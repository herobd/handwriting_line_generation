import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from base import BaseTrainer
import timeit
from utils import util, string_utils, error_rates
from utils.metainit import metainitRecog
from data_loader import getDataLoader
from collections import defaultdict
import random, json, os
from datasets.hw_dataset import PADDING_CONSTANT
from model.clear_grad import ClearGrad
from model.hw_with_style import correct_pred
from datasets.text_data import TextData
from model.autoencoder import Encoder, EncoderSm, Encoder2, Encoder3, Encoder32
import cv2
import torchvision.utils as vutils

STYLE_MAX=15

class HWWithStyleTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(HWWithStyleTrainer, self).__init__(model, loss, metrics, resume, config, train_logger)
        self.config = config
        if 'loss_params' in config:
            self.loss_params=config['loss_params']
        else:
            self.loss_params={}
        for lossname in self.loss:
            if lossname not in self.loss_params:
                self.loss_params[lossname]={}
        self.lossWeights = config['loss_weights'] if 'loss_weights' in config else {"auto": 1, "recog": 1}
        if data_loader is not None:
            self.batch_size = data_loader.batch_size
            self.data_loader = data_loader
            if 'refresh_data' in dir(self.data_loader.dataset):
                self.data_loader.dataset.refresh_data(None,None,self.logged)
            self.data_loader_iter = iter(data_loader)
        if self.val_step<0:
            self.valid_data_loader=None
            print('Set valid_data_loader to None')
        else:
            self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False

        char_set_path = config['data_loader']['char_file']
        with open(char_set_path) as f:
            char_set = json.load(f)
        self.idx_to_char = {}
        self.num_class = len(char_set['idx_to_char'])+1
        for k,v in char_set['idx_to_char'].items():
            self.idx_to_char[int(k)] = v

        self.use_author_vector = config['model']['author_disc'] if 'author_disc' in config['model'] else False
        self.author_vector_gt = config['model']['author_classifier'] is not None if 'author_classifier' in config['model'] else False
        if self.use_author_vector or self.author_vector_gt:
            self.num_authors = len(self.data_loader.dataset.author_list)
            if self.use_author_vector:
                assert(self.num_authors==self.use_author_vector)
                self.use_real_author=0.5

        self.align_loss = 'align' in config['loss']['auto'] if 'auto' in config['loss'] else False

        self.skip_hwr = config['trainer']['skip_hwr'] if 'skip_hwr' in config['trainer'] else False
        self.skip_auto = config['trainer']['skip_auto'] if 'skip_auto' in config['trainer'] else False
        self.style_hwr = 'hwr' in config['model'] and 'Style' in config['model']['hwr']
        self.center_pad = config['data_loader']['center_pad'] if 'center_pad' in config['data_loader'] else False
        assert(not self.center_pad)
        self.feature_loss = 'feature' in self.loss
        self.hack_style = config['trainer']['hack_style'] if 'hack_style' in config['trainer'] else False
        if self.feature_loss or self.hack_style:
            self.clear_hwr_grad = ClearGrad(self.model.hwr)
            self.model.hwr.setup_save_features()

        self.spacing_loss = 'spacing' in self.loss
        self.spacing_input = config['trainer']['space_input'] if 'space_input' in config['trainer'] else False
        self.to_display={}

        self.style_recon=False

        self.style_together = config['trainer']['style_together'] if 'style_together' in config['trainer'] else False
        self.use_hwr_pred_for_style = config['trainer']['use_hwr_pred_for_style'] if 'use_hwr_pred_for_style' in config['trainer'] else False
        self.hwr_withoutStyle = config['trainer']['hwr_without_style'] if 'hwr_without_style' in config['trainer'] else False

        self.gan_loss = 'discriminator' in config['model']
        self.disc_iters = config['trainer']['disc_iters'] if 'disc_iters' in config['trainer'] else 1

        self.use_mask = 'generator' in config['model'] and ('mask' in config['model']['generator'] or 'Mask' in config['model']['generator'])

        self.lookup_style = 'lookup' in config['model']['style'] or 'Lookup' in config['model']['style']

        text_data_batch_size = config['trainer']['text_data_batch_size'] if 'text_data_batch_size' in config['trainer'] else self.config['data_loader']['batch_size']
        text_words = config['trainer']['text_words'] if 'text_words' in config['trainer'] else False
        if 'a_batch_size' in self.config['data_loader']:
            self.a_batch_size = self.config['data_loader']['a_batch_size']
            text_data_batch_size*=self.config['data_loader']['a_batch_size']
        else:
            self.a_batch_size=1
        #text_data_max_len = config['trainer']['text_data_max_len'] if 'text_data_max_len' in config['trainer'] else 20
        if data_loader is not None:
            if 'text_data' in config['trainer']:
                text_data_max_len = self.data_loader.dataset.max_len()
                characterBalance = config['trainer']['character_balance'] if 'character_balance' in config['trainer'] else False
                text_data_max_len = config['trainer']['text_data_max_len'] if 'text_data_max_len' in config['trainer'] else text_data_max_len
                self.text_data = TextData(config['trainer']['text_data'],config['data_loader']['char_file'],text_data_batch_size,max_len=text_data_max_len,words=text_words,characterBalance=characterBalance) if 'text_data' in config['trainer'] else None
        self.balance_loss = config['trainer']['balance_loss'] if 'balance_loss' in config['trainer'] else False # balance the CTC loss with others as in https://arxiv.org/pdf/1903.00277.pdf
        if self.balance_loss:
            self.parameters = list(model.parameters())
            self.balance_var_x = config['trainer']['balance_var_x'] if 'balance_var_x' in config['trainer'] else None
            if self.balance_loss.startswith('sign_preserve_x'):
                self.balance_x = float(self.balance_loss[self.balance_loss.find('x')+1:])
            self.saved_grads = [] #this will hold the gradients for previous training steps if "no-step" is specified

        self.clear_grad_auto_style = config['trainer']['clear_grad_auto_style'] if 'clear_grad_auto_style' in config['trainer'] else False
        if self.clear_grad_auto_style:
            self.style_extractor_parameters = list(model.style_extractor.parameters())
            self.optimizer_gen_only=None

        self.style_detach = config['trainer']['detach_style'] if 'detach_style' in config['trainer'] else (config['trainer']['style_detach'] if 'style_detach' in config['trainer'] else False)
        self.spaced_label_cache={}
        self.cache_spaced_label = config['trainer']['cache_spaced_label'] if 'cache_spaced_label' in config['trainer'] else True

        self.interpolate_gen_styles = config['trainer']['interpolate_gen_styles'] if 'interpolate_gen_styles' in config['trainer'] else False
        if type(self.interpolate_gen_styles) is str and self.interpolate_gen_styles[:6] == 'extra-':
            self.interpolate_gen_styles_low = -float(self.interpolate_gen_styles[6:])
            self.interpolate_gen_styles_high = 1+float(self.interpolate_gen_styles[6:])
        else:
            self.interpolate_gen_styles_low=0
            self.interpolate_gen_styles_high=1
        self.prev_styles_size = config['trainer']['prev_style_size'] if 'prev_style_size' in config['trainer'] else 100
        self.prev_styles = []
        self.prev_g_styles = []
        self.prev_spacing_styles = []
        self.prev_char_styles = []

        self.sometimes_interpolate = config['trainer']['sometimes_interpolate'] if 'sometimes_interpolate' in config['trainer'] else False
        self.interpolate_freq = config['trainer']['interpolate_freq'] if 'interpolate_freq' in config['trainer'] else 0.5

        if 'align_network' in config['trainer']:
            self.align_network = JoinNet()
            weights = config['trainer']['align_network']
            state_dict=torch.load(config['trainer']['align_network'], map_location=lambda storage, location: storage)
            self.align_network.load_state_dict(state_dict)
            self.align_network.set_requires_grad(False)

        self.no_bg_loss= config['trainer']['no_bg_loss'] if 'no_bg_loss' in config else False
        if self.model.char_style_dim>0:
            self.mix_style = config['trainer']['mix_style']

        
        self.sample_disc = self.curriculum.sample_disc if self.curriculum is not None else False
        #if we are going to sample images from the past for the discriminator, these are to store previous generations
        if self.sample_disc:
            self.new_gen=[]
            self.old_gen=[]
            self.store_new_gen_limit = 10
            self.store_old_gen_limit = config['trainer']['store_old_gen_limit'] if 'store_old_gen_limit' in config['trainer'] else 200
            self.new_gen_freq = config['trainer']['new_gen_freq'] if 'new_gen_freq' in config['trainer'] else 0.7
            self.forget_new_freq = config['trainer']['forget_new_freq'] if 'forget_new_freq'  in config['trainer'] else 0.0
            self.old_gen_cache = config['trainer']['old_gen_cache'] if 'old_gen_cache' in config['trainer'] else os.path.join(self.checkpoint_dir,'old_gen_cache')
            if self.old_gen_cache is not None:
                util.ensure_dir(self.old_gen_cache)
                #check for files in cache, so we can resume with them
                for i in range(self.store_old_gen_limit):
                    path = os.path.join(self.old_gen_cache,'{}.pt'.format(i))
                    if os.path.exists(path):
                        self.old_gen.append(path)
                    else:
                        break

        self.gaurd_style = config['trainer']['gaurd_style'] if 'gaurd_style' in config['trainer'] else False
        self.use_char_set_disc = False


        self.WGAN = config['trainer']['WGAN'] if 'WGAN' in config['trainer'] else False
        self.DCGAN = config['trainer']['DCGAN'] if 'DCGAN' in config['trainer'] else False
        if self.DCGAN:
            self.criterion = torch.nn.BCELoss()

        if 'encoder_weights' in config['trainer']:
            snapshot = torch.load(config['trainer']['encoder_weights'],map_location='cpu')
            encoder_state_dict={}
            for key,value in  snapshot['state_dict'].items():
                if key[:8]=='encoder.':
                    encoder_state_dict[key[8:]] = value
            if 'encoder_type' not in config['trainer'] or config['trainer']['encoder_type']=='normal':
                self.encoder = Encoder()
            elif config['trainer']['encoder_type']=='small':
                self.encoder = EncoderSm()
            elif config['trainer']['encoder_type']=='2':
                self.encoder = Encoder2()
            elif config['trainer']['encoder_type']=='2tight':
                self.encoder = Encoder2(32)
            elif config['trainer']['encoder_type']=='2tighter':
                self.encoder = Encoder2(16)
            elif config['trainer']['encoder_type']=='3':
                self.encoder = Encoder3()
            elif config['trainer']['encoder_type']=='32':
                self.encoder = Encoder32(256)
            else:
                raise NotImplementedError('Unknown encoder type: {}'.format(config['trainer']['encoder_type']))
            self.encoder.load_state_dict( encoder_state_dict )
            if self.with_cuda:
                self.encoder = self.encoder.to(self.gpu)

        self.mask_gen_image_end = config['trainer']['mask_gen_image_end'] if 'mask_gen_image_end' in config['trainer'] else False

        self.print_dir = config['trainer']['print_dir'] if 'print_dir' in config['trainer'] else None
        if self.print_dir is not None:
            util.ensure_dir(self.print_dir)
        self.print_every = config['trainer']['print_every'] if 'print_every' in config['trainer'] else 100
        self.iter_to_print = self.print_every
        self.serperate_print_every = config['trainer']['serperate_print_every'] if 'serperate_print_every' in config['trainer'] else 2500
        self.last_print_images=defaultdict(lambda: 0)
        self.print_next_gen=False
        self.print_next_auto=False


        self.casesensitive = config['trainer']['casesensitive'] if 'casesensitive' in config['trainer'] else True


        if 'alt_data_loader' in config:
            alt_config={'data_loader': config['alt_data_loader'],'validation':{}}
            self.alt_data_loader, alt_valid_data_loader = getDataLoader(alt_config,'train')
            self.alt_data_loader_iter = iter(self.alt_data_loader)
        if 'triplet_data_loader' in config:
            triplet_config={'data_loader': config['triplet_data_loader'],'validation':{}}
            self.triplet_data_loader, triplet_valid_data_loader = getDataLoader(triplet_config,'train')
            self.triplet_data_loader_iter = iter(self.triplet_data_loader)

        if 'metainit' in config['trainer'] and config['trainer']['metainit']:
            batch_size=config['data_loader']['batch_size']
            height=self.data_loader.dataset.img_height
            width=self.data_loader.dataset.max_width
            num_char = config['model']['num_class']
            
            metainitRecog(self.model.hwr,self.loss['recog'],(batch_size,1,height,width),num_char)

    def _to_tensor(self, instance):
        image = instance['image']
        label = instance['label']

        if self.with_cuda:
            if image is not None:
                image = image.to(self.gpu)
            if label is not None:
                label = label.to(self.gpu)
        return image, label

    def _eval_metrics(self, typ,name,output, target):
        if len(self.metrics[typ])>0:
            #acc_metrics = np.zeros(len(self.metrics[typ]))
            met={}
            cpu_output=[]
            for pred in output:
                cpu_output.append(output.cpu().data.numpy())
            target = target.cpu().data.numpy()
            for i, metric in enumerate(self.metrics[typ]):
                met[name+metric.__name__] = metric(cpu_output, target)
            return acc_metrics
        else:
            #return np.zeros(0)
            return {}


    def _train_iteration(self, iteration):
        """
        Training logic for an iteration

        :param iteration: Current training iteration.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        #self.model.eval()
        #print("WARNING EVAL")

        ##tic=timeit.default_timer()
        if self.curriculum:
            lesson =  self.curriculum.getLesson(iteration)
        if self.curriculum and all([l[:3]=='gen' or l=='no-step' for l in lesson]) and self.text_data is not None:
            instance = self.text_data.getInstance()
        else:
            if self.curriculum and 'alt-data' in lesson:
                try:
                    instance = self.alt_data_loader_iter.next()
                except StopIteration:
                    if 'refresh_data' in dir(self.alt_data_loader.dataset):
                        self.alt_data_loader.dataset.refresh_data(None,None,self.logged)
                    self.alt_data_loader_iter = iter(self.alt_data_loader)
                    instance = self.alt_data_loader_iter.next()
            elif self.curriculum and 'triplet-style' in lesson:
                try:
                    instance = self.triplet_data_loader_iter.next()
                except StopIteration:
                    if 'refresh_data' in dir(self.triplet_data_loader.dataset):
                        self.triplet_data_loader.dataset.refresh_data(None,None,self.logged)
                    self.triplet_data_loader_iter = iter(self.triplet_data_loader)
                    instance = self.triplet_data_loader_iter.next()
            else:
                try:
                    instance = self.data_loader_iter.next()
                except StopIteration:
                    if 'refresh_data' in dir(self.data_loader.dataset):
                        self.data_loader.dataset.refresh_data(None,None,self.logged)
                    self.data_loader_iter = iter(self.data_loader)
                    instance = self.data_loader_iter.next()
        ##toc=timeit.default_timer()
        ##print('data: '+str(toc-tic))

        ##tic=timeit.default_timer()

        self.optimizer.zero_grad()
        if self.curriculum:
            if any(['disc' in l or 'author-train' in l for l in lesson]):
                self.optimizer_discriminator.zero_grad()
            if any(['auto-style' in l for l in lesson]) and not self.clear_grad_auto_style:
                self.optimizer_gen_only.zero_grad()
            if 'style-ex-only' in lesson:
                self.optimizer_style_ex_only.zero_grad()

        ##toc=timeit.default_timer()
        ##print('for: '+str(toc-tic))

        ##tic=timeit.default_timer()
        #if self.DCGAN:
        #    return self.run_dcgan(instance['image'])
        if self.curriculum:
            if all([l==0 for l in instance['label_lengths']]):
                    return {}
            if (self.iter_to_print<=0 or self.print_next_gen) and 'gen' in lesson:
                losses,got = self.run_gen(instance,lesson,get=['gen','disc'])
                self.print_images(got['gen'],instance['gt'],got['disc'],typ='gen')
                if self.iter_to_print>0:
                    self.print_next_gen=False
                else:
                    self.print_next_auto=True
                    self.iter_to_print=self.print_every
            elif (self.iter_to_print<=0 or self.print_next_auto) and 'auto' in lesson:
                losses,got = self.run_gen(instance,lesson,get=['recon','recon_gt_mask'])
                self.print_images(got['recon'],instance['gt'],typ='recon',gtImages=instance['image'])
                self.print_images(got['recon_gt_mask'],instance['gt'],typ='recon_gt_mask')
                if self.iter_to_print>0:
                    self.print_next_auto=False
                else:
                    self.print_next_gen=True
                    self.iter_to_print=self.print_every
            else:
                losses = self.run_gen(instance,lesson)
                self.iter_to_print-=1
            pred=None
        else:
            if self.gan_loss:
                #alternate between trainin discrminator and enerator
                if iteration%(1+self.disc_iters)==0:
                    pred, recon, losses = self.run(instance,disc=False)
                else:
                    pred, recon, losses = self.run(instance,disc=True)
            else:
                pred, recon, losses = self.run(instance)

        if losses is None:
            return {}
        loss=0
        recogLoss=0
        authorClassLoss=0
        autoGenLoss=0
        for name in losses.keys():
            losses[name] *= self.lossWeights[name[:-4]]
            if self.balance_loss and 'generator' in name and 'auto-gen' in lesson:
                autoGenLoss += losses[name]
            elif self.balance_loss and 'Recog' in name:
                recogLoss += losses[name]
            elif self.balance_loss and 'authorClass' in name:
                authorClassLoss += losses[name]
            else:
                loss += losses[name]
            losses[name] = losses[name].item()
        if (loss!=0 and (torch.isnan(loss) or torch.isinf(loss))):
            print(losses)
        assert(loss==0 or (not torch.isnan(loss) and not torch.isinf(loss)))
        if pred is not None:
            pred = pred.detach().cpu().numpy()
        if type(loss) is not int:
            loss_item = loss.item()
        else:
            loss_item = loss
        if self.balance_loss:
            if type(autoGenLoss) is not int:
                saved_grad=[]
                loss_item += autoGenLoss.item()
                autoGenLoss.backward(retain_graph=True)
                for p in self.parameters:
                    if p.grad is None:
                        saved_grad.append(None)
                    else:
                        saved_grad.append(p.grad.clone())
                        p.grad.zero_()
                self.saved_grads.append(saved_grad)
            if type(recogLoss) is not int:
                saved_grad=[]
                loss_item += recogLoss.item()
                recogLoss.backward(retain_graph=True)
                for p in self.parameters:
                    if p.grad is None:
                        saved_grad.append(None)
                    else:
                        saved_grad.append(p.grad.clone())
                        p.grad.zero_()
                self.saved_grads.append(saved_grad)
            if type(authorClassLoss) is not int:
                saved_grad=[]
                loss_item += authorClassLoss.item()
                authorClassLoss.backward(retain_graph=True)
                for p in self.parameters:
                    if p.grad is None:
                        saved_grad.append(None)
                    else:
                        saved_grad.append(p.grad.clone())
                        p.grad.zero_()
                self.saved_grads.append(saved_grad)
        else:
            loss += recogLoss+authorClassLoss+autoGenLoss


        if type(loss) is not int:
            loss.backward()
            #for p in self.model.parameters():
            #    if p.grad is not None:
            #        assert(not torch.isnan(p.grad).any())

        if self.balance_loss and "no-step" in lesson:
            saved_grad=[]
            for p in self.parameters:
                if p.grad is None:
                    saved_grad.append(None)
                else:
                    saved_grad.append(p.grad.clone())
                    p.grad.zero_()
            self.saved_grads.append(saved_grad)

        elif self.balance_loss and len(self.saved_grads)>0:
            if 'sign_preserve' in self.balance_loss:
                abmean_Ds=[]
                nonzero_sum=0.0
                nonzero_count=0
                for p in self.parameters:
                    if p.grad is not None:
                        abmean_D = torch.abs(p.grad).mean()
                        abmean_Ds.append(abmean_D)
                        if abmean_D!=0:
                            nonzero_sum+=abmean_D
                            nonzero_count+=1
                    else:
                        abmean_Ds.append(None)
                #incase on zero mean
                if nonzero_count>0:
                    nonzero=nonzero_sum/nonzero_count
                    for i in range(len(abmean_Ds)):
                        if abmean_Ds[i]==0.0:
                            abmean_Ds[i]=nonzero
            if self.balance_loss.startswith('sign_preserve_var'):
                for iterT,mult in self.balance_var_x.items():
                    if int(iterT)<=iteration:
                        multipliers=mult
                        if type(multipliers) is not list:
                            multipliers=[multipliers]
            for gi,saved_grad in enumerate(self.saved_grads):
                if self.balance_loss.startswith('sign_preserve_var'):
                    x=multipliers[gi]
                for i,(R, p) in enumerate(zip(saved_grad, self.parameters)):
                    if R is not None:
                        assert(not torch.isnan(p.grad).any())
                        if self.balance_loss=='sign_preserve':
                            abmean_R = torch.abs(p.grad).mean()
                            if abmean_R!=0:
                                p.grad += R*(abmean_Ds[i]/abmean_R)
                        elif self.balance_loss=='sign_match':
                            match_pos = (p.grad>0)*(R>0)
                            match_neg = (p.grad<0)*(R<0)
                            not_match = ~(match_pos+match_neg)
                            p.grad[not_match] = 0 #zero out where signs don't match
                        elif self.balance_loss=='sign_preserve_fixed':
                            abmean_R = torch.abs(R).mean()
                            if abmean_R!=0:
                                p.grad += R*(abmean_Ds[i]/abmean_R)
                        elif self.balance_loss=='sign_preserve_moreHWR':
                            abmean_R = torch.abs(R).mean()
                            if abmean_R!=0:
                                p.grad += 2*R*(abmean_Ds[i]/abmean_R)
                        elif self.balance_loss.startswith('sign_preserve_var'):
                            abmean_R = torch.abs(R).mean()
                            if abmean_R!=0:
                                p.grad += x*R*(abmean_Ds[i]/abmean_R)
                        elif self.balance_loss.startswith('sign_preserve_x'):
                            abmean_R = torch.abs(R).mean()
                            if abmean_R!=0:
                                p.grad += self.balance_x*R*(abmean_Ds[i]/abmean_R)
                        elif self.balance_loss=='orig':
                            if R.nelement()>16:
                                mean_D = p.grad.mean()
                                mean_R = R.mean()
                                std_D = p.grad.std()
                                std_R = R.std()
                                if std_D==0 and std_R==0:
                                    ratio=1
                                else:
                                    if std_R==0:
                                        std_R = 0.0000001
                                    ratio = std_D/std_R
                                if ratio > 100:
                                    ratio = 100
                                p.grad += (ratio*(R-mean_R)+mean_D)
                            else:
                                match = (p.grad>0)*(R>0)
                                p.grad[match] += R[match]*0.001
                                p.grad[~match] *= .1
                                p.grad[~match] += R[~match]*0.0001
                        else:
                            raise NotImplementedError('Unknown gradient balance method: {}'.format(self.balance_loss))
                        assert(not torch.isnan(p.grad).any())
            self.saved_grads=[]
        #for p in self.model.parameters():
        #    if p.grad is not None:
        #        assert(not torch.isnan(p.grad).any())
        #        p.grad[torch.isnan(p.grad)]=0

        if self.curriculum and 'no-step' not in lesson:
            torch.nn.utils.clip_grad_value_(self.model.parameters(),2)
            #meangrad=0
            #count=0
            for m in self.model.parameters():
                assert(not torch.isnan(m).any())
            #        continue
            #    count+=1
            #    meangrad+=m.grad.data.mean().cpu().item()
            #meangrad/=count

            if 'style-ex-only' in lesson:
                self.optimizer_style_ex_only.step()
            elif 'disc' in lesson or 'auto-disc' in lesson or 'disc-style' in lesson or 'author-train' in lesson:
                self.optimizer_discriminator.step()
            elif  any(['auto-style' in l for l in lesson]):
                if self.clear_grad_auto_style:
                    for p in self.style_extractor_parameters:
                        p.grad*=0 #this still has issues as it will be constantly getting different gradients
                    self.optimizer.step()
                else:
                    self.optimizer_gen_only.step()
            else:
                self.optimizer.step()
        elif not self.curriculum:
            if iteration%(1+self.disc_iters)==0 or not self.gan_loss:
                self.optimizer.step()
            else:
                self.optimizer_discriminator.step()

        #assert(not torch.isnan(self.model.spacer.std).any())
        #for p in self.parameters:
        #    assert(not torch.isnan(p).any())


        loss = loss_item

        gt = instance['gt']
        if pred is not None:
            cer,wer, pred_str = self.getCER(gt,pred)
        else:
            cer=0
            wer=0

        ##toc=timeit.default_timer()
        ##print('bac: '+str(toc-tic))

        #tic=timeit.default_timer()
        metrics={}


        log = {
            'loss': loss,
            **losses,
            #'pred_str': pred_str

            'CER': cer,
            'WER': wer,
            #'meangrad': meangrad,

            **metrics,
        }

        #if iteration%10==0:
        #image=None
        #queryMask=None
        #targetBoxes=None
        #outputBoxes=None
        #outputOffsets=None
        #loss=None
        #torch.cuda.empty_cache()


        return log#
    def _minor_log(self, log):
        ls=''
        for key,val in log.items():
            ls += key
            if type(val) is float:
                number = '{:.6f}'.format(val)
                if number == '0.000000':
                    number = str(val)
                ls +=': {},\t'.format(number)
            else:
                ls +=': {},\t'.format(val)
        self.logger.info('Train '+ls)
        for  key,value in self.to_display.items():
            self.logger.info('{} : {}'.format(key,value))
        self.to_display={}

    def _valid_epoch(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()

        total_loss=0
        total_recogLoss=0
        total_autoLoss=0
        total_losses=defaultdict(lambda: 0)
        total_cer=0
        total_wer=0
        print('validate')
        with torch.no_grad():
            losses = defaultdict(lambda: 0)
            for batch_idx, instance in enumerate(self.valid_data_loader):
                if not self.logged:
                    print('validate: {}/{}'.format(batch_idx,len(self.valid_data_loader)), end='\r')
                if self.curriculum:
                    losses = self.run_gen(instance,self.curriculum.getValid())
                    pred=None
                else:
                    pred, recon, losses = self.run(instance)
            
                for name in losses.keys():
                    losses[name] *= self.lossWeights[name[:-4]]
                    total_loss += losses[name].item()
                    total_losses['val_'+name] += losses[name].item()

                if pred is not None:
                    pred = pred.detach().cpu().numpy()
                    gt = instance['gt']
                    cer,wer,_ = self.getCER(gt,pred)
                    total_cer += cer
                    total_wer += wer
        
        for name in total_losses.keys():
            total_losses[name]/=len(self.valid_data_loader)
        toRet={
                'val_loss': total_loss/len(self.valid_data_loader),
                'val_CER': total_cer/len(self.valid_data_loader),
                'val_WER': total_wer/len(self.valid_data_loader),
                **total_losses
                }
        return toRet

    def onehot(self,label):
        label_onehot = torch.zeros(label.size(0),label.size(1),self.num_class)
        label_onehot.scatter_(2,label.cpu().view(label.size(0),label.size(1),1),1)
        #un-tensorized
        #for i in range(label.size(0)):
        #    for j in range(label.size(1)):
        #        label_onehot[i,j,label[i,j]]=1
        return label_onehot.to(label.device)

    def run(self,instance,get_style=False, disc=False):
        image, label = self._to_tensor(instance)
        label_lengths = instance['label_lengths']
        #gt = instance['gt']
        losses = {}
        if not self.skip_hwr and self.hwr_withoutStyle:
            pred = self.model.hwr(image, None)
        elif self.model.guide_hwr is not None:
            pred = self.model.guide_hwr(image, None)
        spaced_label = None
        if self.lookup_style:
            style = self.model.style_extractor(instance['author'],self.gpu)
        elif (not self.skip_auto or self.style_hwr) and self.model.style_extractor is not None:
            if not self.style_together:
                style = self.model.style_extractor(image)
            else:
                #append all the instances in the batch by the same author together along the width dimension
                if self.use_hwr_pred_for_style:
                    spaced_label = pred.permute(1,2,0)
                else:
                    spaced_label = self.correct_pred(pred,label)
                    spaced_label = self.onehot(spaced_label).permute(1,2,0)
                batch_size,feats,h,w = image.size()
                if 'a_batch_size' in instance:
                    a_batch_size = self.a_batch_size#instance['a_batch_size']
                else:
                    a_batch_size = batch_size
                spaced_len = spaced_label.size(2)
                collapsed_image =  image.permute(1,2,0,3).contiguous().view(feats,h,batch_size//a_batch_size,w*a_batch_size).permute(2,0,1,3)
                collapsed_label = spaced_label.permute(1,0,2).contiguous().view(self.num_class,batch_size//a_batch_size,spaced_len*a_batch_size).permute(1,0,2)
                style = self.model.style_extractor(collapsed_image, collapsed_label)
                #style=style.expand(batch_size,-1)
                style = style.repeat(a_batch_size,1)
                spaced_label = spaced_label.permute(2,0,1)
        elif self.style_hwr:
            style = instance['style'].to(image.device)
        else:
            style=None

        if not self.skip_hwr and not self.hwr_withoutStyle:
            pred = self.model.hwr(image, style)
        if self.hack_style:
            feats = self.model.hwr.saved_features
            fs=[]
            for i in range(len(feats)):
                f = feats[i].mean(dim=3).mean(dim=2)
                fs.append(f)
            style = torch.cat(fs,dim=1)
            
        if self.model.create_mask is not None:
            label_onehot=self.onehot(label)
            if self.with_cuda:
                label_onehot = label_onehot.to(self.gpu)
            counts = self.model.spacer(label_onehot,style)
            spaced = self.model.space(label_onehot,counts)
            top_and_bottom = self.model.create_mask(spaced,style)
            mask = self.model.write_mask(top_and_bottom,image.size())



        if not self.skip_auto:
            ###Autoencoder###
            label_onehot=self.onehot(label)
            if self.with_cuda:
                label_onehot = label_onehot.to(self.gpu)
                
            if self.spacing_loss:
                recon, spacing_pred = self.model.generator(label_onehot,style)
                self.to_display['spacing_pred']= spacing_pred[0].argmax(dim=0)[:20].cpu().detach().numpy()

                spaced_label = self.correct_pred(pred,label)
                
                #spaced_label = spaced_label.permute(1,2,0)
                spaced_label = spaced_label.permute(1,0)
                self.to_display['spacing_label']=spaced_label[0,:20].cpu().numpy()
                if spacing_pred.size(2)>spaced_label.size(1): #how to ensure the two vectors have the same scale. Right now both are //4
                    diff=spacing_pred.size(2)-spaced_label.size(1)
                    assert(not self.center_pad)
                    spaced_label = F.pad(spaced_label,(0,diff), value=0)
                elif spacing_pred.size(2)<spaced_label.size(1):
                    diff=spaced_label.size(1)-spacing_pred.size(2)
                    if diff>5:
                        print('WARNING: HWR output ({}) longer than 1D gen ({})'.format(spaced_label.size(1),spacing_pred.size(2)))
                    assert(not self.center_pad)
                    padding = torch.FloatTensor(spacing_pred.size(0),spacing_pred.size(1),diff).zero_()
                    padding[:,:,0]=1 #onehot BLANK ,
                    #spacing_pred = F.pad(spacing_pred,(0,diff), value=BLANK)
                    spacing_pred = torch.cat((spacing_pred,padding.to(spacing_pred.device)),dim=2)

                spacingLoss = self.loss['spacing'](spacing_pred,spaced_label.contiguous(),**self.loss_params['spacing'])
                losses['spacingLoss'] = spacingLoss
                if 'spacingCTC' in self.loss:
                    spacing_pred = spacing_pred.permute(2,0,1)
                    batch_size = spacing_pred.size(1)
                    spacing_pred_size = torch.IntTensor([spacing_pred.size(0)] * batch_size)
                    spacingCTCLoss = self.loss['spacingCTC'](spacing_pred,label.permute(1,0),spacing_pred_size,label_lengths)
                    losses['spacingCTCLoss'] = spacingCTCLoss
                
            elif self.spacing_input:
                if spaced_label is None:
                    #generate with  spaced_label
                    spaced_label = self.correct_pred(pred,label)
                    spaced_label = self.onehot(spaced_label)
                if self.use_mask:
                    mask = instance['mask'].to(image.device)
                    recon = self.model.generator(spaced_label,style,mask)
                else:
                    recon = self.model.generator(spaced_label,style)
            else:
                recon = self.model.generator(label_onehot,style)

            if not self.align_loss:
                #print('recon {}, image {}'.format(recon.size(3), image.size(3)))
                if recon.size(3)>image.size(3):
                    toPad = recon.size(3)-image.size(3)
                    if self.center_pad:
                        image = F.pad(image,(toPad//2,toPad//2 + toPad%2),value=PADDING_CONSTANT)
                    else:
                        image = F.pad(image,(0,toPad),value=PADDING_CONSTANT)
                elif recon.size(3)<image.size(3):
                    toPad = image.size(3)-recon.size(3)
                    if toPad>50:
                        print('WARNING image {} bigger than recon {}'.format(image.size(3),recon.size(3)))
                    if self.center_pad:
                        recon = F.pad(recon,(toPad//2,toPad//2 + toPad%2),value=PADDING_CONSTANT)
                    else:
                        recon = F.pad(recon,(0,toPad),value=PADDING_CONSTANT)
            
            if 'auto' in self.loss and not disc:
                autoLoss = self.loss['auto'](recon,image,**self.loss_params['auto'])
                if type(autoLoss) is tuple:
                    autoLoss, autoLossScales = autoLoss
                losses['autoLoss']=autoLoss
        else:
            recon=None

        if not self.skip_hwr:
            batch_size = pred.size(1)
            pred_size = torch.IntTensor([pred.size(0)] * batch_size)
            if not self.hwr_frozen and not disc:
                recogLoss = self.loss['recog'](pred,label.permute(1,0),pred_size,label_lengths)
                if torch.isinf(recogLoss).any():
                    recogLoss = 0
                else:
                    losses['recogLoss']=recogLoss

            if self.feature_loss and not disc:
                orig_features = list(self.model.hwr.saved_features) #make a new object
                recon = self.clear_hwr_grad(recon) #this will clear the gradients of hwr upon it's backwards call
                recon_pred = self.model.hwr(recon)
                recon_features = self.model.hwr.saved_features

                #orig_features = torch.cat(orig_features)
                #recon_features = torch.cat(recon_features)
                feature_loss = 0
                for r_f,o_f in zip(recon_features,orig_features):
                    feature_loss += self.loss['feature'](r_f,o_f,**self.loss_params['feature'])
                losses['featureLoss']=feature_loss 
                #feature_loss += 0.05*recon_recogLoss
            else:
                recon_pred=None

            if 'reconRecog' in self.loss:
                if recon_pred is None:
                    recon_pred = self.model.hwr(recon)
                recon_pred_size = torch.IntTensor([recon_pred.size(0)] * batch_size)
                recon_recogLoss = self.loss['reconRecog'](recon_pred,label.permute(1,0),recon_pred_size,label_lengths)
                losses['reconRecogLoss']=recon_recogLoss 
        else:
            pred=None
            recogLoss=0

        #if self.style_recon:
        #    #TODO don't train style extractor from this!
        #    style_recon = self.model.style_extractor(recon)
        #    style_reconLoss = self.loss['style_recon'](style_recon,style,**self.loss_params['style_recon'])

        if self.gan_loss:
            if disc:
                discriminator_pred = self.model.discriminator(torch.cat((image,recon),dim=0))
                if self.style_recon: 
                    discriminator_pred, disc_style = discriminator_pred
                    disc_style_on_real, disc_style_on_fake = torch.chunk(disc_style,2,dim=0)
                    losses['style_reconLoss'] = self.loss['style_recon'](disc_style,style, **self.loss_params['style_recon'])
                discriminator_pred_on_real, discriminator_pred_on_fake = torch.chunk(discriminator_pred,2,dim=0)
                #hinge loss
                disc_loss = F.relu(1.0 - discriminator_pred_on_real).mean() + F.relu(1.0 + discriminator_pred_on_fake).mean()
                losses = {} #we don't want anythin interferrin
                losses['discriminatorLoss']=disc_loss
            else:
                gen_loss = -self.model.discriminator(recon).mean()
                losses['generatorLoss']=gen_loss
                assert(not torch.isnan(losses['generatorLoss']))

        if 'key' in self.loss:
            losses['keyLoss'] = self.loss['key'](self.model.style_extractor.keys1,**self.loss_params['key'])



        if get_style:
            if self.use_hwr_pred_for_style:
                spaced_label = self.correct_pred(pred,label)
                spaced_label = self.onehot(spaced_label)
            return pred, recon, losses, style, spaced_label
        else:
            return pred, recon, losses


    def run_gen(self,instance,lesson,get=[]):
        image, label = self._to_tensor(instance)
        batch_size = label.size(1)
        label_lengths = instance['label_lengths']
        a_batch_size = self.a_batch_size if 'a_batch_size' in instance else None

        autoSGStyle = any(['auto-style' in a for a in lesson])

        losses = {}

        if self.use_author_vector:
            author_vector=torch.zeros(batch_size,self.num_authors+1)
            if 'author_idx' in instance or ('eval' in lesson or 'valid' in lesson):
                for b in range(batch_size):
                    if random.random()<self.use_real_author:
                        author_vector[b,instance['author_idx'][b]+1]=1
                    else:
                        author_vector[b,0]=1
            else:
                author_vector[:,0]=1 #all to unknown

            author_vector = author_vector.to(label.device)
        elif self.author_vector_gt and any(['author' in l for l in lesson]) and not ('eval' in lesson or 'valid' in lesson):
            if self.with_cuda:
                author_vector=torch.LongTensor(instance['author_idx']).to(self.gpu)
            else:
                author_vector=torch.LongTensor(instance['author_idx'])
        else:
            author_vector=None


        if 'recon_gt_mask' in get: #this is just to be able to see both
            lesson = lesson+['not-auto-mask']


        #CACHED spaced_label
        need_spaced_label = any([x in lesson for x in ['count','mask','auto','disc','triplet-style']])
        if need_spaced_label:
            if instance['spaced_label'] is not None:
                self.model.spaced_label = self.model.onehot(instance['spaced_label']).to(label.device)
            elif self.cache_spaced_label:
                #print('self.spaced_label_cache:')
                #print(self.spaced_label_cache.keys())
                found=[None]*label.size(1)
                all_found = True
                for b,name in enumerate(instance['name']):
                    if name in self.spaced_label_cache:
                        found[b] = self.spaced_label_cache[name].to(label.device)
                    else:
                        all_found=False
                if not all_found:
                    self.model.pred = self.pred = self.model.hwr(image, None)
                    to_correct = []
                    to_correct_b=[]
                    to_correct_label=[]
                    for b in range(label.size(1)):
                        if found[b] is None or self.model.pred.size(0)<found[b].size(0): #if we don't have it or we can get a smaller version
                            to_correct.append(self.model.pred[:,b])
                            to_correct_b.append(b)
                            to_correct_label.append(label[:,b])
                    to_correct = torch.stack(to_correct,dim=1)
                    to_correct_label = torch.stack(to_correct_label,dim=1)
                    corrected = correct_pred(to_correct,to_correct_label)
                    corrected = self.model.onehot(corrected)
                    for i,b in enumerate(to_correct_b):
                        name = instance['name'][b]
                        self.spaced_label_cache[name] = corrected[:,i:i+1].cpu().detach()
                        found[b] = corrected[:,i:i+1]
                max_len = 0
                for ten in found:
                    max_len = max(ten.size(0),max_len)
                self.model.spaced_label = torch.FloatTensor(max_len,label.size(1),found[0].size(2)).fill_(0).to(label.device)
                #self.model.spaced_label = torch.cat(found,dim=1)
                for b,ten in enumerate(found):
                    self.model.spaced_label[:ten.size(0),b:b+1,:] = ten

        if 'auto' in lesson or 'auto-disc' in lesson or 'split-auto' in lesson:
            #Running autoencoder

            if 'mask' in instance and instance['mask'] is not None and 'auto-mask' not in lesson and 'eval' not in lesson:
                mask = instance['mask'].to(image.device)
            else:
                mask = None
            if 'not-auto-mask' in lesson:
                if mask is None:
                    if 'mask' in instance:
                         mask_gt = instance['mask'].to(image.device)
                    else:
                         mask_gt=None
                else:
                     mask_gt = mask
                #this should only be used in eval
                #assert('eval' in lesson or 'valid' in lesson)
                recon_gt_mask,style_ = self.model.autoencode(image,label,mask_gt,a_batch_size,center_line=None)
            if 'recon_pred_space' in get:
                style = self.model.extract_style(image,label,a_batch_size)

                recon_pred_space = self.model(label,label_lengths,style)


            if 'auto-mask' in lesson or 'eval' in lesson:
                center_line = instance['center_line']
            else:
                center_line = None
            if 'split-style' in lesson:
                style = self.model.extract_style(image,label,a_batch_size)
                if self.model.spaced_label is None:
                    self.model.spaced_label = self.correct_pred(self.model.pred,label)
                    self.model.spaced_label = self.onehot(self.model.spaced_label)
                changed_image = instance['changed_image']
                if self.with_cuda:
                    changed_image = changed_image.to(self.gpu)
                changed_style = self.model.extract_style(changed_image,label,a_batch_size)

                #becuase we will be generating double images (two for each input), we'll halve the batch here
                image = image[::2]
                changed_image = changed_image[::2]
                self.model.pred = self.model.pred[:,::2]
                self.model.spaced_label = self.model.spaced_label[:,::2]
                label = label[:,::2]
                label_lengths = label_lengths[::2]
                if self.no_bg_loss:
                    instance['fg_mask'] = instance['fg_mask'][::2]
                mask = mask[::2]
                a_batch_size = a_batch_size//2 + a_batch_size%2


                #swap the "character shape" portion of the style vector
                if self.model.char_style_dim>0:
                    g_style,spacing_style,char_style = style
                    g_style = g_style[::2]
                    spacing_style = spacing_style[::2]
                    char_style = char_style[::2]

                    g_changed_style,spacing_changed_style,char_changed_style = changed_style
                    g_changed_style = g_changed_style[::2]
                    spacing_changed_style = spacing_changed_style[::2]
                    char_changed_style = char_changed_style[::2]

                    size = spacing_style.size(1)
                    swapped_spacing_style = torch.cat((spacing_style[:,:size//2],spacing_changed_style[:,size//2:]),dim=1)
                    swapped_spacing_changed_style = torch.cat((spacing_changed_style[:,:size//2],spacing_style[:,size//2:]),dim=1)

                    swapped_style = (g_style,swapped_spacing_style,char_style)
                    swapped_changed_style = g_changed_style,swapped_spacing_changed_style,char_changed_style
                else:
                    style = style[::2]
                    changed_style = changed_style[::2]
                    swapped_style = torch.cat((style[:,:32],changed_style[:,32:]),dim=1)
                    swapped_changed_style = torch.cat((changed_style[:,:32],style[:,32:]),dim=1)

                #append regular and changed, as well as repeat relevant data
                style =  self.cat_styles((swapped_style,swapped_changed_style))#(torch.cat((swapped_style,swapped_changed_style),dim=0)
                image = torch.cat((image,changed_image),dim=0)
                label = label.repeat(1,2)#torch.cat((label,label),dim=1)
                label_lengths = label_lengths.repeat(2)
                mask = mask.repeat(2,1,1,1)#torch.cat((mask,mask),dim=0)
                self.model.spaced_label = self.model.spaced_label.repeat(1,2,1)
                if self.no_bg_loss:
                    instance['fg_mask'] = instance['fg_mask'].repeat(2,1,1,1)
                self.model.pred=None
                

                recon = self.model(label,None,style,mask,self.model.spaced_label)
            elif 'eval' not in lesson or 'recon' in get:
                recon,style = self.model.autoencode(image,label,mask,a_batch_size,center_line=center_line,stop_grad_extractor=autoSGStyle)
                if 'styleReg' in self.loss:
                    losses['styleRegLoss'] = self.loss['styleReg'](style,**self.loss_params['styleReg'])



            if self.sample_disc and 'eval' not in lesson and 'valid' not in lesson:
                self.add_gen_sample(recon,self.model.spaced_label,style,author_vector.cpu() if author_vector is not None else None)

            if self.interpolate_gen_styles and 'eval' not in lesson and 'valid' not in lesson:
                if self.model.char_style_dim>0:
                    g_style,spacing_style,char_style = style
                    for i in range(0,batch_size,a_batch_size):
                        self.prev_g_styles.append(g_style[i].detach().cpu())
                        self.prev_spacing_styles.append(spacing_style[i].detach().cpu())
                        self.prev_char_styles.append(char_style[i].detach().cpu())
                    self.prev_g_styles =self.prev_g_styles[-self.prev_styles_size:]
                    self.prev_spacing_styles =self.prev_spacing_styles[-self.prev_styles_size:]
                    self.prev_char_styles =self.prev_char_styles[-self.prev_styles_size:]
                else:
                    for i in range(0,batch_size,a_batch_size):
                        self.prev_styles.append(style[i].detach().cpu())
                    self.prev_styles =self.prev_styles[-self.prev_styles_size:]
        else:
            style=None
            recon=None

        if 'gen' in lesson or 'disc' in lesson or 'gen-style' in lesson or 'disc-style' in lesson or 'gen' in get:
            #Get style vector for pure generative
            if 'eval' not in lesson and 'valid' not in lesson or not self.interpolate_gen_styles:
                style_gen = self.get_style_gen(batch_size,label.device)
            else:
                #During evaluation, we don't have a bank of previous styles, so we use the previous and current
                if self.model.char_style_dim>0:
                    g_style, spacing_style, char_style = style
                    g_style_gen = torch.empty_like(g_style)
                    spacing_style_gen = torch.empty_like(spacing_style)
                    char_style_gen = torch.empty_like(char_style)
                    for i in range(batch_size//a_batch_size):
                        b=i*a_batch_size
                        b_1 = ((i+1)%(batch_size//a_batch_size))*a_batch_size
                        g_style_gen[b:b+a_batch_size] = 0.5*g_style[b_1:b_1+a_batch_size] + 0.5*g_style[b:b+a_batch_size]
                        spacing_style_gen[b:b+a_batch_size] = 0.5*spacing_style[b_1:b_1+a_batch_size] + 0.5*spacing_style[b:b+a_batch_size]
                        char_style_gen[b:b+a_batch_size] = 0.5*char_style[b_1:b_1+a_batch_size] + 0.5*char_style[b:b+a_batch_size]
                        style_gen = (g_style_gen,spacing_style_gen,char_style_gen)
                else:
                    #style_gen = style
                    style_gen = torch.empty_like(style)
                    for i in range(batch_size//a_batch_size):
                        b=i*a_batch_size
                        b_1 = ((i+1)%(batch_size//a_batch_size))*a_batch_size
                        style_gen[b:b+a_batch_size] = 0.5*style[b_1:b_1+a_batch_size] + 0.5*style[b:b+a_batch_size]
        else:
            style_gen = None

        if 'gen' in lesson or 'disc' in lesson or 'gen' in get:
            #Pure generative
            if 'eval' not in lesson and label.size(0)>self.text_data.max_len:
                if 'auto' not in lesson:
                    label=label[:self.text_data.max_len]
                #label_lengths = [min(l.item(),self.text_data.max_len) for l in label_lengths]
                for b in range(batch_size):
                    label_lengths[b] = min(label_lengths[b],self.text_data.max_len)
            gen_image = self.model(label,label_lengths,style_gen)
            if self.mask_gen_image_end:
                for b in range(batch_size):
                    end_of_text = int(self.model.gen_padded[b]*gen_image.size(3)+0.1)
                    gen_image[b,:,:,:end_of_text] = PADDING_CONSTANT
            if self.sample_disc and 'eval' not in lesson and 'valid' not in lesson:
                if self.use_author_vector:
                    fake_author_vector=torch.zeros(batch_size,self.num_authors+1)
                    fake_author_vector[:,0]=1
                else:
                    fake_author_vector=None
                self.add_gen_sample(gen_image,self.model.gen_spaced,style_gen,fake_author_vector)
        else:
            gen_image = None


        if 'auto' in lesson and 'auto' in self.loss and 'eval' not in lesson:
            paddedImage=False
            if recon.size(3)>image.size(3):
                toPad = recon.size(3)-image.size(3)
                paddedImage=True
                if self.center_pad:
                    image = F.pad(image,(toPad//2,toPad//2 + toPad%2),value=PADDING_CONSTANT)
                else:
                    image = F.pad(image,(0,toPad),value=PADDING_CONSTANT)
            elif recon.size(3)<image.size(3):
                toPad = image.size(3)-recon.size(3)
                if toPad>50:
                    print('WARNING image {} bigger than recon {}'.format(image.size(3),recon.size(3)))
                if self.center_pad:
                    recon = F.pad(recon,(toPad//2,toPad//2 + toPad%2),value=PADDING_CONSTANT)
                else:
                    recon = F.pad(recon,(0,toPad),value=PADDING_CONSTANT)
            if self.no_bg_loss:
                fg_mask = instance['fg_mask']
                if paddedImage:
                    if self.center_pad:
                        fg_mask = F.pad(fg_mask,(toPad//2,toPad//2 + toPad%2),value=0)
                    else:
                        fg_mask = F.pad(fg_mask,(0,toPad),value=0)
                recon_autol = recon*fg_mask
                image_autol = image*fg_mask
            else:
                recon_autol = recon
                image_autol = image
            autoLoss = self.loss['auto'](recon_autol,image_autol,**self.loss_params['auto'])
            if type(autoLoss) is tuple:
                autoLoss, autoLossScales = autoLoss
            losses['autoLoss']=autoLoss
            assert(not torch.isnan(autoLoss))


        if 'align-auto' in lesson:
            aligned_recon = self.align_network(recon,image)
            autoLoss = self.loss['alignAuto'](recon,image,**self.loss_params['alignAuto'])
            if type(autoLoss) is tuple:
                autoLoss, autoLossScales = autoLoss
            losses['alignAutoLoss']=autoLoss

        if 'spacing' in lesson  and ('spacingDirect' in self.loss or 'spacingCTC' in self.loss):
            if  self.model.spacing_pred is None:
                mask = None
                center_line = instance['center_line']
                recon,style = self.model.autoencode(image,label,mask,a_batch_size,center_line=center_line)
            if self.model.spaced_label is None:
                self.model.spaced_label = correct_pred(self.model.pred,label)
                self.model.spaced_label = self.model.onehot(self.model.spaced_label)

            if 'spacingDirect' in self.loss:
                spaced_label = self.model.spaced_label
                if self.model.spacing_pred.size(0)>self.model.spaced_label.size(0):
                    diff = self.model.spacing_pred.size(0)-self.model.spaced_label.size(0)
                    self.model.spacing_pred = self.model.spacing_pred[:-diff]
                elif self.model.spaced_label.size(0)>self.model.spacing_pred.size(0):
                    diff = self.model.spaced_label.size(0)-self.model.spacing_pred.size(0)
                    spaced_label = self.model.spaced_label[:-diff]
                losses['spacingDirectLoss'] = self.loss['spacingDirect'](self.model.spacing_pred,spaced_label)
                assert(torch.isfinite(losses['spacingDirectLoss']))


            if 'spacingCTC' in self.loss:
                index_spaced = self.model.spaced_label.argmax(dim=2)+1 #elevate blanks
                index_spaced_size = torch.IntTensor([index_spaced.size(0)] * batch_size)
                spacing_pred = F.pad(self.model.spacing_pred,(1,0)) #add blank channel
                blank_pred = torch.zeros_like(spacing_pred)
                blank_pred[:,:,0] = 1
                total_len = spacing_pred.size(0)*2
                batch_size = label.size(1)
                spacing_pred = torch.stack((spacing_pred,blank_pred.to(spacing_pred.device)),dim=1).view(total_len,batch_size,-1)
                spacing_pred_size = torch.IntTensor([spacing_pred.size(0)] * batch_size)
                spacingCTCLoss = self.loss['spacingCTC'](spacing_pred,index_spaced.permute(1,0),spacing_pred_size,index_spaced_size)
                if torch.isfinite(spacingCTCLoss).all():
                    losses['spacingCTCLoss'] =spacingCTCLoss

        if 'mask_side' in lesson  and 'mask_side' in self.loss:
            if  self.model.mask_pred is None:
                mask = None
                center_line = instance['center_line']
                recon,style = self.model.autoencode(image,label,mask,a_batch_size,center_line=center_line)
            mask_gt = instance['mask'].to(image.device)
            if mask_gt.size(3)>self.model.mask_pred.size(3):
                diff = mask_gt.size(3) - self.model.mask_pred.size(3)
                mask_gt = mask_gt[:,:,:,:-diff]
            elif self.model.mask_pred.size(3)>mask_gt.size(3):
                diff=self.model.mask_pred.size(3)-mask_gt.size(3)
                self.model.mask_pred = self.model.mask_pred[:,:,:,:-diff]
            mask_sideLoss = self.loss['mask_side'](self.model.mask_pred,mask_gt,**self.loss_params['mask_side'])
            if type(mask_sideLoss) is tuple:
                mask_sideLoss, mask_sideLossScales = mask_sideLoss
            losses['mask_sideLoss'] = mask_sideLoss

        if 'count' in lesson  and 'count' in self.loss and 'eval' not in lesson:
            if 'auto' not in lesson:
                #pred = self.model.hwr(image, None)
                #self.model.spaced_label = correct_pred(pred,label)
                #self.model.spaced_label = self.model.onehot(self.model.spaced_label)
                style = self.model.extract_style(image,label,a_batch_size)
                if self.gaurd_style:
                    style = torch.where(style.abs()>STYLE_MAX,STYLE_MAX*style/style.detach(),style)
                #pred = self.model.hwr(image)
                if '$UNKOWN$' in instance['gt']:
                    psuedo_gt=[]
                    psuedo_labels=[]
                    new_styles=[]
                    new_pred=[]
                    for b in range(batch_size):
                        if instance['gt'][b] == '$UNKOWN$':
                            logits = self.model.pred[:,b]
                            pred_str, raw_pred = string_utils.naive_decode(logits.detach().cpu().numpy())
                            pred_str = string_utils.label2str_single(pred_str, self.idx_to_char, False)
                            if len(pred_str)>0:
                                psuedo_gt.append(pred_str)
                                psuedo_labels.append( torch.from_numpy(string_utils.str2label_single(pred_str, self.data_loader.dataset.char_to_idx).astype(np.int32) ) )
                                new_styles.append(style[b])
                                new_pred.append(self.model.pred[:,b])
                        else:
                            psuedo_labels.append( label[:,b] )
                            new_styles.append(style[b])
                            new_pred.append(self.model.pred[:,b])
                    if len(new_styles)==0:
                        if get:
                            return losses,{}
                        else:
                            return losses
                    batch_size = len(new_styles)
                    max_len = max([pl.size(0) for pl in psuedo_labels])
                    new_ps_lb = []
                    label=None
                    for pl in psuedo_labels:
                        diff = max_len-pl.size(0)
                        if diff>0:
                            new_ps_lb.append( F.pad(pl,(0,diff)).to(style.device) )
                        else:
                            new_ps_lb.append( pl.to(style.device) )
                    label=torch.stack(new_ps_lb,dim=1)
                    style=torch.stack(new_styles,dim=0)
                    self.model.pred=torch.stack(new_pred,dim=1)
                    #spaced_label_m = torch.argmax(self.model.pred,dim=2) This failed due differences in how np.argmax and torch.argmax break ties...

                #else:
                spaced_label_m = correct_pred(self.model.pred,label)
                spaced_label_m = self.model.onehot(spaced_label_m)
            else:
                spaced_label_m = self.model.spaced_label
            #if 'gen' not in lesson:
            label_onehot=self.model.onehot(label)
            if self.style_detach:
                if self.model.char_style_dim>0:
                    style_d = (style[0].detach(),style[1].detach(),style[2].detach())
                else:
                    style_d = style.detach()
            else:
                style_d=style
            self.model.counts = self.model.spacer(label_onehot,style_d)
            index_spaced = spaced_label_m.argmax(dim=2)
            if self.model.count_duplicates:
                gt_counts = torch.FloatTensor(label.size(0),batch_size,2).fill_(0)
                for b in range(batch_size):
                    c=0 #blanks
                    d=0 #repeated char
                    pos=0
                    last=0
                    for i in range(index_spaced.size(0)):
                        index = index_spaced[i,b].item()
                        if index==0 and last==0:
                            c+=1
                        elif last==0 or last==index:
                            d+=1
                            last=index
                        else:
                            assert(label[pos,b].item()==last)
                            gt_counts[pos,b,0]=c
                            gt_counts[pos,b,1]=d
                            if index==0:
                                c=1
                                d=0
                            else:
                                c=0
                                d=1
                            pos+=1
                            last=index
                    #count[pos]=c
                    self.model.counts[pos:]=0
            else:
                gt_counts = torch.FloatTensor(label.size(0),batch_size,1).fill_(0)
                for b in range(batch_size):
                    c=0
                    pos=0
                    last=-1
                    for i in range(index_spaced.size(0)):
                        index = index_spaced[i,b]
                        if index==0 or index==last:
                            c+=1
                        else:
                            assert(label[pos,b].item()==index.item())
                            gt_counts[pos,b,0]=c
                            c=0
                            pos+=1
                        last=index
                    #count[pos]=c
                    self.model.counts[pos:]=0

            #self.DEBUG1=self.model.counts
            #self.DEBUG2=gt_counts
            #use_pred_counts = torch.where(torch.isinf(self.model.counts), torch.zeros_like(self.model.counts), self.model.counts)
            assert(not torch.isinf(self.model.counts).any())
            gt_counts = gt_counts.to(self.model.counts.device)
            #gt_counts = torch.where(torch.isinf(self.model.counts), torch.zeros_like(gt_counts), gt_counts)
            losses['countLoss'] = self.loss['count'](self.model.counts,gt_counts,**self.loss_params['count'])
            assert(not torch.isnan(losses['countLoss']) and not torch.isinf(losses['countLoss']))

        if ('mask' in lesson  or 'mask-gen' in lesson or 'mask-disc' in lesson) and 'eval' not in lesson:
            if 'auto' not in lesson and 'count' not in lesson:
                #pred = self.model.hwr(image, None)
                #self.model.spaced_label = correct_pred(pred,label)
                #self.model.spaced_label = self.model.onehot(self.model.spaced_label)
                style = self.model.extract_style(image,label,a_batch_size)
                #pred = self.model.hwr(image)
                spaced_label_m = correct_pred(self.model.pred,label)
                spaced_label_m = self.model.onehot(spaced_label_m)
            else:
                spaced_label_m = correct_pred(self.model.pred,label)
                spaced_label_m = self.model.onehot(spaced_label_m)
            if self.model.char_style_dim>0:
                style_d = self.model.space_style(spaced_label_m,style)
            else:
                style_d = style
            if self.style_detach:
                if self.model.char_style_dim>0:
                    style_d = (style_d[0].detach(),style_d[1].detach(),style_d[2].detach())
                else:
                    style_d = style_d.detach()
            self.model.top_and_bottom = self.model.create_mask(spaced_label_m,style_d)
            #gt_top_and_bottom = torch.FloatTensor(mask.size(3),batch_size,2)
            #center = getCenterLine(image)
            #for h in range(mask.size(3)):
            gt_top_and_bottom = instance['top_and_bottom']
            if 'gt_mask' in get:
                gt_mask = self.model.write_mask(self.model.top_and_bottom,image.size())
            self.model.top_and_bottom = self.model.top_and_bottom.permute(1,2,0)

            if gt_top_and_bottom.size(2)>self.model.top_and_bottom.size(2):
                diff= gt_top_and_bottom.size(2)-self.model.top_and_bottom.size(2)
                gt_top_and_bottom = gt_top_and_bottom[:,:,0:-diff]
            elif gt_top_and_bottom.size(2)<self.model.top_and_bottom.size(2):
                diff = self.model.top_and_bottom.size(2)-gt_top_and_bottom.size(2)
                gt_top_and_bottom = F.pad(gt_top_and_bottom,(0,diff),'replicate')
            if 'mask' in lesson and 'mask' in self.loss:    
                losses['maskLoss'] = self.loss['mask'](self.model.top_and_bottom,gt_top_and_bottom.to(self.model.top_and_bottom.device),**self.loss_params['mask'])

            if 'mask-gen' in lesson and 'mask-gen' in self.loss:
                losses['mask_generatorLoss'] = -self.model.mask_discriminator(self.model.top_and_bottom.permute(2,0,1)).mean()

            if 'mask-disc' in lesson:
                num_fake = self.model.top_and_bottom.size(0)
                discriminator_pred = self.model.mask_discriminator(torch.cat((self.model.top_and_bottom,gt_top_and_bottom.to(self.model.top_and_bottom.device)),dim=0).permute(2,0,1))
                discriminator_pred_on_fake = discriminator_pred[:num_fake]
                discriminator_pred_on_real = discriminator_pred[num_fake:]
                disc_loss = F.relu(1.0 - discriminator_pred_on_real).mean() + F.relu(1.0 + discriminator_pred_on_fake).mean()
                losses['mask_discriminatorLoss']=disc_loss
                assert(not torch.isnan(disc_loss))

        if ('auto' in lesson or ('count' in lesson and not self.style_detach)) and self.model.vae and self.model.training:
            #losses['VAE_KLDLoss'] = 0.5 * torch.sum(
            #                                torch.pow(self.model.mu, 2) +
            #                                torch.pow(self.model.sigma, 2) -
            #                                torch.log(1e-8 + torch.pow(self.model.sigma, 2)) - 1
            #                               ).sum() / batch_size
            tooBig = self.model.sigma>100
            if tooBig.any():
                self.model.sigma = self.model.sigma/torch.where(tooBig,100*self.model.sigma.detach(),torch.ones_like(self.model.sigma))
            e_distribution = torch.distributions.Normal(self.model.mu,self.model.sigma)
            div= torch.distributions.kl_divergence(e_distribution, torch.distributions.Normal(0,1))
            losses['VAE_KLDLoss'] = div[(~torch.isnan(div)) & (~torch.isinf(div))].mean()
            assert(not torch.isnan(losses['VAE_KLDLoss']) and not torch.isinf(losses['VAE_KLDLoss']))
            if losses['VAE_KLDLoss']>10: #ugh, this is a hack... was to fix training failure which I believe was fixed by balanving gradients
                losses['VAE_KLDLoss']/=10*losses['VAE_KLDLoss'].detach()
            #assert(losses['VAE_KLDLoss']<100)
            #losses['VAE_KLDLoss'] = torch.distributions.kl_divergence(e_distribution, torch.distributions.Normal(0,1)).mean()
            assert(not torch.isnan(losses['VAE_KLDLoss']))
            assert(not torch.isinf(losses['VAE_KLDLoss']))


        if 'auto' in lesson and 'feature' in self.loss and 'eval' not in lesson:
            #recon = self.clear_hwr_grad(recon) #this will clear the gradients of hwr upon it's backwards call
            pred = self.model.hwr(image)
            orig_features = list(self.model.hwr.saved_features) #make a new object
            recon_pred = self.model.hwr(recon)
            recon_features = self.model.hwr.saved_features

            #orig_features = torch.cat(orig_features)
            #recon_features = torch.cat(recon_features)
            feature_loss = 0
            for r_f,o_f in zip(recon_features,orig_features):
                feature_loss += self.loss['feature'](r_f,o_f,**self.loss_params['feature'])
            losses['featureLoss']=feature_loss 
            #feature_loss += 0.05*recon_recogLoss
        else:
            recon_pred = None

        if 'auto' in lesson and 'perceptual' in self.loss and 'eval' not in lesson:
            if image.size(3)>recon.size(3):
                diff = image.size(3)-recon.size(3)
                    
                if diff>1:
                    print('Warning, different sizes between image {} and recon {}'.format(image.size(),recon.size()))
                recon = F.pad(recon,(diff//2,diff//2 +diff%2))
            elif image.size(3)<recon.size(3):
                diff = recon.size(3)-image.size(3)
                if diff>1:
                    print('Warning, different sizes between image {} and recon {}'.format(image.size(),recon.size()))
                image = F.pad(image,(diff//2,diff//2 +diff%2))
            both_i = torch.cat( (image,recon), dim=0)
            if both_i.size(3)<40:
                diff = 40-both_i.size(3)
                both_i = F.pad(both_i,(diff//2,diff//2 +diff%2))
            both_f = self.encoder(both_i)
            orig_features,recon_features=zip(*[torch.chunk(b,2,dim=0) for b in both_f])
            #orig_features, recon_features = torch.chunk(both_f,2,dim=0)
            perceptual_loss = 0
            for r_f,o_f in zip(recon_features,orig_features):
                perceptual_loss += self.loss['perceptual'](r_f,o_f,**self.loss_params['perceptual'])
            losses['perceptualLoss']=perceptual_loss 
        if 'auto' in lesson and 'perceptualDisc' in self.loss and 'eval' not in lesson:
            if image.size(3)>recon.size(3):
                diff = image.size(3)-recon.size(3)
                    
                if diff>50:
                    print('Warning, different sizes between image {} and recon {}'.format(image.size(),recon.size()))
                recon = F.pad(recon,(diff//2,diff//2 +diff%2))
            elif image.size(3)<recon.size(3):
                diff = recon.size(3)-image.size(3)
                if diff>50:
                    print('Warning, different sizes between image {} and recon {}'.format(image.size(),recon.size()))
                image = F.pad(image,(diff//2,diff//2 +diff%2))
            
            orig_features = self.model.discriminator(self.model.spaced_label,style,image,return_features=True,author_vector=author_vector)
            recon_features = self.model.discriminator(self.model.spaced_label,style,recon,return_features=True,author_vector=author_vector)
            perceptualDisc_loss = 0
            for r_f,o_f in zip(recon_features,orig_features):
                perceptualDisc_loss += self.loss['perceptualDisc'](r_f,o_f,**self.loss_params['perceptualDisc'])
            losses['perceptualDiscLoss']=perceptualDisc_loss 
            assert(not torch.isnan(perceptualDisc_loss))

        if 'auto' in lesson and 'reconRecog' in self.loss and 'eval' not in lesson:
            if recon_pred is None:
                recon_pred = self.model.hwr(recon)
            recon_pred_size = torch.IntTensor([recon_pred.size(0)] * batch_size)
            recon_recogLoss = self.loss['reconRecog'](recon_pred,label.permute(1,0),recon_pred_size,label_lengths)
            losses['reconRecogLoss']=recon_recogLoss 

        if 'gen' in lesson and 'genRecog' in self.loss and 'eval' not in lesson:
            gen_pred = self.model.hwr(gen_image)
            gen_pred_size = torch.IntTensor([gen_pred.size(0)] * batch_size)
            gen_recogLoss = self.loss['genRecog'](gen_pred,label.permute(1,0),gen_pred_size,label_lengths)
            if torch.isfinite(gen_recogLoss):
                losses['genRecogLoss']=gen_recogLoss 



        #Get generated and real data to match sizes
        if 'gen' in lesson or 'disc' in lesson:
            if ('auto' in lesson or 'auto-disc' in lesson) and 'eval' not in lesson:
                if recon.size(3)>gen_image.size(3):
                    diff = recon.size(3)-gen_image.size(3)
                    gen_image = F.pad(gen_image,(0,diff,0,0),'replicate')
                elif recon.size(3)<gen_image.size(3):
                    diff = -(recon.size(3)-gen_image.size(3))
                    recon = F.pad(recon,(0,diff,0,0),'replicate')
                fake = torch.cat((recon,gen_image),dim=0)
                if self.model.cond_disc:
                    if self.model.char_style_dim>0:
                        fake_style = self.cat_styles( 
                                (self.model.space_style(self.model.spaced_label,style),
                                 self.model.spaced_style) )
                    else:
                        fake_style = self.cat_styles((style,style_gen))
                    if self.model.spaced_label.size(0) > self.model.gen_spaced.size(0):
                        diff = self.model.spaced_label.size(0)-self.model.gen_spaced.size(0)
                        self.model.gen_spaced = self.model.gen_spaced.permute(1,2,0)
                        self.model.gen_spaced = F.pad(self.model.gen_spaced,(0,diff),'constant')
                        self.model.gen_spaced[:,0,-diff:]=1
                        self.model.gen_spaced = self.model.gen_spaced.permute(2,0,1)
                    elif self.model.spaced_label.size(0) < self.model.gen_spaced.size(0):
                        diff = -(self.model.spaced_label.size(0)-self.model.gen_spaced.size(0))
                        self.model.spaced_label = self.model.spaced_label.permute(1,2,0)
                        self.model.spaced_label = F.pad(self.model.spaced_label,(0,diff),'constant')
                        self.model.spaced_label[:,0,-diff:]=1
                        self.model.spaced_label = self.model.spaced_label.permute(2,0,1)
                    fake_label = torch.cat((self.model.spaced_label,self.model.gen_spaced),dim=1)
            else:
                fake = gen_image
                if self.model.char_style_dim>0:
                    fake_style = self.model.spaced_style #style_gen
                else:
                    fake_style = style_gen
                fake_label = self.model.gen_spaced
            if style is None and self.model.cond_disc and image is not None and self.model.style_extractor is not None:
                style = self.model.extract_style(image,None,a_batch_size)
            if self.model.char_style_dim>0 and style is not None:

                if self.model.spaced_label is None:
                    self.model.spaced_label = correct_pred(self.model.pred,label)
                    self.model.spaced_label = self.model.onehot(self.model.spaced_label)
                real_style = self.model.space_style(self.model.spaced_label,style,image.device)
            else:
                real_style = style
            if self.model.spaced_label is None and image is not None:
                if self.model.pred is None:
                    self.model.pred = self.model.hwr(image)
                self.model.spaced_label = correct_pred(self.model.pred,label)
                self.model.spaced_label = self.model.onehot(self.model.spaced_label)
            real_label = self.model.spaced_label
            assert(not self.model.cond_disc or fake_label is not None)
        elif 'auto-gen' in lesson or 'auto-disc' in lesson:
            fake = recon
            if self.model.char_style_dim>0:
                fake_style = self.model.spaced_style
                real_style = self.model.spaced_style
            else:
                fake_style = style
                real_style = style
            fake_label = self.model.spaced_label
            real_label = self.model.spaced_label
            assert(fake_label is not None)
            assert(real_label is not None)
        elif 'sample-disc' in lesson:
            if self.model.style_extractor is not None:
                with torch.no_grad():
                    self.model.pred=None
                    style = self.model.extract_style(image,None,a_batch_size)
            elif self.model.pred is None:
                self.model.pred = self.model.hwr(image)
            spaced_label_m = correct_pred(self.model.pred,label)
            spaced_label_m = self.model.onehot(spaced_label_m)
            if self.model.char_style_dim>0:
                real_style = self.model.space_style(spaced_label_m,style,image.device)
            else:
                real_style = style
            real = image
            real_label = spaced_label_m

            fake,fake_label,fake_style,fake_author_vector = self.sample_gen(batch_size)
            if fake is None:
                return None
            fake = fake.to(image.device)

            if self.model.char_style_dim>0:
                fake_style = self.model.space_style(fake_label,fake_style,image.device)
                fake_style = [s.to(image.device) for s in fake_style]
            elif fake_style is not None:
                fake_style = fake_style.to(image.device)
            fake_label = fake_label.to(image.device)


        if 'char-disc' in lesson:
            if self.use_char_set_disc:

                chars_batch=[]
                chars_target=[]
                chars_target=torch.FloatTensor(2)
                for i in range(2):
                    charIdx = random.choice(self.char_idx_we_care_about)
                    num_fake_chars = random.randint(0,batch_size)
                    num_real_chars = batch_size-num_fake_chars
                    fake_chars = self.sample_gen_char(num_fake_chars,charIdx)
                    real_chars = self.sample_real_char(num_real_chars,charIdx)
                    if num_real_chars>0 and num_fake_chars>0:
                        chars_batch.append(torch.cat((fake_chars,real_chars),dim=0))
                    elif num_fake_chars==0:
                        chars_batch.append(real_chars)
                    elif num_real_chars==0:
                        chars_batch.append(fake_chars)
                    chars_target[i]=num_real_chars/batch_size


                char_pred = self.char_discriminator(chars_batch)
                
                losses['charDiscLoss'] = self.loss['charDiscSuper'](char_pred,chars_target)



        if 'disc' in lesson or 'auto-disc' in lesson or 'sample-disc' in lesson:
            #WHERE DISCRIMINATOR LOSS IS COMPUTED
            if fake.size(3)>image.size(3):
                diff = fake.size(3)-image.size(3)
                image = F.pad(image,(0,diff,0,0),'replicate')
            elif fake.size(3)<image.size(3):
                diff = -(fake.size(3)-image.size(3))
                fake = F.pad(fake,(0,diff,0,0),'replicate')
                #image = image[:,:,:,:-diff]
            ##DEBUG
            #for i in range(batch_size):
            #    im = ((1-image[i,0])*127).cpu().numpy().astype(np.uint8)
            #    cv2.imwrite('test/real{}.png'.format(i),im)
            #for i in range(batch_size):
            #    im = ((1-fake[i,0])*127).cpu().numpy().astype(np.uint8)
            #    cv2.imwrite('test/fake{}.png'.format(i),im)
            #print(lesson)
            #import pdb;pdb.set_trace()

            if self.model.cond_disc:
                if fake_label is None:
                    print('error, fake_label none, leson {}'.format(lesson))
                if real_label is None:
                    print('error, real_label none, leson {}'.format(lesson))
                if fake_label.size(0) > real_label.size(0):
                    diff = fake_label.size(0)-real_label.size(0)
                    real_label = real_label.permute(1,2,0)
                    real_label = F.pad(real_label,(0,diff),'replicate')
                    real_label = real_label.permute(2,0,1)
                elif fake_label.size(0) < real_label.size(0):
                    diff = -(fake_label.size(0)-real_label.size(0))
                    #real_label = real_label[:-diff]
                    fake_label = fake_label.permute(1,2,0)
                    fake_label = F.pad(fake_label,(0,diff),'replicate')
                    fake_label = fake_label.permute(2,0,1)
                if self.use_author_vector:
                    fake_author_vector = fake_author_vector.to(author_vector.device)
                    comb_author = torch.cat((author_vector,fake_author_vector),dim=0)
                else:
                    comb_author = None

                disc_input = torch.cat((image,fake.detach()),dim=0)
                discriminator_pred = self.model.discriminator(
                                    torch.cat((real_label,fake_label),dim=1).detach(),
                                    self.cat_styles((real_style,fake_style),True,length=real_label.size(0)),
                                    disc_input,
                                    author_vector=comb_author)
            else:
                discriminator_pred = self.model.discriminator(torch.cat((image,fake),dim=0).detach())
            #if self.style_recon: 
                #discriminator_pred, disc_style = discriminator_pred
                #disc_style_on_real = disc_style[0:image.size(0)]
                #disc_style_on_fake = disc_style[0:image.size(0)]
                #losses['style_reconLoss'] = self.loss['style_recon'](disc_style,style, **self.loss_params['style_recon'])
            if self.WGAN:
                #Improved W-GAN
                assert(len(discriminator_pred)==1)
                disc_pred_real = discriminator_pred[0][:image.size(0)].mean()
                disc_pred_fake = discriminator_pred[0][image.size(0):].mean()
                ep = torch.empty(batch_size).uniform_()
                ep = ep[:,None,None,None].expand(image.size(0),image.size(1),image.size(2),image.size(3)).cuda()
                hatImgs = ep*image.detach() + (1-ep)*fake.detach()
                hatImgs.requires_grad_(True)
                disc_interpolates = self.model.discriminator(None,None,hatImgs)[0]
                gradients = autograd.grad(outputs=disc_interpolates, inputs=hatImgs,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

                gradients = gradients.view(gradients.size(0), -1)                              
                grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10#gp_lambda

                disc_loss = disc_pred_fake - disc_pred_real + grad_penalty
            elif self.DCGAN:
                disc_loss=0
                for i in range(len(discriminator_pred)): #iterate over different disc losses
                    discriminator_pred_on_real = torch.sigmoid(discriminator_pred[i][:image.size(0)])
                    discriminator_pred_on_fake = torch.sigmoid(discriminator_pred[i][image.size(0):])
                    disc_loss += F.binary_cross_entropy(discriminator_pred_on_real,torch.ones_like(discriminator_pred_on_real)) +  F.binary_cross_entropy(discriminator_pred_on_fake,torch.zeros_like(discriminator_pred_on_fake))

            else:
                #hinge loss
                disc_loss=0
                for i in range(len(discriminator_pred)): #iterate over different disc losses
                    discriminator_pred_on_real = discriminator_pred[i][:image.size(0)]
                    discriminator_pred_on_fake = discriminator_pred[i][image.size(0):]
                    disc_loss += F.relu(1.0 - discriminator_pred_on_real).mean() + F.relu(1.0 + discriminator_pred_on_fake).mean()
                disc_loss /= len(discriminator_pred)

            losses['discriminatorLoss']=disc_loss
        if ('gen' in lesson or 'auto-gen' in lesson) and 'eval' not in lesson:
            #WHERE GENERATOR LOSS IS COMPUTED
            if self.model.cond_disc:
                gen_pred = self.model.discriminator(fake_label.detach(),self.cat_styles([fake_style],True),fake,author_vector=author_vector)
            else:
                gen_pred = self.model.discriminator(fake)
            gen_loss=0
            predicted_disc=[]
            if self.DCGAN:
                for gp in gen_pred:
                    gp=torch.sigmoid(gp)
                    gen_loss = F.binary_cross_entropy(gp,torch.ones_like(gp))
                    if 'disc' in get:
                        predicted_disc.append(gp.mean(dim=1).detach().cpu())
            else:
                for gp in gen_pred:
                    gen_loss -= gp.mean()
                    if 'disc' in get:
                        if len(gp.size())>1:
                            predicted_disc.append(gp.mean(dim=1).detach().cpu())
                        else:
                            predicted_disc.append(gp.detach().cpu())
            gen_loss/=len(gen_pred)
            losses['generatorLoss']=gen_loss
        else:
            predicted_disc=None

        if 'gen-style' in lesson and 'eval' not in lesson:
            assert(self.model.char_style_dim==0)
            gen_loss = -self.model.style_discriminator(style_gen).mean()
            losses['style_generatorLoss']=gen_loss
        if 'disc-style' in lesson and 'eval' not in lesson:
            assert(self.model.char_style_dim==0)
            if style is None:
                style = self.model.extract_style(image,label,a_batch_size)
            num_fake = style_gen.size(0)
            discriminator_pred = self.model.style_discriminator(torch.cat((style_gen,style),dim=0))
            discriminator_pred_on_fake = discriminator_pred[:num_fake]
            discriminator_pred_on_real = discriminator_pred[num_fake:]
            disc_loss = F.relu(1.0 - discriminator_pred_on_real).mean() + F.relu(1.0 + discriminator_pred_on_fake).mean()
            losses['style_discriminatorLoss']=disc_loss

        if 'recon-auto-style' in lesson: 
            style_recon = self.model.extract_style(recon,label,a_batch_size)
            if self.model.char_style_dim>0:
                losses['reconAutoStyleLoss'] = 0
                losses['reconAutoStyleLoss'] += self.loss['reconAutoStyle'](style_recon[0],style[0])
                losses['reconAutoStyleLoss'] += self.loss['reconAutoStyle'](style_recon[1],style[1])
                losses['reconAutoStyleLoss'] += self.loss['reconAutoStyle'](style_recon[2],style[2])
            else:
                losses['reconAutoStyleLoss'] = self.loss['reconAutoStyle'](style_recon,style)
        if 'gen-auto-style' in lesson and 'eval' not in lesson:
            style_recon = self.model.extract_style(gen_image,label,1)
            if self.model.char_style_dim>0:
                losses['genAutoStyleLoss'] = 0
                losses['genAutoStyleLoss'] += self.loss['genAutoStyle'](style_recon[0],style_gen[0])
                losses['genAutoStyleLoss'] += self.loss['genAutoStyle'](style_recon[1],style_gen[1])
                losses['genAutoStyleLoss'] += self.loss['genAutoStyle'](style_recon[2],style_gen[2])
            else:
                losses['genAutoStyleLoss'] = self.loss['genAutoStyle'](style_recon,style_gen)

        if  ('gen' in lesson or 'auto' in lesson) and 'key' in self.loss:
            losses['keyLoss'] = self.loss['key'](self.model.style_extractor.keys1,**self.loss_params['key'])

        if 'style-super' in lesson:
            if style_gen is None:
                style_gen = self.get_style_gen(batch_size,label.device)
            if gen_image is None:
                gen_image = self.model(label,label_lengths,style_gen)
            found_style = self.model.extract_style(gen_image.detach(),label,1)
            assert(self.model.char_style_dim==0)
            losses['styleSuperLoss'] = self.loss['styleSuper'](found_style,style_gen)

        if 'triplet-style' in lesson:
            if style is None:
                style = self.model.extract_style(image,label,a_batch_size)
            styleOne = style[::a_batch_size,...]
            #anchor = styleOne[::3]
            #truthy = styleOne[1::3]
            #falsy = styleOne[2::3]
            losses['tripletStyleLoss'] = self.loss['tripletStyle'](styleOne,instance['author'][::a_batch_size])

        if 'author-classify' in lesson and not ('valid' in lesson or 'eval' in lesson):
            author_pred = self.model.author_classifier(recon)
            losses['authorClassLoss'] = self.loss['authorClass'](author_pred,author_vector,**self.loss_params['authorClass'])
        if 'author-train' in lesson and not ('valid' in lesson or 'eval' in lesson):
            author_pred = self.model.author_classifier(image)
            losses['authorTrainLoss'] = self.loss['authorTrain'](author_pred,author_vector,**self.loss_params['authorTrain'])


        if get:
            if (len(get)>1 or get[0]=='style') and 'name' in instance:
                got={'name': instance['name']}
            else:
                got={}
            for name in get:
                if name=='recon':
                    got[name] = recon.cpu().detach()
                elif name=='recon_gt_mask':
                    got[name] = recon_gt_mask.cpu().detach()
                elif name=='recon_pred_space':
                    got[name] = recon_pred_space.cpu().detach()
                elif name=='gen_image':
                    got[name] = gen_image.cpu().detach()
                elif name=='gen_img':
                    got[name] = gen_image.cpu().detach()
                elif name=='gen':
                    if gen_image is not None:
                        got[name] = gen_image.cpu().detach()
                    else:
                        print('ERROR, gen_image is None, lesson: {}'.format(lesson))
                        #got[name] = None
                elif name=='pred':
                    if self.model.pred is None:
                        self.model.pred = self.model.hwr(image, None)   
                    got[name] = self.model.pred.cpu().detach()
                elif name=='spaced_label':
                    if self.model.spaced_label is None:
                        if self.model.pred is None:
                            self.model.pred = self.model.hwr(image, None)   
                        self.model.spaced_label = correct_pred(self.model.pred,label)
                        #self.model.spaced_label = self.model.onehot(self.model.spaced_label)
                    got[name] = self.model.spaced_label.cpu().detach()
                elif name=='mask':
                    got[name] = self.model.mask
                elif name=='gt_mask':
                    got[name] = gt_mask.cpu().detach()
                elif name=='gen_mask':
                    got[name] = self.model.gen_mask.cpu().detach()
                elif name=='style':
                    got[name] = style.cpu().detach()
                elif name=='author':
                    got[name] = instance['author']
                elif name=='gt':
                    got[name] = instance['gt']
                elif name=='disc':
                    got[name] = predicted_disc
                else:
                    raise ValueError("Unknown get [{}]".format(name))
            ret = (losses, got)
        else:
            ret = losses
        self.model.spaced_label=None
        self.model.mask=None
        self.model.gen_mask=None
        self.model.top_and_bottom=None
        self.model.counts=None
        self.model.pred=None
        self.model.spacing_pred=None
        self.model.mask_pred=None
        self.model.gen_spaced=None
        self.model.spaced_style=None
        self.model.mu=None
        self.model.sigma=None
        if 'feature' in self.loss:
            self.model.hwr.saved_features=[None]*len( self.model.hwr.saved_features)
        return ret

    def getCER(self,gt,pred,individual=False):
        cer=0
        wer=0
        if individual:
            all_cer=[]
        pred_strs=[]
        for i,gt_line in enumerate(gt):
            logits = pred[:,i]
            pred_str, raw_pred = string_utils.naive_decode(logits)
            pred_str = string_utils.label2str_single(pred_str, self.idx_to_char, False)
            this_cer = error_rates.cer(gt_line, pred_str, self.casesensitive)
            cer+=this_cer
            if individual:
                all_cer.append(this_cer)
            pred_strs.append(pred_str)
            wer += error_rates.wer(gt_line, pred_str, self.casesensitive)
        cer/=len(gt)
        wer/=len(gt)
        if individual:
            return cer,wer, pred_strs, all_cer
        return cer,wer, pred_strs

    def correct_pred(self,pred,label):
        #Get optimal alignment
        #use DTW
        # introduce blanks at front, back, and inbetween chars
        label_with_blanks = torch.LongTensor(label.size(0)*2+1, label.size(1)).zero_()
        label_with_blanks[1::2]=label.cpu()
        pred_use = pred.cpu().detach()

        batch_size=pred_use.size(1)
        label_len=label_with_blanks.size(0)
        pred_len=pred_use.size(0)

        dtw = torch.FloatTensor(pred_len+1,label_len+1,batch_size).fill_(float('inf'))
        dtw[0,0]=0
        w = max(pred_len//2, abs(pred_len-label_len))
        for i in range(1,pred_len+1):
            dtw[i,max(1, i-w):min(label_len, i+w)+1]=0
        history = torch.IntTensor(pred_len,label_len,batch_size)
        for i in range(1,pred_len+1):
            for j in range(max(1, i-w), min(label_len, i+w)+1):
                cost = 1-pred_use[i-1,torch.arange(0,batch_size).long(),label_with_blanks[j-1,:]]
                per_batch_min, history[i-1,j-1] = torch.min( torch.stack( (dtw[i-1,j],dtw[i-1,j-1],dtw[i,j-1]) ), dim=0)
                dtw[i,j] = cost + per_batch_min
        new_labels = []
        maxlen = 0
        for b in range(batch_size):
            new_label = []
            i=pred_len-1
            j=label_len-1
            #accum += allCosts[b,i,j]
            new_label.append(label_with_blanks[j,b])
            while(i>0 or j>0):
                if history[i,j,b]==0:
                    i-=1
                elif history[i,j,b]==1:
                    i-=1
                    j-=1
                elif history[i,j,b]==2:
                    j-=1
                #accum+=allCosts[b,i,j]
                new_label.append(label_with_blanks[j,b])
            new_label.reverse()
            maxlen = max(maxlen,len(new_label))
            new_label = torch.stack(new_label,dim=0)
            new_labels.append(new_label)

        new_labels = [ F.pad(l,(0,maxlen-l.size(0)),value=0) for l in new_labels]
        new_label = torch.LongTensor(maxlen,batch_size)
        for b,l in enumerate(new_labels):
            new_label[:l.size(0),b]=l

        #set to one hot at alignment
        #new_label = self.onehot(new_label)
        #fuzzy other neighbor preds
        #TODO

        return new_label.to(label.device)

    def get_style_gen(self,batch_size,device):
        if (self.interpolate_gen_styles and (len(self.prev_styles)>0 or len(self.prev_g_styles)>0)) and (not self.sometimes_interpolate or self.interpolate_freq>random.random()):
            if self.model.char_style_dim>0:
                if self.mix_style:
                    indexes_g = np.random.randint(0,len(self.prev_g_styles),(batch_size,2))
                    mix_g = np.random.uniform(self.interpolate_gen_styles_low,self.interpolate_gen_styles_high,batch_size)
                    indexes_spacing = np.random.randint(0,len(self.prev_g_styles),(batch_size,2))
                    mix_spacing = np.random.uniform(self.interpolate_gen_styles_low,self.interpolate_gen_styles_high,batch_size)
                    indexes_char = np.random.randint(0,len(self.prev_g_styles),(batch_size,2))
                    mix_char = np.random.uniform(self.interpolate_gen_styles_low,self.interpolate_gen_styles_high,batch_size)
                    new_styles_g=[]
                    new_styles_spacing=[]
                    new_styles_char=[]
                    for b in range(batch_size):
                        new_style_g = self.prev_g_styles[indexes_g[b,0]]*mix_g[b] + self.prev_g_styles[indexes_g[b,1]]*(1-mix_g[b])
                        new_styles_g.append(new_style_g)
                        new_style_spacing = self.prev_spacing_styles[indexes_spacing[b,0]]*mix_spacing[b] + self.prev_spacing_styles[indexes_spacing[b,1]]*(1-mix_spacing[b])
                        new_styles_spacing.append(new_style_spacing)
                        new_style_char = self.prev_char_styles[indexes_char[b,0]]*mix_char[b] + self.prev_char_styles[indexes_char[b,1]]*(1-mix_char[b])
                        new_styles_char.append(new_style_char)
                else:
                    indexes = np.random.randint(0,len(self.prev_g_styles),(batch_size,2))
                    mix = np.random.uniform(self.interpolate_gen_styles_low,self.interpolate_gen_styles_high,batch_size)
                    new_styles_g=[]
                    new_styles_spacing=[]
                    new_styles_char=[]
                    for b in range(batch_size):
                        new_style_g = self.prev_g_styles[indexes[b,0]]*mix[b] + self.prev_g_styles[indexes[b,1]]*(1-mix[b])
                        new_styles_g.append(new_style_g)
                        new_style_spacing = self.prev_spacing_styles[indexes[b,0]]*mix[b] + self.prev_spacing_styles[indexes[b,1]]*(1-mix[b])
                        new_styles_spacing.append(new_style_spacing)
                        new_style_char = self.prev_char_styles[indexes[b,0]]*mix[b] + self.prev_char_styles[indexes[b,1]]*(1-mix[b])
                        new_styles_char.append(new_style_char)
                style_g_gen = torch.stack(new_styles_g,dim=0).to(device)
                style_spacing_gen = torch.stack(new_styles_spacing,dim=0).to(device)
                style_char_gen = torch.stack(new_styles_char,dim=0).to(device)
                style_gen = (style_g_gen, style_spacing_gen, style_char_gen)
            else:
                indexes = np.random.randint(0,len(self.prev_styles),(batch_size,2))
                mix = np.random.uniform(self.interpolate_gen_styles_low,self.interpolate_gen_styles_high,batch_size)
                new_styles=[]
                for b in range(batch_size):
                    new_style = self.prev_styles[indexes[b,0]]*mix[b] + self.prev_styles[indexes[b,1]]*(1-mix[b])
                    new_styles.append(new_style)
                style_gen = torch.stack(new_styles,dim=0).to(device)
        elif self.model.vae or self.model.style_from_normal is None:
            if self.model.char_style_dim>0:
                style_g_gen = torch.FloatTensor(batch_size,self.model.style_dim).normal_().to(device)
                style_spacing_gen = torch.FloatTensor(batch_size,self.model.char_style_dim).normal_().to(device)
                style_char_gen = torch.FloatTensor(batch_size,self.model.num_class,self.model.char_style_dim).normal_().to(device)
                style_gen = (style_g_gen, style_spacing_gen, style_char_gen)
            else:
                style_gen = torch.FloatTensor(batch_size,self.model.style_dim).normal_().to(device)
        else:
            sample = torch.FloatTensor(batch_size,self.model.style_dim//2).normal_()
            style_gen = self.model.style_from_normal(sample.to(device))
        return style_gen

    def cat_styles(self,styles,detach=False,length=None):
        if styles[0] is None:
            return None
        if self.model.char_style_dim>0:
            g, spacing, char = zip(*styles)
            if length is not None:
                spacing = list(spacing)
                if (len(spacing[0])==3):
                    for i in range(len(spacing)):
                        if length > spacing[i].size(0):
                            diff = length-spacing[i].size(0)
                            spacing[i] = spacing[i].permute(1,2,0)
                            spacing[i] = F.pad(spacing[i],(0,diff),'replicate')
                            spacing[i] = spacing[i].permute(2,0,1)
                        elif length < spacing[i].size(0):
                            diff = -(length-spacing[i].size(0))
                            spacing[i] = spacing[i][:-diff]
                    
                if (len(spacing[0])==3):
                    if detach:
                        return (torch.cat(g,dim=0).detach(), torch.cat(spacing,dim=1).detach(), torch.cat(char,dim=0).detach())
                    else:
                        return (torch.cat(g,dim=0), torch.cat(spacing,dim=1), torch.cat(char,dim=0))
                else:
                    if detach:
                        return (torch.cat(g,dim=0).detach(), torch.cat(spacing,dim=0).detach(), torch.cat(char,dim=0).detach())
                    else:
                        return (torch.cat(g,dim=0), torch.cat(spacing,dim=0), torch.cat(char,dim=0))
            else:
                if (len(spacing[0])==3):
                    if detach:
                        return (torch.cat(g,dim=0).detach(), torch.cat(spacing,dim=1).detach(), torch.cat(char,dim=0).detach())
                    else:
                        return (torch.cat(g,dim=0), torch.cat(spacing,dim=1), torch.cat(char,dim=0))
                else:
                    if detach:
                        return (torch.cat(g,dim=0).detach(), torch.cat(spacing,dim=0).detach(), torch.cat(char,dim=0).detach())
                    else:
                        return (torch.cat(g,dim=0), torch.cat(spacing,dim=0), torch.cat(char,dim=0))
        else:
            
            catted=  torch.cat(styles,dim=0)
            if detach:
                return catted.detach()
            else:
                return catted

    def sample_gen(self,batch_size):
        images=[]
        labels=[]
        styles=[]
        authors=[]

        max_w=0
        max_l=0
        for b in range(batch_size):
            if (random.random()<self.new_gen_freq or len(self.old_gen)<10) and len(self.new_gen)>0:
                #new
                if self.use_author_vector:
                    image,label,style,author= self.new_gen[0]
                else:
                    image,label,style= self.new_gen[0]
                self.new_gen = self.new_gen[1:]
            elif  len(self.old_gen)>0:
                i = random.randint(0,len(self.old_gen)-1)
                if self.use_author_vector:
                    if self.old_gen_cache is not None:
                        image,label,style,author = torch.load(self.old_gen[i])
                    else:
                        image,label,style,author = self.old_gen[i]
                else:
                    if self.old_gen_cache is not None:
                        image,label,style = torch.load(self.old_gen[i])
                    else:
                        image,label,style = self.old_gen[i]
            else:
                return None,None,None
            images.append(image)
            max_w = max(max_w,image.size(3))
            labels.append(label)
            max_l = max(max_l,label.size(0))
            styles.append(style)
            if self.use_author_vector:
                assert(author.size(0)==1)
                authors.append(author)
        for b in range(batch_size):
            if images[b].size(3)<max_w:
                diff = max_w -  images[b].size(3)
                images[b] = F.pad( images[b], (0,diff),value=PADDING_CONSTANT)
            if labels[b].size(0)<max_l:
                diff = max_l -  labels[b].size(0)
                labels[b] = F.pad( labels[b].permute(1,2,0), (0,diff),value=PADDING_CONSTANT).permute(2,0,1)
        assert(len(images)==batch_size)
        assert(len(styles)==batch_size)
        return torch.cat(images,dim=0), torch.cat(labels,dim=1), self.cat_styles(styles), torch.cat(authors) if len(authors)>0 else None

    def add_gen_sample(self,images,labels,styles,authors=None):
        w = images.size(2)
        hw=w//2
        batch_size = images.size(0)
        images = images.cpu().detach()
        labels = labels.cpu().detach()
        if self.model.char_style_dim>0:
            styles = (styles[0].cpu().detach(), styles[1].cpu().detach(), styles[2].cpu().detach())
        else:
            styles = styles.cpu().detach()

        for b in range(batch_size):
            if self.model.char_style_dim>0:
                #assert(len(styles[1].size())==3)
                #inst = (images[b:b+1],labels[:,b:b+1],(styles[0][b:b+1],styles[1][:,b:b+1],styles[2][b:b+1]))
                assert(len(styles[1].size())==2)
                inst = (images[b:b+1],labels[:,b:b+1],(styles[0][b:b+1],styles[1][b:b+1],styles[2][b:b+1]))
            elif self.use_author_vector:
                inst = (images[b:b+1],labels[:,b:b+1],styles[b:b+1],authors[b:b+1])
            else:
                inst = (images[b:b+1],labels[:,b:b+1],styles[b:b+1])
            if len(self.new_gen)>= self.store_new_gen_limit:
                old = self.new_gen[0]
                self.new_gen = self.new_gen[1:]+[inst]

                if len(self.old_gen)>= self.store_old_gen_limit:
                    if random.random() > self.forget_new_freq:
                        change = random.randint(0,len(self.old_gen)-1)
                        if self.old_gen_cache is not None:
                            torch.save(old,self.old_gen[change])
                        else:
                            self.old_gen[change] = old
                else:
                    if self.old_gen_cache is not None:
                        path = os.path.join(self.old_gen_cache,'{}.pt'.format(len(self.old_gen)))
                        torch.save(old,path)
                        self.old_gen.append(path)
                    else:
                        self.old_gen.append(old)
            else:
                self.new_gen.append(inst)

            if self.use_char_set_disc: #started, but not finished
                chars = labels[:,b].argmax(1)
                positions = torch.range(chars.size(0))[chars>0]
                #positions = (0.5+positions)*scale
                scale = images.size(3)/chars.size(0)
                for pos in positions:
                    charIdx = chars[pos].item()
                    if charIdx in self.char_idx_we_care_about:
                        x=(0.5+pos)*scale
                        #take square crop around location of character
                        imgPatch = images[b:b+1,:,:,x-hw:x+hw] #might be short if at edge of image
                        if len(self.new_gen_char[charIdx])>= self.store_new_gen_limit:
                            old = self.new_gen_char[charIdx][0]
                            self.new_gen_char[charIdx] = self.new_gen_char[charIdx][1:]+[char_inst]

                            if len(self.old_gen_char[charIdx])>= self.store_old_gen_limit:
                                if random.random() > self.forget_new_freq:
                                    change = random.randint(0,len(self.old_gen_char[charIdx])-1)
                                    if self.old_gen_char[charIdx] is not None:
                                        torch.save(old,self.old_gen_char[charIdx][change])
                                    else:
                                        self.old_gen_char[charIdx][change] = old
                            else:
                                if self.old_gen_cache is not None:
                                    path = os.path.join(self.old_gen_cache,'{}_{}.pt'.format(charIdx,len(self.old_gen_char[charIdx])))
                                    torch.save(old,path)
                                    self.old_gen_char[charIdx].append(path)
                                else:
                                    self.old_gen_char[charIdx].append(old)
                        else:
                            self.new_gen_char[charIdx].append(inst)

    def print_images(self,images,text,disc=None,typ='gen',gtImages=None):
        if self.print_dir is not None:
            images = 1-images.detach()
            nrow = max(1,2048//images.size(3))
            if self.iteration-self.last_print_images[typ]>=self.serperate_print_every:
                iterP = self.iteration
                self.last_print_images[typ]=self.iteration
                #vutils.save_image(images,
                #        os.path.join(self.print_dir,'{}_samples_{}.png'.format(typ,self.iteration)),
                #        nrow=nrow,
                #        normalize=True)
                #with open(os.path.join(self.print_dir,'{}_text_{}.txt'.format(typ,self.iteration)),'w') as f:
                #    if disc is None or len(disc)==0:
                #        f.write('\n'.join(text))
                #    else:
                #        for i,t in enumerate(text):
                #            f.write(t)
                #            for v in disc:
                #                f.write(', {}'.format(v[i].mean().item()))
                #            f.write('\n')
                #self.last_print_images[typ]=self.iteration
            else:
                iterP = 'latest'
                #vutils.save_image(images.detach(),
                #        os.path.join(self.print_dir,'{}_samples_latest.png'.format(typ)),
                #        nrow=nrow,
                #        normalize=True)
                ##images = ((1-images[:,0])*128).cpu().numpy().astype(np.uint8)
                ##for b in range(images.shape[0]):
                ##    path = os.path.join(self.print_dir,'gen_{}.png'.format(b))
                ##    cv2.imwrite(path,images[b])
                #with open(os.path.join(self.print_dir,'{}_text_latest.txt'.format(typ)),'w') as f:
                #    if disc is None or len(disc)==0:
                #        f.write('\n'.join(text))
                #    else:
                #        for i,t in enumerate(text):
                #            f.write(t)
                #            for v in disc:
                #                f.write(', {}'.format(v[i].mean().item()))
                #            f.write('\n')
            vutils.save_image(images,
                    os.path.join(self.print_dir,'{}_samples_{}.png'.format(typ,iterP)),
                    nrow=nrow,
                    normalize=True)
            if gtImages is not None:
                gtImages = 1-gtImages.detach()
                vutils.save_image(gtImages,
                        os.path.join(self.print_dir,'{}_gt_{}.png'.format(typ,iterP)),
                        nrow=nrow,
                        normalize=True)
            if typ=='gen': #reduce clutter, GT should be visible from GT image
                with open(os.path.join(self.print_dir,'{}_text_{}.txt'.format(typ,iterP)),'w') as f:
                    if disc is None or len(disc)==0:
                        f.write('\n'.join(text))
                    else:
                        for i,t in enumerate(text):
                            f.write(t)
                            for v in disc:
                                if i < v.size(0):
                                    f.write(', {}'.format(v[i].mean().item()))
                            f.write('\n')
            print('printed {} images, iter: {}'.format(typ,self.iteration))

