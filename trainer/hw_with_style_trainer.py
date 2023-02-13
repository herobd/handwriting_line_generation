import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from base import BaseTrainer
import timeit
from utils import util, string_utils, error_rates
from data_loader import getDataLoader
from collections import defaultdict
import random, json, os
from datasets.hw_dataset import PADDING_CONSTANT
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
        This is both for the HWR (pre)training and the GAN training.
        The main difference is whether it has a curriculum or not (curriculum is needed for GAN/autoencoder stuff).

    """
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(HWWithStyleTrainer, self).__init__(model, loss, metrics, resume, config, train_logger)
        self.config = config

        #Get losses set up
        if 'loss_params' in config:
            self.loss_params=config['loss_params']
        else:
            self.loss_params={}
        for lossname in self.loss:
            if lossname not in self.loss_params:
                self.loss_params[lossname]={}
        self.lossWeights = config['loss_weights'] if 'loss_weights' in config else {"auto": 1, "recog": 1}

        #set up data
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



        self.skip_hwr = config['trainer']['skip_hwr'] if 'skip_hwr' in config['trainer'] else False
        self.skip_auto = config['trainer']['skip_auto'] if 'skip_auto' in config['trainer'] else False

        self.to_display={}

        self.style_recon=False


        self.gan_loss = 'discriminator' in config['model']
        self.disc_iters = config['trainer']['disc_iters'] if 'disc_iters' in config['trainer'] else None


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


        self.style_detach = config['trainer']['detach_style'] if 'detach_style' in config['trainer'] else (config['trainer']['style_detach'] if 'style_detach' in config['trainer'] else False)

        #setup for keeping history of style vectors
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


        self.no_bg_loss= config['trainer']['no_bg_loss'] if 'no_bg_loss' in config else False

        

        self.use_char_set_disc = False


        #Setup encoder for perceptual loss
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


        #set up saving images during training
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
            return met
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
        if self.curriculum:
            lesson =  self.curriculum.getLesson(iteration)
        if self.curriculum and all([l[:3]=='gen' or l=='no-step' for l in lesson]) and self.text_data is not None:
            instance = self.text_data.getInstance()
        else:
            try:
                instance = self.data_loader_iter.next()
            except StopIteration:
                if 'refresh_data' in dir(self.data_loader.dataset):
                    self.data_loader.dataset.refresh_data(None,None,self.logged)
                self.data_loader_iter = iter(self.data_loader)
                instance = self.data_loader_iter.next()

        self.optimizer.zero_grad()
        if self.curriculum:
            if any(['disc' in l for l in lesson]):
                self.optimizer_discriminator.zero_grad()


        if self.curriculum:
            #Do GAN training
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
                losses,got = self.run_gen(instance,lesson,get=['recon'])
                self.print_images(got['recon'],instance['gt'],typ='recon',gtImages=instance['image'])
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
            #Do HWR training
            pred,  losses = self.run_hwr(instance)
            recon = None

        if losses is None:
            return {}

        loss=0
        recogLoss=0
        autoGenLoss=0
        for name in losses.keys():
            losses[name] *= self.lossWeights[name[:-4]]
            if self.balance_loss and 'generator' in name and 'auto-gen' in lesson:
                autoGenLoss += losses[name]
            elif self.balance_loss and 'Recog' in name:
                recogLoss += losses[name]
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
        else:
            loss += recogLoss+autoGenLoss


        if type(loss) is not int:
            loss.backward()

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

            #in case of zero mean
            if nonzero_count>0:
                nonzero=nonzero_sum/nonzero_count
                for i in range(len(abmean_Ds)):
                    if abmean_Ds[i]==0.0:
                        abmean_Ds[i]=nonzero

            #get the right multipliers for this iteration
            for iterT,mult in self.balance_var_x.items():
                if int(iterT)<=iteration:
                    multipliers=mult
                    if type(multipliers) is not list:
                        multipliers=[multipliers]

            #actually change the grandients
            for gi,saved_grad in enumerate(self.saved_grads):
                x=multipliers[gi]
                for i,(R, p) in enumerate(zip(saved_grad, self.parameters)):
                    if R is not None:
                        assert(not torch.isnan(p.grad).any())
                        abmean_R = torch.abs(R).mean()
                        if abmean_R!=0:
                            p.grad += x*R*(abmean_Ds[i]/abmean_R)
            self.saved_grads=[]

        if self.curriculum and 'no-step' not in lesson:
            #Do an optimizer step with accumulated+balanced grandients
            torch.nn.utils.clip_grad_value_(self.model.parameters(),2)
            for m in self.model.parameters():
                assert(not torch.isnan(m).any())

            if 'disc' in lesson or 'auto-disc' in lesson:
                self.optimizer_discriminator.step()
            else:
                self.optimizer.step()
        elif not self.curriculum:
            #HWR pre-training optimizer step
            self.optimizer.step()


        loss = loss_item

        gt = instance['gt']
        if pred is not None:
            cer,wer, pred_str = self.getCER(gt,pred)
        else:
            cer=0
            wer=0

        metrics={}


        log = {
            'loss': loss,
            **losses,

            'CER': cer,
            'WER': wer,

            **metrics,
        }



        return log


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
                    pred, recon, losses = self.run_hwr(instance)
            
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
        return label_onehot.to(label.device)

    #Run training pass for HWR
    def run_hwr(self,instance):
        image, label = self._to_tensor(instance)
        label_lengths = instance['label_lengths']
        
        losses = {}
        pred = self.model.hwr(image, None)

        batch_size = pred.size(1)
        pred_size = torch.IntTensor([pred.size(0)] * batch_size)
        recogLoss = self.loss['recog'](pred,label.permute(1,0),pred_size,label_lengths)
        if torch.isinf(recogLoss).any():
            recogLoss = 0
        else:
            losses['recogLoss']=recogLoss


        return pred, losses


    #run training pass for GAN
    def run_gen(self,instance,lesson,get=[]):
        image, label = self._to_tensor(instance)
        batch_size = label.size(1)
        label_lengths = instance['label_lengths']
        a_batch_size = self.a_batch_size if 'a_batch_size' in instance else None


        losses = {}




        #CACHED spaced_label
        need_spaced_label = any([x in lesson for x in ['count','auto','disc']])
        if need_spaced_label:
            if instance['spaced_label'] is not None:
                self.model.spaced_label = self.model.onehot(instance['spaced_label']).to(label.device)

        if 'auto' in lesson:
            #Running autoencoder

            if 'recon_pred_space' in get:
                style = self.model.extract_style(image,label,a_batch_size)

                recon_pred_space = self.model(label,label_lengths,style)


            if 'eval' not in lesson or 'recon' in get:
                recon,style = self.model.autoencode(image,label,a_batch_size)
                if 'styleReg' in self.loss:
                    losses['styleRegLoss'] = self.loss['styleReg'](style,**self.loss_params['styleReg'])



            if self.interpolate_gen_styles and 'eval' not in lesson and 'valid' not in lesson:
                for i in range(0,batch_size,a_batch_size):
                    self.prev_styles.append(style[i].detach().cpu())
                self.prev_styles =self.prev_styles[-self.prev_styles_size:]
        else:
            style=None
            recon=None

        if 'gen' in lesson or 'disc' in lesson or 'gen' in get:
            #Get style vector for pure generative
            if 'eval' not in lesson and 'valid' not in lesson or not self.interpolate_gen_styles:
                style_gen = self.get_style_gen(batch_size,label.device)
            else:
                #During evaluation, we don't have a bank of previous styles, so we use the previous and current
                
                style_gen = torch.empty_like(style)
                for i in range(batch_size//a_batch_size):
                    b=i*a_batch_size
                    b_1 = ((i+1)%(batch_size//a_batch_size))*a_batch_size
                    style_gen[b:b+a_batch_size] = 0.5*style[b_1:b_1+a_batch_size] + 0.5*style[b:b+a_batch_size]


            #Generate!
            if 'eval' not in lesson and label.size(0)>self.text_data.max_len:
                if 'auto' not in lesson:
                    label=label[:self.text_data.max_len]
                
                for b in range(batch_size):
                    label_lengths[b] = min(label_lengths[b],self.text_data.max_len)
            gen_image = self.model(label,label_lengths,style_gen)
        else:
            style_gen = None
            gen_image = None



        if 'auto' in lesson and 'auto' in self.loss and 'eval' not in lesson:
            #Pad the reconstructed image to match original image
            paddedImage=False
            if recon.size(3)>image.size(3):
                toPad = recon.size(3)-image.size(3)
                paddedImage=True
                image = F.pad(image,(0,toPad),value=PADDING_CONSTANT)
            elif recon.size(3)<image.size(3):
                toPad = image.size(3)-recon.size(3)
                if toPad>50:
                    print('WARNING image {} bigger than recon {}'.format(image.size(3),recon.size(3)))
                recon = F.pad(recon,(0,toPad),value=PADDING_CONSTANT)
            if self.no_bg_loss:
                fg_mask = instance['fg_mask']
                if paddedImage:
                    fg_mask = F.pad(fg_mask,(0,toPad),value=0)
                recon_autol = recon*fg_mask
                image_autol = image*fg_mask
            else:
                recon_autol = recon
                image_autol = image

            #Compute reconstruction loss
            autoLoss = self.loss['auto'](recon_autol,image_autol,**self.loss_params['auto'])
            if type(autoLoss) is tuple:
                autoLoss, autoLossScales = autoLoss
            losses['autoLoss']=autoLoss
            assert not torch.isnan(autoLoss)




        if 'count' in lesson  and 'count' in self.loss and 'eval' not in lesson:
            #The spacing module
            if 'auto' not in lesson:
                style = self.model.extract_style(image,label,a_batch_size)
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

                
                spaced_label_m = correct_pred(self.model.pred,label)
                spaced_label_m = self.model.onehot(spaced_label_m)
            else:
                spaced_label_m = self.model.spaced_label
            
            label_onehot=self.model.onehot(label)
            if self.style_detach:
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
                    self.model.counts[pos:]=0

            assert not torch.isinf(self.model.counts).any()
            gt_counts = gt_counts.to(self.model.counts.device)
            losses['countLoss'] = self.loss['count'](self.model.counts,gt_counts,**self.loss_params['count'])
            assert not torch.isnan(losses['countLoss']) and not torch.isinf(losses['countLoss'])



        recon_pred = None

        if 'auto' in lesson and 'perceptual' in self.loss and 'eval' not in lesson:
            #Perceptual loss for autoencoding
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
            
            perceptual_loss = 0
            for r_f,o_f in zip(recon_features,orig_features):
                perceptual_loss += self.loss['perceptual'](r_f,o_f,**self.loss_params['perceptual'])
            losses['perceptualLoss']=perceptual_loss 


        #Generation guiding recognition losses
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

            else:
                fake = gen_image
            
        elif 'auto-gen' in lesson:
            fake = recon




        if 'disc' in lesson:
            #WHERE DISCRIMINATOR LOSS IS COMPUTED
            if fake.size(3)>image.size(3):
                diff = fake.size(3)-image.size(3)
                image = F.pad(image,(0,diff,0,0),'replicate')
            elif fake.size(3)<image.size(3):
                diff = -(fake.size(3)-image.size(3))
                fake = F.pad(fake,(0,diff,0,0),'replicate')

            discriminator_pred = self.model.discriminator(torch.cat((image,fake.detach()),dim=0))
            #hinge loss
            disc_loss=0
            for i in range(len(discriminator_pred)): #iterate over different disc losses (resolutions)
                discriminator_pred_on_real = discriminator_pred[i][:image.size(0)]
                discriminator_pred_on_fake = discriminator_pred[i][image.size(0):]
                disc_loss += F.relu(1.0 - discriminator_pred_on_real).mean() + F.relu(1.0 + discriminator_pred_on_fake).mean()
            disc_loss /= len(discriminator_pred)

            losses['discriminatorLoss']=disc_loss

        if ('gen' in lesson or 'auto-gen' in lesson) and 'eval' not in lesson:
            #WHERE GENERATOR LOSS IS COMPUTED
            gen_pred = self.model.discriminator(fake)
            gen_loss=0
            predicted_disc=[]
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
            indexes = np.random.randint(0,len(self.prev_styles),(batch_size,2))
            mix = np.random.uniform(self.interpolate_gen_styles_low,self.interpolate_gen_styles_high,batch_size)
            new_styles=[]
            for b in range(batch_size):
                new_style = self.prev_styles[indexes[b,0]]*mix[b] + self.prev_styles[indexes[b,1]]*(1-mix[b])
                new_styles.append(new_style)
            style_gen = torch.stack(new_styles,dim=0).to(device)
        elif self.model.vae or self.model.style_from_normal is None:
            style_gen = torch.FloatTensor(batch_size,self.model.style_dim).normal_().to(device)
        else:
            sample = torch.FloatTensor(batch_size,self.model.style_dim//2).normal_()
            style_gen = self.model.style_from_normal(sample.to(device))
        return style_gen



    def print_images(self,images,text,disc=None,typ='gen',gtImages=None):
        if self.print_dir is not None:
            images = 1-images.detach()
            nrow = max(1,2048//images.size(3))
            if self.iteration-self.last_print_images[typ]>=self.serperate_print_every:
                iterP = self.iteration
                self.last_print_images[typ]=self.iteration
            else:
                iterP = 'latest'
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

