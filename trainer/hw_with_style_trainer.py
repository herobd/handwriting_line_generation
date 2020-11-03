# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from base import BaseTrainer
import timeit
from utils import util, string_utils, error_rates
from collections import defaultdict
import random, json
from datasets.hw_dataset import PADDING_CONSTANT
from model.clear_grad import ClearGrad
from datasets.text_data import TextData

class HWWithStyleTrainer(BaseTrainer):
    # AdobePatentID="P9297-US"
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
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.data_loader_iter = iter(data_loader)
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False

        char_set_path = config['data_loader']['char_file']
        with open(char_set_path) as f:
            char_set = json.load(f)
        self.idx_to_char = {}
        self.num_class = len(char_set['idx_to_char'])+1
        for k,v in char_set['idx_to_char'].items():
            self.idx_to_char[int(k)] = v

        self.align_loss = 'align' in config['loss']['auto'] if 'auto' in config['loss'] else False

        self.skip_hwr = config['trainer']['skip_hwr'] if 'skip_hwr' in config['trainer'] else False
        self.skip_auto = config['trainer']['skip_auto'] if 'skip_auto' in config['trainer'] else False
        self.style_hwr = 'hwr' in config['model'] and 'Style' in config['model']['hwr']
        self.center_pad = config['data_loader']['center_pad'] if 'center_pad' in config['data_loader'] else True

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
        if 'a_batch_size' in self.config['data_loader']:
            text_data_batch_size*=self.config['data_loader']['a_batch_size']
        self.text_data = TextData(config['trainer']['text_data'],config['data_loader']['char_file'],text_data_batch_size) if 'text_data' in config['trainer'] else None
        self.balance_loss = config['trainer']['balance_loss'] if 'balance_loss' in config['trainer'] else False # balance the CTC loss with others as in https://arxiv.org/pdf/1903.00277.pdf
        if self.balance_loss:
            self.parameters = list(model.parameters())

        self.style_detach = config['trainer']['detach_style'] if 'detach_style' in config['trainer'] else (config['trainer']['style_detach'] if 'style_detach' in config['trainer'] else False)
        self.spaced_label_cache={}

        self.interpolate_gen_styles = config['trainer']['interpolate_gen_styles'] if 'interpolate_gen_styles' in config['trainer'] else False
        if type(self.interpolate_gen_styles) is str and self.interpolate_gen_styles[:6] == 'extra-':
            self.interpolate_gen_styles_low = -float(self.interpolate_gen_styles[6:])
            self.interpolate_gen_styles_high = 1+float(self.interpolate_gen_styles[6:])
        else:
            self.interpolate_gen_styles_low=0
            self.interpolate_gen_styles_high=1
        self.prev_styles_size = config['trainer']['prev_style_size'] if 'prev_style_size' in config['trainer'] else 100
        self.prev_styles = []

        if 'align_network' in config['trainer']:
            self.align_network = JoinNet()
            weights = config['trainer']['align_network']
            state_dict=torch.load(config['trainer']['align_network'], map_location=lambda storage, location: storage)
            self.align_network.load_state_dict(state_dict)
            self.align_network.set_requires_grad(False)

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
        if self.curriculum and all([l[:3]=='gen' for l in lesson]) and self.text_data is not None:
            instance = self.text_data.getInstance()
        else:
            try:
                instance = self.data_loader_iter.next()
            except StopIteration:
                self.data_loader_iter = iter(self.data_loader)
                instance = self.data_loader_iter.next()
        ##toc=timeit.default_timer()
        ##print('data: '+str(toc-tic))
        
        ##tic=timeit.default_timer()

        self.optimizer.zero_grad()

        ##toc=timeit.default_timer()
        ##print('for: '+str(toc-tic))

        ##tic=timeit.default_timer()
        if self.curriculum:
            losses = self.run_gen(instance,lesson)
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
    
        loss=0
        recogLoss=0
        for name in losses.keys():
            losses[name] *= self.lossWeights[name[:-4]]
            if self.balance_loss and 'Recog' in name:
                recogLoss += losses[name]
            else:
                loss += losses[name]
            losses[name] = losses[name].item()
        assert(not torch.isnan(loss) and not torch.isinf(loss))
        if pred is not None:
            pred = pred.detach().cpu().numpy()
        loss_item = loss.item()
        saved_grad=[]
        if self.balance_loss and type(recogLoss) is not int:
            loss_item += recogLoss.item()
            recogLoss.backward(retain_graph=True)
            for p in self.parameters:
                if p.grad is None:
                    saved_grad.append(None)
                else:
                    saved_grad.append(p.grad.clone())
                    p.grad.zero_()
        loss.backward()

        if self.balance_loss:
            for R, p in zip(saved_grad, self.parameters):
                if R is not None:
                    if self.balance_loss=='sign_preserve':
                        abmean_D = torch.abs(p.grad).mean()
                        abmean_R = torch.abs(p.grad).mean()
                        if abmean_R!=0:
                            p.grad += R*(abmean_D/abmean_R)
                    elif self.balance_loss=='sign_match':
                        match_pos = (p.grad>0)*(R>0)
                        match_neg = (p.grad<0)*(R<0)
                        not_match = ~(match_pos|match_neg)
                        p.grad[not_match] = 0 #zero out where signs don't match
                    else:
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
                    assert(not torch.isnan(p.grad).any())
            saved_grad=None
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad[torch.isnan(p.grad)]=0


        torch.nn.utils.clip_grad_value_(self.model.parameters(),2)
        meangrad=0
        count=0
        for m in self.model.parameters():
            if m.grad is None:
                continue
            count+=1
            meangrad+=m.grad.data.mean()
        meangrad/=count

        if self.curriculum:
            if 'disc' in lesson or 'auto-disc' in lesson or 'disc-style' in lesson:
                self.optimizer_discriminator.step()
            elif any(['auto-style' in l for l in lesson]):
                self.optimizer_gen_only.step()
            else:
                self.optimizer.step()
        else:
            if iteration%(1+self.disc_iters)==0 or not self.gan_loss:
                self.optimizer.step()
            else:
                self.optimizer_discriminator.step()
        loss = loss_item

        gt = instance['gt']
        if pred is not None:
            cer, pred_str = self.getCER(gt,pred)
        else:
            cer=0

        ##toc=timeit.default_timer()
        ##print('bac: '+str(toc-tic))

        #tic=timeit.default_timer()
        metrics={}


        log = {
            'loss': loss,
            **losses,
            #'pred_str': pred_str

            'CER': cer,
            'meangrad': meangrad,

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
                    total_cer += self.getCER(gt,pred)[0]
        
        for name in total_losses.keys():
            total_losses[name]/=len(self.valid_data_loader)
        toRet={
                'val_loss': total_loss/len(self.valid_data_loader),
                'val_CER': total_cer/len(self.valid_data_loader),
                **total_losses
                }
        return toRet

    def onehot(self,label):
        label_onehot = torch.zeros(label.size(0),label.size(1),self.num_class)
        #label_onehot[label]=1
        #TODO tensorize
        for i in range(label.size(0)):
            for j in range(label.size(1)):
                label_onehot[i,j,label[i,j]]=1
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
                    a_batch_size = instance['a_batch_size']
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
                #import pdb;pdb.set_trace()
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
                #import pdb;pdb.set_trace()
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
                    if toPad>5:
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

        if 'recon_gt_mask' in get: #this is just to be able to see both
            lesson.append('not-auto-mask')

        #CACHED spaced_label
        #print('lesson: {}'.format(lesson))
        if 'count' in lesson or 'mask' in lesson or  'auto' in lesson: #we need spaced_label
            if instance['spaced_label'] is not None:
                self.model.spaced_label = self.model.onehot(instance['spaced_label']).to(label.device)
            else:
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
                    corrected = self.model.correct_pred(to_correct,to_correct_label)
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

        if 'auto' in lesson or 'auto-disc' in lesson:
            a_batch_size = instance['a_batch_size'] if 'a_batch_size' in instance else None
            if 'mask' in instance and instance['mask'] is not None and 'auto-mask' not in lesson:
                mask = instance['mask'].to(image.device)
            else:
                mask = None
            if 'not-auto-mask' in lesson:
                recon_gt_mask,style_ = self.model.autoencode(image,label,mask,a_batch_size,center_line=None)
            if 'auto-mask' in lesson:
                center_line = instance['center_line']
            else:
                center_line = None
            recon,style = self.model.autoencode(image,label,mask,a_batch_size,center_line=center_line)

            if self.interpolate_gen_styles:
                for i in range(0,batch_size,a_batch_size):
                    self.prev_styles.append(style[i].detach().cpu())
                self.prev_styles =self.prev_styles[:self.prev_styles_size]

        else:
            style=None
            recon=None

        if 'gen' in lesson or 'disc' in lesson or 'gen-style' in lesson or 'disc-style' in lesson:
            style_gen = self.get_style_gen(batch_size,label.device)
        else:
            style_gen = None
        if 'gen' in lesson or 'disc' in lesson:
            gen_image = self.model(label,label_lengths,style_gen)
        else:
            gen_image = None

        losses = {}

        if 'auto' in lesson and 'auto' in self.loss:
            if recon.size(3)>image.size(3):
                toPad = recon.size(3)-image.size(3)
                if self.center_pad:
                    image = F.pad(image,(toPad//2,toPad//2 + toPad%2),value=PADDING_CONSTANT)
                else:
                    image = F.pad(image,(0,toPad),value=PADDING_CONSTANT)
            elif recon.size(3)<image.size(3):
                toPad = image.size(3)-recon.size(3)
                if toPad>5:
                    print('WARNING image {} bigger than recon {}'.format(image.size(3),recon.size(3)))
                if self.center_pad:
                    recon = F.pad(recon,(toPad//2,toPad//2 + toPad%2),value=PADDING_CONSTANT)
                else:
                    recon = F.pad(recon,(0,toPad),value=PADDING_CONSTANT)
            autoLoss = self.loss['auto'](recon,image,**self.loss_params['auto'])
            if type(autoLoss) is tuple:
                autoLoss, autoLossScales = autoLoss
            losses['autoLoss']=autoLoss
        if 'align-auto' in lesson:
            aligned_recon = self.align_network(recon,image)
            autoLoss = self.loss['alignAuto'](recon,image,**self.loss_params['alignAuto'])
            if type(autoLoss) is tuple:
                autoLoss, autoLossScales = autoLoss
            losses['alignAutoLoss']=autoLoss

        if 'spacing' in lesson  and ('spacingDirect' in self.loss or 'spacingCTC' in self.loss):
            if  self.model.spacing_pred is None:
                a_batch_size = instance['a_batch_size'] if 'a_batch_size' in instance else None
                mask = None
                center_line = instance['center_line']
                recon,style = self.model.autoencode(image,label,mask,a_batch_size,center_line=center_line)
            if self.model.spaced_label is None:
                self.model.spaced_label = self.model.correct_pred(self.model.pred,label)
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
                a_batch_size = instance['a_batch_size'] if 'a_batch_size' in instance else None
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

        if 'count' in lesson  and 'count' in self.loss:
            if 'auto' not in lesson:
                #pred = self.model.hwr(image, None)
                #self.model.spaced_label = self.model.correct_pred(pred,label)
                #self.model.spaced_label = self.model.onehot(self.model.spaced_label)
                a_batch_size = instance['a_batch_size'] if 'a_batch_size' in instance else None
                style = self.model.extract_style(image,label,a_batch_size)
            #if 'gen' not in lesson:
            label_onehot=self.model.onehot(label)
            if self.style_detach:
                style_d = style.detach()
            else:
                style_d=style
            self.model.counts = self.model.spacer(label_onehot,style_d)
            if self.model.spaced_label is None:
                self.model.spaced_label = self.model.correct_pred(self.model.pred,label)
                self.model.spaced_label = self.model.onehot(self.model.spaced_label)
            index_spaced = self.model.spaced_label.argmax(dim=2)
            if self.model.count_duplicates:
                gt_counts = torch.FloatTensor(label.size(0),batch_size,2).fill_(0)
                for b in range(batch_size):
                    c=0
                    d=0
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

            losses['countLoss'] = self.loss['count'](self.model.counts,gt_counts.to(self.model.counts.device),**self.loss_params['count'])
            assert(not torch.isnan(losses['countLoss']))

        if 'mask' in lesson  or 'mask-gen' in lesson or 'mask-disc' in lesson:
            if 'auto' not in lesson:
                #pred = self.model.hwr(image, None)
                #self.model.spaced_label = self.model.correct_pred(pred,label)
                #self.model.spaced_label = self.model.onehot(self.model.spaced_label)
                a_batch_size = instance['a_batch_size'] if 'a_batch_size' in instance else None
                style = self.model.extract_style(image,label,a_batch_size)
            if self.style_detach:
                style_d = style.detach()
            else:
                style_d=style
            if self.model.spaced_label is None:
                self.model.spaced_label = self.model.correct_pred(self.model.pred,label)
                self.model.spaced_label = self.model.onehot(self.model.spaced_label)
            self.model.top_and_bottom = self.model.create_mask(self.model.spaced_label,style_d)
            #gt_top_and_bottom = torch.FloatTensor(mask.size(3),batch_size,2)
            #center = getCenterLine(image)
            #for h in range(mask.size(3)):
            gt_top_and_bottom = instance['top_and_bottom']
            if 'mask' in get:
                self.model.mask = self.model.write_mask(self.model.top_and_bottom,image.size())
            self.model.top_and_bottom = self.model.top_and_bottom.permute(1,2,0)

            if gt_top_and_bottom.size(2)>self.model.top_and_bottom.size(2):
                diff= gt_top_and_bottom.size(2)-self.model.top_and_bottom.size(2)
                gt_top_and_bottom = gt_top_and_bottom[:,:,diff//2:-(diff//2 + diff%2)]
            elif gt_top_and_bottom.size(2)<self.model.top_and_bottom.size(2):
                diff = self.model.top_and_bottom.size(2)-gt_top_and_bottom.size(2)
                gt_top_and_bottom = F.pad(gt_top_and_bottom,(diff//2,diff//2 + diff%2),'replicate')
            if 'mask' in lesson:    
                losses['maskLoss'] = self.loss['mask'](self.model.top_and_bottom,gt_top_and_bottom.to(self.model.top_and_bottom.device),**self.loss_params['mask'])

            if 'mask-gen' in lesson:
                losses['mask_generatorLoss'] = -self.model.mask_discriminator(self.model.top_and_bottom.permute(2,0,1)).mean()

            if 'mask-disc' in lesson:
                num_fake = self.model.top_and_bottom.size(0)
                discriminator_pred = self.model.mask_discriminator(torch.cat((self.model.top_and_bottom,gt_top_and_bottom.to(self.model.top_and_bottom.device)),dim=0).permute(2,0,1))
                discriminator_pred_on_fake = discriminator_pred[:num_fake]
                discriminator_pred_on_real = discriminator_pred[num_fake:]
                disc_loss = F.relu(1.0 - discriminator_pred_on_real).mean() + F.relu(1.0 + discriminator_pred_on_fake).mean()
                losses['mask_discriminatorLoss']=disc_loss

        if 'auto' in lesson and 'feature' in self.loss:
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

        if 'auto' in lesson and 'reconRecog' in self.loss:
            if recon_pred is None:
                recon_pred = self.model.hwr(recon)
            recon_pred_size = torch.IntTensor([recon_pred.size(0)] * batch_size)
            recon_recogLoss = self.loss['reconRecog'](recon_pred,label.permute(1,0),recon_pred_size,label_lengths)
            losses['reconRecogLoss']=recon_recogLoss 

        if 'gen' in lesson and 'genRecog' in self.loss:
            gen_pred = self.model.hwr(gen_image)
            gen_pred_size = torch.IntTensor([gen_pred.size(0)] * batch_size)
            gen_recogLoss = self.loss['genRecog'](gen_pred,label.permute(1,0),gen_pred_size,label_lengths)
            if torch.isfinite(gen_recogLoss):
                losses['genRecogLoss']=gen_recogLoss 

        #if self.style_recon:
        #    #TODO don't train style extractor from this!
        #    style_recon = self.model.style_extractor(recon)
        #    style_reconLoss = self.loss['style_recon'](style_recon,style,**self.loss_params['style_recon'])

        if 'gen' in lesson or 'disc' in lesson:
            if 'auto' in lesson or 'auto-disc' in lesson:
                if recon.size(3)>gen_image.size(3):
                    diff = recon.size(3)-gen_image.size(3)
                    gen_image = F.pad(gen_image,(diff//2,diff//2 + diff%2,0,0),'replicate')
                elif recon.size(3)<gen_image.size(3):
                    diff = -(recon.size(3)-gen_image.size(3))
                    recon = F.pad(recon,(diff//2,diff//2 + diff%2,0,0),'replicate')
                fake = torch.cat((recon,gen_image),dim=0)
                image.size(0)
            else:
                fake = gen_image
        elif 'auto-gen' in lesson or 'auto-disc' in lesson:
            fake = recon
        if 'disc' in lesson or 'auto-disc' in lesson:
            if fake.size(3)>image.size(3):
                diff = fake.size(3)-image.size(3)
                image = F.pad(image,(diff//2,diff//2 + diff%2,0,0),'replicate')
            elif fake.size(3)<image.size(3):
                diff = -(fake.size(3)-image.size(3))
                #fake = F.pad(fake,(diff//2,diff//2 + diff%2,0,0),'replicate')
                image = image[:,:,:,:-diff]
            discriminator_pred = self.model.discriminator(torch.cat((image,fake),dim=0))
            #if self.style_recon: 
                #discriminator_pred, disc_style = discriminator_pred
                #disc_style_on_real = disc_style[0:image.size(0)]
                #disc_style_on_fake = disc_style[0:image.size(0)]
                #losses['style_reconLoss'] = self.loss['style_recon'](disc_style,style, **self.loss_params['style_recon'])
            discriminator_pred_on_real = discriminator_pred[:image.size(0)]
            discriminator_pred_on_fake = discriminator_pred[image.size(0):]
            #hinge loss
            disc_loss = F.relu(1.0 - discriminator_pred_on_real).mean() + F.relu(1.0 + discriminator_pred_on_fake).mean()

            losses['discriminatorLoss']=disc_loss
        if 'gen' in lesson:
            gen_loss = -self.model.discriminator(fake).mean()
            losses['generatorLoss']=gen_loss
            assert(not torch.isnan(losses['generatorLoss']))
        elif 'auto-gen' in lesson:
            gen_loss = -self.model.discriminator(recon).mean()
            losses['generatorLoss']=gen_loss

        if 'gen-style' in lesson:
            gen_loss = -self.model.style_discriminator(style_gen).mean()
            losses['style_generatorLoss']=gen_loss
        if 'disc-style' in lesson:
            if style is None:
                style = self.model.extract_style(image,label,instance['a_batch_size'])
            num_fake = style_gen.size(0)
            discriminator_pred = self.model.style_discriminator(torch.cat((style_gen,style),dim=0))
            discriminator_pred_on_fake = discriminator_pred[:num_fake]
            discriminator_pred_on_real = discriminator_pred[num_fake:]
            disc_loss = F.relu(1.0 - discriminator_pred_on_real).mean() + F.relu(1.0 + discriminator_pred_on_fake).mean()
            losses['style_discriminatorLoss']=disc_loss

        if 'recon-auto-style' in lesson: 
            style_recon = self.model.extract_style(recon,label,instance['a_batch_size'])
            losses['reconAutoStyleLoss'] = self.loss['reconAutoStyle'](style_recon,style)
        if 'gen-auto-style' in lesson:
            style_recon = self.model.extract_style(gen_image,label,1)
            losses['genAutoStyleLoss'] = self.loss['genAutoStyle'](style_recon,style_gen)

        if  ('gen' in lesson or 'auto' in lesson) and 'key' in self.loss:
            losses['keyLoss'] = self.loss['key'](self.model.style_extractor.keys1,**self.loss_params['key'])

        if 'style-super' in lesson:
            if style_gen is None:
                style_gen = self.get_style_gen(batch_size,label.device)
            if gen_image is None:
                gen_image = self.model(label,label_lengths,style_gen)
            found_style = self.model.extract_style(gen_image.detach(),label,1)
            losses['styleSuperLoss'] = self.loss['styleSuper'](found_style,style_gen)



        if get:
            got={'name': instance['name']}
            for name in get:
                if name=='recon':
                    got[name] = recon
                elif name=='recon_gt_mask':
                    got[name] = recon_gt_mask
                elif name=='gen_image':
                    got[name] = gen_image
                elif name=='gen_img':
                    got[name] = gen_image
                elif name=='gen':
                    got[name] = gen_image
                elif name=='pred':
                    if self.model.pred is None:
                        self.model.pred = self.model.hwr(image, None)   
                    got[name] = self.model.pred
                elif name=='spaced_label':
                    if self.model.spaced_label is None:
                        if self.model.pred is None:
                            self.model.pred = self.model.hwr(image, None)   
                        self.model.spaced_label = self.model.correct_pred(self.model.pred,label)
                        #self.model.spaced_label = self.model.onehot(self.model.spaced_label)
                    got[name] = self.model.spaced_label
                elif name=='mask':
                    got[name] = self.model.mask
                elif name=='gen_mask':
                    got[name] = self.model.gen_mask
                elif name=='style':
                    got[name] = style
                elif name=='style':
                    got[name] = style
                elif name=='author':
                    got[name] = instance['author']
                elif name=='gt':
                    got[name] = instance['gt']
                else:
                    raise ValueError("Unknown get [{}]".format(name))
            self.model.spaced_label=None
            self.model.mask=None
            self.model.gen_mask=None
            self.model.top_and_bottom=None
            self.model.counts=None
            self.model.pred=None
            self.model.spacing_pred=None
            self.model.mask_pred=None
            #self.
            return losses, got
        else:
            self.model.spaced_label=None
            self.model.mask=None
            self.model.gen_mask=None
            self.model.top_and_bottom=None
            self.model.counts=None
            self.model.pred=None
            self.model.spacing_pred=None
            self.model.mask_pred=None
            return losses

    def getCER(self,gt,pred,individual=False):
        cer=0
        if individual:
            all_cer=[]
        pred_strs=[]
        for i,gt_line in enumerate(gt):
            logits = pred[:,i]
            pred_str, raw_pred = string_utils.naive_decode(logits)
            pred_str = string_utils.label2str_single(pred_str, self.idx_to_char, False)
            this_cer = error_rates.cer(gt_line, pred_str)
            cer+=this_cer
            if individual:
                all_cer.append(this_cer)
            pred_strs.append(pred_str)
        cer/=len(gt)
        if individual:
            return cer, pred_strs, all_cer
        return cer, pred_strs

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
        if self.interpolate_gen_styles and len(self.prev_styles)>0:
            indexes = np.random.randint(0,len(self.prev_styles),(batch_size,2))
            mix = np.random.uniform(self.interpolate_gen_styles_low,self.interpolate_gen_styles_high,batch_size)
            new_styles=[]
            for b in range(batch_size):
                new_style = self.prev_styles[indexes[b,0]]*mix[b] + self.prev_styles[indexes[b,1]]*(1-mix[b])
                new_styles.append(new_style)
            style_gen = torch.stack(new_styles,dim=0).to(device)
        else:
            sample = torch.FloatTensor(batch_size,self.model.style_dim//2).normal_()
            style_gen = self.model.style_from_normal(sample.to(device))
        return style_gen
