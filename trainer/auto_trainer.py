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
from datasets.text_data import TextData

class AutoTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(AutoTrainer, self).__init__(model, loss, metrics, resume, config, train_logger)
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
            self.data_loader_iter = iter(data_loader)
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False

        self.center_pad=False
        self.no_bg_loss=False

        char_set_path = config['data_loader']['char_file']
        with open(char_set_path) as f:
            char_set = json.load(f)
        self.idx_to_char = {}
        self.num_class = len(char_set['idx_to_char'])+1
        for k,v in char_set['idx_to_char'].items():
            self.idx_to_char[int(k)] = v


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

        losses = self.run_gen(instance)
    
        loss=0
        recogLoss=0
        for name in losses.keys():
            losses[name] *= self.lossWeights[name[:-4]]
            loss += losses[name]
            losses[name] = losses[name].item()
        loss_item = loss.item()
        loss.backward()

        #for p in self.model.parameters():
        #    if p.grad is not None:
        #        p.grad[torch.isnan(p.grad)]=0


        torch.nn.utils.clip_grad_value_(self.model.parameters(),2)
        #meangrad=0
        #count=0
        #for m in self.model.parameters():
        #    if m.grad is None:
        #        continue
        #    count+=1
        #    meangrad+=m.grad.data.mean()
        #meangrad/=count

        self.optimizer.step()
        loss = loss_item


        ##toc=timeit.default_timer()
        ##print('bac: '+str(toc-tic))

        #tic=timeit.default_timer()
        metrics={}


        log = {
            'loss': loss,
            **losses,
            #'pred_str': pred_str

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
        #for  key,value in self.to_display.items():
        #    self.logger.info('{} : {}'.format(key,value))
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
        total_losses=defaultdict(lambda: 0)
        total_cer=0
        total_wer=0
        print('validate')
        with torch.no_grad():
            losses = defaultdict(lambda: 0)
            for batch_idx, instance in enumerate(self.valid_data_loader):
                if not self.logged:
                    print('validate: {}/{}'.format(batch_idx,len(self.valid_data_loader)), end='\r')
                if 'recog' in self.loss:
                    get=['pred']
                else:
                    get=['none']
                losses,got = self.run_gen(instance,get)
            
                for name in losses.keys():
                    losses[name] *= self.lossWeights[name[:-4]]
                    total_loss += losses[name].item()
                    total_losses['val_'+name] += losses[name].item()
                if 'recog' in self.loss:
                    pred = got['pred']
                    pred = pred.detach().cpu().numpy()
                    gt = instance['gt']
                    cer,wer,_ = self.getCER(gt,pred)
                    total_cer += cer
                    total_wer += wer

        
        for name in total_losses.keys():
            total_losses[name]/=len(self.valid_data_loader)
        toRet={
                'val_loss': total_loss/len(self.valid_data_loader),
                **total_losses
                }
        if 'recog' in self.loss:
            toRet['val_CER']= total_cer/len(self.valid_data_loader)
            toRet['val_WER']= total_wer/len(self.valid_data_loader)
        return toRet

    def onehot(self,label):
        label_onehot = torch.zeros(label.size(0),label.size(1),self.num_class)
        #label_onehot[label]=1
        #TODO tensorize
        for i in range(label.size(0)):
            for j in range(label.size(1)):
                label_onehot[i,j,label[i,j]]=1
        return label_onehot.to(label.device)


    def run_gen(self,instance,get=[]):
        image, label = self._to_tensor(instance)
        if image.size(3)%8>0:
            toPad = 8 - image.size(3)%8
            image = F.pad(image,(toPad//2,toPad//2 + toPad%2),value=PADDING_CONSTANT)
        if 'recog' in self.loss:
            recon,pred = self.model(image)
        else:
            recon = self.model(image)
        losses={}
        if 'auto' in self.loss:
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
                if toPad>5:
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

        if 'recog' in self.loss:
            batch_size = pred.size(1)
            pred_size = torch.IntTensor([pred.size(0)] * batch_size)
            label_lengths = instance['label_lengths']
            recogLoss=self.loss['recog'](pred,label.permute(1,0),pred_size,label_lengths)
            #assert(not torch.isinf(losses['recogLoss']))
            if not torch.isinf(recogLoss):
                losses['recogLoss'] = recogLoss
        if len(get)>0:
            got = {}
            for name in get:
                if name=='recon':
                    got['recon']=recon
                elif name=='pred':
                    got['pred']=pred
                elif name=='none':
                    pass
                else:
                    print('Error, unknown get: {}'.format(name))
            return losses, got
        return losses

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
            this_cer = error_rates.cer(gt_line, pred_str)
            cer+=this_cer
            if individual:
                all_cer.append(this_cer)
            pred_strs.append(pred_str)
            wer += error_rates.wer(gt_line, pred_str)
        cer/=len(gt)
        wer/=len(gt)
        if individual:
            return cer,wer, pred_strs, all_cer
        return cer,wer, pred_strs
