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
from data_loader import getDataLoader

class HWRWithSynthTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(HWRWithSynthTrainer, self).__init__(model, loss, metrics, resume, config, train_logger)
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
        
        if 'synth_data' in config:
            config_synth= {'data_loader':config['synth_data'],
                           'validation':{}}
            self.synth_data_loader, _ = getDataLoader(config_synth,'train')
            self.authors_of_interest = self.synth_data_loader.dataset.authors_of_interest
        else:
            self.authors_of_interest = None
        self.synth_data_loader_iter = None


        #self.gen = config['gen_model$']

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
        if self.curriculum and 'synth' in lesson:
            if self.synth_data_loader_iter is None:
                self.synth_data_loader_iter = self.refresh_synth_data()
            try:
                instance = self.synth_data_loader_iter.next()
            except StopIteration:
                #self.synth_data_loader.dataset.refresh()
                self.synth_data_loader_iter = self.refresh_synth_data()
                instance = self.synth_data_loader_iter.next()
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

        pred, losses = self.run(instance)
    
        loss=0
        recogLoss=0
        for name in losses.keys():
            losses[name] *= self.lossWeights[name[:-4]]
            loss += losses[name]
            losses[name] = losses[name].item()
        #assert(not torch.isnan(loss) and not torch.isinf(loss))
        if pred is not None:
            pred = pred.detach().cpu().numpy()
        if not torch.isnan(loss) and not torch.isinf(loss):
            loss_item = loss.item()
           
            loss.backward()

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

            self.optimizer.step()
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
        else:
            gt = instance['gt']
            if pred is not None:
                cer, pred_str = self.getCER(gt,pred)
            else:
                cer=0
            log = {
                    'CER': cer
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
        #self.to_display={}

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
        author_cer=0
        author_count=0
        print('validate')
        with torch.no_grad():
            losses = defaultdict(lambda: 0)
            for batch_idx, instance in enumerate(self.valid_data_loader):
                if not self.logged:
                    print('validate: {}/{}'.format(batch_idx,len(self.valid_data_loader)), end='\r')
                pred, losses = self.run(instance)
            
                for name in losses.keys():
                    losses[name] *= self.lossWeights[name[:-4]]
                    total_loss += losses[name].item()
                    total_losses['val_'+name] += losses[name].item()

                if pred is not None:
                    pred = pred.detach().cpu().numpy()
                    gt = instance['gt']
                    cer, pred_str, cer_ind = self.getCER(gt,pred,True)
                    total_cer += cer
                    if self.authors_of_interest is not None:
                        for b,author in enumerate(instance['author']):
                            if  author in self.authors_of_interest:
                                author_cer += cer_ind[b]
                                author_count += 1
        
        for name in total_losses.keys():
            total_losses[name]/=len(self.valid_data_loader)
        toRet={
                'val_loss': total_loss/len(self.valid_data_loader),
                'val_CER': total_cer/len(self.valid_data_loader),
                **total_losses
                }
        if self.authors_of_interest is not None:
            toRet['val_author_CER']= author_cer/author_count
        return toRet

    def onehot(self,label):
        label_onehot = torch.zeros(label.size(0),label.size(1),self.num_class)
        #label_onehot[label]=1
        #TODO tensorize
        for i in range(label.size(0)):
            for j in range(label.size(1)):
                label_onehot[i,j,label[i,j]]=1
        return label_onehot.to(label.device)

    def run(self,instance):
        image, label = self._to_tensor(instance)
        label_lengths = instance['label_lengths']

        losses = {}

        pred = self.model(image)
        batch_size = pred.size(1)
        pred_size = torch.IntTensor([pred.size(0)] * batch_size)
        recogLoss = self.loss['recog'](pred,label.permute(1,0),pred_size,label_lengths)
        losses['recogLoss']=recogLoss

        return pred, losses

    def refresh_synth_data(self):
        """
        Recreate the synthetic dataset
        """
        self.model = self.model.cpu()
        self.gen = self.gen.to(self.gpu)
        self.gen.eval()

        self.synth_data_loader.dataset.refresh_data(self.gen,self.gpu,self.logged)
        #num_synth = self.synth_data_loader.dataset.gen_batches

        #print('refreshing sythetic')
        #with torch.no_grad():
        #    for i in range(num_synth):
        #        style,label,label_lengths,gt = self.synth_data_loader.dataset.sample()
        #        if self.with_cuda:
        #            style = style.to(self.gpu)
        #            label = label.to(self.gpu)
        #        if not self.logged:
        #            print('generate: {}/{}'.format(i,num_synth), end='\r')
        #        generated = self.gen(label,label_lengths,style)
        #        self.synth_data_loader.dataset.save(i,generated,gt)
        self.model = self.model.to(self.gpu)
        self.gen = self.gen.cpu()
        return iter(self.synth_data_loader)


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
