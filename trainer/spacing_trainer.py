import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from base import BaseTrainer
import timeit
from collections import defaultdict


class SpacingTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(SpacingTrainer, self).__init__(model, loss, metrics, resume, config, train_logger)
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
        self.to_display={}


    def _to_tensor(self, instance):
        input = instance['input']
        label = instance['label']
        style = instance['style']

        if self.with_cuda:
            input = input.to(self.gpu)
            style = style.to(self.gpu)
            if label is not None:
                label = label.to(self.gpu)
        return input,style, label

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
        batch_idx = (iteration-1) % len(self.data_loader)
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
        for name in losses.keys():
            losses[name] *= self.lossWeights[name[:-4]]
            loss += losses[name]
            losses[name] = losses[name].item()
        assert(not torch.isnan(loss))
        if pred is not None:
            pred = pred.detach().cpu().numpy()
        loss_item = loss.item()
        loss.backward()


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


        ##toc=timeit.default_timer()
        ##print('bac: '+str(toc-tic))

        #tic=timeit.default_timer()
        metrics={}


        log = {
            'loss': loss,
            **losses,
            #'pred_str': pred_str

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
                ls +=': {:.6f},\t'.format(val)
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
        print('validate')
        with torch.no_grad():
            losses = defaultdict(lambda: 0)
            for batch_idx, instance in enumerate(self.valid_data_loader):
                pred, losses = self.run(instance)
            
                for name in losses.keys():
                    losses[name] *= self.lossWeights[name[:-4]]
                    total_loss += losses[name].item()
                    total_losses['val_'+name] += losses[name].item()


        
        for name in total_losses.keys():
            total_losses[name]/=len(self.valid_data_loader)
        toRet={
                'val_loss': total_loss/len(self.valid_data_loader),
                **total_losses
                }
        return toRet


    def run(self,instance):
        input, style, label = self._to_tensor(instance)
        empty_mask = torch.FloatTensor(batch)
        _,pred,_ = self.model(input,style,empty_mask)#,label.size(0))
        #self.to_display['pred']= pred[0].argmax(dim=0)[:20].cpu().detach().numpy()
        #self.to_display['label']=label[0,:20].cpu().numpy()

        if 'count' in self.loss:
            losses['countLoss'] = self.loss['count'](pred,label,**self.loss_params['count'])
        if 'spacing' in self.loss:
            spacingLoss = self.loss['spacing'](pred.permute(1,2,0).contiguous(),label.permute(1,0).contiguous(),**self.loss_params['spacing'])
            losses['spacingLoss'] = spacingLoss
        if 'CTC' in self.loss:
            #pred = pred.permute(2,0,1)
            batch_size = pred.size(1)
            pred_size = torch.IntTensor([pred.size(0)] * batch_size)
            CTCLoss = self.loss['CTC'](pred,label,pred_size,label_lengths)
            losses['CTCLoss'] = CTCLoss
        return pred, losses

