import torch.nn.functional as F
import torch
#from robust_loss_pytorch import AdaptiveLossFunction

def pyramidL1Loss(gen,gt,weights=[1],pool='max', lossType='l1'):
    assert(gt.size(1)==1 and gen.size(1)==1)
    batch_size=gen.size(0)
    gen = (gen+1)/2 #(center so background is zero, ink is 1)
    gt = (gt+1)/2

    losses=[]
    loss=0
    #for s,weight in reversed(list(enumerate(weights))):
    for s,weight in enumerate(weights):
        #scale = 1/(2**s)
        #if scale != 1:
        if s>0:
            #s_gen = F.interpolate(gen,scale_factor=scale,mode='bilinear')
            #s_gt = F.interpolate(gt,scale_factor=scale,mode='bilinear')
            if pool=='max':
                s_gen = F.max_pool2d(s_gen,2)
                s_gt = F.max_pool2d(s_gt,2)
            elif pool=='avg' or pool=='average':
                s_gen = F.avg_pool2d(s_gen,2)
                s_gt = F.avg_pool2d(s_gt,2)
            else:
                raise NotImplemented('unknown pool: '+pool)
        else:
            s_gen = gen
            s_gt = gt

        if lossType=='l1':
            l=F.l1_loss(s_gen,s_gt)
        elif lossType=='hinge':
            diff = (s_gen-s_gt).abs()
            diff *= (diff>0.02).float()
            l=diff.mean()
        elif lossType=='robust':
            l=AdaptiveLossFunction(s_gen,s_gt)
            
        losses.append(l.item())
        loss += weight*l
    return loss,losses
