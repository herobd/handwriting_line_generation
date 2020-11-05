import torch.nn.functional as F
import torch


def alignedL1Loss(gen,gt,weights=[1],center_bias=0.1):
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
            s_gen = F.max_pool2d(s_gen,2)
            s_gt = F.max_pool2d(s_gt,2)
        else:
            s_gen = gen
            s_gt = gt

        pad = min(s_gen.size(3),s_gt.size(3))//4
        padded_gen = F.pad(s_gen,(pad,pad),value=0)
        padded_gen = padded_gen.permute(1,0,2,3) #swap channel and batch
        #s_gt = s_gt = s_gt.permute(1,0,2,3) #swap channel and batch
        corr = F.conv2d(padded_gen,s_gt,groups=batch_size)
        corr = corr[0,:,0,:]
        corr += ( torch.cat( (torch.arange(0,corr.size(1)//2 + corr.size(1)%2),torch.arange(corr.size(1)//2-1,-1,-1)) )/(corr.size(1)*s_gen.size(2)*center_bias) ).float().to(corr.device)
        loc = corr.argmax(dim=1)
        loc += gt.size(3)-pad

        padded_gen = padded_gen.permute(1,0,2,3) #swap back

        minLoc = loc.min()

        padded_gen = padded_gen[:,:,:,minLoc:]
        #print(loc-pad)
        loc-=minLoc
        maxLoc = max(padded_gen.size(3)-pad, s_gt.size(3)+loc.max())

        padded_gen = padded_gen[:,:,:,:maxLoc]
        if padded_gen.size(3)<maxLoc:
            padded_gen = F.pad(padded_gen,(0,maxLoc-padded_gen.size(3)),value=0)
        stacked_gt = torch.zeros(batch_size,1,s_gt.size(2),maxLoc).to(s_gt.device)
        for b in range(batch_size):
            stacked_gt[b,:,:,loc[b]:loc[b]+s_gt.size(3)] = s_gt[b]

        l=F.l1_loss(padded_gen,stacked_gt)
        losses.append(l.item())
        loss += weight*l
    return loss,losses
