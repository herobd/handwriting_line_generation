import timeit
import torch
import torch.nn.functional as F

def DTWLoss(pred,gt, downsize=0,lossType='l1', window=70):
    for i in range(downsize):
        pred=F.avg_pool2d(pred,2)
        gt=F.avg_pool2d(gt,2)
    pred_len = pred.size(3)
    gt_len = gt.size(3)
    batch_size = pred.size(0)

    #total_lossType=0
    #for b in range(batch_size):
    #    dtw = torch.FloatTensor(pred_len+1,gt_len+1).fill_(float('inf'))

    #    dtw[:,0]=0
    #    dtw[0,:]=0

    #    for i in range(1,pred_len+1):
    #        for j in range(1,gt_len+1):
    #            cost = F.l1_lossType(pred[b,:,:,i-1],gt[b,:,:,j-1])
    #            dtw[i,j] = cost + min(  dtw[i-1,j],     #insertion
    #                                    dtw[i-1,j-1],   #match
    #                                    dtw[i,j-1],     #deletion
    #                                    )
    #total_lossType += dtw[-1,-1]   

    #tic=timeit.default_timer()
    pred_ex = pred.view(batch_size,-1,pred_len,1).expand(-1,-1,-1,gt_len)
    gt_ex = gt.view(batch_size,-1,1,gt_len).expand(-1,-1,pred_len,-1)
    #print('setup dtw {}'.format(timeit.default_timer()-tic))
    #tic=timeit.default_timer()
    if lossType=='l1':
        allCosts = F.l1_loss(pred_ex,gt_ex,reduction='none').mean(dim=1)
    elif lossType=='hinge':
        diff = torch.abs(pred_ex-gt_ex)
        diff *= (diff>0.05).float()
        allCosts = diff.mean(dim=1)
    else:
        raise NotImplmentedError('unknown loss for DTW: '+lossTyle)
    #print('allCosts {}'.format(timeit.default_timer()-tic))

    #tic=timeit.default_timer()
    #allCosts = allCosts.cpu()
    #print('to gpu {}'.format(timeit.default_timer()-tic))

    ##DTW code based on https://en.wikipedia.org/wiki/Dynamic_time_warping
    #w = max(70, abs(pred_len-gt_len)) #window size
    #dtw = torch.FloatTensor(pred_len+1,gt_len+1,batch_size).fill_(float('inf')).to(pred.device)
    #dtw[0,0]=0
    #for i in range(1,pred_len+1):
    #    #for j in range(max(1, i-w), min(gt_len, i+w)+1):
    #    #    dtw[i,j]=0
    #    dtw[i,max(1, i-w):min(gt_len, i+w)+1]=0
    #tic=timeit.default_timer()
    #for i in range(1,pred_len+1):
    #    for j in range(max(1, i-w), min(gt_len, i+w)+1):
    #        cost = allCosts[:,i-1,j-1] #F.l1_loss(pred[:,:,:,i],gt[:,:,:,j],reduction='none').mean(dim=1).mean(dim=1)
    #        per_batch_min,_ = torch.min( torch.stack( (dtw[i-1,j],dtw[i-1,j-1],dtw[i,j-1]) ), dim=0)
    #        dtw[i,j] = cost + per_batch_min
    #print('dtw {}'.format(timeit.default_timer()-tic))
    #total_loss = dtw[-1,-1].mean()#.to(pred.device)

    ##NEW
    #DTW code based on https://en.wikipedia.org/wiki/Dynamic_time_warping
    #we compute the DTW off of the GPU and GRADIENT FREE (otherwise the computation graph is massive)
    #do do tract the alignemnt
    if type(window) is int:
        w=window
    elif type(window) is float:
        w=int(window*max(pred_len,gt_len))
    else:
        w=int(0.2*max(pred_len,gt_len))
    w = max(w, abs(pred_len-gt_len)) #window size
    dtw = torch.FloatTensor(pred_len+1,gt_len+1,batch_size).fill_(float('inf'))#.to(pred.device)
    allCosts_d = allCosts.detach().cpu()
    dtw[0,0]=0
    for i in range(1,pred_len+1):
        #for j in range(max(1, i-w), min(gt_len, i+w)+1):
        #    dtw[i,j]=0
        dtw[i,max(1, i-w):min(gt_len, i+w)+1]=0
    #tic=timeit.default_timer()
    history = torch.IntTensor(pred_len,gt_len,batch_size)
    for i in range(1,pred_len+1):
        for j in range(max(1, i-w), min(gt_len, i+w)+1):
            cost = allCosts_d[:,i-1,j-1] #F.l1_loss(pred[:,:,:,i],gt[:,:,:,j],reduction='none').mean(dim=1).mean(dim=1)
            per_batch_min, history[i-1,j-1] = torch.min( torch.stack( (dtw[i-1,j],dtw[i-1,j-1],dtw[i,j-1]) ), dim=0)
            dtw[i,j] = cost + per_batch_min
    #print('dtw {}'.format(timeit.default_timer()-tic))
    #tic=timeit.default_timer()
    #Now we compute an loss using the alignment
    accum = 0
    for b in range(batch_size):
        i=pred_len-1
        j=gt_len-1
        accum += allCosts[b,i,j]
        while(i>0 or j>0):
            if history[i,j,b]==0:
                i-=1
            elif history[i,j,b]==1:
                i-=1
                j-=1
            elif history[i,j,b]==2:
                j-=1
            accum+=allCosts[b,i,j]


    #print('re-run {}'.format(timeit.default_timer()-tic))
    total_loss = accum

    return total_loss
