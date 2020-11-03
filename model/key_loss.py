# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
import torch
import torch.nn.functional as F

def pushMinDist(values,dist='l2',thresh=0.5):

    a = values.view(values.size(0),values.size(1),1,values.size(2)).expand(-1,-1,values.size(1),-1)
    b = values.view(values.size(0),1,values.size(1),values.size(2)).expand(-1,values.size(1),-1,-1)
    if dist=='l1':
        d = (a-b).abs().mean(dim=3)
    elif dist=='l2':
        d = ((a-b)**2).mean(dim=3)
    else:
        raise NotImplemented("Unknown distance: {}".format(dist))
    d[:,range(values.size(1)), range(values.size(1))]=float('inf')
    minD,minL = d.min(dim=1)
    minD = torch.where(minD>thresh,torch.zeros_like(minD).to(minD.device),minD)
    return -1*minD.sum()


