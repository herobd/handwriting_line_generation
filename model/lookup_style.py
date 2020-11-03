# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
import torch
from torch import nn
import torch.nn.functional as F

class LookupStyle(nn.Module):
    def __init__(self, style_dim):
        super(LookupStyle, self).__init__()

        #self.styles = nn.Parameter(torch.FloatTensor(numAuthors,style_dim).normal_(0,0.01))
        #self.idMap = {}
        #self.idIndex = 0
        self.styles = nn.ParameterDict()
        self.style_dim = style_dim

    def forward(self, authorIds,device=None):
        #if authorId in self.idMap:
        #    return self.styles[self.idMap[authorId]]
        #else:
        #    self.idMap[authorId] = self.idIndex
        #    self.idIndex+=1
        ret=[]
        for authorId in authorIds:
            if authorId not in self.styles:
            #    if device is not None:
            #        self.styles[authorId] = nn.Parameter(torch.FloatTensor(self.style_dim).normal_(0,0.01).to(device))
            #    else:
            #        self.styles[authorId] = nn.Parameter(torch.FloatTensor(self.style_dim).normal_(0,0.01))
                ret.append(torch.FloatTensor(self.style_dim).fill_(0).to(device))
            else:
                ret.append(self.styles[authorId])
        return torch.stack(ret,dim=0)

    def add_authors(self,author_iter):
        for authorId in author_iter:
            self.styles[authorId] = nn.Parameter(torch.FloatTensor(self.style_dim).normal_(0,0.01))

