# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
import torch
from torch import nn
import torch.nn.functional as F
from .MUNIT_networks import Conv2dBlock
from .net_builder import getGroupSize
import copy, math
from .attention import MultiHeadedAttention

class NewHWStyleEncoder(nn.Module):
    def __init__(self, input_dim, dim, style_dim, norm, activ, pad_type, n_class, num_keys=16, frozen_keys=False, global_pool=False):
        super(NewHWStyleEncoder, self).__init__()
        self.global_pool=global_pool
        self.down = []
        self.down += [Conv2dBlock(input_dim, dim, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)]
        #self.down += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.down += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)] #32, 16
            dim *= 2
            self.down += [Conv2dBlock(dim, dim, 3, 1, (1,1,0,0), norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.down += [Conv2dBlock(dim, dim, 4, (2,1), (1,1,0,0), norm=norm, activation=activ, pad_type=pad_type)] #6, 1
        self.down = nn.Sequential(*self.down)
        prepped_size = style_dim
        self.prep = nn.Sequential(
                        nn.Conv1d(dim+n_class, prepped_size, 5, 1, 2),
                        nn.ReLU(True),
                        nn.MaxPool1d(2,2),
                        nn.Conv1d(prepped_size, prepped_size, 3, 1, 1),
                        nn.ReLU(True),
                        nn.Conv1d(prepped_size, prepped_size, 3, 1, 1)
                        )

        if global_pool:
            prepped_size//=2 # as we split half of the features off to do global pooling on
            style_dim//=2
        if prepped_size>64:
            heads=prepped_size//64
        else:
            heads=2
        self.mhAtt1 = MultiHeadedAttention(heads,prepped_size)
        #self.mhAtt2 = MultiHeadedAttention(heads,prepped_size)
        #self.mhAtt3 = MultiHeadedAttention(heads,prepped_size)
        if frozen_keys:
            self.put_on_gpu=True#False
            self.keys1 = nn.Parameter(torch.FloatTensor(1,num_keys,prepped_size).normal_(), requires_grad=False)
        else:
            self.put_on_gpu=True
            self.keys1 = nn.Parameter(torch.FloatTensor(1,num_keys,prepped_size).normal_())
        #self.keys2 = nn.Parameter(nn.FloatTensor(1,num_keys,prepped_size).normal_())
        #self.keys3 = nn.Parameter(nn.FloatTensor(1,num_keys,prepped_size).normal_())
        #self.comb1 = nn.Sequential(
        #                nn.LeakyReLU(0.2,True),
        #                nn.Linear(2*prepped_size*num_keys,style_dim)
        #                )
        #self.comb2 = nn.Sequential(
        #                nn.LeakyReLU(0.1,True),
        #                nn.Linear(prepped_size*num_keys+style_dim,style_dim)
        #                )
        self.reduceQuarters=nn.ModuleList()
        for i in range(4):
            self.reduceQuarters.append( nn.Sequential(
                            nn.LeakyReLU(0.2,True),
                            nn.Linear(prepped_size*num_keys//4,style_dim),
                            nn.Dropout(0.05,True),
                            nn.LeakyReLU(0.01,True),
                            ) )
        self.reduceWhole = nn.Sequential(
                        nn.Linear(4*style_dim,2*style_dim),
                        nn.Dropout(0.01,True),
                        nn.LeakyReLU(0.01,True),
                        nn.Linear(2*style_dim,style_dim)
                        )

    def forward(self, x,recog):
        if not self.put_on_gpu:
            self.put_on_gpu=True
            self.keys1=self.keys1.to(x.device)
        batch_size=x.size(0)
        x = self.down(x)
        #x must have height of 1!
        x = x.view(batch_size,x.size(1),x.size(3))
        diff = x.size(2)-recog.size(2)
        if diff>0:
            recog = F.pad(recog,(diff//2,(diff//2)+diff%2),mode='replicate')
        elif diff<0:
            x = F.pad(x,(-diff//2,(-diff//2)+(-diff)%2),mode='replicate')

        x = torch.cat((x,recog),dim=1)
        x = self.prep(x)

        if self.global_pool:
            x,g_x = torch.chunk(x,2,dim=1)
            g_style = F.adaptive_avg_pool1d(g_x,1).view(batch_size,-1)
        #attOut = torch.zero(x.size(1))
        #for att in self.mhAtts:
        x = x.permute(0,2,1)
        att = self.mhAtt1(self.keys1.expand(batch_size,-1,-1),x,x)
        #import pdb;pdb.set_trace() #batch,head,query,width
        #att2 = self.mhAtt2(self.keys2,x,x))
        #att3 = self.mhAtt3(self.keys3,x,x))
        vector = att.view(batch_size,att.size(1)*att.size(2))
        #vector2 = att2.view(batch_size,att2.size(1)*att2.size(2))
        #vector3 = att3.view(batch_size,att3.size(1)*att3.size(2))
        #vector = self.comb1(torch.cat((vector,vector1,dim=1))
        #vector = self.comb2(torch.cat((vector,vector2,dim=1)

        #return vector
        quarterDim = vector.size(1)//4
        reducedQ=[]
        vq = torch.chunk(vector,4,dim=1)
        for i in range(4):
            reducedQ.append( self.reduceQuarters[i](vq[i]) )
            #reducedQ.append( self.reduceQuarters[i](vector[:,i*quarterDim:(i+1)*quarterDim].copy()) )
        a_style = self.reduceWhole(torch.cat(reducedQ,dim=1))
        assert(torch.isfinite(a_style).all())
        assert(torch.isfinite(g_style).all())
        if self.global_pool:
            return torch.cat((g_style,a_style),dim=1)
        else:
            return a_style
            

