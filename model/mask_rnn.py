# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
import torch
import torch.nn as nn
from model.net_builder import getGroupSize

class CountRNN(nn.Module):
    def __init__(self,class_size,style_size, hidden_size=128,n_out=1):
        super(CountRNN, self).__init__()

        self.rnn = nn.GRU(class_size+style_size, hidden_size, bidirectional=True, dropout=0.25, num_layers=2)
        self.out = nn.Linear(hidden_size * 2, n_out)

        self.mean = nn.Parameter(torch.FloatTensor(1,n_out).fill_(2))
        self.std = nn.Parameter(torch.FloatTensor(1,n_out).fill_(1))

    def forward(self, input, style):
        ##pad input by one so we predict before and after the string  #Actually, we won't predict after, since that gets padded out anyways
        #zero = torch.zeros(1,input.size(1),input.size(2)).to(input.device)
        #input = torch.cat((input,zero),dim=0)
        style = style[None,...].expand(input.size(0),-1,-1)
        recurrent, _ = self.rnn(torch.cat((input,style),dim=2))
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.out(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output*self.std + self.mean

class CreateMaskRNN(nn.Module):
    def __init__(self,class_size,style_size, hidden_size=128):
        super(CreateMaskRNN, self).__init__()

        self.rnn1 = nn.GRU(class_size+style_size, hidden_size, bidirectional=True, dropout=0.25, num_layers=2)
        self.upsample = nn.Sequential(  
                                        nn.ConvTranspose1d(2*hidden_size,hidden_size,kernel_size=4,stride=2,padding=0),
                                        nn.GroupNorm(getGroupSize(hidden_size),hidden_size),
                                        nn.ReLU(inplace=True),
                                        nn.ConvTranspose1d(hidden_size,hidden_size//2,kernel_size=4,stride=2,padding=0),
                                        nn.GroupNorm(getGroupSize(hidden_size//2),hidden_size//2),
                                        nn.ReLU(inplace=True)
                                        )
        self.rnn2 = nn.GRU(hidden_size//2, hidden_size//2, bidirectional=True, dropout=0.25, num_layers=2)
        self.out = nn.Linear(hidden_size, 2)

        self.mean = nn.Parameter(torch.FloatTensor(1).fill_(10))
        self.std = nn.Parameter(torch.FloatTensor(1).fill_(5))

    def forward(self, input, style):
        style = style[None,...].expand(input.size(0),-1,-1)
        recurrent, _ = self.rnn1(torch.cat((input,style),dim=2))
        spatial = recurrent.permute(1,2,0)# from len,batch,channel to batch,channel,len
        recurrent=None
        spatial = self.upsample(spatial)
        recurrent = spatial.permute(2,0,1)# from batch,channel,len to len,batch,channel
        spatial=None
        recurrent, _ = self.rnn2(recurrent)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h) # flatten len and batch together
        recurrent=None

        output = self.out(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1) # reshape to get len,batch,channel

        return output*self.std + self.mean


class TopAndBottomDiscriminator(nn.Module):
    def __init__(self,use_derivitive=False):
        super(TopAndBottomDiscriminator, self).__init__()
        n_in = 4 if use_derivitive else 2
        self.use_derivitive=use_derivitive
        self.rnn = nn.GRU(n_in,128,bidirectional=True, dropout=0.25, num_layers=2)
        self.out = nn.Linear(256, 1)
    def forward(self,input):
        if self.use_derivitive:
            diff =  input[1:] - input[:-1]
            input = torch.cat((input[:-1],diff),dim=2)
        r,_ = self.rnn(input)
        T, b, h = r.size()
        t_r = r.view(T*b,h)
        output = self.out(t_r)
        return output.view(T,b).mean(dim=0)
