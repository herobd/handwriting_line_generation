import torch
import torch.nn as nn
from model.net_builder import getGroupSize
from model.pure_gen import PixelNorm

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

class CountCNN(nn.Module):
    def __init__(self,class_size,style_size, hidden_size=128,n_out=1, emb_style=0):
        super(CountCNN, self).__init__()

        self.cnn = nn.Sequential(
                nn.Conv1d(class_size+style_size,hidden_size,kernel_size=3,stride=1,padding=1),
                nn.GroupNorm(getGroupSize(hidden_size),hidden_size),
                nn.Dropout2d(0.1),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size,hidden_size//2,kernel_size=3,stride=1,padding=1),
                nn.GroupNorm(getGroupSize(hidden_size//2),hidden_size//2),
                nn.Dropout2d(0.1),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size//2,hidden_size//4,kernel_size=3,stride=1,padding=1),
                nn.GroupNorm(getGroupSize(hidden_size//4),hidden_size//4),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size//4,n_out,kernel_size=1,stride=1,padding=0),
                )

        if n_out==1 or n_out>2:
            self.mean = nn.Parameter(torch.FloatTensor(1,n_out).fill_(2))
            self.std = nn.Parameter(torch.FloatTensor(1,n_out).fill_(1))
        else:
            self.mean = nn.Parameter(torch.FloatTensor([2.0,0.0])) #These are educated guesses to give the net a good place to start
            self.std = nn.Parameter(torch.FloatTensor([1.5,0.5]))
        
        if emb_style>0:
            if type(emb_style) is float:
                drop=0.125
            else:
                drop=0.5
            layers = [PixelNorm()]
            for i in range(int(emb_style)):
                layers.append(nn.Linear(style_size, style_size))
                layers.append(nn.Dropout(drop,True))
                layers.append(nn.LeakyReLU(0.2,True))
            self.emb_style = nn.Sequential(*layers)
        else:
            self.emb_style = None

    def forward(self, input, style):
        if self.emb_style is not None:
            style = self.emb_style(style)
        input = input.permute(1,2,0)
        style = style[...,None].expand(-1,-1,input.size(2))
        output = self.cnn(torch.cat((input,style),dim=1))

        output = output.permute(2,0,1) #Back to Temporal,batch,channel
        
        assert(not torch.isnan(output).any() and not torch.isinf(output).any())
        assert(not torch.isnan(self.std).any())
        assert(not torch.isnan(self.mean).any())
        return output*self.std + self.mean

class CreateMaskRNN(nn.Module):
    def __init__(self,class_size,style_size, hidden_size=128,shallow=False):
        super(CreateMaskRNN, self).__init__()

        if shallow:
            self.rnn1 = nn.GRU(class_size+style_size, hidden_size, bidirectional=True, dropout=0.25, num_layers=1)
            self.upsample = nn.Sequential(  
                                            nn.ConvTranspose1d(2*hidden_size,hidden_size,kernel_size=4,stride=2,padding=0),
                                            nn.GroupNorm(getGroupSize(hidden_size),hidden_size),
                                            nn.ReLU(inplace=True),
                                            nn.ConvTranspose1d(hidden_size,hidden_size//2,kernel_size=4,stride=2,padding=0),
                                            nn.GroupNorm(getGroupSize(hidden_size//2),hidden_size//2),
                                            nn.ReLU(inplace=True)
                                            )
            self.rnn2 = nn.GRU(hidden_size//2, hidden_size//2, bidirectional=True, dropout=0.25, num_layers=1)
        else:
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

        assert(not torch.isnan(output).any())
        assert(not torch.isnan(self.std).any())
        assert(not torch.isnan(self.mean).any())

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
