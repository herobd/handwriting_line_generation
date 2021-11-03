import torch
import torch.nn as nn
from utils.util import getGroupSize
from model.pure_gen import PixelNorm


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
        

    def forward(self, input, style):
        input = input.permute(1,2,0)
        style = style[...,None].expand(-1,-1,input.size(2))
        output = self.cnn(torch.cat((input,style),dim=1))

        output = output.permute(2,0,1) #Back to Temporal,batch,channel
        
        assert(not torch.isnan(output).any() and not torch.isinf(output).any())
        assert(not torch.isnan(self.std).any())
        assert(not torch.isnan(self.mean).any())
        return output*self.std + self.mean

