import torch
import torch.nn as nn
from .MUNIT_networks import SpacingRNN
from base import BaseModel
class Spacer(BaseModel):
    def __init__(self,config):
        super(Spacer,self).__init__(config)
        self.rnn = SpacingRNN(5+5**2,6)

    def forward(self,input,style,length):
        batch_size=style.size(0)
        style_t = style.view(1,batch_size,-1).expand(input.size(0),-1,-1)

        out=self.rnn(torch.cat([input,style_t],dim=2),length)
        return out
