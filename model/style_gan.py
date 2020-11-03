# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
import torch
import torch.nn as nn

class SGLayer(nn.Module):
    def def __init__(self,?):
        super(GBlock,self).__init__()

    def forward(self,input):
        image,style,noise = input
        #if self.upsample:
            
        x = self.conv(image) #transpose if upsample?
        if self.upsample:
            x = self.blur(x)
        x = self.apply_noise(x,noise)
        x = self.apply_bias(x)
        x = self.act(x)
        #pixel norm?
        x = self.instanceNorm(x)
        x = apply_style(x,style)
        #x = self.adaIN1(x)
