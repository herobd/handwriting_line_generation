import torch.nn as nn
import torch
import torch.nn.functional as F
from .discriminator import SpectralNorm
from .net_builder import getGroupSize
from .attention import MultiHeadedAttention, PositionalEncoding

class AuthorClassifier(nn.Module):

    def __init__(self,class_size,dim):
        super(AuthorClassifier, self).__init__()
        leak=0.1
        keepWide=True
        self.class_size=class_size

        self.conv1 = nn.Sequential(
                nn.Conv2d(1, dim, 7, stride=1, padding=(0,3 if keepWide else 0)),
                nn.GroupNorm(getGroupSize(dim),dim), # Experiments with other GAN showed better results not using spectral on first layer
                nn.LeakyReLU(leak,True),
                nn.Conv2d(dim, dim, 3, stride=1, padding=(0,1 if keepWide else 0)),
                nn.GroupNorm(getGroupSize(dim),dim),
                nn.LeakyReLU(leak,True),
                nn.AvgPool2d(2),
                nn.Conv2d(dim, 2*dim, 3, stride=1, padding=(0,1 if keepWide else 0)),
                nn.GroupNorm(getGroupSize(2*dim),2*dim),
                nn.Dropout2d(0.05,True),
                nn.LeakyReLU(leak,True),
                nn.Conv2d(2*dim, 2*dim, 3, stride=1, padding=(0,1 if keepWide else 0)),
                nn.GroupNorm(getGroupSize(2*dim),2*dim),
                nn.LeakyReLU(leak,True),
                nn.AvgPool2d(2),
                nn.Conv2d(2*dim, 2*dim, 3, stride=1, padding=(0,1 if keepWide else 0)),
                nn.GroupNorm(getGroupSize(2*dim),2*dim),
                nn.LeakyReLU(leak,True),
                nn.AvgPool2d(2),
                nn.Conv2d(2*dim, 4*dim, 3, stride=1, padding=(0,1 if keepWide else 0)),
                nn.GroupNorm(getGroupSize(4*dim),4*dim),
                nn.Dropout2d(0.05,True),
                nn.LeakyReLU(leak,True),
                nn.Conv2d(4*dim, 2*dim, 3, stride=1, padding=(0,1 if keepWide else 0)), #after this it should be flat
                nn.GroupNorm(getGroupSize(2*dim),2*dim),
                nn.Dropout2d(0.025,True),
                nn.LeakyReLU(leak,True),
                nn.AvgPool2d((1,2)), #flat, so only operate horz
                nn.Conv2d(2*dim, 4*dim, (1,3), stride=1, padding=(0,1 if keepWide else 0)),
                nn.GroupNorm(getGroupSize(4*dim),4*dim),
                nn.Dropout2d(0.025,True),
                nn.LeakyReLU(leak,True),
                nn.Conv2d(4*dim, 4*dim, (1,3), stride=1, padding=(0,1 if keepWide else 0)),
                nn.GroupNorm(getGroupSize(4*dim),4*dim),
                nn.Dropout2d(0.025,True),
                nn.LeakyReLU(leak,True),
                nn.AvgPool2d((1,2)), #flat, so only operate horz
                nn.Conv2d(4*dim, 4*dim, (1,3), stride=1, padding=(0,1 if keepWide else 0)),
                nn.GroupNorm(getGroupSize(4*dim),4*dim),
                nn.Dropout2d(0.025,True),
                nn.LeakyReLU(leak,True),
                nn.Conv2d(4*dim, class_size, 1, stride=1, padding=(0,0)),
                )




        #self.fc = SpectralNorm(nn.Linear(w_g * w_g * 512, 1))

    def forward(self, x):
        batch_size = x.size(0)
        return F.adaptive_avg_pool2d(self.conv1(x),1).view(batch_size,self.class_size)
