# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required

from torch import Tensor
from torch.nn import Parameter

channels = 1
leak = 0.1
#w_g = 4


class DownDiscriminator(nn.Module):
    def __init__(self):
        super(DownDiscriminator, self).__init__()

        self.conv1 = SpectralNorm(nn.Conv2d(channels, 64, 3, stride=1, padding=(1,1)))

        self.conv2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(0,0)))
        self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, padding=(0,0)))
        self.conv4 = SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(0,0)))
        self.conv5 = SpectralNorm(nn.Conv2d(128, 256, 3, stride=1, padding=(0,0)))
        self.conv6 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(0,0)))
        self.conv7 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(0,0)))
        self.final = SpectralNorm(nn.Conv2d(256, 1, (1,3), stride=1, padding=(0,0)))


        #self.fc = SpectralNorm(nn.Linear(w_g * w_g * 512, 1))

    def forward(self, x):
        m = x
        m = nn.LeakyReLU(leak)(self.conv1(m))
        m = nn.LeakyReLU(leak)(self.conv2(m))
        m = nn.LeakyReLU(leak)(self.conv3(m))
        m = nn.LeakyReLU(leak)(self.conv4(m))
        m = nn.LeakyReLU(leak)(self.conv5(m))
        m = nn.LeakyReLU(leak)(self.conv6(m))
        m = nn.LeakyReLU(leak)(self.conv7(m))
        m = self.final(m)
        return F.adaptive_avg_pool2d(m,1)

class TwoScaleDiscriminator(nn.Module):
    def __init__(self):
        super(TwoScaleDiscriminator, self).__init__()

        self.conv1 = SpectralNorm(nn.Conv2d(channels, 64, 3, stride=1, padding=(1,1)))

        self.conv2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(0,0)))
        self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, padding=(0,0)))
        self.conv4 = SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(0,0)))
        self.conv5 = SpectralNorm(nn.Conv2d(128, 256, 3, stride=1, padding=(0,0)))
        self.conv6 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(0,0)))
        self.conv7 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(0,0)))
        self.final = SpectralNorm(nn.Conv2d(256, 1, (1,3), stride=1, padding=(0,0)))
        self.finalHigh = SpectralNorm(nn.Conv2d(256, 1, (3,3), stride=1, padding=(0,0)))


        #self.fc = SpectralNorm(nn.Linear(w_g * w_g * 512, 1))

    def forward(self, x):
        m = x
        m = nn.LeakyReLU(leak)(self.conv1(m))
        m = nn.LeakyReLU(leak)(self.conv2(m))
        m = nn.LeakyReLU(leak)(self.conv3(m))
        m = nn.LeakyReLU(leak)(self.conv4(m))
        m = nn.LeakyReLU(leak)(self.conv5(m))
        mL = nn.LeakyReLU(leak)(self.conv6(m))
        mL = nn.LeakyReLU(leak)(self.conv7(mL))
        mL = self.final(mL)
        mH = self.finalHigh(m)
        return 0.5*F.adaptive_avg_pool2d(mH,1) + 0.5*F.adaptive_avg_pool2d(mL,1)

class TwoScaleBetterDiscriminator(nn.Module):
    def __init__(self, more_low=False,dim=64, global_pool=False):
        super(TwoScaleBetterDiscriminator, self).__init__()

        convs1 = []
        if more_low:
            convs1+= [
                SpectralNorm(nn.Conv2d(channels, dim, 7, stride=1, padding=(3,3))),
                nn.LeakyReLU(leak,True),
                SpectralNorm(nn.Conv2d(dim, dim, 3, stride=1, padding=(1,1))),
                nn.LeakyReLU(leak,True),
                ]
        else:
            convs1+= [
                SpectralNorm(nn.Conv2d(channels, dim, 3, stride=1, padding=(1,1))),
                nn.LeakyReLU(leak,True),
                ]
        convs1+= [
                SpectralNorm(nn.Conv2d(dim, dim, 4, stride=2, padding=(0,0))),
                nn.LeakyReLU(leak,True),
                SpectralNorm(nn.Conv2d(dim, 2*dim, 3, stride=1, padding=(0,0))),
                nn.Dropout(0.05,True),
                nn.LeakyReLU(leak,True),
                ]


        self.convs1 = nn.Sequential(*convs1)
        self.convs2 = nn.Sequential(
                SpectralNorm(nn.Conv2d(2*dim, 2*dim, 4, stride=2, padding=(0,0))),
                nn.LeakyReLU(leak,True),
                SpectralNorm(nn.Conv2d(2*dim, 4*dim, 3, stride=1, padding=(0,0))),
                nn.Dropout(0.05,True),
                nn.LeakyReLU(leak,True),
                #SpectralNorm(nn.Conv2d(4*dim, 4*dim, 4, stride=2, padding=(0,0))),
                #nn.Dropout(0.05,True),
                #nn.LeakyReLU(leak,True),
                )
        self.finalMed = nn. Sequential(
                #SpectralNorm(nn.Conv2d(4*dim, 4*dim, 3, stride=1, padding=(0,0))),
                #nn.Dropout(0.05,True),
                #nn.LeakyReLU(leak,True),
                SpectralNorm(nn.Conv2d(4*dim, 1, 1, stride=1, padding=(0,0))),
                )
        #self.finalLow = nn. Sequential(
        #        SpectralNorm(nn.Conv2d(4*dim, 4*dim, 4, stride=2, padding=(0,0))),
        #        nn.Dropout(0.05,True),
        #        nn.LeakyReLU(leak,True),
        #        SpectralNorm(nn.Conv2d(4*dim, 1, 1, stride=1, padding=(0,0))),
        #        )
        self.finalHigh = nn. Sequential(
                SpectralNorm(nn.Conv2d(2*dim, 2*dim, (3,3), stride=1, padding=(0,0))),
                nn.Dropout(0.05,True),
                nn.LeakyReLU(leak,True),
                SpectralNorm(nn.Conv2d(2*dim, 1, 1, stride=1, padding=(0,0))),
                )
        self.global_pool = global_pool
        if global_pool:
            self.convs3 = nn.Sequential(
                    SpectralNorm(nn.Conv2d(4*dim, 4*dim, 4, stride=2, padding=(0,0))),
                    nn.Dropout(0.05,True),
                    nn.LeakyReLU(leak,True),
                    SpectralNorm(nn.Conv2d(4*dim, 4*dim, 3, stride=1, padding=(0,0))),
                    nn.Dropout(0.05,True),
                    nn.LeakyReLU(leak,True),
                    )
            self.gp_final = nn.Linear(4*dim,1)

        #self.fc = SpectralNorm(nn.Linear(w_g * w_g * 512, 1))

    def forward(self, x):
        m = self.convs1(x)
        mL = self.convs2(m)
        #pL = self.finalLow(mL)
        pM = self.finalMed(mL)
        pH = self.finalHigh(m)
        #return 0.5*F.adaptive_avg_pool2d(mH,1) + 0.5*F.adaptive_avg_pool2d(mL,1)
        batch_size = x.size(0)
        if self.global_pool:
            mL = self.convs3(mL)
            gp = F.adaptive_avg_pool2d(mL,1).view(batch_size,-1)
            gp = self.gp_final(gp)

            pM = F.adaptive_avg_pool2d(pM,1).view(batch_size,-1)
            pH = F.adaptive_avg_pool2d(pH,1).view(batch_size,-1)
            return torch.cat( (pM,pH,gp), dim=1)#.view(-1)
        else:
            return torch.cat( (pM.view(batch_size,-1),pH.view(batch_size,-1)), dim=1)#.view(-1)

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


