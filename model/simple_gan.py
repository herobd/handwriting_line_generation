import torch
import torch.nn as nn
from .discriminator import SpectralNorm

class SimpleGen(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1, ngpu=1):
        super(SimpleGen, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 16
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 32
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 64
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 128
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 256
        )
        self.gen=self.main

    def forward(self, content=None,style=None,mask=None,return_intermediate=False,device=None,batch_size=None):
        noise_h_size = 13 #content.size(0)//8 +1
        if content is not None:
            batch_size = content.size(1)
            noise = torch.randn(batch_size, self.nz, 1, noise_h_size, device=content.device)
        else:
            noise = torch.randn(batch_size, self.nz, 1, noise_h_size, device=device)
        if noise.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, noise, range(self.ngpu))
        else:
            output = self.main(noise)
        return output


class SimpleDisc(nn.Module):
    def __init__(self, nc=1, ndf=64, ngpu=1):
        super(SimpleDisc, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 256/600
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 128/300
            #SpectralNorm( nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False) ),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 2),
            nn.GroupNorm(4,ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 64/150
            SpectralNorm( nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False) ),
            #nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False) ,
            #nn.BatchNorm2d(ndf * 4),
            #nn.GroupNorm(8,ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 32/75
            SpectralNorm( nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False) ),
            #nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False) ,
            #nn.BatchNorm2d(ndf * 8),
            #nn.GroupNorm(8,ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 16/37
            nn.Conv2d(ndf * 8, 1, (4,5), 1, 0, bias=False),
            # state size. 1 x 1 x 12/33
            #nn.Sigmoid() 
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return [output.view(input.size(0), -1)]
