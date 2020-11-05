
import torch.nn as nn
import torch
import torch.nn.functional as F
from .discriminator import SpectralNorm
import cv2

class TestImageDiscriminator(nn.Module):
    def __init__(self):
        super(TestImageDiscriminator, self).__init__()
        self.image = cv2.imread('../data/synthABC/0.png',0)
        self.image = torch.from_numpy(self.image).float()[None,None,:,:]
        self.image =1-(self.image//127)
        self.image = self.image.cuda()
    def forward(self, label,style,x,return_features=False):
        #import pdb;pdb.set_trace()
        if x.size(3)<self.image.size(3):
            use_image = self.image[:,:,:,:x.size(3)]
        else:
            use_image = self.image
        return [-torch.abs(x-use_image).mean(dim=2).mean(dim=2)]
class TestCondDiscriminator(nn.Module):

    def __init__(self,class_size,style_size,dim=64, global_pool=True, no_high=True,keepWide=True,use_style=True,use_pixel_stats=False,use_cond=True,global_only=False,pool_med=True):
        super(TestCondDiscriminator, self).__init__()
        self.no_high=no_high
        self.use_style=use_style
        self.use_cond=use_cond
        self.use_pM=not global_only
        self.pool_med=pool_med
        if not use_cond:
            class_size=0
        leak=0.1
        if use_style:
            style_proj1_size=dim
            style_proj2_size=2*dim
            self.style_proj1 = nn.Linear(style_size,style_proj1_size,bias=False)
            self.style_proj2 = nn.Linear(style_size,style_proj2_size,bias=False)
        else:
            style_proj1_size=0
            style_proj2_size=0

        self.in_conv = nn.Sequential(
                SpectralNorm(nn.Conv2d(1, dim, 7, stride=1, padding=(0,3 if keepWide else 0))),
                nn.LeakyReLU(leak,True)
                )

        convs1= [
                SpectralNorm(nn.Conv2d(style_proj1_size+dim, dim, 3, stride=1, padding=(0,1 if keepWide else 0))),
                nn.LeakyReLU(leak,True),
                nn.AvgPool2d(2),
                SpectralNorm(nn.Conv2d(dim, 2*dim, 3, stride=1, padding=(0,1 if keepWide else 0))),
                nn.Dropout2d(0.05,True),
                nn.LeakyReLU(leak,True),
                ]


        self.convs1 = nn.Sequential(*convs1)
        self.convs2 = nn.Sequential(
                SpectralNorm(nn.Conv2d(style_proj2_size+2*dim, 2*dim, 3, stride=1, padding=(0,1 if keepWide else 0))),
                nn.LeakyReLU(leak,True),
                nn.AvgPool2d(2),
                )
        self.convs3 = nn.Sequential(
                SpectralNorm(nn.Conv2d(class_size+2*dim, 2*dim, 3, stride=1, padding=(0,1 if keepWide else 0))),
                nn.LeakyReLU(leak,True),
                nn.AvgPool2d(2),
                SpectralNorm(nn.Conv2d(2*dim, 4*dim, 3, stride=1, padding=(0,1 if keepWide else 0))),
                nn.Dropout2d(0.05,True),
                nn.LeakyReLU(leak,True),
                #SpectralNorm(nn.Conv2d(4*dim, 4*dim, 4, stride=2, padding=(0,0))),
                #nn.Dropout2d(0.05,True),
                #nn.LeakyReLU(leak,True),
                )
        self.finalMed = nn. Sequential(
                #SpectralNorm(nn.Conv2d(4*dim, 4*dim, 3, stride=1, padding=(0,0))),
                #nn.Dropout2d(0.05,True),
                #nn.LeakyReLU(leak,True),
                SpectralNorm(nn.Conv2d(4*dim, 1, 3, stride=1, padding=(0,1 if keepWide else 0))),
                )
        #self.finalLow = nn. Sequential(
        #        SpectralNorm(nn.Conv2d(4*dim, 4*dim, 4, stride=2, padding=(0,0))),
        #        nn.Dropout2d(0.05,True),
        #        nn.LeakyReLU(leak,True),
        #        SpectralNorm(nn.Conv2d(4*dim, 1, 1, stride=1, padding=(0,0))),
        #        )
        if not no_high:
            self.finalHigh = nn. Sequential(
                    SpectralNorm(nn.Conv2d(style_proj2_size+2*dim, 2*dim, (3,3), stride=1, padding=(0,0))),
                    nn.Dropout2d(0.05,True),
                    nn.LeakyReLU(leak,True),
                    SpectralNorm(nn.Conv2d(2*dim, 1, 1, stride=1, padding=(0,0))),
                    )
        self.global_pool = global_pool
        self.use_pixel_stats = use_pixel_stats
        if global_pool:
            self.convs4 = nn.Sequential(
                    SpectralNorm(nn.Conv2d(4*dim, 2*dim, 3, stride=1, padding=(0,1 if keepWide else 0))), #after this it should be flat
                    nn.Dropout2d(0.025,True),
                    nn.LeakyReLU(leak,True),
                    nn.AvgPool2d((1,2)), #flat, so only operate horz
                    SpectralNorm(nn.Conv2d(2*dim, 4*dim, (1,3), stride=1, padding=(0,1 if keepWide else 0))),
                    nn.Dropout2d(0.025,True),
                    nn.LeakyReLU(leak,True),
                    )
            SIZE=1*5
            self.gp_final = nn.Sequential(
                    nn.Linear(SIZE*4*dim + (2 if use_pixel_stats else 0),2*dim),
                    nn.LeakyReLU(leak,True),
                    nn.Linear(2*dim,1)
                    )
        else:
            assert(not use_pixel_stats)

        #self.fc = SpectralNorm(nn.Linear(w_g * w_g * 512, 1))

    def forward(self, label,style,x,return_features=False):
        if x.size(3)<94:
            diff = 80-x.size(3)
            x = F.pad(x,(diff//2,diff//2+diff%2))
        elif x.size(3)>94:
            print('warning image is width {}'.format(x.size(3)))
            x = x[...,:94]
        if self.use_style:
            style1 = self.style_proj1(style)
            style2 = self.style_proj2(style)
        batch_size = x.size(0)
        if self.use_pixel_stats:
            mean_pix = x.view(batch_size,-1).mean(dim=1)
            var_pix = x.view(batch_size,-1).var(dim=1)
        
        x = self.in_conv(x) #64x58xW
        if self.use_style:
            x = torch.cat((x,style1[:,:,None,None].expand(-1,-1,x.size(2),x.size(3))), dim=1) #128x58xW

        m = self.convs1(x) #128x26xW
        if self.use_style:
            m = torch.cat((m,style2[:,:,None,None].expand(-1,-1,m.size(2),m.size(3))), dim=1) #256x26xW
        #print('{} ?= 256x26xW'.format(m.size()))

        mL = self.convs2(m) #128x12xW
        #print('{} ?= 128x12xW'.format(mL.size()))
        if self.use_cond:
            label = label.permute(1,2,0)
            label = label[:,:,None,:mL.size(3)].expand(-1,-1,mL.size(2),-1)
            if label.size(3)<mL.size(3):
                diff = mL.size(3)-label.size(3)
                label = F.pad(label,(diff//2+diff%2,diff//2))
            mL = torch.cat((mL,label), dim=1) #128+Cx12xW
        mL = self.convs3(mL) #256x4xW
        #print('{} ?= 128x3xW'.format(mL.size()))
        #pL = self.finalLow(mL)
        if return_features:
            return mL,self.convs4(mL)
        pM = self.finalMed(mL)
        if not self.no_high:
            pH = self.finalHigh(m)
        #return 0.5*F.adaptive_avg_pool2d(mH,1) + 0.5*F.adaptive_avg_pool2d(mL,1)
        if self.global_pool:
            mL = self.convs4(mL)
            #gp = F.adaptive_avg_pool2d(mL,1).view(batch_size,-1)
            gp = mL.view(batch_size,-1)
            if self.use_pixel_stats:
                gp = torch.cat((gp,mean_pix[:,None],var_pix[:,None]),dim=1)
            gp = self.gp_final(gp)

            pM = pM.view(batch_size,-1)
            if not self.no_high:
                pH = pH.view(batch_size,-1)
                if self.pool_med:
                    return [(pM.mean()+pH.mean()+gp.mean())/3]
                return [pM,pH,gp]
            
            if self.use_pM:
                if self.pool_med:
                    return [(pM.mean()+gp.mean())]
                return [pM,gp]
            else:
                return [gp]
        elif not self.no_high:
            return torch.cat( (pM.view(batch_size,-1),pH.view(batch_size,-1)), dim=1)#.view(-1)
        else:
            return pM.view(batch_size,-1)










class TestSmallCondDiscriminator(nn.Module):

    def __init__(self,class_size,style_size,dim=64, global_pool=True, no_high=True,keepWide=True,use_style=True,use_pixel_stats=False,use_cond=True,global_only=False,pool_med=True):
        super(TestSmallCondDiscriminator, self).__init__()
        self.no_high=no_high
        self.use_style=use_style
        self.use_cond=use_cond
        self.use_pM=not global_only
        self.pool_med=pool_med
        if not use_cond:
            class_size=0
        leak=0.1
        if use_style:
            style_proj1_size=dim
            style_proj2_size=2*dim
            self.style_proj1 = nn.Linear(style_size,style_proj1_size,bias=False)
            self.style_proj2 = nn.Linear(style_size,style_proj2_size,bias=False)
        else:
            style_proj1_size=0
            style_proj2_size=0

        self.in_conv = nn.Sequential(
                SpectralNorm(nn.Conv2d(1, dim, 7, stride=1, padding=(0,3 if keepWide else 0))),
                nn.LeakyReLU(leak,True)
                )

        convs1= [
                SpectralNorm(nn.Conv2d(style_proj1_size+dim, dim, 3, stride=1, padding=(0,1 if keepWide else 0))),
                nn.LeakyReLU(leak,True),
                nn.AvgPool2d(2),
                SpectralNorm(nn.Conv2d(dim, 2*dim, 3, stride=1, padding=(0,1 if keepWide else 0))),
                nn.Dropout2d(0.05,True),
                nn.LeakyReLU(leak,True),
                ]


        self.convs1 = nn.Sequential(*convs1)
        self.convs2 = nn.Sequential(
                SpectralNorm(nn.Conv2d(style_proj2_size+2*dim, 2*dim, 3, stride=1, padding=(0,1 if keepWide else 0))),
                nn.LeakyReLU(leak,True),
                #nn.AvgPool2d(2),
                )
        self.convs3 = nn.Sequential(
                SpectralNorm(nn.Conv2d(class_size+2*dim, 2*dim, 3, stride=1, padding=(0,1 if keepWide else 0))),
                nn.LeakyReLU(leak,True),
                nn.AvgPool2d(2),
                SpectralNorm(nn.Conv2d(2*dim, 4*dim, 3, stride=1, padding=(0,1 if keepWide else 0))),
                nn.Dropout2d(0.05,True),
                nn.LeakyReLU(leak,True),
                #SpectralNorm(nn.Conv2d(4*dim, 4*dim, 4, stride=2, padding=(0,0))),
                #nn.Dropout2d(0.05,True),
                #nn.LeakyReLU(leak,True),
                )
        self.global_pool = global_pool
        self.use_pixel_stats = use_pixel_stats
        if global_pool:
            self.convs4 = nn.Sequential(
                    SpectralNorm(nn.Conv2d(4*dim, 2*dim, 3, stride=1, padding=(0,1 if keepWide else 0))), #after this it should be flat
                    nn.Dropout2d(0.025,True),
                    nn.LeakyReLU(leak,True),
                    nn.AvgPool2d((1,2)), #flat, so only operate horz
                    SpectralNorm(nn.Conv2d(2*dim, 4*dim, (1,3), stride=1, padding=(0,1 if keepWide else 0))),
                    nn.Dropout2d(0.025,True),
                    nn.LeakyReLU(leak,True),
                    )
            SIZE=1*5
            self.gp_final = nn.Sequential(
                    nn.Linear(SIZE*4*dim + (2 if use_pixel_stats else 0),2*dim),
                    nn.LeakyReLU(leak,True),
                    nn.Linear(2*dim,1)
                    )
        else:
            assert(not use_pixel_stats)

        #self.fc = SpectralNorm(nn.Linear(w_g * w_g * 512, 1))

    def forward(self, label,style,x,return_features=False):
        if x.size(3)<48:
            diff = 48-x.size(3)
            x = F.pad(x,(0,diff))
        elif x.size(3)>48:
            print('warning image is width {}'.format(x.size(3)))
            x = x[...,:48]
        if self.use_style:
            style1 = self.style_proj1(style)
            style2 = self.style_proj2(style)
        batch_size = x.size(0)
        if self.use_pixel_stats:
            mean_pix = x.view(batch_size,-1).mean(dim=1)
            var_pix = x.view(batch_size,-1).var(dim=1)
        x = self.in_conv(x) #64x58xW
        if self.use_style:
            x = torch.cat((x,style1[:,:,None,None].expand(-1,-1,x.size(2),x.size(3))), dim=1) #128x58xW

        m = self.convs1(x) #128x26xW
        if self.use_style:
            m = torch.cat((m,style2[:,:,None,None].expand(-1,-1,m.size(2),m.size(3))), dim=1) #256x26xW
        #print('{} ?= 256x26xW'.format(m.size()))

        mL = self.convs2(m) #128x12xW
        #print('{} ?= 128x12xW'.format(mL.size()))
        if self.use_cond:
            label = label.permute(1,2,0)
            label = label[:,:,None,:mL.size(3)].expand(-1,-1,mL.size(2),-1)
            if label.size(3)<mL.size(3):
                diff = mL.size(3)-label.size(3)
                label = F.pad(label,(diff//2+diff%2,diff//2))
            mL = torch.cat((mL,label), dim=1) #128+Cx12xW
        mL = self.convs3(mL) #256x4xW
        #print('{} ?= 128x3xW'.format(mL.size()))
        #pL = self.finalLow(mL)
        if return_features:
            return mL,self.convs4(mL)
        #return 0.5*F.adaptive_avg_pool2d(mH,1) + 0.5*F.adaptive_avg_pool2d(mL,1)
        mL = self.convs4(mL)
        #gp = F.adaptive_avg_pool2d(mL,1).view(batch_size,-1)
        gp = mL.view(batch_size,-1)
        if self.use_pixel_stats:
            gp = torch.cat((gp,mean_pix[:,None],var_pix[:,None]),dim=1)
        gp = self.gp_final(gp)
        return [gp]
