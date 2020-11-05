import torch.nn as nn
import torch
import torch.nn.functional as F
from .discriminator import SpectralNorm

leak=0.05

class CondDiscriminator(nn.Module):
    def __init__(self,class_size,style_size,dim=64, global_pool=True, wide=False):
        super(CondDiscriminator, self).__init__()

        self.style_proj1 = nn.Linear(style_size,dim,bias=False)
        self.style_proj2 = nn.Linear(style_size,2*dim,bias=False)

        self.in_conv = nn.Sequential(
                SpectralNorm(nn.Conv2d(1, dim, 7, stride=1, padding=(0,3 if wide else 0))),
                nn.LeakyReLU(leak,True)
                )

        convs1= [
                SpectralNorm(nn.Conv2d(dim+dim, dim, 1, stride=1, padding=(0,1 if wide else 0))),
                nn.LeakyReLU(leak,True),
                SpectralNorm(nn.Conv2d(dim, 2*dim, 4, stride=2, padding=(0,1 if wide else 0))),
                nn.LeakyReLU(leak,True),
                SpectralNorm(nn.Conv2d(2*dim, 2*dim, 3, stride=1, padding=(0,1 if wide else 0))),
                nn.Dropout2d(0.05,True),
                nn.LeakyReLU(leak,True),
                ]


        self.convs1 = nn.Sequential(*convs1)
        self.convs2 = nn.Sequential(
                SpectralNorm(nn.Conv2d(2*dim+2*dim, 2*dim, 4, stride=2, padding=(0,1 if wide else 0))),
                nn.LeakyReLU(leak,True)
                )
        self.convs3 = nn.Sequential(
                SpectralNorm(nn.Conv2d(class_size+2*dim, 2*dim, 1, stride=1, padding=(0,0))),
                nn.LeakyReLU(leak,True),
                SpectralNorm(nn.Conv2d(2*dim, 4*dim, 3, stride=1, padding=(0,1 if wide else 0))),
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
                SpectralNorm(nn.Conv2d(4*dim, 1, 3, stride=1, padding=(0,1 if wide else 0))),
                )
        #self.finalLow = nn. Sequential(
        #        SpectralNorm(nn.Conv2d(4*dim, 4*dim, 4, stride=2, padding=(0,1 if wide else 0))),
        #        nn.Dropout2d(0.05,True),
        #        nn.LeakyReLU(leak,True),
        #        SpectralNorm(nn.Conv2d(4*dim, 1, 1, stride=1, padding=(0,1 if wide else 0))),
        #        )
        self.finalHigh = nn. Sequential(
                SpectralNorm(nn.Conv2d(2*dim+2*dim, 2*dim, (3,3), stride=1, padding=(0,1 if wide else 0))),
                nn.Dropout2d(0.05,True),
                nn.LeakyReLU(leak,True),
                SpectralNorm(nn.Conv2d(2*dim, 1, 1, stride=1, padding=(0,0))),
                )
        self.global_pool = global_pool
        if global_pool:
            self.convs4 = nn.Sequential(
                    SpectralNorm(nn.Conv2d(4*dim, 2*dim, 4, stride=2, padding=(0,1 if wide else 0))),
                    nn.Dropout2d(0.025,True),
                    nn.LeakyReLU(leak,True),
                    SpectralNorm(nn.Conv2d(2*dim, 4*dim, 3, stride=1, padding=(0,1 if wide else 0))),
                    nn.Dropout2d(0.025,True),
                    nn.LeakyReLU(leak,True),
                    )
            self.gp_final = nn.Sequential(
                    nn.Linear(4*dim,2*dim),
                    nn.LeakyReLU(leak,True),
                    nn.Linear(2*dim,1)
                    )

        #self.fc = SpectralNorm(nn.Linear(w_g * w_g * 512, 1))

    def forward(self, label,style,x):
        style1 = self.style_proj1(style)
        style2 = self.style_proj2(style)
        
        x = self.in_conv(x) #64x58xW
        x = torch.cat((x,style1[:,:,None,None].expand(-1,-1,x.size(2),x.size(3))), dim=1) #128x58xW

        m = self.convs1(x) #128x25xW
        m = torch.cat((m,style2[:,:,None,None].expand(-1,-1,m.size(2),m.size(3))), dim=1) #256x25xW

        mL = self.convs2(m) #128x10?11?xW
        label = label.permute(1,2,0)
        label = label[:,:,None,:mL.size(3)].expand(-1,-1,mL.size(2),-1)
        if label.size(3)<mL.size(3):
            diff = mL.size(3)-label.size(3)
            label = F.pad(label,(diff//2+diff%2,diff//2))
        mL = torch.cat((mL,label), dim=1)
        mL = self.convs3(mL)
        #pL = self.finalLow(mL)
        pM = self.finalMed(mL)
        pH = self.finalHigh(m)
        #return 0.5*F.adaptive_avg_pool2d(mH,1) + 0.5*F.adaptive_avg_pool2d(mL,1)
        batch_size = x.size(0)
        if self.global_pool:
            mL = self.convs4(mL)
            gp = F.adaptive_avg_pool2d(mL,1).view(batch_size,-1)
            gp = self.gp_final(gp)

            pM = F.adaptive_avg_pool2d(pM,1).view(batch_size,-1)
            pH = F.adaptive_avg_pool2d(pH,1).view(batch_size,-1)
            return torch.cat( (pM,pH,gp), dim=1)#.view(-1)
        else:
            return torch.cat( (pM.view(batch_size,-1),pH.view(batch_size,-1)), dim=1)#.view(-1)
