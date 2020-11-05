import torch.nn as nn
import torch
import torch.nn.functional as F
from .discriminator import SpectralNorm
from .net_builder import getGroupSize
from .attention import MultiHeadedAttention, PositionalEncoding

class CondDiscriminatorAP(nn.Module):

    def __init__(self,class_size,style_size,dim=64, global_pool=True, no_high=True,keepWide=True,use_style=True,use_pixel_stats=False,use_cond=True,global_only=False,use_low=False,add_noise_img=False,add_noise_cond=False,dist_map_content=False,convs3NoGroup=False,use_authors_size=0,use_attention=False,use_med=True,small=False):
        super(CondDiscriminatorAP, self).__init__()
        self.no_high=no_high
        self.use_pM=not global_only
        self.use_low = use_low
        self.use_med = use_med
        self.use_style=use_style
        self.use_cond=use_cond
        self.use_attention = use_attention
        self.add_noise_img=add_noise_img
        self.add_noise_cond=add_noise_cond
        self.dist_map_content=dist_map_content
        assert((not use_low) or (not global_pool))
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
        if use_authors_size is not None and use_authors_size>0:
            assert(not use_style)
            style_proj2_size=use_authors_size

        self.in_conv = nn.Sequential(
                nn.Conv2d(1, dim, 7, stride=1, padding=(0,3 if keepWide else 0)),
                nn.GroupNorm(getGroupSize(dim),dim), # Experiments with other GAN showed better results not using spectral on first layer
                nn.LeakyReLU(leak,True)
                )
    
        convs1_pad_v = 0 if not small else 1
        convs1= [
                SpectralNorm(nn.Conv2d(style_proj1_size+dim, dim, 3, stride=1, padding=(convs1_pad_v,1 if keepWide else 0))),
                nn.LeakyReLU(leak,True)
                ]
        if not small:
            convs1.append(nn.AvgPool2d(2),)
        convs1+=[
                SpectralNorm(nn.Conv2d(dim, 2*dim, 3, stride=1, padding=(convs1_pad_v,1 if keepWide else 0))),
                nn.Dropout2d(0.05,True),
                nn.LeakyReLU(leak,True),
                ]


        self.convs1 = nn.Sequential(*convs1)
        self.convs2 = nn.Sequential(
                SpectralNorm(nn.Conv2d(style_proj2_size+2*dim, 2*dim, 3, stride=1, padding=(0,1 if keepWide else 0))),
                nn.LeakyReLU(leak,True),
                nn.AvgPool2d(2),
                )
        if convs3NoGroup:
            convs3=[SpectralNorm(nn.Conv2d(class_size+2*dim, 2*dim, 3, stride=1, padding=(0,1 if keepWide else 0)))]
        else:
            convs3=[nn.Conv2d(class_size+2*dim, 2*dim, 3, stride=1, padding=(0,1 if keepWide else 0)),
                    nn.GroupNorm(getGroupSize(2*dim),2*dim)]
        convs3 += [
                nn.LeakyReLU(leak,True),
                nn.AvgPool2d(2),
                SpectralNorm(nn.Conv2d(2*dim, 4*dim, 3, stride=1, padding=(0,1 if keepWide else 0))),
                nn.Dropout2d(0.05,True),
                nn.LeakyReLU(leak,True),
                #SpectralNorm(nn.Conv2d(4*dim, 4*dim, 4, stride=2, padding=(0,0))),
                #nn.Dropout2d(0.05,True),
                #nn.LeakyReLU(leak,True),
                ]
        self.convs3 = nn.Sequential(*convs3)
        if self.use_med:
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
            self.gp_final = nn.Sequential(
                    nn.Linear(4*dim + (2 if use_pixel_stats else 0),2*dim),
                    nn.LeakyReLU(leak,True),
                    nn.Linear(2*dim,1)
                    )
        else:
            assert(not use_pixel_stats)
            if self.use_low:
                self.convs4 = nn.Sequential(
                    SpectralNorm(nn.Conv2d(4*dim, 2*dim, 3, stride=1, padding=(0,1 if keepWide else 0))), #after this it should be flat
                    nn.Dropout2d(0.025,True),
                    nn.LeakyReLU(leak,True),
                    nn.AvgPool2d((1,2)), #flat, so only operate horz
                    SpectralNorm(nn.Conv2d(2*dim, 4*dim, (1,3), stride=1, padding=(0,1 if keepWide else 0))),
                    nn.Dropout2d(0.025,True),
                    nn.LeakyReLU(leak,True),
                    SpectralNorm(nn.Conv2d(4*dim, 4*dim, (1,3), stride=1, padding=(0,1 if keepWide else 0))),
                    nn.Dropout2d(0.025,True),
                    nn.LeakyReLU(leak,True),
                    nn.AvgPool2d((1,2)), #flat, so only operate horz
                    SpectralNorm(nn.Conv2d(4*dim, 4*dim, (1,3), stride=1, padding=(0,1 if keepWide else 0))),
                    nn.Dropout2d(0.025,True),
                    nn.LeakyReLU(leak,True),
                    SpectralNorm(nn.Conv2d(4*dim, 1, 1, stride=1, padding=(0,0))),
                    )
            elif self.use_attention:
                self.convs4 = nn.Sequential(
                    SpectralNorm(nn.Conv2d(4*dim, 2*dim, 3, stride=1, padding=(0,1 if keepWide else 0))), #after this it should be flat
                    nn.Dropout2d(0.025,True),
                    nn.LeakyReLU(leak,True),
                    nn.AvgPool2d((1,2)), #flat, so only operate horz
                    SpectralNorm(nn.Conv2d(2*dim, 4*dim, (1,3), stride=1, padding=(0,1 if keepWide else 0))),
                    )
                self.posEnc = PositionalEncoding(4*dim,0.025)
                heads = max((4*dim)//32,2)
                self.attLayer1 = MultiHeadedAttention(heads,4*dim)
                self.attLinear1 = nn.Linear(4*dim,4*dim)
                self.attNorm1 = nn.GroupNorm(getGroupSize(4*dim),4*dim)
                self.attLayer2 = MultiHeadedAttention(heads,4*dim)
                self.attLinearPred = nn.Sequential(
                        SpectralNorm(nn.Linear(4*dim,4*dim)),
                        nn.Dropout2d(0.1,True),
                        nn.LeakyReLU(leak,True),
                        SpectralNorm(nn.Linear(4*dim,1))
                        )




        #self.fc = SpectralNorm(nn.Linear(w_g * w_g * 512, 1))

    def forward(self, label,style,x,return_features=False,author_vector=None):
        if self.add_noise_img:
            x=x+torch.randn_like(x)*0.7 #less strong
        if self.add_noise_cond:
            label=label+torch.randn_like(label)*0.7 #less strong
        if self.use_style:
            style1 = self.style_proj1(style)
            style2 = self.style_proj2(style)
        batch_size = x.size(0)
        if self.use_pixel_stats:
            mean_pix = x.view(batch_size,-1).mean(dim=1)
            var_pix = x.view(batch_size,-1).var(dim=1)
        
        x = self.in_conv(x) #64x58xW /x26x/
        if self.use_style:
            x = torch.cat((x,style1[:,:,None,None].expand(-1,-1,x.size(2),x.size(3))), dim=1) #128x58xW

        m = self.convs1(x) #128x26xW /x26x/
        if self.use_style:
            m = torch.cat((m,style2[:,:,None,None].expand(-1,-1,m.size(2),m.size(3))), dim=1) #256x26xW
        elif author_vector is not None:
            m = torch.cat((m,author_vector[:,:,None,None].expand(-1,-1,m.size(2),m.size(3))), dim=1)
        #print('{} ?= 256x26xW'.format(m.size()))

        mL = self.convs2(m) #128x12xW
        #print('{} ?= 128x12xW'.format(mL.size()))
        if self.use_cond:
            label = label.permute(1,2,0)
            if self.dist_map_content:
                for b in range(batch_size):
                    for x in range(label.size(2)):
                        idx = label[b,:,x].argmax()
                        if idx!=0:
                            if x>0:
                                label[b,idx,x-1]=max(label[b,idx,x-1],0.6)
                            if x>1:
                                label[b,idx,x-2]=max(label[b,idx,x-2],0.1)
                            if x<label.size(2)-1:
                                label[b,idx,x+1]=max(label[b,idx,x+1],0.6)
                            if x<label.size(2)-2:
                                label[b,idx,x+2]=max(label[b,idx,x+2],0.1)
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
        if self.use_med:
            pM = self.finalMed(mL)
        if not self.no_high:
            pH = self.finalHigh(m)
        #return 0.5*F.adaptive_avg_pool2d(mH,1) + 0.5*F.adaptive_avg_pool2d(mL,1)
        if self.global_pool:
            mL = self.convs4(mL)
            gp = F.adaptive_avg_pool2d(mL,1).view(batch_size,-1)
            if self.use_pixel_stats:
                gp = torch.cat((gp,mean_pix[:,None],var_pix[:,None]),dim=1)
            gp = self.gp_final(gp)

            #if self.pool_med:
            #    pM = F.adaptive_avg_pool2d(pM,1).view(batch_size,-1)
            #else:
            pM = pM.view(batch_size,-1)
            if not self.no_high:
                pH = pH.view(batch_size,-1)
                return [pM,pH,gp]
            if self.use_pM:
                return [pM,gp]#.view(-1)
            else:
                return [gp]
        elif not self.no_high:
            return [pM.view(batch_size,-1), pH.view(batch_size,-1)] # torch.cat( (pM.view(batch_size,-1),pH.view(batch_size,-1)), dim=1)#.view(-1)
        elif self.use_low:
            pL = self.convs4(mL)
            if self.use_med:
                return [pM.view(batch_size,-1),pL.view(batch_size,-1)]
            else:
                return [pL.view(batch_size,-1)]
            #return torch.cat( (pM.view(batch_size,-1),pL.view(batch_size,-1)), dim=1)#.view(-1)
        elif self.use_attention:
            mL = self.convs4(mL)
            batch_size = mL.size(0)
            channels = mL.size(1)
            length = mL.size(3)
            data = mL.view(batch_size,channels,length).permute(0,2,1) #swap to batch,horz,channels for attnetion
            data = self.posEnc(data)
            data1 = data.permute(0,2,1)
            #if self.use_attention=='across batches':
            #    data1=data1.view(1,batch_size*length,channels)
            data = self.attLayer1(data,data,data)
            data = data.permute(0,2,1)
            data2 = self.attNorm1((data+data1))
            #if not self.use_attention=='across batches':
            data = data2.permute(0,2,1).contiguous()
            data = data.view(batch_size*length,channels)
            data = self.attLinear1(data)
            data = data.view(batch_size,length,channels)
            data = data.permute(0,2,1)
            data3 = self.attNorm1(data+data2)
            #if not self.use_attention=='across batches':
            #    data3 = data3.view(batch_size,length,channels)
            data = data3.permute(0,2,1)
            data = self.attLayer2(data,data,data)
            data = data.permute(0,2,1)
            data = self.attNorm1(data+data3)
            #if not self.use_attention=='across batches':
            #    data = data.view(1,batch_size*length,channels)
            data = data.permute(0,2,1).contiguous().view(batch_size*length,channels)
            pred = self.attLinearPred(data)
            pred = pred.view(batch_size,length)
            return [pM.view(batch_size,-1),pred]
        else:
            return [pM.view(batch_size,-1)]
