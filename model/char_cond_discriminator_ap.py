import torch.nn as nn
import torch
import torch.nn.functional as F
from .discriminator import SpectralNorm
from collections import defaultdict

class CharDisc(nn.Module):
    def __init__(self,input_dim,style_size):
        super(CharDisc, self).__init__()
        self.conv = nn.Sequential(
                        SpectralNorm(nn.Conv2d(input_dim,input_dim//2,3)),
                        nn.LeakyReLU(0.1,True)
                        )
        self.fc = nn.Sequential(
                        nn.Linear(input_dim//2+style_size,input_dim//2),
                        nn.ReLU(True),
                        nn.Linear(input_dim//2,1)
                        )


    def forward(self,x,style=None):
        batch_size = x.size(0)
        x = self.conv(x)
        x = F.adaptive_avg_pool2d(x,1).view(batch_size,-1)
        if style is not None:
            pred = self.fc( torch.cat((x, style),dim=1) )
        else:
            pred = self.fc(x)
        return pred

class CharCondDiscriminatorAP(nn.Module):
    def __init__(self,class_size,style_size,char_style_size,dim=64, global_pool=True, no_high=True,keepWide=True,use_style=True):
        super(CharCondDiscriminatorAP, self).__init__()
        self.no_high=no_high
        self.window=2
        self.n_class=class_size
        leak=0.1
        self.use_style=use_style
        if not use_style:
            char_style_size=0
            self.char_style_size=0
            style_proj1_size=0
            style_proj2_size=0
        else:
            self.char_style_size=char_style_size
            style_proj1_size=dim//2
            style_proj2_size=dim
            self.style_proj1 = nn.Linear(style_size,style_proj1_size,bias=False)
            self.style_proj2 = nn.Linear(style_size,style_proj2_size,bias=False)

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
                SpectralNorm(nn.Conv2d(class_size+char_style_size+2*dim, 2*dim, 3, stride=1, padding=(0,1 if keepWide else 0))),
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
                    nn.Linear(4*dim,2*dim),
                    nn.LeakyReLU(leak,True),
                    nn.Linear(2*dim,1)
                    )

        #self.fc = SpectralNorm(nn.Linear(w_g * w_g * 512, 1))
        self.char_disc = nn.ModuleList()
        for char_n in range(self.n_class):
            self.char_disc.append( CharDisc(dim*4,char_style_size) )
                                        

    def forward(self, label,style,x,return_features=False,author_vector=None):
        assert(author_vector is None)
        batch_size=x.size(0)
        if self.use_style:
            g_style,spaced_style,char_style = style
            style1 = self.style_proj1(g_style)
            style2 = self.style_proj2(g_style)
        
        x = self.in_conv(x) #64x58xW
        if self.use_style:
            x = torch.cat((x,style1[:,:,None,None].expand(-1,-1,x.size(2),x.size(3))), dim=1) #128x58xW

        m = self.convs1(x) #128x26xW
        if self.use_style:
            m = torch.cat((m,style2[:,:,None,None].expand(-1,-1,m.size(2),m.size(3))), dim=1) #256x26xW
        #print('{} ?= 256x26xW'.format(m.size()))

        mL = self.convs2(m) #128x12xW
        #print('{} ?= 128x12xW'.format(mL.size()))
        label = label.permute(1,2,0) #permute to Batch,Channel,Width
        text_chars = label[:,:,:].argmax(dim=1)
        label = label[:,:,None,:].expand(-1,-1,mL.size(2),-1)
        if self.use_style:
            style_char = spaced_style.permute(1,2,0) #permute to Batch,Channel,Width
            style_char = style_char[:,:,None,:].expand(-1,-1,mL.size(2),-1) #add height
            style_and_label = torch.cat((style_char,label),dim=1)
        else:
            style_and_label = label
        #print('label: {}, mL:{}'.format(label.size(3),mL.size(3)))
        #assert(label.size(3)-mL.size(3) < 10)
        style_and_label = style_and_label[:,:,:,:mL.size(3)]
        if style_and_label.size(3)<mL.size(3):
            diff = mL.size(3)-style_and_label.size(3)
            style_and_label = F.pad(style_and_label,(diff//2+diff%2,diff//2))
        else:
            diff=0



        mL = torch.cat((mL,style_and_label), dim=1) #128+Cx12xW
        mL = self.convs3(mL) #256x4xW

        if return_features:
            return mL

        #gather patches for each character in line, we combine across batches
        char_patches=defaultdict(list)
        char_info=defaultdict(list)
        char_style_for_patches=defaultdict(list)
        for b in range(batch_size):
            for i_orig in range(0,text_chars.size(1)):
                char_n = text_chars[b,i_orig].item()
                if char_n>0:
                    i=(i_orig+diff//2+diff%2)//2 #as mL has gone through 2x2 avg pool 
                    #if i == mL.size(3)-1:
                    #    i-=1
                    if i < mL.size(3):
                        left = max(0,i-self.window)
                        pad_left = left-(i-self.window)
                        right = min(mL.size(3)-1,i+self.window)
                        pad_right = (i+self.window)-right
                        wind = mL[b:b+1,:,:,left:right+1]
                        if pad_left>0 or pad_right>0:
                            wind = F.pad(wind,(pad_left,pad_right))
                        #assert(wind.size(3)==5)
                        char_patches[char_n].append(wind)
                        char_info[char_n].append( b )
                        #store appropriate batches char style
                        if self.use_style:
                            char_style_for_patches[char_n].append( char_style[b:b+1,char_n] )
                    else:
                        print('alginement error (char condDisc), text: {}, image: {}, i_orig: {}, i: {}, mL: {}'.format(text_chars.size(1),style_and_label.size(3), i_orig, i, mL.size(3)))

        #make char predictions and then sort back to appropriate batches
        char_preds = [ [] for b in range(batch_size)]
        for char_n, patches in char_patches.items():
            patches = torch.cat(patches,dim=0)
            if self.use_style:
                char_style_for_char_n = torch.cat(char_style_for_patches[char_n],dim=0)
                char_n_pred = self.char_disc[char_n](patches,char_style_for_char_n)
            else:
                char_n_pred = self.char_disc[char_n](patches)
            for i,b in enumerate(char_info[char_n]):
                char_preds[b].append(char_n_pred[i])
        
        #average char_preds for each batch
        #pChar = torch.FloatTensor(batch_size).fill_(0).to(x.device)
        pChar=[]
        for b in range(batch_size):
            if len(char_preds[b])>0:
                pChar+=char_preds[b]
                #pChar[b] = torch.stack(char_preds[b],dim=0).sum(dim=0)
        if len(pChar)>0:
            pChar = torch.stack(pChar,dim=0)
        else:
            pChar = torch.FloatTensor(batch_size).fill_(0).to(x.device)

        #print('{} ?= 128x3xW'.format(mL.size()))
        #pL = self.finalLow(mL)
        pM = self.finalMed(mL)
        if not self.no_high:
            pH = self.finalHigh(m)
        #return 0.5*F.adaptive_avg_pool2d(mH,1) + 0.5*F.adaptive_avg_pool2d(mL,1)
        batch_size = x.size(0)
        if self.global_pool:
            mL = self.convs4(mL)
            gp = F.adaptive_avg_pool2d(mL,1).view(batch_size,-1)
            gp = self.gp_final(gp)

            pM = F.adaptive_avg_pool2d(pM,1).view(batch_size,-1)
            if not self.no_high:
                pH = F.adaptive_avg_pool2d(pH,1).view(batch_size,-1)
                return (pM+pH+gp)/6 + pChar/2
            else:
                return (pM+gp)/4 + pChar/2
        elif not self.no_high:
            return (pM.view(batch_size,-1) +pH.view(batch_size,-1))/4 + pChar/2
        else:
            return [pM,pChar]
