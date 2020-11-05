
from torch import nn
import torch
import torch.nn.functional as F
from .MUNIT_networks import Conv2dBlock, GenWithSkips, ResBlocks

class CharSpacedDecoderWithMask(nn.Module):
    #def __init__(self, n_class, n_resTopA=1, n_resLow=1, n_resLowest=1, n_resTopB=1, dim=256, output_dim=1, style_size=256, res_norm='adain', activ='relu', pad_type='zero', space_style_size=32):
    def __init__(self, n_class, n_res1=2, n_res2=2, n_res3=1, dim=128, output_dim=1, style_size=256, res_norm='adain', activ='relu', pad_type='zero', char_style_size=32, dist_map_text=False,use_skips=False, noise=False, extra_text=False):
        super(CharSpacedDecoderWithMask, self).__init__()
        self.extra_text = extra_text
        self.char_style_size = char_style_size
        dim1D=dim
        input_size = n_class+char_style_size
        self.conv1D=[]
        self.conv1D += [Conv2dBlock(input_size ,dim1D, (1,3), (1,1), (0,1), norm='none', activation=activ, pad_type=pad_type, transpose=True)] #1xc
        self.conv1D += [Conv2dBlock(dim1D ,dim1D, (1,3), (1,1), (0,1), norm=res_norm, activation=activ, pad_type=pad_type, transpose=True)]  #1xc
        #self.conv1D += [ResBlocks(n_res1, dim1D, res_norm, activ, pad_type=pad_type, fixed=True)]
        #self.conv1D += [nn.Upsample(scale_factor=2),                                                                 #1x2c1
        #                Conv2dBlock(dim1D, dim1D // 2, 3, 1, 1, norm='ln', activation=activ, pad_type=pad_type)]
        self.conv1D += [Conv2dBlock(dim1D ,dim1D//2, (1,4), (1,2), (0,1), norm=res_norm, activation=activ, pad_type=pad_type, transpose=True)]  #16xc
        dim1D //= 2
        #self.conv1D += [ResBlocks(n_res2, dim1D, res_norm, activ, pad_type=pad_type, fixed=True)]
        #self.conv1D += [nn.Upsample(scale_factor=2),                                                                 #64x4c
        #                Conv2dBlock(dim1D, dim1D // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
        self.conv1D += [Conv2dBlock(dim1D ,dim1D, (1,4), (1,2), (0,1), norm=res_norm, activation=activ, pad_type=pad_type, transpose=True)]  #16xc
        #dim1D //= 2
        #self.conv1D += [ResBlocks(n_res3, dim1D, res_norm, activ, pad_type=pad_type, fixed=True)]
        self.conv1D += [Conv2dBlock(dim1D, dim1D, (1,3), 1, (0,2), norm='none', activation=activ, pad_type=pad_type)]

        if extra_text:
            dim1D+=n_class

        self.conv1D = nn.Sequential(*self.conv1D)
        if use_skips:
            self.conv2D = GenWithSkips(dim1D,dim,output_dim,n_res1,n_res2,n_res3,res_norm,activ,pad_type)
        else:
            self.conv2D = nn.Sequential(
                    Conv2dBlock(dim1D+1,dim, 3, 1, 1, norm=res_norm, activation=activ, pad_type=pad_type),
                    Conv2dBlock(dim,2*dim, (4,4), (2,2), 1, norm=res_norm, activation=activ, pad_type=pad_type),
                    Conv2dBlock(2*dim,4*dim, (4,4), (2,2), 1, norm=res_norm, activation=activ, pad_type=pad_type),
                    ResBlocks(n_res1, 4*dim, res_norm, activ, pad_type=pad_type, fixed=True),
                    Conv2dBlock(4*dim,2*dim, (4,4), (2,2), 1, norm=res_norm, activation=activ, pad_type=pad_type, transpose=True),
                    ResBlocks(n_res2, 2*dim, res_norm, activ, pad_type=pad_type, fixed=True),
                    Conv2dBlock(2*dim,dim, (4,4), (2,2), 1, norm=res_norm, activation=activ, pad_type=pad_type, transpose=True),
                    ResBlocks(n_res3, dim, res_norm, activ, pad_type=pad_type, fixed=True),
                    Conv2dBlock(dim,output_dim, (5,5), (1,1), 2, norm='none', activation='tanh', pad_type=pad_type),
                    )

        #self.conv2DTopA = nn.Sequential(
        #        Conv2dBlock(dim1D+1,dim, 3, 1, 1, norm=res_norm, activation=activ, pad_type=pad_type),
        #        ResBlocks(n_resTopA, dim, res_norm, activ, pad_type=pad_type, fixed=True),
        #        )
        #self.conv2DLow = nn.Sequential(
        #        Conv2dBlock(dim,dim, (4,4), (2,2), (1,1), norm=res_norm, activation=activ, pad_type=pad_type),
        #        ResBlocks(n_resLow, dim, res_norm, activ, pad_type=pad_type, fixed=True),
        #        )
        #self.conv2DLowest = nn.Sequential(
        #        Conv2dBlock(dim,dim, (4,4), (2,2), (1,1), norm=res_norm, activation=activ, pad_type=pad_type),
        #        ResBlocks(n_resLowest, dim, res_norm, activ, pad_type=pad_type, fixed=True),
        #        )
        #self.conv2DLowestUp = nn.Sequential(
        #        ResBlocks(1, dim, res_norm, activ, pad_type=pad_type, fixed=True),
        #        Conv2dBlock(dim,dim, (4,4), (2,2), (0,1), norm=res_norm, activation=activ, pad_type=pad_type, transpose=True),
        #        )
        #self.conv2DLowUp = nn.Sequential(
        #        Conv2dBlock(2*dim,dim, 3, 1, 1, norm=res_norm, activation=activ, pad_type=pad_type),
        #        ResBlocks(1, dim, res_norm, activ, pad_type=pad_type, fixed=True),
        #        Conv2dBlock(dim,dim, (4,4), (2,2), (0,1), norm=res_norm, activation=activ, pad_type=pad_type, transpose=True),
        #        #nn.ConstantPad2d((1,1,0,0),0)
        #        )
        #self.conv2DTopB = nn.Sequential(
        #        Conv2dBlock(2*dim,dim, 3, 1, 1, norm=res_norm, activation=activ, pad_type=pad_type),
        #        ResBlocks(n_resTopA, dim, res_norm, activ, pad_type=pad_type, fixed=True),
        #        Conv2dBlock(dim,output_dim, (5,5), (1,1), (2,2), norm='none', activation='tanh', pad_type=pad_type),
        #        )

        mid_size = (style_size+char_style_size)//2
        #self.space_proj = nn.Sequential( nn.Linear(style_size,mid_size),nn.ReLU(inplace=True),nn.Linear(mid_size,char_style_size) )
        self.dist_map_text=dist_map_text

        if noise:
            self.noise_scale = nn.Parameter(torch.FloatTensor(dim1D).fill_(0.5))

    def forward(self, mask, text,style, noise=False):
        g_style,spaced_style,char_style = style
        #AdaIN has been updated with style already
        #content: batch X char(onehot) X h(1) X width (this assumes the content is already spaced in the width dimension
        #style: batch X channels

        if self.dist_map_text:
            batch_size = mask.size(0)
            max_len=10
            add=None
            for b in range(batch_size):
                #print('batch {}'.format(b))
                start=0
                curIdx=-1
                for x in range(text.size(3)):
                    idx = text[b,:,0,x].argmax()
                    #print('x:{}, start:{}, curIdx:{}, idx:{}'.format(x,start,curIdx,idx))
                    if idx!=0 and x-start>0:
                        if curIdx==-1 and x-start>max_len:
                            start = x-max_len
                        step = (1-0.1)/((x-start+1)/2)
                        #print('step {} = (1-0.1)/(({}-{}+1)/2)'.format(step,x,start))
                        v=1-step
                        for xd in range((x-start)//2 +((x-start)%2)):
                            if idx!=-1:
                                text[b,idx,0,x-(xd+1)]=v
                                #print('text[{},{},{},{}]={}'.format(b,idx,0,x-(xd+1),v))
                            if curIdx!=-1:
                                text[b,curIdx,0,start+xd]=v
                                #print('text[{},{},{},{}]={}'.format(b,curIdx,0,start+xd,v))
                            v-=step

                    if idx!=0:
                        start=x+1
                        curIdx=idx
                x = text.size(3)
                if x-start>max_len:
                    start = x-max_len
                step = (1-0.1)/((x-start+1)/2)
                v=1-step
                for xd in range((x-start)//2 +1):
                    if curIdx!=-1:
                        text[b,curIdx,0,start+xd]=v
                        #print('pext[{},{},{},{}]={}'.format(b,curIdx,0,start+xd,v))
                    v-=step

        style = spaced_style.permute(1,2,0)
        style = style[:,:,None,:] #add height


        oneD = torch.cat( (text, style), dim=1)
        oneD = self.conv1D(oneD)
        oneD = oneD[:,:,:,:mask.size(3)] #clip to same size as input
        if oneD.size(3)<mask.size(3):
            diff = mask.size(3)-oneD.size(3)
            oneD = F.pad(oneD,(diff//2,diff//2+diff%2))
        im = oneD.expand(-1,-1,mask.size(2),-1)
        if noise:
            noise = torch.randn_like(im) * self.noise_scale[None,:,None,None]
            im = im+noise
        if self.extra_text:
            more_text = F.interpolate(text,(1,im.size(3)))
            more_text = more_text.expand(-1,-1,im.size(2),-1)
            im = torch.cat((mask,im,more_text),dim=1)
        else:
            im = torch.cat((mask,im),dim=1)
        oneD=None
        
        im = self.conv2D(im)
        #imTop = self.conv2DTopA(im)
        #imLow = self.conv2DLow(imTop)
        #imLowest = self.conv2DLowest(imLow)
        #imLowB = self.conv2DLowestUp(imLowest)
        #if imLowB.size(3)<imLow.size(3):
        #    diff = imLow.size(3)-imLowB.size(3)
        #    imLowB = F.pad(imLowB,(diff//2,diff//2+diff%2))
        #imLow = torch.cat((imLow,imLowB),dim=1)
        #imTopB = self.conv2DLowUp(imLow)
        #if imTopB.size(3)<imTop.size(3):
        #    diff = imTop.size(3)-imTopB.size(3)
        #    imTopB = F.pad(imTopB,(diff//2,diff//2+diff%2))
        #imTop = torch.cat((imTop, imTopB),dim=1)
        #im = self.conv2DTopB(imTop)
        return im
