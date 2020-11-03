# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from datasets.hw_dataset import PADDING_CONSTANT
from .net_builder import getGroupSize
from .attention import MultiHeadedAttention, PositionalEncoding
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

class ContentEncoderHW(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoderHW, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)




class HWDecoder(nn.Module):
    def __init__(self, n_class, n_res1, n_res2, n_res3, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(HWDecoder, self).__init__()
        self.model=[]
        self.model += [Conv2dBlock(n_class ,dim, (4,3), (4,3), 0, norm='none', activation=activ, pad_type=pad_type, transpose=True)] #4x3c
        self.model += [Conv2dBlock(dim ,dim, (4,4), (2,2), 1, norm=res_norm, activation=activ, pad_type=pad_type, transpose=True)]  #8x6c
        self.model += [Conv2dBlock(dim ,dim, (4,3), (2,1), 1, norm=res_norm, activation=activ, pad_type=pad_type, transpose=True)]  #16x6c
        self.model += [ResBlocks(n_res1, dim, res_norm, activ, pad_type=pad_type)]
        self.model += [nn.Upsample(scale_factor=2),                                                                 #32x12c1
                        Conv2dBlock(dim, dim // 2, 3, 1, 1, norm='ln', activation=activ, pad_type=pad_type)]
        dim //= 2
        self.model += [ResBlocks(n_res2, dim, res_norm, activ, pad_type=pad_type)]
        self.model += [nn.Upsample(scale_factor=2),                                                                 #64x24c
                        Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
        dim //= 2
        self.model += [ResBlocks(n_res3, dim, res_norm, activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]

        self.model = nn.Sequential(*self.model)

    def forward(self, content,style=None): #, noise=None):
        #AdaIN has been updated with style already
        #import pdb;pdb.set_trace()
        return self.model(content)

class AttentionDecoder(nn.Module):
    def __init__(self, n_class, n_res1, n_res2, n_res3, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(AttentionDecoder, self).__init__()
        self.conv=[]
        self.conv += [Conv2dBlock(emb_size ,dim, (4,3), (4,1), (0,1), norm='none', activation=activ, pad_type=pad_type, transpose=True)] #4xc
        self.conv += [Conv2dBlock(dim ,dim, (4,3), (2,1), 1, norm=res_norm, activation=activ, pad_type=pad_type, transpose=True)]  #8xc
        self.conv += [Conv2dBlock(dim ,dim, (4,3), (2,1), 1, norm=res_norm, activation=activ, pad_type=pad_type, transpose=True)]  #16xc
        self.conv += [ResBlocks(n_res1, dim, res_norm, activ, pad_type=pad_type)]
        self.conv += [nn.Upsample(scale_factor=2),                                                                 #32x2c1
                        Conv2dBlock(dim, dim // 2, 3, 1, 1, norm='ln', activation=activ, pad_type=pad_type)]
        dim //= 2
        self.conv += [ResBlocks(n_res2, dim, res_norm, activ, pad_type=pad_type)]
        self.conv += [nn.Upsample(scale_factor=2),                                                                 #64x4c
                        Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
        dim //= 2
        self.conv += [ResBlocks(n_res3, dim, res_norm, activ, pad_type=pad_type)]
        self.conv += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]

        self.conv = nn.Sequential(*self.conv)

        self.up_factor=4


        self.embContent = [ Conv1dBlock(n_class+loc_size+spacing_style_size,emb_size,3,1,0,norm='none', activation=activ, pad_type=pad_type),
                            Conv1dBlock(emb_size,emb_size,3,1,0,norm='none', activation=activ, pad_type=pad_type),
                            AccumSum(emb_size//2),
                            Conv1dBlock(emb_size,emb_size,3,1,0,norm='none', activation=activ, pad_type=pad_type),
                            Conv1dBlock(emb_size,emb_size,3,1,0,norm='none', activation=activ, pad_type=pad_type)
                            ]
        self.embContent = nn.Sequential(self.embContent)

    def forward(self, content,style,len): #, noise=None):
        #AdaIN has been updated with style already
        #content: batch X char(onehot) X width
        #style: batch X channels
        cWidth = content.size(2)
        spacing_style = self.space_proj(style)
        content_emb = torch.concat((content,spacing_style[...,None].expand(-1,-1,cWidth),self.loc_emb[:,:,:cWidth]),dim=1) #append one-hot-char, style, location
        content_emb = self.embContent(content_emb)

        start_len = len//self.up_factor

        im = self.loc_emb[:,:,:len].expand(content.size(0),-1,-1)
        im = self.att1(content_emb,im)
        im = self.conv(im[:,:,None,:])
        #im = self.att2(content_emb,im) #or self attention?
        #im = self.conv2(im)

        toPad = len-im.size(3)
        im = F.pad(im,(toPad//2,toPad//2 + toPad%2),value=PADDING_CONSTANT)

        return im

class Deep1DDecoder(nn.Module):
    def __init__(self, n_class, n_res1, n_res2, n_res3, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        emb_size=min(dim//2,n_class*2)
        super(Deep1DDecoder, self).__init__()
        self.deep1D =   [   Conv2dBlock(n_class,emb_size,(1,3),(1,3),0,norm='none', activation=activ, pad_type=pad_type, transpose=True),   #x3
                            Conv2dBlock(emb_size,emb_size,(1,5),1,(0,2),norm='adain', activation=activ, pad_type=pad_type),
                            Conv2dBlock(emb_size,emb_size,(1,2),(1,2),0,norm='none', activation=activ, pad_type=pad_type, transpose=True),  #x2
                            Conv2dBlock(emb_size,emb_size,(1,5),1,(0,2),norm='none', activation=activ, pad_type=pad_type),
                            Conv2dBlock(emb_size,emb_size,(1,5),1,(0,2),norm='adain', activation=activ, pad_type=pad_type),
                            Conv2dBlock(emb_size,emb_size,(1,2),(1,2),0,norm='none', activation=activ, pad_type=pad_type, transpose=True),  #x2
                            Conv2dBlock(emb_size,emb_size,(1,7),1,(0,3),norm='adain', activation=activ, pad_type=pad_type),
                            Conv2dBlock(emb_size,emb_size,(1,7),1,(0,3),norm='adain', activation=activ, pad_type=pad_type),
                            Conv2dBlock(emb_size,emb_size,(1,5),1,(0,2),norm='group', activation=activ, pad_type=pad_type),
                            ]
        self.deep1D = nn.Sequential(*self.deep1D)

        self.conv=[]
        self.conv += [Conv2dBlock(emb_size ,dim, (4,3), (1,1), (0,1), norm='none', activation=activ, pad_type=pad_type, transpose=True)] #4xc
        self.conv += [Conv2dBlock(dim ,dim, (4,3), (2,1), 1, norm=res_norm, activation=activ, pad_type=pad_type, transpose=True)]  #8xc
        self.conv += [Conv2dBlock(dim ,dim, (4,3), (2,1), 1, norm=res_norm, activation=activ, pad_type=pad_type, transpose=True)]  #16xc
        self.conv += [ResBlocks(n_res1, dim, res_norm, activ, pad_type=pad_type, fixed=True)]
        self.conv += [nn.Upsample(scale_factor=2),                                                                 #32x2c1
                        Conv2dBlock(dim, dim // 2, 3, 1, 1, norm='ln', activation=activ, pad_type=pad_type)]
        dim //= 2
        self.conv += [ResBlocks(n_res2, dim, res_norm, activ, pad_type=pad_type, fixed=True)]
        self.conv += [nn.Upsample(scale_factor=2),                                                                 #64x4c
                        Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
        dim //= 2
        self.conv += [ResBlocks(n_res3, dim, res_norm, activ, pad_type=pad_type, fixed=True)]
        self.conv += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]

        self.conv = nn.Sequential(*self.conv)


    def forward(self, content,style=None): #, noise=None):
        #AdaIN has been updated with style already
        #content: batch X char(onehot) X h(1) X width
        #style: batch X channels

        im = self.deep1D(content)
        im = self.conv(im)
        return im

class Deep1DDecoderWithStyle(nn.Module):
    def __init__(self, n_class, n_res1, n_res2, n_res3, dim, output_dim, style_size, res_norm='adain', activ='relu', pad_type='zero',intermediate=False):
        emb_size=min(dim//2,n_class*2)
        space_style_size=32
        super(Deep1DDecoderWithStyle, self).__init__()
        self.deep1D =   [   Conv2dBlock(n_class+space_style_size,emb_size,(1,3),(1,3),0,norm='none', activation=activ, pad_type=pad_type, transpose=True),   #x3
                            Conv2dBlock(emb_size,emb_size,(1,5),1,(0,2),norm='adain', activation=activ, pad_type=pad_type),
                            Conv2dBlock(emb_size,emb_size,(1,2),(1,2),0,norm='none', activation=activ, pad_type=pad_type, transpose=True),  #x2
                            Conv2dBlock(emb_size,emb_size,(1,5),1,(0,2),norm='adain', activation=activ, pad_type=pad_type),
                            Conv2dBlock(emb_size,emb_size,(1,5),1,(0,2),norm='none', activation=activ, pad_type=pad_type),
                            Conv2dBlock(emb_size,emb_size,(1,2),(1,2),0,norm='none', activation=activ, pad_type=pad_type, transpose=True),  #x2
                            Conv2dBlock(emb_size,emb_size,(1,7),1,(0,3),norm='adain', activation=activ, pad_type=pad_type),
                            Conv2dBlock(emb_size,emb_size,(1,7),1,(0,3),norm='adain', activation=activ, pad_type=pad_type),
                            Conv2dBlock(emb_size,emb_size,(1,5),1,(0,2),norm='group', activation=activ, pad_type=pad_type),
                            ]
        self.deep1D = nn.Sequential(*self.deep1D)

        self.conv=[]
        self.conv += [Conv2dBlock(emb_size ,dim, (4,3), (1,1), (0,1), norm='none', activation=activ, pad_type=pad_type, transpose=True)] #4xc
        self.conv += [Conv2dBlock(dim ,dim, (4,3), (2,1), 1, norm=res_norm, activation=activ, pad_type=pad_type, transpose=True)]  #8xc
        self.conv += [Conv2dBlock(dim ,dim, (4,3), (2,1), 1, norm=res_norm, activation=activ, pad_type=pad_type, transpose=True)]  #16xc
        self.conv += [ResBlocks(n_res1, dim, res_norm, activ, pad_type=pad_type, fixed=True)]
        self.conv += [nn.Upsample(scale_factor=2),                                                                 #32x2c1
                        Conv2dBlock(dim, dim // 2, 3, 1, 1, norm='ln', activation=activ, pad_type=pad_type)]
        dim //= 2
        self.conv += [ResBlocks(n_res2, dim, res_norm, activ, pad_type=pad_type, fixed=True)]
        self.conv += [nn.Upsample(scale_factor=2),                                                                 #64x4c
                        Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
        dim //= 2
        self.conv += [ResBlocks(n_res3, dim, res_norm, activ, pad_type=pad_type, fixed=True)]
        self.conv += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]

        self.conv = nn.Sequential(*self.conv)

        mid_size = (style_size+space_style_size)//2
        self.space_proj = nn.Sequential( nn.Linear(style_size,mid_size),nn.ReLU(inplace=True),nn.Linear(mid_size,space_style_size) )

        self.intermediate = intermediate
        if self.intermediate:
            self.spacing =  Conv2dBlock(emb_size,n_class,1,1,0,norm='none', activation='none')

    def forward(self, content,style): #, noise=None):
        #AdaIN has been updated with style already
        #content: batch X char(onehot) X h(1) X width
        #style: batch X channels
        spacing_style = self.space_proj(style)

        im1D = torch.cat( (content, spacing_style.view(spacing_style.size(0),spacing_style.size(1),1,1).expand(-1,-1,content.size(2),content.size(3))), dim=1)
        im1D = self.deep1D(im1D)
        im = self.conv(im1D)
        if self.intermediate:
            spacing = self.spacing(im1D)
            #spacing = F.softmax(spacing.view(spacing.size(0),spacing.size(1),-1),dim=1)
            spacing = spacing.view(spacing.size(0),spacing.size(1),-1)
            return im, spacing
        else:
            return im

class SpacedDecoderWithStyle(nn.Module):
    def __init__(self, n_class, n_res1, n_res2, n_res3, dim, output_dim, style_size, res_norm='adain', activ='relu', pad_type='zero', space_style_size=32):
        super(SpacedDecoderWithStyle, self).__init__()

        self.conv=[]
        self.conv += [Conv2dBlock(n_class+space_style_size ,dim, (4,3), (1,1), (0,1), norm='none', activation=activ, pad_type=pad_type, transpose=True)] #4xc
        self.conv += [Conv2dBlock(dim ,dim, (4,3), (2,1), 1, norm=res_norm, activation=activ, pad_type=pad_type, transpose=True)]  #8xc
        self.conv += [Conv2dBlock(dim ,dim, (4,3), (2,1), 1, norm=res_norm, activation=activ, pad_type=pad_type, transpose=True)]  #16xc
        self.conv += [ResBlocks(n_res1, dim, res_norm, activ, pad_type=pad_type, fixed=True)]
        self.conv += [nn.Upsample(scale_factor=2),                                                                 #32x2c1
                        Conv2dBlock(dim, dim // 2, 3, 1, 1, norm='ln', activation=activ, pad_type=pad_type)]
        dim //= 2
        self.conv += [ResBlocks(n_res2, dim, res_norm, activ, pad_type=pad_type, fixed=True)]
        self.conv += [nn.Upsample(scale_factor=2),                                                                 #64x4c
                        Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
        dim //= 2
        self.conv += [ResBlocks(n_res3, dim, res_norm, activ, pad_type=pad_type, fixed=True)]
        self.conv += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]

        self.conv = nn.Sequential(*self.conv)

        mid_size = (style_size+space_style_size)//2
        self.space_proj = nn.Sequential( nn.Linear(style_size,mid_size),nn.ReLU(inplace=True),nn.Linear(mid_size,space_style_size) )

    def forward(self, content,style): #, noise=None):
        #AdaIN has been updated with style already
        #content: batch X char(onehot) X h(1) X width (this assumes the content is already spaced in the width dimension
        #style: batch X channels
        spacing_style = self.space_proj(style)

        im = torch.cat( (content, spacing_style.view(spacing_style.size(0),spacing_style.size(1),1,1).expand(-1,-1,content.size(2),content.size(3))), dim=1)
        im = self.conv(im)
        return im

class GenWithSkips(nn.Module):
    def __init__(self,dim1D,dim,output_dim,n_res1,n_res2,n_res3,res_norm,activ,pad_type):
        super(GenWithSkips, self).__init__()
        self.inconv=        Conv2dBlock(dim1D+1,dim, 3, 1, 1, norm=res_norm, activation=activ, pad_type=pad_type)
        self.downconv=        Conv2dBlock(dim,2*dim, (4,4), (2,2), 1, norm=res_norm, activation=activ, pad_type=pad_type)
        self.downupconv= nn.Sequential(
                Conv2dBlock(2*dim,4*dim, (4,4), (2,2), 1, norm=res_norm, activation=activ, pad_type=pad_type),
                ResBlocks(n_res1, 4*dim, res_norm, activ, pad_type=pad_type, fixed=True),
                Conv2dBlock(4*dim,2*dim, (4,4), (2,2), 1, norm=res_norm, activation=activ, pad_type=pad_type, transpose=True),
                )
        self.upconv = nn.Sequential(
                ResBlocks(n_res2, 2*dim, res_norm, activ, pad_type=pad_type, fixed=True),
                Conv2dBlock(2*dim,dim, (4,4), (2,2), 1, norm=res_norm, activation=activ, pad_type=pad_type, transpose=True),
                )
        self.finalconv = nn.Sequential(
                ResBlocks(n_res3, dim, res_norm, activ, pad_type=pad_type, fixed=True),
                Conv2dBlock(dim,output_dim, (5,5), (1,1), 2, norm='none', activation='tanh', pad_type=pad_type),
                )
        
    def forward(self,x):
        x=self.inconv(x)
        xd_in = self.downconv(x)
        xd_out = self.downupconv(xd_in)
        diff = xd_in.size(3)-xd_out.size(3)
        if diff>0:
            xd_in = xd_in[:,:,:,diff//2:-(diff//2+diff%2)]
        x_out = self.upconv(xd_in+xd_out)
        diff = x.size(3)-x_out.size(3)
        if diff>0:
            x = x[:,:,:,diff//2:-(diff//2+diff%2)]
        out = self.finalconv(x+x_out)
        return out

class SpacedDecoderWithMask(nn.Module):
    #def __init__(self, n_class, n_resTopA=1, n_resLow=1, n_resLowest=1, n_resTopB=1, dim=256, output_dim=1, style_size=256, res_norm='adain', activ='relu', pad_type='zero', space_style_size=32):
    def __init__(self, n_class, n_res1=2, n_res2=2, n_res3=1, dim=128, output_dim=1, style_size=256, res_norm='adain', activ='relu', pad_type='zero', space_style_size=32, dist_map_text=False,use_skips=False):
        super(SpacedDecoderWithMask, self).__init__()
        dim1D=dim
        self.conv1D=[]
        self.conv1D += [Conv2dBlock(n_class+space_style_size ,dim1D, (1,3), (1,1), (0,1), norm='none', activation=activ, pad_type=pad_type, transpose=True)] #1xc
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

        mid_size = (style_size+space_style_size)//2
        self.space_proj = nn.Sequential( nn.Linear(style_size,mid_size),nn.ReLU(inplace=True),nn.Linear(mid_size,space_style_size) )
        self.dist_map_text=dist_map_text

    def forward(self, mask, text,style): #, noise=None):
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

        spacing_style = self.space_proj(style)


        oneD = torch.cat( (text, spacing_style.view(spacing_style.size(0),spacing_style.size(1),1,1).expand(-1,-1,text.size(2),text.size(3))), dim=1)
        oneD = self.conv1D(oneD)
        oneD = oneD[:,:,:,:mask.size(3)] #clip to same size as input
        if oneD.size(3)<mask.size(3):
            diff = mask.size(3)-oneD.size(3)
            oneD = F.pad(oneD,(diff//2,diff//2+diff%2))
        im = torch.cat((mask,oneD.expand(-1,-1,mask.size(2),-1)),dim=1)
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

class Shallow1DDecoderWithStyle(nn.Module):
    def __init__(self, n_class, n_res1, n_res2, n_res3, dim, output_dim, style_size, res_norm='adain', activ='relu', pad_type='zero', initsize=2, intermediate=False):
        emb_size=dim #min(dim//2,n_class*2)
        space_style_size=32
        super(Shallow1DDecoderWithStyle, self).__init__()
        self.deep1D =   [   Conv2dBlock(n_class+space_style_size,emb_size,(1,initsize),(1,initsize),0,norm='none', activation=activ, pad_type=pad_type, transpose=True),   #xinitsize
                            Conv2dBlock(emb_size,emb_size,(1,5),1,(0,2),norm='adain', activation=activ, pad_type=pad_type),
                            Conv2dBlock(emb_size,emb_size,(1,4),(1,2),(0,1),norm='none', activation=activ, pad_type=pad_type, transpose=True),  #x2
                            Conv2dBlock(emb_size,emb_size,(1,7),1,(0,3),norm='adain', activation=activ, pad_type=pad_type),
                            Conv2dBlock(emb_size,emb_size,(1,7),1,(0,3),norm='adain', activation=activ, pad_type=pad_type),
                            ]
        if initsize>2:
            self.deep1D = (self.deep1D[:-1] + 
                            [ Conv2dBlock(emb_size,emb_size,(1,4),(1,2),(0,1),norm='none', activation=activ, pad_type=pad_type, transpose=True)] +
                            self.deep1D[-1:] )
        self.deep1D = nn.Sequential(*self.deep1D)

        self.conv=[]
        self.conv += [Conv2dBlock(emb_size ,dim, (4,3), (1,1), (0,1), norm='none', activation=activ, pad_type=pad_type, transpose=True)] #4xc
        self.conv += [Conv2dBlock(dim ,dim, (4,3), (2,1), 1, norm=res_norm, activation=activ, pad_type=pad_type, transpose=True)]  #8xc
        self.conv += [Conv2dBlock(dim ,dim, (4,3), (2,1), 1, norm=res_norm, activation=activ, pad_type=pad_type, transpose=True)]  #16xc
        self.conv += [ResBlocks(n_res1, dim, res_norm, activ, pad_type=pad_type, fixed=True)]
        self.conv += [nn.Upsample(scale_factor=2),                                                                 #32x2c1
                        Conv2dBlock(dim, dim // 2, 3, 1, 1, norm='ln', activation=activ, pad_type=pad_type)]
        dim //= 2
        self.conv += [ResBlocks(n_res2, dim, res_norm, activ, pad_type=pad_type, fixed=True)]
        self.conv += [nn.Upsample(scale_factor=2),                                                                 #64x4c
                        Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
        dim //= 2
        self.conv += [ResBlocks(n_res3, dim, res_norm, activ, pad_type=pad_type, fixed=True)]
        self.conv += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]

        self.conv = nn.Sequential(*self.conv)

        mid_size = (style_size+space_style_size)//2
        self.space_proj = nn.Sequential( nn.Linear(style_size,mid_size),nn.ReLU(inplace=True),nn.Linear(mid_size,space_style_size) )

        self.intermediate = intermediate
        if self.intermediate:
            self.spacing =  Conv2dBlock(emb_size,n_class,1,1,0,norm='none', activation='none')

    def forward(self, content,style): #, noise=None):
        #AdaIN has been updated with style already
        #content: batch X char(onehot) X h(1) X width
        #style: batch X channels
        spacing_style = self.space_proj(style)

        im1D = torch.cat( (content, spacing_style.view(spacing_style.size(0),spacing_style.size(1),1,1).expand(-1,-1,content.size(2),content.size(3))), dim=1)
        im1D = self.deep1D(im1D)
        im = self.conv(im1D)
        if self.intermediate:
            spacing = self.spacing(im1D)
            #spacing = F.softmax(spacing.view(spacing.size(0),spacing.size(1),-1),dim=1)
            spacing = spacing.view(spacing.size(0),spacing.size(1),-1)
            return im, spacing
        else:
            return im

def sharpen(input,p,dim=2):
    assert(dim==2)
    p = p.view(-1,1,1)
    powed = input**p
    summed = powed.sum(dim=dim)
    return powed/summed.view(-1,1,1)
    

class SpacingRNN(nn.Module):
    def __init__(self,in_size,out_size):
        super(SpacingRNN, self).__init__()
        self.out_size = out_size
        self.net = nn.LSTMCell(in_size,out_size)
        self.shift_predict = nn.Sequential(
                                nn.Linear(out_size,2),
                                #nn.Sigmoid()
                                )
        self.shift_weights = torch.FloatTensor(1,1,3).zero_()
        self.shift_weights[0,0,0]=1
        self.on_gpu=False

    def forward(self,input,length):
        input = input.permute(1,2,0) #from W,B.C to B,C,W
        batch_size=input.size(0)
        attention_vector = torch.FloatTensor(input.size(0),1,input.size(2)).zero_()
        attention_vector[:,:,0]=1 #start attention on first input char
        attention_vector=attention_vector.to(input.device)
        if not self.on_gpu:
            self.shift_weights=self.shift_weights.to(input.device)
            self.on_gpu=True

        #output = torch.FloatTensor(length,batch_size,self.out_size)
        output = [None]*length
        for t in range(length):
            time_in = (input*attention_vector).sum(dim=2)
            if t ==0:
                output[t],memory = self.net(time_in)
            else:
                output[t],memory = self.net(time_in,(output[t-1],memory))

            if t<length-1:
                shift_p = self.shift_predict(output[t])
                sharpen_factor = F.relu(shift_p[:,1])+1
                shift_p = torch.sigmoid(shift_p[:,0])
                #print('{}/{} : {}'.format(t,length,shift_p[0]))
                attention_vector = (1-shift_p.view(batch_size,1,1))*attention_vector + shift_p.view(batch_size,1,1)*F.conv1d(attention_vector,self.shift_weights,padding=1)
                #attention_vector = F.softmax(attention_vector,dim=2)
                attention_vector = sharpen(attention_vector,sharpen_factor)
            #import pdb;pdb.set_trace()
        return torch.stack(output,dim=0)
class SpacingTrans(nn.Module):
    def __init__(self,in_size,out_size,style_size):
        super(SpacingTrans, self).__init__()
        self.qEmbed = nn.Linear(style_size,in_size)
        self.posEmbed = PositionalEncoding(in_size,0.05,2000)
        heads = min(in_size//32,8)
        self.mhAtt = MultiHeadedAttention(heads,in_size)
        self.final = nn.Sequential( nn.Linear(in_size,out_size), 
                                    nn.GroupNorm(getGroupSize(out_size),out_size),
                                    nn.ReLU(True)
                                    )

    def forward(self,input,style,length):
        batch_size = input.size(1)
        queries = style.view(batch_size,1,style.size(1)).expand(-1,length,-1)
        queries = self.qEmbed(queries.contiguous().view(batch_size*length,style.size(1))).view(batch_size,length,-1)
        queries = self.posEmbed(queries)
        input = input.permute(1,0,2)
        input = self.posEmbed(input)
        result = self.mhAtt(queries,input,input)

        result = self.final(result.view(batch_size*length,result.size(2))).view(batch_size,length,-1)
        return result.permute(1,0,2)



class RNNDecoder(nn.Module):
    def __init__(self, n_class, n_res1, n_res2, n_res3, dim, output_dim, style_size, res_norm='adain', activ='relu', pad_type='zero', intermediate=False, space_style_size=32):
        super(RNNDecoder, self).__init__()
        self.emb_size=dim #min(dim//2,n_class*2)
        self.embed1D =  Conv2dBlock(n_class+space_style_size,self.emb_size,(1,1),(1,1),0,norm='none', activation=activ, pad_type=pad_type, transpose=True)
        self.rnn1 = SpacingRNN(self.emb_size,self.emb_size)
        self.rnn2 = nn.LSTM(self.emb_size,self.emb_size, bidirectional=True, dropout=0, num_layers=1)

        self.conv=[]
        self.conv += [Conv2dBlock(self.emb_size ,dim, (4,3), (1,1), (0,1), norm='none', activation=activ, pad_type=pad_type, transpose=True)] #4xc
        self.conv += [Conv2dBlock(dim ,dim, (4,3), (2,1), 1, norm=res_norm, activation=activ, pad_type=pad_type, transpose=True)]  #8xc
        self.conv += [Conv2dBlock(dim ,dim, (4,3), (2,1), 1, norm=res_norm, activation=activ, pad_type=pad_type, transpose=True)]  #16xc
        self.conv += [ResBlocks(n_res1, dim, res_norm, activ, pad_type=pad_type, fixed=True)]
        self.conv += [nn.Upsample(scale_factor=2),                                                                 #32x2c1
                        Conv2dBlock(dim, dim // 2, 3, 1, 1, norm='ln', activation=activ, pad_type=pad_type)]
        dim //= 2
        self.conv += [ResBlocks(n_res2, dim, res_norm, activ, pad_type=pad_type, fixed=True)]
        self.conv += [nn.Upsample(scale_factor=2),                                                                 #64x4c
                        Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
        dim //= 2
        self.conv += [ResBlocks(n_res3, dim, res_norm, activ, pad_type=pad_type, fixed=True)]
        self.conv += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]

        self.conv = nn.Sequential(*self.conv)

        mid_size = (style_size+space_style_size)//2
        self.space_proj = nn.Sequential( nn.Linear(style_size,mid_size),nn.ReLU(inplace=True),nn.Linear(mid_size,space_style_size) )

        self.intermediate = intermediate
        if self.intermediate:
            self.spacing =  Conv2dBlock(self.emb_size,n_class,1,1,0,norm='none', activation='none')

    def forward(self, content,style): #, noise=None):
        #AdaIN has been updated with style already
        #content: batch X char(onehot) X h(1) X width
        #style: batch X channels
        batch_size=content.size(0)
        spacing_style = self.space_proj(style)

        data1D = torch.cat( (content, spacing_style.view(batch_size,spacing_style.size(1),1,1).expand(-1,-1,content.size(2),content.size(3))), dim=1)
        data1D = self.embed1D(data1D)
        dataRNN = data1D.view(batch_size,self.emb_size,-1).permute(2,0,1)
        length = 12*dataRNN.size(0)
        dataRNN = self.rnn1(dataRNN,length)
        dataRNN,_ = self.rnn2(dataRNN)
        data1D = dataRNN.permute(1,2,0).contiguous().view(batch_size,self.emb_size,1,-1)
        im = self.conv(data1D)
        if self.intermediate:
            spacing = self.spacing(data1D)
            #spacing = F.softmax(spacing.view(spacing.size(0),spacing.size(1),-1),dim=1)
            spacing = spacing.view(spacing.size(0),spacing.size(1),-1)
            return im, spacing
        else:
            return im

class NewRNNDecoder(nn.Module):
    def __init__(self, n_class, n_res1, n_res2, n_res3, dim, output_dim, style_size, res_norm='adain', activ='relu', pad_type='zero', intermediate=True, space_style_size=32):
        super(NewRNNDecoder, self).__init__()
        self.emb_size=dim #min(dim//2,n_class*2)
        self.embed1D =  Conv2dBlock(n_class+space_style_size,self.emb_size,(1,1),(1,1),0,norm='none', activation='none', pad_type=pad_type, transpose=True)
        self.trans = SpacingTrans(self.emb_size,self.emb_size,space_style_size)
        self.rnn2 = nn.LSTM(self.emb_size,self.emb_size, bidirectional=True, dropout=0, num_layers=1)

        dim1D=dim
        self.conv1D=[]
        #self.conv1D += [Conv2dBlock(n_class+space_style_size ,dim1D, (1,3), (1,1), (0,1), norm='none', activation=activ, pad_type=pad_type, transpose=True)] #1xc
        #self.conv1D += [Conv2dBlock(self.emb_size ,dim1D, (1,3), (1,1), (0,1), norm=res_norm, activation=activ, pad_type=pad_type, transpose=True)]  #1xc
        #self.conv1D += [ResBlocks(n_res1, dim1D, res_norm, activ, pad_type=pad_type, fixed=True)]
        #self.conv1D += [nn.Upsample(scale_factor=2),                                                                 #1x2c1
        #                Conv2dBlock(dim1D, dim1D // 2, 3, 1, 1, norm='ln', activation=activ, pad_type=pad_type)]
        self.conv1D += [Conv2dBlock(self.emb_size ,dim1D, (1,4), (1,2), (0,1), norm=res_norm, activation=activ, pad_type=pad_type, transpose=True)]  #16xc
        #self.conv1D += [ResBlocks(n_res2, dim1D, res_norm, activ, pad_type=pad_type, fixed=True)]
        #self.conv1D += [nn.Upsample(scale_factor=2),                                                                 #64x4c
        #                Conv2dBlock(dim1D, dim1D // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
        self.conv1D += [Conv2dBlock(dim1D ,dim1D, (1,4), (1,2), (0,1), norm=res_norm, activation=activ, pad_type=pad_type, transpose=True)]  #16xc
        #dim1D //= 2
        #self.conv1D += [ResBlocks(n_res3, dim1D, res_norm, activ, pad_type=pad_type, fixed=True)]
        self.conv1D += [Conv2dBlock(dim1D, dim1D, (1,3), 1, (0,1), norm='none', activation=activ, pad_type=pad_type)]

        self.conv1D = nn.Sequential(*self.conv1D)

        self.conv2D = conv2DTopA = nn.Sequential(
                Conv2dBlock(dim1D+1,dim, 3, 1, 1, norm=res_norm, activation=activ, pad_type=pad_type),
                Conv2dBlock(dim,2*dim, (4,4), (2,2), 1, norm=res_norm, activation=activ, pad_type=pad_type),
                Conv2dBlock(2*dim,2*dim, (4,4), (2,2), 1, norm=res_norm, activation=activ, pad_type=pad_type),
                ResBlocks(n_res1, 2*dim, res_norm, activ, pad_type=pad_type, fixed=True),
                Conv2dBlock(2*dim,2*dim, (4,4), (2,2), 1, norm=res_norm, activation=activ, pad_type=pad_type, transpose=True),
                ResBlocks(n_res2, 2*dim, res_norm, activ, pad_type=pad_type, fixed=True),
                Conv2dBlock(2*dim,dim, (4,4), (2,2), 1, norm=res_norm, activation=activ, pad_type=pad_type, transpose=True),
                ResBlocks(n_res3, dim, res_norm, activ, pad_type=pad_type, fixed=True),
                )

        self.final = Conv2dBlock(dim,output_dim, (5,5), (1,1), 2, norm='none', activation='tanh', pad_type=pad_type)


        mid_size = (style_size+space_style_size)//2
        self.space_proj = nn.Sequential( nn.Linear(style_size,mid_size),nn.ReLU(inplace=True),nn.Linear(mid_size,space_style_size) )

        self.intermediate = intermediate
        if self.intermediate:
            self.spacing =  Conv2dBlock(self.emb_size,n_class,1,1,0,norm='none', activation='logsoftmax')
            self.mask =  Conv2dBlock(dim,1,3,1,1,norm='none', activation='tanh')


    def forward(self, distance_map,content,style,return_intermediate=False): #, noise=None):
        #AdaIN has been updated with style already
        #content: batch X char(onehot) X h(1) X width
        #style: batch X channels

        length = distance_map.size(3)//8

        batch_size=content.size(0)
        spacing_style = self.space_proj(style)

        data1D = torch.cat( (content, spacing_style.view(batch_size,spacing_style.size(1),1,1).expand(-1,-1,content.size(2),content.size(3))), dim=1)
        data1D = self.embed1D(data1D)
        dataRNN = data1D.view(batch_size,self.emb_size,-1).permute(2,0,1)
        dataRNN = self.trans(dataRNN,spacing_style,length)
        dataRNN,_ = self.rnn2(dataRNN)
        data1D = dataRNN.permute(1,2,0).contiguous().view(batch_size,self.emb_size,1,-1)
        
        updata1D = self.conv1D(data1D)
        updata1D = updata1D[:,:,:,:distance_map.size(3)] #clip to same size as input
        if updata1D.size(3)<distance_map.size(3):
            diff = distance_map.size(3)-updata1D.size(3)
            #updata1D = F.pad(updata1D,(diff//2,diff//2+diff%2))
            updata1D = F.pad(updata1D,(0,diff))
        data2D = torch.cat((distance_map,updata1D.expand(-1,-1,distance_map.size(2),-1)),dim=1)
        data2D = self.conv2D(data2D)
        im = self.final(data2D)
        #data2D = self.convUp(data1D)
        #data2D = torch.cat((data2D,distance_map),dim=1)
        #im = self.conv(data2D)
        if self.intermediate and return_intermediate:
            spacing = self.spacing(data1D)
            spacing = F.softmax(spacing,dim=1).view(spacing.size(0),spacing.size(1),-1).permute(2,0,1)
            #spacing = F.softmax(spacing.view(spacing.size(0),spacing.size(1),-1),dim=1)
            #spacing = spacing.view(spacing.size(0),spacing.size(1),-1)
            
            mask = self.mask(data2D)
            return im, spacing, mask
        else:
            return im

def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2*m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]

def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2*m.num_features
    return num_adain_params


