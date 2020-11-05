"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
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

##################################################################################
# Discriminator
##################################################################################

class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, params):
        super(MsImageDis, self).__init__()
        self.n_layer = params['n_layer']
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.activ = params['activ']
        self.num_scales = params['num_scales']
        self.pad_type = params['pad_type']
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

##################################################################################
# Generator
##################################################################################

class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, params):
        super(AdaINGen, self).__init__()
        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']

        # style encoder
        self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, input_dim, res_norm='adain', activ=activ, pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

    def forward(self, images):
        # reconstruct an image
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode(self, images):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        return content, style_fake

    def decode(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params


class VAEGen(nn.Module):
    # VAE architecture
    def __init__(self, input_dim, params):
        super(VAEGen, self).__init__()
        dim = params['dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']

        # content encoder
        self.enc = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc.output_dim, input_dim, res_norm='in', activ=activ, pad_type=pad_type)

    def forward(self, images):
        # This is a reduced VAE implementation where we assume the outputs are multivariate Gaussian distribution with mean = hiddens and std_dev = all ones.
        hiddens = self.encode(images)
        if self.training == True:
            noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
            images_recon = self.decode(hiddens + noise)
        else:
            images_recon = self.decode(hiddens)
        return images_recon, hiddens

    def encode(self, images):
        hiddens = self.enc(images)
        noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
        return hiddens, noise

    def decode(self, hiddens):
        images = self.dec(hiddens)
        return images


##################################################################################
# Encoder and Decoders
##################################################################################

class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
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

class StyleEncoderHW(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoderHW, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
            self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)[:,:,0,0]
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

class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

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
        xd_in=None
        xd_out=None
        diff = x.size(3)-x_out.size(3)
        if diff>0:
            x = x[:,:,:,diff//2:-(diff//2+diff%2)]
        x = self.finalconv(x+x_out)
        return x

class SpacedDecoderWithMask(nn.Module):
    #def __init__(self, n_class, n_resTopA=1, n_resLow=1, n_resLowest=1, n_resTopB=1, dim=256, output_dim=1, style_size=256, res_norm='adain', activ='relu', pad_type='zero', space_style_size=32):
    def __init__(self, n_class, n_res1=2, n_res2=2, n_res3=1, dim=128, output_dim=1, style_size=256, res_norm='adain', activ='relu', pad_type='zero', space_style_size=32, dist_map_text=False,use_skips=False, noise=False,extra_text=False):
        super(SpacedDecoderWithMask, self).__init__()
        self.extra_text = extra_text
        dim1D=max(dim,int(n_class*1.25))
        input_size = n_class+space_style_size
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

        mid_size = (style_size+space_style_size)//2
        self.space_proj = nn.Sequential( nn.Linear(style_size,mid_size),nn.ReLU(inplace=True),nn.Linear(mid_size,space_style_size) )
        self.dist_map_text=dist_map_text

        if noise:
            self.noise_scale = nn.Parameter(torch.FloatTensor(dim1D).fill_(0.5))

    def forward(self, mask, text,style, noise=False):
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
        spacing_style=None
        
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
##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero', fixed=False):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, fixed= (2 if i>0 else 1) if fixed else False)]
        if fixed:
            #add norm+activation at the end
            # initialize normalization
            norm_dim = dim
            if norm == 'bn':
                self.model.append(nn.BatchNorm2d(norm_dim))
            elif norm == 'in':
                #self.model.append(nn.InstanceNorm2d(norm_dim, track_running_stats=True)
                self.model.append(nn.InstanceNorm2d(norm_dim))
            elif norm == 'ln':
                self.model.append(LayerNorm(norm_dim))
            elif norm == 'adain':
                self.model.append(AdaptiveInstanceNorm2d(norm_dim))
            elif norm == 'none' or norm == 'sn':
                pass
            elif norm == 'group':
                self.model.append(nn.GroupNorm(getGroupSize(norm_dim),norm_dim))
            else:
                assert 0, "Unsupported normalization: {}".format(norm)

            # initialize activation
            if activation == 'relu':
                self.model.append(nn.ReLU(inplace=True))
            elif activation == 'lrelu':
                self.model.append(nn.LeakyReLU(0.2, inplace=True))
            elif activation == 'prelu':
                self.model.append(nn.PReLU())
            elif activation == 'selu':
                self.model.append(nn.SELU(inplace=True))
            elif activation == 'tanh':
                self.model.append(nn.Tanh())
            elif activation == 'logsoftmax':
                self.model.append(nn.LogSoftmax(dim=1))
            elif activation == 'none':
                pass
            else:
                assert 0, "Unsupported activation: {}".format(activation)

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero', fixed=False):
        super(ResBlock, self).__init__()

        model = []
        if not fixed or fixed==1:
            model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm if not fixed else 'none', activation='none', pad_type=pad_type)]
        else:
            model += [  Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type, reverse=True),
                        Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type, reverse=True)
                        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', transpose=False, reverse=False):
        super(Conv2dBlock, self).__init__()
        self.reverse=reverse
        self.use_bias = True
        # initialize padding
        if transpose:
            self.pad = lambda x: x
        elif pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        elif norm == 'group':
            self.norm = nn.GroupNorm(getGroupSize(norm_dim),norm_dim)
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'logsoftmax':
            self.activation = nn.LogSoftmax(dim=1)
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias, padding=padding)
            if norm == 'sn':
                raise NotImplemented('easy to do')
        elif norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        if not self.reverse:
            x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        if self.reverse:
            x = self.conv(self.pad(x))
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

##################################################################################
# VGG network definition
##################################################################################
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        # relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        # relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        # relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)
        # relu4_3 = h

        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        relu5_3 = h

        return relu5_3
        # return [relu1_2, relu2_2, relu3_3, relu4_3]

##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

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

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
