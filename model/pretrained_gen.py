from torch import nn
import torch
import torch.nn.functional as F
from datasets.hw_dataset import PADDING_CONSTANT
from .net_builder import getGroupSize
from .MUNIT_networks import Conv2dBlock, GenWithSkips,ResBlocks
from .autoencoder import DecoderNoSkip

class Print(nn.Module):
    def __init__(self):
        super(Print,self).__init__()
    def forward(self,x):
        print(x.size())
        return x

class PretrainedGen(nn.Module):
    #def __init__(self, n_class, n_resTopA=1, n_resLow=1, n_resLowest=1, n_resTopB=1, dim=256, output_dim=1, style_size=256, res_norm='adain', activ='relu', pad_type='zero', space_style_size=32):
    def __init__(self, n_class, n_res1=1, n_res2=1, n_res3=1, dim=32, output_dim=1, style_size=256, res_norm='adain', activ='relu', pad_type='zero', space_style_size=32,  noise=False, decoder_type='no skip', decoder_weights=None):
        super(PretrainedGen, self).__init__()
        self.char_spec = False
        if decoder_type == 'no skip':
            self.convUp = DecoderNoSkip()
            up_in = 512
            up_dim = 32
        elif decoder_type == '2':
            self.convUp = DecoderNoSkip(256)
            up_in = 256
            up_dim = 32
        elif decoder_type == '2tight':
            self.convUp = DecoderNoSkip(32)
            up_in = 32
            up_dim = 32
        else:
            raise NotImplementedError('Unknown decoder type: {}'.format(decoder_type))
        if decoder_weights is not None:
            self.convUp.load_state_dict(decoder_weights)
        self.convUp = nn.Sequential(*list(self.convUp.up_conv1.children())[:-2]) #remove Tanh and final conv layer
        #self.convUp = nn.ConvTranspose2d(up_in,up_dim,64)

        mask_enc_size=64
        self.encode_mask = nn.Sequential(
                        Conv2dBlock(1 ,32, (5,5), (1,1), 2, norm='group', activation=activ, pad_type=pad_type),
                        nn.MaxPool2d(2), #32
                        Conv2dBlock(32 ,32, (3,3), (1,1), 1, norm='group', activation=activ, pad_type=pad_type),
                        Conv2dBlock(32 ,32, (3,3), (1,1), 1, norm='group', activation=activ, pad_type=pad_type),
                        nn.MaxPool2d(2), #16
                        Conv2dBlock(32 ,32, (3,3), (1,1), 1, norm='group', activation=activ, pad_type=pad_type),
                        Conv2dBlock(32 ,32, (3,3), (1,1), 1, norm='group', activation=activ, pad_type=pad_type),
                        #nn.MaxPool2d(2), #8
                        Conv2dBlock(32 ,64, (4,3), (2,1), (1,1,1,1), norm='group', activation=activ, pad_type=pad_type), #6
                        Conv2dBlock(64 ,64, (3,3), (1,1), (1,1,0,0), norm='group', activation=activ, pad_type=pad_type), #4
                        Conv2dBlock(64 ,64, (3,3), (1,1), (1,1,0,0), norm='group', activation=activ, pad_type=pad_type), #4
                        Conv2dBlock(64 ,mask_enc_size, (4,3), (1,1), (1,1,0,0), norm='group', activation=activ, pad_type=pad_type),
                        )

        dim1D=up_in
        input_size = n_class+space_style_size+mask_enc_size
        self.conv1D=[]
        self.conv1D += [Conv2dBlock(input_size ,dim1D, (1,3), (1,1), (1,1,0,0), norm='none', activation=activ, pad_type=pad_type)] #1xc
        self.conv1D += [Conv2dBlock(dim1D ,dim1D, (1,4), (1,2), (1,0,0,0), norm='none', activation=activ, pad_type=pad_type)] #1xc
        self.conv1D += [Conv2dBlock(dim1D ,up_in, (1,3), (1,1), 0, norm=res_norm, activation=activ, pad_type=pad_type)]  #1xc
        self.conv1D = nn.Sequential(*self.conv1D)



        self.convFine = GenWithSkips(up_dim,dim,output_dim,n_res1,n_res2,n_res3,res_norm,activ,pad_type)


        if not self.char_spec:
            mid_size = (style_size+space_style_size)//2
            self.space_proj = nn.Sequential( nn.Linear(style_size,mid_size),nn.ReLU(inplace=True),nn.Linear(mid_size,space_style_size) )

        if noise:
            self.noise_scale = nn.Parameter(torch.FloatTensor(dim1D).fill_(0.5))

    def forward(self, mask, text,style, noise=False):
        #AdaIN has been updated with style already
        #content: batch X char(onehot) X h(1) X width (this assumes the content is already spaced in the width dimension
        #style: batch X channels

        if self.char_spec:
            g_style,spaced_style,char_style = style
            spacing_style = spaced_style.permute(1,2,0)
            spacing_style = spacing_style[:,:,None,:] #add height
        else:
            spacing_style = self.space_proj(style)
            spacing_style = spacing_style.view(spacing_style.size(0),spacing_style.size(1),1,1).expand(-1,-1,text.size(2),text.size(3))

        mask_1d = self.encode_mask(mask)

        if mask_1d.size(3)>text.size(3):
            diff = mask_1d.size(3)-text.size(3)
            mask_1d = mask_1d[:,:,:,diff//2:-(diff//2 + diff%2)]
        elif mask_1d.size(3)<text.size(3):
            diff = text.size(3)-mask_1d.size(3)
            mask_1d = F.pad(mask_1d,(diff//2,diff//2+diff%2))
        im = torch.cat( (text, spacing_style,mask_1d), dim=1)
        im = self.conv1D(im)
        im = self.convUp(im)
        #print('gen up size:{}, mask size:{}'.format(im.size(),mask.size()))
        if im.size(3)<mask.size(3):
            diff = mask.size(3)-im.size(3)
            im = F.pad(im,(diff//2,diff//2+diff%2))
        elif im.size(3)>mask.size(3):
            diff = im.size(3)- mask.size(3)
            im = im[:,:,:,diff//2:-(diff//2+diff%2)]
        if noise:
            noise = torch.randn_like(im) * self.noise_scale[None,:,None,None]
            im = im+noise
        im = torch.cat((mask,im),dim=1)
        spacing_style=None
        
        im = self.convFine(im)
        return im
