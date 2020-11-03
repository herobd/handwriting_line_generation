# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
from base import BaseModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import cv2
from model.cnn_lstm import CRNN
from model.cnn_only_hwr import CNNOnlyHWR
from model.style_hwr import AdaINStyleHWR, HyperStyleHWR, DeformStyleHWR
from model.deform_hwr import DeformHWR
from model.MUNIT_networks import StyleEncoderHW, HWDecoder, MLP, Deep1DDecoder, Deep1DDecoderWithStyle, Shallow1DDecoderWithStyle, SpacedDecoderWithStyle, NewRNNDecoder, SpacedDecoderWithMask
from model.MUNIT_networks import get_num_adain_params, assign_adain_params
from model.style import NewHWStyleEncoder
from model.lookup_style import LookupStyle
from model.discriminator import Discriminator, DownDiscriminator, TwoScaleDiscriminator, TwoScaleBetterDiscriminator
from model.mask_rnn import CountRNN, CreateMaskRNN, TopAndBottomDiscriminator
from model.discriminator import SpectralNorm
from skimage import draw
from scipy.ndimage.morphology import distance_transform_edt


class AdaINGen(nn.Module):
    def __init__(self,n_class,style_dim,n_res1=2,n_res2=1,n_res3=0, dim=256, output_dim=1, style_transform_dim=None,activ='lrelu', type="HW", space_style_size=32,dist_map_text=False,use_skips=False):
        super(AdaINGen, self).__init__()
        self.pad=True
        if type=='HW':
            self.gen = HWDecoder(n_class, n_res1, n_res2, n_res3, dim, output_dim, res_norm='adain', activ=activ, pad_type='zero')
        elif type =='Deep1D':
            self.gen = Deep1DDecoder(n_class, n_res1, n_res2, n_res3, dim, output_dim, res_norm='adain', activ=activ, pad_type='zero')
        elif type.startswith('Deep1DWithStyle'):
            intermediate='Space' in type
            self.gen = Deep1DDecoderWithStyle(n_class, n_res1, n_res2, n_res3, dim, output_dim, style_dim, res_norm='adain', activ=activ, pad_type='zero', intermediate=intermediate)
        elif type == 'SpacedWithStyle':
            self.gen = SpacedDecoderWithStyle(n_class, n_res1, n_res2, n_res3, dim, output_dim, style_dim, res_norm='adain', activ=activ, pad_type='zero', space_style_size=space_style_size)
        elif type == 'SpacedWithMask':
            #self.gen = SpacedDecoderWithMask(n_class, n_res1, n_res2, n_res3, n_res1, dim, output_dim, style_dim, res_norm='adain', activ=activ, pad_type='zero', space_style_size=space_style_size)
            self.gen = SpacedDecoderWithMask(n_class, n_res1, n_res2, n_res3, dim, output_dim, style_dim, res_norm='adain', activ=activ, pad_type='zero', space_style_size=space_style_size,dist_map_text=dist_map_text,use_skips=use_skips)
            self.pad=False
        elif type[:18] =='Shallow1DWithStyle':
            if 'Big' in type:
                initsize=3
            else:
                initsize=2
            intermediate='Space' in type
            self.gen = Shallow1DDecoderWithStyle(n_class, n_res1, n_res2, n_res3, dim, output_dim, style_dim, res_norm='adain', activ=activ, pad_type='zero', initsize=initsize, intermediate=intermediate)
        elif 'RNN' in type:
            intermediate=True
            self.gen = NewRNNDecoder(n_class, n_res1, n_res2, n_res3, dim, output_dim, style_dim, res_norm='adain', activ=activ, pad_type='zero', intermediate=intermediate)
        else:
            print('unknown generator: '+type)
            raise NotImplementedError('unknown generator: '+type)
        num_adain_params = get_num_adain_params(self.gen)
        if style_transform_dim is None:
            style_transform_dim=(2*style_dim+num_adain_params)//3
        self.style_transform = MLP(style_dim, num_adain_params, style_transform_dim, 3, norm='none', activ=activ)

    def forward(self, chars, style, mask=None, return_intermediate=None):
        adain_params = self.style_transform(style)
        assign_adain_params(adain_params, self.gen)
        content = chars.permute(1,2,0) #swap [T,b,cls] to [b,cls,T]
        if self.pad:
            content = F.pad(content,(1,1),value=0)
        content = content.view(content.size(0),content.size(1),1,content.size(2)) #now [b,cls,H,W]
        if mask is None:
            return self.gen(content,style)
        elif type(self.gen) is NewRNNDecoder:
            return self.gen(mask,content,style,return_intermediate=return_intermediate)
        else:
            return self.gen(mask,content,style)

    #def assign_adain_params(self, adain_params, model):
    #    # assign the adain_params to the AdaIN layers in model
    #    for m in model.modules():
    #        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
    #            mean = adain_params[:, :m.num_features]
    #            std = adain_params[:, m.num_features:2*m.num_features]
    #            m.bias = mean.contiguous().view(-1)
    #            m.weight = std.contiguous().view(-1)
    #            if adain_params.size(1) > 2*m.num_features:
    #                adain_params = adain_params[:, 2*m.num_features:]

    #def get_num_adain_params(self, model):
    #    # return the number of AdaIN parameters needed by the model
    #    num_adain_params = 0
    #    for m in model.modules():
    #        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
    #            num_adain_params += 2*m.num_features
    #    return num_adain_params

class HWWithStyle(BaseModel):
    def __init__(self, config):
        super(HWWithStyle, self).__init__(config)

        n_downsample = config['style_n_downsample'] if 'style_n_downsample' in config else 3
        input_dim = 1
        self.count_std = 0.1
        self.dup_std = 0.03
        self.image_height=64
        dim = config['style_dim']//4 if 'style_dim' in config else 64
        style_dim = config['style_dim'] if 'style_dim' in config else 256
        self.style_dim = style_dim
        norm = config['style_norm'] if 'style_norm' in config else 'none'
        activ = config['style_activ'] if 'style_activ' in config else 'lrelu'
        pad_type = config['pad_type'] if 'pad_type' in config else 'replicate'

        num_class = config['num_class']
        self.num_class=num_class

        style_type = config['style'] if 'style' in config else 'normal'

        if 'new' in style_type:
            num_keys = config['num_keys'] if 'num_keys' in config else 16
            frozen_keys = config['frozen_keys'] if 'frozen_keys' in config else False
            global_pool = config['global_pool'] if 'global_pool' in config else False
            self.style_extractor = NewHWStyleEncoder(input_dim, dim, style_dim, norm, activ, pad_type, num_class, num_keys=num_keys, frozen_keys=frozen_keys,global_pool=global_pool)
        elif 'ookup' in style_type:
            self.style_extractor = LookupStyle(style_dim)
        elif style_type != 'none':
            self.style_extractor = StyleEncoderHW(n_downsample, input_dim, dim, style_dim, norm, activ, pad_type)
        else:
            self.style_extractor = None
        if 'pretrained_style' in config and config['pretrained_style'] is not None:
            snapshot = torch.load(config['pretrained_style'], map_location='cpu')
            style_state_dict={}
            for key,value in  snapshot['state_dict'].items():
                if key.startswith('style_extractor.'):
                    style_state_dict[key[16:]] = value
            self.style_extractor.load_state_dict( style_state_dict )
            if 'style_frozen' in config and config['style_frozen']:
                self.style_frozen=True
                for param in self.style_extractor.parameters():
                    param.will_use_grad=param.requires_grad
                    param.requires_grad=False

        hwr_type= config['hwr'] if 'hwr' in config else 'CRNN'
        if 'CRNN' in hwr_type:
            if 'group' in hwr_type:
                norm='group'
            else:
                norm='batch'
            use_softmax = 'softmax' in hwr_type
            self.hwr=CRNN(num_class,norm=norm, use_softmax=use_softmax)
        elif 'CNNOnly' in hwr_type:
            self.hwr=CNNOnlyHWR(num_class)
        elif 'AdaINStyle' in hwr_type:
            self.hwr=AdaINStyleHWR(num_class, style_dim=style_dim)
        elif 'HyperStyle' in hwr_type:
            self.hwr=HyperStyleHWR(num_class, style_dim=style_dim)
        elif 'tyle' in hwr_type:
            useED = 'ED' in hwr_type
            deformConv = 'eform' in hwr_type
            appendStyle = 'ppend' in hwr_type
            addStyle = 'add' in hwr_type or 'Add' in hwr_type
            transDeep = 'eep' in hwr_type
            rnnStyle = 'RNN' in hwr_type
            if 'deformConv' in config:
                deformConv = config['deformConv']
            
            if 'xperts' in hwr_type or 'num_experts' in config:
                numExperts = config['num_experts']
            else:
                numExperts=0
            self.hwr=DeformStyleHWR(num_class, style_dim=style_dim, useED=useED, deformConv=deformConv, appendStyle=appendStyle, transDeep=transDeep,numExperts=numExperts,rnnStyle=rnnStyle, addStyle=addStyle)
        elif 'Deform' in hwr_type:
            useED = 'ED' in hwr_type
            self.hwr=DeformHWR(num_class,useED=useED)
        else:
            raise NotImplementedError('unknown HWR model'+hwr_type)
        self.hwr_frozen=False
        if 'pretrained_hwr' in config and config['pretrained_hwr'] is not None:
            snapshot = torch.load(config['pretrained_hwr'], map_location='cpu')
            hwr_state_dict={}
            for key,value in  snapshot['state_dict'].items():
                if key[:4]=='hwr.':
                    hwr_state_dict[key[4:]] = value
            self.hwr.load_state_dict( hwr_state_dict )
            #if 'hwr_frozen' in config and config['hwr_frozen']:
            #    self.hwr_frozen=True
            #    for param in self.hwr.parameters():
            #        param.will_use_grad=param.requires_grad
            #        param.requires_grad=False

        if 'generator' in config and config['generator'] == 'none':
            self.generator = None
        else:
            generator_type = config['generator'] if 'generator' in config else 'HW'
            n_res1 = config['gen_n_res1'] if 'gen_n_res1' in config else 2
            n_res2 = config['gen_n_res2'] if 'gen_n_res2' in config else 1
            n_res3 = config['gen_n_res3'] if 'gen_n_res3' in config else 0
            g_dim = config['gen_dim'] if 'gen_dim' in config else 256
            space_style_size=config['gen_space_style_size'] if 'gen_space_style_size' in config else 32
            dist_map_text = config['dist_map_text_for_gen'] if 'dist_map_text_for_gen' in config else False
            use_skips = config['gen_use_skips'] if 'gen_use_skips' in config else False
            self.generator = AdaINGen(num_class,style_dim, type=generator_type, n_res1=n_res1,n_res2=n_res2,n_res3=n_res3, dim=g_dim, space_style_size=space_style_size,dist_map_text=dist_map_text,use_skips=use_skips)
        if 'pretrained_generator' in config and config['pretrained_generator'] is not None:
            snapshot = torch.load(config['pretrained_generator'], map_location='cpu')
            gen_state_dict={}
            for key,value in  snapshot['state_dict'].items():
                if key[:10]=='generator.':
                    gen_state_dict[key[10:]] = value
            self.generator.load_state_dict( gen_state_dict )

        if 'discriminator' in config:
            if config['discriminator']=='down':
                self.discriminator = DownDiscriminator()
            elif 'two' in config['discriminator'] and 'better' in config['discriminator']:
                more_low = 'more' in config['discriminator'] and 'low' in config['discriminator']
                global_pool = 'global' in config['discriminator']
                dim = 32 if 'half' in config['discriminator'] else 64
                self.discriminator = TwoScaleBetterDiscriminator(more_low=more_low,dim=dim,global_pool=global_pool)
            elif 'two' in config['discriminator']:
                self.discriminator = TwoScaleDiscriminator()
            else:
                self.discriminator = Discriminator()
        if 'pretrained_discriminator' in config and config['pretrained_discriminator'] is not None:
            snapshot = torch.load(config['pretrained_discriminator'], map_location='cpu')
            discriminator_state_dict={}
            for key,value in  snapshot['state_dict'].items():
                if key[:14]=='discriminator.':
                    discriminator_state_dict[key[14:]] = value
            self.discriminator.load_state_dict( discriminator_state_dict )

        if 'spacer' in config and config['spacer']:
            spacer_dim = config['spacer_dim'] if 'spacer_dim' in config else 128
            self.count_duplicates =  type(config['spacer']) is str and 'duplicate' in config['spacer']
            num_out = 2 if self.count_duplicates else 1
            self.spacer=CountRNN(num_class,style_dim,spacer_dim,num_out)
        else:
            self.spacer=None
        if 'pretrained_spacer' in config and config['pretrained_spacer'] is not None:
            snapshot = torch.load(config['pretrained_spacer'], map_location='cpu')
            spacer_state_dict={}
            for key,value in  snapshot['state_dict'].items():
                if key[:7]=='spacer.':
                    spacer_state_dict[key[7:]] = value
            self.spacer.load_state_dict( spacer_state_dict )


        if 'create_mask' in config and config['create_mask']:
            create_mask_dim = config['create_mask_dim'] if 'create_mask_dim' in config else 128
            self.create_mask=CreateMaskRNN(num_class,style_dim,create_mask_dim)
        else:
            self.create_mask=None
        if 'pretrained_create_mask' in config and config['pretrained_create_mask'] is not None:
            snapshot = torch.load(config['pretrained_create_mask'], map_location='cpu')
            create_mask_state_dict={}
            for key,value in  snapshot['state_dict'].items():
                if key[:12]=='create_mask.':
                    create_mask_state_dict[key[12:]] = value
            self.create_mask.load_state_dict( create_mask_state_dict )

        if self.spacer is not None and self.create_mask is not None or ('style_from_normal' in config and config['style_from_normal']):
            self.style_from_normal = nn.Sequential(
                    nn.Linear(style_dim//2,style_dim),
                    nn.ReLU(True),
                    nn.Linear(style_dim,style_dim),
                    nn.ReLU(True),
                    nn.Linear(style_dim,style_dim)
                    )

        if 'guide_hwr' in config:
            snapshot = torch.load(config['guide_hwr'], map_location='cpu')
            hwr_state_dict={}
            for key,value in  snapshot['state_dict'].items():
                if key[:4]=='hwr.':
                    hwr_state_dict[key[4:]] = value
            self.guide_hwr=CRNN(num_class,norm='group',use_softmax=True)
            self.guide_hwr.load_state_dict( hwr_state_dict )
            for param in self.guide_hwr.parameters():
                param.will_use_grad=param.requires_grad
                param.requires_grad=False
        else:
            self.guide_hwr=None

        if 'style_discriminator' in config:
            spec=False
            if type(config['style_discriminator']) is int:
                num_layers=config['style_discriminator']
            elif 'spectral' in config['style_discriminator']:
                num_layers=int(config['style_discriminator'][0])
                spec=True
            else:
                num_layers=1
            prev = style_dim
            layers=[]
            for i in range(1,num_layers):
                if spec:
                    layers.append(SpectralNorm(nn.Linear(prev,prev//2)))
                    layers.append(nn.LeakyReLU(0.01,True))
                else:
                    layers.append(nn.Linear(prev,prev//2))
                    layers.append(nn.ReLU(True))
                prev = prev//2
            if spec:
                layers.append(SpectralNorm(nn.Linear(prev,1)))
            else:
                layers.append(nn.Linear(prev,1))
            self.style_discriminator = nn.Sequential(*layers)
        else:
            self.style_discriminator = None
        if 'mask_discriminator' in config and config['mask_discriminator']:
            use_derivitive = type(config['mask_discriminator']) is str and 'deriv' in config['mask_discriminator']
            self.mask_discriminator = TopAndBottomDiscriminator(use_derivitive=use_derivitive)

        self.clip_gen_mask = config['clip_gen_mask'] if 'clip_gen_mask' in config else None

        self.use_hwr_pred_for_style = config['use_hwr_pred_for_style'] if 'use_hwr_pred_for_style' in config else True
        self.pred = None
        self.spaced_label = None
        self.spacing_pred = None
        self.mask_pred = None

    def forward(self,label,label_lengths,style,mask=None,spaced=None,flat=False):
        batch_size = label.size(1)
        if type(self.generator.gen) is NewRNNDecoder:
            if mask is None:
                size = [batch_size,1,64,1200] #TODO not have this hard coded
                dist_map = self.generate_distance_map(size).to(label.device)
            else:
                dist_map = mask

            label_onehot=self.onehot(label)
            if mask is None:
                gen_img = self.generator(label_onehot,style,dist_map,return_intermediate=False)
                self.spacing_pred = None
                self.mask_pred = None
            else:
                gen_img, spaced,mask = self.generator(label_onehot,style,dist_map,return_intermediate=True)
                self.spacing_pred = spaced
                self.mask_pred = mask
        else:
            if mask is None:
                label_onehot=self.onehot(label)
                self.counts = self.spacer(label_onehot,style)
                spaced = self.insert_spaces(label,label_lengths,self.counts).to(label.device)
                self.top_and_bottom = self.create_mask(spaced,style)
                size = [batch_size,1,self.image_height,self.top_and_bottom.size(0)]
                mask = self.write_mask(self.top_and_bottom,size,flat=flat).to(label.device)
                if self.clip_gen_mask is not None:
                    mask = mask[:,:,:,:self.clip_gen_mask]
                self.gen_mask = mask

            gen_img = self.generator(spaced,style,mask)
        return gen_img

    def autoencode(self,image,label,mask,a_batch_size=None,center_line=None):
        style = self.extract_style(image,label,a_batch_size)
        if self.spaced_label is None:
            self.spaced_label = self.correct_pred(self.pred,label)
            self.spaced_label = self.onehot(self.spaced_label)
        if type(self.generator.gen) is NewRNNDecoder:
            mask = self.generate_distance_map(image.size(),center_line).to(label.device)
        if mask is None:
            top_and_bottom =  self.create_mask(self.spaced_label,style)
            mask = self.write_mask(top_and_bottom,image.size(),center_line=center_line).to(label.device)
            ##DEBUG
            #mask_draw = ((mask+1)*127.5).numpy().astype(np.uint8)
            #for b in range(image.size(0)):
            #    cv2.imshow('pred',mask_draw[b,0])
            #    print('mask show')
            #    cv2.waitKey()
        recon = self.forward(label,None,style,mask,self.spaced_label)

        return recon,style

    def extract_style(self,image,label,a_batch_size=None):
        if self.pred is None:
            self.pred = self.hwr(image, None)
        if self.use_hwr_pred_for_style:
            spaced = self.pred.permute(1,2,0)
        else:
            if self.spaced_label is None:
                self.spaced_label = self.correct_pred(self.pred,label)
                self.spaced_label = self.onehot(self.spaced_label)
            spaced= self.spaced_label.permute(1,2,0)
        batch_size,feats,h,w = image.size()
        if a_batch_size is None:
            a_batch_size = batch_size
        spaced_len = spaced.size(2)
        #append all the instances in the batch by the same author together along the width dimension
        collapsed_image =  image.permute(1,2,0,3).contiguous().view(feats,h,batch_size//a_batch_size,w*a_batch_size).permute(2,0,1,3)
        collapsed_label = spaced.permute(1,0,2).contiguous().view(self.num_class,batch_size//a_batch_size,spaced_len*a_batch_size).permute(1,0,2)
        style = self.style_extractor(collapsed_image, collapsed_label)
        #style=style.expand(batch_size,-1)
        style = style.repeat(a_batch_size,1)
        return style

    def insert_spaces(self,label,label_lengths,counts):
        max_count = max(math.ceil(counts.max()),3)
        lines = []
        max_line_len=0
        batch_size = label.size(1)
        for b in range(batch_size):
            line=[]
            for i in range(label_lengths[b]):
                count = round(np.random.normal(counts[i,b,0].item(),self.count_std))
                if self.count_duplicates:
                    duplicates = round(np.random.normal(counts[i,b,1].item(),self.dup_std))
                else:
                    duplicates=1
                line+=[0]*count + [label[i][b]]*duplicates
            max_line_len = max(max_line_len,len(line))
            lines.append(line)

        spaced = torch.zeros(max_line_len+max_count,batch_size,self.num_class)
        for b in range(batch_size):
            for i,cls in enumerate(lines[b]):
                spaced[i,b,cls]=1
            for i in range(len(lines[b]),spaced.size(0)):
                spaced[i,b,0]=1

        return spaced

    def write_mask(self,top_and_bottom,size, center_line=None,flat=False):
        #generate a center-line
        batch_size, ch, height, width = size
        mask = torch.zeros(*size)

        if center_line is None:
            center = height//2
            max_center = center+int(height*0.2)
            min_center = center-int(height*0.2)
            step = 3*height/2 #this is from utils.util.getCenterValue()
            last_x = 0
            if flat:
                last_y = np.full(batch_size, center)
            else:
                last_y = np.random.normal(center, (center-min_center)/3, batch_size)
            last_y[last_y>max_center]=max_center
            last_y[last_y<min_center]=min_center
            #debug=''
            while last_x<width:
                if flat:
                    next_x=last_x+step
                    next_y=np.full(batch_size, center)
                else:
                    next_x = np.random.normal(last_x+step,step*0.2)
                    next_y = np.random.normal(last_y,(center-min_center)/5,batch_size)
                next_y[next_y>max_center]=max_center
                next_y[next_y<min_center]=min_center
                #debug+='{}, '.format(next_y)

                self.draw_section(last_x,last_y,next_x,next_y,mask,top_and_bottom)

                last_x=next_x
                last_y=next_y
            #print(debug)
        else:
            for x in range(width):
                if x>=width or x>=top_and_bottom.size(0):
                    break
                for b in range(batch_size):
                    top = max(0,int(center_line[b,x]-top_and_bottom[x,b,0].item()))
                    bot = min(height,int(center_line[b,x]+top_and_bottom[x,b,1].item()+1))
                    mask[b,0,top:bot,x]=1
        
        blur_kernel = 31
        blur_padding = blur_kernel // 2
        blur = torch.nn.AvgPool2d((blur_kernel//4,blur_kernel//4), stride=1, padding=(blur_padding//4,blur_padding//4))
        return blur((2*mask)-1)

    def draw_section(self,last_x,last_y,next_x,next_y,mask,top_and_bottom):
        batch_size, ch, height, width = mask.size()
        for x in range(int(last_x),int(next_x)):
            if x>=width or x>=top_and_bottom.size(0):
                break
            progress = (x-int(last_x))/(int(next_x)-int(last_x))
            y = (1-progress)*last_y + progress*next_y

            for b in range(batch_size):
                top = max(0,int(y[b]-top_and_bottom[x,b,0].item()))
                bot = min(height,int(y[b]+top_and_bottom[x,b,1].item()+1))
                mask[b,0,top:bot,x]=1

    def correct_pred(self,pred,label):
        #Get optimal alignment
        #use DTW
        # introduce blanks at front, back, and inbetween chars
        label_with_blanks = torch.LongTensor(label.size(0)*2+1, label.size(1)).zero_()
        label_with_blanks[1::2]=label.cpu()
        pred_use = pred.cpu().detach()

        batch_size=pred_use.size(1)
        label_len=label_with_blanks.size(0)
        pred_len=pred_use.size(0)

        dtw = torch.FloatTensor(pred_len+1,label_len+1,batch_size).fill_(float('inf'))
        dtw[0,0]=0
        w = max(pred_len//2, abs(pred_len-label_len))
        for i in range(1,pred_len+1):
            dtw[i,max(1, i-w):min(label_len, i+w)+1]=0
        history = torch.IntTensor(pred_len,label_len,batch_size)
        for i in range(1,pred_len+1):
            for j in range(max(1, i-w), min(label_len, i+w)+1):
                cost = 1-pred_use[i-1,torch.arange(0,batch_size).long(),label_with_blanks[j-1,:]]
                per_batch_min, history[i-1,j-1] = torch.min( torch.stack( (dtw[i-1,j],dtw[i-1,j-1],dtw[i,j-1]) ), dim=0)
                dtw[i,j] = cost + per_batch_min
        new_labels = []
        maxlen = 0
        for b in range(batch_size):
            new_label = []
            i=pred_len-1
            j=label_len-1
            #accum += allCosts[b,i,j]
            new_label.append(label_with_blanks[j,b])
            while(i>0 or j>0):
                if history[i,j,b]==0:
                    i-=1
                elif history[i,j,b]==1:
                    i-=1
                    j-=1
                elif history[i,j,b]==2:
                    j-=1
                #accum+=allCosts[b,i,j]
                new_label.append(label_with_blanks[j,b])
            new_label.reverse()
            maxlen = max(maxlen,len(new_label))
            new_label = torch.stack(new_label,dim=0)
            new_labels.append(new_label)

        new_labels = [ F.pad(l,(0,maxlen-l.size(0)),value=0) for l in new_labels]
        new_label = torch.LongTensor(maxlen,batch_size)
        for b,l in enumerate(new_labels):
            new_label[:l.size(0),b]=l

        #set to one hot at alignment
        #new_label = self.onehot(new_label)
        #fuzzy other neighbor preds
        #TODO

        return new_label.to(label.device)


    #def onehot(self,label):
    #    label_onehot = torch.zeros(label.size(0),label.size(1),self.num_class)
    #    #label_onehot[label]=1
    #    for i in range(label.size(0)):
    #        for j in range(label.size(1)):
    #            label_onehot[i,j,label[i,j]]=1
    #    return label_onehot.to(label.device)
    def onehot(self,label): #tensorized version
        label_onehot = torch.zeros(label.size(0),label.size(1),self.num_class)
        label_onehot_v = label_onehot.view(label.size(0)*label.size(1),self.num_class)
        label_onehot_v[torch.arange(0,label.size(0)*label.size(1)),label.view(-1).long()]=1
        return label_onehot.to(label.device)


    def generate_distance_map(self,size,center_line=None):
        batch_size = size[0]
        height = size[2]
        width = size[3]
        line_im = np.ones((batch_size,height,width))

        if center_line is None:
            center = height//2
            max_center = center+int(height*0.2)
            min_center = center-int(height*0.2)
            step = 3*height/2 #this is from utils.util.getCenterValue()
            last_x = 0
            last_y = np.random.normal(center, (center-min_center)/3, batch_size)
            last_y[last_y>max_center]=max_center
            last_y[last_y<min_center]=min_center
            #debug=''
            while last_x<width-1:
                next_x = min(np.random.normal(last_x+step,step*0.2), width-1)
                next_y = np.random.normal(last_y,(center-min_center)/5,batch_size)
                next_y[next_y>max_center]=max_center
                next_y[next_y<min_center]=min_center
                #debug+='{}, '.format(next_y)

                #self.draw_section(last_x,last_y,next_x,next_y,mask,top_and_bottom)
                for b in range(batch_size):
                    rr,cc = draw.line(int(last_y[b]),int(last_x),int(next_y[b]),int(next_x))
                    line_im[b,rr,cc]=0

                last_x=next_x
                last_y=next_y
            #print(debug)
        else:
            for x in range(width):
                if x>=center_line.size(1):
                    break
                #TODO there should be a way to tensorize this
                for b in range(batch_size):
                    line_im[b,round(center_line[b,x].item()),x]=0
        maps=[]
        for b in range(batch_size):
            maps.append(torch.from_numpy(distance_transform_edt(line_im[b])).float())
        maps = torch.stack(maps,dim=0)[:,None,:,:]

        maps /= height/2
        maps[maps>1] = 1
        masp = 1-maps

        return maps
