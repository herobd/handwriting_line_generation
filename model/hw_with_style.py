from base import BaseModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import cv2
from model.cnn_lstm import CRNN, SmallCRNN
from model.cnn_lstm_skip import CRNNSkip
from model.cnn_only_hwr import CNNOnlyHWR
from model.google_hwr import GoogleHWR
from model.style_hwr import AdaINStyleHWR, HyperStyleHWR, DeformStyleHWR
from model.deform_hwr import DeformHWR
from model.MUNIT_networks import StyleEncoderHW, HWDecoder, MLP, Deep1DDecoder, Deep1DDecoderWithStyle, Shallow1DDecoderWithStyle, SpacedDecoderWithStyle, NewRNNDecoder, SpacedDecoderWithMask
from model.MUNIT_networks import get_num_adain_params, assign_adain_params
from model.pretrained_gen import PretrainedGen
from model.pure_gen import SpacedGenerator, SpacedUnStyledGenerator, PureGenerator, CharSpacedGenerator, PixelNorm
from model.style import NewHWStyleEncoder
from model.vae_style import VAEStyleEncoder
from model.lookup_style import LookupStyle
from model.discriminator import Discriminator, DownDiscriminator, TwoScaleDiscriminator, TwoScaleBetterDiscriminator
from model.cond_discriminator import CondDiscriminator
from model.cond_discriminator_ap import CondDiscriminatorAP
from model.test_disc import TestCondDiscriminator, TestImageDiscriminator, TestSmallCondDiscriminator
from model.char_style import CharStyleEncoder
from model.char_gen import CharSpacedDecoderWithMask
from model.char_cond_discriminator_ap import CharCondDiscriminatorAP
from model.mask_rnn import CountRNN, CreateMaskRNN, TopAndBottomDiscriminator, CountCNN
from model.char_mask_rnn import CharCountRNN, CharCreateMaskRNN, CharCountCNN
from model.discriminator import SpectralNorm
from model.simple_gan import SimpleGen, SimpleDisc
from model.author_classifier import AuthorClassifier
from skimage import draw
from scipy.ndimage.morphology import distance_transform_edt

def correct_pred(pred,label):
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


class AdaINGen(nn.Module):
    def __init__(self,n_class,style_dim,n_res1=2,n_res2=1,n_res3=0, dim=256, output_dim=1, style_transform_dim=None,activ='lrelu', type="HW", space_style_size=32,dist_map_text=False,use_skips=False,noise=False, char_style_size=0,decoder_weights=None,extra_text=False):
        super(AdaINGen, self).__init__()
        self.pad=True
        self.char_style=False
        if type=='HW':
            self.gen = HWDecoder(n_class, n_res1, n_res2, n_res3, dim, output_dim, res_norm='adain', activ=activ, pad_type='zero')
        elif type =='Deep1D':
            self.gen = Deep1DDecoder(n_class, n_res1, n_res2, n_res3, dim, output_dim, res_norm='adain', activ=activ, pad_type='zero')
        elif type.startswith('Deep1DWithStyle'):
            intermediate='Space' in type
            self.gen = Deep1DDecoderWithStyle(n_class, n_res1, n_res2, n_res3, dim, output_dim, style_dim, res_norm='adain', activ=activ, pad_type='zero', intermediate=intermediate)
        elif type == 'SpacedWithStyle':
            self.gen = SpacedDecoderWithStyle(n_class, n_res1, n_res2, n_res3, dim, output_dim, style_dim, res_norm='adain', activ=activ, pad_type='zero')
        elif type == 'CharSpacedWithMask':
            self.gen = CharSpacedDecoderWithMask(n_class, n_res1, n_res2, n_res3, dim, output_dim, style_dim, res_norm='adain', activ=activ, pad_type='zero', char_style_size=char_style_size,dist_map_text=dist_map_text,use_skips=use_skips,noise=noise,extra_text=extra_text)
            self.pad=False
            self.char_style=True
        elif type == 'SpacedWithMask':
            #self.gen = SpacedDecoderWithMask(n_class, n_res1, n_res2, n_res3, n_res1, dim, output_dim, style_dim, res_norm='adain', activ=activ, pad_type='zero', space_style_size=space_style_size)
            self.gen = SpacedDecoderWithMask(n_class, n_res1, n_res2, n_res3, dim, output_dim, style_dim, res_norm='adain', activ=activ, pad_type='zero', space_style_size=space_style_size,dist_map_text=dist_map_text,use_skips=use_skips,noise=noise,extra_text=extra_text)
            self.pad=False
        elif 'PretrainedGen' in type:
            decoder_type = type[14:]
            #self.gen = SpacedDecoderWithMask(n_class, n_res1, n_res2, n_res3, n_res1, dim, output_dim, style_dim, res_norm='adain', activ=activ, pad_type='zero', space_style_size=space_style_size)
            self.gen = PretrainedGen(n_class, n_res1, n_res2, n_res3, dim, output_dim, style_dim, res_norm='adain', activ=activ, pad_type='zero', space_style_size=space_style_size,noise=noise,decoder_type=decoder_type,decoder_weights=decoder_weights)
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
        adain_params = self.style_transform(style[0] if self.char_style else style)
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
        self.count_std = config['count_std'] if 'count_std' in config else 0.1
        self.dup_std = config['dup_std'] if 'dup_std' in config else 0.03
        self.image_height=64
        dim = config['style_dim']//4 if 'style_dim' in config else 64
        style_dim = config['style_dim'] if 'style_dim' in config else 256
        self.style_dim = style_dim
        char_style_dim = config['char_style_dim'] if 'char_style_dim' in config else 0
        self.char_style_dim = char_style_dim
        norm = config['style_norm'] if 'style_norm' in config else 'none'
        activ = config['style_activ'] if 'style_activ' in config else 'lrelu'
        pad_type = config['pad_type'] if 'pad_type' in config else 'replicate'
        self.max_gen_length = config['max_gen_length'] if 'max_gen_length' in config else 500

        self.noisy_style = config['noisy_style'] if 'noisy_style' in config else False

        num_class = config['num_class']
        self.num_class=num_class

        style_type = config['style'] if 'style' in config else 'normal'

        self.cond_disc=False
        self.vae=False

        if 'pretrained_autoencoder' in config and config['pretrained_autoencoder'] is not None:
            snapshot = torch.load(config['pretrained_autoencoder'], map_location='cpu')
            encoder_state_dict={}
            decoder_state_dict={}
            for key,value in  snapshot['state_dict'].items():
                if key.startswith('encoder.'):
                    encoder_state_dict[key[8:]] = value
                elif key.startswith('decoder.'):
                    decoder_state_dict[key[8:]] = value
        else:
            encoder_state_dict=None
            decoder_state_dict=None

        if 'new' in style_type:
            num_keys = config['num_keys'] if 'num_keys' in config else 16
            frozen_keys = config['frozen_keys'] if 'frozen_keys' in config else False
            global_pool = config['global_pool'] if 'global_pool' in config else False
            attention = config['attention'] if 'attention' in config else True
            dim = config['style_extractor_dim'] if 'style_extractor_dim' in config else dim
            use_pretrained_encoder = config['use_pretrained_encoder'] if 'use_pretrained_encoder' in config else False
            char_pred = config['char_style_pred'] if 'char_style_pred' in config else None
            self.style_extractor = NewHWStyleEncoder(input_dim, dim, style_dim, norm, activ, pad_type, num_class, num_keys=num_keys, frozen_keys=frozen_keys,global_pool=global_pool, attention=attention, use_pretrained_encoder=use_pretrained_encoder, encoder_weights=encoder_state_dict,char_pred=char_pred)
        elif 'char' in style_type:
            vae = 'VAE' in style_type
            self.vae=vae
            if 'none' in style_type:
                self.style_extractor = None
            else:
                small = 'small' in style_type
                global_pool = config['global_pool'] if 'global_pool' in config else False
                dim = config['style_extractor_dim'] if 'style_extractor_dim' in config else dim
                char_dim = config['char_style_extractor_dim'] if 'char_style_extractor_dim' in config else dim*2
                average_found_char_style = config['average_found_char_style']
                num_final_g_spacing_style = config['style_final_g_spacing'] if 'style_final_g_spacing' in config else 1
                num_char_fc = config['style_char_layers'] if 'style_char_layers' in config else 1
                window = config['char_style_window'] if 'char_style_window' in config else 6
                self.style_extractor = CharStyleEncoder(input_dim, dim, style_dim, char_dim, char_style_dim, norm, activ, pad_type, num_class, 
                        global_pool=global_pool,
                        average_found_char_style=average_found_char_style,
                        num_final_g_spacing_style=num_final_g_spacing_style,
                        num_char_fc=num_char_fc,
                        vae=vae,
                        window=window,
                        small=small)
        elif 'VAE' in style_type:
            self.vae=True
            if 'none' in style_type:
                self.style_extractor = None
            else:
                small = 'small' in style_type
                num_keys = config['num_keys'] if 'num_keys' in config else 16
                frozen_keys = config['frozen_keys'] if 'frozen_keys' in config else False
                global_pool = config['global_pool'] if 'global_pool' in config else False
                attention = config['attention'] if 'attention' in config else True
                wider = config['style_vae_wider'] if 'style_vae_wider' in config else False
                dim = config['style_extractor_dim'] if 'style_extractor_dim' in config else dim
                self.style_extractor = VAEStyleEncoder(input_dim, dim, style_dim, norm, activ, pad_type, num_class, num_keys=num_keys, frozen_keys=frozen_keys,global_pool=global_pool,attention=attention,wider=wider,small=small)
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
        if 'CRNNSkip' in hwr_type:
            if 'group' in hwr_type:
                norm='group'
            elif 'no_norm' in hwr_type or 'no norm' in hwr_type:
                norm=None
            else:
                norm='batch'
            use_softmax = True #'softmax' in hwr_type
            pad = 'pad' in hwr_type
            small = 'small' in hwr_type or 'sma32' in hwr_type
            self.hwr=CRNNSkip(num_class,norm=norm, use_softmax=use_softmax,small=small,pad=pad)
        elif 'CRNN' in hwr_type:
            if 'group' in hwr_type:
                norm='group'
            elif 'no_norm' in hwr_type or 'no norm' in hwr_type:
                norm=None
            else:
                norm='batch'
            use_softmax = True #'softmax' in hwr_type
            if 'small' in hwr_type:
                self.hwr=SmallCRNN(num_class,norm=norm, use_softmax=use_softmax)
            else:
                pad = 'less' if 'pad less' in hwr_type else 'pad' in hwr_type
                small = 'sma32' in hwr_type
                self.hwr=CRNN(num_class,norm=norm, use_softmax=use_softmax,small=small,pad=pad)
        elif 'CNNOnly' in hwr_type:
            useGRCL = 'GRCL' in hwr_type
            if 'group' in hwr_type:
                norm='group'
            else:
                norm='batch'
            small = 'small' in hwr_type
            pad = 'pad' in hwr_type
            if pad and 'pad less' in hwr_type:
                pad = 'less'
            self.hwr=CNNOnlyHWR(num_class,useGRCL=useGRCL,norm=norm,small=small,pad=pad)
        elif 'Google' in hwr_type:
            inceptionNorm = 'norm2d' in hwr_type
            grclNorm = 'norm1d' in hwr_type
            reducedContext = 'reducedContext' in hwr_type
            self.hwr=GoogleHWR(num_class,inceptionNorm=inceptionNorm,grclNorm=grclNorm,reducedContext=reducedContext)
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
        elif 'none' in hwr_type:
            self.hwr=None
        else:
            raise NotImplementedError('unknown HWR model: '+hwr_type)
        self.hwr_frozen=False
        if 'pretrained_hwr' in config and config['pretrained_hwr'] is not None:
            snapshot = torch.load(config['pretrained_hwr'], map_location='cpu')
            hwr_state_dict={}
            for key,value in  snapshot['state_dict'].items():
                if key[:4]=='hwr.':
                    hwr_state_dict[key[4:]] = value
            if len(hwr_state_dict)==0:
                hwr_state_dict=snapshot['state_dict']
            self.hwr.load_state_dict( hwr_state_dict )
            #if 'hwr_frozen' in config and config['hwr_frozen']:
            #    self.hwr_frozen=True
            #    for param in self.hwr.parameters():
            #        param.will_use_grad=param.requires_grad
            #        param.requires_grad=False

        if 'generator' in config and config['generator'] == 'none':
            self.generator = None
        elif 'simple' in config['generator']:
            self.generator=SimpleGen()
        elif 'Pure' in config['generator'] and 'no space' in config['generator']:
            g_dim = config['gen_dim'] if 'gen_dim' in config else 256
            n_style_trans = config['n_style_trans'] if 'n_style_trans' in config else 6
            g_depth1 = config['gen_depth1'] if 'gen_depth1' in config else 3
            g_depth2 = config['gen_depth2'] if 'gen_depth2' in config else 2
            self.generator = PureGenerator(num_class,style_dim,g_dim,n_style_trans=n_style_trans,depth1=g_depth1,depth2=g_depth2)
        elif 'Pure' in config['generator'] and 'no style' in config['generator']:
            g_dim = config['gen_dim'] if 'gen_dim' in config else 256
            n_style_trans = config['n_style_trans'] if 'n_style_trans' in config else 6
            no_content = 'no content' in config['generator']
            use_noise = 'use noise' in config['generator']
            small = 'small' in config['generator']
            dist_map_content = config['dist_map_content_for_gen'] if 'dist_map_content_for_gen' in config else False
            self.generator = SpacedUnStyledGenerator(num_class,style_dim,g_dim,n_style_trans=n_style_trans,no_content=no_content,use_noise=use_noise,small=small,dist_map_content=dist_map_content)
        elif 'Pure' in config['generator'] and 'char spec' in config['generator']:
            g_dim = config['gen_dim'] if 'gen_dim' in config else 256
            n_style_trans = config['n_style_trans'] if 'n_style_trans' in config else 6
            dist_map_text = config['dist_map_text_for_gen'] if 'dist_map_text_for_gen' in config else False
            emb_dropout = config['style_emb_dropout'] if 'style_emb_dropout' in config else False
            skip_char_style = config['gen_skip_char_style'] if 'gen_skip_char_style' in config else False
            first1d = config['gen_first_1d'] if 'gen_first_1d' in config else False
            self.generator = CharSpacedGenerator(num_class,style_dim,char_style_dim,g_dim,n_style_trans=n_style_trans,dist_map_content=dist_map_text,emb_dropout=emb_dropout,skip_char_style=skip_char_style,first1d=first1d)
        elif 'Pure' in config['generator']:
            small = 'small' in config['generator']
            g_dim = config['gen_dim'] if 'gen_dim' in config else 256
            n_style_trans = config['n_style_trans'] if 'n_style_trans' in config else 6
            dist_map_text = config['dist_map_text_for_gen'] if 'dist_map_text_for_gen' in config else False
            emb_dropout = config['style_emb_dropout'] if 'style_emb_dropout' in config else False
            append_style = config['gen_append_style'] if 'gen_append_style' in config else False
            self.generator = SpacedGenerator(num_class,style_dim,g_dim,n_style_trans=n_style_trans,dist_map_content=dist_map_text,emb_dropout=emb_dropout,append_style=append_style,small=small)
        else:
            generator_type = config['generator'] if 'generator' in config else 'HW'
            n_res1 = config['gen_n_res1'] if 'gen_n_res1' in config else 2
            n_res2 = config['gen_n_res2'] if 'gen_n_res2' in config else 1
            n_res3 = config['gen_n_res3'] if 'gen_n_res3' in config else 0
            g_dim = config['gen_dim'] if 'gen_dim' in config else 256
            space_style_size=config['gen_space_style_size'] if 'gen_space_style_size' in config else 32
            dist_map_text = config['dist_map_text_for_gen'] if 'dist_map_text_for_gen' in config else False
            use_skips = config['gen_use_skips'] if 'gen_use_skips' in config else False
            noise = config['gen_use_noise'] if 'gen_use_noise' in config else False
            extra_text = config['gen_extra_text'] if 'gen_extra_text' in config else False
            self.generator = AdaINGen(num_class,style_dim, type=generator_type, n_res1=n_res1,n_res2=n_res2,n_res3=n_res3, dim=g_dim, space_style_size=space_style_size,dist_map_text=dist_map_text,use_skips=use_skips,noise=noise,char_style_size=char_style_dim, decoder_weights=decoder_state_dict,extra_text=extra_text)
        if 'pretrained_generator' in config and config['pretrained_generator'] is not None:
            snapshot = torch.load(config['pretrained_generator'], map_location='cpu')
            gen_state_dict={}
            for key,value in  snapshot['state_dict'].items():
                if key[:10]=='generator.':
                    gen_state_dict[key[10:]] = value
            self.generator.load_state_dict( gen_state_dict )

        if 'discriminator' in config and config['discriminator'] is not None:
            add_noise_img = config['disc_add_noise_img'] if 'disc_add_noise_img' in config else False
            add_noise_cond = config['disc_add_noise_cond'] if 'disc_add_noise_cond' in config else False
            if config['discriminator']=='down':
                self.discriminator = DownDiscriminator()
            elif 'simple' in config['discriminator']:
                self.discriminator=SimpleDisc()
            elif 'two' in config['discriminator'] and 'better' in config['discriminator']:
                more_low = 'more' in config['discriminator'] and 'low' in config['discriminator']
                global_pool = 'global' in config['discriminator']
                dim = 32 if 'half' in config['discriminator'] else 64
                self.discriminator = TwoScaleBetterDiscriminator(more_low=more_low,dim=dim,global_pool=global_pool,add_noise_img=add_noise_img)
            elif 'charCondAP' in config['discriminator']:
                dim = 32 if 'half' in config['discriminator'] else 64
                if 'disc_dim' in config:
                    dim = config['disc_dim']
                #keepWide = 'wide' in config['discriminator']
                use_style = 'no style' not in config['discriminator']
                global_pool = 'no global' not in config['discriminator']
                keepWide = True
                self.discriminator = CharCondDiscriminatorAP(num_class,style_dim,char_style_dim,dim,global_pool=global_pool,use_style=use_style,keepWide=keepWide)
                self.cond_disc=True
            elif 'image' in config['discriminator']:
                self.discriminator = TestImageDiscriminator()
                self.cond_disc=True
            elif 'test small' in config['discriminator']:
                dim = 32 if 'half' in config['discriminator'] else 64
                if 'disc_dim' in config:
                    dim = config['disc_dim']
                keepWide = True#'wide' in config['discriminator']
                use_style = False#'no style' not in config['discriminator']
                use_cond = 'no cond' not in config['discriminator']
                use_global = True#'no global' not in config['discriminator']
                global_only = True#'global_only' in config['discriminator']
                pool_med = 'avg' not in config['discriminator']
                use_pixel_stats = False#config['disc_use_pixel_stats'] if 'disc_use_pixel_stats' in config else False
                self.discriminator = TestSmallCondDiscriminator(num_class,style_dim,dim,keepWide=keepWide,use_style=use_style,use_pixel_stats=use_pixel_stats,use_cond=use_cond,global_only=global_only,global_pool=use_global,pool_med=pool_med)
                self.cond_disc=True
            elif 'test' in config['discriminator']:
                dim = 32 if 'half' in config['discriminator'] else 64
                if 'disc_dim' in config:
                    dim = config['disc_dim']
                keepWide = True#'wide' in config['discriminator']
                use_style = False#'no style' not in config['discriminator']
                use_cond = 'no cond' not in config['discriminator']
                use_global = True#'no global' not in config['discriminator']
                global_only = True#'global_only' in config['discriminator']
                pool_med = 'avg' not in config['discriminator']
                use_pixel_stats = False#config['disc_use_pixel_stats'] if 'disc_use_pixel_stats' in config else False
                self.discriminator = TestCondDiscriminator(num_class,style_dim,dim,keepWide=keepWide,use_style=use_style,use_pixel_stats=use_pixel_stats,use_cond=use_cond,global_only=global_only,global_pool=use_global,pool_med=pool_med)
                self.cond_disc=True
            elif 'condAP' in config['discriminator']:
                dim = 32 if 'half' in config['discriminator'] else 64
                if 'disc_dim' in config:
                    dim = config['disc_dim']
                keepWide = True#'wide' in config['discriminator']
                use_style = 'no style' not in config['discriminator']
                use_cond = 'no cond' not in config['discriminator']
                use_global = 'no global' not in config['discriminator']
                global_only = 'global_only' in config['discriminator']
                use_low = 'use low' in config['discriminator']
                use_pixel_stats = config['disc_use_pixel_stats'] if 'disc_use_pixel_stats' in config else False
                dist_map_content = 'dist map' in config['discriminator']
                convs3NoGroup = 'fixConvs3' in config['discriminator'] or 'fixConvs3' in config
                use_author = config['author_disc']+1 if 'author_disc' in config else None
                use_attention = 'use attention' in config['discriminator']
                no_high = 'use high' not in config['discriminator']
                use_med = 'no med' not in config['discriminator']
                small = 'small' in config['discriminator']
                self.discriminator = CondDiscriminatorAP(num_class,style_dim,dim,keepWide=keepWide,use_style=use_style,use_pixel_stats=use_pixel_stats,use_cond=use_cond,global_only=global_only,global_pool=use_global, no_high=no_high, use_low=use_low,add_noise_img=add_noise_img,add_noise_cond=add_noise_cond,dist_map_content=dist_map_content,convs3NoGroup=convs3NoGroup,use_authors_size=use_author,use_attention=use_attention,use_med=use_med,small=small)
                self.cond_disc=True
            elif 'cond' in config['discriminator']:
                dim = 32 if 'half' in config['discriminator'] else 64
                wide = 'wide' in config['discriminator']
                self.discriminator = CondDiscriminator(num_class,style_dim,dim,wide=wide,add_noise_img=add_noise_img,add_noise_cond=add_noise_cond)
                self.cond_disc=True
            elif 'two' in config['discriminator']:
                self.discriminator = TwoScaleDiscriminator()
            elif 'normal' in config['discriminator']:
                self.discriminator = Discriminator()
            elif config['discriminator']!='none':
                raise NotImplementedError('Unknown discriminator: {}'.format(config['discriminator']))
        if 'pretrained_discriminator' in config and config['pretrained_discriminator'] is not None:
            snapshot = torch.load(config['pretrained_discriminator'], map_location='cpu')
            discriminator_state_dict={}
            for key,value in  snapshot['state_dict'].items():
                if key[:14]=='discriminator.':
                    discriminator_state_dict[key[14:]] = value
            self.discriminator.load_state_dict( discriminator_state_dict )

        if 'spacer' in config and config['spacer']:
            self.count_duplicates =  type(config['spacer']) is str and 'duplicate' in config['spacer']
            num_out = 2 if self.count_duplicates else 1
            if config['spacer']=='identity':
                self.spacer = lambda input,style: torch.zeros(input.size(0),input.size(1),num_out).to(input.device)
            else:
                spacer_dim = config['spacer_dim'] if 'spacer_dim' in config else 128
                if type(config['spacer']) is str and 'CNN' in config['spacer']:
                    emb_style = config['spacer_emb_style'] if 'spacer_emb_style' in config else 0
                    fix_dropout=  config['spacer_fix_dropout'] if 'spacer_fix_dropout' in config else False
                    if char_style_dim>0:
                        self.spacer=CharCountCNN(num_class,style_dim,char_style_dim,spacer_dim,num_out,emb_style,fix_dropout)
                    else:
                        self.spacer=CountCNN(num_class,style_dim,spacer_dim,num_out,emb_style)
                else:
                    if char_style_dim>0:
                        self.spacer=CharCountRNN(num_class,style_dim,char_style_dim,spacer_dim,num_out)
                    else:
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
            shallow = config['create_mask_shallow'] if 'create_mask_shallow' in config else False
            if char_style_dim>0:
                self.create_mask=CharCreateMaskRNN(num_class,char_style_dim,create_mask_dim,shallow=shallow)
            else:
                self.create_mask=CreateMaskRNN(num_class,style_dim,create_mask_dim,shallow=shallow)
        else:
            self.create_mask=None
        if 'pretrained_create_mask' in config and config['pretrained_create_mask'] is not None:
            snapshot = torch.load(config['pretrained_create_mask'], map_location='cpu')
            create_mask_state_dict={}
            for key,value in  snapshot['state_dict'].items():
                if key[:12]=='create_mask.':
                    create_mask_state_dict[key[12:]] = value
            self.create_mask.load_state_dict( create_mask_state_dict )

        if 'style_from_normal' in config and config['style_from_normal']:
            self.style_from_normal = nn.Sequential(
                    nn.Linear(style_dim//2,style_dim),
                    nn.ReLU(True),
                    nn.Linear(style_dim,style_dim),
                    nn.ReLU(True),
                    nn.Linear(style_dim,style_dim)
                    )
        else:
            self.style_from_normal = None

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
        if 'pretrained_mask_discriminator' in config and config['pretrained_mask_discriminator'] is not None:
            snapshot = torch.load(config['pretrained_mask_discriminator'], map_location='cpu')
            mask_discriminator_state_dict={}
            for key,value in  snapshot['state_dict'].items():
                if key[:19]=='mask_discriminator.':
                    mask_discriminator_state_dict[key[19:]] = value
            self.mask_discriminator.load_state_dict( mask_discriminator_state_dict )

        if 'author_classifier' in config and config['author_classifier'] is not None:
            num_author = config['author_classifier']
            author_dim = config['author_dim'] if 'author_dim' in config else 64
            self.author_classifier = AuthorClassifier(num_author,author_dim)

        self.clip_gen_mask = config['clip_gen_mask'] if 'clip_gen_mask' in config else None

        self.use_hwr_pred_for_style = config['use_hwr_pred_for_style'] if 'use_hwr_pred_for_style' in config else True
        self.pred = None
        self.spaced_label = None
        self.spacing_pred = None
        self.mask_pred = None
        self.gen_spaced=None
        self.spaced_style=None
        if self.vae:
            self.mu=None
            self.log_sigma=None


        if config['emb_char_style'] if 'emb_char_style' in config else False:
            self.emb_char_style=nn.ModuleList()
            for c in range(num_class): #blank class already included?
                layers = [PixelNorm()]
                for i in range(3):
                    layers.append(nn.Linear(self.char_style_dim,self.char_style_dim))
                    if emb_dropout and i<n_style_trans-1:
                        layers.append(nn.Dropout(0.125))
                    layers.append(nn.LeakyReLU(0.2))
                self.emb_char_style.append(nn.Sequential(*layers))
        else:
            self.emb_char_style = None

    def forward(self,label,label_lengths,style,mask=None,spaced=None,flat=False,center_line=None):
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
                if self.spacer is None:
                    spaced=label
                else:
                    label_onehot=self.onehot(label)
                    self.counts = self.spacer(label_onehot,style)
                    spaced, padded = self.insert_spaces(label,label_lengths,self.counts)
                    spaced = spaced.to(label.device)
                    self.gen_padded = padded
                    if spaced.size(0) > self.max_gen_length:
                        #print('clipping content! {}'.format(spaced.size(0)))
                        diff = self.max_gen_length - spaced.size(0)
                        #cut blanks from the end
                        chars = spaced.argmax(2)
                        for x in range(spaced.size(0)-1,0,-1): #iterate backwards till we meet non-blank
                            if (chars[x]>0).any():
                                break
                        toRemove = min(diff,spaced.size(0)-x+2) #"+2" to pad out a couple blanks
                        if toRemove>0:
                            spaced = spaced[:-toRemove]
                    if spaced.size(0) > self.max_gen_length:
                        diff = self.max_gen_length - spaced.size(0)
                        #cut blanks from the front
                        chars = spaced.argmax(2)
                        for x in range(spaced.size(0)): #iterate forwards till we meet non-blank
                            if (chars[x]>0).any():
                                break
                        toRemove = max(min(diff,x-2),0) #"-2" to pad out a couple blanks
                        if toRemove>0:
                            spaced = spaced[toRemove:]

                        

                    if self.char_style_dim>0:
                        style = self.space_style(spaced,style,spaced.device)
                        self.spaced_style=style

                    if self.cond_disc:
                        self.gen_spaced=spaced
                    if self.create_mask is not None:
                        self.top_and_bottom = self.create_mask(spaced,style)
                        size = [batch_size,1,self.image_height,self.top_and_bottom.size(0)]
                        mask = self.write_mask(self.top_and_bottom,size,flat=flat).to(label.device)
                        if self.clip_gen_mask is not None:
                            mask = mask[:,:,:,:self.clip_gen_mask]
                        self.gen_mask = mask
                    #print('debug. label:{}, spaced:{}, mask:{}'.format(label.size(),spaced.size(),mask.size()))
            elif self.char_style_dim>0:
                if  self.spaced_style is None:
                    style = self.space_style(spaced,style)
                    self.spaced_style=style
                else:
                    style = self.spaced_style #this occurs on an auto-encode with generated mask

            gen_img = self.generator(spaced,style,mask)
        return gen_img

    def autoencode(self,image,label,mask,a_batch_size=None,center_line=None,stop_grad_extractor=False):
        style = self.extract_style(image,label,a_batch_size)
        if stop_grad_extractor:
            if self.char_style_dim>0:
                style = (style[0].detach(),style[1].detach(),style[2].detach())
            else:
                style=style.detach() #This is used when we use the auto-style loss, as wer're using the extractor result as the target
        if self.spaced_label is None:
            self.spaced_label = correct_pred(self.pred,label)
            self.spaced_label = self.onehot(self.spaced_label)
        if type(self.generator.gen) is NewRNNDecoder:
            mask = self.generate_distance_map(image.size(),center_line).to(label.device)
        if mask is None and self.create_mask is not None:
            with torch.no_grad():
                if self.char_style_dim>0:
                    self.spaced_style = self.space_style(self.spaced_label,style)
                    use_style = self.spaced_style
                else:
                    use_style = style
                top_and_bottom =  self.create_mask(self.spaced_label,use_style)
                mask = self.write_mask(top_and_bottom,image.size(),center_line=center_line).to(label.device)
                self.mask = mask.cpu()
        if mask is None and self.create_mask is None:
            mask=0
            ##DEBUG
            #mask_draw = ((mask+1)*127.5).numpy().astype(np.uint8)
            #for b in range(image.size(0)):
            #    cv2.imshow('pred',mask_draw[b,0])
            #    print('mask show')
            #    cv2.waitKey()
        else:
            self.spaced_style = None
        recon = self.forward(label,None,style,mask,self.spaced_label)

        return recon,style

    def extract_style(self,image,label,a_batch_size=None):
        if self.pred is None:
            self.pred = self.hwr(image, None)
        if self.use_hwr_pred_for_style:
            spaced = self.pred.permute(1,2,0)
        else:
            if self.spaced_label is None:
                self.spaced_label = correct_pred(self.pred,label)
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
        if self.vae:
            if self.char_style_dim>0:
                g_mu,g_log_sigma,spacing_mu, spacing_log_sigma,char_mu,char_log_sigma = style
                g_sigma = torch.exp(g_log_sigma)
                g_style = g_mu + g_sigma * torch.randn_like(g_mu)
                spacing_sigma = torch.exp(spacing_log_sigma)
                spacing_style = spacing_mu + spacing_sigma * torch.randn_like(spacing_mu)
                char_sigma = torch.exp(char_log_sigma)
                char_style = char_mu + char_sigma * torch.randn_like(char_mu)
                self.mu=torch.cat( (g_mu,spacing_mu,char_mu.contiguous().view(batch_size//a_batch_size,-1)), dim=1)
                self.sigma=torch.cat( (g_sigma,spacing_sigma,char_sigma.view(batch_size//a_batch_size,-1)), dim=1)
                style = (g_style,spacing_style,char_style)
            else:
                mu,log_sigma = style
                #assert(not torch.isnan(mu).any())
                assert(not torch.isnan(log_sigma).any())
                if self.training:
                    sigma = torch.exp(log_sigma)
                    style = mu + sigma * torch.randn_like(mu)
                    self.mu=mu
                    self.sigma=sigma
                else:
                    sigma = torch.exp(log_sigma)
                    style = mu + sigma * torch.randn_like(mu)*0.8
                    self.mu=mu
                    self.sigma=sigma
        if self.noisy_style:
            if self.char_style_dim>0:
                raise NotImplementedError('haven;t implmented noise for char spec style vectors')
            else:
                var = style.abs().mean()
                style = style+torch.randn_like(style)*var
        if self.char_style_dim>0:
            g_style,spacing_style,char_style = style
            g_style = torch.cat([g_style[i:i+1].repeat(a_batch_size,1) for i in range(g_style.size(0))],dim=0)
            spacing_style = torch.cat([spacing_style[i:i+1].repeat(a_batch_size,1) for i in range(spacing_style.size(0))],dim=0)
            char_style = torch.cat([char_style[i:i+1].repeat(a_batch_size,1,1) for i in range(char_style.size(0))],dim=0)
            style = (g_style,spacing_style,char_style)
        else:
            #style = style.repeat(a_batch_size,1)
            style = torch.cat([style[i:i+1].repeat(a_batch_size,1) for i in range(style.size(0))],dim=0)
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
        padded=[]
        for b in range(batch_size):
            for i,cls in enumerate(lines[b]):
                spaced[i,b,cls]=1
            for i in range(len(lines[b]),spaced.size(0)):
                spaced[i,b,0]=1
            padded.append((spaced.size(0)-len(lines[b]))/spaced.size(0))

        return spaced, padded

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
            while last_x<width:
                if flat:
                    next_x=last_x+step
                    next_y=np.full(batch_size, center)
                else:
                    next_x = np.random.normal(last_x+step,step*0.2)
                    next_y = np.random.normal(last_y,(center-min_center)/5,batch_size)
                next_y[next_y>max_center]=max_center
                next_y[next_y<min_center]=min_center

                self.draw_section(last_x,last_y,next_x,next_y,mask,top_and_bottom)

                last_x=next_x
                last_y=next_y
        else:
            ###DEBUG
            if center_line.size(1)<width:
                center_line = torch.cat((center_line, torch.FloatTensor(batch_size,width-center_line.size(1)).fill_(height//2)),dim=1)
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

    def space_style(self,spaced,style,device=None):
        #spaced is Width x Batch x Channel
        g_style,spacing_style,char_style = style
        if self.emb_char_style is None:
            device = spaced.device
        spacing_style = spacing_style.to(device)
        char_style = char_style.to(device)
        batch_size = spaced.size(1)
        style = torch.FloatTensor(spaced.size(0),batch_size,self.char_style_dim).to(device)
        text_chars = spaced.argmax(dim=2)
        spacing_style = spacing_style[None,:,:] #add temporal dim for broadcast
        #Put character styles in appropriate places. Fill in rest with projected global style
        for b in range(batch_size):
            lastChar = -1
            for x in range(0,text_chars.size(0)):
                if text_chars[x,b]!=0:
                    charIdx = text_chars[x,b]
                    if self.emb_char_style is not None:
                        style[x,b,:] = self.emb_char_style[charIdx](char_style[b,charIdx])
                        style[lastChar+1:x,b,:] = self.emb_char_style[0](spacing_style[:,b]) #broadcast
                    else:
                        style[x,b,:] = char_style[b,charIdx]
                        style[lastChar+1:x,b,:] = spacing_style[:,b] #broadcast
                    lastChar=x
            style[lastChar+1:,b,:] = spacing_style[:,b]
        return (g_style,style,char_style)
