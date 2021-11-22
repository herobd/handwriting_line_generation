from base import BaseModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import cv2
from model.cnn_lstm import CRNN, SmallCRNN
from model.cnn_only_hwr import CNNOnlyHWR
from model.pure_gen import SpacedGenerator
from model.discriminator_ap import DiscriminatorAP
from model.char_style import CharStyleEncoder
from model.count_cnn import CountCNN
from skimage import draw
import os

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


        num_class = config['num_class']
        self.num_class=num_class

        style_type = config['style'] if 'style' in config else 'normal'

        
        self.vae=False


        if 'char' in style_type:
            #vae = 'VAE' in style_type
            self.vae=False
            #if 'none' in style_type:
            #    self.style_extractor = None
            #else:
            small = False
            global_pool = config['style_global_pool'] if 'style_global_pool' in config else False
            dim = config['style_extractor_dim'] if 'style_extractor_dim' in config else dim
            char_dim = config['char_style_extractor_dim'] if 'char_style_extractor_dim' in config else dim*2
            average_found_char_style = config['average_found_char_style']
            num_final_g_spacing_style = 1 #config['style_final_g_spacing'] if 'style_final_g_spacing' in config else 1
            num_char_fc = 1 #config['style_char_layers'] if 'style_char_layers' in config else 1
            window = config['char_style_window'] if 'char_style_window' in config else 6
            self.style_extractor = CharStyleEncoder(input_dim, dim, style_dim, char_dim, char_style_dim, norm, activ, pad_type, num_class, 
                    global_pool=global_pool,
                    average_found_char_style=average_found_char_style,
                    num_final_g_spacing_style=num_final_g_spacing_style,
                    num_char_fc=num_char_fc,
                    vae=self.vae,
                    window=window,
                    small=small)
        else:
            self.style_extractor = None

        hwr_type= config['hwr'] if 'hwr' in config else 'CRNN'
        if 'CRNN' in hwr_type:
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
            
            if 'group' in hwr_type:
                norm='group'
            else:
                norm='batch'
            small = 'small' in hwr_type
            pad = 'pad' in hwr_type
            if pad and 'pad less' in hwr_type:
                pad = 'less'
            self.hwr=CNNOnlyHWR(num_class,norm=norm,small=small,pad=pad)
        elif 'none' in hwr_type:
            self.hwr=None
        else:
            raise NotImplementedError('unknown HWR model: '+hwr_type)
        self.hwr_frozen=False
        if 'pretrained_hwr' in config and config['pretrained_hwr'] is not None:
            if os.path.exists(config['pretrained_hwr']):
                snapshot = torch.load(config['pretrained_hwr'], map_location='cpu')
                hwr_state_dict={}
                for key,value in  snapshot['state_dict'].items():
                    if key[:4]=='hwr.':
                        hwr_state_dict[key[4:]] = value
                if len(hwr_state_dict)==0:
                    hwr_state_dict=snapshot['state_dict']
                self.hwr.load_state_dict( hwr_state_dict )
            elif not config.get('RUN'):
                print('Could not open pretrained HWR weights at '+config['pretrained_hwr'])
                exit(1)

        if 'generator' in config and config['generator'] == 'none':
            self.generator = None
        elif 'Pure' in config['generator']:
            small = 'small' in config['generator']
            g_dim = config['gen_dim'] if 'gen_dim' in config else 256
            n_style_trans = config['n_style_trans'] if 'n_style_trans' in config else 6
            emb_dropout = config['style_emb_dropout'] if 'style_emb_dropout' in config else False
            append_style = config['gen_append_style'] if 'gen_append_style' in config else False
            self.generator = SpacedGenerator(num_class,style_dim,g_dim,n_style_trans=n_style_trans,emb_dropout=emb_dropout,append_style=append_style,small=small)
        else:
            raise NotImplementedError('unknown generator: '+config['generator'])


        if 'discriminator' in config and config['discriminator'] is not None:
            dim = config['disc_dim'] if 'disc_dim' in config else 64
            use_low = 'use low' in config['discriminator']
            use_med = 'no med' not in config['discriminator']
            small = 'small' in config['discriminator']
            self.discriminator = DiscriminatorAP(dim, use_low=use_low,use_med=use_med,small=small)

        if 'spacer' in config and config['spacer']:
            self.count_duplicates =  type(config['spacer']) is str and 'duplicate' in config['spacer']
            num_out = 2 if self.count_duplicates else 1
            spacer_dim = config['spacer_dim'] if 'spacer_dim' in config else 128
            self.spacer=CountCNN(num_class,style_dim,spacer_dim,num_out)
        else:
            self.spacer=None


        self.create_mask=None
        if 'pretrained_create_mask' in config and config['pretrained_create_mask'] is not None:
            snapshot = torch.load(config['pretrained_create_mask'], map_location='cpu')
            create_mask_state_dict={}
            for key,value in  snapshot['state_dict'].items():
                if key[:12]=='create_mask.':
                    create_mask_state_dict[key[12:]] = value
            self.create_mask.load_state_dict( create_mask_state_dict )

        self.style_from_normal = None
        self.guide_hwr=None
        self.style_discriminator = None

        self.use_hwr_pred_for_style = config['use_hwr_pred_for_style'] if 'use_hwr_pred_for_style' in config else True
        self.pred = None
        self.spaced_label = None
        self.spacing_pred = None
        self.mask_pred = None
        self.gen_spaced=None
        self.spaced_style=None



    def forward(self,label,label_lengths,style,spaced=None):
        batch_size = label.size(1)
        
        if spaced is None:
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

                

            self.gen_spaced=spaced

        gen_img = self.generator(spaced,style)
        return gen_img

    def autoencode(self,image,label,a_batch_size=None,stop_grad_extractor=False):
        style = self.extract_style(image,label,a_batch_size)
        if stop_grad_extractor:
            style=style.detach() #This is used when we use the auto-style loss, as wer're using the extractor result as the target
        if self.spaced_label is None:
            self.spaced_label = correct_pred(self.pred,label)
            self.spaced_label = self.onehot(self.spaced_label)
        recon = self.forward(label,None,style,self.spaced_label)

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




    def onehot(self,label): #tensorized version
        label_onehot = torch.zeros(label.size(0),label.size(1),self.num_class)
        label_onehot_v = label_onehot.view(label.size(0)*label.size(1),self.num_class)
        label_onehot_v[torch.arange(0,label.size(0)*label.size(1)),label.view(-1).long()]=1
        return label_onehot.to(label.device)



    def space_style(self,spaced,style,device=None):
        #spaced is Width x Batch x Channel
        g_style,spacing_style,char_style = style
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
                    style[x,b,:] = char_style[b,charIdx]
                    style[lastChar+1:x,b,:] = spacing_style[:,b] #broadcast
                    lastChar=x
            style[lastChar+1:,b,:] = spacing_style[:,b]
        return (g_style,style,char_style)
