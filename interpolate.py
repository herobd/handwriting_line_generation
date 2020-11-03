# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import os
import json
import logging
import argparse
import torch
from model import *
from model.metric import *
from model.loss import *
from logger import Logger
from trainer import *
from data_loader import getDataLoader
from evaluators import *
import math
from collections import defaultdict
import pickle
from glob import glob
import cv2
from utils import string_utils
import random

#from datasets.forms_detect import FormsDetect
#from datasets import forms_detect

logging.basicConfig(level=logging.INFO, format='')

def get_style(config,model,instance, gpu=None):
    lookup_style = 'lookup' in config['model']['style'] or 'Lookup' in config['model']['style']
    style_together = config['trainer']['style_together'] if 'style_together' in config['trainer'] else False
    use_hwr_pred_for_style = config['trainer']['use_hwr_pred_for_style'] if 'use_hwr_pred_for_style' in config['trainer'] else False
    image = instance['image']
    label = instance['label']
    if gpu is not None:
        image = image.to(gpu)
        label = label.to(gpu)
    if lookup_style:
        style = model.style_extractor(instance['author'],gpu)
    else:
        if not style_together:
            style = model.style_extractor(image)
            style = style[0:1]
        else:
            #append all the instances in the batch by the same author together along the width dimension
            pred = model.hwr(image, None)
            num_class = pred.size(2)
            if use_hwr_pred_for_style:
                spaced_label = pred.permute(1,2,0)
            else:
                spaced_label = correct_pred(pred,label)
                spaced_label = onehot(spaced_label).permute(1,2,0)
            batch_size,feats,h,w = image.size()
            if 'a_batch_size' in instance:
                a_batch_size = instance['a_batch_size']
            else:
                a_batch_size = batch_size
            spaced_len = spaced_label.size(2)
            collapsed_image =  image.permute(1,2,0,3).contiguous().view(feats,h,batch_size//a_batch_size,w*a_batch_size).permute(2,0,1,3)
            collapsed_label = spaced_label.permute(1,0,2).contiguous().view(num_class,batch_size//a_batch_size,spaced_len*a_batch_size).permute(1,0,2)
            style = model.style_extractor(collapsed_image, collapsed_label)
            #style=style.expand(batch_size,-1)
            #style = style.repeat(a_batch_size,1)
    return style

def main(resume,saveDir,gpu=None,config=None,addToConfig=None, fromDataset=True):
    np.random.seed(1234)
    torch.manual_seed(1234)
    if resume is not None:
        checkpoint = torch.load(resume, map_location=lambda storage, location: storage)
        print('loaded iteration {}'.format(checkpoint['iteration']))
        if config is None:
            config = checkpoint['config']
        else:
            config = json.load(open(config))
        for key in config['model'].keys():
            if 'pretrained' in key:
                config['model'][key]=None
    else:
        checkpoint = None
        config = json.load(open(config))
    config['optimizer_type']="none"
    config['trainer']['use_learning_schedule']=False
    config['trainer']['swa']=False
    if gpu is None:
        config['cuda']=False
    else:
        config['cuda']=True
        config['gpu']=gpu
    addDATASET=False
    if addToConfig is not None:
        for add in addToConfig:
            addTo=config
            printM='added config['
            for i in range(len(add)-2):
                addTo = addTo[add[i]]
                printM+=add[i]+']['
            value = add[-1]
            if value=="":
                value=None
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            addTo[add[-2]] = value
            printM+=add[-2]+']={}'.format(value)
            print(printM)
            if (add[-2]=='useDetections' or add[-2]=='useDetect') and value!='gt':
                addDATASET=True

    if fromDataset:
        config['data_loader']['batch_size']=1
        config['validation']['batch_size']=1
        data_loader, valid_data_loader = getDataLoader(config,'train')
    

    if checkpoint is not None:
        if 'state_dict' in checkpoint:
            model = eval(config['arch'])(config['model'])
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model = checkpoint['model']
    else:
        model = eval(config['arch'])(config['model'])
    model.eval()
    model.summary()
    if gpu is not None:
        model = model.to(gpu)
    model.count_std=0
    model.dup_std=0

    gt_mask = 'create_mask' not in config['model'] #'mask' in config['model']['generator'] or 'Mask' in config['model']['generator']

    char_set_path = config['data_loader']['char_file']
    with open(char_set_path) as f:
        char_set = json.load(f)
    char_to_idx = char_set['char_to_idx']


    by_author_styles=defaultdict(list)
    by_author_all_ids=defaultdict(set)
    style_loc = config['style_loc'] if 'style_loc' in config else None
    if style_loc is not None:
        if style_loc[-1]!='*':
            style_loc+='*'
        all_style_files = glob(style_loc)
        assert( len(all_style_files)>0)
        for loc in all_style_files:
            #print('loading '+loc)
            with open(loc,'rb') as f:
                styles = pickle.load(f)
            if 'ids' in styles:
                for i in range(len(styles['authors'])):
                    by_author_styles[styles['authors'][i]].append((styles['styles'][i],styles['ids'][i]))
                    by_author_all_ids[styles['authors'][i]].update(styles['ids'][i])
            else:
                for i in range(len(styles['authors'])):
                    by_author_styles[styles['authors'][i]].append((styles['styles'][i],None))

        styles = defaultdict(list)
        authors=set()
        for author in by_author_styles:
            for style, ids in by_author_styles[author]:
                    styles[author].append(style)
            if len(styles[author])>0:
                authors.add(author)
        authors=list(authors)
    else:
        authors = valid_data_loader.dataset.authors
        styles = None

    num_char = config['model']['num_class']
    use_hwr_pred_for_style = config['trainer']['use_hwr_pred_for_style'] if 'use_hwr_pred_for_style' in config['trainer'] else False
    
    with torch.no_grad():
        while True:
            action = input('i/r/s/a/q? ')
            if action=='done' or action=='exit' or 'action'=='quit' or action=='q':
                exit()
            elif action =='a' or action=='authors':
                print(authors)
            elif action =='s' or action=='strech':
                index1=input("batch? ")
                if len(index1)>0:
                    index1=int(index1)
                else:
                    index1=0
                for i,instance1 in enumerate(valid_data_loader):
                    if i==index1:
                        break
                author1 = instance1['author'][0]
                mask = instance1['mask']
                if gpu is not None:
                    mask = mask.to(gpu)
                style1 = get_style(config,model,instance1,gpu)
                image = instance1['image']
                label = instance1['label']
                if gpu is not None:
                    image = image.to(gpu)
                    label = label.to(gpu)
                pred=model.hwr(image, None)
                if use_hwr_pred_for_style:
                    spaced_label = pred
                else:
                    spaced_label = model.correct_pred(pred,label)
                    spaced_label = model.onehot(spaced_label)
                images=interpolate_mask(model,style1, spaced_label, mask)
                for b in range(images[0].size(0)):
                    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    #filename = os.path.join(saveDir,'output{}.mp4'.format(b))
                    #out = cv2.VideoWriter(filename,fourcc, 2.0, (images[0].size(3),images[0].size(2)))
                    for i in range(len(images)):
                        genStep = ((1-images[i][b].permute(1,2,0))*127.5).cpu().numpy().astype(np.uint8)
                        #out.write(genStep)
                        path = os.path.join(saveDir,'gen{}_{}.png'.format(b,i))
                        #print('wrote: {}'.format(path))
                        cv2.imwrite(path,genStep)
            elif action[0]=='r': #randomly selected styles
                num_styles = int(input('number of styles? '))
                step = float(input('step (0.1 is normal)? '))
                text = input('text? ')
                stylesL=[]
                index = random.randint(0,20)
                last_author = None
                for i,instance in enumerate(valid_data_loader):
                    author = instance['author'][0]
                    if i>=index and author!=last_author:
                        print('i: {}, a: {}'.format(i,author))
                        image=instance['image'].to(gpu)
                        label=instance['label'].to(gpu)
                        a_batch_size = instance['a_batch_size']
                        style=model.extract_style(image,label,a_batch_size)[::a_batch_size]
                        stylesL.append(style)
                        last_author=author
                        index += random.randint(20,50)
                        print('next index: {}'.format(index))
                    if len(stylesL)>=num_styles:
                        break
                images=[]
                #step=0.05
                for i in range(num_styles-1):
                    images+=interpolate(model,stylesL[i],stylesL[i+1], text,char_to_idx,step)
                images+=interpolate(model,stylesL[-1],stylesL[0], text,char_to_idx,step)
                for b in range(images[0].size(0)):
                    for i in range(len(images)):
                        genStep = ((1-images[i][b].permute(1,2,0))*127.5).cpu().numpy().astype(np.uint8)
                        #out.write(genStep)
                        path = os.path.join(saveDir,'gen{}_{}.png'.format(b,i))
                        #print('wrote: {}'.format(path))
                        cv2.imwrite(path,genStep)
            else:#if action=='i' or action=='interpolate':
                if fromDataset and styles is None:
                    index1=input("batch? ")
                    if len(index1)>0:
                        index1=int(index1)
                    else:
                        index1=0
                    for i,instance1 in enumerate(valid_data_loader):
                        if i==index1:
                            break
                    author1 = instance1['author'][0]
                    print('author: {}'.format(author1))
                else:
                    author1=input("author? ")
                    if len(author1)==0:
                        author1=authors[0]
                if True: #new way
                    mask=None
                    index=input("batch? ")
                    text=input("text? ")
                    if len(index)>0:
                        index=int(index)
                    else:
                        index=0
                    for i,instance2 in enumerate(valid_data_loader):
                        if i==index:
                            break
                    author2 = instance2['author'][0]
                    print('author: {}'.format(author2))
                    image1 = instance1['image'].to(gpu)
                    label1 = instance1['label'].to(gpu)
                    image2 = instance2['image'].to(gpu)
                    label2 = instance2['label'].to(gpu)
                    a_batch_size = instance1['a_batch_size']
                    #spaced_label = correct_pred(pred,label)
                    #spaced_label = onehot(spaced_label,num_char)
                    if styles is not None:
                        style1 = styles[author1][0]
                        style2 = styles[author2][0]
                        style1=torch.from_numpy(style1)
                        style2=torch.from_numpy(style2)
                    else:
                        style1 = model.extract_style(image1,label1,a_batch_size)[::a_batch_size]
                        style2 = model.extract_style(image2,label2,a_batch_size)[::a_batch_size]
                    images=interpolate(model,style1,style2, text,char_to_idx)
                else: #old
                    index=input("batch? ")
                    if len(index)>0:
                        index=int(index)
                    else:
                        index=0
                    for i,instance in enumerate(valid_data_loader):
                        if i==index:
                            break
                    author2 = instance['author'][0]
                    print('author: {}'.format(author2))
                    text = instance['gt']
                    print (text)
                    image = instance['image'].to(gpu)
                    pred=model.hwr(image, None)
                    label = instance['label']
                    #spaced_label = correct_pred(pred,label)
                    #spaced_label = onehot(spaced_label,num_char)
                    if use_hwr_pred_for_style:
                        spaced_label = pred
                    else:
                        spaced_label = correct_pred(pred,label)
                        spaced_label = onehot(spaced_label)
                    if gt_mask:
                        mask = instance['mask']
                        if gpu is not None:
                            mask = mask.to(gpu)
                    else:
                        mask = None
                    if styles is not None:
                        style1 = styles[author1][0]
                        style2 = styles[author2][0]
                        style1=torch.from_numpy(style1)
                        style2=torch.from_numpy(style2)
                    else:
                        style1 = get_style(config,model,instance1,gpu)
                        style2 = get_style(config,model,instance,gpu)
                    images=interpolate_sp(model,style1,style2, spaced_label, mask,image.size())
                #compile images to gif?
                #import pdb;pdb.set_trace()
                #mask=(mask*255).cpu().permute(0,2,3,1).numpy().astype(np.uint8)
                if mask is not None:
                    mask = ((mask.cpu().permute(0,2,3,1)+1)/2.0).numpy()
                for b in range(images[0].size(0)):
                    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    #filename = os.path.join(saveDir,'output{}.mp4'.format(b))
                    #out = cv2.VideoWriter(filename,fourcc, 2.0, (images[0].size(3),images[0].size(2)))
                    for i in range(len(images)):
                        genStep = ((1-images[i][b].permute(1,2,0))*127.5).cpu().numpy().astype(np.uint8)
                        #out.write(genStep)
                        path = os.path.join(saveDir,'gen{}_{}.png'.format(b,i))
                        #print('wrote: {}'.format(path))
                        cv2.imwrite(path,genStep)
                    #out.release()
                    #print(' wrote: '+filename)
                    if mask is not None:
                        path_mask = os.path.join(saveDir,'mask{}.png'.format(b))
                        cv2.imwrite(path_mask,mask[b])



def interpolate(model,style1,style2,text,char_to_idx,step=0.05):
    batch_size = style1.size(0)
    label = string_utils.str2label_single(text, char_to_idx)
    label = torch.from_numpy(label.astype(np.int32))[:,None].expand(-1,batch_size).to(style1.device).long()
    label_len = torch.IntTensor(batch_size).fill_(len(text))
    results=[]
    for alpha in np.arange(0,1.0,step):
        style = style2*alpha+(1-alpha)*style1
        gen = model(label,label_len,style,flat=True)
        results.append(gen)
    return results
def interpolate_sp(model,style1,style2,spaced_label,mask=None, image_size=0):
    results=[]
    for alpha in np.arange(0,1.0,0.05):
        style = (style2*alpha+(1-alpha)*style1).view(1,-1).expand(spaced_label.size(1),-1)
        if mask is None:
            #gen = model.generator(spaced_label,style)
            model.top_and_bottom = model.create_mask(spaced_label,style)
            mask = model.write_mask(model.top_and_bottom,image_size).to(style1.device)
        #else:
        gen = model.generator(spaced_label,style, mask)
        results.append(gen)
    return results
def interpolate_mask(model,style,spaced_label,mask):
    results=[]
    style = style.view(1,-1).expand(spaced_label.size(1),-1)
    orig_mask = mask
    orig_spaced_label = spaced_label.permute(1,2,0)
    for strechH in np.arange(1,1.11,0.01):
        mask = F.interpolate(orig_mask,scale_factor=(1,strechH),mode='bilinear')
        spaced_label = F.interpolate(orig_spaced_label,scale_factor=strechH,mode='linear').permute(2,0,1)
        gen = model.generator(spaced_label,style, mask)
        results.append(gen)
    for strechV in np.arange(1,1.11,0.01):
        mask = F.interpolate(orig_mask,scale_factor=(strechV,1.1),mode='bilinear')
        gen = model.generator(spaced_label,style, mask)
        results.append(gen)
    for strechH in np.arange(1.1,0.89,-0.01):
        mask = F.interpolate(orig_mask,scale_factor=(1.1,strechH),mode='bilinear')
        spaced_label = F.interpolate(orig_spaced_label,scale_factor=strechH,mode='linear').permute(2,0,1)
        gen = model.generator(spaced_label,style, mask)
        results.append(gen)
    for strechV in np.arange(1.1,0.89,-0.01):
        mask = F.interpolate(orig_mask,scale_factor=(strechV,0.9),mode='bilinear')
        gen = model.generator(spaced_label,style, mask)
        results.append(gen)
    for strech in np.arange(0.9,1.01,0.01):
        mask = F.interpolate(orig_mask,scale_factor=(strech,strech),mode='bilinear')
        spaced_label = F.interpolate(orig_spaced_label,scale_factor=strech,mode='linear').permute(2,0,1)
        gen = model.generator(spaced_label,style, mask)
        results.append(gen)
    return results
def onehot(label,num_class):
    label_onehot = torch.zeros(label.size(0),label.size(1),num_class)
    #label_onehot[label]=1
    #TODO tensorize
    for i in range(label.size(0)):
        for j in range(label.size(1)):
            label_onehot[i,j,label[i,j]]=1
    return label_onehot.to(label.device)
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
    #fuzzy other neighbor preds
    #TODO

    return new_label.to(label.device)

if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='Interactive script to create interpolations (images)')
    parser.add_argument('-c', '--checkpoint', default=None, type=str,
                        help='path to training snapshot (default: None)')
    parser.add_argument('-d', '--savedir', default=None, type=str,
                        help='path to directory to save result images (default: None)')
    parser.add_argument('-g', '--gpu', default=None, type=int,
                        help='gpu number (default: cpu only)')
    parser.add_argument('-f', '--config', default=None, type=str,
                        help='config override')
    parser.add_argument('-a', '--addtoconfig', default=None, type=str,
                        help='Arbitrary key-value pairs to add to config of the form "k1=v1,k2=v2,...kn=vn"')

    args = parser.parse_args()

    addtoconfig=[]
    if args.addtoconfig is not None:
        split = args.addtoconfig.split(',')
        for kv in split:
            split2=kv.split('=')
            addtoconfig.append(split2)

    config = None
    if args.checkpoint is None and args.config is None:
        print('Must provide checkpoint (with -c)')
        exit()

    if args.gpu is not None:
        with torch.cuda.device(args.gpu):
            main(args.checkpoint, args.savedir, gpu=args.gpu,  config=args.config, addToConfig=addtoconfig)
    else:
        main(args.checkpoint, args.savedir, gpu=args.gpu,  config=args.config, addToConfig=addtoconfig)
