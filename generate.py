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

def getImage(filename,img_height=64):
    img = cv2.imread(filename,0)

    if img is None:
        return None

    if img.shape[0] != img_height:
        if img.shape[0] < img_height:
            print("WARNING: upsampling image to fit size")
        percent = float(img_height) / img.shape[0]
        #if img.shape[1]*percent > self.max_width:
        #    percent = self.max_width/img.shape[1]
        img = cv2.resize(img, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)
        if img.shape[0]<img_height:
            diff = img_height-img.shape[0]
            img = np.pad(img,((diff//2,diff//2+diff%2),(0,0)),'constant',constant_values=255)

    if img is None:
        return None

    if len(img.shape)==2:
        img = img[...,None]

    img = img.astype(np.float32)
    img = 1.0 - img / 128.0

    img = torch.from_numpy(img).permute(2,0,1)
    return img

def main(resume,todo_list,gpu=None,config=None,addToConfig=None):
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


    char_set_path = config['data_loader']['char_file']
    with open(char_set_path) as f:
        char_set = json.load(f)
    char_to_idx = char_set['char_to_idx']

    ###
    with torch.no_grad():
        for todo in todo_list:
            if 'interpolate' in todo and todo['interpolate']: 
                #interpolation
                assert(len(todo['text'])==1 and len(todo['out'])==1)

                step = todo['interpolate'] if type(todo['interpolate']) is float else 0.1
                #get style
                style_images_a=[]
                max_len=0
                for image_name in todo['style']:
                    style_images_a.append(getImage(image_name))
                    #print(style_images[-1].size())
                    max_len = max(style_images_a[-1].size(2),max_len)
                style_images = torch.FloatTensor(len(style_images_a),style_images_a[0].size(0),style_images_a[0].size(1),max_len).fill_(-1)
                for b,s_i in enumerate(style_images_a):
                    style_images[b,:,:,:s_i.size(2)]=s_i
                if gpu is not None:
                    style_images = style_images.to(gpu)
                pred = model.hwr(style_images, None)
                spaced_label = pred.permute(1,2,0)
                styles = model.style_extractor(style_images,spaced_label)

                text = todo['text'][0]
                label = string_utils.str2label_single(text, char_to_idx)
                label = torch.from_numpy(label.astype(np.int32))
                label = label[:,None].to(styles.device).long()
                label_len = [len(text)]

                generated=[]
                for i in range(styles.size(0)-1):
                    styleA = styles[i:i+1]
                    styleB = styles[i+1:i+2]
                    for alpha in np.arange(0,1,step):
                        style = (1-alpha)*styleA + alpha*styleB
                        gen = model(label,label_len,style,flat=True)[0]
                        generated.append( ((1-gen.permute(1,2,0))*127.5).cpu().numpy().astype(np.uint8) )
                styleA = styles[-1][None,...]
                styleB = styles[0:1]
                for alpha in np.arange(0,1,step):
                    style = (1-alpha)*styleA + alpha*styleB
                    gen = model(label,label_len,style,flat=True)[0]
                    generated.append( ((1-gen.permute(1,2,0))*127.5).cpu().numpy().astype(np.uint8) )

                outname = todo['out'][0]
                dotpos = outname.rfind('.')
                if dotpos>=0:
                    outname = outname[:dotpos]+'{}'+outname[dotpos:]
                else:
                    outname += '{}'

                for i,image in enumerate(generated):
                    cv2.imwrite(outname.format(i),image)

            else:
                #normal generation

                #get style
                style_images=[]
                for image_name in todo['style']:
                    style_images.append(getImage(image_name))
                    #print(style_images[-1].size())
                style_images = torch.cat(style_images,dim=2)
                style_images = style_images[None,...] #add batch dim
                if gpu is not None:
                    style_images = style_images.to(gpu)
                pred = model.hwr(style_images, None)
                spaced_label = pred.permute(1,2,0)
                style = model.style_extractor(style_images,spaced_label)
                
                #prepare text
                label_len = [len(text) for text in todo['text']]#torch.IntTensor(batch_size).fill_(len(text))
                max_len = max(label_len)
                batch_size = len(todo['text'])
                labels = torch.FloatTensor(max_len,batch_size).fill_(0)
                for b,text in enumerate(todo['text']):
                    label = string_utils.str2label_single(text, char_to_idx)
                    label = torch.from_numpy(label.astype(np.int32))
                    labels[:label.size(0),b] = label
                labels = labels.to(style.device).long()

                #generate!
                style=style.expand(batch_size,-1)
                gen = model(labels,label_len,style,flat=True)
            
                #save
                for b,outname in enumerate(todo['out']):
                    image = ((1-gen[b].permute(1,2,0))*127.5).cpu().numpy().astype(np.uint8)
                    cv2.imwrite(outname,image)
        ###
if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='Script to run generator')
    parser.add_argument('-c', '--checkpoint', default=None, type=str,
                        help='path to training snapshot (default: None)')
    parser.add_argument('-g', '--gpu', default=None, type=int,
                        help='gpu number (default: cpu only)')
    parser.add_argument('-f', '--config', default=None, type=str,
                        help='config override')
    parser.add_argument('-a', '--addtoconfig', default=None, type=str,
                        help='Arbitrary key-value pairs to add to config of the form "k1=v1,k2=v2,...kn=vn"')
    parser.add_argument('-s', '--style', default=None, type=str,
                        help='images specifying style, space seperated. Ignored it -l used.')
    parser.add_argument('-t', '--text', default=None, type=str,
                        help='text to generate. Ignored if -l used.')
    parser.add_argument('-o', '--out',default=None,  type=str,
                        help='out file name. Ignored if -l used')
    parser.add_argument('-l', '--list', default=None, type=str,
            help='json file specifying list of style images and texts to generate. List of objects each with three lists: style, text, out')

    args = parser.parse_args()

    if args.list is not None:
        with open(args.list) as f:
            todo_list = json.load(f)
    else:
        only = {'style': args.style.split(' '),
                'text': [args.text],
                'out': [args.out]
                }
        todo_list=[only]

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
            main(args.checkpoint, todo_list, gpu=args.gpu,  config=args.config, addToConfig=addtoconfig)
    else:
        main(args.checkpoint, todo_list, gpu=args.gpu,  config=args.config, addToConfig=addtoconfig)
