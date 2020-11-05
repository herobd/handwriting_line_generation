import json, pickle

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

from collections import defaultdict
from glob import iglob
import os
import cv2
import numpy as np
import math

from utils import grid_distortion

from utils import string_utils, augmentation
from utils.parseIAM import getLineBoundaries as parseXML
from utils.util import ensure_dir
#import pyexiv2
#import piexif

import random, pickle
PADDING_CONSTANT = -1

def collate(batch):
    batch = [b for b in batch if b is not None]
    #These all should be the same size or error
    assert len(set([b['image'].shape[0] for b in batch])) == 1
    assert len(set([b['image'].shape[2] for b in batch])) == 1

    dim0 = batch[0]['image'].shape[0]
    dim1 = max([b['image'].shape[1] for b in batch])
    dim2 = batch[0]['image'].shape[2]

    all_labels = []
    label_lengths = []

    input_batch = np.full((len(batch), dim0, dim1, dim2), PADDING_CONSTANT).astype(np.float32)
    for i in range(len(batch)):
        b_img = batch[i]['image']
        toPad = (dim1-b_img.shape[1])
        if 'center' in batch[0] and batch[0]['center']:
            toPad //=2
        else:
            toPad = 0
        input_batch[i,:,toPad:toPad+b_img.shape[1],:] = b_img

        l = batch[i]['gt_label']
        all_labels.append(l)
        label_lengths.append(len(l))

    #all_labels = np.concatenate(all_labels)
    label_lengths = torch.IntTensor(label_lengths)
    max_len = label_lengths.max()
    all_labels = [np.pad(l,((0,max_len-l.shape[0]),),'constant') for l in all_labels]
    all_labels = np.stack(all_labels,axis=1)


    images = input_batch.transpose([0,3,1,2])
    images = torch.from_numpy(images)
    labels = torch.from_numpy(all_labels.astype(np.int32))
    #label_lengths = torch.from_numpy(label_lengths.astype(np.int32))

    return {
        "image": images,
        "label": labels,
        "label_lengths": label_lengths,
        "gt": [b['gt'] for b in batch],
    }

class SynthHWDataset(Dataset):
    def __init__(self, dirPath, split, config):
        if split=='train':
            char_set_path = config['char_file']
            with open(char_set_path) as f:
                char_set = json.load(f)
            self.char_to_idx = char_set['char_to_idx']
            self.augmentation = config['augmentation'] if 'augmentation' in config else None
            textfile = config['text_file']
            max_len=config['max_len']
            with open(textfile) as f:
                text = f.readlines()
            self.text=[]
            max_len = config['max_len'] if 'max_len' in config else 20
            for line in text:
                line = line.strip()
                while len(line)>2*max_len:
                    self.text.append(line[:max_len])
                    line = line[max_len:]
                if len(line)>max_len:
                    mid = len(line)//2
                    self.text.append(line[:mid])
                    self.text.append(line[mid:])
                elif len(line)>0:
                    self.text.append(line)

            self.extrapolate = config['extrapolate'] if 'extrapolate' in config else None
            self.set_size = config['set_size']
            self.gen_batch_size = config['gen_batch_size']
            self.gen_batches = self.set_size//self.gen_batch_size
            self.set_size = self.gen_batch_size*self.gen_batches #ensure no mismatch of size
            self.use_before_refresh = config['use_before_refresh']
            self.used=-1

            self.authors_of_interest = config['authors_of_interest'] if 'authors_of_interest' in config else None

            if 'random_style' in config and config['random_style']:
                self.random_style = True
                self.style_dim = config['style_dim']
                self.directory = dirPath
            else:
                self.random_style = False
                style_loc = config['style_loc']
                
                if style_loc[-1]!='*':
                    style_loc+='*'
                styles=defaultdict(list)
                authors=None
                for loc in iglob(style_loc):
                    with open(loc,'rb') as f:
                        data = pickle.load(f)
                    s=data['styles']
                    if len(s.shape)==4:
                        s=s[:,:,0,0]

                    authors = list(data['authors'])
                    for i in range(len(authors)):
                        styles[authors[i]].append(s[i])
                assert(authors is not None)

                if self.authors_of_interest=='query':
                    print('authors in style set: {}'.format(', '.join(styles.keys())))
                    authors = input('target authors:')
                    authors = authors.split(',')
                    self.authors_of_interest = [author.strip() for author in authors]
                
                append = '_'+('_'.join(self.authors_of_interest)) if self.authors_of_interest is not None else ''
                self.directory = dirPath+append

                self.styles=[ l for author,l in styles.items() if self.authors_of_interest is None or author in self.authors_of_interest]
                self.num_authors = len(self.styles)
                assert(self.num_authors>0)

            ensure_dir(self.directory)
            self.gt_filename = os.path.join(self.directory,'gt.txt')

            self.labels = [None]*self.set_size

            self.init_size=0
            cur_files = list(os.listdir(self.directory))
            try:
                with open(self.gt_filename) as f:
                    labels =f.readlines()

                for i in range(min(self.set_size,len(labels))):
                    if '{}.png'.format(i) in cur_files:
                        self.init_size+=1
                        self.labels[i]=labels[i]
                    else:
                        break
                if self.init_size>0:
                    print('Found synth images 0-{} with labels'.format(self.init_size-1))
                    self.init_size = self.init_size//self.gen_batch_size
            except:
                self.init_size=0
            if self.init_size<self.gen_batches:
                self.used=self.use_before_refresh
        else:
            self.set_size=1


    def __len__(self):
        return self.set_size

    def __getitem__(self, idx):

        img_path = os.path.join(self.directory,'{}.png'.format(idx))
        img = cv2.imread(img_path,0)



        if len(img.shape)==2:
            img = img[...,None]
        if self.augmentation is not None:
            #img = augmentation.apply_random_color_rotation(img)
            if 'brightness' in self.augmentation:
                img = augmentation.apply_tensmeyer_brightness(img)
            if 'warp' in self.augmentation:
                img = grid_distortion.warp_image(img)
        if len(img.shape)==2:
            img = img[...,None]

        img = img.astype(np.float32)
        img = 1.0 - img / 128.0

        gt = self.labels[idx]
        if gt is None:
            #metadata = pyexiv2.ImageMetadata(img_path)
            #metadata.read()
            #metadata = piexif.load(img_path)
            #if 'gt' in metadata:
            #    gt = metadata['gt']
            #else:
            print('Error unknown label for image: {}'.format(img_path))
            return self.__getitem__((idx+7)%self.set_size)

        gt_label = string_utils.str2label_single(gt, self.char_to_idx)


        return {
            "image": img,
            "gt": gt,
            "gt_label": gt_label,
            #"author": author
        }

    def sample(self):
        #ri = np.random.choice(self.num_styles,[self.gen_batch_size,2],replace=False)
        #mix = np.random.random(self.gen_batch_size)
        #if self.extrapolate>0:
        #    mix = (2*self.extrapolate+1)*mix - self.extrapolate
        #style = self.styles[ri[:,0]]*mix + self.styles[ri[:,1]]*(1-mix)
        if self.random_style:
            style = torch.FloatTensor(self.gen_batch_size,self.style_dim).normal_()
        else:
            authors = np.random.choice(self.num_authors,[self.gen_batch_size,2],replace=True)
            mix = np.random.random(self.gen_batch_size)
            if self.extrapolate>0:
                mix = (2*self.extrapolate+1)*mix - self.extrapolate
            style = []
            for b in range(self.gen_batch_size):
                style0_i = np.random.choice(len(self.styles[authors[b,0]]))
                style1_i = np.random.choice(len(self.styles[authors[b,1]]))
                style0 = self.styles[authors[b,0]][style0_i]
                style1 = self.styles[authors[b,1]][style1_i]
                style.append(style0*mix[b] + style1*(1-mix[b]))
            style = np.stack(style,axis=0)
            style = torch.from_numpy(style).float()



        all_labels = []
        label_lengths = []
        gt=[]

        for i in range(self.gen_batch_size):
            idx = np.random.randint(0,len(self.text))
            text =  self.text[idx]
            gt.append(text)
            l = string_utils.str2label_single(text, self.char_to_idx)
            all_labels.append(l)
            label_lengths.append(len(l))

        #all_labels = np.concatenate(all_labels)
        label_lengths = torch.IntTensor(label_lengths)
        max_len = label_lengths.max()
        all_labels = [np.pad(l,((0,max_len-l.shape[0]),),'constant') for l in all_labels]
        all_labels = np.stack(all_labels,axis=1)

        label = torch.from_numpy(all_labels.astype(np.int32))

        return style,label,label_lengths,gt


    def save(self,i,images,gt):
        images = (255*(1-images.permute(0,2,3,1))/2).cpu().numpy().astype(np.uint8)
        #f = open(self.gt_filename,'a')
        with open(self.gt_filename,'a') as f:
            for b in range(self.gen_batch_size):
                idx = i*self.gen_batch_size + b
                filename = os.path.join(self.directory,'{}.png'.format(idx))
                cv2.imwrite(filename,images[b])
                self.labels[idx] = gt[b]
                #metadata = pyexiv2.ImageMetadata(filename)
                #metadata.read()
                #metadata.write()
                f.write(gt[b]+'\n')
        #f.close()

    def refresh_data(self,generator,gpu,logged=True):
        self.used+=1
        if self.used >= self.use_before_refresh:
            if self.init_size==0:
                with open(self.gt_filename,'w') as f:
                    f.write('') #erase or start the gt file
            if logged:
                print('refreshing sythetic')
            with torch.no_grad():
                for i in range(self.init_size,self.gen_batches):
                    style,label,label_lengths,gt = self.sample()
                    if gpu is not None:
                        style = style.to(gpu)
                        label = label.to(gpu)
                    if not logged:
                        print('refreshing sythetic: {}/{}'.format(i,self.gen_batches), end='\r')
                    generated = generator(label,label_lengths,style)
                    self.save(i,generated,gt)
            self.init_size=0
            self.used=0
