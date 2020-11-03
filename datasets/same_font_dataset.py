# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
import json

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

from collections import defaultdict
import os
import cv2
import numpy as np
import math

from utils import grid_distortion

from utils import string_utils, augmentation

from TextFlow import text_render

import random
PADDING_CONSTANT = -1

def collate(batch):
    return batch[0]

class SameFontDataset(Dataset):
    def __init__(self,dirPath, split, config):

        self.img_height = config['img_height']

        fontfile = config['fontfile'] if 'fontfile' in config else 'mono_fonts.txt'
        textfile = 'data/lotr.txt'
        self.batch_size = config['a_batch_size']
        assert(config['batch_size']==1)

        #with open(os.path.join(dirPath,'sets.json')) as f:
        with open(textfile) as f:
            text = f.readlines()
        self.text=[]
        for line in text:
            line = line.strip()
            if len(line)>15:
                mid = len(line)//2
                self.text.append(line[:mid])
                self.text.append(line[mid:])
            elif len(line)>0:
                self.text.append(line)

        with open(os.path.join(dirPath,fontfile)) as f:
            self.fonts = [os.path.join(dirPath,line.strip()) for line in f]

        splitC = 30
        if split=='train':
            self.text=self.text[:-splitC]
        else:
            self.text=self.text[-splitC:]
        self.split=split

        char_set_path = config['char_file'] if 'char_file' in config else 'data/lotr_char_set.json'
        with open(char_set_path) as f:
            char_set = json.load(f)
        self.char_to_idx = char_set['char_to_idx']
        self.augmentation = config['augmentation'] if 'augmentation' in config else None
        self.warning=False

        #DEBUG
        if 'overfit' in config and config['overfit']:
            if split=='train':
                self.text = self.text[50:60]
            else:
                self.text = self.text[:10]
            self.fonts = self.fonts[:2]

        self.center = config['center_pad'] #if 'center_pad' in config else False



    def __len__(self):
        return len(self.text)*len(self.fonts)

    def __getitem__(self, idx):

        font_idx = idx%len(self.fonts)
        font = self.fonts[font_idx]
        text_idx = idx//len(self.fonts)
        text = self.text[text_idx]

        batch=[]
        for b in range(self.batch_size):
            if b>0:
                #if self.split=='train':
                #    text_idx = np,random.randint(0,len(self.fonts))
                #else:
                text_idx=(text_idx+41*b)%len(self.text)
                text = self.text[text_idx]


            data = text_render(text, font, bg=0, interval=.8, height=self.img_height+10, params=None)
            img = data['img']
            gt=text

            #TODO pad for font size diff

            if img.shape[0] != self.img_height:
                if img.shape[0] < self.img_height and not self.warning:
                    self.warning = True
                    print("WARNING: upsampling image to fit size")
                percent = float(self.img_height) / img.shape[0]
                img = cv2.resize(img, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)

            if img is None:
                return None

            if len(img.shape)==2:
                img = img[...,None]
            if self.augmentation is not None:
                #img = augmentation.apply_random_color_rotation(img)
                img = augmentation.apply_tensmeyer_brightness(img)
                img = grid_distortion.warp_image(img)

            img = img.astype(np.float32)
            img = img / 128.0 - 1.0


            if len(gt) == 0:
                return None
            gt_label = string_utils.str2label_single(gt, self.char_to_idx)


            batch.append( {
                "image": img,
                "gt": gt,
                "gt_label": gt_label,
                "name": '{}_{}'.format(font_idx,text_idx),
                "center": self.center,
                "author": font_idx
            })
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
            "name": [b['name'] for b in batch],
            "author": [b['author'] for b in batch]
        }
