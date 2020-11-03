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
from utils.parseIAM import getLineBoundariesWithID as parseXML
from utils.util import makeMask
import itertools

import random
PADDING_CONSTANT = -1
def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)

def collate(batch):
    if len(batch)==1:
        return batch[0]
    batch = [b for b in batch if b is not None]
    a_batch_size = len(batch[0]['gt'])

    dim1 = batch[0]['image'].shape[1]
    dim3 = max([b['image'].shape[3] for b in batch])
    dim2 = batch[0]['image'].shape[2]


    max_label_len = max([b['label'].size(0) for b in batch])

    input_batch = torch.full((len(batch)*a_batch_size, dim1, dim2, dim3), PADDING_CONSTANT)
    mask_batch = torch.full((len(batch)*a_batch_size, dim1, dim2, dim3), PADDING_CONSTANT)
    labels_batch = torch.IntTensor(max_label_len,len(batch)*a_batch_size).fill_(0)
    for i in range(len(batch)):
        b_img = batch[i]['image']
        b_mask = batch[i]['mask']
        l = batch[i]['label']
        #toPad = (dim3-b_img.shape[3])
        input_batch[i*a_batch_size:(i+1)*a_batch_size,:,:,0:b_img.shape[3]] = b_img
        mask_batch[i*a_batch_size:(i+1)*a_batch_size,:,:,0:b_img.shape[3]] = b_mask
        labels_batch[0:l.size(0),i*a_batch_size:(i+1)*a_batch_size] = l


    return {
        "image": input_batch,
        "mask": mask_batch,
        "label": labels_batch,
        #"style": torch.cat([b['style'] for b in batch],dim=0),
        #"label_lengths": [l for b in batch for l in b['label_lengths']],
        "label_lengths": torch.cat([b['label_lengths'] for b in batch],dim=0),
        "gt": [l for b in batch for l in b['gt']],
        "author": [l for b in batch for l in b['author']],
        "name": [l for b in batch for l in b['name']],
        "a_batch_size": a_batch_size
    }
class MixedAuthorHWDataset(Dataset):
    def __init__(self, dirPath, split, config):

        self.img_height = config['img_height']
        self.batch_size = config['a_batch_size'] if 'a_batch_size' in config else 1
        #assert(config['batch_size']==1)

        #with open(os.path.join(dirPath,'sets.json')) as f:
        with open(os.path.join('data','lineMixed_sets.json')) as f:
            set_list = json.load(f)[split]

        self.authors = defaultdict(list)
        allLines=defaultdict(dict)
        self.lineIndex = []
        for line_idx, line_id in enumerate(set_list):
            page_id = line_id[:line_id.rfind('-')]
            if page_id not in allLines:
                lines,author = parseXML(os.path.join(dirPath,'xmls',page_id+'.xml'))
                for bounds,trans,id in lines:
                    allLines[page_id][id] = (bounds,trans)
            bounds,trans = allLines[page_id][line_id]
            self.authors[author].append((os.path.join(dirPath,'forms',page_id+'.png'), bounds, trans))
            #self.lineIndex += [(author,i+authorLines) for i in range(len(lines))]
        #minLines=99999
        #for author,lines in self.authors.items():
            #print('{} {}'.format(author,len(lines)))
            #minLines = min(minLines,len(lines))
        #maxCombs = int(nCr(minLines,self.batch_size)*1.2)
        short = config['short'] if 'short' in config else False
        for author,lines in self.authors.items():
            #if split=='train':
            #    combs=list(itertools.combinations(list(range(len(lines))),self.batch_size))
            #    np.random.shuffle(combs)
            #    self.lineIndex += [(author,c) for c in combs[:maxCombs]]
            #else:
            for i in range(len(lines)//self.batch_size):
                ls=[]
                for n in range(self.batch_size):
                    ls.append(self.batch_size*i+n)
                inst = (author,ls)
                self.lineIndex.append(inst)
                if short and i>=short:
                    break
            if short and i>=short:
                continue
            leftover = len(lines)%self.batch_size
            fill = self.batch_size-leftover
            last=[]
            for i in range(fill):
                last.append(i)
            for i in range(leftover):
                last.append(len(lines)-(1+i))
            self.lineIndex.append((author,last))

            #if split=='train':
            #    ss = set(self.lineIndex)
        #self.authors = self.authors.keys()
                

        char_set_path = config['char_file']
        with open(char_set_path) as f:
            char_set = json.load(f)
        self.char_to_idx = char_set['char_to_idx']
        self.augmentation = config['augmentation'] if 'augmentation' in config else None
        self.warning=False

        #DEBUG
        if 'overfit' in config and config['overfit']:
            self.lineIndex = self.lineIndex[:10]

        self.center = False #config['center_pad'] #if 'center_pad' in config else True

        self.mask_post = config['mask_post'] if 'mask_post' in config else []

    def __len__(self):
        return len(self.lineIndex)

    def __getitem__(self, idx):

        inst = self.lineIndex[idx]
        author=inst[0]
        lines=inst[1]
        batch=[]
        for line in lines:
            if line>=len(self.authors[author]):
                line = (line+37)%len(self.authors[author])
            img_path, lb, gt = self.authors[author][line]
            img = cv2.imread(img_path,0)[lb[0]:lb[1],lb[2]:lb[3]] #read as grayscale, crop line

            if img is None:
                return None

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
            img = 1.0 - img / 128.0


            if len(gt) == 0:
                return None
            gt_label = string_utils.str2label_single(gt, self.char_to_idx)


            batch.append( {
                "image": img,
                "gt": gt,
                "gt_label": gt_label,
                "name": '{}_{}'.format(author,line),
                "center": self.center,
                "author": author
            } )
        #batch = [b for b in batch if b is not None]
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
            "mask": makeMask(images,self.mask_post),
            "label": labels,
            "label_lengths": label_lengths,
            "gt": [b['gt'] for b in batch],
            "name": [b['name'] for b in batch],
            "author": [b['author'] for b in batch],
        }
