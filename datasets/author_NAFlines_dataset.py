import json

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

from collections import defaultdict
import os
import cv2
import numpy as np
import math, re

from utils import grid_distortion
from utils.util import ensure_dir

from utils import string_utils, augmentation
from utils.parseIAM import getLineBoundaries as parseXML
from utils.util import makeMask
import itertools, pickle

import random
PADDING_CONSTANT = -1
def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)

def collate(batch):
    if len(batch)==1:
        batch[0]['a_batch_size']=batch[0]['image'].size(0)
        return batch[0]
    batch = [b for b in batch if b is not None]
    a_batch_size = len(batch[0]['gt'])

    dim1 = batch[0]['image'].shape[1]
    dim3 = max([b['image'].shape[3] for b in batch])
    dim2 = batch[0]['image'].shape[2]


    max_label_len = max([b['label'].size(0) for b in batch])
    if batch[0]['spaced_label'] is not None:
        max_spaced_label_len = max([b['spaced_label'].size(0) for b in batch])
    else:
        max_spaced_label_len = None

    input_batch = torch.full((len(batch)*a_batch_size, dim1, dim2, dim3), PADDING_CONSTANT)
    mask_batch = torch.full((len(batch)*a_batch_size, dim1, dim2, dim3), PADDING_CONSTANT)
    if 'fg_mask' in batch[0]:
        fg_masks = torch.full((len(batch)*a_batch_size, 1, dim2, dim3), 0)
    top_and_bottom_batch = torch.full((len(batch)*a_batch_size,2,dim3), 0)
    center_line_batch = torch.full((len(batch)*a_batch_size,dim3), dim2/2)
    labels_batch = torch.IntTensor(max_label_len,len(batch)*a_batch_size).fill_(0)
    if max_spaced_label_len is not None:
        spaced_labels_batch = torch.IntTensor(max_spaced_label_len,len(batch)*a_batch_size).fill_(0)
    else:
        spaced_labels_batch = None

    for i in range(len(batch)):
        b_img = batch[i]['image']
        b_mask = batch[i]['mask']
        b_top_and_bottom = batch[i]['top_and_bottom']
        b_center_line = batch[i]['center_line']
        l = batch[i]['label']
        #toPad = (dim3-b_img.shape[3])
        input_batch[i*a_batch_size:(i+1)*a_batch_size,:,:,0:b_img.shape[3]] = b_img
        mask_batch[i*a_batch_size:(i+1)*a_batch_size,:,:,0:b_img.shape[3]] = b_mask
        if 'fg_mask' in batch[i]:
                fg_masks[i*a_batch_size:(i+1)*a_batch_size,:,:,0:b_img.shape[3]] = batch[i]['fg_mask']
        if b_top_and_bottom is not None:
            top_and_bottom_batch[i*a_batch_size:(i+1)*a_batch_size,:,0:b_img.shape[3]] = b_top_and_bottom
        else:
            top_and_bottom_batch=None
        if b_center_line is not None:
            center_line_batch[i*a_batch_size:(i+1)*a_batch_size,0:b_img.shape[3]] = b_center_line
        else:
            center_line_batch=None
        labels_batch[0:l.size(0),i*a_batch_size:(i+1)*a_batch_size] = l
        if max_spaced_label_len is not None:
            sl = batch[i]['spaced_label']
            spaced_labels_batch[0:sl.size(0),i*a_batch_size:(i+1)*a_batch_size] = sl


    if batch[0]['style'] is None:
        style=None
    else:
        style=torch.cat([b['style'] for b in batch],dim=0)

    toRet= {
        "image": input_batch,
        "mask": mask_batch,
        "top_and_bottom": top_and_bottom_batch,
        "center_line": center_line_batch,
        "label": labels_batch,
        "style": style,
        #"style": torch.cat([b['style'] for b in batch],dim=0),
        #"label_lengths": [l for b in batch for l in b['label_lengths']],
        "label_lengths": torch.cat([b['label_lengths'] for b in batch],dim=0),
        "gt": [l for b in batch for l in b['gt']],
        "spaced_label": spaced_labels_batch,
        "author": [l for b in batch for l in b['author']],
        "name": [l for b in batch for l in b['name']],
        "a_batch_size": a_batch_size
    }
    if 'fg_mask' in batch[0]:
        toRet['fg_mask']=fg_masks
    return toRet
class AuthorNAFlinesDataset(Dataset):
    def __init__(self, dirPath, split, config):
        if 'split' in config:
            split = config['split']
        if split!='train':
            dirPath=os.path.join(dirPath,split)
        subdir = split

        self.img_height = config['img_height']
        self.batch_size = 1#config['a_batch_size']
        self.max_width = config['max_width'] if  'max_width' in config else 1000
        #assert(config['batch_size']==1)
        wtype = config['type']

        self.pad_ends = config['pad_ends'] if 'pad_ends' in config else False


        #with open(os.path.join(dirPath,'sets.json')) as f:
        gt_file = os.path.join(dirPath,'{}.txt'.format(wtype if len(wtype)>0 else 'text'))

        with open(gt_file) as f:
            gt_list = f.readlines()
        self.image_list = [(os.path.join(dirPath,wtype,a[:a.index('|')]),a[a.index('|')+1:].strip()) for a in gt_list]
        self.max_char_len=max([len(p[1]) for p in self.image_list])

        self.fg_masks_dir = config['fg_masks_dir'] if 'fg_masks_dir' in config else None
        self.warning=False

        if self.fg_masks_dir is not None:
            if self.fg_masks_dir[-1]=='/':
                self.fg_masks_dir = self.fg_masks_dir[:-1]
            self.fg_masks_dir+='_{}'.format(self.max_width)
            ensure_dir(self.fg_masks_dir)
            for img_path,gt in self.image_list:
            
                fg_path = os.path.join(self.fg_masks_dir,'{}_{}.png'.format(author,line))
                if not os.path.exists(fg_path):
                    img = cv2.imread(img_path,0)#read as grayscale
                    if img is None:
                        continue

                    if pad_above<0:
                        img = img[-pad_above:,:]
                        pad_above =0
                    if pad_below<0:
                        img = img[:pad_below,:]
                        pad_below =0
                    #if pad_above>0 or pad_below>0:
                    img = img = np.pad(img,((pad_above,pad_below),(10,10)),'constant',constant_values=255)

                    if img.shape[0] != self.img_height:
                        if img.shape[0] < self.img_height and not self.warning:
                            self.warning = True
                            print("WARNING: upsampling image to fit size")
                        percent = float(self.img_height) / img.shape[0]
                        if img.shape[1]*percent > self.max_width:
                            percent = self.max_width/img.shape[1]
                        img = cv2.resize(img, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)
                        if img.shape[0]<self.img_height:
                            diff = self.img_height-img.shape[0]
                            img = np.pad(img,((diff//2,diff//2+diff%2),(0,0)),'constant',constant_values=255)

                    th,binarized = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    binarized = 255-binarized
                    ele = cv2.getStructuringElement(  cv2.MORPH_ELLIPSE, (9,9) )
                    binarized = cv2.dilate(binarized,ele)
                    cv2.imwrite(fg_path,binarized)
                    print('saved fg mask: {}'.format(fg_path))


        char_set_path = config['char_file']
        with open(char_set_path) as f:
            char_set = json.load(f)
        self.char_to_idx = char_set['char_to_idx']
        self.augmentation = config['augmentation'] if 'augmentation' in config else None

        #DEBUG
        if 'overfit' in config and config['overfit']:
            self.lineIndex = self.lineIndex[:10]

        self.center = False #config['center_pad'] #if 'center_pad' in config else True

        if 'style_loc' in config:
            by_author_styles=defaultdict(list)
            by_author_all_ids=defaultdict(set)
            style_loc = config['style_loc']
            if style_loc[-1]!='*':
                style_loc+='*'
            all_style_files = glob(style_loc)
            assert( len(all_style_files)>0)
            for loc in all_style_files:
                #print('loading '+loc)
                with open(loc,'rb') as f:
                    styles = pickle.load(f)
                for i in range(len(styles['authors'])):
                    by_author_styles[styles['authors'][i]].append((styles['styles'][i],styles['ids'][i]))
                    by_author_all_ids[styles['authors'][i]].update(styles['ids'][i])

            self.styles = defaultdict(lambda: defaultdict(list))
            for author in by_author_styles:
                for id in by_author_all_ids[author]:
                    for style, ids in by_author_styles[author]:
                        if id not in ids:
                            self.styles[author][id].append(style)

            for author in self.authors:
                assert(author in self.styles)
        else:
            self.styles=None

        if 'spaced_loc' in config:
            with open(config['spaced_loc'],'rb') as f:
                self.spaced_by_name = pickle.load(f)
            #for name,v in spaced_by_name.items():
            #    author, id = name.split('_')
        else:
            self.spaced_by_name = None

        self.mask_post = config['mask_post'] if 'mask_post' in config else []
        self.mask_random = config['mask_random'] if 'mask_random' in config else False

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        img_path, gt = self.image_list[idx]
        img = 255-cv2.imread(img_path,0)#read as grayscale
        if img is None:
            return None
        if img.shape[1]<25 and img.shape[0]>56:
            img=cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)

        if img.shape[0]< self.img_height:
            pad = self.img_height-img.shape[0]
            img = np.pad(img,((pad//2+pad%2,pad//2),(0,0)),'constant',constant_values=255)

        if self.pad_ends:
            to_pad = self.img_height//2
            img = np.pad(img,((0,0),(to_pad,to_pad)),'constant',constant_values=255)


        if img.shape[0] != self.img_height:
            assert( img.shape[0] > self.img_height)
            percent = float(self.img_height) / img.shape[0]
            #if img.shape[1]*percent > self.max_width:
            #    percent = self.max_width/img.shape[1]
            img = cv2.resize(img, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)
            #if img.shape[0]<self.img_height:
            #    diff = self.img_height-img.shape[0]
            #    img = np.pad(img,((diff//2,diff//2+diff%2),(0,0)),'constant',constant_values=255)
        if 'UNKNOWN' in gt and img.shape[1]>self.max_width:
            diff = img.shape[1]-self.max_width
            start = random.randint(0,diff-1)
            img = img[:,start:start+self.max_width]


        if len(img.shape)==2:
            img = img[...,None]
        if self.fg_masks_dir is not None:
            fg_path = os.path.join(self.fg_masks_dir,'{}_{}.png'.format(author,line))
            fg_mask = cv2.imread(fg_path,0)
            fg_mask = fg_mask/255
            if fg_mask.shape!=img[:,:,0].shape:
                print('Error, fg_mask ({}, {}) not the same size as image ({})'.format(fg_path,fg_mask.shape,img[:,:,0].shape))
                th,fg_mask = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                fg_mask = 255-fg_mask
                ele = cv2.getStructuringElement(  cv2.MORPH_ELLIPSE, (9,9) )
                fg_mask = cv2.dilate(fg_mask,ele)
                fg_mask = fg_mask/255

        if self.augmentation is not None:
            #img = augmentation.apply_random_color_rotation(img)
            img = augmentation.apply_tensmeyer_brightness(img)
            img = grid_distortion.warp_image(img)
            if len(img.shape)==2:
                img = img[...,None]

        img = img.astype(np.float32)
        img = 1.0 - img / 128.0
        #img = (img / 128.0)-1


        #if len(gt) == 0:
        #    gt_label=None
        #else:
        gt_label = string_utils.str2label_single(gt, self.char_to_idx)

        if self.styles:
            style_i = self.npr.choice(len(self.styles[author][id]))
            style = self.styles[author][id][style_i]
        else:
            style=None
        name = img_path[img_path.rfind('/')+1:img_path.rfind('.')]
        spaced_label = None if self.spaced_by_name is None else self.spaced_by_name[img_path]
        if spaced_label is not None:
            assert(spaced_label.shape[1]==1)
        toAppend = {
            "image": img,
            "gt": gt,
            "style": style,
            "gt_label": gt_label,
            "spaced_label": spaced_label,
            "name": name,
            "center": self.center,
            "author": 'unknown'
        }
        if self.fg_masks_dir is not None:
            toAppend['fg_mask'] = fg_mask
        batch=[toAppend]

        #vvv This is all reduntend, but left in for developement speed vvv
        #batch = [b for b in batch if b is not None]
        #These all should be the same size or error
        assert len(set([b['image'].shape[0] for b in batch])) == 1
        assert len(set([b['image'].shape[2] for b in batch])) == 1

        dim0 = batch[0]['image'].shape[0]
        dim1 = max([b['image'].shape[1] for b in batch])
        dim2 = batch[0]['image'].shape[2]

        all_labels = []
        label_lengths = []
        if self.spaced_by_name is not None:
            spaced_labels = []
        else:
            spaced_labels = None
        max_spaced_len=0

        input_batch = np.full((len(batch), dim0, dim1, dim2), PADDING_CONSTANT).astype(np.float32)
        if self.fg_masks_dir is not None:
            fg_masks = np.full((len(batch), dim0, dim1, 1), 0).astype(np.float32)
        for i in range(len(batch)):
            b_img = batch[i]['image']
            toPad = (dim1-b_img.shape[1])
            if 'center' in batch[0] and batch[0]['center']:
                toPad //=2
            else:
                toPad = 0
            input_batch[i,:,toPad:toPad+b_img.shape[1],:] = b_img
            if self.fg_masks_dir is not None:
                fg_masks[i,:,toPad:toPad+b_img.shape[1],0] = batch[i]['fg_mask']

            l = batch[i]['gt_label']
            all_labels.append(l)
            label_lengths.append(len(l))

            if spaced_labels is not None:
                sl = batch[i]['spaced_label']
                spaced_labels.append(sl)
                max_spaced_len = max(max_spaced_len,sl.shape[0])

        #all_labels = np.concatenate(all_labels)
        label_lengths = torch.IntTensor(label_lengths)
        max_len = label_lengths.max()
        all_labels = [np.pad(l,((0,max_len-l.shape[0]),),'constant') for l in all_labels]
        all_labels = np.stack(all_labels,axis=1)
        if self.spaced_by_name is not None:
            spaced_labels = [np.pad(l,((0,max_spaced_len-l.shape[0]),(0,0)),'constant') for l in spaced_labels]
            ddd = spaced_labels
            spaced_labels = np.concatenate(spaced_labels,axis=1)
            spaced_labels = torch.from_numpy(spaced_labels)
            assert(spaced_labels.size(1) == len(batch))


        images = input_batch.transpose([0,3,1,2])
        images = torch.from_numpy(images)
        labels = torch.from_numpy(all_labels.astype(np.int32))
        #label_lengths = torch.from_numpy(label_lengths.astype(np.int32))
        if self.fg_masks_dir is not None:
            fg_masks = fg_masks.transpose([0,3,1,2])
            fg_masks = torch.from_numpy(fg_masks)
        
        if batch[0]['style'] is not None:
            styles = np.stack([b['style'] for b in batch], axis=0)
            styles = torch.from_numpy(styles).float()
        else:
            styles=None
        mask, top_and_bottom, center_line = makeMask(images,self.mask_post, self.mask_random)
        ##DEBUG
        #for i in range(5):
        #    mask2, top_and_bottom2 = makeMask(images,self.mask_post, self.mask_random)
        #    #extra_masks.append(mask2)
        #    mask2 = ((mask2[0,0]+1)/2).numpy().astype(np.uint8)*255
        #    cv2.imshow('mask{}'.format(i),mask2)
        #mask = ((mask[0,0]+1)/2).numpy().astype(np.uint8)*255
        #cv2.imshow('mask'.format(i),mask)
        #cv2.waitKey()
        toRet = {
            "image": images,
            "mask": mask,
            "top_and_bottom": top_and_bottom,
            "center_line": center_line,
            "label": labels,
            "style": styles,
            "label_lengths": label_lengths,
            "gt": [b['gt'] for b in batch],
            "spaced_label": spaced_labels,
            "name": [b['name'] for b in batch],
            "author": [b['author'] for b in batch],
        }
        if self.fg_masks_dir is not None:
            toRet['fg_mask'] = fg_masks
        return toRet

    def max_len(self):
        return self.max_char_len
