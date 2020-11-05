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
from utils.util import ensure_dir
from utils import string_utils, augmentation, normalize_line
from utils.parseIAM import getLineBoundaries as parseXML

import random
PADDING_CONSTANT = 0

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
        "name": [b['name'] for b in batch],
        "a_batch_size": 1,
        "spaced_label": None,
        "center_line": None
    }

class MMDHWDataset(Dataset):
    def __init__(self, dirPath, split, config):

        self.img_height = config['img_height']
        self.dirPath = dirPath
        clean = config['cleaned'] if 'cleaned' in config else False
        if type(clean)==bool:
            clean = 'clean_' if clean else ''
        else:
            clean += '_'
        self.hide_labels = config['hide_labels'] if 'hide_labels' in config else False

        #with open(os.path.join(dirPath,'sets.json')) as f:
        with open(os.path.join(dirPath,'{}train_valid_split.json'.format(clean))) as f:
            self.set_list = json.load(f)[split]
        self.set_list = [inst for inst in self.set_list if '60383' not in inst['img_path']]
        self.max_char_len = max([len(inst['gt']) for inst in self.set_list])


        char_set_path = config['char_file']
        with open(char_set_path) as f:
            char_set = json.load(f)
        self.char_to_idx = char_set['char_to_idx']

        self.augmentation = config['augmentation'] if 'augmentation' in config else None
        self.normalized_dir = config['cache_normalized'] if 'cache_normalized' in config else None
        if self.normalized_dir is not None:
            ensure_dir(self.normalized_dir)

        self.warning=False

        #DEBUG
        if 'overfit' in config and config['overfit']:
            self.lineIndex = self.lineIndex[:10]

        self.center = config['center_pad'] if 'center_pad' in config else False

        self.add_spaces = config['add_spaces'] if 'add_spces' in config else False

    def max_len(self):
        return self.max_char_len

    def __len__(self):
        return len(self.set_list)

    def __getitem__(self, idx):

        inst = self.set_list[idx]
        img_path = inst['img_path']
        gt = inst['gt']
        if self.hide_labels:
            gt='$UNKNOWN$'
        elif self.add_spaces:
            gt = ' '+gt+' '
        img = cv2.imread(os.path.join(self.dirPath,'training',img_path),0)

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
        
        if type(self.augmentation) is str and 'clip' in self.augmentation:
            img = augmentation.mmd_crop(img)
        if len(img.shape)==2:
            img = img[...,None]
        if type(self.augmentation) is str and 'normalization' in  self.augmentation and not readNorm:
            img = normalize_line.deskew(img)
            img = normalize_line.skeletonize(img)
            if self.normalized_dir is not None:
                cv2.imwrite(os.path.join(self.normalized_dir,'{}_{}.png'.format(author,line)),img)
        elif self.augmentation is not None and (type(self.augmentation) is not str or 'warp' in self.augmentation):
            #img = augmentation.apply_random_color_rotation(img)
            if type(self.augmentation) is str and "low" in self.augmentation:
                if random.random()>0.1:
                    img = augmentation.apply_tensmeyer_brightness(img)
                if random.random()>0.01:
                    img = grid_distortion.warp_image(img,w_mesh_std=0.7,h_mesh_std=0.7)
            else:
                img = augmentation.apply_tensmeyer_brightness(img)
                img = grid_distortion.warp_image(img)
        if type(self.augmentation) is str  and 'lines' in self.augmentation:
            img = augmentation.add_random_lines(img)
        if type(self.augmentation) is str and 'crop' in self.augmentation:
            img = augmentation.bad_crop(img)
        #assert(img.shape[0]==32)
        if len(img.shape)==2:
            img = img[...,None]

        img = img.astype(np.float32)
        img = 1.0 - img / 128.0


        if len(gt) == 0:
            return None
        gt_label = string_utils.str2label_single(gt, self.char_to_idx)


        return {
            "image": img,
            "gt": gt,
            "gt_label": gt_label,
            "name": img_path,
            "center": self.center
        }
