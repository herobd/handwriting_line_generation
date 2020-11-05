import json

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

from collections import defaultdict
import os
import cv2
import numpy as np
import scipy
import math

from utils import grid_distortion

from utils import string_utils, augmentation
from utils.parseIAM import getWordAndLineBoundaries as parseXML
import itertools
from utils.util import makeMask

import random
PADDING_CONSTANT = -1
def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)

def collate(batch):
    #if len(batch)==1:
    #    return batch[0]
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
        "label_lengths": [l for b in batch for l in b['label_lengths']],
        "gt": [l for b in batch for l in b['gt']],
        "author": [l for b in batch for l in b['author']],
        "id": [l for b in batch for l in b['id']],
        "name": [l for b in batch for l in b['name']],
        "a_batch_size": a_batch_size
    }

class StyleWordDataset(Dataset):
    def __init__(self, dirPath, split, config):

        self.img_height = config['img_height']
        self.a_batch_size = config['a_batch_size']
        num_style_per_word = config['num_style_per_word'] if 'num_style_per_word' in config else 10
        

        #with open(os.path.join(dirPath,'sets.json')) as f:
        with open(os.path.join('data','sets.json')) as f:
            set_list = json.load(f)[split]

        #self.authors = defaultdict(list)
        self.w_authors = defaultdict(list)
        for page_idx, name in enumerate(set_list):
            w_lines,lines,author = parseXML(os.path.join(dirPath,'xmls',name+'.xml'))
            
            #authorLines = len(self.authors[author])
            #self.authors[author] += [(os.path.join(dirPath,'forms',name+'.png'),)+l for l in lines]
            for words in w_lines:
                self.w_authors[author] += [(os.path.join(dirPath,'forms',name+'.png'),)+w for w in words]
            #self.lineIndex += [(author,i+authorLines) for i in range(len(lines))]
        #minLines=99999
        #for author,lines in self.authors.items():
            #print('{} {}'.format(author,len(lines)))
            #minLines = min(minLines,len(lines))
        #maxCombs = int(nCr(minLines,self.a_batch_size)*1.2)
        min_words = 999999
        for author,words in self.w_authors.items():
            min_words = min(min_words,len(words))
        self.a_batch_size = min(self.a_batch_size,min_words-1)
        npr = np.random.RandomState(123)
        self.index = []
        for author,words in self.w_authors.items():
            #if split=='train':
            #    combs=list(itertools.combinations(list(range(len(lines))),self.a_batch_size))
            #    np.random.shuffle(combs)
            #    self.lineIndex += [(author,c) for c in combs[:maxCombs]]
            #else:
            for wi in range(len(words)):
                less_words = [wj for wj in range(len(words)) if wj!=wi]
                choose = scipy.misc.comb(len(less_words),self.a_batch_size)
                for i in range(int(min(1+(choose-1)/2,num_style_per_word))):
                    indexes = npr.choice(len(less_words),self.a_batch_size)
                    selection = [less_words[i] for i in indexes]
                self.index.append((author,selection))

                

        char_set_path = config['char_file']
        with open(char_set_path) as f:
            char_set = json.load(f)
        self.char_to_idx = char_set['char_to_idx']
        self.augmentation = config['augmentation'] if 'augmentation' in config else None
        self.warning=False

        self.mask_post = config['mask_post'] if 'mask_post' in config else []

        #DEBUG
        if 'overfit' in config and config['overfit']:
            self.index = self.index[:10]


        



    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        inst = self.index[idx]
        author=inst[0]
        words=inst[1]
        batch=[]
        ids=[]
        for word in words:
            if word>=len(self.w_authors[author]):
                word = (word+37)%len(self.w_authors[author])
            img_path, lb, gt, id = self.w_authors[author][word]
            img = cv2.imread(img_path,0)[lb[0]:lb[1],lb[2]:lb[3]] #read as grayscale, crop word
            if img.shape[0]==0 or img.shape[1]==0:
                return self.__getitem__((idx+1)%self.__len__())

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
                "name": '{}_{}'.format(author,word),
                "author": author,
                "id": id
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
            #toPad = (dim1-b_img.shape[1])
            input_batch[i,:,0:b_img.shape[1],:] = b_img

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
            "id": [b['id'] for b in batch]
        }
