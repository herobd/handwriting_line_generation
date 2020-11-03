# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
import random
import json
import torch
import numpy as np
from utils import string_utils
class TextData():
    def __init__(self, textfile = 'data/lotr.txt', char_set_path='', batch_size=1, max_len=20):


        #with open(os.path.join(dirPath,'sets.json')) as f:
        with open(textfile) as f:
            text = f.readlines()
        self.text=[]
        for line in text:
            line = line.strip()
            if len(line)>max_len:
                mid = len(line)//2
                self.text.append(line[:mid])
                self.text.append(line[mid:])
            elif len(line)>0:
                self.text.append(line)

        with open(char_set_path) as f: 
            char_set = json.load(f) 
        self.char_to_idx = char_set['char_to_idx']

        self.batch_size = batch_size

    def getInstance(self):

        all_labels = []
        label_lengths = []
        gt=[]

        for i in range(self.batch_size):
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

        return {'label': torch.from_numpy(all_labels.astype(np.int32)),
                'label_lengths': label_lengths,
                'gt': gt,
                'image': None
                }
