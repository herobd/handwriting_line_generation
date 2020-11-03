# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
import torch.utils.data
import numpy as np
import math
import torch

def collate(batch):
    maxlen = max([b['label'].size(0) for b in batch])
    label=torch.zeros(maxlen,len(batch)).long()
    for i,b in enumerate(batch):
        label[0:b['label'].size(0),i]=b['label']
    return {'input': torch.stack([b['input'] for b in batch],dim=1),
            'label': label,
            'style': torch.stack([b['style'] for b in batch],dim=0)
                }

class SpacingDataset(torch.utils.data.Dataset):

    def __init__(self,config,dirPath=None,split=None):
        self.chars = ['a','b','c','d','e']
        self.style_dim=len(self.chars)**2

    def __len__(self):
        return 100

    def __getitem__(self,index):
        #rule = np.random.randint(numrules)
        seqlen = 10#np.random.randint(10)
        sequence = np.random.choice(len(self.chars),seqlen)
        style = torch.FloatTensor(len(self.chars),len(self.chars)).normal_()

        label = [0]*round(style[sequence[0],sequence[0]].item()*2+3)
        for i in range(seqlen-1):
            label.append(sequence[i]+1)
            label += [0]*round(style[sequence[i],sequence[i+1]].item()*2+3)
        label.append(sequence[-1]+1)
        label += [0]*round(style[sequence[-1],sequence[-1]].item()*2+3)

        onehot = torch.zeros(seqlen,len(self.chars))
        for i,c in enumerate(sequence):
            onehot[i,c]=1
        return {'input': onehot, 
                'label': torch.from_numpy(np.array(label)),
                'style': style.view(-1)}

