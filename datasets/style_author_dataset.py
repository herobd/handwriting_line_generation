# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
import torch.utils.data
import numpy as np
import math, os
import torch
from glob import glob
import pickle


class StyleAuthorDataset(torch.utils.data.Dataset):

    def __init__(self,config,dirPath=None,split=None):
        style_loc = os.path.join(dirPath,config['file_name'])
        if style_loc[-1]!='*':
            style_loc+='*'
        styles=[]
        authors=[]
        for loc in glob(style_loc):
            with open(loc,'rb') as f:
                data = pickle.load(f)
            s=data['styles']
            if len(s.shape)==4:
                s=s[:,:,0,0]
            styles.append(s)

            authors+=list(data['authors'])
        self.styles = np.concatenate(styles,axis=0)
        self.authors = authors

        author_set = set(authors)
        max_author=0
        min_author=9999
        for author in author_set:
            author=int(author)
            if author>max_author:
                max_author=author
            if author<min_author:
                min_author=author
        print('unique authors: {}, min: {}, max: {}'.format(len(author_set),min_author,max_author))

    def __len__(self):
        return self.styles.shape[0]

    def __getitem__(self,index):
        return torch.from_numpy(self.styles[index]).float(), torch.tensor(int(self.authors[index])).long()
              
             

