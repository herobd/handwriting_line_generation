import random,re
import json
import torch
import numpy as np
from utils import string_utils
class TextData():
    def __init__(self, textfile = 'data/lotr.txt', char_set_path='', batch_size=1, max_len=20, words=False, characterBalance=False,hardsplit_newline=False):
        self.max_len=max_len
        self.characterBalance=characterBalance
        if characterBalance:
            self.chars=[c for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ']


        #with open(os.path.join(dirPath,'sets.json')) as f:
        with open(textfile) as f:
            text = f.read()
        #text = text.replace('\n', ' ')#.replace('  ',' ')
        if hardsplit_newline:
            self.text = text.split('\n')
            self.words=True
        else:
            text=re.sub('\s+',' ',text) #This takes a minute on large text, but only needs done once.
            self.text=text
            self.words=words

            if words:
                words = text.strip().split(' ')
                self.text=[]#words
                for word in words:
                    m = re.match(r'[.,:\'"?!]*',word)
                    if m is None or m.span()[0]!=0 or m.span()[1]<len(word):
                        self.text.append(word)
        if len(char_set_path)>0:
            with open(char_set_path) as f: 
                char_set = json.load(f) 
            self.char_to_idx = char_set['char_to_idx']
        else:
            self.char_to_idx = None

        self.batch_size = batch_size

        self.max_len=max_len
        self.min_len=max(max_len-3,1)

    def getInstance(self):

        all_labels = []
        label_lengths = []
        gt=[]

        
        for i in range(self.batch_size):
            if self.words:
                idx=np.random.randint(0,len(self.text))
                text = self.text[idx]
                if len(self.text)>self.max_len:
                    diff = len(self.text) - self.max_len
                    start = random.randint(0,diff)
                    text[start:start+self.max_len]
            else:
                length = random.randint(self.min_len,self.max_len)
                idx = np.random.randint(0,len(self.text)-length)
                if self.characterBalance:
                    startIdx=idx
                    flipped=False
                    goalChar = random.choice(self.chars)
                    while True:
                        text =  self.text[idx:idx+length]
                        if goalChar in text:
                            break
                        idx += length
                        if idx>=len(self.text)-length:
                            flipped=True
                            idx=0
                        if flipped and idx>=startIdx:
                            #this char is not in the text set, so we'll just add it somewhere random
                            r=random.randint(0,len(text))
                            text=text[:r]+goalChar+text[r+1:]
                            break


                else:
                    text =  self.text[idx:idx+length]
                    assert(len(text)>0)
                    if text==' ':
                        text=self.text[idx+1]
            gt.append(text)
            if self.char_to_idx is not None:
                l = string_utils.str2label_single(text, self.char_to_idx)
                all_labels.append(l)
                label_lengths.append(len(l))

        if self.char_to_idx is not None:
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
        else:
            return {
                    
                    'gt': gt,
                    'image': None
                    }
