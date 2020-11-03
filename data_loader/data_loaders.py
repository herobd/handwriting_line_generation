# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
import torch
import torch.utils.data
import numpy as np
#from datasets.cancer import CancerDataset
#from datasets.ai2d import AI2D
#from datasets import forms_detect
#from datasets.forms_detect import FormsDetect
#from datasets import forms_box_detect
#from datasets.forms_box_detect import FormsBoxDetect
#from datasets import ai2d_box_detect
#from datasets import forms_graph_pair
#from datasets import forms_box_pair
#from datasets.forms_box_pair import FormsBoxPair
#from datasets.forms_feature_pair import FormsFeaturePair
#from datasets import forms_feature_pair
#from datasets.forms_pair import FormsPair
#from datasets.forms_lf import FormsLF
#from datasets import random_messages
#from datasets import random_diffusion
from datasets import hw_dataset
from datasets import author_hw_dataset
from datasets import mixed_author_hw_dataset
from datasets import author_word_dataset
from datasets import mixed_author_word_dataset
from datasets import style_word_dataset
#from datasets import font_dataset
#from datasets import same_font_dataset
from datasets import spacing_dataset
from datasets.style_author_dataset import StyleAuthorDataset
#from torchvision import datasets, transforms
from base import BaseDataLoader




def getDataLoader(config,split):
        data_set_name = config['data_loader']['data_set_name']
        data_dir = config['data_loader']['data_dir']
        batch_size = config['data_loader']['batch_size']
        valid_batch_size = config['validation']['batch_size'] if 'batch_size' in config['validation'] else batch_size

        #copy info from main dataloader to validation (but don't overwrite)
        #helps insure same data
        for k,v in config['data_loader'].items():
            if k not in config['validation']:
                config['validation'][k]=v

        if 'augmentation_params' in config['data_loader']:
            aug_param = config['data_loader']['augmentation_params']
        else:
            aug_param = None
        shuffle = config['data_loader']['shuffle']
        if 'num_workers' in config['data_loader']:
            numDataWorkers = config['data_loader']['num_workers']
        else:
            numDataWorkers = 1
        shuffleValid = config['validation']['shuffle']

        if data_set_name=='AI2D':
            dataset=AI2D(dirPath=data_dir, split=split, config=config)
            if split=='train':
                validation=torch.utils.data.DataLoader(dataset.splitValidation(config), batch_size=batch_size, shuffle=shuffleValid, num_workers=numDataWorkers)
            else:
                validation=None
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=numDataWorkers), validation
        elif data_set_name=='FormsDetect':
            return withCollate(FormsDetect,forms_detect.collate,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config)
        elif data_set_name=='FormsBoxDetect':
            return withCollate(FormsBoxDetect,forms_box_detect.collate,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config)
        elif data_set_name=='AI2DBoxDetect':
            return withCollate(ai2d_box_detect.AI2DBoxDetect,ai2d_box_detect.collate,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config)
        elif data_set_name=='FormsBoxPair':
            return withCollate(FormsBoxPair,forms_box_pair.collate,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config)
        elif data_set_name=='FormsGraphPair':
            return withCollate(forms_graph_pair.FormsGraphPair,forms_graph_pair.collate,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config)
        elif data_set_name=='FormsFeaturePair':
            return withCollate(FormsFeaturePair,forms_feature_pair.collate,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config)
        elif data_set_name=='HWDataset':
            return withCollate(hw_dataset.HWDataset,hw_dataset.collate,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config)
        elif data_set_name=='AuthorHWDataset':
            return withCollate(author_hw_dataset.AuthorHWDataset,author_hw_dataset.collate,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config)
        elif data_set_name=='MixedAuthorHWDataset':
            return withCollate(mixed_author_hw_dataset.MixedAuthorHWDataset,mixed_author_hw_dataset.collate,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config)
        elif data_set_name=='AuthorWordDataset':
            return withCollate(author_word_dataset.AuthorWordDataset,author_word_dataset.collate,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config)
        elif data_set_name=='MixedAuthorWordDataset':
            return withCollate(mixed_author_word_dataset.MixedAuthorWordDataset,mixed_author_word_dataset.collate,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config)
        elif data_set_name=='StyleWordDataset':
            return withCollate(style_word_dataset.StyleWordDataset,style_word_dataset.collate,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config)
        elif data_set_name=='FontDataset':
            return withCollate(font_dataset.FontDataset,font_dataset.collate,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config)
        elif data_set_name=='SameFontDataset':
            return withCollate(same_font_dataset.SameFontDataset,same_font_dataset.collate,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config)
        elif data_set_name=='SpacingDataset':
            return withCollate(spacing_dataset.SpacingDataset,spacing_dataset.collate,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config)
        elif data_set_name=='FormsPair':
            return basic(FormsPair,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config)
        elif data_set_name=='FormsLF':
            return basic(FormsLF,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config)
        elif data_set_name=='StyleAuthorDataset':
            return basic(StyleAuthorDataset,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config)
        elif data_set_name=='Cancer':
            if split=='train':
                rot=config['rot'] if 'rot' in config else None
                trainData = CancerDataset(data_dir, train=True, rot=rot)
                trainLoader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=shuffle, num_workers=numDataWorkers)
                validData = CancerDataset(data_dir, train=False)
                validLoader = torch.utils.data.DataLoader(validData, batch_size=batch_size, shuffle=shuffleValid, num_workers=numDataWorkers)
                return trainLoader, validLoader
        elif data_set_name=='RandomMessagesDataset':
            data = random_messages.RandomMessagesDataset(config)
            dataLoader = torch.utils.data.DataLoader(data,batch_size=batch_size, shuffle=shuffle, num_workers=numDataWorkers,collate_fn=random_messages.collate)
            return dataLoader,dataLoader
        elif data_set_name=='RandomDiffusionDataset':
            data = random_diffusion.RandomDiffusionDataset(config)
            dataLoader = torch.utils.data.DataLoader(data,batch_size=batch_size, shuffle=shuffle, num_workers=numDataWorkers,collate_fn=random_diffusion.collate)
            return dataLoader,dataLoader
        else:
            print('Error, no dataloader has no set for {}'.format(data_set_name))
            exit()



def basic(setObj,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config):
    if split=='train':
        trainData = setObj(dirPath=data_dir, split='train', config=config['data_loader'])
        trainLoader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=shuffle, num_workers=numDataWorkers)
        validData = setObj(dirPath=data_dir, split='valid', config=config['validation'])
        validLoader = torch.utils.data.DataLoader(validData, batch_size=valid_batch_size, shuffle=shuffleValid, num_workers=numDataWorkers)
        return trainLoader, validLoader
    elif split=='test':
        testData = setObj(dirPath=data_dir, split='test', config=config['validation'])
        testLoader = torch.utils.data.DataLoader(testData, batch_size=valid_batch_size, shuffle=False, num_workers=numDataWorkers)
    elif split=='merge' or split=='merged' or split=='train-valid' or split=='train+valid':
        trainData = setObj(dirPath=data_dir, split=['train','valid'], config=config['data_loader'])
        trainLoader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=shuffle, num_workers=numDataWorkers)
        validData = setObj(dirPath=data_dir, split=['train','valid'], config=config['validation'])
        validLoader = torch.utils.data.DataLoader(validData, batch_size=valid_batch_size, shuffle=shuffleValid, num_workers=numDataWorkers)
        return trainLoader, validLoader
def withCollate(setObj,collateFunc,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config):
    if split=='train':
        trainData = setObj(dirPath=data_dir, split='train', config=config['data_loader'])
        trainLoader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=shuffle, num_workers=numDataWorkers, collate_fn=collateFunc)
        validData = setObj(dirPath=data_dir, split='valid', config=config['validation'])
        validLoader = torch.utils.data.DataLoader(validData, batch_size=valid_batch_size, shuffle=shuffleValid, num_workers=numDataWorkers, collate_fn=collateFunc)
        return trainLoader, validLoader
    elif split=='test':
        testData = setObj(dirPath=data_dir, split='test', config=config['validation'])
        testLoader = torch.utils.data.DataLoader(testData, batch_size=valid_batch_size, shuffle=False, num_workers=numDataWorkers, collate_fn=collateFunc)
        return testLoader, None
    elif split=='merge' or split=='merged' or split=='train-valid' or split=='train+valid':
        trainData = setObj(dirPath=data_dir, split=['train','valid'], config=config['data_loader'])
        trainLoader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=shuffle, num_workers=numDataWorkers, collate_fn=collateFunc)
        validData = setObj(dirPath=data_dir, split=['train','valid'], config=config['validation'])
        validLoader = torch.utils.data.DataLoader(validData, batch_size=valid_batch_size, shuffle=shuffleValid, num_workers=numDataWorkers, collate_fn=collateFunc)
        return trainLoader, validLoader
    

