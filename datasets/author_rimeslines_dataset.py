import json

import torch
import torch.nn.functional as F
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
from utils.parseRIMESlines import getLineBoundaries as parseXML
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
    if 'changed_image' in batch[0]:
        changed_batch = torch.full((len(batch)*a_batch_size, dim1, dim2, dim3), PADDING_CONSTANT)
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
        if 'changed_image' in batch[i]:
            changed_batch[i*a_batch_size:(i+1)*a_batch_size,:,:,0:b_img.shape[3]] = batch[i]['changed_image']
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

    toRet = {
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
        "author_idx": [l for b in batch for l in b['author_idx']],
        "name": [l for b in batch for l in b['name']],
        "a_batch_size": a_batch_size
    }
    if 'fg_mask' in batch[0]:
        toRet['fg_mask']=fg_masks
    if 'changed_image' in batch[0]:
        toRet['changed_image']=changed_batch
    return toRet


class AuthorRIMESLinesDataset(Dataset):
    def __init__(self, dirPath, split, config):
        if 'split' in config:
            split = config['split']
        if split=='test' or split=='valid':
            if split=='valid':
                print('WARNING: Using test set for validation!')
            xml = os.path.join(dirPath,'lines_eval_2011_annotated.xml')
        else:
            xml = os.path.join(dirPath,'lines_training_2011.xml')

        self.img_height = config['img_height']
        self.batch_size = config['a_batch_size']
        self.no_spaces = config['no_spaces'] if 'no_spaces' in config else False
        self.max_width = config['max_width'] if  'max_width' in config else 3000
        #assert(config['batch_size']==1)
        self.warning=False
        self.dirPath=dirPath

        self.triplet = config['triplet'] if 'triplet' in config else False
        if self.triplet:
            self.triplet_author_size = config['triplet_author_size']
            self.triplet_sample_size = config['triplet_sample_size']

        only_author = config['only_author'] if 'only_author' in config else None
        skip_author = config['skip_author'] if 'skip_author' in config else None

        self.authors = defaultdict(list)
        self.lineIndex = []
        self.max_char_len=0
        self.authors = parseXML(xml)
        self.author_list=list(self.authors.keys())
        self.author_list.sort()
        if only_author is not None:
            raise NotImplementedError('only_author not implemented for RIMES. There arent authors anyways')
        if skip_author is not None:
            raise NotImplementedError('skip_author not implemented for RIMES. There arent authors anyways')
        #minLines=99999
        #for author,lines in self.authors.items():
            #print('{} {}'.format(author,len(lines)))
            #minLines = min(minLines,len(lines))
        #maxCombs = int(nCr(minLines,self.batch_size)*1.2)
        short = config['short'] if 'short' in config else False
        self.max_char_len=0
        for author,lines in self.authors.items():
            self.max_char_len = max(self.max_char_len,max([len(l[2]) for l in lines]))
            if split=='train' and self.batch_size==2:
                combs=list(itertools.combinations(list(range(len(lines))),self.batch_size))
                #np.random.shuffle(combs)
                if short:
                    combs = combs[:short]
                self.lineIndex += [(author,c) for c in combs]
            else:
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
        self.fg_masks_dir = config['fg_masks_dir'] if 'fg_masks_dir' in config else None

        if self.fg_masks_dir is not None:
            if self.fg_masks_dir[-1]=='/':
                self.fg_masks_dir = self.fg_masks_dir[:-1]
            self.fg_masks_dir+='_{}'.format(self.max_width)
            ensure_dir(self.fg_masks_dir)
            for author,lines in self.lineIndex:
                for line in lines:
                    img_path, lb, gt = self.authors[author][line]
                    fg_path = os.path.join(self.fg_masks_dir,'{}_{}.png'.format(author,line))
                    if not os.path.exists(fg_path):
                        img = cv2.imread(img_path,0)[lb[0]:lb[1],lb[2]:lb[3]] #read as grayscale, crop line

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
                        #test_path = os.path.join(fg_masks_dir,'{}_{}_test.png'.format(author,line))
                        ##print(img.shape)
                        #img = np.stack((img,img,img),axis=2)
                        #img[:,:,0]=binarized
                        #cv2.imwrite(test_path,img)
                        #print('saved fg mask: {}'.format(fg_path))

            #if split=='train':
            #    ss = set(self.lineIndex)
        #self.authors = self.authors.keys()
                

        char_set_path = config['char_file']
        with open(char_set_path) as f:
            char_set = json.load(f)
        self.char_to_idx = char_set['char_to_idx']

        self.augmentation = config['augmentation'] if 'augmentation' in config else None
        self.normalized_dir = config['cache_normalized'] if 'cache_normalized' in config else None
        if self.normalized_dir is not None:
            ensure_dir(self.normalized_dir)
        self.max_strech=0.4
        self.max_rot_rad= 45/180 * math.pi

        self.remove_bg = config['remove_bg'] if 'remove_bg' in config else False
        self.include_stroke_aug = config['include_stroke_aug'] if 'include_stroke_aug' in config else False

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
            self.identity_spaced = config['no_spacing_for_spaced'] if 'no_spacing_for_spaced' in config else False

        self.mask_post = config['mask_post'] if 'mask_post' in config else []
        self.mask_random = config['mask_random'] if 'mask_random' in config else False

    def __len__(self):
        return len(self.lineIndex)

    def __getitem__(self, idx):
        if type( self.augmentation) is str and 'affine' in self.augmentation:
            strech = (self.max_strech*2)*np.random.random() - self.max_strech +1
            #self.max_rot_rad = self.max_rot_deg/180 * np.pi
            skew = (self.max_rot_rad*2)*np.random.random() - self.max_rot_rad
        if self.include_stroke_aug:
            thickness_change= np.random.randint(-4,5)
            fg_shade = np.random.random()*0.25 + 0.75
            bg_shade = np.random.random()*0.2
            blur_size = np.random.randint(2,4)
            noise_sigma = np.random.random()*0.02

        batch=[]

        if self.triplet=='hard':
            authors = random.sample(self.authors.keys(),self.triplet_author_size)
            alines=[]
            for author in authors:
                if len(self.authors[author])>=self.triplet_sample_size*self.batch_size:
                    lines = random.sample(range(len(self.authors[author])),self.triplet_sample_size*self.batch_size)
                else:
                    lines = list(range(len(self.authors[author])))
                    random.shuffle(lines)
                    dif = self.triplet_sample_size*self.batch_size-len(self.authors[author])
                    lines += lines[:dif]
                alines += [(author,l) for l in lines]
        else:


            inst = self.lineIndex[idx]
            author=inst[0]
            lines=inst[1]


            alines = [(author,l) for l in lines]
            used_lines = set(lines)
            if self.triplet:
                if len(self.authors[author])<=2*self.batch_size:
                    for l in range(len(self.authors[author])):
                        if l not in used_lines:
                            alines.append((author,l))
                    if len(alines)<2*self.batch_size:
                        dif = 2*self.batch_size - len(alines)
                        for i in range(dif):
                            alines.append(alines[self.batch_size+i])
                else:
                    unused_lines = set(range(len(self.authors[author])))-used_lines
                    for i in range(self.batch_size):
                        l = random.select(unused_lines)
                        unused_lines.remove(l)
                        alines.append((author,l))
                
                other_authors = set(range(len(self.authors)))
                other_authors.remove(author)
                author = random.select(other_authors)
                unused_lines = set(range(len(self.authors[author])))-used_lines
                for i in range(self.batch_size):
                    l = random.select(unused_lines)
                    unused_lines.remove(l)
                    alines.append((author,l))

            

        images=[]
        for author,line in alines:
            if line>=len(self.authors[author]):
                line = (line+37)%len(self.authors[author])
            img_path, lb, gt = self.authors[author][line]
            img_path = os.path.join(self.dirPath,'images_gray',img_path)

            if self.no_spaces:
                gt = gt.replace(' ','')
            if type(self.augmentation) is str and 'normalization' in  self.augmentation and self.normalized_dir is not None and os.path.exists(os.path.join(self.normalized_dir,'{}_{}.png'.format(author,line))):
                img = cv2.imread(os.path.join(self.normalized_dir,'{}_{}.png'.format(author,line)),0)
                readNorm=True
            else:
                img = cv2.imread(img_path,0)
                if img is None:
                    print('Error, could not read image: {}'.format(img_path))
                    return None
                lb[0] = max(lb[0],0)
                lb[2] = max(lb[2],0)
                lb[1] = min(lb[1],img.shape[0])
                lb[3] = min(lb[3],img.shape[1])
                img = img[lb[0]:lb[1],lb[2]:lb[3]] #read as grayscale, crop line
                readNorm=False


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
            elif img.shape[1]> self.max_width:
                percent = self.max_width/img.shape[1]
                img = cv2.resize(img, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)
                if img.shape[0]<self.img_height:
                    diff = self.img_height-img.shape[0]
                    img = np.pad(img,((diff//2,diff//2+diff%2),(0,0)),'constant',constant_values=255)

            if self.augmentation=='affine':
                if img.shape[1]*strech > self.max_width:
                    strech = self.max_width/img.shape[1]
            images.append( (line,gt,img,author) )
            #we split the processing here so that strech will be adjusted for longest image in author batch


        for line,gt,img,author in images:
            if self.fg_masks_dir is not None:
                fg_path = os.path.join(self.fg_masks_dir,'{}_{}.png'.format(author,line))
                fg_mask = cv2.imread(fg_path,0)
                fg_mask = fg_mask/255
                if fg_mask.shape!=img[:,:].shape:
                    print('Error, fg_mask ({}, {}) not the same size as image ({})'.format(fg_path,fg_mask.shape,img[:,:,0].shape))
                    th,fg_mask = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    fg_mask = 255-fg_mask
                    ele = cv2.getStructuringElement(  cv2.MORPH_ELLIPSE, (9,9) )
                    fg_mask = cv2.dilate(fg_mask,ele)
                    fg_mask = fg_mask/255
            else:
                fg_mask=None

                    
            if type(self.augmentation) is str and 'normalization' in  self.augmentation and not readNorm:
                img = normalize_line.deskew(img)
                img = normalize_line.skeletonize(img)
                if self.normalized_dir is not None:
                    cv2.imwrite(os.path.join(self.normalized_dir,'{}_{}.png'.format(author,line)),img)
            if type(self.augmentation) is str and 'affine' in  self.augmentation:
                img,fg_mask = augmentation.affine_trans(img,fg_mask,skew,strech)
            elif self.augmentation is not None and (type(self.augmentation) is not None or 'warp' in self.augmentation):
                #img = augmentation.apply_random_color_rotation(img)
                img = augmentation.apply_tensmeyer_brightness(img)
                img = grid_distortion.warp_image(img)
                assert(fg_mask is None)

            if self.include_stroke_aug:
                new_img = augmentation.change_thickness(img,thickness_change,fg_shade,bg_shade,blur_size,noise_sigma)
                if len(new_img.shape)==2:
                    new_img = new_img[...,None]
                new_img = new_img*2 -1.0

            if len(img.shape)==2:
                img = img[...,None]

            img = img.astype(np.float32)
            if self.remove_bg:
                img = 1.0 - img / 256.0
                #kernel = torch.FloatTensor(7,7).fill_(1/49)
                #blurred_mask = F.conv2d(fg_mask,kernel,padding=3)
                blurred_mask = cv2.blur(fg_mask,(7,7))
                img *= blurred_mask[...,None]
                img = 2*img -1
            else:
                img = 1.0 - img / 128.0



            if len(gt) == 0:
                return None
            gt_label = string_utils.str2label_single(gt, self.char_to_idx)

            if self.styles:
                style_i = self.npr.choice(len(self.styles[author][id]))
                style = self.styles[author][id][style_i]
            else:
                style=None
            name = '{}_{}'.format(author,line)
            if self.identity_spaced:
                spaced_label = gt_label[:,None].astype(np.long)
            else:
                spaced_label = None if self.spaced_by_name is None else self.spaced_by_name[name]
                if spaced_label is not None:
                    assert(spaced_label.shape[1]==1)
            toAppend= {
                "image": img,
                "gt": gt,
                "style": style,
                "gt_label": gt_label,
                "spaced_label": spaced_label,
                "name": name,
                "center": self.center,
                "author": author,
                "author_idx": self.author_list.index(author)
                
                }
            if self.fg_masks_dir is not None:
                toAppend['fg_mask'] = fg_mask
            if self.include_stroke_aug:
                toAppend['changed_image'] = new_img
            batch.append(toAppend)
            
        #batch = [b for b in batch if b is not None]
        #These all should be the same size or error
        assert len(set([b['image'].shape[0] for b in batch])) == 1
        assert len(set([b['image'].shape[2] for b in batch])) == 1

        dim0 = batch[0]['image'].shape[0]
        dim1 = max([b['image'].shape[1] for b in batch])
        dim2 = batch[0]['image'].shape[2]

        all_labels = []
        label_lengths = []
        if self.spaced_by_name is not None or self.identity_spaced:
            spaced_labels = []
        else:
            spaced_labels = None
        max_spaced_len=0

        input_batch = np.full((len(batch), dim0, dim1, dim2), PADDING_CONSTANT).astype(np.float32)
        if self.fg_masks_dir is not None:
            fg_masks = np.full((len(batch), dim0, dim1, 1), 0).astype(np.float32)
        if self.include_stroke_aug:
            changed_batch = np.full((len(batch), dim0, dim1, dim2), PADDING_CONSTANT).astype(np.float32)
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
            if self.include_stroke_aug:
                changed_batch[i,:,toPad:toPad+b_img.shape[1],:] = batch[i]['changed_image']

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
        if self.spaced_by_name is not None or self.identity_spaced:
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
        toRet= {
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
            "author_idx": [b['author_idx'] for b in batch],
        }
        if self.fg_masks_dir is not None:
            toRet['fg_mask'] = fg_masks
        if self.include_stroke_aug:
            changed_images = changed_batch.transpose([0,3,1,2])
            changed_images = torch.from_numpy(changed_images)
            toRet['changed_image']=changed_images
        return toRet

    def max_len(self):
        return self.max_char_len
