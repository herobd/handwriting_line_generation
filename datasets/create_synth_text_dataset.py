from datasets import synth_text_dataset
import math
import sys
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
from utils.util import makeMask
import numpy as np
import torch
import cv2

widths=[]

def display(data):
    batchSize = data['image'].size(0)
    #mask = makeMask(data['image'])
    for b in range(batchSize):
        #print (data['img'].size())
        img = (data['image'][b].permute(1,2,0)+1)/2.0
        label = data['label']
        gt = data['gt'][b]
        #print(label[:data['label_lengths'][b],b])
        print(data['name'][b])
        #if data['spaced_label'] is not None:
        #    print('spaced label:')
        #    print(data['spaced_label'][:,b])
        print(gt)

        widths.append(img.size(1))
        
        draw=False
        if draw :
            #cv2.imshow('line',img.numpy())
            #cv2.imshow('mask',maskb.numpy())
            #cv2.imwrite('out/mask{}.png'.format(b),maskb.numpy()*255)
            #cv2.imwrite('out/fg_mask{}.png'.format(b),fg_mask.numpy()*255)
            #cv2.imwrite('out/img{}.png'.format(b),img.numpy()*255)
            #cv2.imwrite('out/changed_img{}.png'.format(b),changed_img.numpy()*255)
            plt.imshow(img.numpy()[:,:,0], cmap='gray')
            plt.show()

            if data['top_and_bottom'] is not None:
                center_line = data['center_line'][b]
                top_and_bottom = data['top_and_bottom'][b]
                #max_from_center = int(top_and_bottom.max().item())
                #outline = np.zeros((max_from_center*2+1,img.size(1),3),np.uint8)
                outline = np.zeros((img.size(0),img.size(1),3),np.uint8)
                for h in range(img.size(1)):
                    outline[int((center_line[h]-top_and_bottom[0,h]).item()),h] = (255,0,0)
                    outline[int((center_line[h]+top_and_bottom[1,h]).item()),h] = (0,0,255)
                    outline[int(center_line[h]),h] = (0,255,0)
                #cv2.imshow('top_and_bottom',outline)
            #cv2.waitKey()

        #fig = plt.figure()

        #ax_im = plt.subplot()
        #ax_im.set_axis_off()
        #if img.shape[2]==1:
        #    ax_im.imshow(img[0])
        #else:
        #    ax_im.imshow(img)

        #plt.show()
    print('batch complete')


if __name__ == "__main__":
    dirPath = sys.argv[1]
    data=synth_text_dataset.SynthTextDataset(dirPath=dirPath,split='train',config={

        "fontdir": "../data/fonts/",
        "textdir": "../data/",
        "shuffle": False,
        "num_workers": 1,

        "img_height": 32,
        "max_width": 5000,
        "char_file": "../data/english_char_set.json",
        "augmentation": "affine warp invert",
        "gen_type": "NAF",
        "max_chars": 40,
        "min_chars": 1,
        "refresh_self": True,
        "set_size": 2000000,
        "num_processes": 32,
        "use_before_refresh": 99999999999



})
    #data.cluster(start,repeat,'anchors_rot_{}.json')

    dataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True, num_workers=0, collate_fn=synth_text_dataset.collate)
    dataLoaderIter = iter(dataLoader)

        #if start==0:
        #display(data[0])
    dataLoaderIter.next()
    print('done')

