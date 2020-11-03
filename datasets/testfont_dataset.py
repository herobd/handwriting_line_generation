# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
from datasets import font_dataset
import math
import sys
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
import numpy as np
import torch
import cv2

def display(data):
    batchSize = data['image'].size(0)
    for b in range(batchSize):
        #print (data['img'].size())
        img = (data['image'][b].permute(1,2,0)+1)/2.0
        label = data['label']
        gt = data['gt'][b]
        print(label[:data['label_lengths'][b],b])
        print(gt)


        cv2.imshow('line',img.numpy())
        cv2.waitKey()

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
    if len(sys.argv)>2:
        start = int(sys.argv[2])
    else:
        start=0
    if len(sys.argv)>3:
        repeat = int(sys.argv[3])
    else:
        repeat=1
    data=font_dataset.FontDataset(dirPath, split='train',config={
        'img_height': 64,
        'fontfile': 'mono_fonts.txt',
        'center_pad': False,
        'overfit': False
})
    #data.cluster(start,repeat,'anchors_rot_{}.json')

    dataLoader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=False, num_workers=0, collate_fn=font_dataset.collate)
    print('size: {}'.format(len(dataLoader)))
    dataLoaderIter = iter(dataLoader)

        #if start==0:
        #display(data[0])
    for i in range(0,start):
        print(i)
        dataLoaderIter.next()
        #display(data[i])
    try:
        while True:
            #print('?')
            display(dataLoaderIter.next())
    except StopIteration:
        print('done')
