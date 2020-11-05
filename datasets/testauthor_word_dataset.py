from datasets import author_word_dataset
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
    data=author_word_dataset.AuthorWordDataset(dirPath=dirPath,split='train',config={
        'img_height': 64,
        'char_file' : 'data/IAM_char_set.json',
        'batch_size':3,
        'a_batch_size':4,
        "style_loc": "../tmp/saved/Phase1_IAM_spacedS1DmaskLessDeeper_newStyleBigGlobalKL1closer/styles17k.pkl"
})
    #data.cluster(start,repeat,'anchors_rot_{}.json')

    dataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True, num_workers=0, collate_fn=author_word_dataset.collate)
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
