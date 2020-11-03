# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
from datasets import spacing_dataset
import math
import sys
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
import numpy as np
import torch
import cv2

def display(data):
    batchSize= data['style'].size(0)
    for b in range(batchSize):
        print(data['input'][:,b])
        print(data['label'][:,b])
        print(data['style'][b].view(5,5))

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
    dirPath=None   
    data=spacing_dataset.SpacingDataset(dirPath=dirPath,split='train',config={
})
    #data.cluster(start,repeat,'anchors_rot_{}.json')

    dataLoader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True, num_workers=0, collate_fn=spacing_dataset.collate)
    dataLoaderIter = iter(dataLoader)

    try:
        while True:
            #print('?')
            display(dataLoaderIter.next())
    except StopIteration:
        print('done')
