from datasets import author_rimeslines_dataset
import math, os
import sys
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
from utils.util import makeMask
import numpy as np
import torch
import cv2

widths=[]
saveHere='../data/RIMES_line_images_train'
linenum=0

def display(data):
    global saveHere,linenum
    batchSize = data['image'].size(0)
    #mask = makeMask(data['image'])
    gts=[]
    for b in range(batchSize):
        #print (data['img'].size())
        img = (1-data['image'][b].permute(1,2,0))/2.0
        maskb = (data['mask'][b].permute(1,2,0)+1)/2.0
        #changed_img = (data['changed_image'][b].permute(1,2,0)+1)/2.0
        label = data['label']
        gt = data['gt'][b]
        gts.append(gt)
        #print(label[:data['label_lengths'][b],b])
        #print('{}: {}'.format(data['name'][b],gt))
        #if data['spaced_label'] is not None:
        #    print('spaced label:')
        #    print(data['spaced_label'][:,b])
        #print(gt)

        widths.append(img.size(1))
        
        draw=False
        if draw :
            #cv2.imshow('line',img.numpy())
            #cv2.imshow('mask',maskb.numpy())
            #cv2.imwrite('out/mask{}.png'.format(b),maskb.numpy()*255)
            #cv2.imwrite('out/fg_mask{}.png'.format(b),fg_mask.numpy()*255)
            #cv2.imwrite('out/changed_img{}.png'.format(b),changed_img.numpy()*255)

            #cv2.imwrite('out/img{}.png'.format(b),img.numpy()*255)
            cv2.imshow('out/img.png',img.numpy())

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
            cv2.waitKey()
        if saveHere is not None:
            cv2.imwrite(os.path.join(saveHere,'{}.png').format(linenum),img.numpy()*255)
            linenum+=1

        #fig = plt.figure()

        #ax_im = plt.subplot()
        #ax_im.set_axis_off()
        #if img.shape[2]==1:
        #    ax_im.imshow(img[0])
        #else:
        #    ax_im.imshow(img)

        #plt.show()
    print('batch complete')
    return gts


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
    data=author_rimeslines_dataset.AuthorRIMESLinesDataset(dirPath=dirPath,split='train',config={
        'img_height': 64,
        'max_width': 1300,
        'char_file' : '../data/RIMES/characterset_lines.json',
        'center_pad': False,
        'a_batch_size':2,
        #'augmentation': 'normalization',
        'overfit': False,
        'xxshort':0,
})
    print('num authors: {}'.format(len(data.author_list)))
    #data.cluster(start,repeat,'anchors_rot_{}.json')

    dataLoader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True, num_workers=0, collate_fn=author_rimeslines_dataset.collate)
    dataLoaderIter = iter(dataLoader)

        #if start==0:
        #display(data[0])
    for i in range(0,start):
        print(i)
        dataLoaderIter.next()
        #display(data[i])
    gts=[]
    try:
        while True:
            #print('?')
            gts+=display(dataLoaderIter.next())
    except StopIteration:
        print('done')

    print('width mean: {}'.format(np.mean(widths)))
    print('width std: {}'.format(np.std(widths)))
    print('width max: {}'.format(np.max(widths)))
    with open(os.path.join(dirPath,'RIMES_lines_train_gt.txt'),'w') as out:
        for gt in gts:
            out.write(gt+'\n')
