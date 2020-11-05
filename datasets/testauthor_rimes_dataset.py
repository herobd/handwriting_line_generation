from datasets import author_rimes_dataset
import math
import sys, os
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
from utils.util import makeMask
import numpy as np
import torch
import cv2

widths=[]
saveHere='../data/RIMES_word_no_pad_images_train'
linenum=0

def display(data):
    global saveHere,linenum
    batchSize = data['image'].size(0)
    gts=[]
    #mask = makeMask(data['image'])
    for b in range(batchSize):
        #print (data['img'].size())
        img = (1-data['image'][b].permute(1,2,0))/2.0
        maskb = (data['mask'][b].permute(1,2,0)+1)/2.0
        label = data['label']
        gt = data['gt'][b]
        gts.append(gt)
        #print(label[:data['label_lengths'][b],b])
        #print(data['name'][b])
        #if data['spaced_label'] is not None:
        #    print('spaced label:')
        #    print(data['spaced_label'][:,b])
        #print(gt)

        widths.append(img.size(1))
        
        draw=False
        if draw :
            #cv2.imshow('line',img.numpy())
            #cv2.imshow('mask',maskb.numpy())
            #if img.size(1)>1000:
            #cv2.imwrite('out/{}.png'.format(data['name'][b]),img.numpy()*255)
            # print('wrote out/{}.png'.format(data['name'][b]))
            cv2.imwrite('out/mask{}.png'.format(b),maskb.numpy()*255)
            cv2.imwrite('out/img{}.png'.format(b),img.numpy()*255)

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
                for m_y,m_x in gt:
                    outline[int(m_y),int(m_x)] = (255,0,255)
                cv2.imwrite('out/line{}.png'.format(b),outline)
            #cv2.waitKey()
        if saveHere is not None:
            cv2.imwrite(os.path.join(saveHere,'{}.png').format(linenum),img.numpy()*255)
            print(linenum)
            linenum+=1

        #fig = plt.figure()

        #ax_im = plt.subplot()
        #ax_im.set_axis_off()
        #if img.shape[2]==1:
        #    ax_im.imshow(img[0])
        #else:
        #    ax_im.imshow(img)

        #plt.show()
    #print('batch complete')
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
    BATCH_SIZE=1
    data=author_rimes_dataset.AuthorRIMESDataset(dirPath=dirPath,split='train',config={
        'img_height': 64,
        'char_file' : 'data/IAM_char_set.json',
        'center_pad': False,
        'batch_size':BATCH_SIZE,
        'a_batch_size':1,
        'Xmask_post': ['true','thresh'],
        'mask_post': ["thresh","dilateCircle","errodeCircle"],
        #'mask_post': ['thresh','dilateCircle','errodeCircle','smaller', 'errodeCircle','dilateCircle', 'smaller', 'errodeCircle','dilateCircle'],
        #'mask_post': ['thresh','dilateCircle','errodeCircle', 'smaller', 'errodeCircle','dilateCircle',  'dilateCircle','errodeCircle'],
        'mask_random': False,
        'Xaugmentation': True,
        'Xspaced_loc': '../saved/spaced/spaced.pkl',
        'overfit': False,
        'skip_pad': True
})
    #data.cluster(start,repeat,'anchors_rot_{}.json')

    dataLoader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=author_rimes_dataset.collate)
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
    with open(os.path.join(dirPath,'RIMES_words_train_gt.txt'),'w') as out:
        for gt in gts:
            out.write(gt+'\n')
