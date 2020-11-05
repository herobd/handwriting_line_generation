#from skimage import color, io
import os
import numpy as np
import torch
import cv2
from utils import util
import math
from model.loss import *
from collections import defaultdict
#import pickle
from utils import util, string_utils, error_rates
from datasets.hw_dataset import PADDING_CONSTANT

#THRESH=0.5



def getCorners(xyrhw):
    xc=xyrhw[0].item()
    yc=xyrhw[1].item()
    rot=xyrhw[2].item()
    h=xyrhw[3].item()
    w=xyrhw[4].item()
    h = min(30000,h)
    w = min(30000,w)
    tr = ( int(w*math.cos(rot)-h*math.sin(rot) + xc),  int(w*math.sin(rot)+h*math.cos(rot) + yc) )
    tl = ( int(-w*math.cos(rot)-h*math.sin(rot) + xc), int(-w*math.sin(rot)+h*math.cos(rot) + yc) )
    br = ( int(w*math.cos(rot)+h*math.sin(rot) + xc),  int(w*math.sin(rot)-h*math.cos(rot) + yc) )
    bl = ( int(-w*math.cos(rot)+h*math.sin(rot) + xc), int(-w*math.sin(rot)-h*math.cos(rot) + yc) )
    return tl,tr,br,bl
def plotRect(img,color,xyrhw,lineW=1):
    tl,tr,br,bl = getCorners(xyrhw)

    cv2.line(img,tl,tr,color,lineW)
    cv2.line(img,tr,br,color,lineW)
    cv2.line(img,br,bl,color,lineW)
    cv2.line(img,bl,tl,color,lineW)

def StyleWordDataset_eval(config,instance, trainer, metrics, outDir=None, startIndex=None, lossFunc=None):
    def __eval_metrics(data,target):
        acc_metrics = np.zeros((output.shape[0],len(metrics)))
        for ind in range(output.shape[0]):
            for i, metric in enumerate(metrics):
                acc_metrics[ind,i] += metric(output[ind:ind+1], target[ind:ind+1])
        return acc_metrics

    pred, recon, losses, style, spaced = trainer.run(instance,get_style=True)
    images = instance['image'].numpy()
    gt = instance['gt']
    name = instance['name']
    batchSize = len(gt)
    recon = recon.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    style = style.cpu().detach().numpy()
    sum_cer, pred_str, cer = trainer.getCER(gt,pred,individual=True)

    #num_style = style.shape[0]
    #num_in_style = len(instance['id'])//num_style
    #ids=[]
    #authors=[]
    #for i in range(num_style):
    #    #tosave.append( {'style': style[i], 'ids': instance['id'][num_in_style*i:(i+1)*num_in_style]} )
    #    ids.append(instance['id'][num_in_style*i:(i+1)*num_in_style])
    #    authors.append(instance['author'][num_in_style*i])
    #pickle.dump( tosave, open( save_loc, "wb" ) 
    if outDir is not None:
        if 'show_attention' in config:
            rs=np.random.RandomState(0)
            colors = (rs.rand(trainer.model.style_extractor.mhAtt1.h*trainer.model.style_extractor.keys1.size(1),3)*255).astype(np.uint8)
            attn = trainer.model.style_extractor.mhAtt1.attn
            assert(attn.size(0)==1)
            scale = images.shape[3]*images.shape[0]/attn.size(3)
            batch_len = attn.size(3)/images.shape[0]
            c_index=0
            attn_for=defaultdict(list)
            for head in range(attn.size(1)):
                for query in range(attn.size(2)):
                    loc = attn[0,head,query].argmax().item()
                    b = loc//batch_len
                    x_pixel_loc = int((loc%batch_len)*scale)
                    y_pixel_loc = query*images.shape[2]//attn.size(2) #+ head
                    attn_for[b].append((y_pixel_loc,x_pixel_loc,colors[c_index]))
                    #print('h:{}, q:{}, b:{}, ({},{})'.format(head,query,b,x_pixel_loc,y_pixel_loc))
                    c_index+=1
            maxA = attn.max()
            minA = attn.min()
            streched_attn = F.interpolate((attn[0]-minA)/(maxA-minA), size=int(images.shape[3]*batchSize)).cpu()
        for b in range(batchSize):

            image = (1-((1+np.transpose(images[b][:,:,:],(1,2,0)))/2.0)).copy()
            reconstructed = (1-((1+np.transpose(recon[b][:,:,:],(1,2,0)))/2.0)).copy()
            border = np.zeros((image.shape[0],5,image.shape[2]))

            bigPic = np.concatenate((image,border,reconstructed),axis=1)
            border = np.zeros((50,bigPic.shape[1],bigPic.shape[2]))
            bigPic = np.concatenate((bigPic,border),axis=0)

            

            #if image.shape[2]==1:
            #    image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
            cv2.putText(bigPic,'CER: {:.3f}, T: {}'.format(cer[b],pred_str[b]),(0,image.shape[0]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0.9,0.3,0),2,cv2.LINE_AA)
            bigPic*=255
            bigPic = bigPic.astype(np.uint8)
            if 'show_attention' in config:
                if bigPic.shape[2]==1:
                    bigPic = cv2.cvtColor(bigPic,cv2.COLOR_GRAY2RGB)
                #if 'head' in config['show_attention']:
                if 'full' in config['show_attention']:
                    attnImage = np.zeros((attn.size(1)*attn.size(2),bigPic.shape[1],3))
                    for head in range(attn.size(1)):
                        for query in range(attn.size(2)):
                            y_pixel_loc = head + attn.size(1)*query #query*images.shape[2]//attn.size(2) #+ head
                            x_start = int(b*image.shape[1])
                            x_end = int((b+1)*image.shape[1])
                            if head<3:
                                attnImage[y_pixel_loc,0:image.shape[1],head]=streched_attn[head,query,x_start:x_end].numpy()
                            else:
                                attnImage[y_pixel_loc,0:image.shape[1],head%3]=streched_attn[head,query,x_start:x_end].numpy()
                                attnImage[y_pixel_loc,0:image.shape[1],(head+1)%3]=streched_attn[head,query,x_start:x_end].numpy()

                    attnImage*=255
                    attnImage = attnImage.astype(np.uint8)
                    bigPic = np.concatenate((attnImage,bigPic),axis=0)
                    
                else:
                    for y,x,c in attn_for[b]:
                        bigPic[y:y+2,x:x+2]=c
                        #print('{}, {}  ({},{})'.format(x,y,image.shape[1],image.shape[0]))


            saveName = '{}.png'.format(name[b])
            cv2.imwrite(os.path.join(outDir,saveName),bigPic)
            #io.imsave(os.path.join(outDir,saveName),bigPic)
            print('saved: '+os.path.join(outDir,saveName))
            #import pdb;pdb.set_trace()
        
    #return metricsOut
    for name in losses:
        losses[name] = losses[name].item()
    toRet=   { 
            **losses,
            'cer': cer
             }
    a_batch_size = instance['a_batch_size']
    batch_size = batchSize//a_batch_size
    style = np.stack([style[i] for i in range(0,batchSize,a_batch_size)],axis=0)
    authors = instance['author']
    authors = [authors[i] for i in range(0,batchSize,a_batch_size)]
    ids = instance['id']
    ids = [ids[i:i+a_batch_size] for i in range(0,batchSize,a_batch_size)]
    return (
             toRet,
             (style,authors,ids)
            )


