# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
#from skimage import color, io
import os
import numpy as np
import torch
import cv2
from utils import util
import math
from model.loss import *
from collections import defaultdict
import json
from utils import util, string_utils, error_rates
from datasets.hw_dataset import PADDING_CONSTANT

#THRESH=0.5


def _to_tensor(instance,gpu):
    image = instance['image']
    label = instance['label']

    if gpu is not None:
        image = image.to(gpu)
        if label is not None:
            label = label.to(gpu)
    return image, label
def run(instance,model,num_class,gpu,lossWeights):
        image, label = _to_tensor(instance,gpu)
        label_lengths = instance['label_lengths']
        #gt = instance['gt']

        ###Autoencoder###
        style = model.style_extractor(image)
        pred = model.hwr(image)

        label_onehot = torch.zeros(label.size(0),label.size(1),num_class)
        #label_onehot[label]=1
        #TODO tensorize
        for i in range(label.size(0)):
            for j in range(label.size(1)):
                label_onehot[i,j,label[i,j]]=1
        if gpu is not None:
            label_onehot = label_onehot.to(gpu)
        recon = model.generator(label_onehot,style)
        #recon, HACK = model.generator(label_onehot,style)
        #import pdb;pdb.set_trace()
        #print(HACK.argmax(dim=2))

        if recon.size(3)>image.size(3):
            toPad = recon.size(3)-image.size(3)
            image = F.pad(image,(0,toPad),value=PADDING_CONSTANT)
        elif recon.size(3)<image.size(3):
            toPad = image.size(3)-recon.size(3)
            recon = F.pad(recon,(0,toPad),value=PADDING_CONSTANT)

        autoLoss = L1Loss(recon,image) * lossWeights['auto']

        batch_size = pred.size(1)
        pred_size = torch.IntTensor([pred.size(0)] * batch_size)
        if 'recog' in lossWeights:
            recogLoss = CTCLoss(pred,label.permute(1,0),pred_size,label_lengths) * lossWeights['recog']
        else:
            recogLoss = None

        return pred, recon, recogLoss, autoLoss

def getCER(gt,pred,idx_to_char):
        cer=[]
        pred_strs=[]
        for i,gt_line in enumerate(gt):
            logits = pred[:,i]
            pred_str, raw_pred = string_utils.naive_decode(logits)
            pred_str = string_utils.label2str_single(pred_str, idx_to_char, False)
            cer.append(error_rates.cer(gt_line, pred_str))
            pred_strs.append(pred_str)
        return cer, pred_strs

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

def HWDataset_printer(config,instance, model, gpu, metrics, outDir=None, startIndex=None, lossFunc=None):
    def __eval_metrics(data,target):
        acc_metrics = np.zeros((output.shape[0],len(metrics)))
        for ind in range(output.shape[0]):
            for i, metric in enumerate(metrics):
                acc_metrics[ind,i] += metric(output[ind:ind+1], target[ind:ind+1])
        return acc_metrics

    if 'idx_to_char' not in config:
        char_set_path = config['data_loader']['char_file']
        with open(char_set_path) as f:
            char_set = json.load(f)
        idx_to_char = {}
        num_class = len(char_set['idx_to_char'])+1
        for k,v in char_set['idx_to_char'].items():
            idx_to_char[int(k)] = v
        config['idx_to_char'] = idx_to_char
    else:
        idx_to_char = config['idx_to_char']


    num_class=config['model']['num_class']
    lossWeights = config['loss_weights']
    pred, recon, recogLoss, autoLoss = run(instance,model,num_class,gpu,lossWeights)
    if recogLoss is not None:
        recogLoss=recogLoss.item()
    autoLoss=autoLoss.item()
    images = instance['image'].numpy()
    gt = instance['gt']
    name = instance['name']
    batchSize = len(gt)
    recon = recon.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    cer, pred_str = getCER(gt,pred,idx_to_char)
    if outDir is not None:
        for b in range(batchSize):

            image = (1-((1+np.transpose(images[b][:,:,:],(1,2,0)))/2.0)).copy()
            reconstructed = (1-((1+np.transpose(recon[b][:,:,:],(1,2,0)))/2.0)).copy()
            border = np.zeros((image.shape[0],5,image.shape[2]))

            bigPic = np.concatenate((image,border,reconstructed),axis=1)
            border = np.zeros((50,bigPic.shape[1],bigPic.shape[2]))
            bigPic = np.concatenate((bigPic,border),axis=0)

            

            if image.shape[2]==1:
                image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
            cv2.putText(bigPic,'CER: {:.3f}, T: {}'.format(cer[b],pred_str[b]),(0,image.shape[0]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0.9,0.3,0),2,cv2.LINE_AA)


            saveName = '{}.png'.format(name[b])
            cv2.imwrite(os.path.join(outDir,saveName),255*bigPic)
            #io.imsave(os.path.join(outDir,saveName),bigPic)
            #print('saved: '+os.path.join(outDir,saveName))
        
    #return metricsOut
    toRet=   { 
            'autoLoss': autoLoss,
            'cer': cer
             }
    if recogLoss is not None:
        toRet['recogLoss'] = recogLoss

    return (
             toRet,
             (cer,)
            )


