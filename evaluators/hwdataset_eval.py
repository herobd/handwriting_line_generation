#from skimage import color, io
import os
import numpy as np
import torch
import cv2
from utils import util
import math
from model.loss import *
from model.hw_with_style import correct_pred
from collections import defaultdict
import json
from utils import util, string_utils, error_rates
from datasets.hw_dataset import PADDING_CONSTANT
from trainer import *

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

def HWDataset_eval(config,instance, trainer, metrics, outDir=None, startIndex=None, lossFunc=None, toEval=None):
    def __eval_metrics(data,target):
        acc_metrics = np.zeros((output.shape[0],len(metrics)))
        for ind in range(output.shape[0]):
            for i, metric in enumerate(metrics):
                acc_metrics[ind,i] += metric(output[ind:ind+1], target[ind:ind+1])
        return acc_metrics
    
    if type(trainer) is HWRWithSynthTrainer:
        pred, losses = trainer.run(instance)
        out={'pred':pred}
    elif type(trainer) is AutoTrainer:
        losses, out = trainer.run_gen(instance,toEval)
    else:#if type(trainer) is HWWithStyleTrainer:
        if toEval is None:

            pred, recon, losses, style, spaced = trainer.run(instance,get_style=True)
            toEval = ['pred','recon','style','spaced']
            out={}
            if pred is not None:
                out['pred']=pred
            if recon is not None:
                out['recon']=recon
            if style is not None:
                out['style']=style
            if spaced is not None:
                out['spaced']=spaced
        elif type(toEval) is list:
            losses, out = trainer.run_gen(instance,trainer.curriculum.getEval(),toEval)
        else:
            if toEval=='spaced':
                justSpaced(trainer.model,instance,trainer.gpu if trainer.with_cuda else None)
            if toEval=='spacing':
                justSpacing(trainer.model,instance,trainer.gpu if trainer.with_cuda else None)
            elif toEval=='mask':
                justMask(trainer.model,instance,trainer.gpu if trainer.with_cuda else None)
            else:
                raise ValueError('unkwon just: {}'.format(toEval))
            return {}, (None,)


    images = instance['image'].numpy()
    gt = instance['gt']
    name = instance['name']
    batchSize = len(gt)
    #style = style.cpu().detach().numpy()
    if 'pred' in out:
        pred = out['pred'].cpu().detach()
        aligned1 = correct_pred(pred,instance['label'])
        aligned=[]
        for b in range(batchSize):
            #a, raw_aligned = string_utils.naive_decode(aligned1[:,b])
            a = []
            for i in range(len(aligned1[:,b])):
                if aligned1[i,b] != 0 and not ( i > 0 and aligned1[i,b] == aligned1[i-1,b] ):
                    a.append(aligned1[i,b].item())
            aligned.append(string_utils.label2str_single(a, trainer.idx_to_char, False))
        
        pred = pred.numpy()
        sum_cer,sum_wer, pred_str, cer = trainer.getCER(gt,pred,individual=True)
        out['cer'] = cer
        out['pred_str']=pred_str

        for b in range(batchSize):
            print('{} GT:      {}'.format(instance['name'][b],gt[b]))
            print('{} aligned: {}'.format(instance['name'][b],aligned[b]))
            print('{} pred:    {}'.format(instance['name'][b],pred_str[b]))
            print(pred[:,b])
        if 'by_author' in config:
            by_author=defaultdict(list)
            for b,author in enumerate(instance['author']):
                by_author['cer_'+author].append(cer[b])
    
    if outDir is not None:
        for key_name in ['recon','recon_gt_mask','recon_pred_space']:
            if key_name in out:# and 'pred' in out:
                recon = out[key_name].cpu().detach().numpy()
                if 'show_attention' in config:
                    rs=np.random.RandomState(0)
                    colors = (rs.rand(trainer.model.style_extractor.mhAtt1.h*trainer.model.style_extractor.keys1.size(1),3)*255).astype(np.uint8)
                    attn = trainer.model.style_extractor.mhAtt1.attn
                    assert(attn.size(0)==1)
                    #OR
                    #attn = attn.view(batchSize*a_batch_size
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
                    if 'cer_thresh' in config and cer[b]<config['cer_thresh']:
                        continue
                    toColor=False
                    image = (1-((1+np.transpose(images[b][:,:,:],(1,2,0)))/2.0)).copy()
                    if recon is not None:
                        reconstructed = (1-((1+np.transpose(recon[b][:,:,:],(1,2,0)))/2.0)).copy()
                        #border = np.zeros((image.shape[0],5,image.shape[2]))

                        #bigPic = np.concatenate((image,border,reconstructed),axis=1)

                        padded=None
                        if reconstructed.shape[1]>image.shape[1]:
                            #reconstructed=reconstructed[:,:image.shape[1]]
                            dif = -(image.shape[1] -reconstructed.shape[1])
                            image = np.pad(image,((0,0),(dif//2,dif//2+dif%2),(0,0)),mode='constant')
                            padded='real'
                        elif image.shape[1]>reconstructed.shape[1]: #pad
                            dif = image.shape[1] -reconstructed.shape[1]
                            reconstructed = np.pad(reconstructed,((0,0),(dif//2,dif//2+dif%2),(0,0)),mode='constant')
                            padded='gen'
                        border = np.zeros((2,image.shape[1],image.shape[2]))
                        bigPic = np.concatenate((image,border,reconstructed),axis=0)

                        #add color border for visibility in paper figures
                        if bigPic.shape[2]==1 and True:
                            bigPic*=255
                            bigPic = bigPic.astype(np.uint8)
                            bigPic = cv2.cvtColor(bigPic[:,:,0],cv2.COLOR_GRAY2RGB)
                            toColor=True
                            padReal=dif//2 if padded=='real' else 0
                            padGen =dif//2 if padded=='gen' else 0
                            #print('COLOR!')
                            if padReal!=0:
                                bigPic[0,padReal:-padReal,1]=255
                                bigPic[image.shape[0],padReal:-padReal,1]=255
                                bigPic[image.shape[0],padReal:-padReal,0]=0
                                bigPic[0,padReal:-padReal,0]=0
                                bigPic[0,padReal:-padReal,2]=0
                                bigPic[image.shape[0],padReal:-padReal,2]=0
                            else:
                                bigPic[0,:,1]=255
                                bigPic[image.shape[0],:,1]=255
                                bigPic[image.shape[0],:,0]=0
                                bigPic[0,:,0]=0
                                bigPic[0,:,2]=0
                                bigPic[image.shape[0],:,2]=0
                            bigPic[:image.shape[0],padReal,1]=255
                            bigPic[:image.shape[0],-1-padReal,1]=255
                            bigPic[:image.shape[0],padReal,0]=0
                            bigPic[:image.shape[0],-1-padReal,0]=0
                            bigPic[:image.shape[0],padReal,2]=0
                            bigPic[:image.shape[0],-1-padReal,2]=0


                            if padGen != 0:
                                bigPic[image.shape[0]+1,padGen:-padGen,0]=255
                                bigPic[-1,padGen:-padGen,0]=255
                                bigPic[image.shape[0]+1,padGen:-padGen,2]=0
                                bigPic[-1,padGen:-padGen,2]=0
                                bigPic[image.shape[0]+1,padGen:-padGen,1]=0
                                bigPic[-1,padGen:-padGen,1]=0
                            else:
                                bigPic[image.shape[0]+1,:,0]=255
                                bigPic[-1,:,0]=255
                                bigPic[image.shape[0]+1,:,2]=0
                                bigPic[-1,:,2]=0
                                bigPic[image.shape[0]+1,:,1]=0
                                bigPic[-1,:,1]=0

                            bigPic[image.shape[0]+2:,padGen,0]=255
                            bigPic[image.shape[0]+2:,-1-padGen,0]=255
                            bigPic[image.shape[0]+2:,padGen,2]=0
                            bigPic[image.shape[0]+2:,-1-padGen,2]=0
                            bigPic[image.shape[0]+2:,padGen,1]=0
                            bigPic[image.shape[0]+2:,-1-padGen,1]=0
                    else:
                        bigPic=image

                    

                    #if image.shape[2]==1:
                    #    image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
                    if 'pred' in out:
                       border = np.zeros((50,bigPic.shape[1],bigPic.shape[2]))
                       bigPic = np.concatenate((bigPic,border),axis=0)
                       cv2.putText(bigPic,'CER: {:.3f}, T: {}'.format(cer[b],pred_str[b]),(0,image.shape[0]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0.9,0.3,0),2,cv2.LINE_AA)
                    if not toColor:
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



                    saveName = '{}_{}.png'.format(name[b],key_name)
                    if 'cer_thresh' in config:
                        saveName = '{:.3f}_'.format(cer[b])+saveName
                    cv2.imwrite(os.path.join(outDir,saveName),bigPic)
                    #io.imsave(os.path.join(outDir,saveName),bigPic)
                    print('saved: '+os.path.join(outDir,saveName))
                    #import pdb;pdb.set_trace()

        if 'gen' in out or 'gen_img' in out:
            if 'gen' in out and out['gen'] is not None:
                gen = out['gen'].cpu().detach().numpy()
            elif 'gen_img' in out and  out['gen_img'] is not None:
                gen = out['gen_img'].cpu().detach().numpy()
            else:
                #not sure why this happens
                print('ERROR, None for generated images, {}'.format(name))
                gen = np.ones((batchSize,1,5,5))
            for b in range(batchSize):
                generated = (1-((1+np.transpose(gen[b][:,:,:],(1,2,0)))/2.0)).copy() *255
                saveName = 'gen_{}.png'.format(name[b])
                cv2.imwrite(os.path.join(outDir,saveName),generated)
        if 'mask' in out:
            mask = ((1+out['mask'])*127.5).cpu().detach().permute(0,2,3,1).numpy().astype(np.uint8)
            for b in range(batchSize):
                saveName = '{}_mask.png'.format(name[b])
                cv2.imwrite(os.path.join(outDir,saveName),mask[b])
        if 'gen_mask' in out:
            gen_mask = ((1+out['gen_mask'])*127.5).cpu().detach().permute(0,2,3,1).numpy().astype(np.uint8)
            for b in range(batchSize):
                saveName = 'gen_{}_mask.png'.format(name[b])
                cv2.imwrite(os.path.join(outDir,saveName),gen_mask[b])
    #return metricsOut
    for name in losses:
        losses[name] = losses[name].item()
    toRet=   { 
            **losses,
             }
    if 'pred' in out:
        toRet['cer']=sum_cer
        toRet['wer']=sum_wer
    if 'by_author' in config:
        toRet.update(by_author)

    #decode spaced
    #d_spaced = []
    #for b in range(spaced.shape[1]):
    #    string=''
    #    for i in range(spaced.shape[0]):#instance['label_lengths'][b]):
    #        index=spaced[i,b].argmax().item()
    #        if index>0:
    #            string+=trainer.idx_to_char[index]
    #        else:
    #            string+='\0'
    #    d_spaced.append(string)
    return (
             toRet,
             out
            )



def justMask(model,instance,gpu):
    if gpu is not None:
        label = instance['label'].to(gpu)
        image = instance['image'].to(gpu)
    else:
        label = instance['label']
        image = instance['image']
    if 'a_batch_size' in instance:
        a_batch_size = instance['a_batch_size']
    else:
        a_batch_size=None
    label_lengths = instance['label_lengths']
    style = model.extract_style(image,label,a_batch_size)
    if model.spaced_label is None:
        if model.pred is None:
            model.pred = model.hwr(image, None)
        model.spaced_label = model.correct_pred(model.pred,label)
        model.spaced_label = model.onehot(model.spaced_label)
    model.top_and_bottom = model.create_mask(model.spaced_label,style)
    mask = model.write_mask(model.top_and_bottom,image.size())
    import pdb;pdb.set_trace()

    gt_mask = instance['mask']
    mask = ((mask+1)*127.5).numpy().astype(np.uint8)
    gt_mask = ((gt_mask+1)*127.5).numpy().astype(np.uint8)
    for b in range(mask.shape[0]):
        cv2.imshow('pred',mask[b,0])
        cv2.imshow('gt',gt_mask[b,0])
        print('image {}'.format(b))
        cv2.waitKey()
def justSpaced(model,instance,gpu):
    model.count_std=0
    if gpu is not None:
        label = instance['label'].to(gpu)
        image = instance['image'].to(gpu)
    else:
        label = instance['label']
        image = instance['image']
    if model.spaced_label is None:
        if model.pred is None:
            model.pred = model.hwr(image, None)
        model.spaced_label = model.correct_pred(model.pred,label)
        model.spaced_label = model.onehot(model.spaced_label)
    if 'a_batch_size' in instance:
        a_batch_size = instance['a_batch_size']
    else:
        a_batch_size=None
    label_lengths = instance['label_lengths']
    style = model.extract_style(image,label,a_batch_size)
    label_onehot=model.onehot(label)
    model.counts = model.spacer(label_onehot,style)
    spaced = model.insert_spaces(label,label_lengths,model.counts)
    batch_size = label.size(1)
    for b in range(batch_size):
        print('GT')
        print(model.spaced_label[:,b].argmax(1))
        print('Prediction')
        print(spaced[:,b].argmax(1))
def justSpacing(model,instance,gpu):
    model.count_std=0
    if gpu is not None:
        label = instance['label'].to(gpu)
        image = instance['image'].to(gpu)
    else:
        label = instance['label']
        image = instance['image']
    if  model.spacing_pred is None:
        a_batch_size = instance['a_batch_size'] if 'a_batch_size' in instance else None
        mask = None
        center_line = instance['center_line']
        recon,style = model.autoencode(image,label,mask,a_batch_size,center_line=center_line)
    batch_size = label.size(1)
    #model.spacing_pred[:,:,0]=0
    for b in range(batch_size):
        print('Text: {}'.format(instance['gt'][b]))
        print('GT')
        print(model.spaced_label[:,b].argmax(1))
        print('Prediction')
        print(model.spacing_pred[:,b].argmax(1))
