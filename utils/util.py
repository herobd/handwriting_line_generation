import os, math
import struct
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from skimage import draw

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def pt_xyrs_2_xyxy(state):
    out = torch.ones(state.data.shape[0], 5).type(state.data.type())

    x = state[:,:,1:2]
    y = state[:,:,2:3]
    r = state[:,:,3:4]
    s = state[:,:,4:5]

    x0 = -torch.sin(r) * s + x
    y0 = -torch.cos(r) * s + y
    x1 =  torch.sin(r) * s + x
    y1 =  torch.cos(r) * s + y

    return torch.cat([
        state[:,:,0:1],
        x0, y0, x1, y1
    ], 2)
def pt_xyxy_2_xyrs(state):
    out = torch.ones(state.data.shape[0], 5).type(state.data.type())
            
    x0 = state[:,0:1]
    y0 = state[:,1:2]
    x1 = state[:,2:3]
    y1 = state[:,3:4]

    dx = x0-x1
    dy = y0-y1

    d = torch.sqrt(dx**2.0 + dy**2.0)/2.0

    mx = (x0+x1)/2.0
    my = (y0+y1)/2.0

    theta = -torch.atan2(dx, -dy)

    return torch.cat([
        mx, my, theta, d,
        state[:,4:5]
    ], 1)





def makeMask(image, post=[], random=False):
    batch_size=image.size(0)
    if random:
        morph_kernel_dilate = 2*np.random.randint(8,20)+1#11
        if random=='more':
            morph_kernel_errode = morph_kernel_dilate + 2*np.random.randint(-3,4)
        else:
            morph_kernel_errode = morph_kernel_dilate
        h_kernel = 2*np.random.randint(10,20)+1#31
        v_kernel = h_kernel//4 if h_kernel//4%2==1 else h_kernel//4 +1
    else:
        morph_kernel_dilate = 25
        morph_kernel_errode = 25
        h_kernel = 31
        v_kernel = h_kernel//4
    #morph_padding_dilate = morph_kernel_dilate//2
    #morph_padding_errode = morph_kernel_errode//2
    morph_diff = morph_kernel_errode-morph_kernel_dilate
    morph_padding_errode=0
    morph_padding_dilate=0
    if morph_diff>0:
        morph_padding_errode=morph_diff//2
    elif morph_diff<0:
        morph_padding_dilate=-morph_diff//2
    h_padding = h_kernel // 2
    v_padding = v_kernel // 2

    if len(post)>0 and post[0] == 'true':
        post = post[1:]
        h_kernel=3
        v_kernel=3
        h_padding=1
        v_padding=1
        pool = torch.nn.MaxPool2d((v_kernel,h_kernel), stride=1, padding=(v_padding,h_padding))
        blur_kernel = 3
        blur_padding = blur_kernel // 2
        blur = torch.nn.AvgPool2d((blur_kernel,blur_kernel), stride=1, padding=(blur_padding,blur_padding))
    else:
        #pool = torch.nn.MaxPool2d((kernel,kernel//4), stride=1, padding=(padding,padding//4))
        pool = torch.nn.MaxPool2d((v_kernel,h_kernel), stride=1, padding=(v_padding,h_padding))

        blur_kernel = 31
        blur_padding = blur_kernel // 2
        blur = torch.nn.AvgPool2d((blur_kernel//4,blur_kernel//4), stride=1, padding=(blur_padding//4,blur_padding//4))
    #dilate = torch.nn.Conv2D(11,1,5)
    
    pt_img = pool(image)
    #pt_img = blur(image)
    out = torch.empty_like(image)
    for i in range(batch_size):
        #pt_img_b = pt_img.permute(0,2,3).numpy()[i,0]
        pt_img_b = pt_img.numpy()[i,0]
        cummax_img0 = np.maximum.accumulate(pt_img_b, axis=0)
        cummax_img1 = np.maximum.accumulate(pt_img_b[::-1], axis=0)[::-1]
        cummax_img2 = np.maximum.accumulate(pt_img_b, axis=1)
        cummax_img3 = np.maximum.accumulate(pt_img_b[::-1], axis=1)[::-1]
        result = np.minimum(np.minimum(cummax_img0, cummax_img1), np.minimum(cummax_img2, cummax_img3))
        out[i,0] = torch.from_numpy(result)#.permute(2,0,1)
    for task in post:
        if task=='thresh':
            out = out>0.1
        elif task=='smaller':
            morph_kernel_dilate = morph_kernel_dilate//2 +1
            morph_kernel_errode = morph_kernel_errode//2 +1
        elif task=='errode':
            errode_weights = torch.FloatTensor(1,1,morph_kernel_errode,morph_kernel_errode).fill_(1)
            #one-pad to allow edge pixels to be kept
            #out = F.pad(out,(morph_padding,morph_padding,morph_padding,morph_padding),value=1)
            out = F.conv2d(out.float(),errode_weights,stride=1)#,padding=morph_padding)
            out = out>=(morph_kernel_errode**2)
        elif task=='errodeCircle':
            errode_weights = torch.FloatTensor(1,1,morph_kernel_errode,morph_kernel_errode)
            r = morph_kernel_errode//2
            for x in range(morph_kernel_errode):
                for y in range(morph_kernel_errode):
                    errode_weights[0,0,y,x] = float(((y-r)**2 + (x-r)**2) <= (r**2))
            out = F.conv2d(out.float(),errode_weights,stride=1,padding=morph_padding_errode)#,padding=morph_padding)
            out = out>=errode_weights.sum()
        elif task=='dilate':
            dilate_weights = torch.FloatTensor(1,1,morph_kernel_dilate,morph_kernel_dilate).fill_(1)
            out = F.conv_transpose2d(out.float(),dilate_weights,stride=1)#,padding=morph_padding)
            out = out>0.1
        elif task=='dilateCircle':
            dilate_weights = torch.FloatTensor(1,1,morph_kernel_dilate,morph_kernel_dilate)
            r = morph_kernel_dilate//2
            for x in range(morph_kernel_dilate):
                for y in range(morph_kernel_dilate):
                    dilate_weights[0,0,y,x] = float(((y-r)**2 + (x-r)**2) <= (r**2))
            out = F.conv_transpose2d(out.float(),dilate_weights,stride=1,padding=morph_padding_dilate)#,padding=morph_padding)
            out = out>0.1
        elif task=='distance':
            out = out.numpy()
            height = out.shape[2]
            width = out.shape[3]
            window = 3*height
            dists = np.empty(out.shape,np.float32)
            for b in range(batch_size):
                line_im = np.ones((height,width),np.uint8)
                #get mediana
                medians=[]
                sum_x=0
                sum_y=0
                count=1
                y_indexes = np.arange(height)[:,None].repeat(window,axis=1)
                x_indexes = np.arange(window)[None,:].repeat(height,axis=0)
                for x_start in range(0,width-window,window//2):
                    on=out[b,0,:,x_start:x_start+window].sum()
                    if on>0:
                        med_x = (x_indexes*out[b,0,:,x_start:x_start+window]).sum()/on + x_start
                        med_y = (y_indexes*out[b,0,:,x_start:x_start+window]).sum()/on
                        medians.append((med_y,med_x))#x_start+window//2))
                        sum_x+=med_x
                        sum_y+=med_y
                        #assert(med_y<height)
                        count+=1
                med_x = sum_x/count
                med_y = sum_y/count
                if len(medians)>1:
                    slope = (medians[1][0]-medians[0][0])/(medians[1][1]-medians[0][1])
                    distance= -medians[0][1]
                    front_point = [ (med_y + medians[0][0]+slope*distance)/2, 0]
                    slope = (medians[-1][0]-medians[-2][0])/(medians[-1][1]-medians[-2][1])
                    distance= width-1 - medians[-1][1]
                    last_point = [ (med_y + medians[-1][0]+slope*distance)/2, width-1]
                    if last_point[0]<0 or last_point[0]>=height:
                        last_point = (med_y,width-1)
                else:
                    front_point = [med_y,med_x]
                    last_point = [med_y,med_x]
                medians = [front_point]+medians+[last_point]
                for i in range(0,len(medians)-1):
                    if math.isnan(medians[i][0]):
                        medians[i][0] = medians[i+1][0]
                    if math.isnan(medians[i][1]):
                        medians[i][1] = medians[i+1][1]
                for i in range(len(medians)-1,0,-1):
                    if math.isnan(medians[i][0]):
                        medians[i][0] = medians[i-1][0]
                    if math.isnan(medians[i][1]):
                        medians[i][1] = medians[i-1][1]
                for i in range(1,len(medians)): 
                    rr,cc = draw.line(int(medians[i-1][0]),int(medians[i-1][1]),int(medians[i][0]),int(medians[i][1]))
                    line_im[rr,cc]=0
                dists[b] = distance_transform_edt(line_im)
                #out[b,:] = line_im
            max_dist = height//2
            dists/=max_dist
            dists[dists>1]=1
            dists = 1-dists
            out = torch.from_numpy(dists)



        else:
            raise NotImplementedError('unknown makeMask post operation: {}'.format(task))


    if len(post)>0:
        #centersV_t, medians = getCenterValue(out)
        #centersV_t = torch.from_numpy(centersV_t)
        centersV_t = torch.from_numpy(getCenterValue(out))
        centerV = centersV_t[:,None,...]
        height = out.size(2)
        width = out.size(3)
        #ranges = (np.arange(height)+1)[None,None,...,None].repeat(batch_size,axis=0).repeat(width,axis=3)
        ranges = (torch.arange(height)+1)[None,None,...,None].expand(batch_size,-1,-1,width)
        mask_ranges = ranges*out.long()
        bottom = mask_ranges.argmax(dim=2)
        bottom_not_valid = 0==mask_ranges.max(dim=2)[0]
        mask_ranges = ((height+1)-ranges)*out.long()
        top = mask_ranges.argmax(dim=2)
        top_not_valid = 0==mask_ranges.max(dim=2)[0]

        #top_and_bottom = np.concatecate((centerV-top,bottom-centerV),axis=1)
        top_and_bottom = torch.cat((centerV-top.float(),bottom.float()-centerV),dim=1)
        top_and_bottom[:,0][top_not_valid[:,0,:]]=0
        top_and_bottom[:,1][bottom_not_valid[:,0,:]]=0
        if top_and_bottom.max()>200 or top_and_bottom.min()<-200:
            import pdb;pdb.set_trace()
        out = 2*out.float()-1
    else:
        top_and_bottom = None
        centersV_t = None
    #assert(out.size(2)==64)
    return blur(out),top_and_bottom,centersV_t #, medians


def getCenterValue(mask):
    #out = mask>0.8
    mask = mask.numpy()
    batch_size = mask.shape[0]
    height = mask.shape[2]
    width = mask.shape[3]
    window = 3*height
    centers = np.zeros([mask.shape[0],mask.shape[3]],np.float32)
    centers[:] = height/2
    #all_medians=[]
    for b in range(batch_size):
        #line_im = np.ones((height,width),np.uint8)
        #get mediana
        medians=[]
        sum_x=0
        sum_y=0
        count=1
        y_indexes = np.arange(height)[:,None].repeat(window,axis=1)
        x_indexes = np.arange(window)[None,:].repeat(height,axis=0)
        for x_start in range(0,width-window,window//2):
            on=mask[b,0,:,x_start:x_start+window].sum()
            if on>0:
                med_x = (x_indexes*mask[b,0,:,x_start:x_start+window]).sum()/on + x_start
                med_y = (y_indexes*mask[b,0,:,x_start:x_start+window]).sum()/on
                medians.append((med_y,med_x))#x_start+window//2))
                sum_x+=med_x
                sum_y+=med_y
                #assert(med_y<height)
                count+=1
        med_x = sum_x/count
        med_y = sum_y/count
        if len(medians)>1:
            slope = (medians[1][0]-medians[0][0])/(medians[1][1]-medians[0][1])
            distance= -medians[0][1]
            front_point = [ (med_y + medians[0][0]+slope*distance)/2, 0]
            slope = (medians[-1][0]-medians[-2][0])/(medians[-1][1]-medians[-2][1])
            distance= width-1 - medians[-1][1]
            last_point = [ (med_y + medians[-1][0]+slope*distance)/2, width-1]
            if last_point[0]<0 or last_point[0]>=height:
                last_point = (med_y,width-1)
        else:
            on=mask[b,0].sum()
            if on == 0:
                front_point = [height/2,0]
                last_point = [height/2,width-1]
            else:
                y_indexes = np.arange(height)[:,None].repeat(width,axis=1)
                x_indexes = np.arange(width)[None,:].repeat(height,axis=0)
                med_x = (x_indexes*mask[b,0]).sum()/on
                med_y = (y_indexes*mask[b,0]).sum()/on
                front_point = [med_y,0]
                last_point = [med_y,width-1]
        medians = [front_point]+medians+[last_point]
        for i in range(0,len(medians)-1):
            if math.isnan(medians[i][0]):
                medians[i][0] = medians[i+1][0]
            if math.isnan(medians[i][1]):
                medians[i][1] = medians[i+1][1]
        for i in range(len(medians)-1,0,-1):
            if math.isnan(medians[i][0]):
                medians[i][0] = medians[i-1][0]
            if math.isnan(medians[i][1]):
                medians[i][1] = medians[i-1][1]
        for i in range(1,len(medians)): 
            rr,cc = draw.line(int(medians[i-1][0]),int(medians[i-1][1]),int(medians[i][0]),int(medians[i][1]))
            #line_im[rr,cc]=0
            centers[b][cc]=rr
        #all_medians.append(medians)
    return centers#, all_medians

#-------------------------------------------------------------------------------
# Name:        get_image_size
# Purpose:     extract image dimensions given a file path using just
#              core modules
#
# Author:      Paulo Scardine (based on code from Emmanuel VAÏSSE)
#
# Created:     26/09/2013
# Copyright:   (c) Paulo Scardine 2013
# Licence:     MIT
# From:        https://stackoverflow.com/questions/15800704/get-image-size-without-loading-image-into-memory
#-------------------------------------------------------------------------------
class UnknownImageFormat(Exception):
    pass

def get_image_size(file_path):
    """
    Return (width, height) for a given img file content - no external
    dependencies except the os and struct modules from core
    """
    size = os.path.getsize(file_path)

    with open(file_path) as input:
        height = -1
        width = -1
        data = input.read(25)

        if (size >= 10) and data[:6] in ('GIF87a', 'GIF89a'):
            # GIFs
            w, h = struct.unpack("<HH", data[6:10])
            width = int(w)
            height = int(h)
        elif ((size >= 24) and data.startswith('\211PNG\r\n\032\n')
              and (data[12:16] == 'IHDR')):
            # PNGs
            w, h = struct.unpack(">LL", data[16:24])
            width = int(w)
            height = int(h)
        elif (size >= 16) and data.startswith('\211PNG\r\n\032\n'):
            # older PNGs?
            w, h = struct.unpack(">LL", data[8:16])
            width = int(w)
            height = int(h)
        elif (size >= 2) and data.startswith('\377\330'):
            # JPEG
            msg = " raised while trying to decode as JPEG."
            input.seek(0)
            input.read(2)
            b = input.read(1)
            try:
                while (b and ord(b) != 0xDA):
                    while (ord(b) != 0xFF): b = input.read(1)
                    while (ord(b) == 0xFF): b = input.read(1)
                    if (ord(b) >= 0xC0 and ord(b) <= 0xC3):
                        input.read(3)
                        h, w = struct.unpack(">HH", input.read(4))
                        break
                    else:
                        input.read(int(struct.unpack(">H", input.read(2))[0])-2)
                    b = input.read(1)
                width = int(w)
                height = int(h)
            except struct.error:
                raise UnknownImageFormat("StructError" + msg)
            except ValueError:
                raise UnknownImageFormat("ValueError" + msg)
            except Exception as e:
                raise UnknownImageFormat(e.__class__.__name__ + msg)
        else:
            raise UnknownImageFormat(
                "Sorry, don't know how to get information from this file."
            )

    return width, height

def getGroupSize(channels):
    if channels>=32:
        goalSize=8
    else:
        goalSize=4
    if channels%goalSize==0:
        return goalSize
    factors=primeFactors(channels)
    bestDist=9999
    for f in factors:
        if abs(f-goalSize)<=bestDist: #favor larger
            bestDist=abs(f-goalSize)
            bestGroup=f
    return int(bestGroup)
