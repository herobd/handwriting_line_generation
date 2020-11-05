import cv2, math,random
import numpy as np
from skimage.draw import line as sk_line

def tensmeyer_brightness(img, foreground=0, background=0):
    #ret,th = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret,th = cv2.threshold(img ,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if th is None:
        return img

    th = (th.astype(np.float32) / 255)[...,None]
    if len(img.shape)==2:
        img = img[...,None]

    img = img.astype(np.float32)
    img = img + (1.0 - th) * foreground
    img = img + th * background

    img[img>255] = 255
    img[img<0] = 0

    return img.astype(np.uint8)

def apply_tensmeyer_brightness(img, sigma=30, **kwargs):
    random_state = np.random.RandomState(kwargs.get("random_seed", None))
    foreground = random_state.normal(0,sigma)
    background = random_state.normal(0,sigma)

    img = tensmeyer_brightness(img, foreground, background)

    return img


def increase_brightness(img, brightness=0, contrast=1):
    img = img.astype(np.float32)
    img = img * contrast + brightness
    img[img>255] = 255
    img[img<0] = 0

    return img.astype(np.uint8)

def apply_random_brightness(img, b_range=[-50,51], **kwargs):
    random_state = np.random.RandomState(kwargs.get("random_seed", None))
    brightness = random_state.randint(b_range[0], b_range[1])

    img = increase_brightness(img, brightness)

    return input_data

def apply_random_color_rotation(img, **kwargs):
    random_state = np.random.RandomState(kwargs.get("random_seed", None))
    shift = random_state.randint(0,255)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[...,0] = hsv[...,0] + shift
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img


def affine_trans(img,fg_mask,skew,strech):
    m = math.tan(skew)
    h = img.shape[0]/2
    #print('skew:{}, m:{}, strech:{}, h:{}'.format(skew,m,strech,h))
    matrix = np.array( [[strech, m, -h*m],
                        [0,      1, 0]] )
    shape = (int(img.shape[1]*strech),img.shape[0])
    img= cv2.warpAffine(img, matrix, shape, borderValue=255)
    if fg_mask is not None:
        fg_mask= cv2.warpAffine(fg_mask, matrix, shape, borderValue=0)
    return img,fg_mask


def change_thickness(img,size,fg_shade,bg_shade,blur_size,noise_sigma):
    th,new_img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    new_img = 255-new_img
    rad = abs(size) #random.randint(1,size)
    if rad>1:
        ele = cv2.getStructuringElement(  cv2.MORPH_ELLIPSE, (rad,rad) )
        if size>0:
            new_img = cv2.dilate(new_img,ele)
        else:
            summed = new_img.sum()
            temp = new_img
            new_img = cv2.erode(new_img,ele)
            if new_img.sum() < 0.1*summed:
                new_img=temp
    new_img = new_img*1.0/255
    new_img *= fg_shade-bg_shade
    new_img += bg_shade

    rad = blur_size#random.randint(1,blur_size)
    if rad>1:
        new_img=cv2.blur(new_img,(blur_size,blur_size))
    
    #sigma = random.random()*sigma_max
    new_img += np.random.normal(0,noise_sigma,new_img.shape)
    new_img[new_img<0]=0
    new_img[new_img>1]=1

    return new_img

def add_random_lines(img):
    if img.shape[1]<=5:
        return img
    num_lines = random.randint(1,12)
    #print('num: {}'.format(num_lines))
    for i in range(num_lines):
        r = random.random()
        if r<0.4: #horizontal
            angle = random.gauss(0,7) #degrees
            dist = random.gauss(img.shape[1],img.shape[1]*0.4)
        elif r<0.8: #vertivle
            angle = random.gauss(90,7) #degrees
            dist = random.gauss(img.shape[0],img.shape[0]*0.4)
        else: #random
            angle = random.random()*180
            dist = random.gauss(0,1)*max(img.shape)*0.4 + min(img.shape)
        if dist<=0:
            continue

        if random.random()<0.5:
            angle += 180

        angle = angle/180 * np.pi

        type_line = random.choice(['solid','dotted','dashed'])
        thickness = random.randint(1,8)
        blockStart = -thickness//2
        blockEnd = thickness//2 +thickness%2
        darkness = random.random()

        line_img = np.ones(img.shape,np.float64)
        startX = random.randint(0,img.shape[1])
        startY = random.randint(0,img.shape[0])
        endX = round(startX + math.cos(angle)*dist)
        endY = round(startY + math.sin(angle)*dist)

        #print('({},{}) - ({},{}) t:{}, c:{}'.format(startX,startY,endX,endY,thickness,darkness))

        for x,y in zip(*sk_line(startX,startY,endX,endY)):
            if x>=0 and x<img.shape[1] and y>=0 and y<img.shape[0]:
                xBlockStart = min(max(x+blockStart,0),img.shape[1])
                xBlockEnd = min(max(x+blockEnd,0),img.shape[1])
                yBlockStart = min(max(y+blockStart,0),img.shape[1])
                yBlockEnd = min(max(y+blockEnd,0),img.shape[1])
                #print('w [{}:{}, {}:{}
                line_img[yBlockStart:yBlockEnd,xBlockStart:xBlockEnd]=darkness

        #smooth
        line_img = cv2.blur(line_img,(3,3))

        #cv2.imshow('line{}'.format(i),(line_img*255).astype(np.int32))
        
        #cv2.imshow('line{}'.format(i),line_img)

        #add or multiply original image?
        if random.random()<0.5:
            img = (img.astype(np.float64)*line_img).astype(np.uint32)
        else:
            img=img.astype(np.int64)
            img -= ((1-line_img)*220).astype(np.int64)
            img = np.clip(img,0,255).astype(np.uint32)
    #cv2.waitKey()

    return img

def mmd_crop(img):
    if random.random()<0.1 or img.shape[1]<=img.shape[0]:
        return img
    else:
        #in the mmd dataset, we have a lot of excess space on the ends, I'd like to remove that
        profile = img.sum(axis=0)
        kernel = np.array([-1,-1,-1,-1,0,1,1,1,1.0])
        edges = cv2.filter2D(profile.astype(np.float32),-1,kernel)
        assert(edges.shape[0]==profile.shape[0])
        #print([int(edges[i]) for i in range(profile.shape[0])])
        #print('max: {}, mean:{}'.format(edges.max(),edges.mean()))
        edges=np.absolute(edges)
        thresh=500
        max_edge=0
        edge_x=0
        for x in range(6,min(int(img.shape[0]*1.5),profile.shape[0])):
            if edges[x]>thresh and edges[x]>max_edge:
                max_edge=edges[x]
                edge_x=x
            elif edges[x]+10<max_edge:
                break
        max_edge=0
        edge_xr=profile.shape[0]-1
        for x in range(profile.shape[0]-7,profile.shape[0]//2,-1):
            if edges[x]>thresh and edges[x]>max_edge:
                max_edge=edges[x]
                edge_xr=x
            elif edges[x]+10<max_edge:
                break
        imgCrop = img[:,edge_x:edge_xr+1]
        if imgCrop.shape[1]<5:
            return img

        return imgCrop
def bad_crop(img):
    if random.random()<0.1:
        return img
    else:


        h = img.shape[0]
        if random.random()<0.7:
            cropTop = round(random.random()*0.3*h)
        else:
            cropTop = 0
        if random.random()<0.7:
            cropBot = round(-random.random()*0.3*h)
        else:
            cropBot = 0
        if random.random()<0.7 and img.shape[1]>=h*0.9:
            cropLeft = round(random.random()*0.3*h)
        else:
            cropLeft = 0
        if random.random()<0.7 and img.shape[1]>=h*0.9:
            cropRight = round(-random.random()*0.3*h)
        else:
            cropRight = 0

        if cropRight==0:
            cropRight=img.shape[1]
        if random.random()<0.6:
            if cropBot==0:
                cropBot=img.shape[0]
            img = img[cropTop:cropBot,cropLeft:cropRight]
        else:
            img = img[:,cropLeft:cropRight]
            img = np.pad(img,((cropTop,-cropBot),(0,0)),mode='constant',constant_values=img.mean())
        new_h = img.shape[0]
        if new_h==h:
            assert(img.shape[0]>0 and img.shape[1]>0)
            return img
        else:
            ratio = h/new_h
            if new_h ==0 or img.shape[1]==0:
                assert(img.shape[0]>1 and img.shape[1]>1)
                return img
            img =  cv2.resize(img.astype(np.float32),(0,0), fx=ratio, fy=ratio, interpolation = cv2.INTER_CUBIC).astype(np.uint32)
            assert(img.shape[0]>1 and img.shape[1]>1)
            return img
