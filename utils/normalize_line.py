import cv2, math
import numpy as np
import torch
import torch.nn.functional as F
from skimage.morphology import skeletonize as skeletonize_ski

#we'll assume a gray image, [rows,cola,1s]
#angled profile take from https://stackoverflow.com/a/7880726/1018830
def deskew(img,angle_range=0.38, angle_step=0.076, x_step=2): #range 22 degrees, each direction

    v_img = 1 - img/255
    v_img = cv2.GaussianBlur(    v_img, (0,0), 1.5)

    #This will be a two pass method. Course, then refine

    
    max_var=0
    for angle in np.arange(-angle_range,angle_range+0.001,angle_step):
        values=[]
        x_diff = math.tan(angle)*img.shape[0]-1
        for x in range(0,img.shape[1],x_step):
            x_end=x+x_diff
            if x_end<img.shape[1]:
                shift_x = x_end-x
                length = int(np.hypot(x_end-x, img.shape[0]-1))
                xL, yL = np.linspace(x, x_end, length), np.linspace(0, img.shape[0]-1, length)

                v = v_img[yL.astype(np.int), xL.astype(np.int)].sum()/img.shape[0]
                values.append(v)
        var = np.var(values)
        #print('{}: {}'.format(angle,var))
        if var>max_var:
            max_var=var
            best_angle = angle
            best_shift = x_diff
    #print('max var: {}, angle: {}, shift: {}'.format(max_var,best_angle,best_shift))
    max_var=0
    for angle in np.arange(best_angle-angle_step,best_angle+angle_step+0.001,angle_step/3):
        values=[]
        x_diff = math.tan(angle)*img.shape[0]-1
        for x in range(0,img.shape[1],max(1,x_step//2)):
            x_end=x+x_diff
            if x_end<img.shape[1]:
                shift_x = x_end-x
                length = int(np.hypot(x_end-x, img.shape[0]-1))
                xL, yL = np.linspace(x, x_end, length), np.linspace(0, img.shape[0]-1, length)

                v = v_img[yL.astype(np.int), xL.astype(np.int)].sum()/img.shape[0]
                values.append(v)
        var = np.var(values)
        if var>max_var:
            max_var=var
            best_angle2 = angle
            best_shift = x_diff
    #print('REFINE max var: {}, angle: {}, shift: {}'.format(max_var,best_angle,best_shift))


    M = np.array([[1, math.tan(-best_angle), best_shift/2],
                  [0, 1, 0]])

    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),borderValue=255)

    return img


def skeletonize(img):

    #binarize
    ret,th = cv2.threshold(255-img ,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    skeleton = skeletonize_ski(th)

    skeleton = torch.from_numpy(skeleton*255)[None,None,...]
    morph_kernel_dilate=3
    dilate_weights = torch.FloatTensor(1,1,morph_kernel_dilate,morph_kernel_dilate)
    r = morph_kernel_dilate//2
    for x in range(morph_kernel_dilate):
        for y in range(morph_kernel_dilate):
            dilate_weights[0,0,y,x] = float(((y-r)**2 + (x-r)**2) <= (r**2))
    out = F.conv_transpose2d(skeleton.float(),dilate_weights,stride=1,padding=1)#,padding=morph_padding)

    blur_kernel = 3
    blur_padding = blur_kernel // 2
    blur = torch.nn.AvgPool2d((blur_kernel,blur_kernel), stride=1, padding=(blur_padding,blur_padding))
    return 255-blur(out)[0,0].numpy()
