# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
import torch.nn.functional as F
import torch
import utils
#import torch.nn as nn
#from model.alignment_loss import alignment_loss, box_alignment_loss, iou_alignment_loss
#from model.yolo_loss import YoloLoss, YoloDistLoss, LineLoss
#from model.lf_loss import point_loss as lf_point_loss
#from model.lf_loss import special_loss as lf_line_loss
#from model.lf_loss import xyrs_loss as lf_xyrs_loss
#from model.lf_loss import end_pred_loss as lf_end_loss
#from torch.nn import CTCLoss, L1Loss
from model.aligned_l1_loss import alignedL1Loss
from model.pyramid_l1_loss import pyramidL1Loss
from model.dtw_loss import DTWLoss
from model.key_loss import pushMinDist
from torch.nn.functional import cross_entropy

def my_loss(y_input, y_target):
    return F.nll_loss(y_input, y_target)

def sigmoid_BCE_loss(y_input, y_target):
    return F.binary_cross_entropy_with_logits(y_input, y_target)
def MSE(y_input, y_target):
    return F.mse_loss(y_input, y_target.float())
def MSELoss(y_input, y_target):
    return F.mse_loss(y_input, y_target.float())

def L1Loss(input,target):
    return F.l1_loss(input,target)
def HingeLoss(input,target,threshold):
    diff = torch.abs(input-target)
    diff[diff<threshold] = 0
    return diff.mean()
def CTCLoss(input,target,input_len,target_len):
    return F.ctc_loss(input,target,input_len,target_len)

def CrossEntropyLoss1D(input,target, blank_weight=1):
    #press space into batch
    num_class = input.size(1)
    batch_size = input.size(0)
    space_len = input.size(2)
    input = input.permute(0,2,1).contiguous().view(batch_size*space_len,num_class)
    target = target.view(batch_size*space_len)
    if blank_weight!=1:
        if blank_weight=='even':
            blank_loc = target==0
            blank_weight = 1/blank_loc.sum()
        elif blank_weight=='better':
            blank_loc = target==0
            blank_weight = (space_len-blank_loc.sum())/space_len
        elif blank_weight=='max':
            blank_loc = target==0
            blank_weight = (space_len-blank_loc.sum())/space_len
            blank_wieght = max(0.1,blank_weight)
        #weights = torch.FloatTensor(target.size()).one_()
        #weights[blank_loc]=blank_weight
        weights = torch.FloatTensor(num_class).fill_(1)
        weights[0]=blank_weight
        return F.cross_entropy(input,target,weight=weights.to(target.device))
    else:
        return F.cross_entropy(input,target)

#def detect_alignment_loss(predictions, target,label_sizes,alpha_alignment, alpha_backprop):
#    return alignment_loss(predictions, target, label_sizes, alpha_alignment, alpha_backprop)
#def detect_alignment_loss_points(predictions, target,label_sizes,alpha_alignment, alpha_backprop):
#    return alignment_loss(predictions, target, label_sizes, alpha_alignment, alpha_backprop,points=True)

#def lf_point_loss(prediction,target):
#    return point_loss(prediction,target)
#def lf_line_loss(prediction,target):
#    return special_loss(prediction,target)
#def lf_xyrs_loss(prediction,target):
#    return xyrs_loss(prediction,target)
#def lf_end_loss(end_pred,path_xyxy,end_point):
    #    return end_pred_loss(end_pred,path_xyxy,end_point)
