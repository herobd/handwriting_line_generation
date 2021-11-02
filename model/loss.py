import torch.nn.functional as F
import torch
import utils
from torch.nn.functional import cross_entropy

def sigmoid_BCE_loss(y_input, y_target):
    return F.binary_cross_entropy_with_logits(y_input, y_target)
def MSE(y_input, y_target):
    return F.mse_loss(y_input, y_target.float())
def MSELoss(y_input, y_target):
    return F.mse_loss(y_input, y_target.float())
def CrossEntropyLoss(input,target):
    return F.cross_entropy(input,target)

def L1Loss(input,target):
    return F.l1_loss(input,target)
def HingeLoss(input,target,threshold):
    diff = torch.abs(input-target)
    diff[diff<threshold] = 0
    return diff.mean()
def AdaptiveHingeLoss(input,target,threshold):
    batch_size=target.size(0)
    diff = torch.abs(input-target)
    std = torch.std(diff.view(batch_size,-1),dim=1)[:,None,None,None]
    mean = torch.mean(diff.view(batch_size,-1),dim=1)[:,None,None,None]
    diff[torch.abs(diff-mean)/std<threshold] = 0
    return diff.mean()
def CTCLoss(input,target,input_len,target_len):
    ret = F.ctc_loss(input,target,input_len,target_len)
    return torch.where(torch.isinf(ret), torch.zeros_like(ret), ret)

