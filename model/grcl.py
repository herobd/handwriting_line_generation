import torch
from torch import nn
import torch.nn.functional as F
from model.net_builder import getGroupSize
#
class GRCL(nn.Module):
    def __init__(self,chIn,chOut,k,T=2,useNorm=False):
        super(GRCL, self).__init__()
        self.T=T
        self.inL = nn.Sequential(
                nn.Conv1d(chIn,chOut,1),
                nn.ReLU6(True)
                )
        self.conv1 = nn.Sequential(
                nn.Conv1d(chOut,chOut,k,padding=k//2),
                nn.ReLU6(True)
                )
        self.gate1 = nn.Sequential(
                nn.Conv1d(chOut,chOut,k,padding=k//2),
                nn.Sigmoid()
                )
        if useNorm:
            self.normA = nn.GroupNorm(getGroupSize(chOut),chOut)
            self.normB = nn.GroupNorm(getGroupSize(chOut),chOut)
        else:
            self.normA = None
            self.normB = None

    def forward(self,x):
        x = self.inL(x)
        if self.normA is not None:
            x = self.normA(x)
        x_orig = x
        for t in range(self.T):
            x_c = self.conv1(x)
            x_g = self.gate1(x)
            x = x_c*x_g
        if self.normB is not None:
            x = self.normB(x)
        return x+x_orig

        #left = self.w_f(x)
        #right = self.w_g_f(x)
        #x_t = F.relu(left)
        #for i in range(self.T):
        #    inner_left = self.w_r(x_t)
        #    inner_right = F.sigmoid(self.w_g_r(x_t) + right)
        #    x_t = F.relu(left + (inner_left*inner_right))

class NewGRCL(nn.Module):
    def __init__(self,chIn,chOut,k,T=2,useNorm=True,inL=False):
        super(NewGRCL, self).__init__()
        self.T=T
        if inL:
            self.inL = nn.Conv1d(chIn,chOut,1)
        else:
            self.inL = None
        self.conv1 = nn.Conv1d(chOut,chOut,k,padding=k//2)
        self.gate1 = nn.Sequential(
                nn.Conv1d(chOut,chOut,k,padding=k//2),
                nn.Sigmoid()
                )
        if useNorm:
            self.act = nn.Sequential(
                    nn.GroupNorm(getGroupSize(chOut),chOut),
                    nn.ReLU6(True)
                    )
        else:
            self.act = nn.ReLU6(True)

    def forward(self,x):
        if self.inL is not None:
            x = self.inL(x)
        x_orig = x
        for t in range(self.T):
            x = self.act(x)
            x_c = self.conv1(x)
            x_g = self.gate1(x)
            x = x_c*x_g
        return x+x_orig
