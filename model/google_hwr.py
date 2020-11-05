import torch
from torch import nn
from .net_builder import getGroupSize
from .grcl import GRCL

# based on A Scalable Handwritten Text Recognition System
class GoogleHWR(nn.Module):

    def __init__(self, nclass, nc=1, inceptionNorm=False, grclNorm=False,reducedContext=False):
        super(GoogleHWR, self).__init__()
        # assert imgH % 16 == 0, 'imgH has to be a multiple of 16'


        self.cnn = nn.Sequential(
                nn.Conv2d(nc, 256,7, 2, 3), #32xW/2
                Inception(256,32,64,useNorm=inceptionNorm),
                GooglePool(256,256), #16xW/4
                Inception(512,64,128,useNorm=inceptionNorm),
                GooglePool(512,256,1), #8xW/4
                Inception(512,64,128,useNorm=inceptionNorm),
                GoogleReduce(512,5,3,4,8)
                )

        if reducedContext:
            self.cnn1d = nn.Sequential(
                            GRCL(512,512,3,useNorm=grclNorm),
                            GRCL(512,512,3,useNorm=grclNorm),
                            GRCL(512,256,5,useNorm=grclNorm),
                            GRCL(256,128,7,useNorm=grclNorm),
                            nn.Conv1d(128,nclass,1),
                            nn.LogSoftmax(dim=1)
                        )
        else:
            self.cnn1d = nn.Sequential(
                            GRCL(512,512,3,useNorm=grclNorm),
                            GRCL(512,512,3,useNorm=grclNorm),
                            GRCL(512,512,3,useNorm=grclNorm),
                            GRCL(512,512,3,useNorm=grclNorm),
                            GRCL(512,256,5,useNorm=grclNorm),
                            GRCL(256,256,5,useNorm=grclNorm),
                            GRCL(256,256,5,useNorm=grclNorm),
                            GRCL(256,256,5,useNorm=grclNorm),
                            GRCL(256,128,7,useNorm=grclNorm),
                            GRCL(128,128,7,useNorm=grclNorm),
                            GRCL(128,128,7,useNorm=grclNorm),
                            GRCL(128,128,7,useNorm=grclNorm),
                            nn.Conv1d(128,nclass,1),
                            nn.LogSoftmax(dim=1)
                        )


    def forward(self, input, style=None):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        # assert h == 1, "the height of conv must be 1"
        conv = conv.view(b, -1, w)
        conv = self.cnn1d(conv)
        output = conv.permute(2, 0, 1)  # [w, b, c]

        return output

    def setup_save_features(self):
        save_from = [12]
        self.saved_features = [None]*len(save_from)
        def factorySave(i):
            def saveX(module, input ,output):
                self.saved_features[i]=output
            return saveX
        for i,layer in enumerate(save_from):
            self.cnn[layer].register_forward_hook( factorySave(i) )


class Inception(nn.Module):
    def __init__(self,chIn, d1, d2, useNorm=False, useNormEnd=False):
        super(Inception, self).__init__()
        if useNorm:
            self.conv1 = nn.Sequential(
                    nn.Conv2d(chIn,d2,1),
                    nn.GroupNorm(getGroupSize(d2),d2),
                    nn.ReLU6(True)
                    )
            self.conv2 = nn.Sequential(
                    nn.Conv2d(chIn,d1,1),
                    nn.ReLU6(True),
                    nn.Conv2d(d1,d2,3,padding=1),
                    nn.GroupNorm(getGroupSize(d2),d2),
                    nn.ReLU6(True)
                    )
            self.conv3 = nn.Sequential(
                    nn.Conv2d(chIn,d1,1),
                    nn.ReLU6(True),
                    nn.Conv2d(d1,d2,5,padding=2),
                    nn.GroupNorm(getGroupSize(d2),d2),
                    nn.ReLU6(True)
                    )
            self.conv4 = nn.Sequential(
                    nn.AvgPool2d(3,1,padding=1),
                    nn.Conv2d(chIn,d2,1),
                    nn.GroupNorm(getGroupSize(d2),d2),
                    nn.ReLU6(True),
                    )
        else:
            self.conv1 = nn.Sequential(
                    nn.Conv2d(chIn,d2,1),
                    nn.ReLU6(True)
                    )
            self.conv2 = nn.Sequential(
                    nn.Conv2d(chIn,d1,1),
                    nn.ReLU6(True),
                    nn.Conv2d(d1,d2,3,padding=1),
                    nn.ReLU6(True)
                    )
            self.conv3 = nn.Sequential(
                    nn.Conv2d(chIn,d1,1),
                    nn.ReLU6(True),
                    nn.Conv2d(d1,d2,5,padding=2),
                    nn.ReLU6(True)
                    )
            self.conv4 = nn.Sequential(
                    nn.AvgPool2d(3,1,padding=1),
                    nn.Conv2d(chIn,d2,1),
                    nn.ReLU6(True),
                    )
        if useNormEnd:
            self.norm = nn.GroupNorm(getGroupSize(d1+d2+d1+d2),d1+d2+d1+d2)
        else:
            self.norm = None
            
    def forward(self,x):
        result = [
                self.conv1(x),
                self.conv2(x),
                self.conv3(x),
                self.conv4(x)
                ]
        result = torch.cat(result,dim=1)
        if self.norm is not None:
            result = self.norm(result)
        return x+result

class GooglePool(nn.Module):
    def __init__(self,chIn,d,w=2):
        super(GooglePool, self).__init__()
        self.convS = nn.Sequential(
                nn.Conv2d(chIn,d,3,(2,w),padding=(1,w//2)),
                nn.ReLU6(True)
                )
        self.convP = nn.Sequential(
                nn.Conv2d(chIn,d,1),
                nn.ReLU6(True),
                nn.MaxPool2d(3,(2,w),padding=(1,w//2))
                )

    def forward(self,x):
        return torch.cat( (
                            self.convS(x),
                            self.convP(x)
                          ), dim=1)
class GoogleReduce(nn.Module):
    def __init__(self,chIn,f1,f2,f3,f4):
        super(GoogleReduce, self).__init__()
        self.convL = nn.Sequential(
                nn.Conv2d(chIn,128,1,1,padding=0),
                nn.ReLU6(True),
                nn.Conv2d(128,128,(f1,1),1,padding=0),
                nn.ReLU6(True),
                nn.Conv2d(128,128,(1,f2),1,padding=(0,f2//2)),
                nn.ReLU6(True),
                nn.Conv2d(128,256,(f3,1),1,padding=0),
                nn.ReLU6(True),
                )
        self.convR = nn.Sequential(
                nn.Conv2d(chIn,256,(f4,1),1,padding=0),
                nn.ReLU6(True),
                )

    def forward(self,x):
        return torch.cat( (
                            self.convL(x),
                            self.convR(x)
                          ), dim=1)
