# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
import torch
from torch import nn
from .net_builder import getGroupSize


class CNNOnlyHWR(nn.Module):

    def __init__(self, nclass, nc=1, cnnOutSize=512, nh=512, leakyRelu=False, norm='group'):
        super(CNNOnlyHWR, self).__init__()
        # assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 3]
        ps = [1, 1, 1, 1, 1, 0, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, norm=None):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if norm is not None and 'group' in norm:
                cnn.add_module('groupnorm{0}'.format(i), nn.GroupNorm(getGroupSize(nOut),nOut))
            elif norm:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x32x12c
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x16x6c
        convRelu(2, norm)
        convRelu(3)       
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x8x6c
        convRelu(4, norm)
        convRelu(5)                                           # 512x6x6c-2
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x3x6c-2
        convRelu(6, norm)                                     # 512x1x6c-4

        self.cnn = cnn
        size1d=512
        self.cnn1d = nn.Sequential(
                        nn.Conv1d(size1d,size1d,3,1,2,2),
                        nn.GroupNorm(getGroupSize(size1d),size1d),
                        nn.ReLU(True),
                        #nn.Conv1d(size1d,size1d,3,1,0,1),
                        #nn.GroupNorm(getGroupSize(size1d),size1d),
                        #nn.ReLU(True),
                        nn.Conv1d(size1d,size1d,3,1,4,4),
                        nn.GroupNorm(getGroupSize(size1d),size1d),
                        nn.ReLU(True),
                        nn.Conv1d(size1d,size1d,3,1,0,1),
                        nn.GroupNorm(getGroupSize(size1d),size1d),
                        nn.ReLU(True),
                        nn.Conv1d(size1d,size1d,3,1,8,8),
                        nn.GroupNorm(getGroupSize(size1d),size1d),
                        nn.ReLU(True),
                        #nn.Conv1d(size1d,size1d,3,1,0,1),
                        #nn.GroupNorm(getGroupSize(size1d),size1d),
                        #nn.ReLU(True),
                        #nn.Conv1d(size1d,size1d,3,1,16,16),
                        #nn.GroupNorm(getGroupSize(size1d),size1d),
                        #nn.ReLU(True),
                        nn.Conv1d(size1d,nclass,3,1,0,1),
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

