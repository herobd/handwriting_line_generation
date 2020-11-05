import torch
from torch import nn
import torch.nn.functional as F
from .MUNIT_networks import AdaptiveInstanceNorm2d, get_num_adain_params, assign_adain_params, MLP
from .net_builder import getGroupSize
from .elastic_layer import ElasticDeform
#from mmdetection.mmdet.ops.dcn.modules.deform_conv import ModulatedDeformConvPack

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, dropout=0.5, num_layers=2)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class DeformHWR(nn.Module):

    def __init__(self, nclass, nc=1, cnnOutSize=512, nh=512, n_rnn=2, leakyRelu=False, norm='group', useED=False):
        super(DeformHWR, self).__init__()
        # assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 3]
        ps = [1, 1, 1, 1, 1, 0, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]
        deform = [False, False, True, True, True, True, True]
        if useED:
            deform = [False]*7
            ele = [True,True,True,False,False,False,False]
        else:
            ele = [False]*7
        edK = [5,5,3,None,None,None,None]
        edSize = [16,32,64,None,None,None,None]
        edDown = [2,2,1,None,None,None,None]
        edBlur = [9,7,5,None,None,None,None]
        edX = [10,6,4,None,None,None,None]
        edY = [16,12,8,None,None,None,None]

        cnn = nn.Sequential()

        def convRelu(i, norm=None):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            if ele[i]:
                cnn.add_module('elasticDistortion{0}'.format(i),ElasticDeform(nIn,2,edK[i],edSize[i],edDown[i],edBlur[i],edX[i],edY[i]))
            if deform[i]:
                cnn.add_module('conv{0}'.format(i),ModulatedDeformConvPack(nIn, nOut, ks[i], ss[i], ps[i]))
            else:
                cnn.add_module('conv{0}'.format(i), nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
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

        self.rnn = BidirectionalLSTM(cnnOutSize, nh, nclass)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, style=None):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        # assert h == 1, "the height of conv must be 1"
        conv = conv.view(b, -1, w)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        # rnn features
        output = self.rnn(conv)

        return output

    def setup_save_features(self):
        save_from = [18]
        self.saved_features = [None]*len(save_from)
        def factorySave(i):
            def saveX(module, input ,output):
                self.saved_features[i]=output
            return saveX
        for i,layer in enumerate(save_from):
            self.cnn[layer].register_forward_hook( factorySave(i) )

