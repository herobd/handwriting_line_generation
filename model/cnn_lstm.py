import torch
from torch import nn
import torch.nn.functional as F
from utils.util import getGroupSize

from base import BaseModel


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

#This model is intended for images with a height of 64
class CRNN(BaseModel):#(nn.Module):

    def __init__(self, nclass, nc=1, cnnOutSize=512, nh=512, n_rnn=2, leakyRelu=False, norm='batch',use_softmax=False,small=False,pad=False):
        super(CRNN, self).__init__(None)
        # assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        self.use_softmax=use_softmax
        if pad:
            h = 32 if small else 64
            if pad=='less':
                self.pad=nn.ZeroPad2d((h,h,0,0))
            else:
                self.pad=nn.ZeroPad2d((h*2,h*2,0,0))
        else:
            self.pad=None

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
        if not small:
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
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, style=None):
        # conv features
        if self.pad is not None:
            input=self.pad(input)
        if input.size(3)<12:
            diff = 12-input.size(3)
            input = F.pad(input,(diff//2,diff//2 +diff%2))
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        # assert h == 1, "the height of conv must be 1"
        conv = conv.view(b, -1, w)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        # rnn features
        output = self.rnn(conv)

        if self.use_softmax:
            return self.softmax(output)
        else:
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


#This model is intended for images with a height of 24
class SmallCRNN(BaseModel):#(nn.Module):

    def __init__(self, nclass, nc=1, cnnOutSize=512, nh=512, n_rnn=2, leakyRelu=False, norm='batch',use_softmax=False):
        super(SmallCRNN, self).__init__(None)
        # assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        self.use_softmax=use_softmax

        ks = [3, 3, 3, 3, 3, 3, 3]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [128, 128, 256, 256, 512, 512, 512]
        drop = [False,False,True,True,True,True,True]

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
            if drop[i]:
                cnn.add_module('dropout{0}'.format(i),nn.Dropout2d(0.1,True))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        convRelu(1,norm)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 128x12x12c
        convRelu(2,norm)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 256x6x6c
        convRelu(4, norm)
        convRelu(5)       
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x3x6c
        convRelu(6, norm)                                     # 512x1x6c-2

        self.cnn = cnn

        self.rnn = BidirectionalLSTM(cnnOutSize, nh, nclass)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, style=None):
        if input.size(3)<12:
            diff = 12-input.size(3)
            input = F.pad(input,(diff//2,diff//2 +diff%2))
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        # assert h == 1, "the height of conv must be 1"
        conv = conv.view(b, -1, w)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        # rnn features
        output = self.rnn(conv)

        if self.use_softmax:
            return self.softmax(output)
        else:
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

