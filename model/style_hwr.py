# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
import torch
from torch import nn
import torch.nn.functional as F
from .MUNIT_networks import AdaptiveInstanceNorm2d, get_num_adain_params, assign_adain_params, MLP
from .net_builder import getGroupSize
from .elastic_layer import ElasticDeformWithStyle
#from mmdetection.mmdet.ops.dcn.modules.deform_conv import ModulatedDeformConv

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut,style_dim=0):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, dropout=0.5, num_layers=2)
        self.embedding = nn.Linear(nHidden * 2, nOut)

        if style_dim>0:
            #num_layers * num_directions, batch, hidden_size
            trans_size = 2*2*nHidden * 2
            mid_size = (trans_size+style_dim)//2
            self.trans = nn.Sequential(
                            nn.ReLU(),
                            nn.Linear(style_dim,mid_size),
                            nn.ReLU(True),
                            nn.Linear(mid_size,trans_size),
                            nn.ReLU(True)
                            )


    def forward(self, input,style=None):
        if style is None:
            recurrent, _ = self.rnn(input)
        else:
            batch_size = style.size(0)
            style_t = self.trans(style).permute(1,0).contgiuous().view(2*2,batch_size,-1)
            h_0,c_0 = torch.chunk(style_t,2,dim=2)
            recurrent, _ = self.rnn(input,(h_0.contgiuous(),c_0.contgiuous()))
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class AdaINStyleHWR(nn.Module):

    def __init__(self, nclass, nc=1, cnnOutSize=512, nh=512, n_rnn=2, leakyRelu=False, style_dim=256):
        super(AdaINStyleHWR, self).__init__()
        # assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 3]
        ps = [1, 1, 1, 1, 1, 0, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                #cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
                cnn.add_module('adaIN{0}'.format(i),  AdaptiveInstanceNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x32x12c
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x16x6c
        convRelu(2, True)
        convRelu(3)       
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x8x6c
        convRelu(4, True)
        convRelu(5)                                           # 512x6x6c-2
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x3x6c-2
        convRelu(6, True)                                     # 512x1x6c-4

        self.cnn = cnn

        self.rnn = BidirectionalLSTM(cnnOutSize, nh, nclass)
        self.softmax = nn.LogSoftmax()

        num_adain_params = get_num_adain_params(self.cnn)
        #if style_transform_dim is None:
        style_transform_dim=(2*style_dim+num_adain_params)//3
        self.style_transform = MLP(style_dim, num_adain_params, style_transform_dim, 4, norm='none')

    def forward(self, input,style):
        adain_params = self.style_transform(style)
        assign_adain_params(adain_params, self.cnn)
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

class HyperConv(nn.Module):
    def __init__(self,ch_in,ch_out,hyper_in,padding=1):
        super(HyperConv, self).__init__()
        #self.fold = torch.nn.Fold(output_size, kernel_size=3, dilation=1, padding=1, stride=1)
        self.unfold = nn.Unfold(3,padding=padding)
        self.padding=padding
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.num_gen = max(self.ch_out//64,1)
        self.num_weight = (self.ch_out//self.num_gen)*(self.ch_in//self.num_gen)*3*3
        self.num_bias = self.ch_out//self.num_gen
        hyper_dim = int(max(hyper_in, self.num_weight+self.num_bias)*0.25 + 0.5*hyper_in)//self.num_gen
        print('make hyper {}, {} {}'.format(self.num_gen,self.num_weight//self.num_gen,self.num_bias//self.num_gen))
        self.gen = nn.ModuleList()
        for i in range(self.num_gen):
            self.gen.append(nn.Sequential(
                        nn.Linear(hyper_in,hyper_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(hyper_dim,(self.num_weight+self.num_bias))
                        ))
        print('made')


    def forward(self,image,hyper):
        batch_size = image.size(0)
        resH=image.size(2) - 2*(1-self.padding)
        resW=image.size(3) - 2*(1-self.padding)
        weight = torch.FloatTensor(batch_size,self.ch_out,self.ch_in,3,3).to(image.device)
        bias = torch.FloatTensor(batch_size,self.ch_out,1,1).to(image.device)
        g_ch_in = self.ch_in//self.num_gen
        g_ch_out = self.ch_out//self.num_gen
        hyper=hyper.view(hyper.size(0),hyper.size(1))
        for i in range(self.num_gen):
            gened = self.gen[i](hyper)
            weight[:,i*g_ch_out:(i+1)*g_ch_out,i*g_ch_in:(i+1)*g_ch_in,:,:] = gened[:,:self.num_weight].view(batch_size,g_ch_out,g_ch_in,3,3)
            bias[:,i*g_ch_out:(i+1)*g_ch_out] = gened[:,self.num_weight:].view(-1,g_ch_out,1,1)
        #image = F.conv1d(image,weight,padding=1)
        unfolded = self.unfold(image)
        #wieght is ch, dims
        unfolded = unfolded.transpose(1, 2).matmul(weight.view(weight.size(0), weight.size(1), -1).permute(0,2,1)).transpose(1, 2)
        image = F.fold(unfolded, (resH, resW), (1, 1),padding=0) #not sure why padding is 0...
        return image+bias
class HyperStyleHWR(nn.Module):

    def __init__(self, nclass, nc=1, cnnOutSize=512, nh=512, n_rnn=2, leakyRelu=False, style_dim=256, norm='group'):
        super(HyperStyleHWR, self).__init__()
        # assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 3]
        ps = [1, 1, 1, 1, 1, 0, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]


        def convRelu(cnn,i, norm=None):
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

        self.cnn1 = nn.Sequential()
        convRelu(self.cnn1,0)
        self.cnn1.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x32x12c
        self.hyper1 = HyperConv(nm[0],nm[1],style_dim) #convRelu(1)
        self.cnn2 = nn.Sequential(nn.ReLU(True))
        self.cnn2.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x16x6c
        convRelu(self.cnn2,2, norm)
        #convRelu(3)       
        self.hyper2 =HyperConv(nm[2],nm[3],style_dim) #convRelu(1)
        self.cnn3 = nn.Sequential(nn.ReLU(True))
        self.cnn3.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x8x6c
        convRelu(self.cnn3,4, norm)
        #convRelu(5)                                           # 512x6x6c-2
        self.hyper3 =HyperConv(nm[4],nm[5],style_dim,padding=0) #Curtis's skips the padding on this layer convRelu(1)
        self.cnn4 = nn.Sequential(nn.ReLU(True))
        self.cnn4.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x3x6c-2
        convRelu(self.cnn4,6, norm)                                     # 512x1x6c-4


        self.rnn = BidirectionalLSTM(cnnOutSize, nh, nclass)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, style=None):
        # conv features
        conv = self.cnn1(input)
        conv = self.hyper1(conv,style)
        conv = self.cnn2(conv)
        conv = self.hyper2(conv,style)
        conv = self.cnn3(conv)
        conv = self.hyper3(conv,style)
        conv = self.cnn4(conv)
        b, c, h, w = conv.size()
        # assert h == 1, "the height of conv must be 1"
        conv = conv.view(b, -1, w)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        # rnn features
        output = self.rnn(conv)

        return output

    def setup_save_features(self):
        save_from = [2]
        self.saved_features = [None]*len(save_from)
        def factorySave(i):
            def saveX(module, input ,output):
                self.saved_features[i]=output
            return saveX
        for i,layer in enumerate(save_from):
            self.cnn4[layer].register_forward_hook( factorySave(i) )

class DeformConv(nn.Module):
    def __init__(self,ch_in,ch_out,kernel,stride,padding,style_dim):
        super(DeformConv,self).__init__()
        self.deform = ModulatedDeformConv(
                 in_channels=ch_in,
                 out_channels=ch_out,
                 kernel_size=kernel,
                 stride=stride,
                 padding=padding,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=True)
        ch_mid = min(ch_in,style_dim)
        self.conv_offset_mask_prestyle = nn.Sequential(
                                nn.Conv2d(
                                    ch_in,
                                    ch_mid,
                                    kernel_size=kernel,
                                    stride=stride,
                                    padding=padding,
                                    bias=True),
                                nn.ReLU(True)
                                )
        self.conv_offset_mask = nn.Sequential(
                nn.Conv2d(
                    style_dim+ch_mid,
                    style_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True),
                nn.ReLU(True),
                nn.Conv2d(
                    style_dim,
                    3*kernel*kernel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True)	
                )
        self.style_trans=nn.Sequential(
                            nn.ReLU(),
                            nn.Linear(style_dim,style_dim)
                            )
        #according to the deformation conv v2 paper, you should init to no deforamation and 0.5 masking/modulation
        self.conv_offset_mask[2].weight.data.zero_()
        self.conv_offset_mask[2].bias.data.zero_()

    def forward(self,input):
        image,style = input
        cond = self.conv_offset_mask_prestyle(image)
        style = self.style_trans(style)
        style_expanded = style.view(style.size(0),-1,1,1).expand(-1,-1,cond.size(2),cond.size(3))
        cond = torch.cat((cond,style_expanded),dim=1)
        cond = self.conv_offset_mask(cond)
        o1, o2, mask = torch.chunk(cond, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return (self.deform(image,offset,mask), style)

class Split(nn.Module):
    def __init__(self,mod):
        super(Split,self).__init__()
        self.mod = mod
    def forward(self,input):
        image,style = input
        return (self.mod(image),style)

class StyleAppend(nn.Module):
    def __init__(self,style_dim,down=1):
        super(StyleAppend,self).__init__()
        self.trans = nn.Sequential(
                            nn.ReLU(),
                            nn.Linear(style_dim,style_dim//down)
                            )
    def forward(self,input):
        image,style = input
        styleT = self.trans(style)
        styleT = styleT.view(styleT.size(0),styleT.size(1),1,1).expand(-1,-1,image.size(2),image.size(3))
        image = torch.cat((image,styleT),dim=1)
        return (image,style)
class StyleAdd(nn.Module):
    def __init__(self,style_dim,num_feat):
        super(StyleAdd,self).__init__()
        self.trans = nn.Sequential(
                            nn.ReLU(),
                            nn.Linear(style_dim,num_feat)
                            )
    def forward(self,input):
        image,style = input
        styleT = self.trans(style)
        styleT = styleT.view(styleT.size(0),styleT.size(1),1,1).expand(-1,-1,image.size(2),image.size(3))
        imageOut =image+styleT
        return (imageOut,style)


class DeformStyleHWR(nn.Module):

    def __init__(self, nclass, nc=1, cnnOutSize=512, nh=512, n_rnn=2, leakyRelu=False, norm='group', style_dim=256, useED=False, appendStyle=False, deformConv=True, transDeep=False, numExperts=0, rnnStyle=False, addStyle=False):
        super(DeformStyleHWR, self).__init__()
        # assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 3]
        ps = [1, 1, 1, 1, 1, 0, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]
        if deformConv == 'all' or deformConv=='full':
            deform = [True]*7
        elif deformConv:
            deform = [False, False, True, True, True, True, True]
        else:
            deform = [False]*7
        
        if useED:
            ele = [True,True,True,False,False,False,False]
        else:
            ele = [False]*7
        edK = [5,5,3,None,None,None,None]
        edSize = [16,32,64,None,None,None,None]
        edDown = [2,2,1,None,None,None,None]
        edBlur = [9,7,5,None,None,None,None]
        edX = [10,6,4,None,None,None,None]
        edY = [16,12,8,None,None,None,None]

        if appendStyle:
            append = [4,None,None,2,None,None,1]
        else:
            append = [None]*7
        if addStyle:
            add = [True,False,False,True,False,False,True]
        else:
            add = [False]*7

        cnn = nn.Sequential()

        def convRelu(seq,i, norm=None):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            if ele[i]:
                seq.add_module('elasticDistortionStyle{0}'.format(i),ElasticDeformWithStyle(nIn,2,edK[i],edSize[i],edDown[i],edBlur[i],edX[i],edY[i],style_dim))
            if append[i] is not None:
                seq.add_module('appendStyle{0}'.format(i),StyleAppend(style_dim,append[i]))
                nIn+=style_dim//append[i]
            if add[i]:
                seq.add_module('addStyle{0}'.format(i),StyleAdd(style_dim,nIn))
            if deform[i]:
                seq.add_module('conv{0}'.format(i),DeformConv(nIn, nOut,ks[i], ss[i], ps[i], style_dim))
            else:
                seq.add_module('conv{0}'.format(i), Split(nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i])))
            if norm is not None and 'group' in norm:
                seq.add_module('groupnorm{0}'.format(i), Split(nn.GroupNorm(getGroupSize(nOut),nOut)))
            elif norm:
                seq.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                seq.add_module('relu{0}'.format(i),
                               Split(nn.LeakyReLU(0.2, inplace=True)))
            else:
                seq.add_module('relu{0}'.format(i), Split(nn.ReLU(True)))

        convRelu(cnn,0)
        cnn.add_module('pooling{0}'.format(0), Split(nn.MaxPool2d(2, 2)))  # 64x32x12c
        convRelu(cnn,1)
        cnn.add_module('pooling{0}'.format(1), Split(nn.MaxPool2d(2, 2)))  # 128x16x6c

        if numExperts>1:
            expert_modules = nn.ModuleList()
            for i in range(numExperts):
                expert = nn.Sequential()
                convRelu(expert,2, norm)
                convRelu(expert,3)       
                expert.add_module('pooling{0}'.format(2),
                               Split(nn.MaxPool2d((2, 2), (2, 1), (0, 1))))  # 256x8x6c
                convRelu(expert,4, norm)
                convRelu(expert,5)          # 512x6x6c-2
                expert_modules.append(expert)
            cnn.add_module('MoE', MoE(expert_modules,nm[1],style_dim))
        else:
            convRelu(cnn,2, norm)
            convRelu(cnn,3)       
            cnn.add_module('pooling{0}'.format(2),
                           Split(nn.MaxPool2d((2, 2), (2, 1), (0, 1))))  # 256x8x6c
            convRelu(cnn,4, norm)
            convRelu(cnn,5)                                           # 512x6x6c-2
        cnn.add_module('pooling{0}'.format(3),
                       Split(nn.MaxPool2d((2, 2), (2, 1), (0, 1))))  # 512x3x6c-2
        convRelu(cnn,6, norm)                                     # 512x1x6c-4

        self.cnn = cnn
        
        self.rnn_style=rnnStyle
        if rnnStyle:
            self.rnn = BidirectionalLSTM(cnnOutSize, nh, nclass, style_dim=style_dim)
        else:
            self.rnn = BidirectionalLSTM(cnnOutSize, nh, nclass)
        self.softmax = nn.LogSoftmax()
        
        if transDeep:
            self.style_trans=nn.Sequential(
                                nn.LeakyReLU(0.2,True),
                                nn.Linear(style_dim,style_dim),
                                nn.ReLU(True),
                                nn.Linear(style_dim,style_dim),
                                nn.ReLU(True),
                                nn.Linear(style_dim,style_dim),
                                nn.ReLU(True),
                                nn.Linear(style_dim,style_dim),
                                nn.ReLU(True),
                                nn.Linear(style_dim,style_dim),
                                )
        elif style_dim>0:
            self.style_trans=nn.Sequential(
                                nn.LeakyReLU(0.2,True),
                                nn.Linear(style_dim,style_dim),
                                nn.ReLU(True),
                                nn.Linear(style_dim,style_dim),
                                )

    def forward(self, input, style=None):
        # conv features
        if style is not None:
            style = self.style_trans(style)
            style = style.view(style.size(0),style.size(1))
        conv,style = self.cnn((input,style))
        b, c, h, w = conv.size()
        # assert h == 1, "the height of conv must be 1"
        conv = conv.view(b, -1, w)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        # rnn features
        if self.rnn_style:
            output = self.rnn(conv,style)
        else:
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

class MoE(nn.Module):
    def __init__(self,experts,im_ch,style_dim=0):
        super(MoE,self).__init__()
        self.experts=experts
        self.gatingCNN = nn.Sequential(
                            nn.Conv2d(im_ch,im_ch,3),
                            nn.AdaptiveAvgPool2d(1)
                            )
        if style_dim>0:
            self.gating = nn.Sequential(
                            nn.Linear(im_ch+style_dim,style_dim),
                            nn.ReLU(True),
                            #nn.Linear(style_dim,style_dim),
                            #nn.ReLU(True),
                            nn.Linear(style_dim,len(experts)*2)
                            )
            #self.noise = nn.Sequential(
            #                nn.Linear(im_ch+style_dim,style_dim),
            #                nn.Softplus()
            #                )
        else:
            self.gating = nn.Sequential(
                            nn.Linear(im_ch,im_ch),
                            nn.ReLU(True),
                            nn.Linear(im_ch,len(experts)*2)
                            )

    def forward(self,input):
        image,style=input
        batch_size = image.size(0)
        inter_gate = self.gatingCNN(image).view(batch_size,-1)
        if style is not None:
            inter_gate = torch.cat((inter_gate,style),dim=1)
        choice = self.gating(inter_gate)
        choice, noise_control = torch.chunk(choice,2,dim=1)
        choice += torch.empty_like(choice).normal_()*F.softplus(noise_control)
        #keep top k?
        choice = F.softmax(choice,dim=1)
        choice = torch.chunk(choice,len(self.experts),dim=1)

        result=None
        for i,module in enumerate(self.experts):
            r_i, _ = module(input)
            if result is None:
                result = choice[i].view(batch_size,1,1,1)*r_i
            else:
                result += choice[i].view(batch_size,1,1,1)*r_i
        return result, style
