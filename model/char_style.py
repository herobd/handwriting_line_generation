import torch
from torch import nn
import torch.nn.functional as F
from utils.util import getGroupSize
import copy, math
from collections import defaultdict
import random

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', transpose=False, reverse=False):
        super(Conv2dBlock, self).__init__()
        self.reverse=reverse
        self.use_bias = True
        # initialize padding
        if transpose:
            self.pad = lambda x: x
        elif pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        elif norm == 'group':
            self.norm = nn.GroupNorm(getGroupSize(norm_dim),norm_dim)
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'logsoftmax':
            self.activation = nn.LogSoftmax(dim=1)
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias, padding=padding)
            if norm == 'sn':
                raise NotImplemented('easy to do')
        elif norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        if not self.reverse:
            x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        if self.reverse:
            x = self.conv(self.pad(x))
        return x

class CharExtractor(nn.Module):
    def __init__(self, input_dim, dim, style_dim,num_fc=1,small=False):
        super(CharExtractor, self).__init__()
        self.conv1 = nn.Sequential(
                            #nn.GroupNorm(getGroupSize(input_dim),input_dim),
                            nn.ReLU(),
                            nn.Conv1d(input_dim,dim,3,padding=1),
                            nn.GroupNorm(getGroupSize(dim),dim),
                            nn.ReLU(),
                            nn.Conv1d(dim,input_dim,3,padding=1),
                            #nn.GroupNorm(getGroupSize(dim),dim),
                            )
        if small:
            self.conv2 = nn.Sequential(
                                nn.ReLU(),
                                nn.Conv1d(input_dim,2*dim,1),
                                nn.GroupNorm(getGroupSize(2*dim),2*dim),
                                nn.ReLU()
                                )
        else:
            self.conv2 = nn.Sequential(
                                nn.ReLU(),
                                nn.MaxPool1d(2),
                                nn.Conv1d(input_dim,2*dim,3),
                                nn.GroupNorm(getGroupSize(2*dim),2*dim),
                                nn.ReLU()
                                )

        fc = [nn.Linear(2*dim,2*dim),nn.ReLU(True)]
        for i in range(style_dim,num_fc-1):
            fc+=[nn.Linear(2*dim,2*dim),nn.Dropout(0.25,True),nn.ReLU(True)]
        fc.append(nn.Linear(2*dim,style_dim))
        self.fc = nn.Sequential(*fc)

    def forward(self,x):
        res=x
        batch_size = x.size(0)
        x=self.conv1(x)
        x=self.conv2(x+res)
        x = F.adaptive_avg_pool1d(x,1).view(batch_size,-1)
        return self.fc(x)

class CharStyleEncoder(nn.Module):
    def __init__(self, input_dim, dim, style_dim, char_dim, char_style_dim, norm, activ, pad_type, n_class, global_pool=False, average_found_char_style=0,num_final_g_spacing_style=1,num_char_fc=1,vae=False,window=6,small=False):
        super(CharStyleEncoder, self).__init__()
        if vae:
            self.vae=True
            style_dim*=2
            char_style_dim*=2
        else:
            self.vae=False
        self.n_class=n_class
        if char_style_dim>0:
            self.char_style_dim = char_style_dim
            self.average_found_char_style=average_found_char_style if type(average_found_char_style) is float else 0
            self.single_style=False
        else:
            assert(not self.vae)
            self.char_style_dim = style_dim
            char_style_dim = style_dim
            self.single_style=True
        self.window=window
        small_char_ex = window<3
        self.down = []
        self.down += [Conv2dBlock(input_dim, dim, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)] #64
        #self.down += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            if i==0 and small:
                self.down += [Conv2dBlock(dim, 2 * dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)] #32, 16
            else:
                self.down += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)] #32, 16
            dim *= 2
            self.down += [Conv2dBlock(dim, dim, 3, 1, (1,1,0,0), norm=norm, activation=activ, pad_type=pad_type)]
        self.down += [Conv2dBlock(dim, dim, 4, (2,1), (1,1,0,0), norm=norm, activation=activ, pad_type=pad_type)] #6
        self.down += [Conv2dBlock(dim, dim, 4, (2,1), (1,1,0,0), norm='none', activation='none', pad_type=pad_type)] #1
        self.down = nn.Sequential(*self.down)
        prepped_size = dim #style_dim + style_dim//2
        self.prep = nn.Sequential(
                        nn.Conv1d(dim+n_class, prepped_size, 5, 1, 2),
                        nn.ReLU(True),
                        nn.MaxPool1d(2,2),
                        nn.Conv1d(prepped_size, prepped_size, 3, 1, 1),
                        nn.GroupNorm(getGroupSize(prepped_size),prepped_size),
                        nn.ReLU(True),
                        nn.Conv1d(prepped_size, prepped_size, 3, 1, 1),
                        nn.ReLU(True),
                        )
        final_g_spacing_style = [nn.Linear(prepped_size+char_style_dim,prepped_size),nn.ReLU(True)]
        for i in range(num_final_g_spacing_style-1):
            final_g_spacing_style+=[nn.Linear(prepped_size,prepped_size),nn.Dropout(0.25,True),nn.ReLU(True)]
        if self.single_style:
            final_g_spacing_style.append(nn.Linear(prepped_size,style_dim))
        else:
            final_g_spacing_style.append(nn.Linear(prepped_size,style_dim +char_style_dim))
        self.final_g_spacing_style = nn.Sequential(*final_g_spacing_style)

        self.char_extractor = nn.ModuleList()
        if not self.single_style:
            self.fill_pred = nn.ModuleList()
        for n in range(n_class):
            self.char_extractor.append( CharExtractor(dim,char_dim,self.char_style_dim,num_char_fc,small_char_ex) )
            if not self.single_style:
                self.fill_pred.append( nn.Sequential(
                                            nn.Linear(self.char_style_dim,2*self.char_style_dim),
                                            nn.ReLU(True),
                                            nn.Linear(2*self.char_style_dim,self.char_style_dim*n_class)
                                            ) )


    def forward(self, x,recog):
        batch_size=x.size(0)
        x = self.down(x)
        #x must have height of 1!
        x = x.view(batch_size,x.size(1),x.size(3))
        diff = x.size(2)-recog.size(2)
        if diff>0:
            recog = F.pad(recog,(diff//2,(diff//2)+diff%2),mode='replicate')
        elif diff<0:
            x = F.pad(x,(-diff//2,(-diff//2)+(-diff)%2),mode='replicate')

        recogPred = torch.argmax(recog, dim=1) #taking max prediction, may want to to both max and 2nd?
        fill_styles=[ [] for b in range(batch_size)]
        found_chars_style={}
        if self.single_style:
            b_sum = torch.FloatTensor(batch_size).fill_(0).to(x.device)
            total_style = torch.FloatTensor(batch_size,self.char_style_dim).fill_(0).to(x.device)
        for char_n in range(1,self.n_class):
            locs = recogPred==char_n
            if locs.any():
                patches=[]
                b_weight=[]
                for b in range(batch_size):
                    horz_cents = locs[b].nonzero() #x locations of each instance of char_n
                    for horz_cent in horz_cents:
                        #pad out a window
                        left = max(0,horz_cent.item()-self.window)
                        pad_left = left-(horz_cent.item()-self.window)
                        right = min(x.size(2)-1,horz_cent.item()+self.window)
                        pad_right = (horz_cent.item()+self.window)-right
                        wind = x[b:b+1,:,left:right+1]
                        if pad_left>0 or pad_right>0:
                            wind = F.pad(wind,(pad_left,pad_right))
                        assert(wind.size(2)==self.window*2 +1)
                        patches.append(wind)
                        b_weight.append( (b,math.exp(recog[b,char_n,horz_cent.item()])) ) #keep track of batch in pred score
                patches = torch.cat(patches,dim=0)
                char_styles = self.char_extractor[char_n](patches)

                if self.single_style:
                    for i, (b,score) in enumerate(b_weight):
                        total_style[b] += score * char_styles[i]
                        b_sum[b] += score
                else:
                    b_sum=defaultdict(lambda: 0)#[0]*batch_size

                    found_chars_style[char_n]=defaultdict(lambda: torch.FloatTensor(self.char_style_dim).fill_(0).to(x.device))

                    #perform weighted average over all locations of char_n (by batch, of course)
                    for i, (b,score) in enumerate(b_weight):
                        #if b in found_chars_style[char_n]
                        #char_style[b] += score * char_styles[i]
                        #assert(not torch.isnan(char_style[b]).any())
                        found_chars_style[char_n][b] += score * char_styles[i]
                        b_sum[b] += score
                    bs_of_interest = list(found_chars_style[char_n].keys())
                    for b in bs_of_interest:
                        assert(b_sum[b]!=0)
                        found_chars_style[char_n][b] /= b_sum[b]
                    #predict all other character styles, from this char_n style
                    char_style = torch.stack( [found_chars_style[char_n][b] for b in bs_of_interest], dim=0)
                    fill_pred =  self.fill_pred[char_n](char_style)
                    for i,b in enumerate(bs_of_interest):
                        fill_styles[b].append(fill_pred[i])


        if not self.single_style:
            #average fill styles across the predictions from each found character's style
            fill_bs=[]
            for b in range(batch_size):
                if len(fill_styles[b])>0:
                    fill_bs.append( torch.stack(fill_styles[b],dim=0).mean(dim=0) )
                else:
                    fill_bs.append( torch.FloatTensor(self.n_class*self.char_style_dim).fill_(0).to(x.device))
            all_char_style = [list(torch.chunk(styles,self.n_class,dim=0)) for styles in fill_bs] #this is a pain, we chunk to prevent "inplace operation" in the following loop

            #substitute in the styles of the characters we actually found
            for char_n,char_style in found_chars_style.items():
                for b in char_style:
                    if self.average_found_char_style>0:
                        all_char_style[b][char_n] = char_style[b]*(1-self.average_found_char_style) + all_char_style[b][char_n]*(self.average_found_char_style)
                    elif self.average_found_char_style<0:
                        if self.training:
                            mix = random.random()*(-self.average_found_char_style)
                        else:
                            mix = 0.1
                        all_char_style[b][char_n] = char_style[b]*(1-mix) + all_char_style[b][char_n]*(mix)
                    else:
                        all_char_style[b][char_n] = char_style[b]
            all_char_style = [torch.stack(styles,dim=0) for styles in all_char_style] #stack different characters together (for each batch)
            all_char_style = torch.stack(all_char_style,dim=0) #stack batch together
            avg_char_style = all_char_style.sum(dim=1)/self.n_class
        else:
            avg_char_style = torch.where(b_sum[...,None]!=0,total_style/b_sum[...,None],total_style)

        xr = torch.cat((F.relu(x),recog),dim=1)
        xr = self.prep(xr)

        xr = F.adaptive_avg_pool1d(xr,1).view(batch_size,-1)

        comb_style = torch.cat((xr,avg_char_style),dim=1)
        assert(not torch.isnan(comb_style).any())
        comb_style = self.final_g_spacing_style(comb_style)
        if self.single_style:
            return comb_style
        g_style = comb_style[:,self.char_style_dim:]
        spacing_style = comb_style[:,:self.char_style_dim]
        assert(not torch.isnan(comb_style).any())

        if self.vae:
            g_mu,g_log_sigma=g_style.chunk(2,dim=1)
            spacing_mu,spacing_log_sigma=spacing_style.chunk(2,dim=1)
            all_char_mu,all_char_log_sigma=all_char_style.chunk(2,dim=2)

            return g_mu,g_log_sigma,spacing_mu,spacing_log_sigma,all_char_mu,all_char_log_sigma
        else:
            return g_style, spacing_style, all_char_style
            

