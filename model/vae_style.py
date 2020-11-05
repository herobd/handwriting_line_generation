import torch
from torch import nn
import torch.nn.functional as F
from .MUNIT_networks import Conv2dBlock
from .net_builder import getGroupSize
import copy, math
from .attention import MultiHeadedAttention

class VAEStyleEncoder(nn.Module):
    def __init__(self, input_dim, dim, style_dim, norm, activ, pad_type, n_class, num_keys=16, frozen_keys=False, global_pool=False, attention=True,wider=False,small=False):
        super(VAEStyleEncoder, self).__init__()
        self.global_pool=global_pool
        self.attention=attention
        self.down1 = []
        self.down1 += [Conv2dBlock(input_dim, dim, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)]
        if small:
            self.down1 += [Conv2dBlock(dim, 2 * dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)] #32, 16
        else:
            self.down1 += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)] #32, 16
        dim *= 2
        self.down2 = []
        self.down2 += [Conv2dBlock(dim+n_class, dim, 3, 1, (1,1,0,0), norm=norm, activation=activ, pad_type=pad_type)]
        self.down2 += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)] #32, 16
        dim *= 2
        self.down2 += [Conv2dBlock(dim, dim, 3, 1, (1,1,0,0), norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.down2 += [Conv2dBlock(dim, dim, 4, (2,1), (1,1,0,0), norm=norm, activation=activ, pad_type=pad_type)] #6, 1
        self.down1 = nn.Sequential(*self.down1)
        self.down2 = nn.Sequential(*self.down2)
        if wider:
            prepped_size = 2*style_dim
        else:
            prepped_size = style_dim
        self.prep = nn.Sequential(
                        nn.Conv1d(dim+n_class, prepped_size, 5, 1, 2),
                        nn.ReLU(True),
                        nn.MaxPool1d(2,2),
                        nn.Conv1d(prepped_size, prepped_size, 3, 1, 1),
                        nn.ReLU(True),
                        nn.Conv1d(prepped_size, prepped_size, 3, 1, 1)
                        )

        final_out_size = style_dim
        if global_pool and attention:
            prepped_size//=2 # as we split half of the features off to do global pooling on
            style_dim//=2
        if attention:
            if prepped_size>64:
                heads=prepped_size//64
            else:
                heads=2
            self.mhAtt1 = MultiHeadedAttention(heads,prepped_size)
            #self.mhAtt2 = MultiHeadedAttention(heads,prepped_size)
            #self.mhAtt3 = MultiHeadedAttention(heads,prepped_size)
            if frozen_keys:
                self.put_on_gpu=True#False
                self.keys1 = nn.Parameter(torch.FloatTensor(1,num_keys,prepped_size).normal_(), requires_grad=False)
            else:
                self.put_on_gpu=True
                self.keys1 = nn.Parameter(torch.FloatTensor(1,num_keys,prepped_size).normal_())
            #self.keys2 = nn.Parameter(nn.FloatTensor(1,num_keys,prepped_size).normal_())
            #self.keys3 = nn.Parameter(nn.FloatTensor(1,num_keys,prepped_size).normal_())
            #self.comb1 = nn.Sequential(
            #                nn.LeakyReLU(0.2,True),
            #                nn.Linear(2*prepped_size*num_keys,style_dim)
            #                )
            #self.comb2 = nn.Sequential(
            #                nn.LeakyReLU(0.1,True),
            #                nn.Linear(prepped_size*num_keys+style_dim,style_dim)
            #                )
            self.reduceQuarters=nn.ModuleList()
            if global_pool:
                quarter_feat = (prepped_size*num_keys + style_dim)//4
            else:
                quarter_feat = prepped_size*num_keys//4
            for i in range(4):
                self.reduceQuarters.append( nn.Sequential(
                                nn.LeakyReLU(0.2,True),
                                nn.Linear(quarter_feat,style_dim),
                                nn.Dropout(0.05,True),
                                nn.LeakyReLU(0.01,True),
                                ) )
            self.reduceWhole = nn.Sequential(
                            nn.Linear(4*style_dim,2*final_out_size),
                            nn.Dropout2d(0.01,True),
                            nn.LeakyReLU(0.01,True),
                            #nn.Linear(2*final_out_size,final_out_size)
                            #nn.LeakyReLU(0.01,True),
                            )
        else:
            self.put_on_gpu=True
            self.final = nn.Sequential(
                            nn.Linear(prepped_size,2*final_out_size),
                            nn.GroupNorm(getGroupSize(2*final_out_size),2*final_out_size),
                            nn.Dropout(0.01,True),
                            nn.LeakyReLU(0.01,True),
                            )

        self.pred_mu = nn.Linear(2*final_out_size,final_out_size)
        self.pred_log_sigma = nn.Linear(2*final_out_size,final_out_size)

    def forward(self, x,recog):
        if not self.put_on_gpu and self.attention:
            self.put_on_gpu=True
            self.keys1=self.keys1.to(x.device)
        batch_size=x.size(0)
        x = self.down1(x)

        recog_resize = F.interpolate(recog,x.size(3))
        recog_resize = recog_resize[:,:,None,:].expand(-1,-1,x.size(2),-1)
        x = torch.cat((x,recog_resize),dim=1)

        x = self.down2(x)

        #x must have height of 1!
        x = x.view(batch_size,x.size(1),x.size(3))
        diff = x.size(2)-recog.size(2)
        if diff>0:
            recog = F.pad(recog,(diff//2,(diff//2)+diff%2),mode='replicate')
        elif diff<0:
            x = F.pad(x,(-diff//2,(-diff//2)+(-diff)%2),mode='replicate')

        x = torch.cat((x,recog),dim=1)
        x = self.prep(x)

        if self.global_pool:
            if self.attention:
                x,g_x = torch.chunk(x,2,dim=1)
            else:
                g_x = x
            g_style = F.adaptive_avg_pool1d(g_x,1).view(batch_size,-1)
        #attOut = torch.zero(x.size(1))
        #for att in self.mhAtts:
        if self.attention:
            x = x.permute(0,2,1)
            att = self.mhAtt1(self.keys1.expand(batch_size,-1,-1),x,x)
            #import pdb;pdb.set_trace() #batch,head,query,width
            #att2 = self.mhAtt2(self.keys2,x,x))
            #att3 = self.mhAtt3(self.keys3,x,x))
            vector = att.view(batch_size,att.size(1)*att.size(2))
            #vector2 = att2.view(batch_size,att2.size(1)*att2.size(2))
            #vector3 = att3.view(batch_size,att3.size(1)*att3.size(2))
            #vector = self.comb1(torch.cat((vector,vector1,dim=1))
            #vector = self.comb2(torch.cat((vector,vector2,dim=1)
            if self.global_pool:
                vector = torch.cat((g_style,vector),dim=1)

            #return vector
            quarterDim = vector.size(1)//4
            reducedQ=[]
            vq = torch.chunk(vector,4,dim=1)
            for i in range(4):
                reducedQ.append( self.reduceQuarters[i](vq[i]) )
                #reducedQ.append( self.reduceQuarters[i](vector[:,i*quarterDim:(i+1)*quarterDim].copy()) )
            a_style = self.reduceWhole(torch.cat(reducedQ,dim=1))
        else:
            a_style = self.final(g_style)

        mu,sigma = self.pred_mu(a_style), self.pred_log_sigma(a_style)
        return mu,sigma

