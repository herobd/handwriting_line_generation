from base import BaseModel
import torch
from torch import nn
import torch.nn.functional as F
from utils.util import getGroupSize


class Autoencoder(BaseModel):
    def __init__(self, config):
        super(Autoencoder, self).__init__(config)
        if  'type' in config:
            if config['type']=='small':
                out_size=128
                self.encoder = EncoderSm()
                self.decoder = DecoderSm()
            elif config['type'] == 'no skip':
                out_size=256
                self.encoder = Encoder()
                self.decoder = DecoderNoSkip()
            elif config['type'] == '2':
                out_size=256
                self.encoder = Encoder2()
                self.decoder = DecoderNoSkip(256)
            elif config['type'] == '3':
                out_size=512
                self.encoder = Encoder3()
                self.decoder = DecoderNoSkip()
            elif config['type'] == '2tight':
                out_size=32
                self.encoder = Encoder2(32)
                self.decoder = DecoderNoSkip(32)
            elif config['type'] == '2tighter':
                out_size=16
                self.encoder = Encoder2(16)
                self.decoder = DecoderNoSkip(16)
            elif config['type'] == 'smallSpace':
                out_size=4
                self.encoder = EncoderSpace(4)
                self.decoder = DecoderSpace(4)
            elif config['type'] == 'space':
                out_size=8
                self.encoder = EncoderSpace(8)
                self.decoder = DecoderSpace(8)
            elif config['type'] == '32':
                out_size=256
                self.encoder = Encoder32(out_size)
                self.decoder = Decoder32NoSkip(out_size)
            else:
                raise NotImplementedError('Autoencoder, no type: {}'.format(config['type']))
        else:
            self.encoder = Encoder()
            self.decoder = Decoder()

        if 'hwr_batch' in config:
            self.hwr = E_HWR_batch(config['hwr_batch'],out_size)
        elif 'hwr' in config:
            self.hwr = E_HWR(config['hwr'],out_size)
        else:
            self.hwr = None

    def forward(self,x):
        enc,mid = self.encoder(x)
        if self.hwr is None:
            return self.decoder(enc,mid)
        else:
            return self.decoder(enc,mid), self.hwr(enc)

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.down_conv1 = nn.Sequential(
                nn.Conv2d(1,32,5,padding=2),
                nn.GroupNorm(getGroupSize(32),32),
                nn.ReLU(True),
                nn.MaxPool2d(2),
                nn.Conv2d(32,64,3,padding=1),
                #nn.GroupNorm(getGroupSize(64),64)
                #nn.ReLU(True),
                )

        self.conv1 = nn.Sequential(
                nn.GroupNorm(getGroupSize(64),64),
                nn.ReLU(True),
                nn.Conv2d(64,64,3,padding=1),
                nn.GroupNorm(getGroupSize(64),64),
                nn.ReLU(True),
                nn.Conv2d(64,64,3,padding=1),
                )

        self.down_conv2 = nn.Sequential(
                nn.GroupNorm(getGroupSize(64),64),
                nn.ReLU(True),
                nn.MaxPool2d(2),
                nn.Conv2d(64,128,3,padding=1),
                )

        self.conv2 = nn.Sequential(
                nn.GroupNorm(getGroupSize(128),128),
                nn.ReLU(True),
                nn.Conv2d(128,128,3,padding=1),
                nn.GroupNorm(getGroupSize(128),128),
                nn.ReLU(True),
                nn.Conv2d(128,128,3,padding=1),
                )

        self.down_conv3 = nn.Sequential(
                nn.GroupNorm(getGroupSize(128),128),
                nn.ReLU(True),
                nn.MaxPool2d(2),
                nn.Conv2d(128,256,3),
                nn.GroupNorm(getGroupSize(256),256),
                nn.ReLU(True),
                #nn.Conv2d(256,256,3),
                #nn.GroupNorm(getGroupSize(256),256)
                #nn.ReLU(True),
                nn.Conv2d(256,512,(6,3)),
                )


    def forward(self,x):

        x=self.down_conv1(x)
        res=x
        x=self.conv1(x)
        x+=res
        x=self.down_conv2(x)
        res=x
        x=self.conv2(x)
        x+=res
        mid_features=x
        x=self.down_conv3(x)
        return x, mid_features


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()


        self.up_conv1 = nn.Sequential(
                nn.ReLU(False),
                nn.ConvTranspose2d(512,256,(6,3)),
                nn.GroupNorm(getGroupSize(256),256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256,256,3),
                nn.GroupNorm(getGroupSize(256),256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256,128,4,stride=2,padding=1),
                )
        self.up_conv2 = nn.Sequential(
                nn.GroupNorm(getGroupSize(256),256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256,128,3,padding=1),
                nn.GroupNorm(getGroupSize(128),128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128,64,4,stride=2,padding=1),
                nn.GroupNorm(getGroupSize(64),64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64,64,3,padding=1),
                nn.GroupNorm(getGroupSize(64),64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64,32,4,stride=2,padding=1),
                nn.GroupNorm(getGroupSize(32),32),
                nn.ReLU(True),
                nn.ConvTranspose2d(32,1,3,padding=1),
                nn.Tanh(),
                )


    def forward(self,x,mid_features):

        x=self.up_conv1(x)
        if x.size(3) < mid_features.size(3):
            x = F.pad(x,(0,1,0,0),mode='replicate')
        x = torch.cat((x,mid_features),dim=1)
        x=self.up_conv2(x)
        return x




class EncoderSm(nn.Module):

    def __init__(self):
        super(EncoderSm, self).__init__()

        self.down_conv1 = nn.Sequential(
                nn.Conv2d(1,32,5,padding=2),
                nn.GroupNorm(getGroupSize(32),32),
                nn.ReLU(True),
                nn.MaxPool2d(2),
                nn.Conv2d(32,32,1),
                #nn.GroupNorm(getGroupSize(64),64)
                #nn.ReLU(True),
                )

        self.conv1 = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(32,32,3,padding=1),
                nn.GroupNorm(getGroupSize(32),32),
                nn.ReLU(True),
                nn.Conv2d(32,32,3,padding=1),
                )

        self.down_conv2 = nn.Sequential(
                nn.GroupNorm(getGroupSize(32),32),
                nn.ReLU(True),
                nn.MaxPool2d(2),
                nn.Conv2d(32,64,1),
                )

        self.conv2 = nn.Sequential(
                nn.GroupNorm(getGroupSize(64),64),
                nn.ReLU(True),
                nn.Conv2d(64,64,3,padding=1),
                nn.GroupNorm(getGroupSize(64),64),
                nn.ReLU(True),
                nn.Conv2d(64,64,3,padding=1),
                )

        self.down_conv3 = nn.Sequential(
                nn.GroupNorm(getGroupSize(64),64),
                nn.ReLU(True),
                nn.MaxPool2d(2),
                nn.Conv2d(64,128,3),
                nn.GroupNorm(getGroupSize(128),128),
                nn.ReLU(True),
                #nn.Conv2d(256,256,3),
                #nn.GroupNorm(getGroupSize(256),256)
                #nn.ReLU(True),
                nn.Conv2d(128,256,(6,3)),
                )


    def forward(self,x):

        x=self.down_conv1(x)
        res=x
        x=self.conv1(x)
        x+=res
        x=self.down_conv2(x)
        res=x
        x=self.conv2(x)
        x+=res
        mid_features=x
        x=self.down_conv3(x)
        return x, mid_features


class DecoderSm(nn.Module):

    def __init__(self):
        super(DecoderSm, self).__init__()


        self.up_conv1 = nn.Sequential(
                nn.ReLU(False),
                nn.ConvTranspose2d(256,128,(6,3)),
                nn.GroupNorm(getGroupSize(128),128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128,128,3),
                nn.GroupNorm(getGroupSize(128),128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128,64,4,stride=2,padding=1),
                )
        self.up_conv2 = nn.Sequential(
                nn.GroupNorm(getGroupSize(128),128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128,64,3,padding=1),
                nn.GroupNorm(getGroupSize(64),64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64,32,4,stride=2,padding=1),
                nn.GroupNorm(getGroupSize(32),32),
                nn.ReLU(True),
                nn.ConvTranspose2d(32,32,3,padding=1),
                nn.GroupNorm(getGroupSize(32),32),
                nn.ReLU(True),
                nn.ConvTranspose2d(32,32,4,stride=2,padding=1),
                nn.GroupNorm(getGroupSize(32),32),
                nn.ReLU(True),
                nn.ConvTranspose2d(32,1,3,padding=1),
                nn.Tanh(),
                )


    def forward(self,x,mid_features):

        x=self.up_conv1(x)
        if x.size(3) < mid_features.size(3):
            x = F.pad(x,(0,mid_features.size(3)-x.size(3),0,0),mode='replicate')
        if x.size(3) < mid_features.size(3):
            mid_features = F.pad(mid_features,(0,x.size(3)-mid_features.size(3),0,0),mode='replicate')
        x = torch.cat((x,mid_features),dim=1)
        x=self.up_conv2(x)
        return x



class DecoderNoSkip(nn.Module):

    def __init__(self,input_dim=512):
        super(DecoderNoSkip, self).__init__()


        self.up_conv1 = nn.Sequential(
                nn.ReLU(False),
                nn.ConvTranspose2d(input_dim,256,(6,3)),
                nn.GroupNorm(getGroupSize(256),256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256,256,3),
                nn.GroupNorm(getGroupSize(256),256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256,128,4,stride=2,padding=1),
                nn.GroupNorm(getGroupSize(128),128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128,128,3,padding=1),
                nn.GroupNorm(getGroupSize(128),128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128,64,4,stride=2,padding=1),
                nn.GroupNorm(getGroupSize(64),64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64,64,3,padding=1),
                nn.GroupNorm(getGroupSize(64),64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64,32,4,stride=2,padding=1),
                nn.GroupNorm(getGroupSize(32),32),
                nn.ReLU(True),
                nn.ConvTranspose2d(32,1,3,padding=1),
                nn.Tanh(),
                )


    def forward(self,x,mid_features):

        x=self.up_conv1(x)
        return x

class Encoder2(nn.Module):

    def __init__(self,out_dim=256):
        super(Encoder2, self).__init__()

        self.down_conv1 = nn.Sequential(
                nn.Conv2d(1,32,5,padding=2),
                nn.GroupNorm(getGroupSize(32),32),
                nn.ReLU(True),
                nn.AvgPool2d(2),
                nn.Conv2d(32,32,1),
                #nn.GroupNorm(getGroupSize(64),64)
                #nn.ReLU(True),
                )

        self.conv1 = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(32,32,3,padding=1),
                nn.GroupNorm(getGroupSize(32),32),
                nn.Dropout2d(0.1,True),
                nn.ReLU(True),
                nn.Conv2d(32,32,3,padding=1),
                )

        self.down_conv2 = nn.Sequential(
                nn.GroupNorm(getGroupSize(32),32),
                nn.ReLU(True),
                nn.AvgPool2d(2),
                nn.Conv2d(32,64,1),
                )

        self.conv2 = nn.Sequential(
                nn.GroupNorm(getGroupSize(64),64),
                nn.Dropout2d(0.1,True),
                nn.ReLU(True),
                nn.Conv2d(64,64,3,padding=1),
                nn.GroupNorm(getGroupSize(64),64),
                nn.Dropout2d(0.1,True),
                nn.ReLU(True),
                nn.Conv2d(64,64,3,padding=1),
                )

        self.down_conv3 = nn.Sequential(
                nn.GroupNorm(getGroupSize(64),64),
                nn.ReLU(True),
                nn.AvgPool2d(2),
                nn.Conv2d(64,128,3),
                nn.GroupNorm(getGroupSize(128),128),
                nn.Dropout2d(0.1,True),
                nn.ReLU(True),
                #nn.Conv2d(256,256,3),
                #nn.GroupNorm(getGroupSize(256),256)
                #nn.ReLU(True),
                nn.Conv2d(128,out_dim,(6,3)),
                )


    def forward(self,x):

        x=self.down_conv1(x)
        res=x
        x=self.conv1(x)
        x+=res
        x=self.down_conv2(x)
        res=x
        x=self.conv2(x)
        x+=res
        mid_features=x
        x=self.down_conv3(x)
        return x, mid_features
class Encoder3(nn.Module):

    def __init__(self):
        super(Encoder3, self).__init__()

        self.down_conv1 = nn.Sequential(
                nn.Conv2d(1,32,5,padding=2),
                nn.GroupNorm(getGroupSize(32),32),
                nn.ReLU(True),
                nn.AvgPool2d(2),
                nn.Conv2d(32,64,3,padding=1),
                #nn.GroupNorm(getGroupSize(64),64)
                #nn.ReLU(True),
                )

        self.conv1 = nn.Sequential(
                nn.GroupNorm(getGroupSize(64),64),
                nn.Dropout2d(0.1,True),
                nn.ReLU(True),
                nn.Conv2d(64,64,3,padding=1),
                nn.GroupNorm(getGroupSize(64),64),
                nn.Dropout2d(0.1,True),
                nn.ReLU(True),
                nn.Conv2d(64,64,3,padding=1),
                )

        self.down_conv2 = nn.Sequential(
                nn.GroupNorm(getGroupSize(64),64),
                nn.ReLU(True),
                nn.AvgPool2d(2),
                nn.Conv2d(64,128,3,padding=1),
                )

        self.conv2 = nn.Sequential(
                nn.GroupNorm(getGroupSize(128),128),
                nn.Dropout2d(0.1,True),
                nn.ReLU(True),
                nn.Conv2d(128,128,3,padding=1),
                nn.GroupNorm(getGroupSize(128),128),
                nn.Dropout2d(0.1,True),
                nn.ReLU(True),
                nn.Conv2d(128,128,3,padding=1),
                )

        self.down_conv3 = nn.Sequential(
                nn.GroupNorm(getGroupSize(128),128),
                nn.ReLU(True),
                nn.AvgPool2d(2),
                nn.Conv2d(128,256,3),
                nn.GroupNorm(getGroupSize(256),256),
                nn.Dropout2d(0.1,True),
                nn.ReLU(True),
                #nn.Conv2d(256,256,3),
                #nn.GroupNorm(getGroupSize(256),256)
                #nn.ReLU(True),
                nn.Conv2d(256,512,(6,3)),
                )


    def forward(self,x):

        x=self.down_conv1(x)
        res=x
        x=self.conv1(x)
        x+=res
        x=self.down_conv2(x)
        res=x
        x=self.conv2(x)
        x+=res
        mid_features=x
        x=self.down_conv3(x)
        return x, mid_features

class EncoderSpace(nn.Module):

    def __init__(self,out_dim=4):
        super(EncoderSpace, self).__init__()

        self.down_conv1 = nn.Sequential(
                nn.Conv2d(1,32,5,padding=2),
                nn.GroupNorm(getGroupSize(32),32),
                nn.ReLU(True),
                nn.AvgPool2d(2),
                nn.Conv2d(32,32,1),
                #nn.GroupNorm(getGroupSize(64),64)
                #nn.ReLU(True),
                )

        self.conv1 = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(32,32,3,padding=1),
                nn.GroupNorm(getGroupSize(32),32),
                nn.Dropout2d(0.1,True),
                nn.ReLU(True),
                nn.Conv2d(32,32,3,padding=1),
                )

        self.down_conv2 = nn.Sequential(
                nn.GroupNorm(getGroupSize(32),32),
                nn.ReLU(True),
                nn.AvgPool2d(2),
                nn.Conv2d(32,64,1),
                )

        self.conv2 = nn.Sequential(
                nn.GroupNorm(getGroupSize(64),64),
                nn.Dropout2d(0.1,True),
                nn.ReLU(True),
                nn.Conv2d(64,64,3,padding=1),
                nn.GroupNorm(getGroupSize(64),64),
                nn.Dropout2d(0.1,True),
                nn.ReLU(True),
                nn.Conv2d(64,64,3,padding=1),
                )

        self.down_conv3 = nn.Sequential(
                nn.GroupNorm(getGroupSize(64),64),
                nn.ReLU(True),
                nn.AvgPool2d(2),
                nn.Conv2d(64,128,3, padding=1),
                nn.GroupNorm(getGroupSize(128),128),
                nn.Dropout2d(0.1,True),
                nn.ReLU(True),
                #nn.Conv2d(256,256,3),
                #nn.GroupNorm(getGroupSize(256),256)
                #nn.ReLU(True),
                nn.Conv2d(128,out_dim,(3,3), padding=1),
                )


    def forward(self,x):

        x=self.down_conv1(x)
        res=x
        x=self.conv1(x)
        x+=res
        x=self.down_conv2(x)
        res=x
        x=self.conv2(x)
        x+=res
        mid_features=x
        x=self.down_conv3(x)
        return x, mid_features
class DecoderSpace(nn.Module):

    def __init__(self,input_dim=4):
        super(DecoderSpace, self).__init__()


        self.up_conv1 = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(input_dim,256,(3,3),padding=1),
                nn.GroupNorm(getGroupSize(256),256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256,256,3,padding=1),
                nn.GroupNorm(getGroupSize(256),256),
                nn.Dropout2d(0.1,True),
                nn.ReLU(True),
                nn.ConvTranspose2d(256,128,4,stride=2,padding=1),
                nn.GroupNorm(getGroupSize(128),128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128,128,3,padding=1),
                nn.GroupNorm(getGroupSize(128),128),
                nn.Dropout2d(0.1,True),
                nn.ReLU(True),
                nn.ConvTranspose2d(128,64,4,stride=2,padding=1),
                nn.GroupNorm(getGroupSize(64),64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64,64,3,padding=1),
                nn.GroupNorm(getGroupSize(64),64),
                nn.Dropout2d(0.1,True),
                nn.ReLU(True),
                nn.ConvTranspose2d(64,32,4,stride=2,padding=1),
                nn.GroupNorm(getGroupSize(32),32),
                nn.ReLU(True),
                nn.ConvTranspose2d(32,1,3,padding=1),
                nn.Tanh(),
                )


    def forward(self,x,mid_features):

        x=self.up_conv1(x)
        return x

class E_HWR(nn.Module):

    def __init__(self,n_class,n_in):
        super(E_HWR, self).__init__()


        self.classify = nn.Sequential(
                nn.Conv1d(n_in,512,3,1,1,1),
                nn.GroupNorm(getGroupSize(512),512),
                nn.Dropout(0.1,True),
                nn.ReLU(True),
                nn.Conv1d(512,512,3,1,2,2),
                nn.GroupNorm(getGroupSize(512),512),
                nn.Dropout(0.1,True),
                nn.ReLU(True),
                nn.Conv1d(512,512,3,1,4,4),
                nn.GroupNorm(getGroupSize(512),512),
                nn.Dropout(0.1,True),
                nn.ReLU(True),
                nn.Conv1d(512,512,5,1,2,1),
                nn.GroupNorm(getGroupSize(512),512),
                nn.Dropout(0.1,True),
                nn.ReLU(True),

                nn.Conv1d(512,n_class,1),
                nn.LogSoftmax(dim=1)
                )

    def forward(self,x):

        pred = self.classify(x.view(x.size(0),x.size(1),x.size(3)))
        pred = pred.permute(2,0,1) #more temporal dimension first
        return pred
class E_HWR_batch(nn.Module):

    def __init__(self,n_class,n_in):
        super(E_HWR_batch, self).__init__()


        self.classify = nn.Sequential(
                nn.Conv1d(n_in,512,3,1,1,1),
                nn.BatchNorm1d(512),
                nn.Dropout(0.1,True),
                nn.ReLU(True),
                nn.Conv1d(512,512,3,1,2,2),
                nn.BatchNorm1d(512),
                nn.Dropout(0.1,True),
                nn.ReLU(True),
                nn.Conv1d(512,512,3,1,4,4),
                nn.BatchNorm1d(512),
                nn.Dropout(0.1,True),
                nn.ReLU(True),
                nn.Conv1d(512,512,5,1,2,1),
                nn.BatchNorm1d(512),
                nn.Dropout(0.1,True),
                nn.ReLU(True),

                nn.Conv1d(512,n_class,1),
                nn.LogSoftmax(dim=1)
                )

    def forward(self,x):

        pred = self.classify(x.view(x.size(0),x.size(1),x.size(3)))
        pred = pred.permute(2,0,1) #more temporal dimension first
        return pred

class Encoder32(nn.Module):

    def __init__(self,out_dim=256):
        super(Encoder32, self).__init__()

        self.down_conv1 = nn.Sequential(
                nn.Conv2d(1,32,3,padding=1),
                nn.GroupNorm(getGroupSize(32),32),
                nn.ReLU(True),
                nn.Conv2d(32,32,1),
                #nn.GroupNorm(getGroupSize(64),64)
                #nn.ReLU(True),
                )

        self.conv1 = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(32,32,3,padding=1),
                nn.GroupNorm(getGroupSize(32),32),
                nn.Dropout2d(0.1,True),
                nn.ReLU(True),
                nn.Conv2d(32,32,3,padding=1),
                )

        self.down_conv2 = nn.Sequential(
                nn.GroupNorm(getGroupSize(32),32),
                nn.ReLU(True),
                nn.AvgPool2d(2),
                nn.Conv2d(32,64,1),
                )

        self.conv2 = nn.Sequential(
                nn.GroupNorm(getGroupSize(64),64),
                nn.Dropout2d(0.1,True),
                nn.ReLU(True),
                nn.Conv2d(64,64,3,padding=1),
                nn.GroupNorm(getGroupSize(64),64),
                nn.Dropout2d(0.1,True),
                nn.ReLU(True),
                nn.Conv2d(64,64,3,padding=1),
                )

        self.down_conv3 = nn.Sequential(
                nn.GroupNorm(getGroupSize(64),64),
                nn.ReLU(True),
                nn.AvgPool2d(2),
                nn.Conv2d(64,128,3),
                nn.GroupNorm(getGroupSize(128),128),
                nn.Dropout2d(0.1,True),
                nn.ReLU(True),
                #nn.Conv2d(256,256,3),
                #nn.GroupNorm(getGroupSize(256),256)
                #nn.ReLU(True),
                nn.Conv2d(128,out_dim,(6,3)),
                )


    def forward(self,x):

        x=self.down_conv1(x)
        res=x
        x=self.conv1(x)
        x+=res
        x=self.down_conv2(x)
        res=x
        x=self.conv2(x)
        x+=res
        mid_features=x
        x=self.down_conv3(x)
        return x, mid_features
class Decoder32NoSkip(nn.Module):

    def __init__(self,input_dim=512):
        super(Decoder32NoSkip, self).__init__()


        self.up_conv1 = nn.Sequential(
                nn.ReLU(False),
                nn.ConvTranspose2d(input_dim,256,(6,3)),
                nn.GroupNorm(getGroupSize(256),256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256,256,3),
                nn.GroupNorm(getGroupSize(256),256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256,128,4,stride=2,padding=1),
                nn.GroupNorm(getGroupSize(128),128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128,128,3,padding=1),
                nn.GroupNorm(getGroupSize(128),128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128,64,4,stride=2,padding=1),
                nn.GroupNorm(getGroupSize(64),64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64,64,3,padding=1),
                nn.GroupNorm(getGroupSize(64),64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64,32,3,stride=1,padding=1),
                nn.GroupNorm(getGroupSize(32),32),
                nn.ReLU(True),
                nn.ConvTranspose2d(32,1,3,padding=1),
                nn.Tanh(),
                )


    def forward(self,x,mid_features):

        x=self.up_conv1(x)
        return x
