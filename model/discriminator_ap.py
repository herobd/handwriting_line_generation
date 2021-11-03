import torch.nn as nn
from torch.nn import Parameter
import torch
import torch.nn.functional as F
from utils.util import getGroupSize

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class DiscriminatorAP(nn.Module):

    def __init__(self,dim=64, use_low=False,use_med=True,small=False):
        super(DiscriminatorAP, self).__init__()
        self.use_low = use_low
        self.use_med = use_med
        leak=0.1

        self.in_conv = nn.Sequential(
                nn.Conv2d(1, dim, 7, stride=1, padding=(0,3)),
                nn.GroupNorm(getGroupSize(dim),dim), # Experiments with other GAN showed better results not using spectral on first layer
                nn.LeakyReLU(leak,True)
                )
    
        convs1_pad_v = 0 if not small else 1
        convs1= [
                SpectralNorm(nn.Conv2d(dim, dim, 3, stride=1, padding=(convs1_pad_v,1))),
                nn.LeakyReLU(leak,True)
                ]
        if not small:
            convs1.append(nn.AvgPool2d(2),)
        convs1+=[
                SpectralNorm(nn.Conv2d(dim, 2*dim, 3, stride=1, padding=(convs1_pad_v,1))),
                nn.Dropout2d(0.05,True),
                nn.LeakyReLU(leak,True),
                ]


        self.convs1 = nn.Sequential(*convs1)
        self.convs2 = nn.Sequential(
                SpectralNorm(nn.Conv2d(2*dim, 2*dim, 3, stride=1, padding=(0,1))),
                nn.LeakyReLU(leak,True),
                nn.AvgPool2d(2),
                )
        convs3=[nn.Conv2d(2*dim, 2*dim, 3, stride=1, padding=(0,1)),
                nn.GroupNorm(getGroupSize(2*dim),2*dim),
                nn.LeakyReLU(leak,True),
                nn.AvgPool2d(2),
                SpectralNorm(nn.Conv2d(2*dim, 4*dim, 3, stride=1, padding=(0,1))),
                nn.Dropout2d(0.05,True),
                nn.LeakyReLU(leak,True),
                ]
        self.convs3 = nn.Sequential(*convs3)
        if self.use_med:
            self.finalMed = nn. Sequential(
                    SpectralNorm(nn.Conv2d(4*dim, 1, 3, stride=1, padding=(0,1))),
                    )
        if self.use_low:
            self.convs4 = nn.Sequential(
                SpectralNorm(nn.Conv2d(4*dim, 2*dim, 3, stride=1, padding=(0,1))), #after this it should be flat
                nn.Dropout2d(0.025,True),
                nn.LeakyReLU(leak,True),
                nn.AvgPool2d((1,2)), #flat, so only operate horz
                SpectralNorm(nn.Conv2d(2*dim, 4*dim, (1,3), stride=1, padding=(0,1))),
                nn.Dropout2d(0.025,True),
                nn.LeakyReLU(leak,True),
                SpectralNorm(nn.Conv2d(4*dim, 4*dim, (1,3), stride=1, padding=(0,1))),
                nn.Dropout2d(0.025,True),
                nn.LeakyReLU(leak,True),
                nn.AvgPool2d((1,2)), #flat, so only operate horz
                SpectralNorm(nn.Conv2d(4*dim, 4*dim, (1,3), stride=1, padding=(0,1))),
                nn.Dropout2d(0.025,True),
                nn.LeakyReLU(leak,True),
                SpectralNorm(nn.Conv2d(4*dim, 1, 1, stride=1, padding=(0,0))),
                )




        #self.fc = SpectralNorm(nn.Linear(w_g * w_g * 512, 1))

    def forward(self, x,return_features=False):
        batch_size = x.size(0)
        
        x = self.in_conv(x) #64x58xW /x26x/

        m = self.convs1(x) #128x26xW /x26x/

        mL = self.convs2(m) #128x12xW
        mL = self.convs3(mL) #256x4xW
        if return_features:
            return mL,self.convs4(mL)
        if self.use_med:
            pM = self.finalMed(mL)

        if self.use_low:
            pL = self.convs4(mL)
            if self.use_med:
                return [pM.view(batch_size,-1),pL.view(batch_size,-1)]
            else:
                return [pL.view(batch_size,-1)]
            #return torch.cat( (pM.view(batch_size,-1),pL.view(batch_size,-1)), dim=1)#.view(-1)
        else:
            return [pM.view(batch_size,-1)]
