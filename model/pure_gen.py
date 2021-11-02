#from https://github.com/rosinality/style-based-gan-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from math import sqrt
import random


class SpacedGenerator(nn.Module):
    def __init__(self, n_class, style_size, dim=256, output_dim=1, n_style_trans=6, emb_dropout=False, append_style=False,small=False):
        super(SpacedGenerator, self).__init__()
        fused=True
        self.append_style = append_style
        if append_style:
            in_ch = n_class + style_size
        else:
            in_ch = n_class
        self.conv = nn.Sequential(
                StyledConvBlock(in_ch,dim,upsample=False,style_dim=style_size,initial=True),
                StyledConvBlock(dim,dim//2,upsample=True,only_vertical=True,fused=False,style_dim=style_size),
                StyledConvBlock(dim//2,dim//4,upsample=True,only_vertical=True,fused=False,style_dim=style_size),
                StyledConvBlock(dim//4,dim//8,upsample=True,only_vertical=False,fused=fused,style_dim=style_size),
                StyledConvBlock(dim//8,dim//16,upsample=not small,only_vertical=False,fused=fused,style_dim=style_size),
                )

        self.out = nn.Sequential(EqualConv2d(dim//16, output_dim, 1), nn.Tanh())

        layers = [PixelNorm()]
        drop = emb_dropout if type(emb_dropout) is float else 0.5
        for i in range(n_style_trans):
            layers.append(nn.Linear(style_size, style_size))
            if emb_dropout and i<n_style_trans-1:
                layers.append(nn.Dropout(drop,True))
            layers.append(nn.LeakyReLU(0.2,True))

        self.style_emb = nn.Sequential(*layers)
        self.gen=self.conv

    def forward(self, content,style,return_intermediate=False): #, noise=None):
        content = content.permute(1,2,0) #swap [T,b,cls] to [b,cls,T]
        content = content.view(content.size(0),content.size(1),1,content.size(2)) #now [b,cls,H,W]

        style = self.style_emb(style)
        if self.append_style:
            content=torch.cat((content,style[:,:,None,None].expand(-1,-1,1,content.size(3))),dim=1)
        x,_ = self.conv((content,style))
        return self.out(x)

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = nn.Linear(style_dim, in_channel * 2)

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(1, channel, 1, 1)*0.01)

    def forward(self, image, noise):
        return image + self.weight * noise
class BlurFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None


class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None


blur = BlurFunction.apply


class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)
        # return F.conv2d(input, self.weight, padding=1, groups=input.shape[1])


class StyledConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        style_dim=512,
        initial=False,
        upsample=False,
        only_vertical=False,
        fused=False,
    ):
        super().__init__()

        if initial=='1d':
            self.conv1 = nn.ConvTranspose2d(
                in_channel, out_channel, (4,4), padding=(0,0), stride=(1,4)
            )

        elif initial:
            self.conv1 = nn.ConvTranspose2d(
                in_channel, out_channel, (4,3), padding=(0,1)
            )

        else:
            if upsample:
                if fused:
                    self.conv1 = nn.Sequential(
                        FusedUpsample(
                            in_channel, out_channel, kernel_size, padding=padding, only_vertical=only_vertical
                        ),
                        Blur(out_channel),
                    )

                else:
                    if only_vertical:
                        scale = (2,1)
                    else:
                        scale = 2
                    self.conv1 = nn.Sequential(
                        nn.Upsample(scale_factor=scale, mode='nearest'),
                        nn.Conv2d(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),
                    )

            else:
                self.conv1 = nn.Conv2d(
                    in_channel, out_channel, kernel_size, padding=padding
                )

        self.noise1 = equal_lr(NoiseInjection(out_channel))
        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.noise2 = equal_lr(NoiseInjection(out_channel))
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input):
        input, style = input
        #print(input.size())
        out = self.conv1(input)
        out = self.noise1(out, torch.randn_like(out))
        out = self.lrelu1(out)
        out = self.adain1(out, style)
        #print('out: {}'.format(out.size()))

        out = self.conv2(out)
        out = self.noise2(out, torch.randn_like(out))
        out = self.lrelu2(out)
        out = self.adain2(out, style)
        #print('out: {}'.format(out.size()))
        return out, style

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class FusedUpsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0, only_vertical=False):
        super().__init__()

        if only_vertical:
            self.stride = (2,1)
        else:
            self.stride = 2

        weight = torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding
    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4
        out = F.conv_transpose2d(input, weight, self.bias, stride=self.stride, padding=self.pad)

        return out

class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1 if len(input.size())>1 else 0, keepdim=True) + 1e-8)

class UnstyledConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        style_dim=512,
        initial=False,
        upsample=False,
        only_vertical=False,
        fused=False,
        use_noise=False,
        use_second=True,
    ):
        super().__init__()
        self.use_noise=use_noise

        if initial:
            self.conv1 = nn.ConvTranspose2d(
                in_channel, out_channel, (4,3), padding=(0,1)
            )

        else:
            if upsample:
                if fused:
                    self.conv1 = nn.Sequential(
                        FusedUpsample(
                            in_channel, out_channel, kernel_size, padding=padding, only_vertical=only_vertical
                        ),
                        Blur(out_channel),
                    )

                else:
                    if only_vertical:
                        scale = (2,1)
                    else:
                        scale = 2
                    self.conv1 = nn.Sequential(
                        nn.Upsample(scale_factor=scale, mode='nearest'),
                        nn.Conv2d(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),
                    )

            else:
                self.conv1 = nn.Conv2d(
                    in_channel, out_channel, kernel_size, padding=padding
                )

                #self.noise1 = equal_lr(NoiseInjection(out_channel))
        #self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.norm = nn.GroupNorm(4,out_channel)
        self.lrelu1 = nn.LeakyReLU(0.2)
        
        if use_second:
            self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding)
            #self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
            self.norm2 = nn.GroupNorm(4,out_channel)
            self.lrelu2 = nn.LeakyReLU(0.2)
            if self.use_noise:
                self.noise1 = equal_lr(NoiseInjection(out_channel))
                self.noise2 = equal_lr(NoiseInjection(out_channel))
        else:
            self.conv2 = None


    def forward(self, input):
        input, style = input
        out = self.conv1(input)
        if self.use_noise:
            out = self.noise1(out, torch.randn_like(out))
        out = self.lrelu1(out)
        #out = self.adain1(out, style)
        out = self.norm(out)

        if self.conv2 is not None:
            out = self.conv2(out)
            if self.use_noise:
                out = self.noise2(out, torch.randn_like(out))
            out = self.lrelu2(out)
            #out = self.adain2(out, style)
            out = self.norm2(out)
        return out, style
