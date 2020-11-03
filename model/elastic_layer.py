# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import torch
import torch.nn as nn

class ElasticDeform(nn.Module):

    def __init__(self, ic=3, n_layers=3, kernel=5, channels=16, downsample=2, blur=9, max_dx=5, max_dy=10):
        super(ElasticDeform, self).__init__()

        self.layers = nn.ModuleList()

        for idx in range(n_layers-1):
            ps = (kernel-1)//2
            conv = nn.Conv2d(ic, channels, kernel_size=kernel, stride=1, padding=ps)
            nn.init.xavier_uniform_(conv.weight, gain=1.414)
            self.layers.append(conv)
            self.layers.append(nn.ReLU())
            if downsample > 1 and idx < (n_layers-2):
                self.layers.append(nn.AvgPool2d(downsample, stride=downsample, ceil_mode=True))
            ic = channels
        self.dy_pred_layer = nn.Conv2d(ic, 1, kernel_size=3, stride=1, padding=1)
        self.dx_pred_layer = nn.Conv2d(ic, 1, kernel_size=3, stride=1, padding=1)
        self.max_dx = max_dx
        self.max_dy = max_dy

        self.blur_pad = (blur-1) // 2
        self.blur_weights = (1. / blur ** 2) * torch.ones(1, 1, blur, blur)

    def forward(self, x):
        self.blur_weights = self.blur_weights.to(x.device)

        h,w = x.shape[2:]

        original_x = x
        for layer in self.layers:
            x = layer(x)

        dy_pred_small = self.dy_pred_layer(x)
        dy_pred_big = nn.functional.interpolate(dy_pred_small, size=(h,w), mode='bilinear', align_corners=False)
        # tanh gets the offsets to the range (-1,1), which is what is used by grid_sample()
        # we want to scale it down so that we have a maximum pixel displacement of max_delta
        dy_pred_big = (float(self.max_dy) / h) * torch.tanh(dy_pred_big)
        # blur the offsets to get local smoothness
        dy_pred_big = nn.functional.conv2d(dy_pred_big, self.blur_weights, stride=1, padding=self.blur_pad)
        # y_identity with x_identity yields the identity transform when input to nn.functional.grid_sample
        y_identity = torch.linspace(-1., 1., steps=h)[:,None].repeat(1, w)[None,None,:,:].to(x.device)
        y_mapping = (y_identity + dy_pred_big).squeeze(1)

        dx_pred_small = self.dx_pred_layer(x)
        dx_pred_big = nn.functional.interpolate(dx_pred_small, size=(h,w), mode='bilinear', align_corners=False)
        dx_pred_big = (float(self.max_dx) / w) * torch.tanh(dx_pred_big)
        dx_pred_big = nn.functional.conv2d(dx_pred_big, self.blur_weights, stride=1, padding=self.blur_pad)
        x_identity = torch.linspace(-1., 1., steps=w)[None,:].repeat(h, 1)[None,None,:,:].to(x.device)
        x_mapping = (x_identity + dx_pred_big).squeeze(1)

        combined_grid = torch.cat((x_mapping[:,:,:,None], y_mapping[:,:,:,None]), dim=3)
        resampled = nn.functional.grid_sample(original_x, combined_grid, mode='bilinear', padding_mode='border')

        return resampled

class ElasticDeformWithStyle(nn.Module):

    def __init__(self, ic=3, n_layers=3, kernel=5, channels=16, downsample=2, blur=9, max_dx=5, max_dy=10, style_dim=256):
        super(ElasticDeformWithStyle, self).__init__()

        self.layers = nn.ModuleList()

        for idx in range(n_layers):
            ps = (kernel-1)//2
            conv = nn.Conv2d(ic, channels, kernel_size=kernel, stride=1, padding=ps)
            nn.init.xavier_uniform_(conv.weight, gain=1.414)
            self.layers.append(conv)
            self.layers.append(nn.ReLU())
            if downsample > 1 and idx < (n_layers-2):
                self.layers.append(nn.AvgPool2d(downsample, stride=downsample, ceil_mode=True))
            ic = channels
        self.pred_layer = nn.Sequential(
                            nn.Conv2d(ic+style_dim, style_dim, kernel_size=1, stride=1, padding=1),
                            nn.ReLU(True),
                            nn.Conv2d(style_dim, 2, kernel_size=1, stride=1, padding=1)
                            )
        self.pred_layer[2].weight.data /=2
        self.pred_layer[2].bias.data /=2

        self.max_dx = max_dx
        self.max_dy = max_dy

        self.blur_pad = (blur-1) // 2
        self.blur_weights = (1. / blur ** 2) * torch.ones(1, 1, blur, blur)

    def forward(self, input):
        x,style = input
        self.blur_weights = self.blur_weights.to(x.device)

        h,w = x.shape[2:]

        original_x = x
        for layer in self.layers:
            x = layer(x)

        style_expanded = style.view(style.size(0),style.size(1),1,1).expand(-1,-1,x.size(2),x.size(3))
        pred_small = self.pred_layer(torch.cat((x,style_expanded),dim=1))
        dy_pred_small = pred_small[:,0:1]
        dy_pred_big = nn.functional.interpolate(dy_pred_small, size=(h,w), mode='bilinear', align_corners=False)
        # tanh gets the offsets to the range (-1,1), which is what is used by grid_sample()
        # we want to scale it down so that we have a maximum pixel displacement of max_delta
        dy_pred_big = (float(self.max_dy) / h) * torch.tanh(dy_pred_big)
        # blur the offsets to get local smoothness
        dy_pred_big = nn.functional.conv2d(dy_pred_big, self.blur_weights, stride=1, padding=self.blur_pad)
        # y_identity with x_identity yields the identity transform when input to nn.functional.grid_sample
        y_identity = torch.linspace(-1., 1., steps=h)[:,None].repeat(1, w)[None,None,:,:].to(x.device)
        y_mapping = (y_identity + dy_pred_big).squeeze(1)

        dx_pred_small = pred_small[:,1:2]
        dx_pred_big = nn.functional.interpolate(dx_pred_small, size=(h,w), mode='bilinear', align_corners=False)
        dx_pred_big = (float(self.max_dx) / w) * torch.tanh(dx_pred_big)
        dx_pred_big = nn.functional.conv2d(dx_pred_big, self.blur_weights, stride=1, padding=self.blur_pad)
        x_identity = torch.linspace(-1., 1., steps=w)[None,:].repeat(h, 1)[None,None,:,:].to(x.device)
        x_mapping = (x_identity + dx_pred_big).squeeze(1)

        combined_grid = torch.cat((x_mapping[:,:,:,None], y_mapping[:,:,:,None]), dim=3)
        resampled = nn.functional.grid_sample(original_x, combined_grid, mode='bilinear', padding_mode='border')

        return resampled,style

if __name__ == "__main__":
    import torch.optim
    from torchvision import datasets, transforms
    import numpy as np
    import cv2


    def get_target(x):
        y = torch.zeros_like(x)
        y[:,:,1:28,2:28] = x[:,:,0:27,0:26]
        return y

    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, _) in enumerate(train_loader):
            target = get_target(data)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.l1_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                
    def dump(epoch, data, target, output):
        data = data.cpu().numpy()
        target = target.cpu().numpy()
        output = output.cpu().numpy()
        for idx in range(data.shape[0]):
            x = (255 * data[idx,0]).astype(np.uint8)
            t = (255 * target[idx,0]).astype(np.uint8)
            y = (255 * output[idx,0]).astype(np.uint8)
            combined = np.concatenate((y[:,:,np.newaxis], t[:,:,np.newaxis], y[:,:,np.newaxis]), axis=2)
            cv2.imwrite('tmp/%d/%d_input.png' % (epoch, idx), x)
            cv2.imwrite('tmp/%d/%d_target.png' % (epoch, idx), t)
            cv2.imwrite('tmp/%d/%d_output.png' % (epoch, idx), y)
            cv2.imwrite('tmp/%d/%d_combined.png' % (epoch, idx), combined)


    def test(model, device, test_loader, epoch):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            first = True
            for data, _ in test_loader:
                target = get_target(data)
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = nn.functional.l1_loss(output, target)
                test_loss += loss
                if first:
                    dump(epoch, data, target, output)
                first = False

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))


    device = torch.device('cuda')
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=64, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=10, shuffle=False, **kwargs)

    model = ElasticDeform(ic=1, n_layers=3, blur=3, max_dx=5, max_dy=5).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    test(model, device, test_loader, 0)
    for epoch in range(1, 10):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, epoch)



