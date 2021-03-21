

from torchdiffeq import odeint
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import random

import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
from tensorboardX import SummaryWriter

import data_loader

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

class Drift(nn.Module):

    def __init__(self, dim):
        super(Drift, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)

    def forward(self, t, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out    

class Diffusion(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Diffusion, self).__init__()
        self.norm1 = norm(dim_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim_in, dim_out, 3, 1, 1)
        self.norm2 = norm(dim_in)
        self.conv2 = ConcatConv2d(dim_in, dim_out, 3, 1, 1)
        self.fc = nn.Sequential(norm(dim_out), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(dim_out, 1), nn.Sigmoid())
    def forward(self, t, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.fc(out)
        return out

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)
    
class SDENet_mnist(nn.Module):
    def __init__(self, device, num_classes=10, dim = 64):
        super(SDENet_mnist, self).__init__()
        self.device = device
        
        self.downsampling_layers = nn.Sequential(
            nn.Conv2d(1, dim, 3, 1),
            norm(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            norm(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 4, 2, 1),
        ) # [500, 1, 28, 28]  ->  [500, 64, 6, 6]
        self.drift = Drift(dim) # [500, 64, 6, 6]  ->  [500, 64, 6, 6]
        
        self.diffusion = Diffusion(dim, dim) # [500, 64, 6, 6]  ->  [500, 1, 1, 1]
        self.fc_layers = nn.Sequential(norm(dim), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(dim, 10)) # [500, 64, 6, 6] -> [500, 10]
        self.sigma = 500

    def forward(self, x, training_diffusion=False):
        out = self.downsampling_layers(x)
        if not training_diffusion:
            t = 0
            diffusion_term = self.sigma * self.diffusion(t, out).reshape(-1,1,1,1)
            out = odeint(self.drift, diffusion_term, out, torch.tensor([0, 1]).float().to(device), diffusion_mode="fixed")[1]
            final_out = self.fc_layers(out)
        else:
            t = 0
            final_out = self.diffusion(t, out.detach())
        return final_out

def test(net, epoch, test_loader_inDomain):
    net.eval()
    correct = 0
    total = 0

    real_sigma = 0
    fake_sigma = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader_inDomain):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = 0
            for j in range(10):
                current_batch = net(inputs)
                outputs = outputs + F.softmax(current_batch, dim = 1)

            outputs = outputs/10
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            real_sigma = real_sigma + net(inputs, training_diffusion=True).sum().item()
            fake_sigma = fake_sigma + net(2 * torch.randn(batch_size,1, imageSize, imageSize, device = device) + inputs, training_diffusion=True).sum().item()
    net.train()
    
    return 100. * correct/total, real_sigma/total, fake_sigma/total
        # print('Test epoch: {} | Acc: {:.6f} ({}/{})'
        # .format(epoch, 100.*correct/total, correct, total))
    

if __name__== "__main__":
    imageSize = 28
    epoch = 100
    batch_size = 128
    GPU_NUM = 2
    random_num = 777
    device = torch.device('cuda:' + str(GPU_NUM) if torch.cuda.is_available() else 'cpu')


    torch.manual_seed(random_num)
    random.seed(random_num)
    if device == 'cuda':
        cudnn.benchmark = True
        torch.cuda.manual_seed(random_num)

    net = SDENet_mnist(device).to(device)

    real_label = 0
    fake_label = 1

    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.BCELoss()

    optimizer_F = optim.SGD([ {'params': net.downsampling_layers.parameters()}, {'params': net.drift.parameters()},
    {'params': net.fc_layers.parameters()}], lr=0.1, momentum=0.9, weight_decay=5e-4)

    optimizer_G = optim.SGD([ {'params': net.diffusion.parameters()}], lr=0.01, momentum=0.9, weight_decay=5e-4)

    #use a smaller sigma during training for training stability
    net.sigma = 20

    train_loader_inDomain, test_loader_inDomain = data_loader.getDataSet("mnist", batch_size, batch_size, 28)

    writer = SummaryWriter("runs/sde_net")
    total_i = 0
    total_test = 0
    for e in range(epoch):
        for x, y in train_loader_inDomain:
            total_i += 1
            inputs, targets = x.to(device), y.to(device)
            optimizer_F.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_F.step()
            
            
            writer.add_scalar('losses/task_loss', loss, total_i)

            #training with out-of-domain data
            label = torch.full((batch_size,1), real_label, device=device).float()
            optimizer_G.zero_grad()
            predict_in = net(inputs, training_diffusion=True)
            loss_in = criterion2(predict_in, label)
            loss_in.backward()
            writer.add_scalar('losses/loss_in', loss_in, total_i)

            label.fill_(fake_label)
            inputs_out = 2 * torch.randn(batch_size,1, imageSize, imageSize, device = device) + inputs
            predict_out = net(inputs_out, training_diffusion=True)
            loss_out = criterion2(predict_out, label)
            writer.add_scalar('losses/loss_out', loss_out, total_i)

            loss_out.backward()
            optimizer_G.step()
            
            if total_i % 5 == 4:
                total_test += 1
                acc, real_sigma, fake_sigma = test(net, epoch, test_loader_inDomain)
                writer.add_scalar('score/Acc', acc, total_test)
                writer.add_scalar('score/real_sigma', real_sigma, total_test)
                writer.add_scalar('score/fake_sigma', fake_sigma, total_test)