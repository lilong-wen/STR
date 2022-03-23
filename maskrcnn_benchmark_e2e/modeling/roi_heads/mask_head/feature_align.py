# Written by Minghui Liao
import torch 

import torch.nn as nn
import torch.nn.functional as F


gpu_device = torch.device("cuda")
cpu_device = torch.device("cpu")


def conv3x3(in_planes, out_planes, stride=1,pad=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=pad, bias=False)

def conv5x5(in_planes, out_planes, stride=1,pad=2):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=pad, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),)   

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,finalrelu=True,pad=1):
        super(BasicBlock, self).__init__()

        self.conv1_f = conv3x3(inplanes, planes, stride,pad=pad)
        self.bn1_f = nn.BatchNorm2d(planes)
        self.relu_f = nn.ReLU(inplace=True)
        self.conv2_f = conv3x3(planes, planes,pad=pad)
        self.bn2_f = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.finalrelu = finalrelu

    def forward(self, x):
        residual = x
        out = self.conv1_f(x)
        out = self.bn1_f(out)
        out = self.relu_f(out)
        out = self.conv2_f(out)
        out = self.bn2_f(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        if self.finalrelu:
            out = self.relu_f(out)

        return out

class feature_align_for_x_and_target (nn.Module):
    def __init__(self, block=BasicBlock):
        self.inplanes = 64
        super(feature_align_for_x_and_target , self).__init__()
        # self.alphabet = args.alphabet
        self.conv1_f = nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1,bias=True) #size/2
        self.conv2_f = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.conv3_f = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv4_f = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0, bias=True)
        self.bn1_f = nn.BatchNorm2d(64)
        self.relu_f = nn.ReLU(inplace=True)
        self.layer1_f = self._make_layer(block, 64, 2,stride=(2,1))
        self.layer2_f = self._make_layer(block, 128, 2,stride=(2,2)) 

        self.avgpool_f = nn.AdaptiveMaxPool2d((7, 7))
        self.maxpool_f = nn.MaxPool2d(2,stride=(2,1))
        self.down_f = conv1x1(256,128)
        # self.linear = nn.Linear(720, 360)
        # self.linear2 = nn.Linear(64, 360)

        self.upsample_f =  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down_channel_f = double_conv(128+64,128)
        self.padding_f = nn.ZeroPad2d((0,1,0,0))
        self.dropout_f = torch.nn.Dropout2d(p=0.2, inplace=False)

        self.seq_encoder_x = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.Conv2d(128, 128, 3, stride=2, padding=0),
            nn.ReLU(inplace=True),
        )

 
    def _make_layer(self, block, planes, blocks, stride=1,finalrelu = True):
        # block: name of basic blocks
        # planes: channels of filters
        # blocks: number of ResBlocks, each block has two conv layers 
        # self.inplanes: set to constant number(64), and downsample if input channel != output channel 
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion,1),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,1, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(nn.MaxPool2d(stride,stride=stride))
            layers.append(block(self.inplanes, planes,finalrelu = finalrelu))

        return nn.Sequential(*layers)

    def encoder(self,x):

        x = self.conv1_f(x)
        x = self.bn1_f(x)
        x = self.relu_f(x)
        x_1 = self.layer1_f(x)
        x_2 = self.layer2_f(x_1) 

        # upsample
        x_up = self.upsample_f(x_2) 
        if x_up.shape[-1]!=x_1.shape[-1]:
            x_up= self.padding_f(x_up)
        x = torch.cat([x_up, x_1], dim=1)
        x = self.down_channel_f(x) 
        x = self.avgpool_f(x) 
        # x = self.down(x) 
        # x = self.relu_f(x)

        # make similarity maps
        # x = x.view(x.shape[0],-1,x.shape[-1]) 
        x = x/(torch.norm(x,dim=1,keepdim=True) + 1e-6)
        return x


    def forward(self,target, x):
        # encoding  
        target_align = self.encoder(target)
        x_align = self.seq_encoder_x(x)

        return target_align, x_align