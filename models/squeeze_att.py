import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class conv_block_att(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block_att,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(ch_out),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.conv(x)
        return x

#import matplotlib.pyplot as plt

class SqueezeAttentionBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SqueezeAttentionBlock, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = conv_block(ch_in, ch_out)
        self.conv_atten = conv_block_att(ch_in, ch_out)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        H,W = x.shape[-2:]
        
        x_res = self.conv(x)
        y = self.avg_pool(x)
        y = self.conv_atten(y)
        #y = self.upsample(y)
        y = F.interpolate(y,size=(H,W),mode='nearest')

        return (y * x_res) + y