#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 13:25:24 2021

@author: hasan
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

""" Atruos Spatal Pyramid Pooling Module"""

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel, dilation = 1):
        modules = [
            nn.Conv2d(in_channels, 
                      out_channels,
                      kernel,
                      padding=dilation if(kernel>1) else 0,
                      dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)
        
class SeparableASPPConv(nn.Sequential):
    def __init__(self,in_channels,out_channels,kernel,dilation = 1):
        depthwise_conv = nn.Conv2d(in_channels,
                                   in_channels,
                                   kernel,
                                   padding = dilation,
                                   dilation = dilation,
                                   groups = in_channels,
                                   bias = False
                                   )
        pointwise_conv  = nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size = 1,
                                    bias = False)
        super().__init__(depthwise_conv,
                         pointwise_conv,
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True))
        
class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        batch_size,C,H,W = x.shape
        x = self.module(x)
        #print(x.shape)
        x = F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True)
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels,
                 out_channels=256,
                 atrous_rates = [6,12,18],
                 dropout_rate = 0,
                 separable = False):
        super(ASPP, self).__init__()
        
        modules = []
        modules.append(ASPPConv(in_channels, out_channels, 1))
        _c = SeparableASPPConv if(separable) else ASPPConv
        for rate in atrous_rates:
            
            modules.append(_c(in_channels, out_channels, 3,  rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = ASPPConv(len(self.convs) * out_channels, out_channels, 1)

    def forward(self, x):
        res = []
        for i,conv in enumerate(self.convs):
            #print(i)
            res.append(conv(x))
            #print(res[i].shape)
        res = torch.cat(res, dim=1)
        return self.project(res)


class DenseASPPConv(nn.Module):
    def __init__(self,in_channels,
                 mid_channels,
                 out_channels,
                 rate = 3, 
                 separable = False,
                 dropout_rate = 0.0):
        super().__init__()
        self.conv1x1 = ASPPConv(in_channels = in_channels, out_channels = mid_channels, kernel = 1)
        _c = SeparableASPPConv if(separable) else ASPPConv
        self.dilated_conv = _c(in_channels = mid_channels,
                               out_channels = out_channels,
                               kernel = 3,
                               dilation = rate)
        self.drop = nn.Dropout(dropout_rate)
    def forward(self,x):
        x = self.conv1x1(x)
        x = self.dilated_conv(x)
        return self.drop(x)
    
class DenseASPP(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels = 256,
                 inter_channels=256,
                 atrous_rates = [3,6,12,18],
                 dropout_rate = 0,
                 separable = False):
        super().__init__()
        modules = []
        for i,rate in enumerate(atrous_rates):
            modules.append(
                            DenseASPPConv(in_channels = in_channels + i * inter_channels,
                                          mid_channels = mid_channels,
                                          out_channels = inter_channels,
                                          rate = rate,
                                          separable = separable,
                                          dropout_rate=dropout_rate
                                )
                            )
        self.out_channels = in_channels + len(atrous_rates) * inter_channels
        self.convs = nn.ModuleList(modules)
    def forward(self,feats):
        for conv in self.convs:
            x = conv(feats)
            feats = torch.cat([feats,x],dim = 1)
        return feats
        
    
'''    
x = torch.rand((2,128,32,32)).cuda()
m = DenseASPP(128,32,64).cuda()
<<<<<<< HEAD
print(m.out_channels)
y = m(x)
print(y.size())


y = m(x)
'''

