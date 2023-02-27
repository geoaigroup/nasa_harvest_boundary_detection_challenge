import torch
from torch.nn.functional import interpolate,pad

import json
import numpy as np
import pandas as pd

from math import ceil



def load_json(path):
    return json.load(open(path,'r'))

def get_model(model_name):
    if model_name == 'UTAE':
        from models import UTAE
        return UTAE

    elif model_name == 'UNET3D':
        from models import Unet3D
        return Unet3D
    
    elif model_name == 'UNETPP3D':
        from models import UnetPlusPlus3D
        return UnetPlusPlus3D
        
    elif model_name == 'FUNET3D':
        from models import FU3D
        return FU3D
        
    elif model_name == 'refinenetv0':
        from models import RefineNetv0
        return RefineNetv0
        
    elif model_name == 'UNETLSTM':
        from models import UnetLstm
        return UnetLstm
    
    else:
        raise ValueError(f'model {model_name} is not implemented!!!')

def load_trained_model(path,use_last_model=False):
    cfg = json.load(open(f'{path}/configs.json'))
    model = get_model(cfg['model_name'])(**cfg['model'])
    
    if use_last_model:
        checkpoint = torch.load(f'{path}/last_model.pth', map_location='cpu')
    else:
        checkpoint = torch.load(f'{path}/best_model.pth', map_location='cpu')

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model,cfg



def noise_filter(washed,mina):
    values = np.unique(washed)
    for val in values[1:]:
        area = (washed[washed == val]>0).sum()
        if(area<=mina):  
            washed[washed == val] = 0
    return washed

def remove_boundary_positives(img,pixels=20):
    H,W = img.shape[-2:]
    bg = torch.zeros_like(img,dtype=img.dtype,device=img.device)

    s1 = min(pixels,H-1)
    e1 = max(s1+1,H-pixels) 
    s2 = min(pixels,W-1)
    e2 = max(s2+1,W-pixels) 
    
    bg[...,s1:e1,s2:e2] = img[...,s1:e1,s2:e2]
    return bg

def remove_boundary_positives_np(img,pixels=20):
    H,W = img.shape[:2]
    bg = np.zeros_like(img,dtype=img.dtype)

    s1 = min(pixels,H-1)
    e1 = max(s1+1,H-pixels) 
    s2 = min(pixels,W-1)
    e2 = max(s2+1,W-pixels) 
    bg[s1:e1,s2:e2,...] = img[s1:e1,s2:e2,...]
    return bg

def resize_pad(x,padsize=None,resize=None,pad_value = -1):
  
    if padsize is None and resize is None:
        return x
    
    input_shape = x.shape
    if len(input_shape) == 5:
        B,T,C,H,W = input_shape
        x = x.view(B*T,C,H,W)
        
    if resize is not None:
        x = interpolate(x,size=(resize,resize),mode='bilinear')

    if padsize is not None:
        if resize is not None:
            ppix = padsize - resize 
        else:
            ppix = padsize - 256
        s = ppix // 2
        e = ppix - s
        x = pad(x, (s,e,s,e), mode='constant', value=pad_value)

    
    if len(input_shape) == 5:
        H,W = x.shape[-2:]
        x = x.view(B,T,C,H,W)
    
    return x

def unpad_resize(x,padsize=None,resize=None):
    if padsize is None and resize is None:
        return x
    if padsize is not None:
        if resize is not None:
            ppix = padsize - resize 
        else:
            ppix = padsize - 256
        
        s = ppix // 2
        e = ppix - s
        H,W = x.shape[-2:]
        x = x[...,s:H-e,s:W-e]
        
    if resize is not None:
        x = interpolate(x,size=(256,256),mode='bilinear')

    return x

def tta(data,i):
    if i == 0:
        x = data
    elif i == 1:
        x = torch.flip(data.clone(),dims=(-1,))
    elif i == 2:
        x = torch.flip(data.clone(),dims=(-2,))
    elif i == 3:
        x = torch.flip(data.clone(),dims=(-2,-1))
    
    return x

