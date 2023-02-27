import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms.functional import rotate

from torchvision.transforms import InterpolationMode



class TorchRandomRotate(nn.Module):
    def __init__(self, degrees, probability=1.0,interpolation=InterpolationMode.BILINEAR, center=None, fill=0,mask_fill=0):
        super().__init__()
        if not isinstance(degrees,(list,tuple)):
            degrees = (-abs(degrees),abs(degrees))

        self.degrees = degrees
        self.interpolation = interpolation
        self.center = center
        self.fill_value = fill
        self.mask_fill_value = mask_fill
        self.proba = probability

    @staticmethod
    def get_params(degrees) -> float:
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            float: angle parameter to be passed to ``rotate`` for random rotation.
        """
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        return angle
    
    @torch.no_grad()
    def __call__(self,img,mask=None):

        batch_size = img.shape[0]

        for i in range(batch_size):
            
            if random.random() > self.proba:
                continue

            angle = self.get_params(self.degrees)
            img[i,...] = rotate(img[i,...], angle, self.interpolation, False, self.center, self.fill_value)
            #mask = mask.long()
            if mask is not None:
                mask[i,...] =  rotate(mask[i,...], angle, self.interpolation, False, self.center, self.mask_fill_value)
            mask = mask.float()
        if mask is not None:
            mask[mask<0] = self.mask_fill_value
            return img,mask
        return 


class RandomMaskIgnore(nn.Module):

    def __init__(self,min_length=50,max_length=10,proba=0.5,ignore_index=-10):
        super().__init__()

        self.min_length = min_length
        self.max_length = max_length
        self.proba = proba
        self.ignore_index = ignore_index
    

    def generate_random_bbox(self,shape):
        H,W = shape
        L = random.randint(self.min_length,self.max_length)

        t = random.randint(0,H-L)
        b = t + L

        l = random.randint(0,W-L)
        r = l + L

        return (t,l,b,r)
    
    def mask_channel(self,bbox,channel):
        (t,l,b,r) = bbox
        channel[:,t:b,l:r] = self.ignore_index
        return channel
    
    @torch.no_grad()
    def __call__(self,mask):

        B,C,H,W = mask.shape
        for i in range(B):
            if random.random() > self.proba:
                continue
            bbox = self.generate_random_bbox((H,W))
            mask[i,...] = self.mask_channel(bbox,mask[i,...])
        
        return mask

class MaskPixelDrop(nn.Module):

    def __init__(self,neg_drop=50,pos_drop=50,ignore_index=-10):
        super().__init__()

        if not isinstance(neg_drop,tuple):
            neg_drop = (0,neg_drop)
        if not isinstance(pos_drop,tuple):
            pos_drop = (0,pos_drop)
        
        self.neg_drop = neg_drop
        self.pos_drop = pos_drop

        self.ignore_index = ignore_index
    
    @staticmethod
    def get_drop_proba(_range):
        return random.randint(_range[0],_range[1]) / 100
    
    def random_pixel_drop(self,gt,mask,_range):
        Cs,Hs,Ws = mask.nonzero(as_tuple=True)
        proba = self.get_drop_proba(_range)
        max_num = Cs.shape[0]
        drop_count = min(max_num,int(proba * max_num))
        #print(drop_count)
        if drop_count == 0 or max_num == 0:
            return gt

        indexes = random.sample(range(0, max_num), drop_count)
        Cs,Hs,Ws = Cs[indexes].tolist(),Hs[indexes].tolist(),Ws[indexes].tolist()
        gt[Cs,Hs,Ws] = self.ignore_index
        return gt

    @torch.no_grad()
    def __call__(self,mask):
        B,C,H,W = mask.shape
        pos_mask = mask.gt(0)
        neg_mask = mask.eq(0)
        for i in range(B):
            mask[i] = self.random_pixel_drop(mask[i],pos_mask[i],self.pos_drop)
            mask[i] = self.random_pixel_drop(mask[i],neg_mask[i],self.neg_drop)
        return mask