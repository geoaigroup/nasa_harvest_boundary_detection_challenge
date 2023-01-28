import torch
from torch.utils.data import Dataset

import os
import glob
import cv2

import pandas as pd
import rasterio as rio
import numpy as np

import albumentations as A
import matplotlib.pyplot as plt

from scipy.ndimage import binary_dilation
from skimage.morphology import square

Mean = [0.049, 0.083, 0.088, 0.303]
Std = [0.215, 0.276, 0.283, 0.46]
MAX_PIXEL_VALUE = 10000#np.array([ 6200.,  6200.,  9100., 10000.]) #65535
ALL_MONTHS = ['03','04','08','10','11','12']

fpfx = 'nasa_rwanda_field_boundary_competition'


ksize = (4,4)
sigmax =sigmay = (2 * (ksize[0] / 3)** 2)
#print(sigmax)

class RwandaFieldsTrainSet(Dataset):

    def __init__(
        self,
        root = '/home/hasann/Desktop/geogroup/competition_datasets/nasa_rwanda_field_boundary_competition',
        folds_path = '/home/hasann/Desktop/geogroup/competitions/nasa_rwanda_field_boundary_competition/folds.csv',
        fold = 0,
        months = ALL_MONTHS,
        train = True,
        aug_tfm=None,
        resize = None,
        pad = True,
        smooth_mask = False
        ):

        #root = root
        self.img_dir = os.path.join(root,'train_imgs')
        self.mask_dir = os.path.join(root,'train_labels')
        
        df = pd.read_csv(folds_path)

        if train:
            df = df[df['fold'] != fold]
        else:
            df = df[df['fold'] == fold]

        df.reset_index(drop=True,inplace=True)
        #A.Norm

        self.normalize = A.Normalize(mean=Mean,std=Std,max_pixel_value=MAX_PIXEL_VALUE,p=1)
        self.aug_tfm = aug_tfm if aug_tfm is None else A.from_dict(aug_tfm)

        #print(df)
        self.train = train
        self.df = df
        self.months = [m for m in ALL_MONTHS if m in months]
        self.year = '2021'
        self.dates = [int(month) for month in self.months]
        

        self.resize = resize
        self.pad = pad
        self.smooth_mask = smooth_mask
    
    def __len__(self):
        return len(self.df)
    
    
    def pad_img(self,img,value=0.0):
        ###Note this assumes that img has a size divisible by 32
        return cv2.copyMakeBorder(img,top=16,bottom=16,right=16,left=16,borderType=cv2.BORDER_CONSTANT,value=value)

    
    @staticmethod
    def load_channels(path):
        img = []
        for band in ['B01','B02','B03','B04']:
            img.append(rio.open(f'{path}/{band}.tif').read())
        img = np.vstack(img).transpose(1,2,0)
        return img


    
    def __getitem__(self, index,totensor = True):

        row = self.df.iloc[index]
        idx = int(row['idx'])
        idx = f'{idx:02}'

        imgs = [] 
        
        for month in self.months:
            img = self.load_channels(f'{self.img_dir}/{fpfx}_source_train_{idx}_{self.year}_{month}')

            img = self.normalize(image=img)['image']
            
            if self.resize is not None:
                img = cv2.resize(img,self.resize,interpolation=cv2.INTER_CUBIC)

            img = img.transpose(2,0,1)
            imgs.append(img)
        
        mask = rio.open(f'{self.mask_dir}/{fpfx}_labels_train_{idx}/raster_labels.tif').read()
        
        if self.resize is not None:
            mask = cv2.resize(mask[0,:,:],self.resize,interpolation=cv2.INTER_CUBIC)[np.newaxis,...]
        
        if self.aug_tfm is not None:
            
            params = {}
            for i,month in enumerate(self.months):
                img = imgs[i]
                k = 'image' if i == 0 else f'image_{month}'
                params[k] = img.transpose(1,2,0)
            params['mask'] = mask.transpose(1,2,0)
            augs = self.aug_tfm(**params)
            
            imgs = []
            for i,month in enumerate(self.months):
                k = 'image' if i == 0 else f'image_{month}'
                img = augs[k].transpose(2,0,1)
                imgs.append(img)
            
            mask = augs['mask'].transpose(2,0,1)
    

        cb_mask = binary_dilation(mask[0],structure=square(3),iterations=1)
        cb_mask = cb_mask[np.newaxis,...]    

        imgs = np.stack(imgs,axis=0)
        
        if self.train and self.smooth_mask:

            mask = mask[0].astype(np.uint8) *255
            #print(np.unique(mask))
            dist, labels = cv2.distanceTransformWithLabels(~mask, distanceType=cv2.DIST_L2, maskSize=3)
            mask = np.exp(-(dist**2)/sigmax)[np.newaxis,...]
            mask[mask<1e-3]=0.0
        
        if totensor:
            imgs = imgs.astype(np.float32)
            imgs = torch.from_numpy(imgs).float()
            mask = torch.from_numpy(mask).float()
            cb_mask = torch.from_numpy(cb_mask).bool()
            dates = torch.tensor(self.dates).long()
        
        ret = {'x':imgs,'y_gt':mask,'cb_mask':cb_mask,'dates':dates}

        return ret

class RwandaFieldsTestSet(Dataset):

    def __init__(
        self,
        root = '/home/hasann/Desktop/geogroup/competition_datasets/nasa_rwanda_field_boundary_competition',
        months = ALL_MONTHS,
        resize =None,
        aug_tfm=None
        ):
        self.img_dir = os.path.join(root,'test_imgs')
        self.idxs = [os.path.basename(f).split('_2021')[0].split('_')[-1] for f in glob.glob(f'{self.img_dir}/nasa_rwanda_field_boundary_competition_source*')]
        self.idxs = sorted(list(set(self.idxs)))
        self.normalize = A.Normalize(mean=Mean,std=Std,max_pixel_value=MAX_PIXEL_VALUE,p=1)
        #self.aug_tfm = aug_tfm if aug_tfm is None else A.from_dict(aug_tfm)

        self.aug_tfm = aug_tfm if aug_tfm is None else A.from_dict(aug_tfm)

        self.months = [m for m in ALL_MONTHS if m in months]
        self.year = '2021'
        self.dates = [int(month) for month in self.months]
        self.resize = resize

    
    def __len__(self):
        return len(self.idxs)
    
    @staticmethod
    def load_channels(path):
        img = []
        for band in ['B01','B02','B03','B04']:
            img.append(rio.open(f'{path}/{band}.tif').read())
        img = np.vstack(img).transpose(1,2,0)
        return img

    def __getitem__(self, index,totensor = True):

        idx = int(self.idxs[index])
        idx = f'{idx:02}'
        iid = f'{fpfx}_source_test_{idx}'
        
        imgs = [] 
        
        for month in self.months:
            img = self.load_channels(f'{self.img_dir}/{iid}_{self.year}_{month}')
            img = self.normalize(image=img)['image']
            if self.resize is not None:
                img = cv2.resize(img,self.resize,interpolation=cv2.INTER_CUBIC)
            img = img.transpose(2,0,1)
            imgs.append(img)

        if self.aug_tfm is not None:
            
            params = {}
            for i,month in enumerate(self.months):
                img = imgs[i]
                k = 'image' if i == 0 else f'image_{month}'
                params[k] = img.transpose(1,2,0)
            #params['mask'] = mask.transpose(1,2,0)
            augs = self.aug_tfm(**params)
            
            imgs = []
            for i,month in enumerate(self.months):
                k = 'image' if i == 0 else f'image_{month}'
                img = augs[k].transpose(2,0,1)
                imgs.append(img)

        imgs = np.stack(imgs,axis=0)
        
        if totensor:
            imgs = imgs.astype(np.float32)
            imgs = torch.from_numpy(imgs).float()
            dates = torch.tensor(self.dates).long()

        ret = {'x':imgs,'dates':dates,'iid':iid}
        
        return ret

if __name__ == '__main__':

    additional_targets={f'image_{m}':'image' for m in ALL_MONTHS}


    transform = A.Compose(
        [
            A.PadIfNeeded(min_height=448,min_width=448,border_mode=cv2.BORDER_CONSTANT,value=[-1]*4,mask_value=0.0,p=1),
            A.ShiftScaleRotate(
                shift_limit=0.0,scale_limit=0.0,rotate_limit=(0,1),
                border_mode=cv2.BORDER_CONSTANT,value=[-1]*4,mask_value=0.0,p=1.0),
            A.ElasticTransform(
                alpha=1, sigma=10, alpha_affine=20, interpolation=1, 
                border_mode=cv2.BORDER_CONSTANT, value=[-1]*4, mask_value=0, 
                always_apply=False, approximate=False, same_dxdy=False, p=1.0)
        ],
        p=1.0,
        additional_targets=additional_targets
    )

    aug_tfm =A.to_dict(transform)
    dataset = RwandaFieldsTrainSet(aug_tfm=aug_tfm,fold=2,train=True,smooth_mask=False)
    IDX = 2
    ret = dataset.__getitem__(IDX)
    

    imgs,mask,cb_mask =ret['x'],ret['y_gt'],ret['cb_mask']
    fig,axs = plt.subplots(2,4,figsize = (20,10))

    B,C,H,W = imgs.shape
    for i in range(B):
        # print(img.shape)
        img = imgs[i,...].cpu().numpy()        
        img = img.transpose(1,2,0)

        row,col = i // 4, i%4
        
        #print(img.min(axis=(0,1)))
        img = img * 2.0
        img = (img + 1)/ 2
        
        

        rgb = img[:,:,[2,1,0]]
        axs[row,col].imshow(rgb)

    mask = mask.cpu().numpy()[0]
    cb_mask = cb_mask.cpu().numpy()[0]

    axs[1,2].imshow(mask)
    axs[1,3].imshow(cb_mask)
    #print(np.unique(mask))
    plt.show()