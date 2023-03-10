import albumentations as A

from cv2 import BORDER_CONSTANT,INTER_CUBIC,INTER_LINEAR
from math import ceil



epochs = 200
learning_rate = 0.00001
fold = 1

version = 1
batch_size = 4

RESIZE = None
PAD = False
PADSIZE = None

model_name = 'refinenetv0'

if PAD and PADSIZE is None:
    if RESIZE is None:
        PADSIZE = 256 + 32
    else:
        RESIZE = max(256,int(ceil(RESIZE/32)*32))
        PADSIZE = RESIZE + 32

test_transform = A.Compose(
    [   
        A.Resize(height=RESIZE,width=RESIZE,interpolation=INTER_CUBIC,p=1.0) if RESIZE is not None else A.NoOp(),
        A.PadIfNeeded(min_height=PADSIZE,min_width=PADSIZE,border_mode=BORDER_CONSTANT,value=[-1]*4,mask_value=-100,p=1)\
             if PAD else A.NoOp(),
    ],
    p=1.0
    )

transform = A.Compose(
    [
        #A.Resize(height=RESIZE,width=RESIZE,interpolation=INTER_CUBIC,p=1.0) if RESIZE is not None else A.NoOp(),

        #Flips and rotations
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        test_transform,
    ],
    p=1.0)

cfg = {
    
    'train_tfm': A.to_dict(transform) if transform is not None else None,
    'test_tfm' : A.to_dict(test_transform) if test_transform is not None else None,
    
    'resize' : RESIZE,
    'pad' : PAD,
    'padsize' : PADSIZE,

    'random_mask_ignore' : {
        'apply' : True,
        'proba' : 0.8,
        'min_width' : 80,
        'max_width' : 150
    },
    
    'model_name' : model_name,
    
    'model' : dict(
        in_channels = 6,
        out_channels = 1,
        inter_channels = [16],
        aspp = False,
        aspp_dilations = [2,4,8],
        aspp_channels = 16,
        activation='relu',
        norm_type= 'group',
        group_norm_channels=4,
        aspp_dropout = 0.5,
    ),

    'optimizer':
        {
            "name": "Adam",
            "kwargs": {
                "lr": learning_rate,
                "betas": [
                0.9,
                0.99
                ],
                "eps": 1e-08,
                "weight_decay": 0.0001,
                "amsgrad": False
            }
        },
    'scheduler':
        {
            'name' : 'polylrwwp',
            'kwargs':{
                'warmup' : 0,
                'epochs' : epochs,
                'ratio' : 0.9,
            }
        },

    'loss' : dict(
        pos_weight=1.0
        )
    ,

    'dataset': dict(

         root = '/home/hasann/Desktop/geogroup/competition_datasets/nasa_rwanda_field_boundary_competition/refinement_data_v0',
        folds_path = '/home/hasann/Desktop/geogroup/competitions/nasa_rwanda_field_boundary_competition/folds.csv',
        fold = 0,
        #train = True,
        #aug_tfm=None,
        max_distance = 8,
        distance_mode = 'gaussian'
    )

        ,
    'training':
        {   'activation' : 'sigmoid',
            'epochs' : epochs,
            'lr' : learning_rate,
            'batch_size' : batch_size,
            'val_batch_size' : 1,
            'accumulation_steps':1,
            'val_freq' : 5,
            'device' : 'cuda',
            'amp' : False,
            'max_grad_norm' : 1.0,
            'use_mixup':True,
            'double_mixup' : True
        },


    'save_dir' : f'./results_refinement/nasa_rfb_{model_name}_{epochs}epochs_fold{fold}_V{version}',



}

"""
A.OneOf([
            A.Affine(
                scale=1.2, translate_percent=(0,0.02), translate_px=None, 
                rotate=0.0, shear=(0,20.0),cval=[-1]*4, cval_mask=0, mode=BORDER_CONSTANT, fit_output=False, p=1.0),
            A.Affine(
                scale=1.1, translate_percent=(0,0.05), translate_px=None, 
                rotate=0.0, shear=(0,10.0),cval=[-1]*4, cval_mask=0, mode=BORDER_CONSTANT, fit_output=False, p=1.0),
            A.Affine(
                scale=(0.9,1.05), translate_percent=(0,0.08), translate_px=None, 
                rotate=0.0, shear=(0,10.0),cval=[-1]*4, cval_mask=0, mode=BORDER_CONSTANT, fit_output=False, p=1.0),
            A.ElasticTransform(
            alpha=1, sigma=10, alpha_affine=20, interpolation=1, 
            border_mode=BORDER_CONSTANT, value=[-1]*4, mask_value=0, 
            always_apply=False, approximate=False, same_dxdy=False, p=1.0)
        ],p=0.4)
"""