import albumentations as A

from cv2 import BORDER_CONSTANT,INTER_CUBIC,INTER_LINEAR
from math import ceil
MONTHS = ['03','04','08','10','11','12']

epochs = 150
learning_rate = 0.001
fold = 0

version = 4

batch_size = 2

RESIZE = None
PAD = True
PADSIZE = 320
if PAD and PADSIZE is None:
    if RESIZE is None:
        PADSIZE = 256 + 32
    else:
        RESIZE = max(256,int(ceil(RESIZE/32)*32))
        PADSIZE = RESIZE + 32


IN_CHANNELS = 4

additional_targets={f'image_{m}':'image' for m in MONTHS}

test_transform = A.Compose(
    [   
        A.Resize(height=RESIZE,width=RESIZE,interpolation=INTER_CUBIC,p=1.0) if RESIZE is not None else A.NoOp(),
        A.PadIfNeeded(min_height=PADSIZE,min_width=PADSIZE,border_mode=BORDER_CONSTANT,value=[-1]*4,mask_value=0.0,p=1)\
             if PAD else A.NoOp(),
        
    ],
    p=1.0,
    additional_targets=additional_targets
)
transform = A.Compose(
    [
        A.Resize(height=RESIZE,width=RESIZE,interpolation=INTER_CUBIC,p=1.0) if RESIZE is not None else A.NoOp(),

        #Flips and rotations
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        #Color AUgs
        #A.RandomBrightnessContrast(contrast_limit=(-0.05,0.05),brightness_limit=(-0.02,0.02),brightness_by_max=True,p=0.3),

        test_transform,

        A.ShiftScaleRotate(
            shift_limit=0.0,scale_limit=0.0,rotate_limit=(-20,20),
            border_mode=BORDER_CONSTANT,value=[-1]*4,mask_value=0.0,p=0.5),
        
        
    ],
    p=1.0,
    additional_targets=additional_targets
)


model_name = 'FUNET3D'
cfg = {
    
    'train_tfm': A.to_dict(transform) if transform is not None else None,
    'test_tfm' : A.to_dict(test_transform) if test_transform is not None else None,
    
    'resize' : RESIZE,
    'pad' : PAD,
    'padsize' : PADSIZE,
    
    'model_name' : model_name,

    'model' : dict(
        tsteps = len(MONTHS),
        in_channels=IN_CHANNELS, 
        out_3d_channels=4, 
        classes = 1,
        f_maps=32, 
        layer_order='cgr',
        num_groups=4, 
        num_levels=5, 
        conv_kernel_size=(3,3,3),
        pool_kernel_size=(1,2,2),
        conv_padding=(1,1,1)
    ),

    'optimizer':
        {
            "name": "Adam",
            "kwargs": {
                "lr": 0.001,
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
                'warmup' : 40,
                'epochs' : epochs,
                'ratio' : 0.9,
            }
        },

    'loss' : dict(
        alpha = 1.0,
        beta=1.0,
        gamma=0.0,
        pos_weight=1.0,

    )
    ,

    'dataset': dict(

        root = '/home/hasann/Desktop/geogroup/competition_datasets/nasa_rwanda_field_boundary_competition',
        folds_path = '/home/hasann/Desktop/geogroup/competitions/nasa_rwanda_field_boundary_competition/folds.csv',
        fold = fold,
        months = MONTHS,
        #aug_tfm = A.to_dict(transform) if transform is not None else None,
        resize=None,
        smooth_mask = False
        #train = True
    )
        ,
    'training':
        {   'activation' : 'sigmoid',
            'epochs' : epochs,
            'lr' : learning_rate,
            'batch_size' : batch_size,
            'val_batch_size' : 1,
            'accumulation_steps':1,
            'val_freq' : 10,
            'device' : 'cuda',
            'amp' : False,
            'max_grad_norm' : 1.0,
            'use_mixup':True,
            'double_mixup' : False

        },


    'save_dir' : f'./results_Funet3d_new/nasa_rfb_{model_name}_{epochs}epochs_fold{fold}_V{version}',



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