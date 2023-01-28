import albumentations as A

from cv2 import BORDER_CONSTANT,INTER_CUBIC,INTER_LINEAR
from math import ceil
MONTHS = ['03','04','08','10','11','12']

epochs = 150
learning_rate = 0.001
fold = 0

version = 0

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

        #A.GaussianBlur(blur_limit=(3,5),p=0.1),
        #A.GaussNoise(var_limit=0.0002, mean=0, per_channel=True, always_apply=False, p=0.15)
        
    ],
    p=1.0,
    additional_targets=additional_targets
)
#transform = None

#UTAE
#backbone = 'tu-tf_efficientnet_b1'
#backbone = 'resnet34'
#backbone = 'resnet18'
model_name = 'UTAE'
cfg = {
    
    'train_tfm': A.to_dict(transform) if transform is not None else None,
    'test_tfm' : A.to_dict(test_transform) if test_transform is not None else None,
    
    'resize' : RESIZE,
    'pad' : PAD,
    'padsize' : PADSIZE,
    
    'model_name' : model_name,

    'model' : dict(
        input_dim = IN_CHANNELS,
        encoder_widths= [16,32,64,128,256],#[64,64,64,128],
        decoder_widths= [16,32,64,128,256],
        out_conv=[16, 1],
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        agg_mode="att_group",
        encoder_norm="group",
        n_head=16,
        d_model=256,
        d_k=4,
        encoder=False,
        return_maps=False,
        pad_value=0,
        padding_mode='zeros',
        positional_encoding = True
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
                'warmup' : 20,
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
        smooth_mask = False,
        #aug_tfm = A.to_dict(transform) if transform is not None else None,
        resize=None
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


    'save_dir' : f'./results_new/nasa_rfb_{model_name}_{epochs}epochs_fold{fold}_V{version}',



}

