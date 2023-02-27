
def get_config(name):

    if name == 'configs_utae':
        from .configs_utae import cfg
    elif name == 'configs_unet3d':
        from .configs_unet3d import cfg
    elif name == 'configs_unet3d_kd':
        from .configs_unet3d_kd import cfg
    elif name == 'configs_unetplusplus3d':
        from .configs_unetplusplus3d import cfg
    elif name == 'configs_fully3dunet':
        from .configs_fully3dunet import cfg
    elif name == 'configs_unetlstm':
        from .configs_unetlstm import cfg
    elif name == 'configs_tsvit':
        from .configs_tsvit import cfg
    
    else:
        raise ValueError(f'configs is not a Valid config file!!!')
    
    return cfg