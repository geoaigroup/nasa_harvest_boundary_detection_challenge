
def get_config(name):

    if name == 'configs_utae':
        from .configs_utae import cfg
    elif name == 'configs_unet3d':
        from .configs_unet3d import cfg
    elif name == 'configs_unetplusplus3d':
        from .configs_unetplusplus3d import cfg
    
    else:
        raise ValueError(f'configs is not a Valid config file!!!')
    
    return cfg