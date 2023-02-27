import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, MultiStepLR,LambdaLR
# polylr

class PolyLR_WWP(LambdaLR):
    def __init__(self, optimizer, epochs, warmup,ratio=0.9):
        warmup = min(epochs,max(0,warmup-1))
        decay_epochs = epochs - warmup
        xlambda = lambda x : 1.0 if(x<warmup) else (1 - ((x - warmup) / decay_epochs ) ** ratio)
        super().__init__(optimizer, xlambda)

class PolyLR(PolyLR_WWP):
    def __init__(self, optimizer, epochs, ratio=0.9):
        super().__init__(optimizer, epochs,0,ratio)


# schedulers
def get_scheduler(name, optimizer, **kwargs):
    name = name.lower()
    if (name == 'polylr'):
        return PolyLR(optimizer=optimizer, **kwargs)  # kwargs : (epochs , ratio)
    elif (name == 'polylrwwp'):
        return PolyLR_WWP(optimizer=optimizer, **kwargs)  # kwargs : (epochs , ratio)
    elif (name == 'multisteplr'):
        return MultiStepLR(optimizer=optimizer, **kwargs)
    elif (name == 'cosine-anneal'):
        return CosineAnnealingLR(optimizer=optimizer, last_epoch=-1, **kwargs)  # kwargs : (T_max, eta_min)
    elif (name == 'cosine-anneal-wm'):
        return CosineAnnealingWarmRestarts(optimizer=optimizer, last_epoch=-1,
                                           **kwargs)  # kwargs: (T_0, T_mult, eta_min)
    else:
        raise ValueError('optimizer not found')