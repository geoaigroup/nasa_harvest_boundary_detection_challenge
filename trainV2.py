import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler,autocast

from dataset import RwandaFieldsTrainSet

from utils import (
    get_scheduler,get_optimizer,
    DiceScore,AverageMeter,
    DiceLoss,SoftBCEWithLogitsLoss,MaskedBCELoss,
    KeypointFocalLoss,SmoothKeypointFocalLoss,
    Mixup)

from configs import get_config


import time
import datetime
import random

import json
import os

from tqdm import tqdm
import numpy as np



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
    
    else:
        raise ValueError(f'model {model_name} is not implemented!!!')


def save_model(path,model,epoch,score,lr):
    if not path.endswith('.pth'):
        path += '/best_model.pth'
    torch.save(
        {
            'state_dict' : model.state_dict(),
            'epoch' : epoch,
            'best_score' : score,
            'learning_rate' : lr
        },
        path)


def save_configs(path, configs):
    f = open(os.path.join(path, 'configs.json'), "w+")
    json.dump(configs, f, indent=3)
    f.close()

def save_logs(path,logs):
    with open(path+'/logs.txt','w+') as f:
        f.write(logs)

def make_dir(path):
    try:
        os.makedirs(path)
    except:
        pass

def initialize_params(cfg):

    global max_grad_norm, accum_steps, device, alpha, beta, gamma,POS_ENCOD,use_mixup,double_mixup

    #GLOBAL PARAMS
    if 'positional_encoding' in cfg['model'].keys():
        POS_ENCOD = cfg['model']['positional_encoding']
    else:
        POS_ENCOD = False

    max_grad_norm= cfg['training']['max_grad_norm']
    accum_steps = cfg['training']['accumulation_steps']
    device = cfg['training']['device']
    
    use_mixup = cfg['training']['use_mixup']
    double_mixup = cfg['training']['double_mixup']
    if use_mixup:
        global mixup
        mixup = Mixup(0.5) 
        mixup.to(device)

    #GLOBAL LOSS PARAMS
    loss_cfg = cfg['loss']

    alpha,beta,gamma = [loss_cfg[w] for w in ['alpha','beta','gamma']]

    train_dataset = RwandaFieldsTrainSet(
        train=True,
        aug_tfm=cfg['train_tfm'],
        **cfg['dataset']
    )

    val_dataset = RwandaFieldsTrainSet(
        train=False,
        aug_tfm=cfg['test_tfm'],
        **cfg['dataset']
    )


    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=cfg['training']['batch_size'],
                                  shuffle=True,
                                  sampler=None,
                                  batch_sampler=None,
                                  num_workers=4,
                                  pin_memory=True)

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=cfg['training']['val_batch_size'],
                                shuffle=True,
                                sampler=None,
                                batch_sampler=None,
                                num_workers=4,
                                pin_memory=True)

    scaler = GradScaler() if cfg['training']['amp'] else None



    model = get_model(cfg['model_name'])(**cfg['model'])
    #print(model)
    
    optimizer = get_optimizer(
        params=model.parameters(),
        name=cfg['optimizer']['name'],
        **cfg['optimizer']['kwargs'])

    scheduler = get_scheduler(
        optimizer=optimizer,
        name=cfg['scheduler']['name'],
        **cfg['scheduler']['kwargs'])

    focal_criterion = SmoothKeypointFocalLoss(eps=0.2)
    dice_criterion = DiceLoss(mode='binary',from_logits=True,smooth=0.0,eps=1e-7)
    #bce_criterion = SoftBCEWithLogitsLoss(pos_weight=torch.tensor([loss_cfg['pos_weight']]))

    #if gamma > 0:
    #    cb_criterion = MaskedBCELoss(device=device)
    #else:
    #    cb_criterion = None

    model.to(device)
    #dice_criterion.to(device)
    #bce_criterion.to(device)
    focal_criterion.to(device)

    
    return model,dice_criterion,focal_criterion,optimizer,scheduler,train_dataloader,val_dataloader,scaler



def extract_batch(batch,device='cuda'):
    x = batch['x'].to(device)
    y_gt = batch['y_gt'].to(device)
    cb_mask = batch['cb_mask'].to(device)
    dates = batch['dates'].to(device)

    return x,y_gt,cb_mask,dates

def train_epoch(model,dice_criterion,focal_criterion,optimizer,train_dataloader,epoch = 0,scaler=None,amp=True):
    model.train()
    model.zero_grad()

    loader = tqdm(train_dataloader)

    dice_loss_meter = AverageMeter()
    focal_loss_meter = AverageMeter()


    dice_scorer_30 = DiceScore(threshold=0.3)
    dice_scorer_50 = DiceScore(threshold=0.5)
    dice_scorer_80 = DiceScore(threshold=0.8)


    for batch_idx,batch in enumerate(loader):
        
        x,y_gt,cb_mask,dates = extract_batch(batch)
        #print(x.shape,y_gt.shape,cb_mask.shape)
        if use_mixup:
            x,y_gt,cb_mask = mixup(x,y_gt,cb_mask)
            if double_mixup:
                x,y_gt,cb_mask = mixup(x,y_gt,cb_mask)


        with autocast(enabled=amp):
            if POS_ENCOD:
                logits = model(x,dates)
            else:
                logits = model(x)
    
        #dice_loss = dice_criterion(logits,y_gt)
        #bce_loss = bce_criterion(logits,y_gt)
        
        y_pred = torch.sigmoid(logits)

        dice_loss = dice_criterion(logits,y_gt)
        focal_loss = focal_criterion(y_pred,y_gt)

        loss = alpha * dice_loss + beta * focal_loss

        loss = loss / accum_steps

        if amp:
            scaler.scale(loss).backward()
       
        else:
            loss.backward()
            
        if ((batch_idx + 1) % accum_steps == 0) or (batch_idx + 1 == len(loader)):
            if amp:
                scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) 
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) 
                optimizer.step()
            
            optimizer.zero_grad()



        y_pred = y_pred.detach()
            


        ###Update Loss Meters#####
        dice_loss_meter.update(dice_loss.item())
        focal_loss_meter.update(focal_loss.item())

        

        focal_loss = focal_loss_meter.get_update()
        dice_loss = dice_loss_meter.get_update()

        ###calculate Dice Scores#####
        f1_30 = dice_scorer_30(y_pred,y_gt,thresh_gt=True)
        f1_50 = dice_scorer_50(y_pred,y_gt,thresh_gt=True)
        f1_80 = dice_scorer_80(y_pred,y_gt,thresh_gt=True)

        
        s = f'TrainEpoch[{epoch}]: losses : dice = {dice_loss} -- focal = {focal_loss:.4} -- F1scores = ({f1_30:.4} | {f1_50:.4} | {f1_80:.4})'
        #break
        loader.set_description_str(s)
        #break
    return s

def val_epoch(model,dice_criterion,focal_criterion,val_dataloader,epoch = 0):
    model.eval()
    model.zero_grad()

    loader = tqdm(val_dataloader)
    
    focal_loss_meter = AverageMeter()
    dice_loss_meter = AverageMeter()

    dice_scorer_30 = DiceScore(threshold=0.3)
    dice_scorer_50 = DiceScore(threshold=0.5)
    dice_scorer_80 = DiceScore(threshold=0.8)


    for batch_idx,batch in enumerate(loader):
        
        x,y_gt,cb_mask,dates = extract_batch(batch)
        #print(x.shape,y_gt.shape,cb_mask.shape)
        with torch.no_grad():
            if POS_ENCOD:
                logits = model(x,dates)
            else:
                logits = model(x)
            y_pred = torch.sigmoid(logits)

        dice_loss = dice_criterion(logits,y_gt)
        focal_loss = focal_criterion(y_pred,y_gt)

        loss = alpha * dice_loss + beta * focal_loss
        

        ###Update Loss Meters#####
        dice_loss_meter.update(dice_loss.item())
        focal_loss_meter.update(focal_loss.item())

        

        focal_loss = focal_loss_meter.get_update()
        dice_loss = dice_loss_meter.get_update()

        ###calculate Dice Scores#####
        f1_30 = dice_scorer_30(y_pred,y_gt)
        f1_50 = dice_scorer_50(y_pred,y_gt)
        f1_80 = dice_scorer_80(y_pred,y_gt)


        s = f'ValEpoch[{epoch}]: losses : dice = {dice_loss} -- focal = {focal_loss:.4} -- F1scores = ({f1_30:.4} | {f1_50:.4} | {f1_80:.4})'
        #break
        loader.set_description_str(s)
        #break


    scores = {
            'f1_30' : f1_30,'f1_50':f1_50,'f1_80':f1_80,
            'focal_loss' : focal_loss,
    }

    return s,scores

def train_model(cfg):

    ###some configs###
    save_dir = cfg['save_dir']
    num_epochs = cfg['training']['epochs']
    device = cfg['training']['device']
    amp = cfg['training']['amp']
    val_freq = cfg['training']['val_freq']
    
    make_dir(save_dir)
    save_configs(save_dir,cfg)
    
    ###___________###
    model,dice_criterion,focal_criterion,optimizer,scheduler,train_dataloader,val_dataloader,scaler = initialize_params(cfg)

    best_score = -9999.99
    logs = ''
    for epoch in range(num_epochs):
        lr = optimizer.param_groups[0]['lr']
        s_train = train_epoch(
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        dice_criterion=dice_criterion,
                        focal_criterion=focal_criterion,
                        train_dataloader=train_dataloader,
                        scaler=scaler,
                        amp=amp,
                        )
        logs+= '>>>' + s_train + '\n'
        if (epoch+1) % val_freq  == 0 or (epoch+1) == num_epochs:
            s_val,scores = val_epoch(
                                epoch= epoch,
                                model=model,
                                dice_criterion=dice_criterion,
                                focal_criterion=focal_criterion,
                                val_dataloader=val_dataloader, 
                                )

            score = scores['f1_50']
            logs+= '>>>' + s_val
            

            if score >= best_score:
                s_save = f'Current DiceScore [{score:.5}] is better than previous best DiceScore [{best_score:.5}] ---> Saving Model!!! \n'
                best_score=score
                print(s_save)
                logs+= '>>>' + s_save + '\n'
                save_model(save_dir,model,epoch,score,lr)
            
        save_model(save_dir+'/last_model.pth',model,epoch,best_score,lr)


        save_logs(save_dir,logs)

        scheduler.step()


def set_seed(seed=911):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    from configs import get_config
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',type=str,default='configs_unetplusplus3d_focal')
    args = parser.parse_args()

    config_file = args.config_file
    cfg = get_config(config_file)

    set_seed(911)
    start = time.time()
    train_model(cfg)
    end = time.time()
    t = str(datetime.timedelta(seconds=round(end-start)))
    print(f'Training done in {t} hours !!!')