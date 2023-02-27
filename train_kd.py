import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler,autocast

from dataset import RwandaFieldsTrainSet

from utils import (
    get_scheduler,get_optimizer,
    DiceScore,AverageMeter,
    DiceLoss,SoftBCEWithLogitsLoss,MaskedBCELoss,TverskyLoss,FocalLoss,
    Mixup)
from utils.test_utils import load_trained_model

from configs import get_config


import time
import datetime
import random

import json
import os

from tqdm import tqdm
import numpy as np


def center_hole_mask(crop,length):
    mask = np.ones((length,length))

    s = length // 2 - crop // 2
    e = s + crop

    mask[s:e,s:e] = 0.0

    return mask

def get_model(model_name):
    print(f'Model Architecture : {model_name}')
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

    elif model_name == 'UNETLSTM':
        from models import UnetLstm
        return UnetLstm
    
    elif model_name == 'TSVIT':
        from models import TSViT
        return TSViT
    
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

    global max_grad_norm, accum_steps, device, alpha, beta, gamma,\
        POS_ENCOD,use_mixup,double_mixup,use_chole,CW,RR,RM,PD,bce_per_img

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
        mixup = Mixup(0.5,ignore_index=-100) 
        mixup.to(device)

    #GLOBAL LOSS PARAMS
    loss_cfg = cfg['loss']
    bce_per_img = loss_cfg['bce_per_img']
    alpha,beta,gamma = [loss_cfg[w] for w in ['alpha','beta','gamma']]

    use_chole = loss_cfg['use_chole'] > 0
    
    if use_chole:
        CW = loss_cfg['chole_weight']
        global center_hole
        center_hole = center_hole_mask(loss_cfg['use_chole'],length=cfg['padsize'])
        center_hole = torch.from_numpy(center_hole).float().to(device).unsqueeze(0).unsqueeze(0)
  

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

    if 'use_aspp' not in cfg['model']:
        drop_last=False
    else:
        drop_last=cfg['model']['use_aspp']

    RR = cfg['random_rotate']['apply']

    if RR:
        from utils.transforms import TorchRandomRotate
        global rot_after_mixup 
        rot_after_mixup = TorchRandomRotate(
            degrees=cfg['random_rotate']['angle'],
            probability=cfg['random_rotate']['proba'],
            fill=-1,
            mask_fill=-100
            )
        rot_after_mixup.to(device=device)
        print(rot_after_mixup)
    
    RM = cfg['random_mask_ignore']['apply']
    if RM:
        from utils.transforms import RandomMaskIgnore
        global random_mask_ignore
        random_mask_ignore = RandomMaskIgnore(
            ignore_index=-100,
            min_length= cfg['random_mask_ignore']['min_width'],
            max_length= cfg['random_mask_ignore']['min_width'],
            proba= cfg['random_mask_ignore']['proba']
            )
        random_mask_ignore.to(device=device)
        print(random_mask_ignore)

    PD = cfg['mask_pixel_drop']['apply']
    if PD:
        from utils import MaskPixelDrop
        global mask_pixel_drop 
        mask_pixel_drop = MaskPixelDrop(
            neg_drop=cfg['mask_pixel_drop']['neg_drop'],
            pos_drop = cfg['mask_pixel_drop']['pos_drop'],
            ignore_index=-100
        )
        mask_pixel_drop.to(device=device)
        print(mask_pixel_drop)



    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=cfg['training']['batch_size'],
                                  shuffle=True,
                                  sampler=None,
                                  batch_sampler=None,
                                  num_workers=4,
                                  pin_memory=True,
                                  drop_last=drop_last
                                  
                                  )

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

    global dice_name,bce_name

    if loss_cfg['dice_fn_w'] == 0.5:
        dice_criterion = DiceLoss(mode='binary',from_logits=True,smooth=0.0,eps=1e-7,ignore_index=-100,log_loss=True)
        dice_name = 'dice'
    else:
        fn_w = loss_cfg['dice_fn_w']
        fp_w = 1.0 - fn_w
        dice_name = f'tverksy_{fn_w}'
        dice_criterion = TverskyLoss(mode='binary',from_logits=True,smooth=0.0,eps=1e-7,alpha=fp_w,beta=fn_w,log_loss=True,ignore_index=-100)

    bce_reduction = 'none' if bce_per_img else 'mean'
    if loss_cfg['use_focal']:
        bce_name = 'focal'
        bce_criterion = FocalLoss(alpha=0.25,gamma=2.0,mode='binary',ignore_index=-100,reduction=bce_reduction)
    else:
        bce_name = 'bce'
        bce_criterion = SoftBCEWithLogitsLoss(pos_weight=torch.tensor([loss_cfg['pos_weight']]),ignore_index=-100,reduction=bce_reduction)

    if gamma > 0:
        cb_criterion = MaskedBCELoss(device=device)
    else:
        cb_criterion = None

    global kl_criterion,Temp,teachers

    kl_criterion = nn.L1Loss()#nn.KLDivLoss(log_target=True)
    kd_cfg = cfg['kd']
    Temp = kd_cfg['temperature']
    teachers = []
    for path in kd_cfg['teachers']:
        teacher,_ = load_trained_model(path,use_last_model=False)
        teacher.to(device)
        teachers.append(teacher)
    #print(teachers)

    kl_criterion.to(device)
    model.to(device)
    dice_criterion.to(device)
    bce_criterion.to(device)

    
    return model,dice_criterion,bce_criterion,cb_criterion,optimizer,scheduler,train_dataloader,val_dataloader,scaler



def extract_batch(batch,device='cuda'):
    x = batch['x'].to(device)
    y_gt = batch['y_gt'].to(device)
    cb_mask = batch['cb_mask'].to(device)
    dates = batch['dates'].to(device)

    return x,y_gt,cb_mask,dates

def train_epoch(model,dice_criterion,bce_criterion,cb_criterion,optimizer,train_dataloader,epoch = 0,scaler=None,amp=True):
    model.train()
    model.zero_grad()

    loader = tqdm(train_dataloader)
    
    dice_loss_meter = AverageMeter()
    bce_loss_meter = AverageMeter()
    kl_loss_meter = AverageMeter()

    close_error = gamma > 0
    
    if close_error:
        cb_loss_meter = AverageMeter()
    
    if use_chole:
        ch_loss_meter = AverageMeter()

    dice_scorer_30 = DiceScore(threshold=0.3,ignore_index=-100)
    dice_scorer_50 = DiceScore(threshold=0.5,ignore_index=-100)
    dice_scorer_80 = DiceScore(threshold=0.8,ignore_index=-100)


    for batch_idx,batch in enumerate(loader):
        
        x,y_gt,cb_mask,dates = extract_batch(batch)
        #print(dates)
        #print(x.shape,y_gt.shape,cb_mask.shape)
        
        if use_mixup:
            x,y_gt,cb_mask = mixup(x,y_gt,cb_mask)
            if double_mixup:
                x,y_gt,cb_mask = mixup(x,y_gt,cb_mask)
        #print(y_gt.min())
        if RR:
            x,y_gt = rot_after_mixup(x,y_gt)
        if RM:
            y_gt = random_mask_ignore(y_gt)
        if PD:
            y_gt = mask_pixel_drop(y_gt)
            #print(y_gt.min())
        with autocast(enabled=amp):
            if POS_ENCOD:
                logits = model(x,dates)
            else:
                logits = model(x)

        dice_loss = dice_criterion(logits,y_gt)
        bce_loss = bce_criterion(logits,y_gt)
        if bce_per_img:
            bce_loss = bce_loss.mean(dim=(-1,-2)).mean()
            #print(bce_loss.shape)
        
        loss = alpha * dice_loss + beta * bce_loss
        
        if use_chole:
            chole_dice_loss = dice_criterion(logits,y_gt*center_hole - 100 * (1 - center_hole))
            loss = loss + CW * chole_dice_loss
        
        if close_error:
            cb_loss = cb_criterion(logits,y_gt,cb_mask)
            loss = loss + gamma * cb_loss

        kl_loss = 0
        for teacher in teachers:
            with torch.no_grad():
                teacher_logits = teacher(x)
                #teacher_logits = torch.sigmoid(teacher_logits/Temp)
                #teacher_logits = (teacher_logits > 0.5).type(teacher_logits.dtype)
            
            kl_loss = kl_loss + kl_criterion(torch.sigmoid(logits/Temp),torch.sigmoid(teacher_logits/Temp))
            #

        loss = loss + kl_loss

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



        with torch.no_grad():
            y_pred = torch.sigmoid(logits)


        ###Update Loss Meters#####
        dice_loss_meter.update(dice_loss.item())
        bce_loss_meter.update(bce_loss.item())
        kl_loss_meter.update(kl_loss.item())

        if close_error:
            cb_loss_meter.update(cb_loss.item())
        if use_chole:
            ch_loss_meter.update(chole_dice_loss.item())
        

        dice_loss = dice_loss_meter.get_update()
        bce_loss = bce_loss_meter.get_update()
        kl_loss = kl_loss_meter.get_update()
        
        if close_error:
            cb_loss = cb_loss_meter.get_update()
        if use_chole:
            ch_loss = ch_loss_meter.get_update()

        ###calculate Dice Scores#####
        f1_30 = dice_scorer_30(y_pred,y_gt,thresh_gt=True)
        f1_50 = dice_scorer_50(y_pred,y_gt,thresh_gt=True)
        f1_80 = dice_scorer_80(y_pred,y_gt,thresh_gt=True)

        
        s = f'TrainEpoch[{epoch}]: losses : {dice_name} = {dice_loss:.4}| {bce_name} = {bce_loss:.4}| kl = {kl_loss:.4}'
        
        if close_error:
            s += f' close_bce = {cb_loss:.4}|'
        if use_chole:
            s += f' chole_dice = {ch_loss:.4}|'

        s += f'-- F1scores = ({f1_30:.4} | {f1_50:.4} | {f1_80:.4})'
        #break
        loader.set_description_str(s)
        #break
    return s

def val_epoch(model,dice_criterion,bce_criterion,cb_criterion,val_dataloader,epoch = 0):
    model.eval()
    model.zero_grad()

    loader = tqdm(val_dataloader)
    
    dice_loss_meter = AverageMeter()
    bce_loss_meter = AverageMeter()
    
    close_error = gamma > 0
    
    if close_error:
        cb_loss_meter = AverageMeter()
    
    dice_scorer_30 = DiceScore(threshold=0.3,ignore_index=-100)
    dice_scorer_50 = DiceScore(threshold=0.5,ignore_index=-100)
    dice_scorer_80 = DiceScore(threshold=0.8,ignore_index=-100)


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
        bce_loss = bce_criterion(logits,y_gt)
        if bce_per_img:
            bce_loss = bce_loss.mean(dim=(-1,-2)).mean()
        loss = alpha * dice_loss + beta * bce_loss

        if close_error:
            cb_loss = cb_criterion(logits,y_gt,cb_mask)
            loss = loss + gamma * cb_loss
        ###Update Loss Meters#####
        dice_loss_meter.update(dice_loss.item())
        bce_loss_meter.update(bce_loss.item())
        if close_error:
            cb_loss_meter.update(cb_loss.item())
        

        dice_loss = dice_loss_meter.get_update()
        bce_loss = bce_loss_meter.get_update()
        if close_error:
            cb_loss = cb_loss_meter.get_update()

        ###calculate Dice Scores#####
        f1_30 = dice_scorer_30(y_pred,y_gt)
        f1_50 = dice_scorer_50(y_pred,y_gt)
        f1_80 = dice_scorer_80(y_pred,y_gt)

        if close_error:
            s = f'ValEpoch[{epoch}]: losses : {dice_name} = {dice_loss:.4}| {bce_name} = {bce_loss:.4}| close_bce = {cb_loss:.4} -- F1scores = ({f1_30:.4} | {f1_50:.4} | {f1_80:.4})'

        else:
            s = f'ValEpoch[{epoch}]: losses : {dice_name} = {dice_loss:.4}| {bce_name} = {bce_loss:.4} -- F1scores = ({f1_30:.4} | {f1_50:.4} | {f1_80:.4})'
        #break
        loader.set_description_str(s)
        #break


    scores = {
            'f1_30' : f1_30,'f1_50':f1_50,'f1_80':f1_80,
            f'{dice_name}_loss' : dice_loss,f'{bce_name}_loss' : bce_loss,
    }
    if close_error:
        scores['cb_loss'] = cb_loss
    return s,scores

def train_model(cfg,save_last=True):

    ###some configs###
    save_dir = cfg['save_dir']
    num_epochs = cfg['training']['epochs']
    device = cfg['training']['device']
    amp = cfg['training']['amp']
    val_freq = cfg['training']['val_freq']
    
    make_dir(save_dir)
    save_configs(save_dir,cfg)
    
    ###___________###
    model,dice_criterion,bce_criterion,cb_criterion,optimizer,scheduler,train_dataloader,val_dataloader,scaler = initialize_params(cfg)

    best_score = -9999.99
    logs = ''
    for epoch in range(num_epochs):
        lr = optimizer.param_groups[0]['lr']
        s_train = train_epoch(
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        bce_criterion=bce_criterion,
                        dice_criterion=dice_criterion,
                        cb_criterion=cb_criterion,
                        train_dataloader=train_dataloader,
                        scaler=scaler,
                        amp=amp,
                        )
        logs+= '>>>' + s_train + '\n'
        if (epoch+1) % val_freq  == 0 or (epoch+1) == num_epochs:
            s_val,scores = val_epoch(
                                epoch= epoch,
                                model=model,
                                bce_criterion=bce_criterion,
                                dice_criterion=dice_criterion,
                                cb_criterion=cb_criterion,
                                val_dataloader=val_dataloader, 
                                )

            score = scores['f1_50']
            logs+= '>>>' + s_val + '\n'
            

            if score >= best_score:
                s_save = f'Current DiceScore [{score:.5}] is better than previous best DiceScore [{best_score:.5}] ---> Saving Model!!! \n' 
                best_score=score
                print(s_save)
                logs+= '>>>' + s_save + '\n'
                save_model(save_dir,model,epoch,score,lr)
        
        if save_last:
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
    parser.add_argument('--config_file',type=str,default='configs_unet3d_kd')
    args = parser.parse_args()

    config_file = args.config_file
    cfg = get_config(config_file)

    set_seed(911)
    start = time.time()
    train_model(cfg)
    end = time.time()
    t = str(datetime.timedelta(seconds=round(end-start)))
    print(f'Training done in {t} hours !!!')