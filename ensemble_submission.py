import torch

import os
import gc
import argparse

import numpy as np
import pandas as pd

from dataset import RwandaFieldsTestSet

from utils.test_utils import load_trained_model,resize_pad,unpad_resize,noise_filter,tta

import matplotlib.pyplot as plt
from skimage.measure import label

from tqdm import tqdm

Thresh= 0.5
remove_noise = 20

ignored_models = [
        'nasa_rfb_UNET3D_tu-tf_efficientnetv2_s_300epochs_fold1_V21',
        'nasa_rfb_UNET3D_tu-tf_efficientnetv2_s_300epochs_fold2_V21'
        ]

def inference(root,models,pad_sizes,resize_sizes,device='cuda',hr_idxs=None):

    dataset = RwandaFieldsTestSet(
        root=root,
        months=['03','04','08','10','11','12'],
        resize=None,
        aug_tfm=None,
        include_nir=True)

    data = []
    iids = []

    for i in range(len(dataset)):
        ret = dataset.__getitem__(i,totensor=True)
        imgs,iid =ret['x'],ret['iid']
        iids.append(iid)
        data.append(imgs)

    data_orig = torch.stack(data,dim=0)
    preds = []

    for mid,model in enumerate(tqdm(models)):
        preds_tta = [] 
        data = resize_pad(data_orig,padsize=pad_sizes[mid],resize=resize_sizes[mid],pad_value=-1)        
        model.to(device=device)
        
        for i in range(4):
            x = tta(data,i)
            x = x.to(device=device)

            with torch.no_grad():
                y_pred = model(x)
                y_pred = torch.sigmoid(y_pred) 

            y_pred = tta(y_pred,i)
            y_pred = unpad_resize(y_pred,padsize=pad_sizes[mid],resize=resize_sizes[mid])
    
            x = x.cpu()

            preds_tta.append(y_pred)
        
        preds_tta = torch.cat(preds_tta,dim=1).mean(dim=1).unsqueeze(1)
        preds_tta = preds_tta.cpu()
        
        preds.append(preds_tta)
        model.cpu()
    
    preds = torch.cat(preds,dim=1).to(device=device)

    if hr_idxs is not None:
        preds_hr = preds.clone()[:,hr_idxs,:,:]
        preds_hr = preds_hr.mean(dim=1).cpu().numpy()

    preds = preds.mean(dim=1)
    preds = preds.cpu().numpy()

    if hr_idxs is not None:
        return preds,preds_hr,iids,data_orig

    return preds,iids,data_orig

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--input_dir',type=str,default='./final_models')
    arg('--data_dir',type=str,\
        default='/home/hasann/Desktop/geogroup/competition_datasets/nasa_rwanda_field_boundary_competition')

    args = parser.parse_args()

    models_dir = args.input_dir 
    data_dir = args.data_dir

    ### LOAD Models Weights ####
    model_paths = [os.path.join(models_dir,f) for f in os.listdir(models_dir) if f not in ignored_models]

    models = []
    resize_sizes = []
    pad_sizes = []

    model_id = [model_path.split('/')[-1] for model_path in model_paths]
    for i,model_path in enumerate(model_paths):
        model,cfg = load_trained_model(model_path,use_last_model=False)
        models.append(model)
        pad_sizes.append(cfg['padsize'] if cfg['padsize'] is not None else 320)
        resize_sizes.append(cfg['resize'] if cfg['resize'] is not None else 256)

    #### Inference ######
    preds,iids,_ = inference(
        root = data_dir,
        models=models,
        pad_sizes=pad_sizes,
        resize_sizes=resize_sizes,
        hr_idxs=None
        )

    del models
    torch.cuda.empty_cache()
    gc.collect()

    ###Prepare Submision csv ###
    predictions_dictionary = {}
    for i in range(len(iids)):

        iid = iids[i]
        tile_id = str(iid.split('_')[-1]).zfill(2)

        raw_pred = preds[i]
        pred = (raw_pred.copy() >= Thresh).astype(np.uint8)

        if remove_noise > 0:
            washed = label(pred,background=0,connectivity=2)
            washed = noise_filter(washed,mina=remove_noise)
            pred = (washed > 0).astype(np.uint8)

        predictions_dictionary.update([(str(tile_id), pd.DataFrame(pred))])


    dfs = []
    for key, value in predictions_dictionary.items():
        ftd = value.unstack().reset_index().rename(columns={'level_0': 'row', 'level_1': 'column', 0: 'label'})
        ftd['tile_row_column'] = f'Tile{key}_' + ftd['row'].astype(str) + '_' + ftd['column'].astype(str)
        ftd = ftd[['tile_row_column', 'label']]
        dfs.append(ftd)

    ####Save Submissio csv ####
    sub = pd.concat(dfs)
    sub.to_csv(f'./final_submission.csv', index = False)
    #print(sub['label'].value_counts())