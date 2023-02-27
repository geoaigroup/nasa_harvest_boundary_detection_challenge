import os
import random
import numpy as np
import torch
import argparse
import gc

from train import train_model 
from train_kd import train_model as train_model_kd
from utils.test_utils import load_json

from tqdm import tqdm

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

    #special_model_config = 'nasa_rfb_UNET3D_tu-tf_efficientnetv2_s_200epochs_fold2_V15'

    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--configs_dir',type=str,default='./final_models_configs')
    arg('--out_dir',type=str,default='./final_models')
    arg('--data_dir',type=str,\
        default='/home/hasann/Desktop/geogroup/competition_datasets/nasa_rwanda_field_boundary_competition')
    arg('--folds_path',type=str,default='./folds.csv')

    args = parser.parse_args()

    configs_dir = args.configs_dir
    out_dir = args.out_dir 
    data_dir = args.data_dir
    folds_path = args.folds_path

    os.makedirs(out_dir,exist_ok=True)

    config_titles = []
    config_titles_kd = []

    set_seed(911)
    
    for f in sorted(os.listdir(configs_dir)):
        is_kd = f.split('_')[2] == 'KD'
        if is_kd:
            config_titles_kd.append(f)
        else:
            config_titles.append(f)
    
    print('Training 1st stage Models...')

    ###First Training Models without Knowledge Distillation###

    for config_title in config_titles:

        print(f'\n\nModel Config : {config_title}')

        config_cfg = load_json(os.path.join(configs_dir,config_title,'configs.json'))
        
        config_cfg['save_dir'] = os.path.join(out_dir,config_title)
        config_cfg['dataset']['root'] = data_dir
        config_cfg['dataset']['folds_path'] = folds_path
        
        train_model(config_cfg,save_last=False)

        torch.cuda.empty_cache()
        gc.collect()

    ###Second Training Models with Knowledge Distillation###

    print('Training 2nd stage Models...')

    for config_title in config_titles_kd:

        print(f'\n\nModel Config : {config_title}')

        config_cfg = load_json(os.path.join(configs_dir,config_title,'configs.json'))
        
        config_cfg['save_dir'] = os.path.join(out_dir,config_title)
        config_cfg['dataset']['root'] = data_dir
        config_cfg['dataset']['folds_path'] = folds_path
        
        teachers_titles = [tt.split('/')[-1] for tt in config_cfg['kd']['teachers']]
        teachers_paths = [os.path.join(out_dir,tt) for tt in teachers_titles]

        config_cfg['kd']['teachers'] = teachers_paths

        train_model_kd(config_cfg,save_last=False)

        torch.cuda.empty_cache()
        gc.collect()
    
    print('Finished Training all Models...')


