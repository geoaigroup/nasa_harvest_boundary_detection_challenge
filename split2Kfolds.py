import pandas as pd
import numpy as np

import os
import glob
import random

SEED = 911
K = 4

img_dir = '/home/hasann/Desktop/geogroup/competition_datasets/nasa_rwanda_field_boundary_competition/train_imgs'
random.seed(SEED)

def main():

    
    df = {'iid':[],'fold':[],'idx':[]}
    iids = [os.path.basename(f).split('_2021')[0] for f in glob.glob(f'{img_dir}/nasa_rwanda_field_boundary_competition_source*')]
    iids = list(set(iids))
    random.shuffle(iids)

    indexes = [f.split('_')[-1] for f in iids]
    folds = np.arange(start=0,stop=len(iids)) % K

    df['iid'] = iids
    df['fold'] = folds
    df['idx'] = indexes
    df = pd.DataFrame(df)
    print(df)
    print(df['fold'].value_counts())
    df.to_csv('folds5.csv',index=False)

if __name__ =='__main__':
    main()
