
import time
import os
import numpy as np
import pandas as pd
import random
import gc
import pickle
import math

#
from sklearn.preprocessing import LabelEncoder
#

import lightgbm as lgb

import torch
import torch.nn as nn
import torch.nn.functional as F

## log

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 
    
    
seed_everything(seed=42)

NUM_FOLD = 5  
# 
CUR_MODEL_IDX = 0
NUM_CLASSES = 13


import json
from sklearn.cluster import KMeans  
#
from collections import Counter
import xgboost as xgb

import warnings 
warnings.filterwarnings('ignore')  

print(f'lgb: {lgb.__version__}')
print(f'xgb: {xgb.__version__}')


INPUT_DATA = os.environ['INPUT_DATA']     ## "/home/my_user/model_ecaas_agrifieldnet_bronze/data/input/"
OUTPUT_DATA = os.environ['OUTPUT_DATA']   ## "/home/my_user/model_ecaas_agrifieldnet_bronze/data/output/"

print(f'INPUT_DATA: {INPUT_DATA}')
print(f'OUTPUT_DATA: {OUTPUT_DATA}')

data_dir = f'{INPUT_DATA}'   ## path to dataset: ref_agrifieldnet_competition_v1

### Read Data
# contains all band values for test fields
test_data = pd.read_csv(f'{OUTPUT_DATA}other_data/test_data_full_B.csv')
test_data['field_id'] = test_data['field_id'].astype(int)
print(test_data.shape)

# contains geo features for test fields
test_features = pd.read_csv(f'{OUTPUT_DATA}other_data/features/test_features.csv')
print(test_features.shape)


### Aggregate features
def quantile_025(series):
    return np.quantile(series, q=0.25)

def quantile_075(series):
    return np.quantile(series, q=0.75)


def create_group_features(df, tile_info, is_train):
    
    ## df -- train_data/test_data
    ## tile_info -- train_features/test_features
    ## is_train -- bool indicator of train_data
    
    selected_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    
    df['Vegetation_Idx'] = (df['B08'] - df['B04']) / (df['B08'] + df['B04'])   ## (B8-B4)/(B8+B4)
    
    features_list = selected_bands + ['Vegetation_Idx']    
    
    feat_dict = {
        col: [np.mean, np.min, np.max, np.median, quantile_025, quantile_075] for col in features_list   
        }
    
    # create aggregation features
    features = df.groupby('field_id').agg(feat_dict).reset_index()    
    features.columns = ['_'.join(col) for col in features.columns] 
    features = features.rename(columns={'field_id_': 'field_id'})
    
    # create field_size feature
    size_df = df.groupby('field_id').size().reset_index(name='field_size')
    features = features.merge(size_df, on='field_id', how='left')
    df = features
    
    # merge geo features: center_x, center_y
    df = df.merge(tile_info[['field_id', 'center_x', 'center_y']], on='field_id', how='left')
    print(f'df after tile_info merge: {df.shape}')
    
    # for train fields: merge with target 
    if is_train:
        
        df = df.merge(field_crop_pair, on='field_id')   
        
        le = LabelEncoder()
        df['target'] = le.fit_transform(df['crop_id'].values)
        
        with open('./pickles/le_lgb.pkl', 'wb') as f: 
            pickle.dump(le, f)
            
        
    df = df.reset_index(drop=True)
    
    return df
  

test = create_group_features(test_data.copy(), test_features.copy(), is_train=False)
print(test.shape, test.index)


