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

# 
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

data_dir = f'{INPUT_DATA}'   ## path to dataset: contains ref_agrifieldnet_competition_v1

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

### Create geo cluster features
def compute_cl_distances(df, cl_centers):
    ## df -- df/test
    ## cl_centers -- coordinates of cluster centers from KMeans model
    
    for i in range(len(cl_centers)):
        
        df[f'geo_dist_cl_{i}'] = ((df['center_x'] - cl_centers[i, 0])**2 + (df['center_y'] - cl_centers[i, 1])**2 )**0.5 
        df[f'geo_dist_cl_x_{i}'] = df['center_x'] - cl_centers[i, 0]
        df[f'geo_dist_cl_y_{i}'] = df['center_y'] - cl_centers[i, 1]  
        
        
    return df
    
    
def run_kmeans(test):    ## dropped: df
        
    with open(f'{INPUT_DATA}checkpoint/kmeans/kmeans_geo.pkl', 'rb') as f:
        km = pickle.load(f)
        
    cl_centers = km.cluster_centers_
    print(cl_centers)
        
    X_test = test[['center_x', 'center_y']].copy()
    test_labels = km.predict(X_test)  ## predict 5 geo cluster labels for test fields
    print(f'test_labels: {test_labels.shape}')
        
    test['geo_cluster'] = test_labels
    
    # One Hot Encoding for geo cluster labels 
    test = pd.get_dummies(test, columns=['geo_cluster'])
    
    # Create geo distance of fields to all cluster centers
    test = compute_cl_distances(test, cl_centers)
    
    return test
    
    
test = run_kmeans(test.copy())   ## dropped: df
print(test.shape, test.index)
test.to_csv(f'{OUTPUT_DATA}other_data/features/test_features_stats2.csv', index=False)


### LGB models
def train_and_evaluate_lgb(test, n_fold=5, seed=42):  ## drop train
    
    seed_everything(seed=seed)
    
    drop_columns = ['field_id', 'crop_id', 'fold', 'target', 'weight'] 
    num_features = [col for col in test.columns if col not in drop_columns]    ## train->test
    
    x_test = test[num_features]
    test_predictions = np.zeros((x_test.shape[0], NUM_CLASSES))    ### probas
    
    for fold_num in range(n_fold):
            
        model = lgb.Booster(model_file=f'{INPUT_DATA}checkpoint/lgb_models/lgb_fold_{fold_num}_seed_{seed}.txt')
        test_predictions += model.predict(x_test) / n_fold  
         

    return test_predictions
    

predictions_all = []   ## collect all test predictions here 
     
NUM_FOLD = 10
CUR_MODEL_IDX = 0
predictions = train_and_evaluate_lgb(test.copy(), n_fold=NUM_FOLD, seed=CUR_MODEL_IDX)
predictions_all.append(predictions)

CUR_MODEL_IDX = 10
predictions = train_and_evaluate_lgb(test.copy(), n_fold=NUM_FOLD, seed=CUR_MODEL_IDX)
predictions_all.append(predictions)

CUR_MODEL_IDX = 20
predictions = train_and_evaluate_lgb(test.copy(), n_fold=NUM_FOLD, seed=CUR_MODEL_IDX)
predictions_all.append(predictions)

CUR_MODEL_IDX = 30 
predictions = train_and_evaluate_lgb(test.copy(), n_fold=NUM_FOLD, seed=CUR_MODEL_IDX)
predictions_all.append(predictions)

CUR_MODEL_IDX = 40
predictions = train_and_evaluate_lgb(test.copy(), n_fold=NUM_FOLD, seed=CUR_MODEL_IDX)
predictions_all.append(predictions)

print(len(predictions_all))


### check: correct read of label_json  
def create_submit(preds, test):
    
    with open(f'{data_dir}ref_agrifieldnet_competition_v1/ref_agrifieldnet_competition_v1_labels_train/ref_agrifieldnet_competition_v1_labels_train_001c1/ref_agrifieldnet_competition_v1_labels_train_001c1.json') as ll:
        label_json = json.load(ll)
        
    crop_dict = {asset.get('values')[0]:asset.get('summary') for asset in label_json['assets']['raster_labels']['file:values']}
    
    with open(f'{INPUT_DATA}checkpoint/pickles/le_lgb.pkl', 'rb') as f:  
        le = pickle.load(f)
    
    crop_columns = [crop_dict.get(i) for i in le.classes_]
    sub = pd.DataFrame(columns = ['field_id'] + crop_columns)
    sub['field_id'] = test['field_id']
    sub[crop_columns] = preds 
    
    return sub
    
    
sub = create_submit(np.array(predictions_all).mean(axis=0), test.copy())
print(sub.shape)
print(sub.isnull().sum().sum())

sub.to_csv(f'{OUTPUT_DATA}sub_bst.csv', index=False)   #### SAVING





