import os
import glob
import json
import getpass  
import rasterio  
import numpy as np
import pandas as pd
from tqdm import tqdm
#from radiant_mlhub import Dataset  ## to drop

import pickle
import rasterio.merge
import rasterio.plot

### 
INPUT_DATA = os.environ['INPUT_DATA']     ## "/home/my_user/model_ecaas_agrifieldnet_bronze/data/input/"
OUTPUT_DATA = os.environ['OUTPUT_DATA']   ## "/home/my_user/model_ecaas_agrifieldnet_bronze/data/output/"

print(f'INPUT_DATA: {INPUT_DATA}')
print(f'OUTPUT_DATA: {OUTPUT_DATA}')

data_dir = f'{INPUT_DATA}'   ## path to dataset: ref_agrifieldnet_competition_v1


### create_dirs
os.makedirs(f'{OUTPUT_DATA}other_data', exist_ok=True)   ## save preprocessed data, created features, field images
os.makedirs(f'{OUTPUT_DATA}other_data/features', exist_ok=True)
os.makedirs(f'{OUTPUT_DATA}other_data/images', exist_ok=True)  ## 
os.makedirs(f'{OUTPUT_DATA}other_data/images/basic', exist_ok=True)  ## 
#os.makedirs(f'{OUTPUT_DATA}other_data/images/basic/train', exist_ok=True)
os.makedirs(f'{OUTPUT_DATA}other_data/images/basic/test', exist_ok=True)
#os.makedirs(f'{OUTPUT_DATA}other_data/images/pretrain', exist_ok=True)  

#os.makedirs('./models', exist_ok=True) 
#os.makedirs('./logs', exist_ok=True)
#os.makedirs('./pickles', exist_ok=True)
#os.makedirs('./submissions', exist_ok=True)

#### START PREPROCESSING
Full_bands = ['B01', 'B02', 'B03', 'B04','B05', 'B06', 'B07', 'B08','B8A', 'B09', 'B11', 'B12']
selected_bands = Full_bands   ## choose all bands

# define dataset collection_id , assets and necessary paths to collections
main = 'ref_agrifieldnet_competition_v1'   ## Dataset ID, fetch by this ID
assets = ['field_ids', 'raster_labels']   

source_collection = f'{main}_source'   ## ds.collections
#train_label_collection = f'{main}_labels_train'
test_label_collection = f'{main}_labels_test'  

###### Prepare Test data  ###
with open (f'{data_dir}{main}/{test_label_collection}/collection.json') as f: 
    test_json = json.load(f)
    
test_folder_ids = [i['href'].split('_')[-1].split('.')[0] for i in test_json['links'][4:]]
test_field_paths = [f'{data_dir}{main}/{test_label_collection}/{test_label_collection}_{i}/field_ids.tif' for i in test_folder_ids]

print(len(test_folder_ids), test_folder_ids[:5])
print(len(test_field_paths), test_field_paths[:1])

competition_test_data = pd.DataFrame(test_folder_ids , columns=['unique_folder_id'])
competition_test_data['field_paths'] = test_field_paths
competition_test_data.to_csv(f'{OUTPUT_DATA}other_data/competition_test_data.csv', index=False)


### Create tile features
def create_tile_features(df):
    ## df - competition_train_data/competition_test_data  
    
    tile_info = []
    tile_ids_all = []
    field_ids_all = []
    
    for i, x in enumerate(df.itertuples(index=False)): 
        
        with rasterio.open(x.field_paths) as src:    ## read tile with several field_ids
            
            field_data = src.read()[0]  ## np array: (256, 256)
            field_ids = np.unique(field_data).tolist()  ## with zero field_id value
            
            field_ids_all.extend(field_ids)
            tile_ids_all.extend([x.unique_folder_id] * len(field_ids))
            
            bounds = src.bounds
            tile_info.append((x.unique_folder_id, bounds.left, bounds.bottom, bounds.right, bounds.top))
              
        
    field_tile_df = pd.DataFrame({'field_id': field_ids_all, 
                                  'folder_id': tile_ids_all})
    field_tile_df = field_tile_df.loc[field_tile_df['field_id'] != 0]   ## drop zero field_id
    print(f'field_tile_df: {field_tile_df.shape}')
        
    tile_info = pd.DataFrame(tile_info, columns=['folder_id', 'left', 'bottom', 'right', 'top'])
        
    field_tile_df = field_tile_df.merge(tile_info, on='folder_id', how='left')
    field_tile_df = field_tile_df.reset_index(drop=True)
    print(f'field_tile_df: {field_tile_df.shape}')
    
    return field_tile_df
    

test_field_tile_df = create_tile_features(competition_test_data.copy())
print(test_field_tile_df.shape)
print(test_field_tile_df.isnull().sum().sum())
print(test_field_tile_df.index)
test_field_tile_df.to_csv(f'{OUTPUT_DATA}other_data/test_field_tile_df.csv', index=False)


### Extract field images
def extract_field_images(df, field_tile_df, out_dir):
    ## df -- competition_train_data/competition_test_data
    ## field_tile_df -- train_field_tile_df/test_field_tile_df
    ## out_dir -- dir to save field bands
    
    print(f'selected_bands: {len(selected_bands)}')   ### all 12 bands
    
    field_tile_dict = field_tile_df.groupby('field_id')['folder_id'].apply(list).to_dict()  # field_id -> folder_id
    print(f'field_tile_dict: {len(field_tile_dict)}')
    
    meta_info = []
    
    ### iterate through field_ids
    for i, (field_id, tile_ids) in tqdm(enumerate(field_tile_dict.items())):
        
        field_paths = df.loc[df['unique_folder_id'].isin(tile_ids), 'field_paths'].values.tolist()
        
        merged_field, _ = rasterio.merge.merge(field_paths)   ## merge chips
        merged_field = merged_field[0]   # (1, H, W)    -> (H, W) 
        
        field_mask = (merged_field == field_id).astype(int)  ## mask of current field_id
         
        ### extract field bounds
        mask_nonzero = field_mask.nonzero()
        up, down = mask_nonzero[0].min(), mask_nonzero[0].max()  
        left, right = mask_nonzero[1].min(), mask_nonzero[1].max()
        
        field_mask_crop = field_mask[up:down+1, left:right+1]  ## cropped field mask 
        
        band_img_all = np.zeros((down-up+1, right-left+1, len(selected_bands)), dtype=np.uint8)   ### (H, W, C)
        
        ### iterate through all bands
        for band_idx, band in enumerate(selected_bands):
            
            band_paths = [f'{data_dir}{main}/{source_collection}/{source_collection}_{tile_id}/{band}.tif' for tile_id in tile_ids]
            
            band_img, _ = rasterio.merge.merge(band_paths)   # (1, H, W);  merge all chips with current field_id
            band_crop = band_img[0, up:down+1, left:right+1]   ## crop pixels with current field
            band_crop_masked = band_crop * field_mask_crop   ## extract pixels which belong to field
            band_img_all[:, :, band_idx] = band_crop_masked
            
            
        ### save band_img_all
        out_path = os.path.join(out_dir, f'img_{field_id}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(band_img_all, f)

        ### add meta info
        meta_info.append((field_id, out_path, band_img_all.shape[0], band_img_all.shape[1]))
        
        
    meta_info = pd.DataFrame(meta_info, columns=['field_id', 'img_path', 'H', 'W'])
    print(f'meta_info: {meta_info.shape}')
    
    
    return meta_info
    

test_meta_info = extract_field_images(competition_test_data.copy(), test_field_tile_df.copy(), 
                     f'{OUTPUT_DATA}other_data/images/basic/test/')

print(test_meta_info.shape)
print(test_meta_info.isnull().sum().sum())
print(test_meta_info.index)
print(len(os.listdir(f'{OUTPUT_DATA}other_data/images/basic/test/')))
test_meta_info.to_csv(f'{OUTPUT_DATA}other_data/test_meta_info.csv', index=False)


### Create geo features
def extract_field_features(df, field_tile_df):
    ## df -- competition_train_data/competition_test_data
    ## field_tile_df -- train_field_tile_df/test_field_tile_df
    
    print(f'selected_bands: {len(selected_bands)}')   
    
    field_tile_dict = field_tile_df.groupby('field_id')['folder_id'].apply(list).to_dict()
    print(f'field_tile_dict: {len(field_tile_dict)}')
    
    meta_info = []  ## collect field features here
    
    ### iterate through field_ids
    for i, (field_id, tile_ids) in tqdm(enumerate(field_tile_dict.items())):
        
        field_paths = df.loc[df['unique_folder_id'].isin(tile_ids), 'field_paths'].values.tolist()
        
        merged_field, merged_transform = rasterio.merge.merge(field_paths)  ## merge chips
        merged_field = merged_field[0]   # (1, H, W)    -> (H, W) 
        
        field_mask_bool = (merged_field == field_id)   
        field_mask = field_mask_bool.astype(int)
         
        mask_nonzero = field_mask.nonzero()
        up, down = mask_nonzero[0].min(), mask_nonzero[0].max()   
        left, right = mask_nonzero[1].min(), mask_nonzero[1].max()
        
        ## geo coordinates of field center
        center_x, center_y = merged_transform * (int((left + right)/2), int((up + down)/2))
        
        ## sizes of field crop rectangle
        H = down-up+1
        W = right-left+1
        aspect_ratio = H / W
        
        ### add meta info
        meta_info.append((field_id, center_x, center_y, H, W, aspect_ratio))
        
        
    meta_info = pd.DataFrame(meta_info, columns=['field_id', 'center_x', 'center_y', 'H', 'W', 'aspect_ratio'])
    print(f'meta_info: {meta_info.shape}')
    meta_info = meta_info.reset_index(drop=True)
    
    return meta_info
    

test_features = extract_field_features(competition_test_data.copy(), test_field_tile_df.copy())
print(test_features.shape)
print(test_features.isnull().sum().sum())
print(test_features.index)
test_features.to_csv(f'{OUTPUT_DATA}other_data/features/test_features.csv', index=False)


### Extract bands features
img_sh = 256
n_selected_bands= len(selected_bands)   ## all 12 channels

n_obs = 1  # imagery per chip(no time series)

def feature_extractor(data_ ,   path ):
    '''
        data_: Dataframe with 'field_paths' and 'unique_folder_id' columns
        path: Path to source collections files

        returns: pixel dataframe with corresponding field_ids
        '''
    
    X = np.empty((0, n_selected_bands * n_obs))  ## (0, 12*1)
    X_tile = np.empty((img_sh * img_sh, 0))      ## (256*256, 0)
    X_arrays = []
        
    field_ids = np.empty((0, 1))

    for idx, tile_id in tqdm(enumerate(data_['unique_folder_id'])):   ## tile_id: '28852', 'd987c', 'ca1d4'...
        
        field_src =   rasterio.open( data_['field_paths'].values[idx])   ## open: field_ids.tif
        field_array = field_src.read(1)  ## np array: (256, 256)
        field_ids = np.append(field_ids, field_array.flatten())  ## field_ids: array with ndim = 1
        
        bands_src = [rasterio.open(f'{data_dir}{main}/{path}/{path}_{tile_id}/{band}.tif') for band in selected_bands]
        bands_array = [np.expand_dims(band.read(1).flatten(), axis=1) for band in bands_src]
        
        X_tile = np.hstack(bands_array)  ## (256*256, 12) 

        X_arrays.append(X_tile)
        

    X = np.concatenate(X_arrays)  ## (256*256*n_tiles, 12)
    
    data = pd.DataFrame(X, columns=selected_bands)
    data['field_id'] = field_ids
    print(f'data: {data.shape}')   

    return data[data['field_id']!=0]   


test_data = feature_extractor(competition_test_data,  source_collection)
print(test_data.shape)
test_data.to_csv(f'{OUTPUT_DATA}other_data/test_data_full_B.csv', index=False)



