import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'

# Where to save the figures
PROJECT_ROOT_DIR = "."
PROJECT_SAVE_DIR = "Figure_PDFs"

import os
if not (os.path.isdir(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR)):
    print('Figure directory didn''t exist, creating now.')
    os.mkdir(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR)
else:
    print('Figure directory exists.')

# Ignore warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from scipy import stats
import copy
import pickle as pkl
from tqdm import tqdm
import joblib
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.ensemble import RandomForestRegressor

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Define a function to save future figures to PDFs
def savepdf(fig,name):
    fig.savefig(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR+'/'+name+'.pdf')

# To evaluate the statistics between predicted and true PM2.5
def eval_stat(y_train_pred, y_train):
    Rsquared = stats.spearmanr(y_train_pred, y_train.ravel())[0]
    pvalue = stats.spearmanr(y_train_pred, y_train.ravel())[1]
    Rsquared_pearson = stats.pearsonr(y_train_pred, y_train.ravel())[0]
    pvalue_pearson = stats.pearsonr(y_train_pred, y_train.ravel())[1]
    return Rsquared, pvalue, Rsquared_pearson, pvalue_pearson

# To plot the predicted and true PM2.5 along with the calculated statistics
def plot_result(y_train_pred, y_train, Rsquared, pvalue, Rsquared_pearson, pvalue_pearson, plot_label="train", save=True, 
                fig_name="", lower_bound=0, upper_bound=100, spatial_R=-1):
    plt.rcParams.update({'mathtext.default':  'regular' })
    my_prediction = copy.copy(y_train_pred)
    fig, ax = plt.subplots(figsize = (8,8))
    ax.scatter(y_train, my_prediction,color = 'orange', edgecolors=(0, 0, 0),  s = 100)
    ax.plot([lower_bound, upper_bound], [lower_bound, upper_bound], 'k--', lw=4)
    ax.set_xlabel('True $PM_{2.5}$ ($\mu $g m$^{-3}$)', size = 20)
    ax.set_ylabel('Predicted $PM_{2.5}$ ($\mu $g m$^{-3}$)', size = 20)
    ax.tick_params(labelsize = 15)
    fig.text(0.15, 0.84, 'Spearman r = '+ str(round(Rsquared,2)), color='black', weight='roman', 
    fontsize=16)
    fig.text(0.15, 0.80, 'Spearman p-value = '+ str(round(pvalue,2)), color='black', weight='roman', 
    fontsize=16)
    plt.axis('tight')
    fig.text(0.15, 0.76, 'Pearson r = '+ str(round(Rsquared_pearson,2)), color='black', weight='roman', 
    fontsize=16)
    fig.text(0.15, 0.72, 'Pearson p-value = '+ str(round(pvalue_pearson,3)), color='black', weight='roman', 
    fontsize=16)
    fig.text(0.15, 0.68, 'RMSE = '+ str(round(np.sqrt(metrics.mean_squared_error(y_train, my_prediction)),2)), 
    color='black', weight='roman', fontsize=16)
    fig.text(0.15, 0.64, 'MAE = '+ str(round(metrics.mean_absolute_error(y_train, my_prediction),2)), 
    color='black', weight='roman', fontsize=16)
    fig.text(0.15, 0.60, '% error = '+ str(round(metrics.mean_absolute_error(y_train, my_prediction)/np.mean(y_train)*100,1))+'%', 
    color='black', weight='roman', fontsize=16)
    if spatial_R != -1:
        fig.text(0.15, 0.56, 'Spatial Pearson r = ' + str(round(spatial_R, 2)), color='black', weight='roman', 
        fontsize=16)
    if plot_label == "train":
        fig.text(0.615, 0.188, 'training dataset', bbox=dict(facecolor='grey', alpha=0.9),color='black', weight='roman',
        fontsize=20)
    else:
        fig.text(0.615, 0.188, 'test dataset', bbox=dict(facecolor='grey', alpha=0.9),color='black', weight='roman',
        fontsize=20)
        
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    if save:
        savepdf(fig, fig_name)
    pass
    del fig, ax
    return

# To plot the spatial R and RMSE for the predicted and true PM2.5 along with the calculated statistics
def spatialRPlot(color, y_test_ref,  y_test_ref_pred_raw, plot_label = 'test', save=False, fig_name=""):
    Rsquared, pvalue, Rsquared_pearson, pvalue_pearson = eval_stat(y_test_ref_pred_raw, y_test_ref)
    
    y_train_pred_mlpr,y_train, Rsquared, pvalue, Rsquared_pearson, pvalue_pearson = y_test_ref_pred_raw, y_test_ref, Rsquared, pvalue, Rsquared_pearson, pvalue_pearson
        
    plt.rcParams.update({'mathtext.default':  'regular' })
    my_prediction = y_train_pred_mlpr
    fig, ax = plt.subplots(figsize = (8,8))
    ax.scatter(y_train, my_prediction, color = color,alpha =1, edgecolors='navy',  s = 100)
    ax.plot([50, 150], [50, 150], 'k--', lw=4)
    ax.set_xlabel('True $PM_{2.5}$ ($\mu $g m$^{-3}$)', size = 25)
    ax.set_ylabel('Predicted $PM_{2.5}$ ($\mu $g m$^{-3}$)', size = 25)
    ax.tick_params(labelsize = 25)
    horozontal_ax = 0.05
    vertical_offset = 0.2
    ax.text(horozontal_ax, 0.72+vertical_offset, 'Spatial Pearson r = '+ str(round(Rsquared_pearson,2)), color='black', weight='roman',
    fontsize=25,transform=ax.transAxes)
    ax.text(horozontal_ax, 0.65+vertical_offset, 'p-value = '+ str(round(pvalue_pearson,3)), color='black', weight='roman',
    fontsize=25,transform=ax.transAxes)
    ax.text(horozontal_ax, 0.58+vertical_offset, 'RMSE = '+ str(round(np.sqrt(metrics.mean_squared_error(y_train, my_prediction)),2)), 
    color='black', weight='roman', fontsize=25, transform=ax.transAxes)
   
    if plot_label == "train":
        ax.text(0.575, 0.014, 'training dataset', bbox=dict(facecolor='grey', alpha=0.9),color='black', weight='roman',
        fontsize=25,transform=ax.transAxes)
    else:
        ax.text(0.665, 0.0190, 'test dataset', bbox=dict(facecolor='grey', alpha=0.9),color='black', weight='roman',
        fontsize=25,transform=ax.transAxes) 
    plt.tight_layout()
    if save:
        savepdf(fig, fig_name)
    pass
    del fig, ax
    return

# Image dataset used for downstream supervised task with regular MSE loss
class MyPM25Dataset(Dataset):

    def __init__(self, root_dir, crop_dim=0, img_transform=None, mode='train', holdout=None, train_stations=-1, 
                 requires_meteo=False, meteo_model=None, rf_train=None, rf_test=None):
        """
        Args:
            root_dir (string): Directory of PM2.5 data
            crop_dim (int, optional): Dimension for cropping
            mode ('train' or 'test', optional): Whether the dataset is for
                training or testing
            img_transform (callable, optional): Optional transform to be applied
                on an image.
            holdout (list of station index): Must be specified if mode is 'test'
            train_stations (integer): Number of stations to be used for training
            requires_meteo (boolean): Whether to use meteorological features or not
            meteo_model (optional): RandomTreesEmbedding model
            rf_train (optional): Train predictions using Random Forest model
            rf_test (optional): Test predictions using Random Forest model
        """
        if mode not in ['train', 'test']:
            raise Exception('Mode must be either \'train\' or \'test\'.')
        if mode == 'test' and (not holdout and train_stations == -1):
            raise Exception('Must specify the holdout test set or the number of training stations')
        if holdout and train_stations != -1:
            raise Exception('Either specify holdout stations or specify the number of training stations.')
        if requires_meteo and not meteo_model:
            raise Exception('If meteo features are required, you must pass in a model to transform the meteo features.')
        if requires_meteo:
            if mode == 'train' and rf_train is None:
                raise Exception('Please pass in training predictions from Random Forest model')
            elif mode == 'test' and rf_test is None:
                raise Exception('Please pass in test predictions from Random Forest model')
        
        # Pass in parameters
        self.crop_dim = crop_dim
        self.img_transform = img_transform
        self.mode = mode
        self.holdout = holdout
        self.train_stations = train_stations
        self.requires_meteo = requires_meteo
        self.y_train_pred_rf = rf_train
        self.y_test_pred_rf = rf_test
        
        # Private variables
        self.img_train_PM25, self.img_test_PM25 = [], []
        self.PM25_train, self.PM25_test = [], []
        self.train_stations_seen = set()
        
        self.meteo_raw = []
        self.meteo_raw_train, self.meteo_raw_test = [], []
        self.meteo_transformed_train, self.meteo_transformed_test = [], []
        
        
        # Load images, meteo features and targets for PM2.5 data
        with open(root_dir, "rb") as fp:
            images = pkl.load(fp)
            if self.train_stations != -1:
                for data_point in tqdm(images, position=0, leave=True):
                    if data_point['Station_index'] not in self.train_stations_seen:
                        self.train_stations_seen.add(data_point['Station_index'])
                    if len(self.train_stations_seen) <= self.train_stations:
                        self.img_train_PM25.append(data_point['Image'])
                        self.PM25_train.append(data_point['PM25'])
                        if self.requires_meteo:
                            self.meteo_raw_train.append(data_point['Meteo'].values)
                    else:
                        self.img_test_PM25.append(data_point['Image'])
                        self.PM25_test.append(data_point['PM25'])
                        if self.requires_meteo:
                            self.meteo_raw_test.append(data_point['Meteo'].values)
            else:
                for data_point in tqdm(images, position=0, leave=True):
                    if data_point['Station_index'] not in holdout:
                        self.img_train_PM25.append(data_point['Image'])
                        self.PM25_train.append(data_point['PM25'])
                        if self.requires_meteo:
                            self.meteo_raw_train.append(data_point['Meteo'].values)
                    else:
                        self.img_test_PM25.append(data_point['Image'])
                        self.PM25_test.append(data_point['PM25'])
                        if self.requires_meteo:
                            self.meteo_raw_test.append(data_point['Meteo'].values)
        
        if self.requires_meteo:
            # Transform the meteo features to increase the representation power
            self.meteo_transformed_train = meteo_model.transform(self.meteo_raw_train).toarray()
            self.meteo_transformed_test = meteo_model.transform(self.meteo_raw_test).toarray()
            
        # Remove unnecessary data
        if self.mode == 'train':
            del self.img_test_PM25, self.PM25_test, self.meteo_transformed_test
        else:
            del self.img_train_PM25, self.PM25_train, self.meteo_transformed_train
        del self.meteo_raw, self.meteo_raw_train, self.meteo_raw_test
            
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.img_train_PM25)
        else:
            return len(self.img_test_PM25)
        
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Get images, transformed meteo features and targets
        if self.mode == 'train':
            img = self.img_train_PM25[idx]
            target = self.PM25_train[idx]
            if self.requires_meteo:
                meteo = self.meteo_transformed_train[idx]
                target_pred = self.y_train_pred_rf[idx]
        else:
            img = self.img_test_PM25[idx]
            target = self.PM25_test[idx] 
            if self.requires_meteo:
                meteo = self.meteo_transformed_test[idx]
                target_pred = self.y_test_pred_rf[idx]
        # Crop the image if crop_dim is specified
        if self.crop_dim != 0:
            crop = transforms.Compose([transforms.ToPILImage(), 
                                       transforms.CenterCrop((self.crop_dim, self.crop_dim)),
                                       transforms.ToTensor()])
            img = crop(img)
        # Perform data augmentation if transform function is specified
        if self.img_transform:
            img = self.img_transform(img)
        
        if self.requires_meteo:
            return img, meteo, target, target_pred
        else:
            return img, target

# Temporally sorted image dataset used for downstream supervised task with weighted MSE loss
class MyPM25DatasetSorted(Dataset):

    def __init__(self, root_dir, img_transform=None, mode='train', holdout=None, train_stations=-1, 
                 requires_meteo=False, meteo_model=None, rf_train=None, rf_test=None):
        """
        Args:
            root_dir (string): Directory of PM2.5 data
            mode ('train' or 'test', optional): Whether the dataset is for
                training or testing
            img_transform (callable, optional): Optional transform to be applied
                on an image.
            target_transform (boolean, optional): If true, then normalize y
            holdout (list of station index): Must be specified if mode is 'test'
            train_stations (integer): Number of stations to be used for training
            requires_meteo (boolean): Whether to use meteorological features or not
            meteo_model (optional): RandomTreesEmbedding model
            rf_train (optional): Train predictions using Random Forest model
            rf_test (optional): Test predictions using Random Forest model
        """
        if mode not in ['train', 'test']:
            raise Exception('Mode must be either \'train\' or \'test\'.')
        if mode == 'test' and (not holdout and train_stations == -1):
            raise Exception('Must specify the holdout test set or the number of training stations')
        if holdout and train_stations != -1:
            raise Exception('Either specify holdout stations or specify the number of training stations.')
        if requires_meteo and not meteo_model:
            raise Exception('If meteo features are required, you must pass in a model to transform the meteo features.')
        if requires_meteo:
            if mode == 'train' and rf_train is None:
                raise Exception('Please pass in training predictions from Random Forest model')
            elif mode == 'test' and rf_test is None:
                raise Exception('Please pass in test predictions from Random Forest model')
        
        # Pass in parameters
        self.img_transform = img_transform
        self.mode = mode
        self.holdout = holdout
        self.train_stations = train_stations
        self.requires_meteo = requires_meteo
        self.y_train_pred_rf = rf_train
        self.y_test_pred_rf = rf_test
        
        # Private variables
        self.img_train_PM25, self.img_test_PM25 = [], []
        self.PM25_train, self.PM25_test = [], []
        self.train_stations_seen = set()
        
        self.meteo_raw = []
        self.meteo_raw_train, self.meteo_raw_test = [], []
        self.meteo_transformed_train, self.meteo_transformed_test = [], []
        
        
        # Load images, meteo features and targets for PM2.5 data
        with open(root_dir, "rb") as fp:
            # Sort the images based on their timestamp
            images = pkl.load(fp)
            images.sort(key=lambda x: x['Meteo'].name)
            cur_timestamp_train, cur_timestamp_test = None, None
            if self.train_stations != -1:
                for data_point in images:
                    if data_point['Station_index'] not in self.train_stations_seen:
                        self.train_stations_seen.add(data_point['Station_index'])
                    if len(self.train_stations_seen) <= self.train_stations:
                        if data_point['Meteo'].name != cur_timestamp_train:
                            cur_timestamp_train = data_point['Meteo'].name
                            self.img_train_PM25.append([])
                            self.PM25_train.append([])
                            if self.requires_meteo:
                                self.meteo_raw_train.append([])
                        self.img_train_PM25[-1].append(data_point['Image'])
                        self.PM25_train[-1].append(data_point['PM25'])
                        if self.requires_meteo:
                            self.meteo_raw_train[-1].append(data_point['Meteo'].values)
                    else:
                        if data_point['Meteo'].name != cur_timestamp_test:
                            cur_timestamp_test = data_point['Meteo'].name
                            self.img_test_PM25.append([])
                            self.PM25_test.append([])
                            if self.requires_meteo:
                                self.meteo_raw_test.append([])
                        self.img_test_PM25[-1].append(data_point['Image'])
                        self.PM25_test[-1].append(data_point['PM25'])
                        if self.requires_meteo:
                            self.meteo_raw_test[-1].append(data_point['Meteo'].values)
            else:
                for data_point in tqdm(images, position=0, leave=True):
                    if data_point['Station_index'] not in holdout:
                        if data_point['Meteo'].name != cur_timestamp_train:
                            cur_timestamp_train = data_point['Meteo'].name
                            self.img_train_PM25.append([])
                            self.PM25_train.append([])
                            if self.requires_meteo:
                                self.meteo_raw_train.append([])
                        self.img_train_PM25[-1].append(data_point['Image'])
                        self.PM25_train[-1].append(data_point['PM25'])
                        if self.requires_meteo:
                            self.meteo_raw_train[-1].append(data_point['Meteo'].values)
                    else:
                        if data_point['Meteo'].name != cur_timestamp_test:
                            cur_timestamp_test = data_point['Meteo'].name
                            self.img_test_PM25.append([])
                            self.PM25_test.append([])
                            if self.requires_meteo:
                                self.meteo_raw_test.append([])
                        self.img_test_PM25[-1].append(data_point['Image'])
                        self.PM25_test[-1].append(data_point['PM25'])
                        if self.requires_meteo:
                            self.meteo_raw_test[-1].append(data_point['Meteo'].values)
        
        # Perform data augmentation if transform function is specified
        if self.img_transform:
            for i in range(len(self.img_train_PM25)):
                for j in range(len(self.img_train_PM25[i])):
                    self.img_train_PM25[i][j] = self.img_transform(self.img_train_PM25[i][j])
            for i in range(len(self.img_test_PM25)):
                for j in range(len(self.img_test_PM25[i])):
                    self.img_test_PM25[i][j] = self.img_transform(self.img_test_PM25[i][j])
        
        if self.requires_meteo:
            # Transform the meteo features to increase the representation power
            for meteo_day in tqdm(self.meteo_raw_train, position=0, leave=True):
                self.meteo_transformed_train.append(meteo_model.transform(meteo_day).toarray())
            for meteo_day in tqdm(self.meteo_raw_test, position=0, leave=True):
                self.meteo_transformed_test.append(meteo_model.transform(meteo_day).toarray())
            self.meteo_transformed_train = np.array(self.meteo_transformed_train)
            self.meteo_transformed_test = np.array(self.meteo_transformed_test)
            
        # Remove unnecessary data
        if self.mode == 'train':
            del self.img_test_PM25, self.PM25_test, self.meteo_transformed_test
        else:
            del self.img_train_PM25, self.PM25_train, self.meteo_transformed_train
        del self.meteo_raw, self.meteo_raw_train, self.meteo_raw_test
            
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.img_train_PM25)
        else:
            return len(self.img_test_PM25)
        
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Get images, transformed meteo features and targets
        if self.mode == 'train':
            img = self.img_train_PM25[idx]
            target = self.PM25_train[idx]
            if self.requires_meteo:
                meteo = self.meteo_transformed_train[idx]
                target_pred = self.y_train_pred_rf[idx]
        else:
            img = self.img_test_PM25[idx]
            target = self.PM25_test[idx] 
            if self.requires_meteo:
                meteo = self.meteo_transformed_test[idx]
                target_pred = self.y_test_pred_rf[idx]
        
        if self.requires_meteo:
            return img, meteo, target, target_pred
        else:
            return img, target

# Load Random Trees Embedding and Random Forest model for regular MSE loss
def loadRTandRFModel(root_dir, rt_dir, rf_dir, holdout):
    with open(root_dir, "rb") as fp:
        meteo_raw, meteo_raw_train, y_train, meteo_raw_test, y_test = [], [], [], [], []
        for data_point in pkl.load(fp):
            meteo_raw.append(data_point['Meteo'].values)
            if data_point['Station_index'] not in holdout:
                meteo_raw_train.append(data_point['Meteo'].values)
                y_train.append(data_point['PM25'])
            else:
                meteo_raw_test.append(data_point['Meteo'].values)
                y_test.append(data_point['PM25'])
        meteo_raw = np.array(meteo_raw)
        meteo_raw_train, y_train = np.array(meteo_raw_train), np.array(y_train)
        meteo_raw_test, y_test = np.array(meteo_raw_test), np.array(y_test)
    
    # Load Random Trees Embedding Model
    rt_model = joblib.load(rt_dir)
    
    # Load Random Forest Model
    print("Loading Random Forest Model...")
    rf_model = joblib.load(rf_dir)
    
    # Transform the meteo features
    meteo_transformed_train = rt_model.transform(meteo_raw_train).toarray()
    meteo_transformed_test = rt_model.transform(meteo_raw_test).toarray()
    del meteo_raw
    
    return rt_model, rf_model, meteo_transformed_train, y_train, meteo_transformed_test, y_test

# Load Random Trees Embedding and Random Forest model with temporally sorted images for weighted MSE loss
def loadRTandRFModelSorted(root_dir, rt_dir, rf_dir, holdout):
    with open(root_dir, "rb") as fp:
        # Sort the images based on their timestamp
        images = pkl.load(fp)
        images.sort(key=lambda x: x['Meteo'].name)
        meteo_raw_train, y_train, meteo_raw_test, y_test = [], [], [], []
        cur_timestamp_train, cur_timestamp_test = None, None
        for data_point in images:
            if data_point['Station_index'] not in holdout:
                if data_point['Meteo'].name != cur_timestamp_train:
                    cur_timestamp_train = data_point['Meteo'].name
                    meteo_raw_train.append([])
                    y_train.append([])
                meteo_raw_train[-1].append(data_point['Meteo'].values)
                y_train[-1].append(data_point['PM25'])
            else:
                if data_point['Meteo'].name != cur_timestamp_test:
                    cur_timestamp_test = data_point['Meteo'].name
                    meteo_raw_test.append([])
                    y_test.append([])
                meteo_raw_test[-1].append(data_point['Meteo'].values)
                y_test[-1].append(data_point['PM25'])
        meteo_raw_train, y_train = np.array(meteo_raw_train), np.array(y_train)
        meteo_raw_test, y_test = np.array(meteo_raw_test), np.array(y_test)
    
    # Load Random Trees Embedding Model
    rt_model = joblib.load(rt_dir)
    
    # Load Random Forest Model
    print("Loading Random Forest Model...")
    rf_model = joblib.load(rf_dir)
    
    # Transform the meteo features
    meteo_transformed_train, meteo_transformed_test = [], []
    for meteo_day in tqdm(meteo_raw_train, position=0, leave=True):
        meteo_transformed_train.append(rt_model.transform(meteo_day).toarray())
    for meteo_day in tqdm(meteo_raw_test, position=0, leave=True):
        meteo_transformed_test.append(rt_model.transform(meteo_day).toarray())
    
    meteo_transformed_train = np.array(meteo_transformed_train)
    meteo_transformed_test = np.array(meteo_transformed_test)
    
    return rt_model, rf_model, meteo_transformed_train, y_train, meteo_transformed_test, y_test

# Make predictions with the loaded Random Forest model for regular MSE loss
def predictWithRF(rf_model, meteo_transformed_train, meteo_transformed_test):
    y_train_pred_rf = rf_model.predict(meteo_transformed_train)
    y_test_pred_rf = rf_model.predict(meteo_transformed_test)
    return y_train_pred_rf, y_test_pred_rf

# Make predictions with the loaded Random Forest model and temporally sorted images for weighted MSE loss
def predictWithRFSorted(rf_model, meteo_transformed_train, meteo_transformed_test):
    y_train_pred_rf, y_test_pred_rf = [], []
    print("Predicting with Random Forest Model...")
    for meteo_day in tqdm(meteo_transformed_train, position=0, leave=True):
        y_train_pred_rf.append(rf_model.predict(meteo_day))
    for meteo_day in tqdm(meteo_transformed_test, position=0, leave=True):
        y_test_pred_rf.append(rf_model.predict(meteo_day))
        
    y_train_pred_rf = np.array(y_train_pred_rf)
    y_test_pred_rf = np.array(y_test_pred_rf)
    
    return y_train_pred_rf, y_test_pred_rf

# Initialize the data loader for CNN models with regular MSE loss
def initializeCNNdata(root_dir, img_transform, batch_size, holdout=None, train_stations=-1, requires_meteo=False, rt_model=None, rf_train=None, rf_test=None):
    if requires_meteo:
        if (rt_model is None) or (rf_train is None) or (rf_test is None):
            raise Exception("Must specify rt_model, rf_train and rf_test.")
        train_dataset_PM25 = MyPM25Dataset(root_dir=root_dir, img_transform=img_transform, mode='train', holdout=holdout, 
                                           train_stations=train_stations, requires_meteo=requires_meteo, meteo_model=rt_model, rf_train=rf_train)
        test_dataset_PM25 = MyPM25Dataset(root_dir=root_dir, img_transform=img_transform, mode='test', holdout=holdout, 
                                          train_stations=train_stations, requires_meteo=requires_meteo, meteo_model=rt_model, rf_test=rf_test)
    else:
        train_dataset_PM25 = MyPM25Dataset(root_dir=root_dir, img_transform=img_transform, mode='train', 
                                           holdout=holdout, train_stations=train_stations, requires_meteo=requires_meteo)
        test_dataset_PM25 = MyPM25Dataset(root_dir=root_dir, img_transform=img_transform, mode='test', 
                                          holdout=holdout, train_stations=train_stations, requires_meteo=requires_meteo)
    train_dataloader_PM25 = DataLoader(train_dataset_PM25, batch_size=batch_size, shuffle=True, num_workers=2, worker_init_fn=np.random.seed(2020))
    train_dataloader_PM25_for_test = DataLoader(train_dataset_PM25, batch_size=batch_size, shuffle=False)
    test_dataloader_PM25 = DataLoader(test_dataset_PM25, batch_size=batch_size, shuffle=False)
    print(len(train_dataset_PM25), len(test_dataset_PM25))
    if requires_meteo:
        return train_dataloader_PM25, train_dataloader_PM25_for_test, test_dataloader_PM25, train_dataset_PM25[0][1].shape[0]
    else:
        return train_dataloader_PM25, train_dataloader_PM25_for_test, test_dataloader_PM25

# Initialize the data loader for CNN models with weighted MSE loss
def initializeSortedCNNdata(root_dir, img_transform, batch_size, holdout=None, train_stations=-1, requires_meteo=False, rt_model=None, rf_train=None, rf_test=None):
    if requires_meteo:
        if (rt_model is None) or (rf_train is None) or (rf_test is None):
            raise Exception("Must specify rt_model, rf_train and rf_test.")
        train_dataset_PM25 = MyPM25DatasetSorted(root_dir=root_dir, img_transform=img_transform, mode='train', holdout=holdout, 
                                                 train_stations=train_stations, requires_meteo=requires_meteo, meteo_model=rt_model, rf_train=rf_train)
        test_dataset_PM25 = MyPM25DatasetSorted(root_dir=root_dir, img_transform=img_transform, mode='test', holdout=holdout, 
                                                train_stations=train_stations, requires_meteo=requires_meteo, meteo_model=rt_model, rf_test=rf_test)
    else:
        train_dataset_PM25 = MyPM25DatasetSorted(root_dir=root_dir, img_transform=img_transform, mode='train', 
                                                 holdout=holdout, train_stations=train_stations, requires_meteo=requires_meteo)
        test_dataset_PM25 = MyPM25DatasetSorted(root_dir=root_dir, img_transform=img_transform, mode='test', 
                                                holdout=holdout, train_stations=train_stations, requires_meteo=requires_meteo)
    train_dataloader_PM25 = DataLoader(train_dataset_PM25, batch_size=batch_size, shuffle=True, num_workers=2, worker_init_fn=np.random.seed(2020))
    train_dataloader_PM25_for_test = DataLoader(train_dataset_PM25, batch_size=batch_size, shuffle=False)
    test_dataloader_PM25 = DataLoader(test_dataset_PM25, batch_size=batch_size, shuffle=False)
    print(len(train_dataset_PM25), len(test_dataset_PM25))
    if requires_meteo:
        return train_dataloader_PM25, train_dataloader_PM25_for_test, test_dataloader_PM25, train_dataset_PM25[0][1][0].shape[0]
    else:
        return train_dataloader_PM25, train_dataloader_PM25_for_test, test_dataloader_PM25

# Get the all the station names for testing only
def getTestStations(root_dir, holdout):
    with open(root_dir, "rb") as fp:
        test_stations = []
        for data_point in pkl.load(fp):
            if data_point['Station_index'] in holdout:
                test_stations.append(data_point['Station_index'])
    return test_stations

# Get all the station names
def getAllStations(root_dir):
    with open(root_dir, "rb") as fp:
        stations = []
        for data_point in pkl.load(fp):
            if data_point['Station_index'] not in stations:
                stations.append(data_point['Station_index'])
    return stations

# Calculate spatial Pearson R and RMSE of all stations for testing
def calculateSpatial(y_test_pred, y_test, test_stations):
    df = pd.DataFrame({'y_test_pred': y_test_pred, 'y_test': y_test, 'test_stations': test_stations}).groupby(['test_stations']).mean()
    test_station_avg_pred = np.array(df.y_test_pred)
    test_station_avg = np.array(df.y_test)
    _, _, Rsquared_pearson, _ = eval_stat(test_station_avg_pred, test_station_avg)
    rmse = np.sqrt(metrics.mean_squared_error(test_station_avg, test_station_avg_pred))
    return Rsquared_pearson, rmse, test_station_avg_pred, test_station_avg

# Create all applicable plots
def plot_all(current_epochs, encoder_name, fig_size, loss_train, loss_test, y_train_pred, y_train, y_test_pred, y_test, 
             station_avg_pred, station_avg, spatial_R, spatial_R_test=None, spatial_rmse_test=None, train_stations=-1):
    # Plot and save the train and test loss over epochs
    plt.clf()
    fig = plt.plot(figsize=(16, 16))
    ax = plt.gca()
    epochs = range(current_epochs)
    ax.plot(epochs, loss_train, color='b', linewidth=0.5, label='Train loss')
    ax.plot(epochs, loss_test, color='r', linewidth=0.5, label='Test loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Train and test loss of predicting ground-level PM2.5')
    ax.legend()
    if train_stations > 0:
        savepdf(ax.figure, 'PM2.5_train_test_loss_' + encoder_name + '_self_supervision_train_stations_' + str(train_stations))
    else:
        savepdf(ax.figure, 'PM2.5_train_test_loss_' + encoder_name + '_self_supervision')
    del fig, ax
    
    # Plot the spatial R and RMSE if applicable
    if spatial_R_test:
        plt.clf()
        fig = plt.plot(figsize=(16, 16))
        ax = plt.gca()
        epochs = range(current_epochs)
        ax.plot(epochs, spatial_R_test, color='r', linewidth=0.5, label='Test spatial R')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Spatial R')
        ax.set_title('Test spatial R of predicting ground-level PM2.5')
        ax.legend()
        if train_stations > 0:
            savepdf(ax.figure, 'PM2.5_test_spatial_R_' + encoder_name + '_self_supervision_train_stations_' + str(train_stations))
        else:
            savepdf(ax.figure, 'PM2.5_test_spatial_R_' + encoder_name + '_self_supervision')
        del fig, ax
    
    if spatial_rmse_test:
        plt.clf()
        fig = plt.plot(figsize=(16, 16))
        ax = plt.gca()
        epochs = range(current_epochs)
        ax.plot(epochs, spatial_rmse_test, color='r', linewidth=0.5, label='Test spatial RMSE')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Spatial RMSE')
        ax.set_title('Test spatial RMSE of predicting ground-level PM2.5')
        ax.legend()
        if train_stations > 0:
            savepdf(ax.figure, 'PM2.5_test_spatial_RMSE_' + encoder_name + '_self_supervision_train_stations_' + str(train_stations))
        else:
            savepdf(ax.figure, 'PM2.5_test_spatial_RMSE_' + encoder_name + '_self_supervision')
        del fig, ax
    
    # Plot and save the train set predictions
    y_train_pred, y_train = np.squeeze(y_train_pred), np.squeeze(y_train)
    Rsquared, pvalue, Rsquared_pearson, pvalue_pearson = eval_stat(y_train_pred, y_train)
    if train_stations > 0:
        plot_result(y_train_pred, y_train, Rsquared, pvalue, Rsquared_pearson, pvalue_pearson, plot_label='train', save=True, 
                    fig_name='PM2.5_self_supervision_train_' + encoder_name + '_train_stations_' + str(train_stations), lower_bound=0, upper_bound=fig_size)
    else:
        plot_result(y_train_pred, y_train, Rsquared, pvalue, Rsquared_pearson, pvalue_pearson, plot_label='train', save=True, 
                    fig_name='PM2.5_self_supervision_train_' + encoder_name, lower_bound=0, upper_bound=fig_size)
    
    # Plot and save the test set predictions
    y_test_pred, y_test = np.squeeze(y_test_pred), np.squeeze(y_test)
    Rsquared, pvalue, Rsquared_pearson, pvalue_pearson = eval_stat(y_test_pred, y_test)
    if train_stations > 0:
        plot_result(y_test_pred, y_test, Rsquared, pvalue, Rsquared_pearson, pvalue_pearson, plot_label='test', save=True, 
                    fig_name='PM2.5_self_supervision_test_' + encoder_name + '_train_stations_' + str(train_stations), lower_bound=0, upper_bound=fig_size, spatial_R=spatial_R)
    else:
        plot_result(y_test_pred, y_test, Rsquared, pvalue, Rsquared_pearson, pvalue_pearson, plot_label='test', save=True, 
                    fig_name='PM2.5_self_supervision_test_' + encoder_name, lower_bound=0, upper_bound=fig_size, spatial_R=spatial_R)
    
    # Plot and save the spatial R predictions
    if train_stations > 0:
        spatialRPlot('dodgerblue', station_avg, station_avg_pred, plot_label='test', save=True, 
                     fig_name='PM2.5_self_supervision_test_spatial_R_' + encoder_name + '_train_stations_' + str(train_stations))
    else:
        spatialRPlot('dodgerblue', station_avg, station_avg_pred, plot_label='test', save=True, 
                     fig_name='PM2.5_self_supervision_test_spatial_R_' + encoder_name)