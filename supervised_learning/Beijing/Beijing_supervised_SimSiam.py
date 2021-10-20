import sys
import numpy as np
import os
import pickle as pkl
sys.path.insert(0, '../../model_utils')

import torch
import copy
from torch import optim
from torchvision import transforms
from sklearn import metrics

from cnn_models import ResNet_SimCLR_SimSiam_no_meteo, ResNet_SimCLR_SimSiam_joint_meteo
from supervised_utils import eval_stat, plot_result, calculateSpatial, spatialRPlot, plot_all
from supervised_utils import getTestStations
from supervised_utils import loadRTandRFModel, predictWithRF
from supervised_utils import initializeCNNdata
from train_test_utils import run_with_regular_loss

# To make this notebook's output stable across runs
np.random.seed(2020)
torch.manual_seed(2020)
torch.cuda.manual_seed(2020)
torch.cuda.manual_seed_all(2020)
torch.backends.cudnn.deterministic = True

def run_supervised_SimSiam(requires_meteo=False, train_stations=-1, lr=5e-7):
    root_dir = '../../data/Beijing_labeled.pkl'
    img_transform = transforms.ToTensor()
    holdout = ['25', '26', '27', '28', '29', '30', '31', '32', '33', '34']
    test_stations = getTestStations(root_dir, holdout=holdout)
    batch_size = 4
    fig_size = 500
    scale_factor = 0.95
    scaler = None
    
    # Build Random Trees Embedding and Random Forest Model
    if requires_meteo:
        rt_dir = '../../rt_rf_checkpoint/rt_model_Beijing.pkl'
        rf_dir = '../../rt_rf_checkpoint/ML_RF_singlemet_Beijing.pkl'
        rt_model, rf_model, meteo_transformed_train, PM_train, meteo_transformed_test, PM_test = loadRTandRFModel(root_dir, rt_dir, rf_dir, holdout)
        y_train_pred_rf, y_test_pred_rf = predictWithRF(rf_model, meteo_transformed_train, meteo_transformed_test)
    
    # Initialize the data for CNN
    if requires_meteo:
        train_loader, train_loader_for_test, test_loader, transformed_meteo_dim = initializeCNNdata(root_dir, img_transform, batch_size, holdout=holdout, 
                                                                                            train_stations=train_stations, requires_meteo=True, rt_model=rt_model, 
                                                                                            rf_train=y_train_pred_rf, rf_test=y_test_pred_rf)
    else:
        train_loader, train_loader_for_test, test_loader, scaler = initializeCNNdata(root_dir, img_transform, batch_size, 
                                                                     holdout=holdout, train_stations=train_stations, requires_meteo=False, normalized=False)
    # Visualize the Random Forest predictions
    if requires_meteo:
        Rsquared, pvalue, Rsquared_pearson, pvalue_pearson = eval_stat(y_train_pred_rf, PM_train)
        plot_result(y_train_pred_rf, PM_train, Rsquared, pvalue, Rsquared_pearson, pvalue_pearson, plot_label='train', save=True, 
                    fig_name='PM2.5_RF_train_train_stations_' + str(train_stations), lower_bound=0, upper_bound=fig_size)
        Rsquared, pvalue, Rsquared_pearson, pvalue_pearson = eval_stat(y_test_pred_rf, PM_test)
        spatial_R_rf, spatial_rmse_rf, station_avg_rf_pred, station_avg_rf = calculateSpatial(y_test_pred_rf, PM_test, test_stations)
        plot_result(y_test_pred_rf, PM_test, Rsquared, pvalue, Rsquared_pearson, pvalue_pearson, plot_label='test', save=True, 
                    fig_name='PM2.5_RF_test_train_stations_' + str(train_stations), lower_bound=0, upper_bound=fig_size, spatial_R=spatial_R_rf)
        spatialRPlot('dodgerblue', station_avg_rf, station_avg_rf_pred, plot_label='test', save=True, 
                     fig_name='PM2.5_RF_test_spatial_R_train_stations_' + str(train_stations))
        
    # Run supervised learning
    max_epochs = 500
    early_stopping_threshold = 20
    early_stopping_metric = 'spatial_rmse'
    encoder_name = 'resnet50_SimSiam'
    # ssl_path = '../../model_checkpoint/encoder_params_resnet18_spatiotemporal_Beijing_SimSiam.pkl'
    ssl_path = '../../model_checkpoint/encoder_params_resnet50_spatiotemporal_Beijing_SimSiam.pkl'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if requires_meteo:
        model = ResNet_SimCLR_SimSiam_joint_meteo(ssl_path, transformed_meteo_dim, backbone='resnet50').to(device)
    else:
        model = ResNet_SimCLR_SimSiam_no_meteo(ssl_path, backbone='resnet50').to(device)
        
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.6, weight_decay=0.1)
    # optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.01, 0.01), weight_decay=0.1)
    gamma = 0.005
    exp_func = lambda epoch: np.exp(-gamma*epoch)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=exp_func)
    if requires_meteo:
        y_train_pred, y_train, y_test_pred, y_test, loss_train, loss_test, spatial_R_test, spatial_rmse_test, current_epochs = run_with_regular_loss(
            model, optimizer, device, train_loader, train_loader_for_test, test_loader, 
            encoder_name=encoder_name, 
            max_epochs=max_epochs, 
            save_model=False, 
            lr_scheduler=scheduler, 
            early_stopping_threshold=early_stopping_threshold, 
            early_stopping_metric=early_stopping_metric, 
            requires_meteo=True, 
            scale_factor=scale_factor, 
            test_stations=test_stations
        )
    else:
        y_train_pred, y_train, y_test_pred, y_test, loss_train, loss_test, spatial_R_test, spatial_rmse_test, current_epochs = run_with_regular_loss(
            model, optimizer, device, train_loader, train_loader_for_test, test_loader, 
            encoder_name=encoder_name, 
            max_epochs=max_epochs, 
            save_model=False, 
            # lr_scheduler=scheduler, 
            early_stopping_threshold=early_stopping_threshold, 
            early_stopping_metric=early_stopping_metric, 
            requires_meteo=False, 
            test_stations=test_stations, 
            scaler=scaler
        )
    
    if scaler is not None:
        y_train_pred, y_train = scaler.inverse_transform(y_train_pred), scaler.inverse_transform(y_train)
        y_test_pred, y_test = scaler.inverse_transform(y_test_pred), scaler.inverse_transform(y_test)
    
    # Calculate spatial Pearson R
    spatial_R, spatial_rmse, station_avg_pred, station_avg = calculateSpatial(y_test_pred, y_test, test_stations)
    
    # Save spatial statistics
    result_stats = {'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)), 'Spatial_R': spatial_R, 'Spatial_RMSE': spatial_rmse}
    result_path = './model_results/results_SimSiam.pkl'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'ab') as fp:
        pkl.dump(result_stats, fp)
    
    # Visualize and save results
    if requires_meteo:
        plot_all(current_epochs, encoder_name, fig_size, loss_train, loss_test, y_train_pred, y_train, 
                 y_test_pred, y_test, station_avg_pred, station_avg, spatial_R, spatial_R_test, spatial_rmse_test, train_stations=train_stations, line_range=[0, 50])
    else:
        plot_all(current_epochs, encoder_name, fig_size, loss_train, loss_test, y_train_pred, y_train, 
                 y_test_pred, y_test, station_avg_pred, station_avg, spatial_R, train_stations=train_stations, line_range=[0, 100])

def cli_main():
    stations_num = [1, 5, 10, 15, 20, 25]
    lrs_no_meteo = [3e-6, 1.5e-6, 9e-7, 7e-7, 6e-7, 5e-7]   # ResNet50
    lrs_meteo = []   # ResNet50

    for i in range(len(stations_num)):
        # run_supervised_SimSiam(requires_meteo=True, train_stations=stations_num[i], lr=lrs_meteo[i])
        run_supervised_SimSiam(requires_meteo=False, train_stations=stations_num[i], lr=lrs_no_meteo[i])

if __name__ == '__main__':
    cli_main()