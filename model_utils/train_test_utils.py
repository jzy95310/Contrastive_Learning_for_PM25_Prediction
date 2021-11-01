import torch
# import numpy as np   ## Used for debugging
# import matplotlib.pyplot as plt   ## Used for debugging
from torch import nn
import copy
from supervised_utils import calculateSpatial

# Train function with regular MSE loss
def train_with_regular_loss(model, device, train_loader, criterion, optimizer, epoch, requires_meteo=False, scale_factor=-1, 
                            residual_factor=1):       
#     model.eval()
#     for m in model.modules():
#         if isinstance(m, torch.nn.Dropout):
#             m.train()
    y_pred = torch.empty(0).to(device)
    y_true = torch.empty(0).to(device)
    if requires_meteo:
        for batch_idx, (img, meteo, target, target_pred) in enumerate(train_loader):
            img, meteo, target, target_pred = img.to(device), meteo.to(device), torch.squeeze(target.to(device)), torch.squeeze(target_pred.to(device))
            optimizer.zero_grad()
            output = torch.squeeze(model(img, meteo))
            if len(output.shape) == 0:
                continue
            prediction = residual_factor * output + scale_factor * target_pred.float()
            if scale_factor != -1:
                loss = criterion(prediction, target.float())   # residue = y - y_pred_rf
            else:
                loss = criterion(output + target_pred.float(), target.float())
            loss.backward()
            optimizer.step()
            if scale_factor != -1:
                y_pred = torch.cat((y_pred, residual_factor * output + scale_factor * target_pred.float()))
            else:
                y_pred = torch.cat((y_pred, output + target_pred.float()))
            y_true = torch.cat((y_true, target))
    else:
        for batch_idx, (img, target) in enumerate(train_loader):
            img, target = img.to(device), torch.squeeze(target.to(device))
            optimizer.zero_grad()
            output = torch.squeeze(model(img))
            if len(output.shape) == 0:
                continue
            loss = criterion(output, target.float())
            loss.backward()
            optimizer.step()
            y_pred = torch.cat((y_pred, output))
            y_true = torch.cat((y_true, target))
        
    train_loss = criterion(y_pred, y_true)
    print('Train Epoch: {} Loss: {:.6f}'.format(epoch, train_loss))

# Test function with regular MSE loss
def test_with_regular_loss(model, device, test_loader, criterion, use_train=False, requires_meteo=False, scale_factor=-1,
                           residual_factor=1, test_stations=None, scaler=None):  
    model.eval()
    y_pred = torch.empty(0).to(device)
    y_true = torch.empty(0).to(device)
    if requires_meteo:
        with torch.no_grad():
            for img, meteo, target, target_pred in test_loader:
                img, meteo, target, target_pred = img.to(device), meteo.to(device), torch.squeeze(target.to(device)), torch.squeeze(target_pred.to(device))
                output = torch.squeeze(model(img, meteo))
                if len(output.shape) == 0:
                    continue
                if scale_factor != -1:
                    y_pred = torch.cat((y_pred, residual_factor * output + scale_factor * target_pred))   # output is the predicted residue, y_pred is the predicted PM2.5
                else:
                    y_pred = torch.cat((y_pred, output + target_pred))
                y_true = torch.cat((y_true, target))   # y_true is the true PM2.5
    else:
        with torch.no_grad():
            for img, target in test_loader:
                img, target = img.to(device), torch.squeeze(target.to(device))
                output = torch.squeeze(model(img))
                if len(output.shape) == 0:
                    continue
                y_pred = torch.cat((y_pred, output))
                y_true = torch.cat((y_true, target))
    
    test_loss = criterion(y_pred, y_true)
    if test_stations:
        if scaler is not None:
            spatial_R, spatial_rmse, _, _ = calculateSpatial(scaler.inverse_transform(y_pred.cpu().numpy()), scaler.inverse_transform(y_true.cpu().numpy()), test_stations)
        else:
            spatial_R, spatial_rmse, _, _ = calculateSpatial(y_pred.cpu().numpy(), y_true.cpu().numpy(), test_stations)
    
    if use_train:
        print('Train set Loss: {:.4f}'.format(test_loss))
    else:
        print('Test set Loss: {:.4f}'.format(test_loss))
        if test_stations:
            print('Test spatial RMSE: {:.4f}'.format(spatial_rmse))
    
    if test_stations:
        return y_pred, y_true, test_loss, spatial_R, spatial_rmse
    else:
        return y_pred, y_true, test_loss

# Run training and testing with regular MSE loss
def run_with_regular_loss(model, optimizer, device, train_loader, train_loader_for_test, test_loader, encoder_name, max_epochs=500, save_model=False, lr_scheduler=None, 
                          early_stopping_threshold=-1, early_stopping_metric='test_loss', requires_meteo=False, scale_factor=-1, residual_factor=1, test_stations=None, scaler=None):
    assert early_stopping_metric in [None, 'test_loss', 'spatial_r', 'spatial_rmse'], "Early stopping metric should be one of the [test_loss, spatial_r, spatial_rmse]."
    
    criterion_train = nn.MSELoss(reduction='mean')
    criterion_test = nn.MSELoss(reduction='mean')
    
    y_train_pred_final, y_test_pred_final = torch.empty(0), torch.empty(0)
    y_train_final, y_test_final = torch.empty(0), torch.empty(0)
    loss_train_arr, loss_test_arr = [], []
    spatial_R_test_arr, spatial_rmse_test_arr = [], []
    loss_test_smallest, spatial_rmse_test_smallest, spatial_R_test_largest, early_stopping_count = 1e9, 1e9, 0, 0
    current_epochs = max_epochs + 0
    for epoch in range(1, max_epochs + 1):
        train_with_regular_loss(model, device, train_loader, criterion_train, optimizer, epoch, requires_meteo=requires_meteo, scale_factor=scale_factor, residual_factor=residual_factor)
        if lr_scheduler is not None:
            lr_scheduler.step()
        y_train_pred, y_train, loss_train = test_with_regular_loss(model, device, train_loader_for_test, criterion_test, use_train=True, requires_meteo=requires_meteo, scale_factor=scale_factor, residual_factor=residual_factor)
        y_test_pred, y_test, loss_test, spatial_R_test, spatial_rmse_test = test_with_regular_loss(model, device, test_loader, criterion_test, use_train=False, requires_meteo=requires_meteo, scale_factor=scale_factor, residual_factor=residual_factor, test_stations=test_stations, scaler=scaler)
        loss_train_arr.append(loss_train)
        loss_test_arr.append(loss_test)
        spatial_rmse_test_arr.append(spatial_rmse_test)
        spatial_R_test_arr.append(spatial_R_test)
        if early_stopping_metric == 'test_loss':
            if loss_test < loss_test_smallest:
                early_stopping_count = 0
                loss_test_smallest = loss_test + 0
                y_train_pred_final = copy.copy(y_train_pred)
                y_test_pred_final = copy.copy(y_test_pred)
                y_train_final = copy.copy(y_train)
                y_test_final = copy.copy(y_test)
            else:
                early_stopping_count += 1
                if early_stopping_threshold > 0 and early_stopping_count > early_stopping_threshold:
                    current_epochs = epoch + 0
                    print('Early stopping criterion reached. Exiting...')
                    break
        elif early_stopping_metric == 'spatial_rmse':
            if spatial_rmse_test < spatial_rmse_test_smallest:
                early_stopping_count = 0
                spatial_rmse_test_smallest = spatial_rmse_test + 0
                y_train_pred_final = copy.copy(y_train_pred)
                y_test_pred_final = copy.copy(y_test_pred)
                y_train_final = copy.copy(y_train)
                y_test_final = copy.copy(y_test)
            else:
                early_stopping_count += 1
                if early_stopping_threshold > 0 and early_stopping_count > early_stopping_threshold:
                    current_epochs = epoch + 0
                    print('Early stopping criterion reached. Exiting...')
                    break
        elif early_stopping_metric == 'spatial_r':
            if spatial_R_test > spatial_R_test_largest:
                early_stopping_count = 0
                spatial_R_test_largest = spatial_R_test + 0
                y_train_pred_final = copy.copy(y_train_pred)
                y_test_pred_final = copy.copy(y_test_pred)
                y_train_final = copy.copy(y_train)
                y_test_final = copy.copy(y_test)
            else:
                early_stopping_count += 1
                if early_stopping_threshold > 0 and early_stopping_count > early_stopping_threshold:
                    current_epochs = epoch + 0
                    print('Early stopping criterion reached. Exiting...')
                    break
        else:
            y_train_pred_final = copy.copy(y_train_pred)
            y_test_pred_final = copy.copy(y_test_pred)
            y_train_final = copy.copy(y_train)
            y_test_final = copy.copy(y_test)
    
    if (save_model):
        if requires_meteo:
            torch.save(model.state_dict(), '/work/zj63/Contrastive_learning_for_PM25_prediction/model_checkpoint/pipeline_params_' + encoder_name + '_meteo.pkl')
        else:
            torch.save(model.state_dict(), '/work/zj63/Contrastive_learning_for_PM25_prediction/model_checkpoint/pipeline_params_' + encoder_name + '_no_meteo.pkl')
    
    if test_stations:
        return y_train_pred_final.cpu().numpy(), y_train_final.cpu().numpy(), y_test_pred_final.cpu().numpy(), y_test_final.cpu().numpy(), loss_train_arr, loss_test_arr, spatial_R_test_arr, spatial_rmse_test_arr, current_epochs
    else:
        return y_train_pred_final.cpu().numpy(), y_train_final.cpu().numpy(), y_test_pred_final.cpu().numpy(), y_test_final.cpu().numpy(), loss_train_arr, loss_test_arr, current_epochs

# Train function with weighted MSE loss
def train_with_weighted_loss(model, device, train_loader, criterion, optimizer, epoch, requires_meteo=False, scale_factor=-1, 
                             residual_factor=1, spatial_factor=0):       
#     model.eval()
#     for m in model.modules():
#         if isinstance(m, torch.nn.Dropout):
#             m.train()
    y_pred = torch.empty(0).to(device)
    y_true = torch.empty(0).to(device)
    if requires_meteo:
        for batch_idx, (img, meteo, target, target_pred) in enumerate(train_loader):
            img, meteo = torch.squeeze(torch.stack(img)).to(device), torch.squeeze(torch.tensor(meteo)).to(device)
            target, target_pred = torch.squeeze(torch.tensor(target)).to(device), torch.squeeze(torch.tensor(target_pred)).to(device)
            if (len(img.shape) < 4):
                img, meteo = torch.unsqueeze(img, 0), torch.unsqueeze(meteo, 0)
                target, target_pred = torch.unsqueeze(target, 0), torch.unsqueeze(target_pred, 0)
            optimizer.zero_grad()
            output = torch.squeeze(model(img, meteo))
            if len(output.shape) == 0:
                continue
            if scale_factor != -1:
                prediction = residual_factor * output + scale_factor * target_pred.float()    # residue = y - scale * y_pred_rf
                loss_ind = criterion(prediction, target.float())   
                loss_spatial = criterion(prediction - torch.mean(prediction), target.float() - torch.mean(target.float()))
                loss = loss_ind + spatial_factor * loss_spatial
                loss.backward()
            else:
                loss = criterion(output + target_pred.float(), target.float())
                loss.backward()
            optimizer.step()
            if scale_factor != -1:
                y_pred = torch.cat((y_pred, prediction))
            else:
                y_pred = torch.cat((y_pred, output + target_pred.float()))
            y_true = torch.cat((y_true, target))
    else:
        for batch_idx, (img, target) in enumerate(train_loader):
            img = torch.squeeze(torch.stack(img)).to(device)
            target = torch.squeeze(torch.tensor(target)).to(device)
            if (len(img.shape) < 4):
                img = torch.unsqueeze(img, 0)
                target = torch.unsqueeze(target, 0)
            optimizer.zero_grad()
            output = torch.squeeze(model(img))
            if (len(output.shape) == 0):
                continue
            loss_ind = criterion(output, target.float())
            loss_spatial = criterion(output - torch.mean(output), target.float() - torch.mean(target.float()))
            loss = loss_ind + spatial_factor * loss_spatial
            loss.backward()
            optimizer.step()
            y_pred = torch.cat((y_pred, output))
            y_true = torch.cat((y_true, target))
        
    train_loss = criterion(y_pred, y_true)
    print('Train Epoch: {} Loss: {:.6f}'.format(epoch, train_loss))

# Test function with weighted MSE loss
def test_with_weighted_loss(model, device, test_loader, criterion, use_train=False, requires_meteo=False, scale_factor=-1, 
                            residual_factor=1, test_stations=None):  
    model.eval()
    y_pred = torch.empty(0).to(device)
    y_true = torch.empty(0).to(device)
    if requires_meteo:
        with torch.no_grad():
            for img, meteo, target, target_pred in test_loader:
                img, meteo = torch.squeeze(torch.stack(img)).to(device), torch.squeeze(torch.tensor(meteo)).to(device)
                target, target_pred = torch.squeeze(torch.tensor(target)).to(device), torch.squeeze(torch.tensor(target_pred)).to(device)
                if (len(img.shape) < 4):
                    img, meteo = torch.unsqueeze(img, 0), torch.unsqueeze(meteo, 0)
                    target, target_pred = torch.unsqueeze(target, 0), torch.unsqueeze(target_pred, 0)
                output = torch.squeeze(model(img, meteo))
                if (len(output.shape) == 0):
                    output = torch.unsqueeze(output, 0)
                if scale_factor != -1:
                    y_pred = torch.cat((y_pred, residual_factor * output + scale_factor * target_pred))   # output is the predicted residue, y_pred is the predicted PM2.5
                else:
                    y_pred = torch.cat((y_pred, output + target_pred))
                y_true = torch.cat((y_true, target))   # y_true is the true PM2.5

    else:
        with torch.no_grad():
            for img, target in test_loader:
                img, target = torch.squeeze(torch.stack(img)).to(device), torch.squeeze(torch.tensor(target)).to(device)
                if (len(img.shape) < 4):
                    img = torch.unsqueeze(img, 0)
                    target = torch.unsqueeze(target, 0)
                output = torch.squeeze(model(img))
                if (len(output.shape) == 0):
                    output = torch.unsqueeze(output, 0)
                y_pred = torch.cat((y_pred, output))
                y_true = torch.cat((y_true, target))
    
    test_loss = criterion(y_pred, y_true)
    if test_stations:
        if scaler is not None:
            spatial_R, spatial_rmse, _, _ = calculateSpatial(scaler.inverse_transform(y_pred.cpu().numpy()), scaler.inverse_transform(y_true.cpu().numpy()), test_stations)
        else:
            spatial_R, spatial_rmse, _, _ = calculateSpatial(y_pred.cpu().numpy(), y_true.cpu().numpy(), test_stations)
    
    if use_train:
        print('Train set Loss: {:.4f}'.format(test_loss))
    else:
        print('Test set Loss: {:.4f}'.format(test_loss))
        if test_stations:
            print('Test spatial RMSE: {:.4f}'.format(spatial_rmse))
    
    if test_stations:
        return y_pred, y_true, test_loss, spatial_R, spatial_rmse
    else:
        return y_pred, y_true, test_loss

# Run training and testing with weighted MSE loss
def run_with_weighted_loss(model, optimizer, device, train_loader, train_loader_for_test, test_loader, encoder_name, max_epochs=500, save_model=False, lr_scheduler=None, 
                           early_stopping_threshold=-1, early_stopping_metric='test_loss', requires_meteo=False, scale_factor=-1, residual_factor=1, spatial_factor=0, test_stations=None, scaler=None):
    assert early_stopping_metric in [None, 'test_loss', 'spatial_r', 'spatial_rmse'], "Early stopping metric should be one of the [test_loss, spatial_r, spatial_rmse]."
    
    criterion_train = nn.MSELoss(reduction='mean')
    criterion_test = nn.MSELoss(reduction='mean')
    
    y_train_pred_final, y_test_pred_final = torch.empty(0), torch.empty(0)
    y_train_final, y_test_final = torch.empty(0), torch.empty(0)
    loss_train_arr, loss_test_arr = [], []
    spatial_R_test_arr, spatial_rmse_test_arr = [], []
    loss_test_smallest, spatial_rmse_test_smallest, spatial_R_test_largest, early_stopping_count = 1e9, 1e9, 0, 0
    current_epochs = max_epochs + 0
    for epoch in range(1, max_epochs + 1):
        train_with_weighted_loss(model, device, train_loader, criterion_train, optimizer, epoch, requires_meteo=requires_meteo, scale_factor=scale_factor, residual_factor=residual_factor, spatial_factor=spatial_factor)
        if lr_scheduler is not None:
            lr_scheduler.step()
        y_train_pred, y_train, loss_train = test_with_weighted_loss(model, device, train_loader_for_test, criterion_test, use_train=True, requires_meteo=requires_meteo, scale_factor=scale_factor, residual_factor=residual_factor)
        y_test_pred, y_test, loss_test, spatial_R_test, spatial_rmse_test = test_with_weighted_loss(model, device, test_loader, criterion_test, use_train=False, requires_meteo=requires_meteo, scale_factor=scale_factor, residual_factor=residual_factor, test_stations=test_stations, scaler=scaler)
        loss_train_arr.append(loss_train)
        loss_test_arr.append(loss_test)
        spatial_rmse_test_arr.append(spatial_rmse_test)
        spatial_R_test_arr.append(spatial_R_test)
        if early_stopping_metric == 'test_loss':
            if loss_test < loss_test_smallest:
                early_stopping_count = 0
                loss_test_smallest = loss_test + 0
                y_train_pred_final = copy.copy(y_train_pred)
                y_test_pred_final = copy.copy(y_test_pred)
                y_train_final = copy.copy(y_train)
                y_test_final = copy.copy(y_test)
            else:
                early_stopping_count += 1
                if early_stopping_threshold > 0 and early_stopping_count > early_stopping_threshold:
                    current_epochs = epoch + 0
                    print('Early stopping criterion reached. Exiting...')
                    break
        elif early_stopping_metric == 'spatial_rmse':
            if spatial_rmse_test < spatial_rmse_test_smallest:
                early_stopping_count = 0
                spatial_rmse_test_smallest = spatial_rmse_test + 0
                y_train_pred_final = copy.copy(y_train_pred)
                y_test_pred_final = copy.copy(y_test_pred)
                y_train_final = copy.copy(y_train)
                y_test_final = copy.copy(y_test)
            else:
                early_stopping_count += 1
                if early_stopping_threshold > 0 and early_stopping_count > early_stopping_threshold:
                    current_epochs = epoch + 0
                    print('Early stopping criterion reached. Exiting...')
                    break
        elif early_stopping_metric == 'spatial_r':
            if spatial_R_test > spatial_R_test_largest:
                early_stopping_count = 0
                spatial_R_test_largest = spatial_R_test + 0
                y_train_pred_final = copy.copy(y_train_pred)
                y_test_pred_final = copy.copy(y_test_pred)
                y_train_final = copy.copy(y_train)
                y_test_final = copy.copy(y_test)
            else:
                early_stopping_count += 1
                if early_stopping_threshold > 0 and early_stopping_count > early_stopping_threshold:
                    current_epochs = epoch + 0
                    print('Early stopping criterion reached. Exiting...')
                    break
        else:
            y_train_pred_final = copy.copy(y_train_pred)
            y_test_pred_final = copy.copy(y_test_pred)
            y_train_final = copy.copy(y_train)
            y_test_final = copy.copy(y_test)
    
    if (save_model):
        if requires_meteo:
            torch.save(model.state_dict(), '/work/zj63/Contrastive_learning_for_PM25_prediction/model_checkpoint/pipeline_params_' + encoder_name + '_meteo.pkl')
        else:
            torch.save(model.state_dict(), '/work/zj63/Contrastive_learning_for_PM25_prediction/model_checkpoint/pipeline_params_' + encoder_name + '_no_meteo.pkl')
    
    if test_stations:
        return y_train_pred_final.cpu().numpy(), y_train_final.cpu().numpy(), y_test_pred_final.cpu().numpy(), y_test_final.cpu().numpy(), loss_train_arr, loss_test_arr, spatial_R_test_arr, spatial_rmse_test_arr, current_epochs
    else:
        return y_train_pred_final.cpu().numpy(), y_train_final.cpu().numpy(), y_test_pred_final.cpu().numpy(), y_test_final.cpu().numpy(), loss_train_arr, loss_test_arr, current_epochs