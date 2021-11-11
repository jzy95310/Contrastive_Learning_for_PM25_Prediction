# Contrastive_Learning_for_PM25_Prediction
This repository presents a contrastive learning framework for predicting ground-level PM2.5 from high-resolution micro-satellite imagery. Two contrastive learning frameworks, SimCLR and SimSiam, are tested and then extended to formulate a new framework called Spatiotemporal Contrastive Learning (SCL). The satellite imagery and PM2.5 data is obtained from 2 cities: Delhi and Beijing. The original framework is implemented in Pytorch Lightning Bolts repository: https://github.com/PyTorchLightning/lightning-bolts. 

## Structure and Organization
The structure of this repository is given below:
- `contrastive_learning`: This module contains scripts for unsupervised pre-training with unlabeled satellite images by using regular contrastive learning (SimCLR and SimSiam) frameworks and SCL frameworks.
- `contrastive_models`: This module contains the backbone architecture of original SimCLR and SimSiam frameworks as well as the corresponding data augmentation functions.
- `data`: This module contains the satellite imagery, PM2.5 data and pre-trained weights for both contrastive and supervised learning tasks. Please contact ziyang.jiang@duke.edu for data if you want to replicate the experiments. If you want to run the experiment with your own data, please follow the require format given in this module.
- `model_utils`: This module contains all the utility functions for both contrastive and supervised learning tasks.
- `rt_rf_checkpoint`: This module contains the pre-trained parameters for Random Forest and Totally Random Trees Embedding models, which are used for predicting PM2.5 with multi-modal data (e.g. satellite images, meteorology, AOD, etc.). This is considered as future work and is beyond the scope of this paper.
- `supervised_learning`: This module contains scripts for training and testing the pre-trained model with satellite images and corresponding PM2.5 labels. 
