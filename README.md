# Contrastive_Learning_for_PM25_Prediction
This repository presents a contrastive learning framework for predicting ground-level PM2.5 from high-resolution micro-satellite imagery. Two contrastive learning frameworks, SimCLR and SimSiam, are tested and then extended to formulate a new framework called Spatiotemporal Contrastive Learning (SCL). The satellite imagery and PM2.5 data is obtained from 2 cities: Delhi and Beijing. The original framework is implemented in Pytorch Lightning Bolts repository: https://github.com/PyTorchLightning/lightning-bolts. 

## Structure of the Repository
The structure of this repository is given below:
- `contrastive_learning`: This module contains scripts for unsupervised pre-training with unlabeled satellite images by using regular contrastive learning (SimCLR and SimSiam) frameworks and SCL frameworks.
- `contrastive_models`: This module contains the backbone architecture of original SimCLR and SimSiam frameworks as well as the corresponding data augmentation functions.
- `data`: This module should contain the satellite imagery, PM2.5 data and pre-trained weights for both contrastive and supervised learning tasks.
- `model_utils`: This module contains all the utility functions for both contrastive and supervised learning tasks.
- `rt_rf_checkpoint`: This module contains the pre-trained parameters for Random Forest and Totally Random Trees Embedding models, which are used for predicting PM2.5 with multi-modal data (e.g. satellite images, meteorology, AOD, etc.). This is considered as future work and is beyond the scope of this paper.
- `supervised_learning`: This module contains scripts for training and testing the pre-trained model with satellite images and corresponding PM2.5 labels. 

## Related Work
Contrastive Learning
- T. Chen, S. Kornblith, M. Norouzi, et al., "A simple framework for contrastive learning of visual representations," in International conference on machine learning, PMLR, 2020, pp. 1597-1607.
- T. Chen, S. Kornblith, K. Swersky, et al., "Big self-supervised models are strong semi-supervised learners," arXiv preprint arXiv:2006.10029, 2020. https://github.com/google-research/simclr
- X. Chen and K. He, "Exploring simple siamese representation learning," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 15750-15758. https://github.com/facebookresearch/simsiam
- W. Falcon et al., Pytorch lightning, https://github.com/PyTorchLightning/pytorch-lightning, 2019.

PM2.5 Prediction from Satellite Imagery
- T. Zheng, M. H. Bergin, S. Hu, et al., "Estimating ground-level pm2.5 using micro-satellite images by a convolutional neural network and random forest approach," Atmospheric Environment, vol. 230, 2020. doi: 10.1016/j.atmosenv.2020.117451.
- T. Zheng, M. Bergin, G. Wang, et al., "Local pm2.5 hotspot detector at 300 m resolution: A random forest-convolutional neural network joint model jointly trained on satellite images and meteorology," Remote Sensing, vol. 13, 2021. doi: 10.3390/rs13071356.

## Conducting Experiments
Before conducting experiments, please first prepare the data in the format as specified in the file `requirements.md` under the `data` directory, or contact ziyang.jiang@duke.edu if you want to replicate the experiments in the paper. Next, run the scripts named in the `xxx_pretrain_SimCLR_xxx` or `xxx_pretrain_SimSiam_xxx` format under the `contrastive learning` directory. After that, the pre-trained model weights will be saved to the `data` directory. Finally, run the scripts named in the `xxx_supervised_xxx` format under the `supervised_learning` directory. This will generate the visualizations of PM2.5 predictions and the corresponding statistics.
