import torch
from torch import nn
import torchvision.models as models
import matplotlib.pyplot as plt
for _ in range(2):
    try:
        from pl_bolts.models.self_supervised.resnets import resnet18 as resnet18_simclr_simsiam
        from pl_bolts.models.self_supervised.resnets import resnet50 as resnet50_simclr_simsiam
    except Exception as err:
        pass

# CNN model with ResNet architecture pretrained on ImageNet without meteorology
class ResNet_ImageNet_pretrained_no_meteo(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, trainable=False):
        super(ResNet_ImageNet_pretrained_no_meteo, self).__init__()
        
        # Load the pretrained weights on ImageNet
        if backbone == 'resnet18':
            self.net = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet50':
            self.net = models.resnet50(pretrained=pretrained)
        elif backbone == 'alexnet':
            self.net = models.alexnet(pretrained=pretrained)
        else:
            raise Exception("The specified backbone is not defined.")
        
        # Freeze all feature extraction layers in the encoder
        for param in self.net.parameters():
            param.requires_grad = trainable
        
        # Model initialization
        self.fc1 = nn.Linear(1000, 512)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)
    
    
    def forward(self, image):
        img_features = self.net(image)
        img_features = torch.flatten(img_features, 1)
#         plt.imshow(img_features.cpu().detach().numpy())
#         plt.colorbar()
#         plt.show()
        img_features = self.fc1(img_features)
        x = self.relu(img_features)
        x = self.dropout(x)
        x = self.fc2(x.float())
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# CNN model with ResNet architecture pretrained on ImageNet with meteorology
class ResNet_ImageNet_pretrained_joint_meteo(nn.Module):
    def __init__(self, transformed_meteo_size, backbone='resnet18', pretrained=True, trainable=False):
        super(ResNet_ImageNet_pretrained_joint_meteo, self).__init__()
        
        # Load the pretrained weights on ImageNet
        if backbone == 'resnet18':
            self.net = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet50':
            self.net = models.resnet50(pretrained=pretrained)
        elif backbone == 'alexnet':
            self.net = models.alexnet(pretrained=pretrained)
        else:
            raise Exception("The specified backbone is not defined.")
        
        # Freeze all feature extraction layers in the encoder
        for param in self.net.parameters():
            param.requires_grad = trainable
        
        # Model initialization
        self.fc1 = nn.Linear(1000+transformed_meteo_size, 512)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)
    
    
    def forward(self, image, transformed_meteo_features):
        img_features = self.net(image)
        img_features = torch.flatten(img_features, 1)
        # Concatenate image representations with transformed meteo features
        x = torch.cat((img_features, transformed_meteo_features), dim=-1)
        x = self.fc1(x.float())
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x.float())
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# CNN model with ResNet architecture pretrained with SimCLR/SimSiam framework without meteorology
class ResNet_SimCLR_SimSiam_no_meteo(nn.Module):
    def __init__(self, ssl_path, backbone='resnet18', resnet_trainable=False):
        super(ResNet_SimCLR_SimSiam_no_meteo, self).__init__()
        
        # Load the pretrained weights using spatiotemporal contrastive learning
        if backbone == 'resnet18':
            resnet = resnet18_simclr_simsiam(
                first_conv=True, 
                maxpool1=True, 
                pretrained=False, 
                return_all_feature_maps=False
            )
        else:
            resnet = resnet50_simclr_simsiam(
                first_conv=True, 
                maxpool1=True, 
                pretrained=False, 
                return_all_feature_maps=False
            )
        checkpoint_resnet = torch.load(ssl_path)
        resnet.load_state_dict(checkpoint_resnet)
        
        # Freeze all feature extraction layers in the encoder
        for param in resnet.parameters():
            param.requires_grad = resnet_trainable
        
        # Model initialization
        self.resnet_pretrained = resnet
        if backbone == 'resnet18':
            in_features = 512
        else:
            in_features = 2048
        self.fc1 = nn.Linear(in_features, 512)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)
    
    
    def forward(self, image):
        img_features = self.resnet_pretrained(image)[0]
#         plt.imshow(img_features.cpu().detach().numpy())
#         plt.colorbar()
#         plt.show()
        img_features = self.fc1(img_features)
        x = self.relu(img_features)
        x = self.dropout(x)
        x = self.fc2(x.float())
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# CNN model with ResNet architecture pretrained with SimCLR/SimSiam framework with meteorology
class ResNet_SimCLR_SimSiam_joint_meteo(nn.Module):
    def __init__(self, ssl_path, transformed_meteo_size, backbone='resnet18', resnet_trainable=False):
        super(ResNet_SimCLR_SimSiam_joint_meteo, self).__init__()
        
        # Load the pretrained weights using spatiotemporal contrastive learning
        if backbone == 'resnet18':
            resnet = resnet18_simclr_simsiam(
                first_conv=True, 
                maxpool1=True, 
                pretrained=False, 
                return_all_feature_maps=False
            )
        else:
            resnet = resnet50_simclr_simsiam(
                first_conv=True, 
                maxpool1=True, 
                pretrained=False, 
                return_all_feature_maps=False
            )
        checkpoint_resnet = torch.load(ssl_path)
        resnet.load_state_dict(checkpoint_resnet)
        
        # Freeze all feature extraction layers in the encoder
        for param in resnet.parameters():
            param.requires_grad = resnet_trainable
        
        # Model initialization
        self.resnet_pretrained = resnet
        if backbone == 'resnet18':
            in_features = 512
        else:
            in_features = 2048
        self.fc1 = nn.Linear(in_features+transformed_meteo_size, 512)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)
    
    
    def forward(self, image, transformed_meteo_features):
        img_features = self.resnet_pretrained(image)[0]
        # Concatenate image representations with transformed meteo features
        x = torch.cat((img_features, transformed_meteo_features), dim=-1)
        x = self.fc1(x.float())
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# ########################################################################################
# MIT License

# Copyright (c) 2022 Ziyang Jiang

# Permission is hereby granted, free of charge, to any person obtaining a copy of this 
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, 
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
# permit persons to whom the Software is furnished to do so, subject to the following 
# conditions:

# The above copyright notice and this permission notice shall be included in all copies 
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE 
# OR OTHER DEALINGS IN THE SOFTWARE.
# ########################################################################################