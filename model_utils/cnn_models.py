import torch
from torch import nn
import torchvision.models as models
for _ in range(2):
    try:
        from pl_bolts.models.self_supervised.resnets import resnet50 as resnet50_simclr_simsiam
        from pl_bolts.models.self_supervised.swav.swav_resnet import resnet50 as resnet50_swav
    except Exception as err:
        pass

# CNN model with ResNet50 architecture pretrained on ImageNet without meteorology
class ResNet50_ImageNet_pretrained_no_meteo(nn.Module):
    def __init__(self):
        super(ResNet50_ImageNet_pretrained_no_meteo, self).__init__()
        
        # Load the pretrained weights on ImageNet
        resnet50 = models.resnet50(pretrained=True)
        
        # Freeze all feature extraction layers in the encoder
        for param in resnet50.parameters():
            param.requires_grad = False
        
        # Model initialization
        self.resnet_pretrained = resnet50
        self.fc1 = nn.Linear(self.resnet_pretrained.fc.out_features, 256)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
    
    
    def forward(self, image):
        img_features = self.resnet_pretrained(image)
        img_features = torch.flatten(img_features, 1)
        img_features = self.fc1(img_features)
        x = self.relu(img_features)
        x = self.dropout(x)
        x = self.fc2(x.float())
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# CNN model with ResNet50 architecture pretrained on ImageNet with meteorology
class ResNet50_ImageNet_pretrained_joint_meteo(nn.Module):
    def __init__(self, transformed_meteo_size):
        super(ResNet50_ImageNet_pretrained_joint_meteo, self).__init__()
        
        # Load the pretrained weights on ImageNet
        resnet50 = models.resnet50(pretrained=True)
        
        # Freeze all feature extraction layers in the encoder
        for param in resnet50.parameters():
            param.requires_grad = False
        
        # Model initialization
        self.resnet_pretrained = resnet50
        self.fc1 = nn.Linear(self.resnet_pretrained.fc.out_features+transformed_meteo_size, 512)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
    
    
    def forward(self, image, transformed_meteo_features):
        img_features = self.resnet_pretrained(image)
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

# CNN model with ResNet50 architecture pretrained with SimCLR/SimSiam framework without meteorology
class ResNet50_SimCLR_SimSiam_no_meteo(nn.Module):
    def __init__(self, ssl_path):
        super(ResNet50_SimCLR_SimSiam_no_meteo, self).__init__()
        
        # Load the pretrained weights using spatiotemporal contrastive learning
        model_resnet50 = resnet50_simclr_simsiam(
            first_conv=True, 
            maxpool1=True, 
            pretrained=False, 
            return_all_feature_maps=False
        )
        checkpoint_resnet50 = torch.load(ssl_path)
        model_resnet50.load_state_dict(checkpoint_resnet50)
        
        # Freeze all feature extraction layers in the encoder
        for param in model_resnet50.parameters():
            param.requires_grad = False
        
        # Model initialization
        self.resnet_pretrained = model_resnet50
        in_features = 2048
        self.fc1 = nn.Linear(in_features, 256)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
    
    
    def forward(self, image):
        img_features = self.resnet_pretrained(image)[0]
        img_features = self.fc1(img_features)
        x = self.relu(img_features)
        x = self.dropout(x)
        x = self.fc2(x.float())
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# CNN model with ResNet50 architecture pretrained with SimCLR/SimSiam framework with meteorology
class ResNet50_SimCLR_SimSiam_joint_meteo(nn.Module):
    def __init__(self, ssl_path, transformed_meteo_size):
        super(ResNet50_SimCLR_SimSiam_joint_meteo, self).__init__()
        
        # Load the pretrained weights using spatiotemporal contrastive learning
        model_resnet50 = resnet50_simclr_simsiam(
            first_conv=True, 
            maxpool1=True, 
            pretrained=False, 
            return_all_feature_maps=False
        )
        checkpoint_resnet50 = torch.load(ssl_path)
        model_resnet50.load_state_dict(checkpoint_resnet50)
        
        # Freeze all feature extraction layers in the encoder
        for param in model_resnet50.parameters():
            param.requires_grad = False
        
        # Model initialization
        self.resnet_pretrained = model_resnet50
        in_features = 2048
        self.fc1 = nn.Linear(in_features+transformed_meteo_size, 512)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
    
    
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