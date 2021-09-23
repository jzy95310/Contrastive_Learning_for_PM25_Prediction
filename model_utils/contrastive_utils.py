import numpy as np
import torch
import pickle as pkl
from tqdm import tqdm

from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors


# Image dataset used for contrastive learning (pre-training)
class MySSLImageDataset(Dataset):
    """
    Args:
        root_dir (string): Directory with all the images.
        mode ('train', 'val' or 'test', optional): Whether the dataset is for
            training or validation
        transform (callable, optional): Optional transform to be applied
            on a sample.
        batch_size (int, optional): Fake batch size for each monthly data. 
            In dataloader we will set batch_size = 1 since we want all data in one batch to share the same timestamp
        train_val_ratio (float, optional): Ratio of train and validation data
        
    """

    def __init__(self, root_dir, mode='train', transform=None, batch_size=10, train_val_ratio=-1):
        if mode not in ['train', 'val']:
            raise Exception('Mode must be either \'train\' or \'val\'')
        if mode == 'train' and train_val_ratio != -1:
            raise Exception('Should not set train_val_ratio in train mode')
        if mode == 'val' and (train_val_ratio > 1 or train_val_ratio <= 0):
            raise Exception('train_val_ratio must be in the range (0, 1]')
        
        # Pass in parameters
        self.mode = mode
        self.transform = transform
        self.batch_size = batch_size
        self.train_val_ratio = train_val_ratio
        
        # Private variables
        self.train_index, self.val_index = [], []
        self.train_images, self.val_images = [], []    # Store the intervals of images for each batch
        
        with open(root_dir, "rb") as fp:
            grids = pkl.load(fp)
        
        # Split the data into train and validation set based on TIMESTAMPS
        split_idx = int(self.train_val_ratio*len(grids)) if self.train_val_ratio != -1 else -1
        train_data_size, val_data_size = [], []
        for i in tqdm(range(len(grids)), position=0, leave=True):
            train_data_size.append(len(grids[i]))
            if split_idx == -1:
                for j in range(len(grids[i])):
                    self.train_images.append(grids[i][j])
            else:
                if i < split_idx:
                    for j in range(len(grids[i])):
                        self.train_images.append(grids[i][j])
                else:
                    val_data_size.append(len(grids[i]))
                    for j in range(len(grids[i])):
                        self.train_images.append(grids[i][j])
                        self.val_images.append(grids[i][j])
        
        # Slice the data based on batch_size
        self.train_index = self.divide_data(train_data_size)
        self.val_index = self.divide_data(val_data_size)
        
        # Remove unnecessary data
        if self.mode == 'train':
            del grids, self.val_images, self.val_index
        else:
            del grids, self.train_images, self.train_index

                
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_index)
        else:
            return len(self.val_index)

        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.mode == 'train':
            batch = self.train_images[self.train_index[idx][0]:self.train_index[idx][1]]
        else:
            batch = self.val_images[self.val_index[idx][0]:self.val_index[idx][1]]
        
        if self.transform:
            batch = self.transform(batch)
            
        return batch, idx
    
    
    def divide_data(self, data_size):
        res = []
        start_index, end_index = 0, 0
        for length in data_size:
            div = length // self.batch_size
            mod = length % self.batch_size
            for i in range(div):
                end_index += self.batch_size
                res.append([start_index, end_index])
                start_index = end_index + 0
                
            if mod != 0:
                end_index += mod
                if mod > 1:
                    # If there is just one image in current batch, then discard this image
                    # since this will cause problem in BatchNorm layer
                    res.append([start_index, end_index])
                start_index = end_index + 0
        return res

# Transform function used to find the spatiotemporal nearest neighbor of an image
class SpatiotemporalTransform(object):
    """
    Args:
        input_height (int): Cropping dimension of input images
    
    """
    def __init__(self, input_height):
        """
        Put extra data augmentation here.
        
        """
        self.input_height = input_height
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
#             transforms.RandomResizedCrop(size=self.input_height),
#             transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        
        
    def __call__(self, batch):
        coordinates = [[sample['lat'], sample['lon']] for sample in batch]
        if len(coordinates) > 1:
            nbrs = NearestNeighbors(n_neighbors=2, metric=self.distance).fit(coordinates)
            _, indices = nbrs.kneighbors(coordinates)
        else:
            indices = [[0, 0]]
        
        xi = [batch[idx[0]]['Image'] for idx in indices]
        xj = [batch[idx[1]]['Image'] for idx in indices]
        
        xi = torch.stack([self.transform(np.uint8(img)) for img in xi])
        xj = torch.stack([self.transform(np.uint8(img)) for img in xj])
        
        return xi, xj
    
    
    def distance(self, p1, p2):
        """
        Calculate the Euclidean distance (specified in decimal degrees)
        
        """
        lat1, lon1 = p1
        lat2, lon2 = p2
        
        return np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)