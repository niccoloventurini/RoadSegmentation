import numpy as np
from preprocces import *
from torch.utils.data import Dataset
from helpers import *
from config import RoadSegmentationConfig
        
class RoadSegmentationDataset(Dataset):
    """ Road Segmentation dataset class

    Args:
        data_path (str): Path to the images
        gt_path (str): Path to the ground truth
        configuration (RoadSegmentationConfig): Configuration class
        train (bool): If True, the dataset is used for training
        device (str): Device to use, default is None
        ps (int): Patch size
    """
    def __init__(self, data_path, gt_path, configuration:RoadSegmentationConfig, train: bool, device = None, ps=PATCH_SIZE):
        self.train = train
        self.device = device
        imgs, gts = load_data(data_path,gt_path,train,device,configuration, ps) #load the data and the groundtruths
        divide = not configuration.norm #if we normalize the images, we don't divide by 255
        if gts is not None: 
            self.gt = [mask_to_tensor(gt,device) for gt in gts] #transform the groundtruths into tensors
        self.data = [image_into_tensor(img, device, divide = divide) for img in imgs] #transform the images into tensors

    def __len__(self):
        """ Returns the length of the dataset 
        
        Returns:
            int: Length of the dataset
        
        """
        return len(self.data) #return the length of the data

    def __getitem__(self, index):
        """ Returns the item at index
        
        Args:
            index (int): Index of the item to return

        Returns:
            (list[Tensor], list[Tensor]): 
        """
        if self.train:
            return self.data[index], self.gt[index] #return the data and the groundtruths
        else:
            return self.data[index] #return the data