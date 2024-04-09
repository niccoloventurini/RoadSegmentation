import numpy as np
import torch
from PIL import Image
# from preprocces import crop
from torchvision import transforms
import matplotlib.image as mpimg
from parameters import *
import os

#transform an image into a tensor, since when importing we are sure that we get np.arrays
def image_into_tensor(img:np.ndarray, device = None, divide = False):
    """ Transform an image into a tensor

    Args:
        img (np.ndarray): Image to transform
        device (str): Device to use, default is None
        divide (bool): If True, the image is divided by 255, the default is False

    Returns:
        tensor: Tensor of the image
    """
    img = np.transpose(img,[2,0,1]) #change the order of the dimensions
    tensor = torch.Tensor(img) #transform the image into a tensor
    if divide:
        tensor = tensor / 255 #divide by 255 to have values between 0 and 1
    tensor.unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device) #send the tensor to the device
    return tensor

#transform a mask into a tensor, since when importing we are sure that we get np.arrays
def mask_to_tensor(mask: np.ndarray, device = None):
    """ Transform a mask into a tensor

    Args:
        mask (np.ndarray): Mask to transform
        device (str): Device to use, default is None

    Returns:
        tensor: Tensor of the mask
    """
    tensor = transforms.ToTensor()(mask) #transform the image into a tensor
    tensor = torch.round(tensor) #round the values to have only 0 and 1
    if device is not None:
        tensor = tensor.to(device) #send the tensor to the device
    return tensor[0,:,:][None,:,:]

#transform a tensor into a mask, since when importing we are sure that we get np.arrays
def transform_to_patch(pixels, threshold):
    """ Transform a tensor into a mask

    Args:
        pixels (tensor): Tensor to transform
        threshold (int): Threshold to use
        
    Returns:
        (int): 0 or 1, depending on the mean of the pixels if it is above the threshold return 1, otherwise 0
    """
    m = np.mean(pixels) #compute the mean of the pixels
    if m  > threshold:
        return 1 #if the mean is above the threshold, we return 1
    else:
        return 0 #if the mean is below the threshold, we return 0

#transform an image into a np array, since when importing we are sure that we get np.arrays
def images_to_np_array(images):
    """ Transform a list of images into a np.array
    Args:
        images (list): List of images to transform
    
    Returns:
        np.array: np.array of the images
    """
    return np.array([np.array(img) for img in images]) #transform the images into np.arrays

#function to save a prediction changing the threshold to create the image for the report
def report(prediction, th):
    """ Save the prediction with a threshold, to create the image for the report

    Args:
        prediction (np.array): Prediction to save
        th (int): Threshold to use
    
    Returns:
        np.array: np.array of the prediction
    """
    new_prediction = np.zeros(prediction.shape)
    new_prediction[prediction > th] = 150 #if the prediction is above the threshold, we put 150, so it is visible in the image
    return new_prediction

#take a predected image and translate it into a prediction on patches
def transform_prediction_to_patch(img, id, patch_size=16, step=16, th=0.25):
    """ Transform a predicted image into a prediction on patches

    Args:
        img (np.array): Image to transform
        id (int): Id of the image
        patch_size (int): Size of the patch, the default is 16
        step (int): Step to use, the default is 16
        th (int): Threshold to use, the default is 0.25

    Returns:
        list: List of patches
        list: List of ids
    """
    prs = []
    ids = []
    for j in range(0,img.shape[1],step):
        for i in range(0, img.shape[0],step):
            threshold = th 
            prs.append(transform_to_patch(img[i:i+patch_size,j:j+patch_size],threshold)) #transform the image into a patch
            ids.append("{:03d}_{}_{}".format(id, j, i)) #create the id of the patch
    return prs, ids

#takes a np.array from images, and transform into PIL image (which is more usable to deal with chages on the image)
def PIL_Images_from_np_array(images):
    """ Transform a np.array into PIL images
    
    Args:
        images (np.array): Array of images to transform

    Returns:
        list: List of PIL images
    """
    return list(map(Image.fromarray, images)) #transform the np.arrays into PIL images

#split between test and train set because we want to see the performance of our models
def split_data(x, y):
    """ Split the data into train and test set

    Args:
        x (list): List of images
        y (list): List of masks

    Returns:
        list: List of images for the train set
        list: List of masks for the train set
        list: List of images for the test set
        list: List of masks for the test set
    """
    if SEED is not None:
        np.random.seed(SEED) 
    indices = np.arange(len(x)) #create an array of indices
    np.random.shuffle(indices) #shuffle the indices

    split_point = int(len(indices) * (1 - TEST_SIZE)) #compute the split point

    train_x, train_y = np.array(x)[indices[:split_point]], np.array(y)[indices[:split_point]] #split the data for the train set
    test_x, test_y = np.array(x)[indices[split_point:]], np.array(y)[indices[split_point:]] #split the data for the test set

    return train_x, train_y, test_x, test_y

# load images
def load_image(infilename):
    """ Load an image

    Args:
        infilename (str): Path of the image

    Returns:
        np.array: np.array of the image
    """
    data = mpimg.imread(infilename) #load the image
    return data

# Extract 6-dimensional features consisting of average RGB color as well as variance
def extract_features(img):
    """ Extract features from an image

    Args:
        img (np.array): Image to extract features from

    Returns:
        np.array: np.array of the features
    """
    feat_m = np.mean(img, axis=(0,1)) #compute the mean of the image
    feat_v = np.var(img, axis=(0,1)) #compute the variance of the image
    feat = np.append(feat_m, feat_v) #append the mean and the variance
    return feat

# Compute features for each image patch
def value_to_class(v):
    """ Compute the class of a patch

    Args:
        v (np.array): Array of values   

    Returns:
        int: 0 or 1, depending on the sum of the values if it is above the threshold return 1, otherwise 0
    """
    df = np.sum(v) 
    if df > THRESHOLD:
        return 1 #if the sum of the values is above the threshold, we return 1
    else:
        return 0 #if the sum of the values is below the threshold, we return 0
    
# Load training images and extract features for each patch
def plot_prediction(prediction, dir):
    """ Plot the prediction and save it

    Args:
        prediction (np.array): Prediction to plot
        dir (str): Directory to save the image

    Returns:
        png: Image of the prediction
    """
    img = Image.fromarray(prediction)
    if img.mode != 'RGB':
        img = img.convert('RGB') #convert the image into RGB
    img_path = os.path.join(dir, "prediction.png") #create the path of the image
    img.save(img_path)  #save the image
    return img