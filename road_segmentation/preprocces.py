from PIL import Image, ImageEnhance, ImageFilter 
import numpy as np
from helpers import images_to_np_array, PIL_Images_from_np_array
from parameters import *
from config import RoadSegmentationConfig
import cv2

# normalize images between 0 and 1
def normalize_images(data):
    """ Normalize the images between 0 and 1 
    
    Args:
        data (np.array): Images to normalize
        
    Returns:
        np.array: Normalized images
    """
    for i, image in enumerate(data):
        image = image / 255
        data[i] = (image - image.mean(axis=(0, 1), dtype='float64')) / (image.std(axis=(0, 1), dtype='float64')) #normalize the image
    return data

# augment images
def augment_images(data):
    """ Augment the images, flipping, rotating, blurring and changing brightness

    Args:
        data (list): Images to augment

    Returns:
        list: Augmented images
    """
    augmented_images = []

    # Flip, rotate, blur and change brightness of the images
    for image in data:
        augmented_imgs = [
            image.transpose(Image.FLIP_LEFT_RIGHT),
            image.transpose(Image.ROTATE_90),
            image.transpose(Image.ROTATE_180),
            image.transpose(Image.ROTATE_270),
            image.transpose(Image.FLIP_TOP_BOTTOM),
            image.filter(ImageFilter.GaussianBlur(4)),
        ]

        color_shift = ImageEnhance.Color(image)
        augmented_imgs.append(color_shift.enhance(0.5)) #increase brightness

        for angle in range(10, 61, 10):
            augmented_imgs.append(image.rotate(angle, resample=Image.BICUBIC))
        augmented_images.extend(augmented_imgs)

    return augmented_images

# apply CLAHE to the dataset
def apply_clahe_to_dataset(data, clip_limit=2.0, tile_grid_size=(8, 8)):
    """ Apply CLAHE to the dataset, enhancing the contrast of the images

    Args:
        data (list): Images to enhance
        clip_limit (float): Threshold for contrast limiting, default is 2.0
        tile_grid_size (tuple): Size of grid for histogram equalization, default is (8, 8)

    Returns:
        np.array: Enhanced images
    """
    enhanced_data = []

    # Apply CLAHE to each image in the dataset
    for image in data:
        # Convert the image to grayscale if it's a color image
        if len(image.shape) == 3:  # Check if the image is color
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Create a CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        # Apply CLAHE to the grayscale image
        clahe_image = clahe.apply(gray)

        # If the original image is color, merge the enhanced channel with the original channels
        if len(image.shape) == 3:
            clahe_image = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)

        enhanced_data.append(clahe_image)

    return np.array(enhanced_data)

# crop images into patches
def crop(image, width, height):
    """ Crop the image into patches

    Args:
        image (np.array): Image to crop
        width (int): Width of the patches
        height (int): Height of the patches

    Returns:
        list: List of patches
    """
    res = []
    for i in range(0,image.shape[1],height):
        for j in range(0,image.shape[0],width):
            if len(image.shape) == 2: #if the image is grayscale
                res.append(image[j:j + width, i : i + height]) #crop the image
            else:
                res.append(image[j:j + width, i : i + height, :]) #crop the image
    return res


# aplly preprocessing to the data
def preprocess(data, gts, configuration:RoadSegmentationConfig, train: bool, ps):
    """ Preprocess the data and the groundtruths
    
    Args:
        data (list): Images to preprocess
        gts (list): Groundtruths to preprocess
        configuration (RoadSegmentationConfig): Configuration of the preprocessing
        train (bool): If True, the data is training data
        ps (int): Patch size
        
    Returns:
        np.array: Preprocessed images
        np.array: Preprocessed groundtruths
    """
    data = images_to_np_array(data)

    if train:
        gts = images_to_np_array(gts) #transform the groundtruths into np.arrays
    else:
        gts = None

    if configuration.norm:
        data = normalize_images(data) #normalize the images

    if not train:
        return data, None

    if configuration.aug:
        data = PIL_Images_from_np_array(data) #transform the images into PIL images

        if gts is not None:
            gts = PIL_Images_from_np_array(gts) #transform the groundtruths into PIL images

        augmented_images = augment_images(data) #augment the images

        if gts is not None:
            augmented_groundtruths = augment_images(gts) #augment the groundtruths
            gts.extend(augmented_groundtruths) #add the augmented groundtruths to the groundtruths

        data.extend(augmented_images) #add the augmented images to the images
        data = images_to_np_array(data) #transform the images into np.arrays

        if gts is not None:
            gts = images_to_np_array(gts) #transform the groundtruths into np.arrays

    if configuration.divide_patches and gts is not None: 
        patch_size = ps 
        data = [crop(image, patch_size,patch_size) for image in data] #crop the images
        data = np.asarray([data[i][j] for i in range(len(data)) for j in range(len(data[i]))]) #transform the images into np.arrays
        gts = [crop(image, patch_size, patch_size) for image in gts] #crop the groundtruths
        gts = np.asarray([gts[i][j] for i in range(len(gts)) for j in range(len(gts[i]))]) #transform the groundtruths into np.arrays
        data = images_to_np_array(data) #transform the images into np.arrays
        gts = images_to_np_array(gts) #transform the groundtruths into np.arrays

    if configuration.clahe:
        data= apply_clahe_to_dataset(data) #apply CLAHE to the images
        gts= apply_clahe_to_dataset(gts) #apply CLAHE to the groundtruths

    return data, gts

# load data
def load_data(path_data, path_gts, train : bool, device,configuration:RoadSegmentationConfig, ps):
    """ Load the data and preprocess it

    Args:
        path_data (list): Paths of the images
        path_gts (list): Paths of the groundtruths
        train (bool): If True, the data is training data
        device (str): Device to use, default is None
        configuration (RoadSegmentationConfig): Configuration of the preprocessing
        ps (int): Patch size

    Returns:
        np.array: Preprocessed images
        np.array: Preprocessed groundtruths
    """
    data = [Image.open(img) for img in path_data] #load the images
    
    gts = []
    if path_gts is not None:
        gts = [Image.open(gt) for gt in path_gts] #load the groundtruths

    if train:
        data, gts = preprocess(data, gts, configuration,True, ps) #apply preprocessing to the data and the groundtruths
    else:
        data, _ = preprocess(data, None, configuration, False, ps) #apply preprocessing to the data

    return data, gts