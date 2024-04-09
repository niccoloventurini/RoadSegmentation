class RoadSegmentationConfig():
    """ Configuration class for Road Segmentation task 

    Args:
        divide_patches (bool): If True, the images are divided into patches 
        normalize (bool): If True, the images are normalized
        augment (bool): If True, the images are augmented
        clahe (bool): If True, the images are enhanced with CLAHE
    
    """
    def __init__(self, divide_patches, normalize, augment, clahe):
        self.divide_patches= divide_patches
        self.norm=normalize
        self.aug=augment
        self.clahe=clahe