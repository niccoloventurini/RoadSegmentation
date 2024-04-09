import segmentation_models_pytorch as smp
import torchvision.models as models

def get_deeplabplus(encoder_name = 'resnet34'):
    """ Get the DeepLabV3+ model 
    
    Args:
        encoder_name (str): Name of the encoder to use, default is 'resnet34'
        
    Returns:
        segmentation_models_pytorch.DeepLabV3Plus: DeepLabV3+ model
    """
    net = smp.DeepLabV3Plus(
    encoder_name=encoder_name,
    encoder_depth = 5,
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                     # model output channels (number of classes in your dataset)
    )

    return net


def deeplab_model(encoder_name = 'resnet34'):
    """ Get the DeepLabV3 model
    
    Args:   
        encoder_name (str): Name of the encoder to use, default is 'resnet34'
        
    Returns:    
        segmentation_models_pytorch.DeepLabV3: DeepLabV3 model
    """
    net = smp.DeepLabV3(
        encoder_name=encoder_name,
        encoder_depth=5,
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
    )

    return net

def get_resnet50(classes=1):
    """ Get the ResNet50 model

    Args:
        classes (int): Number of classes, default is 1

    Returns:
        torchvision.models.resnet.ResNet: ResNet50 model
    """
    #Create a DeepLabV3 model with ResNet50 backbone
    net = smp.DeepLabV3(
        encoder_name='resnet50',  # Use ResNet50 as the encoder
        encoder_depth=5,
        encoder_weights="imagenet",  # Use pre-trained weights for encoder initialization
        in_channels=3,  # Model input channels (3 for RGB images)
        classes=classes,  # Model output channels (1 for binary segmentation)
    )
    return net
