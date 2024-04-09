import numpy as np

# apply postprocessing to the prediction
def postprocess(y):
    """ Apply postprocessing to the prediction

    Args:
        y (np.array): Prediction to postprocess

    Returns:
        list: Postprocessed prediction
    """
    res = []
    y = np.array(y)
    sz = y[0][0].shape[0] #get the size of the image
    y = y.reshape((-1,6,sz,sz)) #reshape the prediction
    for images in y:
        one = np.fliplr(images[1]) #flip the image
        two = np.rot90(images[2], k=3) #rotate the image
        three = np.rot90(images[3], k=2) #rotate the image
        four = np.rot90(images[4],k=1) #rotate the image
        five = np.flipud(images[5]) #flip the image

        m = np.stack([images[0], one, two, three, four, five]) #stack the images
        m = np.mean(m,axis=0) #compute the mean of the images
        res.append(m) #append the mean to the result
    return res
