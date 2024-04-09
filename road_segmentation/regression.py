import os
import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from helpers import *
from preprocces import crop

# load data for the logistic regression train
def load_data_log_reg_train(path_data, path_gts):
    """ Load the data for the logistic regression train
    
    Args:
        path_data (str): Path of the images
        path_gts (str): Path of the groundtruths
        
    Returns:
        np.array: Features
        np.array: Labels
    """
    files = os.listdir(path_data) #get the files
    n=len(files) #number of files
    imgs = [load_image(path_data + files[i]) for i in range(n)] #load the images
    gt_dir = path_gts  # ground truth directory
    print("Load " + str(n) + " images") 

    gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)] #load the ground truth images 

    img_patches = [crop(imgs[i], 16, 16) for i in range(n)] #crop the images into patches
    gt_patches = [crop(gt_imgs[i], 16, 16) for i in range(n)] #crop the ground truth images into patches

    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

    X = np.asarray([ extract_features(img_patches[i]) for i in range(len(img_patches))]) #extract features
    Y = np.asarray([value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))]) #get the labels
    return X, Y

# load data for the logistic regression test
def load_data_log_reg_test(path_data):
    """ Load the data for the logistic regression test

    Args:
        path_data (str): Path of the images

    Returns:
        np.array: Features
    """
    files = os.listdir(path_data) #get the files
    n=len(files)
    imgs=[]
    for j in range (n):
        imgs.append(load_image(path_data + files[j] + "/" + files[j] + ".png")) #load the images 

    print("Load " + str(n) + " images")
    img_patches = [crop(imgs[i], 16, 16) for i in range(n)] #crop the images into patches

    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    X = np.asarray([ extract_features(img_patches[i]) for i in range(len(img_patches))]) #extract features

    return X


# logistic regression
def log_regression (x, y, x_test):
    """ Logistic regression model 

    Args:
        x (np.array): Features
        y (np.array): Labels
        x_test (np.array): Features of the test set

    Returns:
        np.array: Predictions
    """

    logreg = linear_model.LogisticRegression(C=1e5, class_weight="balanced") #define the model
    logreg.fit(x, y) #train the model
    # Predict on the training set to print
    predict_test = logreg.predict(x_test)
    # Predict on the train (to verify logreg.predict working)
    predict=logreg.predict(x)

    # Get non-zeros in prediction and grountruth arrays
    predict_n = np.nonzero(predict)[0]
    y_n = np.nonzero(y)[0]

    TPR = len(list(set(y_n) & set(predict_n))) / float(len(predict)) #compute the true positive rate
    accuracy = accuracy_score(y, predict) #compute the accuracy
    f1 = f1_score(y, predict) #compute the f1 score
    print('True positive rate = ' + str(TPR))
    print('Accuracy score = ' + str(accuracy))
    print('F1 score = ' + str(f1))

    print (predict_test)
    return predict_test

# create the submission file for the logistic regression
def log_reg_submit (predict_test):
    """ Create the submission file for the logistic regression

    Args:
        predict_test (np.array): Predictions
    """
    img_ids = range(1,len(predict_test)+1) #get the ids of the images
    ret_ids = []
    ret_labels = []
    predict_test=np.asarray(predict_test) #transform the prediction into an array
    k=0
    j=0
    l=1
    for pr, i in zip(predict_test,img_ids): #create the submission file
        if (j>592):
          j=0
          k+=16
        if (k>592):
          j=0
          k=0
          l+=1

        id="{:03d}_{}_{}".format(l, k,j) #get the id of the image
        j+=16
        ret_labels.append(pr) 
        ret_ids.append(id)

    pd.DataFrame({'id': ret_ids, 'prediction' : ret_labels}).to_csv(SUBMISSION_PATH,index=False) #create the submission file