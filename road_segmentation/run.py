import torch
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

from config import RoadSegmentationConfig
from dataset import RoadSegmentationDataset
from model import RoadSegmentationModel
from torch.utils.data import DataLoader
from regression import *
from helpers import split_data
from parameters import *
from validation import *
import sys
sys.path.append("..")

#define all the experiment, 4 is the best one
EXPERIMENT = 4

if EXPERIMENT == 1:
    divide_patches = True
    normalize = False
    augment = False
    training = True
    submmission = True
    clahe=True
elif EXPERIMENT == 2:
    divide_patches = True
    normalize = True
    augment = False
    training = True
    submmission = True
    clahe=True
elif EXPERIMENT == 3:
    divide_patches = True
    normalize = False
    augment = True
    training = True
    submmission = True
    clahe=True
elif EXPERIMENT == 4:
    divide_patches = True
    normalize = True
    augment = True
    training = True
    submmission = True
    clahe=False
    baseline = False
    validation = False
    THRESHOLD = 0.2
    LEARNING_RATE = 0.0003
 
elif EXPERIMENT == 5:
    divide_patches = True
    normalize = True
    augment = True
    training = True
    submmission = True
    clahe=True
    baseline = False

#experiment 6 is the one used for the baseline
if EXPERIMENT == 6:
    divide_patches = False
    normalize = False
    augment = False
    training = False
    submmission = False
    clahe = False
    baseline = True

def main():
    """ Run the model with the specified parameters and return the results

    Returns:
        list: F1 score
        list: Accuracy    
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    configuration = RoadSegmentationConfig(augment=augment, normalize=normalize, divide_patches=divide_patches, clahe=clahe) #define the configuration

    data = sorted(glob(DATA_PATH + "/training/images/*.png")) #get the images
    gts = sorted(glob(DATA_PATH + "/training/groundtruth/*.png")) #get the ground truth images

    if baseline :
        submission_images = sorted(glob(DATA_PATH + "/test_set_images/*/*"), #get the submission images
        key = lambda x: int(x.split('/')[-2].split('_')[-1])) 
        x, y = load_data_log_reg_train(DATA_PATH + "/training/images/", DATA_PATH + "/training/groundtruth/") #load the data for the logistic regression
        x_test = load_data_log_reg_test(DATA_PATH + "/test_set_images/") #load the data for the logistic regression
        predict_regression=log_regression (x, y, x_test) #logistic regression
        log_reg_submit (predict_regression) #create the submission file
        return

    optimal_th = THRESHOLD
    optimal_lr = LR
    optimal_ps = PATCH_SIZE

    # cross validation
    if validation:
        print("Starting Cross Validation over Foreground Threshold")
        optimal_th,_ = cross_validation(data,gts,THRESHOLD_VALIDATION_VECTOR,"threshold",len(data))
        print("Starting Cross Validation over Learning Rate")
        optimal_lr,_ = cross_validation(data,gts,LEARNING_RATE_VALIDATION_VECTOR,"learning rate",len(data))
        print("Starting Cross Validation over Patch size ")
        optimal_ps,_ = cross_validation(data,gts,PATCH_SIZE_VALIDATION_VECTOR,"patch size",len(data))

    train_data, train_labels, test_data, test_labels = split_data(data,gts) #split the data between train and test set

    if submmission:
        # run training on full dataset
        train_data = data
        train_labels = gts

    train_set = RoadSegmentationDataset(train_data,train_labels,configuration, True, DEVICE, optimal_ps) #create the train set
    test_set = RoadSegmentationDataset(test_data,test_labels,configuration,True,DEVICE, optimal_ps) #create the test set
    evaluation_dataset = RoadSegmentationDataset(test_data,test_labels,configuration,False,DEVICE, optimal_ps) #create the evaluation set

    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True) #create the train loader
    test_loader = DataLoader(test_set,BATCH_SIZE, shuffle=False)  #create the test loader
    evaluation_loader = DataLoader(evaluation_dataset,1, shuffle=False) #create the evaluation loader

    model = RoadSegmentationModel(DEVICE, th= optimal_th, lr = optimal_lr, max_iter = MAX_ITER, ps= optimal_ps) #create the model

    if training:
        if submmission:
            results = model.train(train_loader, test_loader,False,None,augment) #train the model
            f1 = None
            acc = None
        else: 
            results = model.train(train_loader, test_loader,True,evaluation_loader,False) #train the model
            f1 = results['f1'] #get the f1 score
            acc = results['accuracy'] #get the accuracy
        loss = results['train_loss'] #get the train loss
        test_loss = results['test_loss'] #get the test loss
        print("TRAINING LOSS = " + str(loss[len(loss)-1]))
        print("TEST LOSS = " + str(test_loss[len(test_loss)-1]))
    if submmission:
        submission_images = sorted(glob(DATA_PATH + "/test_set_images/*/*"), 
            key = lambda x: int(x.split('\\')[-2].split('_')[-1])) #get the submission images 
    

        test_set = RoadSegmentationDataset(submission_images, None, configuration, False, DEVICE) #create the test set
        test_loader = DataLoader(test_set,1) #create the test loader

        model.submit(test_loader) #create the submission file

    return f1, acc

__name__ = "__main__"

if __name__ == "__main__":
    f1,acc = main()
