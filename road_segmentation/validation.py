import numpy as np
from parameters import *
from helpers import *
from config import RoadSegmentationConfig
from dataset import RoadSegmentationDataset
from model import RoadSegmentationModel
from torch.utils.data import DataLoader

# build the k indices for cross validation
def build_k_indices(N, k_fold, seed = SEED):
    """ Build k indices for k-fold cross-validation.
    
    Args:
        N (int): Number of data
        k_fold (int): Number of folds
        seed (int): Seed for the random permutation, default = SEED
        
    Returns:
        2D array: k indices, shape = (k_fold, N/k_fold) to split data into k fold
    """
    num_row = N
    interval = int(num_row / k_fold) #size of each fold
    np.random.seed(seed)
    indices = np.random.permutation(num_row) #random permutation of the indices
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)] #split the indices into k_fold
    return np.array(k_indices)

# cross validation step
def cross_validation_step(data, gts, k_indices, k, th, lr, ps):
    """ One step of cross validation

    Args:
        data (np.array): Data
        gts (np.array): Ground truth
        k_indices (2D array): k indices
        k (int): k index
        th (float): Threshold
        lr (float): Learning rate
        ps (int): Patch size
    
    Returns:
        float: F1 score
    """
    train_data = np.delete(data, k_indices[k], axis = 0) #get the train data
    train_gts = np.delete(gts, k_indices[k], axis = 0) #get the train labels
    test_x = []
    test_y = []
    for id in k_indices[k]:
        test_x.append(data[id]) #get the test data
        test_y.append(gts[id]) #get the test labels
    configuration=RoadSegmentationConfig (augment=True, normalize=True, divide_patches=True, clahe=True) #define the configuration

    train_set = RoadSegmentationDataset(train_data,train_gts,configuration, True, DEVICE, ps) #create the train set
    test_set = RoadSegmentationDataset(test_x,test_y,configuration,True,DEVICE, ps) #create the test set
    evaluation_dataset = RoadSegmentationDataset(test_x,test_y,configuration,False,DEVICE, ps) #create the evaluation set

    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True) #create the train loader
    test_loader = DataLoader(test_set,BATCH_SIZE, shuffle=False) #create the test loader
    evaluation_loader = DataLoader(evaluation_dataset,1, shuffle=False) #create the evaluation loader

    model = RoadSegmentationModel(DEVICE,th=th,lr=lr,max_iter=10) #create the model
    results = model.train(train_loader, test_loader,True,evaluation_loader,False) #train the model
    f1_score = results["f1"] #get the f1 score
    return f1_score

# cross validation over the patch size
def validation_over_patch_size(data, gts, k_indices, k, ps):
    """ Cross validation over the patch size
    
    Args:
        data (np.array): Data
        gts (np.array): Ground truth
        k_indices (2D array): k indices
        k (int): k index
        ps (int): Patch size
    
    Returns:
        float: F1 score
    """
    return cross_validation_step(data,gts,k_indices,k,THRESHOLD,LR, ps)

# cross validation over the threshold
def validation_over_threshold(data, gts, k_indices, k, threshold):
    """ Cross validation over the threshold
    
    Args:   
        data (np.array): Data
        gts (np.array): Ground truth
        k_indices (2D array): k indices
        k (int): k index
        threshold (float): Threshold
    
    Returns:
        float: F1 score
    """
    return cross_validation_step(data,gts,k_indices,k,threshold,LR, PATCH_SIZE)

# cross validation over the learning rate
def validation_over_learning_rate(data, gts, k_indices, k, lr):
    """ Cross validation over the learning rate 

    Args:
        data (np.array): Data
        gts (np.array): Ground truth
        k_indices (2D array): k indices
        k (int): k index
        lr (float): Learning rate

    Returns:
        float: F1 score
    """
    return cross_validation_step(data,gts,k_indices,k,THRESHOLD,lr, PATCH_SIZE)

# cross validation
def cross_validation(data, gts, parameters, parameter_name, N, seed = SEED, k_fold = K_FOLD):
    """ Cross validation over a parameter, which is parameter_name searching the best one in parameters

    Args:
        data (np.array): Data
        gts (np.array): Ground truth
        parameters (np.array): Parameters to try
        parameter_name (str): Parameter name to try
        N (int): Number of data
        seed (int): Seed for the random permutation, default = SEED
        k_fold (int): Number of folds, default = K_FOLD

    Returns:
        float: Optimal parameter
        float: Best performance
    """
    k_indices = build_k_indices(N,k_fold,seed) #build the k indices
    best_performance = -1 
    optimal_parameter = -1

    for parameter in parameters:
        print("Trying " + str(parameter_name) + " = " + str(parameter))
        avg_performance = 0 #average performance
        performances = np.zeros(k_fold) #performances for each fold

        for k in range(k_fold):
            performance = 0
            if parameter_name == "threshold":
                performance = validation_over_threshold(data,gts,k_indices,k,parameter)[-1] #get the performance for the threshold
            elif parameter_name == "learning rate":
                performance = validation_over_learning_rate(data,gts,k_indices,k,parameter)[-1] #get the performance for the learning rate
            elif parameter_name == "patch size":
                performance = validation_over_patch_size(data,gts,k_indices,k,parameter)[-1] #get the performance for the patch size

            avg_performance = performance + avg_performance 
            performances[k] = performance #save the performance for the fold

        avg_performance = avg_performance / k_fold #compute the average performance

        print("Cross-Validation for " + parameter_name + " = " + str(parameter) + " with f1_score = " + str(avg_performance))
        if best_performance == -1 or avg_performance > best_performance: #if the performance is better than the best performance
            best_performance = avg_performance
            optimal_parameter = parameter

    print("Optimal Patameter for " + parameter_name + " = " + str(optimal_parameter))
    return optimal_parameter, best_performance