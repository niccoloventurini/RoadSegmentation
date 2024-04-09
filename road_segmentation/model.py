import torch
from torch import nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from pretrained_network import *
from sklearn.metrics import f1_score, accuracy_score
from helpers import *
from parameters import *
from postprocess import postprocess

class RoadSegmentationModel(nn.Module):
    def __init__(self, device, lr = LR, th =THRESHOLD, max_iter = MAX_ITER, ps=PATCH_SIZE):
        """ Initialize the model
        
        Args:
            device (str): Device to use
            lr (float): Learning rate, default is LR
            th (float): Foreground threshold, default is THRESHOLD
            max_iter (int): Maximum number of iterations, default is MAX_ITER
            ps (int): Patch size, default is PATCH_SIZE
            
        """
        super().__init__()
        self.device = device
        self.pre_trained_network = get_deeplabplus()
        self.criterion = nn.BCEWithLogitsLoss()
        self.pre_trained_network.to(self.device)
        self.th = th
        self.lr = lr
        self.max_iter = max_iter
        self.ps = ps
    
    #forward pass
    def forward(self, data):
        """ Forward pass of the model

        Args:
            data (list): List of data to use

        Returns:
            tensor: Prediction of the model
        """
        data_x = data[0].to(self.device) #data to the device
        return self.pre_trained_network(data_x) #return the prediction

    #train the model for one epoch
    def train_epoch(self, loader, optimizer):
        """ Train the model for one epoch 
        
        Args:
            loader (torch.utils.data.DataLoader): loader for the data to use
            optimizer (torch.optim): Optimizer to use
            
        Returns:
            float: Loss of the epoch
        """
        loss = []
        self.pre_trained_network.train() #set the model in training mode
        for batch in tqdm(loader): 
            pr = self.forward(batch) #predictions
            y = batch[1].to(self.device) #labels
            l = self.criterion(pr,y).to(self.device) #loss
            optimizer.zero_grad() #set the gradients to zero
            l.backward() #compute the gradients
            optimizer.step() #update the weights
            l = l.cpu().detach().numpy() #compute loss
            loss.append(l)
        return np.mean(loss)

    #test the model for one epoch
    def test_epoch(self, loader):
        """ Test the model for one epoch 

        Args:
            loader (torch.utils.data.DataLoader): loader for the data to use

        Returns:
            float: Loss of the epoch
        """
        self.pre_trained_network.eval() #set the model in evaluation mode
        loss = []
        with torch.no_grad():
            for batch in tqdm(loader):
                pr = self.forward(batch) #predictions
                y = batch[1].to(self.device) #labels
                l = self.criterion(pr,y).to(self.device) #loss
                l = l.cpu().detach().numpy() #compute loss
                loss.append(l)
        return np.mean(loss)

    #get the score of the model
    def get_score(self,loader, do_postprocessing=True):
        """ Get the score of the model 

        Args:
            loader (torch.utils.data.DataLoader): loader for the data to use
            do_postprocessing (bool): If True, apply postprocessing, default is True

        Returns:
            (float, float): F1 score and accuracy score 
        """
        self.pre_trained_network.eval()
        prs = []
        ys = []
        first_predictions = self.make_prediction(loader,do_postprocessing) #make a prediction
        masks = loader.dataset.gt #get the groundtruths
        #transform the prediction into patches
        for pr, y in zip(first_predictions, masks):
            labels, _ = transform_prediction_to_patch(pr,1, patch_size =self.ps, th=self.th) #transform the prediction into patches
            prs.extend(labels) #extend the list of predictions
            y = y[0].cpu().detach().numpy() #get the mask
            real_labels, _ = transform_prediction_to_patch(y,1, patch_size =self.ps, th=self.th) #transform the mask into patches
            ys.extend(real_labels) #extend the list of labels
        return f1_score(ys,prs), accuracy_score(ys,prs)

    #make a prediction
    def make_prediction(self, loader, do_postprocessing):
        """ Make a prediction

        Args:   
            loader (torch.utils.data.DataLoader): loader for the data to use
            do_postprocessing (bool): If True, apply postprocessing, default is True

        Returns:
            list: List of predictions
        """
        pr = []
        self.pre_trained_network.eval() #set the model in evaluation mode
        sigmoid = torch.nn.Sigmoid() #apply sigmoid to the output
        with torch.no_grad(): 
            for batch in tqdm(loader):
                p = sigmoid(self.pre_trained_network(batch.to(self.device))).cpu().detach().numpy() #apply sigmoid to the output
                pr.append(p[0][0]) #append the prediction
        if do_postprocessing:
            pr = postprocess(pr) #apply postprocessing
        return pr

    #train the model
    def train(self, train_loader, test_loader, evaluate, evaluate_loader, do_postprocessing):
        """ Train the model

        Args:
            train_loader (torch.utils.data.DataLoader): loader for the training data
            test_loader (torch.utils.data.DataLoader): loader for the test data
            evaluate (bool): If True, evaluate the model, default is True
            evaluate_loader (torch.utils.data.DataLoader): loader for the evaluation data
            do_postprocessing (bool): If True, apply postprocessing, default is True

        Returns:
            dict: Dictionary containing the results
        """
        optimizer = torch.optim.Adam(self.pre_trained_network.parameters(), lr=self.lr) #define the optimizer
        losses = []
        test_losses = []
        accuracies = []
        f1s = []
        best_loss = {'loss': float('inf'), 'epoch': 0}
        i = 1
        while True:
            print("EPOCH " + str(i))
            l_train = self.train_epoch(train_loader, optimizer) #train the model
            print("epoch trained, now testing") 
            losses.append(l_train) #append the loss
            l_test = self.test_epoch(test_loader) #test the model
            test_losses.append(l_test) #append the loss
            if evaluate:
                print("Test Loss for epoch " + str(i) + " = " + str(l_test))
            else:
                print("Train Loss for epoch " + str(i) + " = " + str(l_test))
            f1 = 0
            acc = 0
            if evaluate:
                f1, acc = self.get_score(evaluate_loader, do_postprocessing) #get the score
                print("LOSS = " + str(l_test) + " F1 = " + str(f1) + " ACCURACY = " + str(acc))

            f1s.append(f1)
            accuracies.append(acc)

            if l_test < best_loss['loss']: 
                best_loss['loss'] = l_test
                best_loss['epoch'] = i

            if i == self.max_iter - 1:
                break

            i += 1
        
        # save the results into a dictionary
        results = {}
        results['train_loss'] = losses
        results['f1'] = f1s
        results['accuracy'] = accuracies
        results['test_loss'] = test_losses
        return results

    #create the submission file
    def submit(self, test_loader):
        """ Create the submission file 

        Args:
            test_loader (torch.utils.data.DataLoader): loader for the test data
        """
        prs = self.make_prediction(test_loader,False) #make a prediction
        img_ids = range(1,len(prs)+1) #get the ids of the images

        ret_ids = []
        ret_labels = []

        for pr, i in zip(prs,img_ids):
            #create the image thanks to the prediction, of the 6 image of the test set
            if i == 6:
                pred = report(pr, self.th) # save the prediction to create the image for the report
                plot_prediction(pred, SUBMISSION_DIR) #save the image

            labels, ids = transform_prediction_to_patch(pr,i,th=self.th) #transform the prediction into patches
            for label in labels:
                ret_labels.append(label) #extend the list of predictions
            for id in ids:
                ret_ids.append(id) #extend the list of ids

        pd.DataFrame({'id': ret_ids, 'prediction' : ret_labels}).to_csv(SUBMISSION_PATH,index=False) #create the submission file