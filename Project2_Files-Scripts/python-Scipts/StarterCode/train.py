import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim

import time
import sys 
sys.path.append("../../utils")
from utils import NoiseDatsetLoader

dtype = torch.float32
#you can change this to "cuda" to run your code on GPU
cpu = torch.device('cpu')


def checkTestingAccuracy(dataloader,model):
    ## This Function Checks Accuracy on Testing Dataset for the model passed
    ## This function should return the loss the Testing Dataset

    ## Before running on Testing Dataset the model should be in Evaluation Mode    
    model.eval()
    totalLoss = []
    loss_mse = nn.MSELoss()
    for t, temp in enumerate(dataloader):
        NoisyImage = temp['NoisyImage'].to(device=cpu,dtype=dtype)
        referenceImage = temp['image'].to(device=cpu,dtype=dtype)

        ## For Each Test Image Calculate the MSE loss with respect to Reference Image 
        ## Return the mean the total loss on the whole Testing Dataset
        ## ************* Start of your Code *********************** ##
        

        raise NotImplementedError
        

        ## ************ End of your code ********************   ##


def trainingLoop(dataloader,model,optimizer,nepochs):
    ## This function Trains your model for 'nepochs'
    ## Using optimizer and model your passed as arguments
    ## On the data present in the DataLoader class passed
    ##
    ## This function return the final loss values

    model = model.to(device=cpu)    
    
    ## Our Loss function for this exercise is fixed to MSELoss
    loss_function = nn.MSELoss()
    loss_array =[]
    for e in range(nepochs):
            print("Epoch", e)
            for t, temp in enumerate(dataloader):
                ## Before Training Starts, put model in Training Mode
                model.train()   
                NoisyImage = temp['NoisyImage'].to(device=cpu,dtype=dtype)
                referenceImage = temp['image'].to(device=cpu,dtype=dtype)
                ## Pass your input images through the model
                ## Be sure to set the gradient as Zero in Optmizer before backward pass. Hint:- zero_grad()
                ## Step the otimizer on after backward pass
                ## calcualte the loss value using the ground truth and the output image
                ## Assign the value computed by the loss function to a varible named 'loss'

                ## Due to dataset being Gray you may have to use unsqueeze function here

                ## ************* Start of your Code *********************** ##

                raise NotImplementedError

                ## ************ End of your code ********************   ##
                loss_array.append(loss.cpu().detach().numpy())
            print("Training loss: ",loss)
    return loss



def main():
    TrainingSet = NoiseDatsetLoader(csv_file='TrainingDataSet.csv', root_dir_noisy='TrainingDataset')
    TestingSet  = NoiseDatsetLoader(csv_file='TestingDataSet.csv' , root_dir_noisy='TestingDataset')

    ## Batch Size is a hyper parameter, You may need to play with this paramter to get a more better network
    batch_size=16

    ## DataLoader is a pytorch Class for iterating over a dataset
    dataloader_train  = DataLoader(TrainingSet,batch_size=batch_size,num_workers=4)
    dataloader_test   = DataLoader(TestingSet,batch_size=1)

    ## Declare your Model/Models in the space below
    ## You should try atleast 3 models. 
    ## Model 1:- Declare a model with one conv2d filter with 1 input channel and output channel
    ## Model 2:-  Declare a model with five conv2d filters, with input channel size of first filter as 1 and output channel size of last filter as 1.
    ##            All other intermediate channels you can change as you see fit( use a maximum of 8 or 16 channel inbetween layers, otherwise the model might take a huge amount of time to train).
    ##            Add batchnorm2d layers between each convolution layer for faster convergence.
    ## Model 3:-  Add Non Linear activation in between convolution layers from Model 2

    ## ************* Start of your Code *********************** ##


    raise NotImplementedError
    model =    

    ## ************ End of your code ********************   ##
   

 
    ## Optimizer
    ## Please Declare An Optimizer for your model. We suggest you use SGD
    ## ************* Start of your Code *********************** ##

    raise NotImplementedError
    learning_rate =

    weight_decay  = 
    epochs        =   

    optimizer     = 
    
    ## ************ End of your code ********************   ##


    ## Train Your Model. Complete the implementation of trainingLoop function above 
    valMSE = trainingLoop(dataloader_train,model,optimizer,epochs)

    ## Test Your Model. Complete the implementation of checkTestingAccuracy function above 
    testMSE = checkTestingAccuracy(dataloader_test,model)
    print("Mean Square Error for the testing Set for the trained model is ", testMSE)
    
    model.eval() # Put you model in Eval Mode

    ## Plot graph of loss vs epoch
    ## ************* Start of your Code *********************** ##

    raise NotImplementedError

    ## ************ End of your code ********************   ##
    
    ## Plot some of the Testing Dataset images by passing them through the trained model
    ## ************* Start of your Code *********************** ##

    raise NotImplementedError    

    ## ************ End of your code ********************   ##

if __name__ == "__main__":

    main()

