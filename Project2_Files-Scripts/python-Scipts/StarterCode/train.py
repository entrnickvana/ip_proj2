import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import sys
import os
import code
sys.path.append("../../utils")
from utils import NoiseDatsetLoader

dtype = torch.float32
#you can change this to "cuda" to run your code on GPU
cpu = torch.device('cpu')
#cpu = torch.device('cuda')


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
        curr_loss = loss_mse(NoisyImage, referenceImage)
        print("CURR_LOSS: ", curr_loss)
        totalLoss.append(loss_mse(NoisyImage, referenceImage))

    return np.mean(totalLoss)
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
                print("Batch index t: ", t)
                #print("temp: ", temp)                
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
                ## Pass your input images through the model
                #NoisyImage = torch.unsqueeze(NoisyImage, 1)
                #referenceImage = torch.unsqueeze(referenceImage, 1)
                
                output = model(NoisyImage) #Forward Pass
                
                #t0 = referenceImage.detach().numpy()[0][0]
                #t1 = referenceImage.detach().numpy()[1][0]                
                #t2 = referenceImage.detach().numpy()[2][0]                
                #t3 = referenceImage.detach().numpy()[3][0]
                #
                #n0 = NoisyImage.detach().numpy()[0][0]
                #n1 = NoisyImage.detach().numpy()[1][0]                
                #n2 = NoisyImage.detach().numpy()[2][0]                
                #n3 = NoisyImage.detach().numpy()[3][0]
                #
                #out0 = output.detach().numpy()[0][0]
                #out1 = output.detach().numpy()[1][0]
                #out2 = output.detach().numpy()[2][0]
                #out3 = output.detach().numpy()[3][0]
                #
                ##d0 = np.subtract(t0, out0)
                ##d1 = np.subtract(t1, out1)
                ##d2 = np.subtract(t2, out2)
                ##d3 = np.subtract(t3, out3)
                #
                #d0 = t0 - out0
                #d1 = t1 - out1
                #d2 = t2 - out2
                #d3 = t3 - out3
                #
                #dsqr_0 = np.square(d0)
                #dsqr_1 = np.square(d1)
                #dsqr_2 = np.square(d2)
                #dsqr_3 = np.square(d3)
                #
                #print("d0--------------------------\n", d0)
                #print("d1--------------------------\n", d1)
                #print("d2--------------------------\n", d2)
                #print("d3--------------------------\n", d3)
                #
                #plt.subplot(441)
                #plt.imshow(NoisyImage.detach().numpy()[0][0], cmap='gray')
                #plt.subplot(442)
                #plt.imshow(referenceImage.detach().numpy()[0][0], cmap='gray')
                #plt.subplot(443)                
                #plt.imshow(output.detach().numpy()[0][0], cmap='gray')
                #
                #print('weight before: \n', model[0].weight)
                #print('Grad weight before: \n', model[0].weight.grad)                           
                ### Be sure to set the gradient as Zero in Optmizer before backward pass. Hint:- zero_grad()
                #
                #print('after opt: \n', model[0].weight)
                #print('Grad after opt: \n', model[0].weight.grad)                           
                
                ## Step the otimizer after backward pass
                MSE_loss = loss_function(output, referenceImage) #Calculate loss
                #debugprint('after mse calc: \n', model[0].weight)
                #debugprint('Grad mse calc: \n', model[0].weight.grad)                           
                optimizer.zero_grad()  # Set gradient to zero
                           
                MSE_loss.backward() #Perform backward pass
                #debugprint('after mse_backward: \n', model[0].weight)
                #debugprint('Grad mse_backward: \n', model[0].weight.grad)                           
                           
                optimizer.step()  # Optimizer step
                #debugprint('after opt step: \n', model[0].weight)
                #debugprint('Grad opt step: \n', model[0].weight.grad)
                #plt.show()                
                           
                ## calcualte the loss value using the ground truth and the output image                                
                
                ## Assign the value computed by the loss function to a varible named 'loss'
                #loss = loss_function(output, referenceImage)
                loss = MSE_loss

                #code.interact(local=locals())                
                
                ## ************ End of your code ********************   ##
                loss_array.append(loss.cpu().detach().numpy())
                print("t: ", t, "Loss: ", loss.cpu().detach().numpy())
            print("Training loss: ",loss)
    code.interact(local=locals())
    return loss



def main():
    TrainingSet = NoiseDatsetLoader(csv_file='TrainingDataSet.csv', root_dir_noisy='TrainingDataset')
    TestingSet  = NoiseDatsetLoader(csv_file='TestingDataSet.csv' , root_dir_noisy='TestingDataset')

    ## Batch Size is a hyper parameter, You may need to play with this paramter to get a more better network
    batch_size=16

    ## DataLoader is a pytorch Class for iterating over a dataset
    #dataloader_train  = DataLoader(TrainingSet,batch_size=batch_size,num_workers=4)
    dataloader_train  = DataLoader(TrainingSet,batch_size=batch_size,num_workers=0)    
    dataloader_test   = DataLoader(TestingSet,batch_size=1)

    ## Declare your Model/Models in the space below
    ## You should try atleast 3 models. 
    ## Model 1:- Declare a model with one conv2d filter with 1 input channel and output channel
    ## Model 2:-  Declare a model with five conv2d filters, with input channel size of first filter as 1 and output channel size of last filter as 1.
    ##            All other intermediate channels you can change as you see fit( use a maximum of 8 or 16 channel inbetween layers, otherwise the model might take a huge amount of time to train).
    ##            Add batchnorm2d layers between each convolution layer for faster convergence.
    ## Model 3:-  Add Non Linear activation in between convolution layers from Model 2

    ## ************* Start of your Code *********************** ##

    # Model: 1 input channel to 1 output channel, kernel size 5 x 5 with padding of 2
    model = torch.nn.Sequential(torch.nn.Conv2d( 1, 1, kernel_size=(5,5), padding=2)) #28x28    
    #model = torch.nn.Sequential(torch.nn.Conv2d(1, 1, kernel_size=(5,5)), padding=2)

    ## ************ End of your code ********************   ##
 
    ## Optimizer
    ## Please Declare An Optimizer for your model. We suggest you use SGD
    ## ************* Start of your Code *********************** ##

    learning_rate = 1e-9

    weight_decay  = 1e-3
    epochs        = 10  

    #optimizer     = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    optimizer     = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5, weight_decay=weight_decay)    
    
    ## ************ End of your code ********************   ##

    ## Train Your Model. Complete the implementation of trainingLoop function above 
    valMSE = trainingLoop(dataloader_train, model, optimizer, epochs)

    ## Test Your Model. Complete the implementation of checkTestingAccuracy function above 
    testMSE = checkTestingAccuracy(dataloader_test, model)
    print("Mean Square Error for the testing Set for the trained model is ", testMSE)
    
    model.eval() # Put you model in Eval Mode

    code.interact(local=locals())

    ## Plot graph of loss vs epoch
    ## ************* Start of your Code *********************** ##
    plt.plot(epochs, valMSE)
    plt.show()
    
    
    #raise NotImplementedError

    ## ************ End of your code ********************   ##
    
    ## Plot some of the Testing Dataset images by passing them through the trained model
    ## ************* Start of your Code *********************** ##

    #raise NotImplementedError    

    ## ************ End of your code ********************   ##

if __name__ == "__main__":

    main()

