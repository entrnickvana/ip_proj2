import pandas as pd
from skimage import io
import numpy as np
import random
import skimage as ski
import csv
import os


def writeCSV(csv_file_name, Trainingdictlist):
    csv_columns = ['RefImageName','NoiseType','NoisyImage']
    try:
        with open(csv_file_name, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in Trainingdictlist:
                writer.writerow(data)
    except IOError:
        print("I/O error")


def noiseAddtion(image,noiseIndicator):
    # This function gets an image of size m x n. Range of values is (0,1).
    # noiseIndicator is type of noise that needs to be added to the image
    # noiseIndicator == 0 indicates an addition of gaussian noise with mean 0 and var 0.08
    # noiseIndicator == 1 indicates an addtion of Salt and Pepper noise with intensity variation of 0.08
    # noiseIndicator == 2 indicates an addition of poisson noise
    # noiseIndicator == 3 indicates an addition of Speckle noise of mean 0 and var 0.05
    
    ## This function should return a noisy version of the input image
    
    ##  ***************** Your Code starts here ***************** ##
    
    raise NotImplementedError
    
    ## ***************** Your Code ends here ***************** ##
    return noisy

def main(train, numberOfSamples,noisyIndicatorLow,noisyIndicatorHigh):
    root_dir = '../../utils/GrayScale' 
    if(train==1):
        name_csv = pd.read_csv('../../utils/file_name_train.csv')
        csv_file_name = "TrainingDataSet.csv"
        directoryName = "TrainingDataset/"
        Trainingdictlist = [dict() for x in range(numberOfSamples)]
        if not os.path.exists(directoryName):
            os.makedirs(directoryName)
    else:
        name_csv = pd.read_csv('../../utils/file_name_test.csv')
        csv_file_name = "TestingDataSet.csv"
        directoryName = "TestingDataset/"
        Trainingdictlist = [dict() for x in range(numberOfSamples)]
        if not os.path.exists(directoryName):
            os.makedirs(directoryName)

    for i in range(numberOfSamples):
        r2 = random.randint(0,len(name_csv)-1) # Choose an image randomly from the dataset
        # Read the image from a path
        img_name = os.path.join(root_dir,name_csv.iloc[r2, 0]) 
        image    = io.imread(img_name) 
        
        # Normalize the image to range (0,1) 
        M,N      = image.shape
        maximumPixelValue = np.max(image)
        image = image/maximumPixelValue  
        
        # Choosing the noise randomly 
        r1 = random.randint(2,3)
        noisyImage = noiseAddtion(image,r1)
        Trainingdictlist[i]={'RefImageName':img_name,'NoiseType': r1,'NoisyImage':str(i)+'.png'}
        io.imsave(directoryName+str(i)+'.png',noisyImage)
    writeCSV(csv_file_name, Trainingdictlist)

if __name__ == "__main__":
    main(1,800,0,3) ## creating 400 Samples of Training Data
    main(0,400,0,3) ## creating 200 Samples of Testing Data

