
NOTE:- YOU HAVE TO COMPLETE THE CODE INSIDE THE TAGs"Start of your code" and "End of you code"

Steps to Complete the Assignment:-

Make sure you have the following python packages:
1. numpy
2. pandas
3. matplotlib
4. opencv-python
5. skimage
6. pytorch
7. PIL

Part 1:- Creating a Noisy DataSet for trainging and testing the models

	a. Complete the function "noiseAddtion" in CreateNoisy.py

	b. After the function is complete run the following commands
		python CreateNoisy.py 
        Above executions create Noisy Training and testing dataset from the function you completed.



Part 2:- Completing the train.py for training

	a. Complete both "trainingLoop" function and "checkTestingAccuracy" functions in the file train.py
	if you don't understand what to do, review the pytorch tutorial.

	b. After both functions are complete. Declare you model.
    
    c. Assign initial values to hyper parameters(Learning rate, Weight decay, epochs)

    d. Set Up your optimizer

    e. If you have done everything right till this point your model should be able to train. You should see the reduce in your loss.
    You can run the code at this point to debug for any issues by setting number of epochs as 1
    python train.py

    f. In main(), include your code to plot graph of loss values vs epoch

	g. In main(),  include your code to plot sample test tmages. Both noisy and denoised version in the same plot.
