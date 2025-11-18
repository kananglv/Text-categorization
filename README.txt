1) The code in "All pre-processing"

Essentially, all comments are in the file however what is important:

When you will connect google drive to the page in google colab, there will
be lines of code that will require you to enter the path of a file , please 
check that before running.

In the beginning just rename "USE ME" file into a file with the "bydate" name 
download the "bydate" file in our google disk and run it.

When you will reach the SC pre-processing section, replace the file paths
with the correct test or train data path in google disk. Then when you will
reach "file_path" code block stage , enter the respective path of pre-processed test or train data set in the window input and keep using the same path until the end. 

2) This part of the code can be run natively from command line
Place your training data in a folder named "DATA FOR THE MODEL TRAIN"
Place your testing data in a folder named "DATA FOR THE MODEL TEST"
Put both python scripts in the same parent folder that has both sets of data
If the script doesn't run, adjust the absolute path from "C:\Users\Karim\Downloads\DATA FOR THE MODEL TEST" and the same for test to your path, if necessary
Run the script: python ensemble_learning_combinations.py
Run the script: python hyperparameter_tuning.py
Please note that hyperparameter_tuning.py may take a long time to run as it tests a lot of hyperparameter combinations
This part of the code was run in the following environment:
numpy==1.23.5
pandas==2.2.2
tensorflow==2.12.0
keras==2.12.0
sklearn==1.4.2
