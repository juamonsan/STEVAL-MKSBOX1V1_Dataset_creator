#----------------------------------------------------------------------
#
# NPZ_CREATOR.PY
#
# Creates NPZ files from CSV structure 
# for each data folder in desired path
#
# TODO: Paths, random_seed... as command parameters 
#
#----------------------------------------------------------------------
#
# Author: Juan M. Montes
# GitHub: https://github.com/juamonsan
#
#----------------------------------------------------------------------

#----------------------------------------------------------------------
#IMPORTS
#----------------------------------------------------------------------

from tensorflow.keras.utils import to_categorical

import sys

# More imports

import os #for system folders and files management
import pathlib #for system folders and files management
import pandas as pd #for Dataframe handling

from datetime import datetime #getting current time for naming the result file

import numpy as np
import math
import csv

import re #for character extraction

from sklearn.model_selection import train_test_split

#----------------------------------------------------------------------
# IMPORTS END
#----------------------------------------------------------------------


#----------------------------------------------------------------------
# DEFINITIONS
#----------------------------------------------------------------------


# TRAIN SET CREATOR
def Train_Set_Creator(train_folder, random_seed):
    full_path = train_folder

    Current_file_path = os.path.join(full_path,os.listdir(full_path)[0])

    Current_file = open(Current_file_path) # Open first file for counting rows

    freq_samples=int(re.sub('[^0-9]','', sensor_folder[-7:-2])) #extracts the frequency from the directory name

    num_samples= len(os.listdir(full_path)) #number of .csv files in total
    tam_samples= len(Current_file.readlines())#number of lectures in a window sample (rows in each .csv)
    num_features = 3 #axis number (3 for accel) TODO: read column number from the file

    ##print("Number of samples: {}\nSample size: {}\nNumber of features: {}".format(num_samples,tam_samples,num_features))
    ##print("Frequency (from file folder name): {}".format(freq_samples))
    ##print("-----------------------\nPlot example for first file:\n-----------------------")

    file_class=int(Current_file_path[-6:-4])-1 # Class is in the filename, at the end, just before file extension
    ##print("The class for this file is {}".format(file_class))

    Current_file.close() # Close file

    DataX=np.empty([num_samples, tam_samples, num_features]) # Empty numpy array we will fill with data from .csv files
    DataY=np.empty([num_samples])

    for i in range (num_samples):
        Current_file_path = os.path.join(full_path,os.listdir(full_path)[i])
        Current_CSV=np.genfromtxt (Current_file_path, delimiter=",")
        DataX[i]=Current_CSV
        if np.any(np.isnan(DataX[i])):
            print(i)
            for k in range(0,len(DataX[i])):
                if np.any(np.isnan(DataX[i][k])):
                    print("File {} in line {} is NaN".format(i,k))
        file_class=int(Current_file_path[-6:-4])
        DataY[i]=file_class-1 #first class is class 1, but now we start from 0

    DataY = to_categorical(DataY)        
    print("The shape of DataX array is: {}".format(np.shape(DataX))) # We check if the shape is correct
    if np.any(np.isnan(DataX)):
        print("DANGER!! NaN values found in DataX!! Check exactly where above")
        
    print("The shape of DataY array is: {}".format(np.shape(DataY))) 
    if np.any(np.isnan(DataY)):
        print("DANGER!! NaN values found in DataY!!")

    # train_test_split uses a random seed int value from arguments
    X_train, X_test, y_train, y_test = train_test_split(DataX, DataY, test_size=0.15, random_state=random_seed, stratify=DataY) 
    print("X_train shape: {} || X_test shape: {}".format(np.shape(X_train), np.shape(X_test))) # We check if the shape is correct
    print("y_train shape: {} || y_test shape: {}".format(np.shape(y_train), np.shape(y_test)))

    return freq_samples, num_samples, tam_samples, num_features, X_train, X_test, y_train, y_test 


#TEST SET CREATOR
def Test_Set_Creator(full_path_test):    

    Current_file_path_test = os.path.join(full_path_test,os.listdir(full_path_test)[0])

    Current_file_test = open(Current_file_path_test) # Open first file for counting rows

    num_samples_test= len(os.listdir(full_path_test)) #number of .csv files in total
    tam_samples_test= len(Current_file_test.readlines())#number of lectures in a window sample (rows in each .csv)
    num_features_test = 3 #axis number (3 for accel) TODO: read column number from the file

    file_class_test=int(Current_file_path_test[-6:-4])-1 # Class is in the filename, at the end, just before file extension

    Current_file_test.close() # Close file
    
    TestDataX=np.empty([num_samples_test, tam_samples_test, num_features_test]) # Empty numpy array we will fill with data from .csv files
    TestDataY=np.empty([num_samples_test])

    for i in range (num_samples_test):
        Current_file_path_test = os.path.join(full_path_test,os.listdir(full_path_test)[i])
        Current_CSV=np.genfromtxt (Current_file_path_test, delimiter=",")
        TestDataX[i]=Current_CSV

        file_class=int(Current_file_path_test[-6:-4])
        TestDataY[i]=file_class-1 #first class is class 1, but now we start from 0

    TestDataY = to_categorical(TestDataY)
    
    if np.any(np.isnan(TestDataX)):
        print("DANGER!! NaN values found in DataX!!")
    
    if np.any(np.isnan(TestDataY)):
        print("DANGER!! NaN values found in DataY!!")

    return TestDataX, TestDataY
 
 
#----------------------------------------------------------------------
# DEFINITIONS END
#----------------------------------------------------------------------


#----------------------------------------------------------------------
# MAIN CODE START
#----------------------------------------------------------------------

train_folder="DATA\\1_Pump_3_Class_PM_dataset_(full_processed)\\train"
test_folder="DATA\\1_Pump_3_Class_PM_dataset_(full_processed)\\test"
now = datetime.now() # current date and time
output_folder="DATA\\NPZ_DATASET_{}".format(now.strftime("%Y%m%d_%H%M%S")) #Added auto-naming of output (from current date and time)

random_seed=100; 
#random_seed=random.randint(0,1000); #Uncomment for random splits between train and validation

for i in range (0, len(os.listdir(train_folder))):
    current_train_folder=os.path.join(train_folder,os.listdir(train_folder)[i]) #access folders by order
    current_test_folder=os.path.join(test_folder,os.listdir(test_folder)[i])   #test folder must have the exact same folders that train folder has    
    sensor_folder=os.path.basename(os.path.normpath(current_train_folder))   #sensor name by folder name for naming files
    
    current_output_folder=os.path.join(output_folder,sensor_folder)
    if not os.path.exists(current_output_folder): 
                    os.makedirs(current_output_folder)
    
    freq_samples, num_samples, tam_samples, num_features, X_train, X_test, y_train, y_test=Train_Set_Creator(current_train_folder, random_seed)
    TestDataX, TestDataY=Test_Set_Creator(current_test_folder)
    
    np.savez(os.path.join(current_output_folder,'train.npz'), x=X_train, y=y_train)
    np.savez(os.path.join(current_output_folder,'validation.npz'), x=X_test, y=y_test)
    np.savez(os.path.join(current_output_folder,'test.npz'), x=TestDataX, y=TestDataY)
