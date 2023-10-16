#----------------------------------------------------------------------
#
# SUPERTRAINER.PY
#
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
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import LSTM, GRU, Dropout
from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.utils import to_categorical

import sys

from tensorflow.python.client import device_lib

# More imports
import matplotlib.pyplot as plt #for plotting raw data
import os #for system folders and files management
import pathlib #for system folders and files management
import pandas as pd #for Dataframe handling

from datetime import datetime #getting current time for naming the result file

from IPython.display import display # for folder selection dialog

import numpy as np
import math
import csv

import re #for character extraction

#for metrics and plots
import sklearn.model_selection
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import seaborn as sns #for plotting confusion matrix

from sklearn.model_selection import train_test_split

#----------------------------------------------------------------------
# IMPORTS END
#----------------------------------------------------------------------


#----------------------------------------------------------------------
# DEFINITIONS
#----------------------------------------------------------------------

def create_1layer_lstm_model(nodes,drp,input_shape):
    
    #input_shape = (None, 550, 3)

    rnn_model = Sequential()

    rnn_model.add(BatchNormalization(batch_input_shape = input_shape))
    rnn_model.add(LSTM((nodes)))
    rnn_model.add(Dropout(drp))
    rnn_model.add(Dense(3, activation='softmax'))
    
    return rnn_model


def create_1layer_gru_model(nodes,drp,input_shape):
    
    #input_shape = (None, 550, 3)

    rnn_model = Sequential()

    rnn_model.add(BatchNormalization(batch_input_shape = input_shape))
    rnn_model.add(GRU((nodes)))
    rnn_model.add(Dropout(drp))
    rnn_model.add(Dense(3, activation='softmax'))
    
    return rnn_model


def create_2layer_lstm_model(nodes_1stl,nodes_2ndl,drp,input_shape):
    
    #input_shape = (None, 550, 3)

    rnn_model = Sequential()

    rnn_model.add(BatchNormalization(batch_input_shape = input_shape))
    rnn_model.add(LSTM((nodes_1stl), return_sequences=True))
    rnn_model.add(Dropout(drp))
    rnn_model.add(LSTM((nodes_2ndl)))
    rnn_model.add(Dropout(drp))
    rnn_model.add(Dense(3, activation='softmax'))
    
    return rnn_model


def create_2layer_gru_model(nodes_1stl,nodes_2ndl,drp,input_shape):
    
    #input_shape = (None, 550, 3)

    rnn_model = Sequential()

    rnn_model.add(BatchNormalization(batch_input_shape = input_shape))
    rnn_model.add(GRU((nodes_1stl), return_sequences=True))
    rnn_model.add(Dropout(drp))
    rnn_model.add(GRU((nodes_2ndl)))
    rnn_model.add(Dropout(drp))
    rnn_model.add(Dense(3, activation='softmax'))
    
    return rnn_model


def Model_creation(RNN_type, layers, nodes, sensor_model_folder, freq_samples, tam_samples, num_features):

    #Parameters
    nodes_1 = nodes # Specify nodes for first layer 32
    nodes_2 = int(nodes_1/2) # Specify nodes for second layer 16
    drp = 0.3 # Specify droput rate

    input_shape=(None,tam_samples, num_features) #this should not be changed since it depends on the loaded file

    if RNN_type == 'GRU':
        if layers == 1:
            model_RNN = create_1layer_gru_model(nodes_1,drp,input_shape)
            model_RNN_name = 'GRU_1l_{0:0>2}_{1:0>4}Hz'.format(nodes_1,freq_samples)
        elif layers == 2:
            model_RNN = create_2layer_gru_model(nodes_1,nodes_2,drp,input_shape)
            model_RNN_name = 'GRU_2l_{0:0>2}_{1:0>2}_{2:0>4}Hz'.format(nodes_1,nodes_2,freq_samples)
        else:
            print("ERROR, incorrect parameters")
    elif RNN_type == 'LSTM':
        if layers == 1:
            model_RNN = create_1layer_lstm_model(nodes_1,drp,input_shape)
            model_RNN_name = 'LSTM_1l_{0:0>2}_{1:0>4}Hz'.format(nodes_1,freq_samples)
        elif layers == 2:
            model_RNN = create_2layer_lstm_model(nodes_1,nodes_2,drp,input_shape)
            model_RNN_name = 'LSTM_2l_{0:0>2}_{1:0>2}_{2:0>4}Hz'.format(nodes_1,nodes_2,freq_samples)
        else:
            print("ERROR, incorrect parameters")
    else:
        print("ERROR, incorrect parameters")

          
    #model_RNN.summary()

    #Model save

           
    #set filename    
    filename_txt="Summary_{}.txt".format(model_RNN_name)

    #save summary report in a txt
    with open(os.path.join(sensor_model_folder,filename_txt), 'w') as f:
        model_RNN.summary(print_fn=lambda x: f.write(x + '\n'))
    f.close()

    #Compile

    opt = tf.keras.optimizers.Adam(learning_rate=0.002)

    model_RNN.compile(loss='categorical_crossentropy',
            optimizer=opt,
            metrics=["accuracy"])
    
    return model_RNN, model_RNN_name


def Model_trainer(model_RNN, model_RNN_name, sensor_model_folder, X_train, y_train):

    #Train

    class train_print_cb(keras.callbacks.Callback):

        def on_epoch_end(self, epoch, logs=None):
            keys = list(logs.keys())
            watch1 = keys[0] # watching first key (loss)
            print(f'epoch {epoch} {watch1}: {logs[watch1]:.3f}          ', end = '\r')


    epochs = 1000
    batch_size = 32

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "best_model.h5", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=120, verbose=1),
        train_print_cb()
    ]
    #print("Training starting soon...")
    with tf.device('/gpu:0'):
        history = model_RNN.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_data = (X_test, y_test),
            verbose=0,
        )
        
    #print("Training done!!")

    #Model save
    best_model_RNN_name = 'best_model_{}.h5'.format(model_RNN_name)

    model_RNN.save(os.path.join(sensor_model_folder, best_model_RNN_name))
    #print("Model succesfully saved in {}".format(sensor_model_folder))

    #save progression graph
    metric = "accuracy"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")

    figname_pdf="MT_{}.pdf".format(model_RNN_name)
    figname_png="MT_{}.png".format(model_RNN_name)
    plt.savefig(os.path.join(sensor_model_folder,figname_pdf))
    plt.savefig(os.path.join(sensor_model_folder,figname_png))

    #plt.show()
    plt.close()
    return model_RNN


def Model_evaluator(model, sensor_model_folder, model_RNN_name, TestDataX, TestDataY):

    pred = model.predict(TestDataX)

    # CONF MATRIX
    y_pred = np.argmax(pred, axis = 1)
    p = sklearn.metrics.confusion_matrix(TestDataY.argmax(axis=1), y_pred, labels=[0,1,2])
    p_norm = sklearn.metrics.confusion_matrix(TestDataY.argmax(axis=1), y_pred, labels=[0,1,2], normalize='true')

    T_lables = ['STOP','NEW','OLD']    

    ax= plt.subplot()

    # Set values format
    values = ["{0:0.0f}".format(x) for x in p.flatten()]

    # Find percentages and set format
    percentages = ["{0:.1%}".format(x) for x in p_norm.flatten()]

    # Combine classes, values and percentages to show 
    combined = [f"{i}\n{j}" for i, j in zip(values, percentages)]
    combined = np.asarray(combined).reshape(3,3)

    sns.heatmap(p_norm, annot=combined, fmt='', ax=ax, cmap="Greens", cbar_kws={"ticks":[0,0.5,1], "format":'%.1f'});  #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(T_lables); ax.yaxis.set_ticklabels(T_lables);

    # save the figure
    figname_pdf="CM_{}.pdf".format(model_RNN_name)
    figname_png="CM_{}.png".format(model_RNN_name)
    plt.savefig(os.path.join(sensor_model_folder,figname_pdf))
    plt.savefig(os.path.join(sensor_model_folder,figname_png))

    plt.close()

    # REPORT
    report = sklearn.metrics.classification_report(TestDataY.argmax(axis=1),
                                                   y_pred,
                                                   target_names=['STOP','NEW', 'OLD'],
                                                   output_dict=True)

    report_filename="Report_{}.csv".format(model_RNN_name)

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(os.path.join(sensor_model_folder,report_filename))
    
# TRAIN SET CREATOR
def Train_Set_Creator(train_folder):
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


    X_train, X_test, y_train, y_test = train_test_split(DataX, DataY, test_size=0.15, random_state=100, stratify=DataY)
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

    #Select root folder train
    #Select root folder test
    #iterate folders inside root folders
    #  for each folder, iterate different NN: GRU and LSTM with 1-2 layers, 2-64 nodes, and validate


        
#set to True for having debug info on the rest of the code
tf.debugging.set_log_device_placement(False)

train_folder="DATA\\1_Pump_3_Class_PM_dataset_(full_processed)\\train"
test_folder="DATA\\1_Pump_3_Class_PM_dataset_(full_processed)\\test"
now = datetime.now() # current date and time
output_folder="DATA\\models_{}".format(now.strftime("%Y%m%d_%H%M%S")) #Added auto-naming of output (from current date and time)
start_at_folder=0 #For skipping first folders (if 0 starts from the beginning)

for i in range(start_at_folder, len(os.listdir(train_folder))): 
    print("---------------\nStarting folder {} of {}\n---------------".format(i+1, len(os.listdir(train_folder))))
    current_train_folder=os.path.join(train_folder,os.listdir(train_folder)[i]) #access folders by order
    current_test_folder=os.path.join(test_folder,os.listdir(test_folder)[i])   #test folder must have the exact same folders that train folder has    
    sensor_folder=os.path.basename(os.path.normpath(current_train_folder))   #sensor name by folder name for naming files
    
    current_output_folder=os.path.join(output_folder,sensor_folder)
    if not os.path.exists(current_output_folder): 
                    os.makedirs(current_output_folder)
    
    freq_samples, num_samples, tam_samples, num_features, X_train, X_test, y_train, y_test=Train_Set_Creator(current_train_folder)
    TestDataX, TestDataY=Test_Set_Creator(current_test_folder)
    for layers in range (1,3,1):
        nodes=2
        while nodes <=64:
            print("GRU {} layers and {} nodes".format(layers, nodes))
            model, model_name=Model_creation("GRU", layers, nodes, current_output_folder, freq_samples, tam_samples, num_features)
            #print(model_name)
            model=Model_trainer(model, model_name, current_output_folder, X_train, y_train)
            Model_evaluator(model, current_output_folder, model_name, TestDataX, TestDataY)        
            nodes=nodes*2
            
        nodes=2
        while nodes <=64:
            print("LSTM {} layers and {} nodes".format(layers, nodes))
            model, model_name=Model_creation("LSTM", layers, nodes, current_output_folder, freq_samples, tam_samples, num_features)
            #print(model_name)
            model=Model_trainer(model, model_name, current_output_folder, X_train, y_train)
            Model_evaluator(model, current_output_folder, model_name, TestDataX, TestDataY)        
            nodes=nodes*2
            
        



