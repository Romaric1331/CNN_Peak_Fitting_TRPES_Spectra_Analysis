# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 12:54:14 2024

@author: ajulien
"""


#%% Update Keras
!pip install keras --upgrade


#%% Importation of packages
import pickle
import numpy as np
from tensorflow.python.client import device_lib
import keras
import tensorflow as tf
import ML_models as MLm
from time import time
import os
from google.colab import files


#%% Inputs
folder = "/content/"

epochs = 50


#%% Load database
database_folder = folder
with open(database_folder+"Training_database.pkl", 'rb') as f:
    energy_range, train_peak_label, train_peak, train_peak_param = pickle.load(f)
train_n = len(train_peak_label)
energy_n = len(energy_range)

with open(database_folder+"Validation_database.pkl", 'rb') as f:
    energy_range, val_peak_label, val_peak, val_peak_param = pickle.load(f)
val_n = len(val_peak_label)


#%% Check compatibility with system
device_lib.list_local_devices()
print(keras.__version__)
print(tf.__version__)
print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))


#%% Redistributing labels for functional API
train_center = []
train_width = []
train_amp = []
train_peak_number = []
for i in range(train_n):
    try:
        train_center.append(train_peak_label[i][0])
        train_width.append(train_peak_label[i][1])
        train_amp.append(train_peak_label[i][2])
        train_peak_number.append(train_peak_label[i][3])
    except:
        print("Reached error at " + str(i) + " step")

training_data = [train_peak,[np.array(train_center),
                             np.array(train_width),
                             np.array(train_amp),
                             np.array(train_peak_number)]]

val_center = []
val_width = []
val_amp = []
val_peak_number = []
for i in range(val_n):
    try:
        val_center.append(val_peak_label[i][0])
        val_width.append(val_peak_label[i][1])
        val_amp.append(val_peak_label[i][2])
        val_peak_number.append(val_peak_label[i][3])
    except:
        print("Reached error at " + str(i) + " step")

validation_data = [val_peak,[np.array(val_center),
                             np.array(val_width),
                             np.array(val_amp),
                             np.array(val_peak_number)]]


train_val_database_str = '\n'.join(["Size of training database: "+str(train_n),
                                    "Size of validation database: "+str(val_n)])
print(train_val_database_str)


#%% Define function called for each model, taking care of model summary, compilation, training and history
def train_validate_model(model,training_data,validation_data,epochs,model_name,model_folder):

    # Compile, setup, train and validate model
    model_file = model_folder+model_name+".keras"
    t0 = time()
    model = MLm.compile_setup_train_validate(model,training_data,validation_data,epochs,model_file)
    train_time = time()-t0
    train_time_str = "Training time for "+str(epochs)+" epochs: "+format(train_time,".0f")+" s"

    # Save history
    with open(model_folder+model_name+"_history.pkl", 'wb') as f:
        pickle.dump(model.history, f)
    
    return train_time_str


#%% Sparse_densenet
model_name = "Sparse_densenet"
print(model_name)

# Model folder
model_folder = data_folder+model_name+"/"
os.makedirs(model_folder,exist_ok=True)

# Define model structure
model = MLm.define_sparse_densenet(energy_n)

# Print, plot summary, compile, setup, train, validate model and save history
train_time_str = train_validate_model(model,training_data,
                                      validation_data,epochs,model_name,model_folder)
print(train_time_str)

# Print text summary
with open(model_folder+model_name+"_training_validation_summary.txt", 'w') as f:
    f.writelines(model_name)
    f.writelines('\n')
    f.writelines(train_val_database_str)
    f.writelines('\n')
    f.writelines(train_time_str)
    f.close()

files.download(model_folder+model_name+".keras")
files.download(model_folder+model_name+"_history.pkl")
files.download(model_folder+model_name+"_training_validation_summary.txt")

#%% SEResNet
model_name = "SEResNet"
print(model_name)

# Model folder
model_folder = data_folder+model_name+"/"
os.makedirs(model_folder,exist_ok=True)

# Define model structure
model = MLm.define_SEResNet(energy_n)

# Print, plot summary, compile, setup, train, validate model and save history
train_time_str = train_validate_model(model,training_data,
                                      validation_data,epochs,model_name,model_folder)
print(train_time_str)

# Print text summary
with open(model_folder+model_name+"_training_validation_summary.txt", 'w') as f:
    f.writelines(model_name)
    f.writelines('\n')
    f.writelines(train_val_database_str)
    f.writelines('\n')
    f.writelines(train_time_str)
    f.close()

files.download(model_folder+model_name+".keras")
files.download(model_folder+model_name+"_history.pkl")
files.download(model_folder+model_name+"_training_validation_summary.txt")


#%% ResNet
model_name = "ResNet"
print(model_name)

# Model folder
model_folder = data_folder+model_name+"/"
os.makedirs(model_folder,exist_ok=True)

# Define model structure
model = MLm.define_ResNet(energy_n)

# Print, plot summary, compile, setup, train, validate model and save history
train_time_str = train_validate_model(model,training_data,
                                      validation_data,epochs,model_name,model_folder)
print(train_time_str)

# Print text summary
with open(model_folder+model_name+"_training_validation_summary.txt", 'w') as f:
    f.writelines(model_name)
    f.writelines('\n')
    f.writelines(train_val_database_str)
    f.writelines('\n')
    f.writelines(train_time_str)
    f.close()

files.download(model_folder+model_name+".keras")
files.download(model_folder+model_name+"_history.pkl")
files.download(model_folder+model_name+"_training_validation_summary.txt")


#%% VGGNet
model_name = "VGGNet"
print(model_name)

# Model folder
model_folder = data_folder+model_name+"/"
os.makedirs(model_folder,exist_ok=True)

# Define model structure
model = MLm.define_VGGNet(energy_n)

# Print, plot summary, compile, setup, train, validate model and save history
train_time_str = train_validate_model(model,training_data,
                                      validation_data,epochs,model_name,model_folder)
print(train_time_str)

# Print text summary
with open(model_folder+model_name+"_training_validation_summary.txt", 'w') as f:
    f.writelines(model_name)
    f.writelines('\n')
    f.writelines(train_val_database_str)
    f.writelines('\n')
    f.writelines(train_time_str)
    f.close()

files.download(model_folder+model_name+".keras")
files.download(model_folder+model_name+"_history.pkl")
files.download(model_folder+model_name+"_training_validation_summary.txt")


#%% Alex_ZFNet
model_name = "Alex_ZFNet"
print(model_name)

# Model folder
model_folder = data_folder+model_name+"/"
os.makedirs(model_folder,exist_ok=True)

# Define model structure
model = MLm.define_Alex_ZFNet(energy_n)

# Print, plot summary, compile, setup, train, validate model and save history
train_time_str = train_validate_model(model,training_data,
                                      validation_data,epochs,model_name,model_folder)
print(train_time_str)

# Print text summary
with open(model_folder+model_name+"_training_validation_summary.txt", 'w') as f:
    f.writelines(model_name)
    f.writelines('\n')
    f.writelines(train_val_database_str)
    f.writelines('\n')
    f.writelines(train_time_str)
    f.close()

files.download(model_folder+model_name+".keras")
files.download(model_folder+model_name+"_history.pkl")
files.download(model_folder+model_name+"_training_validation_summary.txt")


#%% LeNet
model_name = "LeNet"
print(model_name)

# Model folder
model_folder = data_folder+model_name+"/"
os.makedirs(model_folder,exist_ok=True)

# Define model structure
model = MLm.define_LeNet(energy_n)

# Print, plot summary, compile, setup, train, validate model and save history
train_time_str = train_validate_model(model,training_data,
                                      validation_data,epochs,model_name,model_folder)
print(train_time_str)

# Print text summary
with open(model_folder+model_name+"_training_validation_summary.txt", 'w') as f:
    f.writelines(model_name)
    f.writelines('\n')
    f.writelines(train_val_database_str)
    f.writelines('\n')
    f.writelines(train_time_str)
    f.close()

files.download(model_folder+model_name+".keras")
files.download(model_folder+model_name+"_history.pkl")
files.download(model_folder+model_name+"_training_validation_summary.txt")




