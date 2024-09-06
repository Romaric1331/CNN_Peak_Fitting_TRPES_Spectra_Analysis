# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 09:24:14 2024

@author: ajulien & Romaric
"""

#%% Importation of packages
import pickle
import numpy as np
from tensorflow.python.client import device_lib
import keras
import tensorflow as tf
import ML_models as MLm
from time import time
import os
import tensorflow_probability as tfp
import json
#%% Inputs
main_data_folder = "C:/Users/rsallustre/Documents/XPS_fitting_tests/"
session_name = "fifth_test_07-05-24"
data_folder = main_data_folder+session_name+"/"

epochs = 63


#%% Load database
database_folder = data_folder+"Database/"
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
        print("Reached error at " + str(i) +   " step")

validation_data = [val_peak,[np.array(val_center),
                             np.array(val_width),
                             np.array(val_amp),
                             np.array(val_peak_number)]]


train_val_database_str = '\n'.join(["Size of training database: "+str(train_n),
                                    "Size of validation database: "+str(val_n)])
print(train_val_database_str)


#%% Define function called for each model, taking care of model summary, compilation, training and history
# def train_validate_model(model,training_data,validation_data,epochs,model_name,model_folder):
#     # Print summary
#     model_summary_file = model_folder+model_name+"_summary.txt"
#     MLm.print_summary(model,model_summary_file)

#     # Plot summary
#     model_summary_file = model_folder+model_name+"_summary.png"
#     MLm.plot_summary(model,model_summary_file)

  
#     model.compile(optimizer='adam', loss='mae', metrics={'total_center3': 'mae', 'total_width3': 'mae', 'total_amp3': 'mae', 'total_peak_number3': 'mae'})

#     # Train the model
#     t0 = time()
#     history = model.fit(
#         x=training_data[0],
#         y=training_data[1],
#         epochs=epochs,
#         batch_size=32,
#         validation_data=validation_data,

#     )
#     train_time = time() - t0
#     train_time_str = f"Training time for {epochs} epochs: {format(train_time, '.0f')} s"

#   # Save the training history using json
#     with open(os.path.join(model_folder, f"{model_name}_history.json"), 'w') as f:
#         json.dump(history.history, f)
#         # Save the model in Keras format
#     model_file = os.path.join(model_folder, f"{model_name}.keras")
#     model.save(model_file)
#     model.save(model_file, save_format='tf')
    
#     return train_time_str
#%% Define Hyperparameter

import os
import json
import time
from tensorflow.keras.optimizers import Adam

def train_validate_model(model, training_data, validation_data, epochs, model_name, model_folder):
    # Print summary
    model_summary_file = os.path.join(model_folder, f"{model_name}_summary.txt")
    MLm.print_summary(model, model_summary_file)

    # Plot summary
    model_summary_plot = os.path.join(model_folder, f"{model_name}_summary.png")
    MLm.plot_summary(model, model_summary_plot)

    # Define the loss and metrics with the correct layer names
    losses = {
        'center_set1': 'mae', 'width_set1': 'mae', 'amplitude_set1': 'mae', 'peak_number_set1': 'mae',
        'center_set2': 'mae', 'width_set2': 'mae', 'amplitude_set2': 'mae', 'peak_number_set2': 'mae',
        'center_set3': 'mae', 'width_set3': 'mae', 'amplitude_set3': 'mae', 'peak_number_set3': 'mae',
        'center_set4': 'mae', 'width_set4': 'mae', 'amplitude_set4': 'mae', 'peak_number_set4': 'mae'
    }

    metrics = {
        'center_set1': 'mae', 'width_set1': 'mae', 'amplitude_set1': 'mae', 'peak_number_set1': 'mae',
        'center_set2': 'mae', 'width_set2': 'mae', 'amplitude_set2': 'mae', 'peak_number_set2': 'mae',
        'center_set3': 'mae', 'width_set3': 'mae', 'amplitude_set3': 'mae', 'peak_number_set3': 'mae',
        'center_set4': 'mae', 'width_set4': 'mae', 'amplitude_set4': 'mae', 'peak_number_set4': 'mae'
    }

    # Create the optimizer
    optimizer = Adam(
        learning_rate=0.000241,  # Example learning rate
    )

    # Compile the model with the correct loss and metrics
    model.compile(
        optimizer=optimizer,
        loss=losses,
        metrics=metrics
    )

    # Train the model
    t0 = time.time()
    history = model.fit(
        x=training_data[0],
        y=training_data[1],
        epochs=epochs,
        batch_size=32,
        validation_data=validation_data,
    )
    train_time = time.time() - t0
    train_time_str = f"Training time for {epochs} epochs: {format(train_time, '.0f')} s"

    # Save the training history using json
    with open(os.path.join(model_folder, f"{model_name}_history.json"), 'w') as f:
        json.dump(history.history, f)

    # Save the model in Keras format
    model_file = os.path.join(model_folder, f"{model_name}.keras")
    model.save(model_file)
    model.save(model_file, save_format='tf')
    
    return train_time_str

#%% H_Bayesian_Sparse_densenet
# model_name = "H_Bayesian_sparse_densenet" 
# print(model_name)

# # Model folder
# model_folder = data_folder+model_name+"/"
# os.makedirs(model_folder,exist_ok=True)

# model = MLm.define_H_bayesian_sparse_densenet(energy_n)

# # # Print, plot summary, compile, setup, train, validate model and save history
# train_time_str = train_validate_model(model,training_data,
#                                       validation_data,epochs,model_name,model_folder)
# print(train_time_str)


# # 
# # # Print text summary
# with open(model_folder+"Training_validation_summary.txt", 'w') as f:
#     f.writelines(model_name)
#     f.writelines('\n')
#     f.writelines(train_val_database_str)
#     f.writelines('\n')
#     f.writelines(train_time_str)
#     f.close()
#%% Bayesian_Sparse_densenet
# model_name = "Bayesian_sparse_densenet" 
# print(model_name)

# # Model folder
# model_folder = data_folder+model_name+"/"
# os.makedirs(model_folder,exist_ok=True)

# model = MLm.define_bayesian_sparse_densenet(energy_n)

# # # Print, plot summary, compile, setup, train, validate model and save history
# train_time_str = train_validate_model(model,training_data,
#                                       validation_data,epochs,model_name,model_folder)
# print(train_time_str)


# # 
# # # Print text summary
# with open(model_folder+"Training_validation_summary.txt", 'w') as f:
#     f.writelines(model_name)
#     f.writelines('\n')
#     f.writelines(train_val_database_str)
#     f.writelines('\n')
#     f.writelines(train_time_str)
#     f.close()


#%% BayesianCNN_densenet

# model_name = "BayesianCNN" 
# print(model_name)

# # Model folder
# model_folder = data_folder+model_name+"/"
# os.makedirs(model_folder,exist_ok=True)

# model = MLm.define_bayesian_cnn(energy_n)

# # # Print, plot summary, compile, setup, train, validate model and save history
# train_time_str = train_validate_model(model,training_data,
#                                       validation_data,epochs,model_name,model_folder)
# print(train_time_str)

# # # Print text summary
# with open(model_folder+"Training_validation_summary.txt", 'w') as f:
#     f.writelines(model_name)
#     f.writelines('\n')
#     f.writelines(train_val_database_str)
#     f.writelines('\n')
#     f.writelines(train_time_str)
#     f.close()


model_name = "BayesianCNN" 
print(model_name)

# Model folder
model_folder = data_folder + model_name + "/"
os.makedirs(model_folder, exist_ok=True)

# Define the model with four sets of outputs
model = MLm.define_bayesian_cnn(energy_n)

# Print, compile, setup, train, validate model, and save history
train_time_str = train_validate_model(
    model,
    training_data,
    validation_data,
    epochs,
    model_name,
    model_folder
)
print(train_time_str)

# Prepare training and validation summary string
train_val_summary = '\n'.join([
    f"Model Name: {model_name}",
    f"Training Time for {epochs} epochs: {train_time_str}",
    f"Size of Training Database: {len(training_data[0])}",
    f"Size of Validation Database: {len(validation_data[0])}"
])

# Print text summary
with open(model_folder + "Training_validation_summary.txt", 'w') as f:
    f.write(train_val_summary)


 #%% Sparse_densenet
# model_name = "Sparse_densenet" 
# print(model_name)

# # Model folder
# model_folder = data_folder+model_name+"/"
# os.makedirs(model_folder,exist_ok=True)

# # Define model structure
# model = MLm.define_sparse_densenet(energy_n)

# # # Print, plot summary, compile, setup, train, validate model and save history
# train_time_str = train_validate_model(model,training_data,
#                                       validation_data,epochs,model_name,model_folder)
# print(train_time_str)


# # # Print text summary
# with open(model_folder+"Training_validation_summary.txt", 'w') as f:
#     f.writelines(model_name)
#     f.writelines('\n')
#     f.writelines(train_val_database_str)
#     f.writelines('\n')
#     f.writelines(train_time_str)
#     f.close()

#%% SEResNet
# # =============================================================================
# model_name = "SEResNet"
# print(model_name)

# # Model folder
# model_folder = data_folder+model_name+"/"
# os.makedirs(model_folder,exist_ok=True)

# # Define model structure
# model = MLm.define_SEResNet(energy_n)  

# # Print, plot summary, compile, setup, train, validate model and save history
# train_time_str = train_validate_model(model,training_data,
#                                       validation_data,epochs,model_name,model_folder)
# print(train_time_str)

# # Print text summary
# with open(model_folder+"Training_validation_summary.txt", 'w') as f:
#     f.writelines(model_name)
#     f.writelines('\n')
#     f.writelines(train_val_database_str)
#     f.writelines('\n')
#     f.writelines(train_time_str)
#     f.close() 

# 
# 
# #%% ResNet
# model_name = "ResNet"
# print(model_name)
# 
# # # Model folder
# model_folder = data_folder+model_name+"/"
# os.makedirs(model_folder,exist_ok=True)
# 
#  # Define model structure
# model = MLm.define_ResNet(energy_n)
# 
#  # Print, plot summary, compile, setup, train, validate model and save history
# train_time_str = train_validate_model(model,training_data,
#                                        validation_data,epochs,model_name,model_folder)
# print(train_time_str)
# 
#  # Print text summary
# with open(model_folder+"Training_validation_summary.txt", 'w') as f:
#      f.writelines(model_name)
#      f.writelines('\n')
#      f.writelines(train_val_database_str)
#      f.writelines('\n')
#      f.writelines(train_time_str)
#      f.close() 
# 
# 
# 
# # #%% VGGNet
# cmodel_name = "VGGNet"
# print(model_name)
# 
# # # Model folder
# model_folder = data_folder+model_name+"/"
# os.makedirs(model_folder,exist_ok=True)
# 
# # # Define model structure
# model = MLm.define_VGGNet(energy_n)
# 
# # # Print, plot summary, compile, setup, train, validate model and save history
# train_time_str = train_validate_model(model,training_data,
#                                      validation_data,epochs,model_name,model_folder)
# print(train_time_str)
# 
# # # Print text summary
# with open(model_folder+"Training_validation_summary.txt", 'w') as f:
#      f.writelines(model_name)
#      f.writelines('\n')
#      f.writelines(train_val_database_str)
#      f.writelines('\n')
#      f.writelines(train_time_str)
#      f.close()
# 
# 
# 
# # #%% Alex_ZFNet
# model_name = "Alex_ZFNet"
# print(model_name)
# 
# # # Model folder
# model_folder = data_folder+model_name+"/"
# os.makedirs(model_folder,exist_ok=True)
# 
# # # Define model structure
# model = MLm.define_Alex_ZFNet(energy_n)
# 
# # # Print, plot summary, compile, setup, train, validate model and save history
# train_time_str = train_validate_model(model,training_data,
#                                       validation_data,epochs,model_name,model_folder)
# print(train_time_str)
# 
# # # Print text summary
# with open(model_folder+"Training_validation_summary.txt", 'w') as f:
#      f.writelines(model_name)
#      f.writelines('\n')
#      f.writelines(train_val_database_str)
#      f.writelines('\n')
#      f.writelines(train_time_str)
#      f.close()
# 
# 
# 
#%% LeNet
# model_name = "LeNet"
# print(model_name)

# # Model folder
# model_folder = data_folder+model_name+"/"
# os.makedirs(model_folder,exist_ok=True)

# # Define model structure
# model = MLm.define_LeNet(energy_n)

# # Print, plot summary, compile, setup, train, validate model and save history
# train_time_str = train_validate_model(model,training_data,
#                                       validation_data,epochs,model_name,model_folder)
# print(train_time_str)

# # Print text summary
# with open(model_folder+"Training_validation_summary.txt", 'w') as f:
#     f.writelines(model_name)
#     f.writelines('\n')
#     f.writelines(train_val_database_str)
#     f.writelines('\n')
#     f.writelines(train_time_str)
#     f.close()

# 
# 
# 
# =============================================================================
