# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 13:15:29 2024

@author: ajulien
"""

#%% Importation of packages
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import ML_models as MLm
import os
import time

from ML_models import MyConvLayer, scale_invariant_kl


#%% Inputs
main_data_folder = "C:/Users/rsallustre/Documents/XPS_fitting/"
session_name = "Tenth_test_25-06-24"
data_folder = main_data_folder+session_name+"/"


#%% Load test database
database_folder = data_folder+"Database/"
with open(database_folder+"Test_database.pkl", 'rb') as f:
    energy_range, test_peak_label, test_peak, test_peak_param = pickle.load(f)
test_n = len(test_peak_label)
energy_n = len(energy_range)

test_center = []
test_width = []

test_amp = []
test_peak_number = []
for i in range(test_n):
    try:
        test_center.append(test_peak_label[i][0])
        test_width.append(test_peak_label[i][1])
        test_amp.append(test_peak_label[i][2])
        test_peak_number.append(test_peak_label[i][3])
    except:
        print("Reached error at " + str(i) + " step")

test_data = [test_peak,[np.array(test_center),
                        np.array(test_width),
                        np.array(test_amp),
                        np.array(test_peak_number)]]

test_database_str = "Size of test database: "+str(test_n)
print(test_database_str)


import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import load_model
import time

# def scale_invariant_kl(q, p, _):
#     return tfp.distributions.kl_divergence(q, p) / tf.cast(tf.reduce_prod(q.batch_shape_tensor()), tf.float32)

model_name = "BayesianCNN"
model_folder = data_folder + model_name + "/"
model_file = model_folder + model_name + ".keras"

custom_objects = {
    'MyConvLayer': MyConvLayer,
    'scale_invariant_kl': scale_invariant_kl,
    'tfp': tfp,
}

try:
    model = tf.keras.models.load_model(model_file, custom_objects=custom_objects)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Test the model
t0 = time.time()
bayesian_cnn_test, model_MAE = MLm.test_model(model, test_data)
fit_time = (time.time() - t0) / test_n

# Format and print the results
model_MAE_str = '\n'.join([
    f"{model_name}:",
    f"MAE on center: {model_MAE[0]:.2f}",
    f"MAE on width: {model_MAE[1]:.2f}",
    f"MAE on amplitude: {model_MAE[2]:.2f}",
    f"MAE on peak number: {model_MAE[3]:.2f}",
    f"Avg fit time: {fit_time:.2e} s"
])
print(model_MAE_str)

# Save the results to a file
with open(model_folder + "Test_summary.txt", 'w') as f:
    f.write(test_database_str + '\n')
    f.write(model_MAE_str)
# model_name = "BayesianCNN"
# custom_objects = {
#     'MyConvLayer': MyConvLayer,
#     'scale_invariant_kl': scale_invariant_kl
# }
# model_folder = data_folder + model_name + "/"
# model_file = model_folder + model_name + ".keras"

# # Load the model with custom objects
# model = load_model(model_file, custom_objects=custom_objects)

# # Test the model
# t0 = time.time()
# bayesian_cnn_test, model_MAE = MLm.test_model(model, test_data)
# fit_time = (time.time() - t0) / test_n

# # Format and print the results
# model_MAE_str = '\n'.join([
#     f"{model_name}:",
#     f"MAE on center: {model_MAE[0]:.2f}",
#     f"MAE on width: {model_MAE[1]:.2f}",
#     f"MAE on amplitude: {model_MAE[2]:.2f}",
#     f"MAE on peak number: {model_MAE[3]:.2f}",
#     f"Avg fit time: {fit_time:.2e} s"
# ])
# print(model_MAE_str)

# # Save the results to a file
# with open(model_folder + "Test_summary.txt", 'w') as f:
#     f.write(test_database_str + '\n')
#     f.write(model_MAE_str)


# %% Sparse_densenet
model_name = "Sparse_densenet"

model_folder = data_folder+model_name+"/"
model_file = model_folder+model_name+".keras"

model = load_model(model_file)
t0 = time.time()
sparse_densenet_test,model_MAE = MLm.test_model(model,test_data)
fit_time = (time.time()-t0)/test_n

model_MAE_str = '\n'.join([model_name+":",
                            "MAE on center: "+format(model_MAE[0],".2f"),
                            "MAE on width: "+format(model_MAE[1],".2f"),
                            "MAE on amplitude: "+format(model_MAE[2],".2f"),
                            "MAE on peak number: "+format(model_MAE[3],".2f"),
                            "Avg fit time: "+format(fit_time,".2e")+" s"])
print(model_MAE_str)

# Print text summary
with open(model_folder+"Test_summary.txt", 'w') as f:
    f.writelines(test_database_str)
    f.writelines('\n')
    f.writelines(model_MAE_str)
    f.close()
    



    
# #%% SEResNet
# model_name = "SEResNet"
# model_folder = data_folder+model_name+"/"
# model_file = model_folder+model_name+".keras"

# model = load_model(model_file)
# t0 = time.time()
# SEResNet_test,model_MAE = MLm.test_model(model,test_data)
# fit_time = (time.time()-t0)/test_n

# model_MAE_str = '\n'.join([model_name+":",
#                            "MAE on center: "+format(model_MAE[0],".2f"),
#                            "MAE on width: "+format(model_MAE[1],".2f"),
#                            "MAE on amplitude: "+format(model_MAE[2],".2f"),
#                            "MAE on peak number: "+format(model_MAE[3],".2f"),
#                            "Avg fit time: "+format(fit_time,".2e")+" s"])
# print(model_MAE_str)

# # Print text summary
# with open(model_folder+"Test_summary.txt", 'w') as f:
#     f.writelines(test_database_str)
#     f.writelines('\n')
#     f.writelines(model_MAE_str)
#     f.close()


# #%% ResNet
# model_name = "ResNet"
# model_folder = data_folder+model_name+"/"
# model_file = model_folder+model_name+".keras"

# model = load_model(model_file)
# t0 = time.time()
# ResNet_test,model_MAE = MLm.test_model(model,test_data)
# fit_time = (time.time()-t0)/test_n

# model_MAE_str = '\n'.join([model_name+":",
#                            "MAE on center: "+format(model_MAE[0],".2f"),
#                            "MAE on width: "+format(model_MAE[1],".2f"),
#                            "MAE on amplitude: "+format(model_MAE[2],".2f"),
#                            "MAE on peak number: "+format(model_MAE[3],".2f"),
#                            "Avg fit time: "+format(fit_time,".2e")+" s"])
# print(model_MAE_str)

# # Print text summary
# with open(model_folder+"Test_summary.txt", 'w') as f:
#     f.writelines(test_database_str)
#     f.writelines('\n')
#     f.writelines(model_MAE_str)
#     f.close()


#%% VGGNet
# =============================================================================
# model_name = "VGGNet"
# model_folder = data_folder+model_name+"/"
# model_file = model_folder+model_name+".keras"
# 
# model = load_model(model_file)
# t0 = time.time()
# VGGNet_test,model_MAE = MLm.test_model(model,test_data)
# fit_time = (time.time()-t0)/test_n
# 
# model_MAE_str = '\n'.join([model_name+":",
#                            "MAE on center: "+format(model_MAE[0],".2f"),
#                            "MAE on width: "+format(model_MAE[1],".2f"),
#                            "MAE on amplitude: "+format(model_MAE[2],".2f"),
#                            "MAE on peak number: "+format(model_MAE[3],".2f"),
#                            "Avg fit time: "+format(fit_time,".2e")+" s"])
# print(model_MAE_str)
# 
# # Print text summary
# with open(model_folder+"Test_summary.txt", 'w') as f:
#     f.writelines(test_database_str)
#     f.writelines('\n')
#     f.writelines(model_MAE_str)
#     f.close()
# =============================================================================


# #%% Alex_ZFNet
# model_name = "Alex_ZFNet"
# model_folder = data_folder+model_name+"/"
# model_file = model_folder+model_name+".keras"

# model = load_model(model_file)
# t0 = time.time()
# Alex_ZFNet_test,model_MAE = MLm.test_model(model,test_data)
# fit_time = (time.time()-t0)/test_n

# model_MAE_str = '\n'.join([model_name+":",
#                            "MAE on center: "+format(model_MAE[0],".2f"),
#                            "MAE on width: "+format(model_MAE[1],".2f"),
#                            "MAE on amplitude: "+format(model_MAE[2],".2f"),
#                            "MAE on peak number: "+format(model_MAE[3],".2f"),
#                            "Avg fit time: "+format(fit_time,".2e")+" s"])
# print(model_MAE_str)

# # Print text summary
# with open(model_folder+"Test_summary.txt", 'w') as f:
#     f.writelines(test_database_str)
#     f.writelines('\n')
#     f.writelines(model_MAE_str)
#     f.close()


# #%% LeNet
# model_name = "LeNet"
# model_folder = data_folder+model_name+"/"
# model_file = model_folder+model_name+".keras"

# model = load_model(model_file)
# t0 = time.time()
# LeNet_test,model_MAE = MLm.test_model(model,test_data)
# fit_time = (time.time()-t0)/test_n

# model_MAE_str = '\n'.join([model_name+":",
#                            "MAE on center: "+format(model_MAE[0],".2f"),
#                            "MAE on width: "+format(model_MAE[1],".2f"),
#                            "MAE on amplitude: "+format(model_MAE[2],".2f"),
#                            "MAE on peak number: "+format(model_MAE[3],".2f"),
#                            "Avg fit time: "+format(fit_time,".2e")+" s"])
# print(model_MAE_str)

# # Print text summary
# with open(model_folder+"Test_summary.txt", 'w') as f:
#     f.writelines(test_database_str)
#     f.writelines('\n')
#     f.writelines(model_MAE_str)
#     f.close()


#%% Plot one example of fitted spectrum
# figures_folder = data_folder+"Figures_models_test/"
# os.makedirs(figures_folder,exist_ok=True)

# result_i = 102

# # Define pseudo-Voigt function
# def pseudo_Voigt(a, b, c, x):
#     beta = 5.09791537e-01
#     gamma = 4.41140472e-01
#     y = c * ((0.7 * np.exp(-np.log(2) * (x - a) ** 2 / (beta * b) ** 2))
#         + (0.3 / (1 + (x - a) ** 2 / (gamma * b) ** 2)))
#     return y


# plt.figure(figsize=(720/100, 360/100))
# plt.plot(energy_range,test_peak[result_i],label="Test spectrum")
# plt.plot(energy_range,pseudo_Voigt(test_peak_label[result_i][0],test_peak_label[result_i][1],
#                                    test_peak_label[result_i][2],energy_range),label="True major peak")
# plt.plot(energy_range,pseudo_Voigt(*sparse_densenet_test[result_i,0:3],energy_range), label="Sparse densenet")
# plt.plot(energy_range,pseudo_Voigt(*sparse_densenet_test[result_i,0:3],energy_range), label="BayesianCNN")
# # plt.plot(energy_range,pseudo_Voigt(*SEResNet_test[result_i,0:3],energy_range), label="SEResNet")
# # plt.plot(energy_range,pseudo_Voigt(*ResNet_test[result_i,0:3],energy_range), label="ResNet")
# # #plt.plot(energy_range,pseudo_Voigt(*VGGNet_test[result_i,0:3],energy_range), label="VGGNet")
# # plt.plot(energy_range,pseudo_Voigt(*Alex_ZFNet_test[result_i,0:3],energy_range), label="Alex ZFNet")
# # plt.plot(energy_range,pseudo_Voigt(*LeNet_test[result_i,0:3],energy_range), label="LeNet")
# plt.grid(True)
# #plt.ylim(0,2)
# plt.xlabel("Energy")
# plt.ylabel("Intensity")
# plt.legend()
# plt.tight_layout()
# plt.savefig(figures_folder+"Fit_example_"+str(result_i)+".jpg",dpi=300)

figures_folder = data_folder + "Figures_models_test/"
os.makedirs(figures_folder, exist_ok=True)

# Ensure result_i is within the valid range
result_i = min(102, len(test_peak) - 1)  # This will use 98 if test_peak has 99 elements

# Define pseudo-Voigt function
def pseudo_Voigt(a, b, c, x):
    beta = 5.09791537e-01
    gamma = 4.41140472e-01
    y = c * ((0.7 * np.exp(-np.log(2) * (x - a) ** 2 / (beta * b) ** 2))
        + (0.3 / (1 + (x - a) ** 2 / (gamma * b) ** 2)))
    return y

plt.figure(figsize=(720/100, 360/100))
plt.plot(energy_range, test_peak[result_i], label="Test spectrum")
plt.plot(energy_range, pseudo_Voigt(test_peak_label[result_i][0], test_peak_label[result_i][1],
                                   test_peak_label[result_i][2], energy_range), label="True major peak")
plt.plot(energy_range, pseudo_Voigt(*sparse_densenet_test[result_i,0:3], energy_range), label="Sparse densenet")
plt.plot(energy_range, pseudo_Voigt(*bayesian_cnn_test[result_i,0:3], energy_range), label="BayesianCNN")

plt.grid(True)
#plt.ylim(0,2)
plt.xlabel("Energy")
plt.ylabel("Intensity")
plt.legend()
plt.tight_layout()
plt.savefig(figures_folder + f"Fit_example_{result_i}.jpg", dpi=300)
plt.close()  # Close the figure to free up memory

# Print some information about the arrays
print(f"Number of test spectra: {len(test_peak)}")
print(f"Shape of sparse_densenet_test: {sparse_densenet_test.shape}")
print(f"Shape of bayesian_cnn_test: {bayesian_cnn_test.shape}")
print(f"Using result_i: {result_i}")

