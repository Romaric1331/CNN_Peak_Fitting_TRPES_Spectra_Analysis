# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 13:15:29 2024

@author: ajulien & Romaric
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
main_data_folder = "C:/Users/rsallustre/Documents/XPS_fitting_tests/"
session_name = "fifth_test_07-05-24"
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

# %% Bayesian_sparse_densenet
# model_name = "Bayesian_sparse_densenet"

# model_folder = data_folder + model_name + "/"
# model_file = model_folder + model_name + ".keras"

# model = load_model(model_file)
# t0 = time.time()
# Bayesian_sparse_densenet_test, model_MAE, model_MSE, model_RMSE = MLm.test_model(model, test_data)
# fit_time = (time.time() - t0) / test_n

# # Prepare the summary strings
# model_MAE_str = '\n'.join([
#     model_name + ":",
#     "MAE on center: " + format(model_MAE[0], ".2f"),
#     "MAE on width: " + format(model_MAE[1], ".2f"),
#     "MAE on amplitude: " + format(model_MAE[2], ".2f"),
#     "MAE on peak number: " + format(model_MAE[3], ".2f")
# ])

# model_MSE_str = '\n'.join([
#     "MSE on center: " + format(model_MSE[0], ".2f"),
#     "MSE on width: " + format(model_MSE[1], ".2f"),
#     "MSE on amplitude: " + format(model_MSE[2], ".2f"),
#     "MSE on peak number: " + format(model_MSE[3], ".2f")
# ])

# model_RMSE_str = '\n'.join([
#     "RMSE on center: " + format(model_RMSE[0], ".2f"),
#     "RMSE on width: " + format(model_RMSE[1], ".2f"),
#     "RMSE on amplitude: " + format(model_RMSE[2], ".2f"),
#     "RMSE on peak number: " + format(model_RMSE[3], ".2f")
# ])

# fit_time_str = "Avg fit time: " + format(fit_time, ".2e") + " s"

# # Print the summary
# print(model_MAE_str)
# print("\n")  # Add a newline for spacing
# print(model_MSE_str)
# print("\n")  # Add a newline for spacing
# print(model_RMSE_str)
# print("\n")  # Add a newline for spacing
# print(fit_time_str)

# # Write the summary to a file
# with open(model_folder + "Test_summary.txt", 'w') as f:
#     f.writelines(test_database_str)
#     f.writelines('\n')
#     f.writelines(model_MAE_str)
#     f.writelines('\n')
#     f.writelines(model_MSE_str)
#     f.writelines('\n')
#     f.writelines(model_RMSE_str)
#     f.writelines('\n')
#     f.writelines(fit_time_str)
#     f.close()

# model_name = "Bayesian_sparse_densenet"

# model_folder = data_folder+model_name+"/"
# model_file = model_folder+model_name+".keras"

# model = load_model(model_file)
# t0 = time.time()
# Bayesian_sparse_densenet_test,model_MAE = MLm.test_model(model,test_data)
# fit_time = (time.time()-t0)/test_n

# model_MAE_str = '\n'.join([model_name+":",
#                             "MAE on center: "+format(model_MAE[0],".2f"),
#                             "MAE on width: "+format(model_MAE[1],".2f"),
#                             "MAE on amplitude: "+format(model_MAE[2],".2f"),
#                             "MAE on peak number: "+format(model_MAE[3],".2f"),
#                             "Avg fit time: "+format(fit_time,".2e")+" s"])
# print(model_MAE_str)

# # Print text summary
# with open(model_folder+"Test_summary.txt", 'w') as f:
#     f.writelines(test_database_str)
#     f.writelines('\n')
#     f.writelines(model_MAE_str)
#     f.close()
# %% BayesianSparse Densenet
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import load_model
import time




# model_name = "Bayesian_sparse_densenet"

# model_folder = data_folder + model_name + "/"
# model_file = model_folder + model_name + ".keras"

# model = load_model(model_file)
# t0 = time.time()
# Bayesian_sparse_densenet_test, model_MAE, model_MSE, model_RMSE = MLm.test_model(model, test_data)
# fit_time = (time.time() - t0) / test_n

# # Prepare the summary strings
# model_MAE_str = '\n'.join([
#     model_name + ":",
#     "MAE on center: " + format(model_MAE[0], ".2f"),
#     "MAE on width: " + format(model_MAE[1], ".2f"),
#     "MAE on amplitude: " + format(model_MAE[2], ".2f"),
#     "MAE on peak number: " + format(model_MAE[3], ".2f")
# ])

# model_MSE_str = '\n'.join([
#     "MSE on center: " + format(model_MSE[0], ".2f"),
#     "MSE on width: " + format(model_MSE[1], ".2f"),
#     "MSE on amplitude: " + format(model_MSE[2], ".2f"),
#     "MSE on peak number: " + format(model_MSE[3], ".2f")
# ])

# model_RMSE_str = '\n'.join([
#     "RMSE on center: " + format(model_RMSE[0], ".2f"),
#     "RMSE on width: " + format(model_RMSE[1], ".2f"),
#     "RMSE on amplitude: " + format(model_RMSE[2], ".2f"),
#     "RMSE on peak number: " + format(model_RMSE[3], ".2f")
# ])

# fit_time_str = "Avg fit time: " + format(fit_time, ".2e") + " s"

# # Print the summary
# print(model_MAE_str)
# print("\n")  # Add a newline for spacing
# print(model_MSE_str)
# print("\n")  # Add a newline for spacing
# print(model_RMSE_str)
# print("\n")  # Add a newline for spacing
# print(fit_time_str)

# # Write the summary to a file
# with open(model_folder + "Test_summary.txt", 'w') as f:
#     f.writelines(test_database_str)
#     f.writelines('\n')
#     f.writelines(model_MAE_str)
#     f.writelines('\n')
#     f.writelines(model_MSE_str)
#     f.writelines('\n')
#     f.writelines(model_RMSE_str)
#     f.writelines('\n')
#     f.writelines(fit_time_str)
#     f.close()



#%% Bayesian CNN



model_name = "BayesianCNN"

model_folder = data_folder + model_name + "/"
model_file = model_folder + model_name + ".keras"

model = load_model(model_file)
t0 = time.time()
BayesianCNN_test, model_MAE, model_MSE, model_RMSE = MLm.test_model(model, test_data)
fit_time = (time.time() - t0) / test_n

# Prepare the summary strings manually without loops
model_MAE_str = (
    f"{model_name}:\n"
    f"MAE on center_set1: {format(model_MAE[0], '.2f')}\n"
    f"MAE on width_set1: {format(model_MAE[1], '.2f')}\n"
    f"MAE on amplitude_set1: {format(model_MAE[2], '.2f')}\n"
    f"MAE on peak_number_set1: {format(model_MAE[3], '.2f')}\n"
    f"MAE on center_set2: {format(model_MAE[4], '.2f')}\n"
    f"MAE on width_set2: {format(model_MAE[5], '.2f')}\n"
    f"MAE on amplitude_set2: {format(model_MAE[6], '.2f')}\n"
    f"MAE on peak_number_set2: {format(model_MAE[7], '.2f')}\n"
    f"MAE on center_set3: {format(model_MAE[8], '.2f')}\n"
    f"MAE on width_set3: {format(model_MAE[9], '.2f')}\n"
    f"MAE on amplitude_set3: {format(model_MAE[10], '.2f')}\n"
    f"MAE on peak_number_set3: {format(model_MAE[11], '.2f')}\n"
    f"MAE on center_set4: {format(model_MAE[12], '.2f')}\n"
    f"MAE on width_set4: {format(model_MAE[13], '.2f')}\n"
    f"MAE on amplitude_set4: {format(model_MAE[14], '.2f')}\n"
    f"MAE on peak_number_set4: {format(model_MAE[15], '.2f')}"
)

model_MSE_str = (
    f"MSE on center_set1: {format(model_MSE[0], '.2f')}\n"
    f"MSE on width_set1: {format(model_MSE[1], '.2f')}\n"
    f"MSE on amplitude_set1: {format(model_MSE[2], '.2f')}\n"
    f"MSE on peak_number_set1: {format(model_MSE[3], '.2f')}\n"
    f"MSE on center_set2: {format(model_MSE[4], '.2f')}\n"
    f"MSE on width_set2: {format(model_MSE[5], '.2f')}\n"
    f"MSE on amplitude_set2: {format(model_MSE[6], '.2f')}\n"
    f"MSE on peak_number_set2: {format(model_MSE[7], '.2f')}\n"
    f"MSE on center_set3: {format(model_MSE[8], '.2f')}\n"
    f"MSE on width_set3: {format(model_MSE[9], '.2f')}\n"
    f"MSE on amplitude_set3: {format(model_MSE[10], '.2f')}\n"
    f"MSE on peak_number_set3: {format(model_MSE[11], '.2f')}\n"
    f"MSE on center_set4: {format(model_MSE[12], '.2f')}\n"
    f"MSE on width_set4: {format(model_MSE[13], '.2f')}\n"
    f"MSE on amplitude_set4: {format(model_MSE[14], '.2f')}\n"
    f"MSE on peak_number_set4: {format(model_MSE[15], '.2f')}"
)

model_RMSE_str = (
    f"RMSE on center_set1: {format(model_RMSE[0], '.2f')}\n"
    f"RMSE on width_set1: {format(model_RMSE[1], '.2f')}\n"
    f"RMSE on amplitude_set1: {format(model_RMSE[2], '.2f')}\n"
    f"RMSE on peak_number_set1: {format(model_RMSE[3], '.2f')}\n"
    f"RMSE on center_set2: {format(model_RMSE[4], '.2f')}\n"
    f"RMSE on width_set2: {format(model_RMSE[5], '.2f')}\n"
    f"RMSE on amplitude_set2: {format(model_RMSE[6], '.2f')}\n"
    f"RMSE on peak_number_set2: {format(model_RMSE[7], '.2f')}\n"
    f"RMSE on center_set3: {format(model_RMSE[8], '.2f')}\n"
    f"RMSE on width_set3: {format(model_RMSE[9], '.2f')}\n"
    f"RMSE on amplitude_set3: {format(model_RMSE[10], '.2f')}\n"
    f"RMSE on peak_number_set3: {format(model_RMSE[11], '.2f')}\n"
    f"RMSE on center_set4: {format(model_RMSE[12], '.2f')}\n"
    f"RMSE on width_set4: {format(model_RMSE[13], '.2f')}\n"
    f"RMSE on amplitude_set4: {format(model_RMSE[14], '.2f')}\n"
    f"RMSE on peak_number_set4: {format(model_RMSE[15], '.2f')}"
)

fit_time_str = "Avg fit time: " + format(fit_time, ".2e") + " s"

# Print the summary
print(model_MAE_str)
print("\n")  # Add a newline for spacing
print(model_MSE_str)
print("\n")  # Add a newline for spacing
print(model_RMSE_str)
print("\n")  # Add a newline for spacing
print(fit_time_str)

# Write the summary to a file
with open(model_folder + "Test_summary.txt", 'w') as f:
    f.writelines(test_database_str)
    f.writelines('\n')
    f.writelines(model_MAE_str)
    f.writelines('\n')
    f.writelines(model_MSE_str)
    f.writelines('\n')
    f.writelines(model_RMSE_str)
    f.writelines('\n')
    f.writelines(fit_time_str)
    f.close()




# model_name = "BayesianCNN"
# model_folder = data_folder + model_name + "/"
# model_file = model_folder + model_name + ".keras"

# custom_objects = {
#     'MyConvLayer': MyConvLayer,
#     'scale_invariant_kl': scale_invariant_kl,
#     'tfp': tfp,
# }


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

#%% Sparse_densenet

# model_name = "Sparse_densenet"

# model_folder = data_folder + model_name + "/"
# model_file = model_folder + model_name + ".keras"

# model = load_model(model_file)
# t0 = time.time()
# Sparse_densenet_test, model_MAE, model_MSE, model_RMSE = MLm.test_model(model, test_data)
# fit_time = (time.time() - t0) / test_n

# # Prepare the summary strings
# model_MAE_str = '\n'.join([
#     model_name + ":",
#     "MAE on center: " + format(model_MAE[0], ".2f"),
#     "MAE on width: " + format(model_MAE[1], ".2f"),
#     "MAE on amplitude: " + format(model_MAE[2], ".2f"),
#     "MAE on peak number: " + format(model_MAE[3], ".2f")
# ])

# model_MSE_str = '\n'.join([
#     "MSE on center: " + format(model_MSE[0], ".2f"),
#     "MSE on width: " + format(model_MSE[1], ".2f"),
#     "MSE on amplitude: " + format(model_MSE[2], ".2f"),
#     "MSE on peak number: " + format(model_MSE[3], ".2f")
# ])

# model_RMSE_str = '\n'.join([
#     "RMSE on center: " + format(model_RMSE[0], ".2f"),
#     "RMSE on width: " + format(model_RMSE[1], ".2f"),
#     "RMSE on amplitude: " + format(model_RMSE[2], ".2f"),
#     "RMSE on peak number: " + format(model_RMSE[3], ".2f")
# ])

# fit_time_str = "Avg fit time: " + format(fit_time, ".2e") + " s"

# # Print the summary
# print(model_MAE_str)
# print("\n")  # Add a newline for spacing
# print(model_MSE_str)
# print("\n")  # Add a newline for spacing
# print(model_RMSE_str)
# print("\n")  # Add a newline for spacing
# print(fit_time_str)

# # Write the summary to a file
# with open(model_folder + "Test_summary.txt", 'w') as f:
#     f.writelines(test_database_str)
#     f.writelines('\n')
#     f.writelines(model_MAE_str)
#     f.writelines('\n')
#     f.writelines(model_MSE_str)
#     f.writelines('\n')
#     f.writelines(model_RMSE_str)
#     f.writelines('\n')
#     f.writelines(fit_time_str)
#     f.close()

# model_name = "Sparse_densenet"

# model_folder = data_folder+model_name+"/"
# model_file = model_folder+model_name+".keras"

# model = load_model(model_file)
# t0 = time.time()
# sparse_densenet_test,model_MAE = MLm.test_model(model,test_data)
# fit_time = (time.time()-t0)/test_n

# model_MAE_str = '\n'.join([model_name+":",
#                             "MAE on center: "+format(model_MAE[0],".2f"),
#                             "MAE on width: "+format(model_MAE[1],".2f"),
#                             "MAE on amplitude: "+format(model_MAE[2],".2f"),
#                             "MAE on peak number: "+format(model_MAE[3],".2f"),
#                             "Avg fit time: "+format(fit_time,".2e")+" s"])
# print(model_MAE_str)

# # Print text summary
# with open(model_folder+"Test_summary.txt", 'w') as f:
#     f.writelines(test_database_str)
#     f.writelines('\n')
#     f.writelines(model_MAE_str)
#     f.close()
    
#%% Resnet


    
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



#%% LeNet
# model_name = "LeNet"
# model_folder = data_folder+model_name+"/"
# model_file = model_folder+model_name+".keras"

# model = load_model(model_file)
# t0 = time.time()
# LeNet_test,model_MAE = MLm.test_model(model,test_data)
# fit_time = (time.time()-t0)/test_n

# model_MAE_str = '\n'.join([model_name+":",
#                             "MAE on center: "+format(model_MAE[0],".2f"),
#                             "MAE on width: "+format(model_MAE[1],".2f"),
#                             "MAE on amplitude: "+format(model_MAE[2],".2f"),
#                             "MAE on peak number: "+format(model_MAE[3],".2f"),
#                             "Avg fit time: "+format(fit_time,".2e")+" s"])
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
# Define pseudo-Voigt function
def pseudo_Voigt(a, b, c, x):
    beta = 5.09791537e-01
    gamma = 4.41140472e-01
    y = c * ((0.7 * np.exp(-np.log(2) * (x - a) ** 2 / (beta * b) ** 2))
        + (0.3 / (1 + (x - a) ** 2 / (gamma * b) ** 2)))
    return y


plt.figure(figsize=(10,5))
plt.plot(energy_range,test_peak[result_i],label="Test spectrum")
plt.plot(energy_range,pseudo_Voigt(test_peak_label[result_i][0],test_peak_label[result_i][1],
                                   test_peak_label[result_i][2],energy_range),label="True major peak")
# plt.plot(energy_range,pseudo_Voigt(*Bayesian_sparse_densenet_test[result_i,0:3],energy_range), label="Bayesian_Sparse densenet")
# plt.plot(energy_range,pseudo_Voigt(*Sparse_densenet_test[result_i,0:3],energy_range), label="Sparse densenet")
#plt.plot(energy_range,pseudo_Voigt(*LeNet_test[result_i,0:3],energy_range), label="LeNet")
# plt.plot(energy_range,pseudo_Voigt(*SEResNet_test[result_i,0:3],energy_range), label="SEResNet")
# plt.plot(energy_range,pseudo_Voigt(*ResNet_test[result_i,0:3],energy_range), label="ResNet")
# plt.plot(energy_range,pseudo_Voigt(*VGGNet_test[result_i,0:3],energy_range), label="VGGNet")
# plt.plot(energy_range,pseudo_Voigt(*Alex_ZFNet_test[result_i,0:3],energy_range), label="Alex ZFNet")
plt.plot(energy_range,pseudo_Voigt(*BayesianCNN_test[result_i,0:3],energy_range), label="BayesianCNN")
plt.grid(True)
plt.ylim(0,4.0)
plt.xlabel("Energy")
plt.ylabel("Intensity")
plt.legend()
plt.tight_layout()
plt.savefig(figures_folder+"Fit_example_"+str(result_i)+".jpg",dpi=300)

