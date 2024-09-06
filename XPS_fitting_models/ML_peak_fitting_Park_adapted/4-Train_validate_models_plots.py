# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 10:32:27 2024

@author: ajulien & Romaric
"""

# %% Importation of packages
import pickle
import json
import matplotlib.pyplot as plt
import os
import numpy as np


# %% Inputs
main_data_folder = "C:/Users/rsallustre/Documents/XPS_fitting/"
session_name = "Tenth_test_25-06-24"
data_folder = main_data_folder+session_name+"/"


# %% Load models history
model_name = "Bayesian_sparse_densenet"
model_folder = data_folder + model_name + "/"
with open(model_folder + model_name + "_history.json", 'r') as f:
    Bayesian_sparse_densenet_history = json.load(f)
    
    
model_name = "Sparse_densenet"
model_folder = data_folder + model_name + "/"
with open(model_folder + model_name + "_history.json", 'r') as f:
    sparse_densenet_history = json.load(f)
    
    
model_name = "BayesianCNN"
model_folder = data_folder + model_name + "/"
with open(model_folder + model_name + "_history.json", 'r') as f:
    bayesian_cnn_history = json.load(f)
    
print(Bayesian_sparse_densenet_history.keys())
print(Bayesian_sparse_densenet_history["loss"][:10])  # Print first 10 values 
print(sparse_densenet_history.keys())
print(bayesian_cnn_history.keys())
print(sparse_densenet_history["loss"][:10])  # Print first 10 values
print(bayesian_cnn_history["loss"][:10])  # Print first 10 values
# model_name = "SEResNet"
# model_folder = data_folder+model_name+"/"
# with open(model_folder+model_name+"_history.pkl", 'rb') as f:
#     SEResNet_history = pickle.load(f)

# model_name = "ResNet"
# model_folder = data_folder+model_name+"/"
# with open(model_folder+model_name+"_history.pkl", 'rb') as f:
#     ResNet_history = pickle.load(f)

# model_name = "VGGNet"
# model_folder = data_folder+model_name+"/"
# with open(model_folder+model_name+"_history.pkl", 'rb') as f:
#     VGGNet_history = pickle.load(f)

# model_name = "Alex_ZFNet"
# model_folder = data_folder+model_name+"/"
# with open(model_folder+model_name+"_history.pkl", 'rb') as f:
#     Alex_ZFNet_history = pickle.load(f)

# model_name = "LeNet"
# model_folder = data_folder+model_name+"/"
# with open(model_folder+model_name+"_history.pkl", 'rb') as f:
#     LeNet_history = pickle.load(f)

# %% Plot training and validation losses



figures_folder = data_folder+"Figures_models_training_validation/"
os.makedirs(figures_folder, exist_ok=True)
max_loss_value = 100

plt.figure(figsize=(12, 6))

# # Plot training loss
# plt.plot(sparse_densenet_history['loss'], label='SparseDenseNet Training')
# plt.plot(bayesian_cnn_history['loss'], label='BayesianCNN Training')
# plt.plot(Bayesian_sparse_densenet_history['loss'], label='BayesianSparseDenseNet Training')
# # Plot validation loss
# plt.plot(sparse_densenet_history['val_loss'], label='SparseDenseNet Validation')
# plt.plot(bayesian_cnn_history['val_loss'], label='BayesianCNN Validation')
# plt.plot(Bayesian_sparse_densenet_history['val_loss'], label='BayesianSparseDenseNet Validation')
# Bayesian_sparse_densenet_history
# plt.ylim([0, max_loss_value])
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)

# Save the figure
plt.savefig(data_folder + 'Figures_models_training_validation/loss_comparison.png', dpi=300)
plt.show()



# # Training Losses
# fig, (ax1) = plt.subplots(1, 1, figsize=(15, 5))
# plt.yscale('log')
# sparse_densenet_history["loss"] = np.array(sparse_densenet_history["loss"])

# ax1.plot(sparse_densenet_history["loss"], label="Sparse DenseNet")
# ax1.set_title("Sparse DenseNet Loss")
# ax1.plot(bayesian_cnn_history["loss"], label="Bayesian_CNN")
# ax1.set_title("Bayesian_cnn Loss")
# ax1.plot(Bayesian_sparse_densenet_history["loss"], label="Bayesian_sparse_densenet")
# ax1.set_title("Training  Loss")
# ax1.set_xlabel("Epoch")
# ax1.set_ylabel("Loss")
# ax1.legend()
# ax1.grid(True)









# def normalize(data):
#     return (data - np.min(data)) / (np.max(data) - np.min(data))

# plt.figure(figsize=(10, 5))
# plt.plot(normalize(sparse_densenet_history["loss"]), label="Sparse DenseNet")
# plt.plot(normalize(bayesian_cnn_history["loss"]), label="Bayesian_sparse_densenet")
# plt.title("Normalized Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Normalized Loss")
# plt.legend()
# plt.grid(True)
# plt.show()

# Total loss during training and validation

plt.figure(figsize=(10, 5))
plt.subplot(121)

plt.plot(sparse_densenet_history["loss"], '+:', label="Sparse densenet")
plt.plot(bayesian_cnn_history["loss"], '+:', label="BayesianCNN")
plt.plot(Bayesian_sparse_densenet_history["loss"], '+:', label="Bayesian_sparse_densenet_history")
# plt.plot(SEResNet_history["loss"], '+:', label="SEResNet")
# plt.plot(ResNet_history["loss"], '+:', label="ResNet")
# plt.plot(VGGNet_history["loss"], '+:', label="VGGNet")
# plt.plot(Alex_ZFNet_history["loss"], '+:', label="Alex ZFNet")
# plt.plot(LeNet_history["loss"], '+:', label="LeNet")
plt.grid(True)
plt.ylim((0,12))
plt.xlabel("Epoch")
plt.ylabel("Total loss")
plt.legend()
plt.title("Training")

plt.subplot(122)
plt.plot(sparse_densenet_history["val_loss"], '+:', label="Sparse densenet")
plt.plot(bayesian_cnn_history["val_loss"], '+:', label="BayesianCNN")
plt.plot(Bayesian_sparse_densenet_history["val_loss"], '+:', label="Bayesian_sparse_densenet")
# plt.plot(SEResNet_history["val_loss"], '+:', label="SEResNet")
# plt.plot(ResNet_history["val_loss"], '+:', label="ResNet")
# plt.plot(VGGNet_history["val_loss"], '+:', label="VGGNet")
# plt.plot(Alex_ZFNet_history["val_loss"], '+:', label="Alex ZFNet")
# plt.plot(LeNet_history["val_loss"], '+:', label="LeNet")
plt.grid(True)
plt.ylim((0, 15))
plt.xlabel("Epoch")
plt.ylabel("Total loss")
plt.legend()
plt.title("Validation")
plt.tight_layout()
plt.savefig(figures_folder + "Total_loss.jpg", dpi=300)

# Loss on center during training and validation
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(sparse_densenet_history["total_center3_mae"], '+:', label="Sparse densenet")
plt.plot(bayesian_cnn_history["total_center3_mae"], '+:', label="BayesianCNN")
plt.plot(Bayesian_sparse_densenet_history["total_center3_mae"], '+:', label="Bayesian_sparse_densenet")
# plt.plot(SEResNet_history["total_center3_mae"], '+:', label="SEResNet")
# plt.plot(ResNet_history["total_center3_mae"], '+:', label="ResNet")
# plt.plot(VGGNet_history["total_center3_mae"], '+:', label="VGGNet")
# plt.plot(Alex_ZFNet_history["total_center3_mae"], '+:', label="Alex ZFNet")
# plt.plot(LeNet_history["total_center3_mae"], '+:', label="LeNet")
plt.grid(True)
plt.ylim((0, 6.0))
plt.xlabel("Epoch")
plt.ylabel("MAE on center")
plt.legend()
plt.title("Training")

plt.subplot(122)
plt.plot(sparse_densenet_history["val_total_center3_mae"], '+:', label="Sparse densenet")
plt.plot(bayesian_cnn_history["val_total_center3_mae"], '+:', label="BayesianCNN")
plt.plot(Bayesian_sparse_densenet_history["val_total_center3_mae"], '+:', label="Bayesian_sparse_densenet")
# plt.plot(SEResNet_history["val_total_center3_mae"], '+:', label="SEResNet")
# plt.plot(ResNet_history["val_total_center3_mae"], '+:', label="ResNet")
# plt.plot(VGGNet_history["val_total_center3_mae"], '+:', label="VGGNet")
# plt.plot(Alex_ZFNet_history["val_total_center3_mae"], '+:', label="Alex ZFNet")
# plt.plot(LeNet_history["val_total_center3_mae"], '+:', label="LeNet")
plt.grid(True)
plt.ylim((0, 6.0))
plt.xlabel("Epoch")
plt.ylabel("MAE on center")
plt.legend()
plt.title("Validation")
plt.tight_layout()
plt.savefig(figures_folder + "Center_loss.jpg", dpi=300)

# Loss on width during training and validation
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(sparse_densenet_history["total_width3_mae"], '+:', label="Sparse densenet")
plt.plot(bayesian_cnn_history["total_width3_mae"], '+:', label="BayesianCNN")
plt.plot(Bayesian_sparse_densenet_history["total_width3_mae"], '+:', label="Bayesian_sparse_densenet")
# plt.plot(SEResNet_history["total_width3_mae"], '+:', label="SEResNet")
# plt.plot(ResNet_history["total_width3_mae"], '+:', label="ResNet")
# plt.plot(VGGNet_history["total_width3_mae"], '+:', label="VGGNet")
# plt.plot(Alex_ZFNet_history["total_width3_mae"], '+:', label="Alex ZFNet")
# plt.plot(LeNet_history["total_width3_mae"], '+:', label="LeNet")
plt.grid(True)
plt.ylim((0, 6.0))
plt.xlabel("Epoch")
plt.ylabel("MAE on width")
plt.legend()
plt.title("Training")

plt.subplot(122)
plt.plot(sparse_densenet_history["val_total_width3_mae"], '+:', label="Sparse densenet")
plt.plot(bayesian_cnn_history["val_total_width3_mae"], '+:', label="BayesianCNN")
plt.plot(Bayesian_sparse_densenet_history["val_total_width3_mae"], '+:', label="Bayesian_sparse_densenet")
# plt.plot(SEResNet_history["val_total_width3_mae"], '+:', label="SEResNet")
# plt.plot(ResNet_history["val_total_width3_mae"], '+:', label="ResNet")
# plt.plot(VGGNet_history["val_total_width3_mae"], '+:', label="VGGNet")
# plt.plot(Alex_ZFNet_history["val_total_width3_mae"], '+:', label="Alex ZFNet")
# plt.plot(LeNet_history["val_total_width3_mae"], '+:', label="LeNet")
plt.grid(True)
plt.ylim((0, 6.0))
plt.xlabel("Epoch")
plt.ylabel("MAE on width")
plt.legend()
plt.title("Validation")
plt.tight_layout()
plt.savefig(figures_folder + "Width_loss.jpg", dpi=300)

# Loss on amplitude during training and validation
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(sparse_densenet_history["total_amp3_mae"], '+:', label="Sparse densenet")
plt.plot(bayesian_cnn_history["total_amp3_mae"], '+:', label="BayesianCNN")
plt.plot(Bayesian_sparse_densenet_history["total_amp3_mae"], '+:', label="Bayesian_sparse_densenet")
# plt.plot(SEResNet_history["total_amp3_mae"], '+:', label="SEResNet")
# plt.plot(ResNet_history["total_amp3_mae"], '+:', label="ResNet")
# plt.plot(VGGNet_history["total_amp3_mae"], '+:', label="VGGNet")
# plt.plot(Alex_ZFNet_history["total_amp3_mae"], '+:', label="Alex ZFNet")
# plt.plot(LeNet_history["total_amp3_mae"], '+:', label="LeNet")
plt.grid(True)
plt.ylim((0, 6.0))
plt.xlabel("Epoch")
plt.ylabel("MAE on amplitude")
plt.legend()
plt.title("Training")

plt.subplot(122)
plt.plot(sparse_densenet_history["val_total_amp3_mae"], '+:', label="Sparse densenet")
plt.plot(bayesian_cnn_history["val_total_amp3_mae"], '+:', label="BayesianCNN")
plt.plot(Bayesian_sparse_densenet_history["val_total_amp3_mae"], '+:', label="Bayesian_sparse_densenet")
# plt.plot(SEResNet_history["val_total_amp3_mae"], '+:', label="SEResNet")
# plt.plot(ResNet_history["val_total_amp3_mae"], '+:', label="ResNet")
# plt.plot(VGGNet_history["val_total_amp3_mae"], '+:', label="VGGNet")
# plt.plot(Alex_ZFNet_history["val_total_amp3_mae"], '+:', label="Alex ZFNet")
# plt.plot(LeNet_history["val_total_amp3_mae"], '+:', label="LeNet")
plt.grid(True)
plt.ylim((0, 6.0))
plt.xlabel("Epoch")
plt.ylabel("MAE on amplitude")
plt.legend()
plt.title("Validation")
plt.tight_layout()
plt.savefig(figures_folder + "Amplitude_loss.jpg", dpi=300)

# Loss on number of peaks during training and validation
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(sparse_densenet_history["total_peak_number3_mae"], '+:', label="Sparse densenet")
plt.plot(bayesian_cnn_history["total_peak_number3_mae"], '+:', label="BayesianCNN")
plt.plot(Bayesian_sparse_densenet_history["total_peak_number3_mae"], '+:', label="Bayesian_sparse_densenet")
# plt.plot(SEResNet_history["total_peak_number3_mae"], '+:', label="SEResNet")
# plt.plot(ResNet_history["total_peak_number3_mae"], '+:', label="ResNet")
# plt.plot(VGGNet_history["total_peak_number3_mae"], '+:', label="VGGNet")
# plt.plot(Alex_ZFNet_history["total_peak_number3_mae"], '+:', label="Alex ZFNet")
# plt.plot(LeNet_history["total_peak_number3_mae"], '+:', label="LeNet")
plt.grid(True)
plt.ylim((0, 6.0))
plt.xlabel("Epoch")
plt.ylabel("MAE on number of peaks")
plt.legend()
plt.title("Training")

plt.subplot(122)
plt.plot(sparse_densenet_history["val_total_peak_number3_mae"], '+:', label="Sparse densenet")
plt.plot(bayesian_cnn_history["val_total_peak_number3_mae"], '+:', label="BayesianCNN")
plt.plot(Bayesian_sparse_densenet_history["val_total_peak_number3_mae"], '+:', label="Bayesian_sparse_densenet")
# plt.plot(SEResNet_history["val_total_peak_number3_mae"], '+:', label="SEResNet")
# plt.plot(ResNet_history["val_total_peak_number3_mae"], '+:', label="ResNet")
# plt.plot(VGGNet_history["val_total_peak_number3_mae"], '+:', label="VGGNet")
# plt.plot(Alex_ZFNet_history["val_total_peak_number3_mae"], '+:', label="Alex ZFNet")
# plt.plot(LeNet_history["val_total_peak_number3_mae"], '+:', label="LeNet")
plt.grid(True)
plt.ylim((0, 6.0))
plt.xlabel("Epoch")
plt.ylabel("MAE on number of peaks")
plt.legend()
plt.title("Validation")
plt.tight_layout()
plt.savefig(figures_folder + "Peak_number_loss.jpg", dpi=300)

