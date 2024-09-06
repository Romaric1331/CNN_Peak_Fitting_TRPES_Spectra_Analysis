# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 17:22:59 2024

@author: ajulien
"""

#%% Importation of packages
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tqdm import tqdm
import time
from Fit_functions import recursive_fit_A,recursive_fit_A_internal_polish,polish_basinhopping,recursive_fit_B,recursive_fit_B_internal_polish
import matplotlib.pyplot as plt


#%% Inputs
main_data_folder = "C:/Users/rsallustre/Documents/XPS_fitting/"
session_name = "Fourteenth__MAE_test_18-07-24"
data_folder = main_data_folder+session_name+"/"

R2_min = 0.995

#%% Load test database
database_folder = data_folder+"Database/"
with open(database_folder+"Test_database.pkl", 'rb') as f:
    energy_range, test_peak_label, test_peak, test_peak_param = pickle.load(f)
test_n = len(test_peak_label)
energy_range_n = len(energy_range)

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


#%% Define recursive fit function
def do_recursive_fits(test_spectra,energy_range,model,model_name,model_folder,R2_min):
    test_n = np.size(test_spectra,0)
    energy_range_n = len(energy_range)
    
    fitted_spectra_A = np.zeros([test_n,energy_range_n,1])
    fitted_params_A = []
    fit_time_A = np.zeros(test_n)
    
    fitted_spectra_A_ip = np.zeros([test_n,energy_range_n,1])
    fitted_params_A_ip = []
    fit_time_A_ip = np.zeros(test_n)
    
    fitted_spectra_A_ep = np.zeros([test_n,energy_range_n,1])
    fitted_params_A_ep = []
    fit_time_A_ep = np.zeros(test_n)
    polish_n_A = 0
    
    fitted_spectra_B = np.zeros([test_n,energy_range_n,1])
    fitted_params_B = []
    fit_time_B = np.zeros(test_n)
    
    fitted_spectra_B_ip = np.zeros([test_n,energy_range_n,1])
    fitted_params_B_ip = []
    fit_time_B_ip = np.zeros(test_n)
    
    fitted_spectra_B_ep = np.zeros([test_n,energy_range_n,1])
    fitted_params_B_ep = []
    fit_time_B_ep = np.zeros(test_n)
    polish_n_B = 0
    
    for test_i in tqdm (range (test_n), desc="Iterative fitting", smoothing=0.1):
        # Number of iterations and sub-peaks is based on first fit
        t0 = time.time()
        fitted_spectra_A[test_i], fitted_params_i = recursive_fit_A(model, test_spectra[test_i], energy_range)
        fitted_params_A.append(fitted_params_i)
        fit_time_A[test_i] = time.time() - t0
        
        # Number of iterations and sub-peaks is based on first fit + curve_fit polish at each intermediate step
        t0 = time.time()
        fitted_spectra_A_ip[test_i],fitted_params_i = recursive_fit_A_internal_polish(model,test_spectra[test_i],energy_range)
        fitted_params_A_ip.append(fitted_params_i)
        fit_time_A_ip[test_i] = time.time()-t0
        
        # Basinhopping polish after fit A
        t0 = time.time()
        fitted_spectra_A_ep[test_i],fitted_params_i,do_polish = polish_basinhopping(test_spectra[test_i],energy_range,fitted_params_A[test_i],fitted_spectra_A[test_i],R2_min)
        fitted_params_A_ep.append(fitted_params_i)
        if do_polish: polish_n_A = polish_n_A+1
        fit_time_A_ep[test_i] = time.time()-t0
        
        # Number of iterations and sub-peaks is based on max value of remainig spectrum
        t0 = time.time()
        fitted_spectra_B[test_i],fitted_params_i = recursive_fit_B(model,test_spectra[test_i],energy_range)
        fitted_params_B.append(fitted_params_i)
        fit_time_B[test_i] = time.time()-t0
        
        # Number of iterations and sub-peaks is based on max value of remainig spectrum + curve_fit polish at each intermediate step
        t0 = time.time()
        fitted_spectra_B_ip[test_i],fitted_params_i = recursive_fit_B_internal_polish(model,test_spectra[test_i],energy_range)
        fitted_params_B_ip.append(fitted_params_i)
        fit_time_B_ip[test_i] = time.time()-t0
        
        # Basinhopping polish after fit B
        t0 = time.time()
        fitted_spectra_B_ep[test_i], fitted_params_i, do_polish = polish_basinhopping(test_spectra[test_i], energy_range, fitted_params_B[test_i], fitted_spectra_B[test_i], R2_min)
        fitted_params_B_ep.append(fitted_params_i)
        if do_polish:
            polish_n_B += 1
        fit_time_B_ep[test_i] = time.time() - t0
    
    avg_time_A = np.mean(fit_time_A)
    avg_time_A_ip = np.mean(fit_time_A_ip)
    avg_time_A_ep = np.sum(fit_time_A_ep)/test_n
    avg_time_B = np.mean(fit_time_B)
    avg_time_B_ip = np.mean(fit_time_B_ip)
    avg_time_B_ep = np.sum(fit_time_B_ep)/test_n
    
    with open(model_folder+model_name+"_recursive_fit_test_A.pkl", 'wb') as f:
        pickle.dump([energy_range,fitted_spectra_A,fitted_params_A], f)
    
    with open(model_folder+model_name+"_recursive_fit_test_A_ip.pkl", 'wb') as f:
        pickle.dump([energy_range,fitted_spectra_A_ip,fitted_params_A_ip], f)
    
    with open(model_folder+model_name+"_recursive_fit_test_A_ep.pkl", 'wb') as f:
        pickle.dump([energy_range,fitted_spectra_A_ep,fitted_params_A_ep], f)
    
    with open(model_folder+model_name+"_recursive_fit_test_B.pkl", 'wb') as f:
        pickle.dump([energy_range,fitted_spectra_B,fitted_params_B], f)
    
    with open(model_folder+model_name+"_recursive_fit_test_B_ip.pkl", 'wb') as f:
        pickle.dump([energy_range,fitted_spectra_B_ip,fitted_params_B_ip], f)
    
    with open(model_folder+model_name+"_recursive_fit_test_B_ep.pkl", 'wb') as f:
        pickle.dump([energy_range,fitted_spectra_B_ep,fitted_params_B_ep], f)
    
    model_iterative_fit_str = '\n'.join([model_name+":",
                                         "Size of test database: "+str(test_n),
                                         "Avg fit time (iterative fit A): "+format(avg_time_A,".2f")+" s per spectrum",
                                         "Avg fit time (iterative fit A + internal polish): "+format(avg_time_A_ip,".2f")+" s per spectrum",
                                         "Avg end polish time after iterative fit A: "+format(avg_time_A_ep,".2f")+" s per spectrum, "+str(polish_n_A)+" spectra treated",
                                         "Avg fit time (iterative fit B): "+format(avg_time_B,".2f")+" s per spectrum",
                                         "Avg fit time (iterative fit B + internal polish): "+format(avg_time_B_ip,".2f")+" s per spectrum",
                                         "Avg end polish time after iterative fit B: "+format(avg_time_B_ep,".2f")+" s per spectrum, "+str(polish_n_B)+" spectra treated"])
    print(model_iterative_fit_str)
    
    # Print text summary
    with open(model_folder+"Recursive_fit_summary.txt", 'w') as f:
        f.writelines(model_iterative_fit_str)
        f.close()


# #%% Bayesian CNN

# import tensorflow as tf
# import tensorflow_probability as tfp


# model_name = "BayesianCNN"
# model_folder = data_folder + model_name + "/"
# model_file = model_folder + model_name + ".keras"

# def scale_invariant_kl(q, p, _):
#     return lambda x: tfp.distributions.kl_divergence(q, p) / tf.cast(tf.reduce_prod(q.batch_shape_tensor()), tf.float32)

# class MyConvLayer(tf.keras.layers.Layer):
#     def __init__(self, filters, kernel_size, strides, padding, kernel_divergence_fn, bias_divergence_fn, activation, name=None, trainable=True, **kwargs):
#         super(MyConvLayer, self).__init__(name=name, trainable=trainable, **kwargs)
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.strides = strides
#         self.padding = padding
#         self.kernel_divergence_fn = kernel_divergence_fn
#         self.bias_divergence_fn = bias_divergence_fn
#         self.activation = activation
#         self.conv_layer = None

#     def build(self, input_shape):
#         self.conv_layer = tfp.layers.Convolution1DFlipout(
#             filters=self.filters,
#             kernel_size=self.kernel_size,
#             strides=self.strides,
#             padding=self.padding,
#             kernel_divergence_fn=self.kernel_divergence_fn,
#             bias_divergence_fn=self.bias_divergence_fn,
#             activation=self.activation,
#             trainable=self.trainable
#         )
#         super(MyConvLayer, self).build(input_shape)

#     def call(self, inputs):
#         return self.conv_layer(inputs)

#     def get_config(self):
#         config = super(MyConvLayer, self).get_config()
#         config.update({
#             'filters': self.filters,
#             'kernel_size': self.kernel_size,
#             'strides': self.strides,
#             'padding': self.padding,  
#             'kernel_divergence_fn': 'scale_invariant_kl',
#             'bias_divergence_fn': 'scale_invariant_kl',
#             'activation': self.activation,
#         })
#         return config

#     @classmethod
#     def from_config(cls, config): 
#         config['kernel_divergence_fn'] = scale_invariant_kl
#         config['bias_divergence_fn'] = scale_invariant_kl
#         return cls(**config)

# custom_objects = {
#     'MyConvLayer': MyConvLayer,
#     'scale_invariant_kl': scale_invariant_kl,
#     'tfp': tfp,
#     'Convolution1DFlipout': tfp.layers.Convolution1DFlipout
# }

# try:
#     BayesianCNN_model = load_model(model_file, custom_objects=custom_objects)
#     print("Model loaded successfully.")
# except Exception as e:
#     print(f"Error loading model: {e}")
    # If loading fails, you might need to recreate the model architecture and load weights
    # BayesianCNN_model = create_bayesian_cnn_model()  # You'd need to define this function
    # BayesianCNN_model.load_weights(model_file)
    # If loading fails, you might need to recreate the model architecture and load weights
    # BayesianCNN_model = create_bayesian_cnn_model()  # You'd need to define this function
    # BayesianCNN_model.load_weights(model_file)
#%% Bayesian Sparse densenet
model_name = "Bayesian_sparse_densenet"
model_folder = data_folder+model_name+"/"
model_file = model_folder+model_name+".keras"
Bayesian_sparse_densenet_model = load_model(model_file)

do_recursive_fits(test_peak[0:2000,:,:],energy_range,Bayesian_sparse_densenet_model,model_name,model_folder,R2_min)
# do_recursive_fits(test_peak,energy_range,Bayesian_sparse_densenet_model,model_name,model_folder,R2_min)

#%% Bayesian Sparse densenet
model_name = "BayesianCNN"
model_folder = data_folder+model_name+"/"
model_file = model_folder+model_name+".keras"
BayesianCNN_model = load_model(model_file)

do_recursive_fits(test_peak[0:2000,:,:],energy_range,BayesianCNN_model,model_name,model_folder,R2_min)
# do_recursive_fits(test_peak,energy_range,BayesianCNN_model,model_name,model_folder,R2_min)
#%% Sparse densenet
model_name = "Sparse_densenet"
model_folder = data_folder+model_name+"/"
model_file = model_folder+model_name+".keras"
Sparse_densenet_model = load_model(model_file)

do_recursive_fits(test_peak[0:2000,:,:],energy_range,Sparse_densenet_model,model_name,model_folder,R2_min)
# do_recursive_fits(test_peak,energy_range,Sparse_densenet_model,model_name,model_folder,R2_min)
#%% LeNet
model_name = "LeNet"
model_folder = data_folder+model_name+"/"
model_file = model_folder+model_name+".keras"
LeNet_model = load_model(model_file)

do_recursive_fits(test_peak[0:2000,:,:],energy_range,LeNet_model,model_name,model_folder,R2_min)


#%%  densenet