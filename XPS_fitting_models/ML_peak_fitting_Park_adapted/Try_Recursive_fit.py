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
session_name = "Twelth_test_11-07-24"
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
        spectrum = test_spectra[test_i].reshape(-1, 1)
        energy = energy_range.reshape(-1, 1)
        # Number of iterations and sub-peaks is based on first fit
        t0 = time.time()
        fitted_spectra_A[test_i],fitted_params_i = recursive_fit_A(model,test_spectra[test_i],energy_range)
        fitted_params_A.append(fitted_params_i)
        fit_time_A[test_i] = time.time()-t0
        
        # Number of iterations and sub-peaks is based on first fit + curve_fit polish at each intermediate step
        t0 = time.time()
        fitted_spectra_A_ep[test_i], fitted_params_i, do_polish = polish_basinhopping(
            spectrum, energy, fitted_params_A[test_i], fitted_spectra_A[test_i].reshape(-1, 1), R2_min)
        fit_time_A_ip[test_i] = time.time()-t0
        
        # Basinhopping polish after fit A
        t0 = time.time()
        fitted_spectra_A_ep[test_i], fitted_params_i, do_polish = polish_basinhopping(spectrum, energy, fitted_params_A[test_i], fitted_spectra_A[test_i], R2_min)
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
        fitted_spectra_B_ep[test_i], fitted_params_i, do_polish = polish_basinhopping(
            spectrum, energy, fitted_params_B[test_i], fitted_spectra_B[test_i].reshape(-1, 1), R2_min)
        if do_polish: polish_n_B = polish_n_B+1
        fit_time_B_ep[test_i] = time.time()-t0
    
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
# #%% BayesianCNN
# model_name = "BayesianCNN"
# model_folder = data_folder+model_name+"/"
# model_file = model_folder+model_name+".keras"
# Bayesian_cnn_model = load_model(model_file)

# do_recursive_fits(test_peak[0:2000,:,:],energy_range,Bayesian_cnn_model,model_name,model_folder,R2_min)

#%% Bayesian Sparse densenet
model_name = "Bayesian_sparse_densenet"
model_folder = data_folder+model_name+"/"
model_file = model_folder+model_name+".keras"
Bayesian_sparse_densenet_model = load_model(model_file)

do_recursive_fits(test_peak[0:2000,:,:],energy_range,Bayesian_sparse_densenet_model,model_name,model_folder,R2_min)

# #%% Sparse densenet
# model_name = "Sparse_densenet"
# model_folder = data_folder+model_name+"/"
# model_file = model_folder+model_name+".keras"
# sparse_densenet_model = load_model(model_file)

# do_recursive_fits(test_peak[0:2000,:,:],energy_range,sparse_densenet_model,model_name,model_folder,R2_min)


# #%% LeNet
# model_name = "LeNet"
# model_folder = data_folder+model_name+"/"
# model_file = model_folder+model_name+".keras"
# LeNet_model = load_model(model_file)

# #do_recursive_fits(test_peak[0:100,:,:],energy_range,LeNet_model,model_name,model_folder,R2_min)

