# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 13:39:26 2024

@author: ajulien & Romaric
"""

#%% Importation of packages
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from Fit_functions import pseudo_Voigt,pseudo_Voigt_3,pseudo_Voigt_4
import matplotlib.colors as mcolors
colors = list(mcolors.TABLEAU_COLORS)
import matplotlib as mpl


#%% Inputs

main_data_folder = "C:/Users/rsallustre/Documents/XPS_fitting/"
session_name = "Fourteenth_test_18-07-24"
data_folder = main_data_folder+session_name+"/"

# model_name = "BayesianCNN"
# model_folder = data_folder+model_name+"/"
model_name = "Bayesian_sparse_densenet"
model_folder = data_folder+model_name+"/"
# model_name = "Sparse_densenet"
# model_folder = data_folder+model_name+"/"
figures_folder = data_folder+"Figures_recursive_fits_test/"+model_name+"/"
os.makedirs(figures_folder,exist_ok=True)

mpl.rcParams['lines.linewidth'] = 2.5
mpl.rcParams['lines.markersize'] = 11
mpl.rcParams['font.size'] = 20
mpl.rcParams['legend.fontsize'] = 20


#%% Sort peaks per underlying area

def sort_sub_peaks(peak_params, energy_range):
    peak_params_sorted = []
    for i in range(len(peak_params)):
        sub_peaks_area = []
        for j in range(len(peak_params[i])):
            try:
                if isinstance(peak_params[i][j], (list, np.ndarray)):
                    x0 = float(peak_params[i][j][0])
                    gamma = float(peak_params[i][j][1])
                    sigma = float(peak_params[i][j][2])
                elif isinstance(peak_params[i][j], (int, float)):
                    x0 = float(peak_params[i][j])
                    gamma = sigma = 0.0  # Default values
                else:
                    raise ValueError(f"Unexpected type for peak_params[{i}][{j}]: {type(peak_params[i][j])}")
                
                area = sum(pseudo_Voigt(energy_range, x0, gamma, sigma))
                sub_peaks_area.append(area)
            except (TypeError, ValueError, IndexError) as e:
                print(f"Error: {e} for peak_params[{i}][{j}]")
                sub_peaks_area.append(0)  # Default value in case of error
        
        sub_peaks_area = np.array(sub_peaks_area)
        sub_peaks_order = np.argsort(sub_peaks_area)[::-1]
        sorted_peaks = []
        for k in sub_peaks_order:
            if k < len(peak_params[i]):
                if isinstance(peak_params[i][k], (list, np.ndarray)):
                    sorted_peak = peak_params[i][k][:3]
                else:
                    sorted_peak = [peak_params[i][k], 0, 0]  # Default values for gamma and sigma
                sorted_peaks.append(np.reshape(np.array(sorted_peak), (3, 1)))
        peak_params_sorted.append(sorted_peaks)
    return peak_params_sorted
#%% Function computing MAE on sub-peaks
def errors_5_sub_peaks(pn_test, pn_fit, peak_params_test, peak_params_fit):
    indices = np.where(pn_test == 5)[0]
    
    if len(indices) == 0:
        return np.zeros((3, 5))  # Return a 3x5 array of zeros if there are no 5-peak spectra
    
    def safe_concatenate(params, index):
        valid_params = []
        for i in indices:
            if i < len(params) and index < len(params[i]):
                if isinstance(params[i][index], (list, np.ndarray)):
                    valid_params.append(np.array(params[i][index][:3]).flatten())
                elif isinstance(params[i][index], (int, float)):
                    valid_params.append(np.array([params[i][index], 0, 0]))
                else:
                    valid_params.append(np.array([0, 0, 0]))
            else:
                valid_params.append(np.array([0, 0, 0]))
        
        if not valid_params:
            return np.zeros((3, 1))  # Return a 3x1 array of zeros if there are no valid parameters
        
        # Ensure all elements have the same shape
        max_length = max(len(param) for param in valid_params)
        valid_params = [np.pad(param, (0, max_length - len(param)), 'constant') for param in valid_params]
        
        return np.array(valid_params).T  # Transpose to get a 3xN array

    test_params = [safe_concatenate(peak_params_test, i) for i in range(5)]
    fit_params = [safe_concatenate(peak_params_fit, i) for i in range(5)]
    
    MAEs = []
    for test, fit in zip(test_params, fit_params):
        if test.size == 0 or fit.size == 0:
            MAEs.append(np.zeros(3))
        else:
            # Ensure test and fit have the same shape
            min_shape = min(test.shape[1], fit.shape[1])
            test = test[:, :min_shape]
            fit = fit[:, :min_shape]
            MAEs.append(np.nanmean(abs(test - fit), axis=1))
    
    return np.array(MAEs).T
def errors_4_sub_peaks(pn_test, pn_fit, peak_params_test, peak_params_fit):
    indices = np.where(pn_test == 4)[0]
    
    def safe_concatenate(params, index):
        return np.concatenate([params[i][index][:3] if i < len(params) and index < len(params[i]) else np.zeros((3,1)) for i in indices], axis=1)

    test_params_0 = safe_concatenate(peak_params_test, 0)
    test_params_1 = safe_concatenate(peak_params_test, 1)
    test_params_2 = safe_concatenate(peak_params_test, 2)
    test_params_3 = safe_concatenate(peak_params_test, 3)
    
    def safe_append(params, i, index):
        if i < len(params) and index < len(params[i]):
            param = params[i][index]
            if isinstance(param, (list, np.ndarray)) and len(param) >= 3:
                return param[:3]
            elif isinstance(param, (int, float)):
                return np.array([param, 0, 0])  # Assume the float is the first parameter
        return np.zeros((3,1)) * np.nan

    fit_params = [[], [], [], []]
    for i in indices:
        for j in range(4):
            fit_params[j].append(safe_append(peak_params_fit, i, j) if pn_fit[i] > j else np.zeros((3,1)) * np.nan)
    
    fit_params = [np.concatenate(params, axis=1) for params in fit_params]
    
    MAEs = [np.nanmean(abs(test - fit), axis=1) for test, fit in zip([test_params_0, test_params_1, test_params_2, test_params_3], fit_params)]
    
    return np.stack(MAEs, axis=1)

def errors_3_sub_peaks(pn_test, pn_fit, peak_params_test, peak_params_fit):
    indices = np.where(pn_test == 3)[0]
    
    def safe_concatenate(params, index):
        return np.concatenate([params[i][index][:3] if i < len(params) and index < len(params[i]) else np.zeros((3,1)) for i in indices], axis=1)

    test_params_0 = safe_concatenate(peak_params_test, 0)
    test_params_1 = safe_concatenate(peak_params_test, 1)
    test_params_2 = safe_concatenate(peak_params_test, 2)
    
    def safe_append(params, i, index):
        if i < len(params) and index < len(params[i]):
            param = params[i][index]
            if isinstance(param, (list, np.ndarray)) and len(param) >= 3:
                return param[:3]
            elif isinstance(param, (int, float)):
                return np.array([param, 0, 0])  # Assume the float is the first parameter
        return np.zeros((3,1)) * np.nan

    fit_params = [[], [], []]
    for i in indices:
        for j in range(3):
            fit_params[j].append(safe_append(peak_params_fit, i, j) if pn_fit[i] > j else np.zeros((3,1)) * np.nan)
    
    fit_params = [np.concatenate(params, axis=1) for params in fit_params]
    
    MAEs = [np.nanmean(abs(test - fit), axis=1) for test, fit in zip([test_params_0, test_params_1, test_params_2], fit_params)]
    
    return np.stack(MAEs, axis=1)
def errors_2_sub_peaks(pn_test, pn_fit, peak_params_test, peak_params_fit):
    indices = np.where(pn_test == 2)[0]
    
    def safe_concatenate(params, index):
        return np.concatenate([params[i][index][:3] if i < len(params) and index < len(params[i]) else np.zeros((3,1)) for i in indices], axis=1)

    test_params_0 = safe_concatenate(peak_params_test, 0)
    test_params_1 = safe_concatenate(peak_params_test, 1)
    
    def safe_append(params, i, index):
        if i < len(params) and index < len(params[i]):
            param = params[i][index]
            if isinstance(param, (list, np.ndarray)) and len(param) >= 3:
                return param[:3]
            elif isinstance(param, (int, float)):
                return np.array([param, 0, 0])  # Assume the float is the first parameter
        return np.zeros((3,1)) * np.nan

    fit_params = [[], []]
    for i in indices:
        for j in range(2):
            fit_params[j].append(safe_append(peak_params_fit, i, j) if pn_fit[i] > j else np.zeros((3,1)) * np.nan)
    
    fit_params = [np.concatenate(params, axis=1) for params in fit_params]
    
    MAEs = [np.nanmean(abs(test - fit), axis=1) for test, fit in zip([test_params_0, test_params_1], fit_params)]
    
    return np.stack(MAEs, axis=1)
def errors_1_sub_peaks(pn_test, pn_fit, peak_params_test, peak_params_fit):
    indices = np.where(pn_test == 1)[0]
    
    def safe_get_params(params, i):
        if i < len(params):
            if isinstance(params[i], (list, np.ndarray)) and len(params[i]) > 0:
                if isinstance(params[i][0], (list, np.ndarray)):
                    return np.array(params[i][0][:3]).flatten()
                else:
                    return np.array(params[i][:3]).flatten()
            elif isinstance(params[i], (int, float)):
                return np.array([params[i], 0, 0])
        return np.array([0, 0, 0])

    test_params_0 = np.array([safe_get_params(peak_params_test, i) for i in indices])
    fit_params_0 = np.array([safe_get_params(peak_params_fit, i) for i in indices])
    
    print(f"test_params_0 shape: {test_params_0.shape}")
    print(f"fit_params_0 shape: {fit_params_0.shape}")
    
    if test_params_0.shape != fit_params_0.shape:
        print(f"Shape mismatch: test_params_0 shape: {test_params_0.shape}, fit_params_0 shape: {fit_params_0.shape}")
        # Ensure both arrays have the same shape
        min_shape = min(test_params_0.shape[0], fit_params_0.shape[0])
        test_params_0 = test_params_0[:min_shape]
        fit_params_0 = fit_params_0[:min_shape]
    
    MAE_0 = np.nanmean(abs(test_params_0 - fit_params_0), axis=1)
    return MAE_0[:, np.newaxis]

def compute_MAE(peak_number_test, peak_number_fit, peak_params_test_sorted, fitted_params):
    if len(peak_number_test) == 0 or len(peak_number_fit) == 0:
        return np.zeros(6)  # Return zeros if there's no data
    
    MAE_peak_number = np.nanmean(abs(peak_number_test - peak_number_fit), axis=0)
    
    MAE_5 = errors_5_sub_peaks(peak_number_test, peak_number_fit, peak_params_test_sorted, fitted_params)
    MAE_4 = errors_4_sub_peaks(peak_number_test, peak_number_fit, peak_params_test_sorted, fitted_params)
    MAE_3 = errors_3_sub_peaks(peak_number_test, peak_number_fit, peak_params_test_sorted, fitted_params)
    MAE_2 = errors_2_sub_peaks(peak_number_test, peak_number_fit, peak_params_test_sorted, fitted_params)
    MAE_1 = errors_1_sub_peaks(peak_number_test, peak_number_fit, peak_params_test_sorted, fitted_params)
    
    return MAE_peak_number, MAE_5, MAE_4, MAE_3, MAE_2, MAE_1


#%% Load test database
database_folder = data_folder+"Database/"
with open(database_folder+"Test_database.pkl", 'rb') as f:
    energy_range, peak_label_test, spectra_test, peak_params_test = pickle.load(f)
test_n = len(peak_label_test)
energy_range_n = len(energy_range)

# Store number of sub-peaks in test database
peak_number_test = np.array([len(peak_params_test[i]) for i in range(test_n)])

# Sort peaks per underlying area
peak_params_test_sorted = sort_sub_peaks(peak_params_test,energy_range)


#%% Load recursive fit results on test database
with open(model_folder+model_name+"_recursive_fit_test_A.pkl", 'rb') as f:
    energy_range,fitted_spectra_A,fitted_params_A = pickle.load(f)
# Sort peaks per underlying area
fitted_params_A_sorted = sort_sub_peaks(fitted_params_A,energy_range)
recursive_fit_n = np.size(fitted_spectra_A,0)

with open(model_folder+model_name+"_recursive_fit_test_A_ip.pkl", 'rb') as f:
    energy_range,fitted_spectra_A_ip,fitted_params_A_ip = pickle.load(f)
# Sort peaks per underlying area
fitted_params_A_ip_sorted = sort_sub_peaks(fitted_params_A_ip,energy_range)

with open(model_folder+model_name+"_recursive_fit_test_A_ep.pkl", 'rb') as f:
    energy_range,fitted_spectra_A_ep,fitted_params_A_ep = pickle.load(f)
# Sort peaks per underlying area
fitted_params_A_ep_sorted = sort_sub_peaks(fitted_params_A_ep,energy_range)

with open(model_folder+model_name+"_recursive_fit_test_B.pkl", 'rb') as f:
    energy_range,fitted_spectra_B,fitted_params_B = pickle.load(f)
# Sort peaks per underlying area
fitted_params_B_sorted = sort_sub_peaks(fitted_params_B,energy_range)

with open(model_folder+model_name+"_recursive_fit_test_B_ip.pkl", 'rb') as f:
    energy_range,fitted_spectra_B_ip,fitted_params_B_ip = pickle.load(f)
# Sort peaks per underlying area
fitted_params_B_ip_sorted = sort_sub_peaks(fitted_params_B_ip,energy_range)

with open(model_folder+model_name+"_recursive_fit_test_B_ep.pkl", 'rb') as f:
    energy_range,fitted_spectra_B_ep,fitted_params_B_ep = pickle.load(f)
# Sort peaks per underlying area
fitted_params_B_ep_sorted = sort_sub_peaks(fitted_params_B_ep,energy_range)


#%% Compute average error indicators
# Store number of sub-peaks according to fit
peak_number_fit_A = np.array([len(fitted_params_A_sorted[i]) for i in range(recursive_fit_n)])

# Compute MAE on parameters estimation
MAE_peak_number_A,MAE_A_5,MAE_A_4,MAE_A_3,MAE_A_2,MAE_A_1 = compute_MAE(peak_number_test[0:recursive_fit_n],
                                                                        peak_number_fit_A,
                                                                        peak_params_test_sorted[0:recursive_fit_n],
                                                                        fitted_params_A_sorted)

# Compute R2 error
SST_A = np.squeeze(np.sum((spectra_test[0:recursive_fit_n]-np.mean(spectra_test[0:recursive_fit_n],1,keepdims=True))**2,1))
SSE_A = np.squeeze(np.sum((spectra_test[0:recursive_fit_n]-fitted_spectra_A)**2,1))
R2_A = 1-SSE_A/SST_A

peak_number_fit_A_ip = []
for i in range(recursive_fit_n):
    try:
        if len(fitted_params_A_ip_sorted) > 0:
            peak_number_fit_A_ip = np.array([len(fitted_params_A_ip_sorted[i]) for i in range(recursive_fit_n)])
        else:
                peak_number_fit_A_ip = np.zeros(recursive_fit_n)  # or handle it as per your application logic

    except IndexError:
        print(f"IndexError occurred at index {i}, length of fitted_params_A_ip_sorted is {len(fitted_params_A_ip_sorted)}")
        raise
peak_number_fit_A_ip = np.array(peak_number_fit_A_ip)

# Compute MAE on parameters estimation
MAE_peak_number_A_ip,MAE_A_ip_5,MAE_A_ip_4,MAE_A_ip_3,MAE_A_ip_2,MAE_A_ip_1 = compute_MAE(peak_number_test[0:recursive_fit_n],
                                                                        peak_number_fit_A_ip,
                                                                        peak_params_test_sorted[0:recursive_fit_n],
                                                                        fitted_params_A_ip_sorted)

# Compute R2 error
SST_A_ip = np.squeeze(np.sum((spectra_test[0:recursive_fit_n]-np.mean(spectra_test[0:recursive_fit_n],1,keepdims=True))**2,1))
SSE_A_ip = np.squeeze(np.sum((spectra_test[0:recursive_fit_n]-fitted_spectra_A_ip)**2,1))
R2_A_ip = 1-SSE_A_ip/SST_A_ip

# Store number of sub-peaks according to fit
peak_number_fit_A_ep = np.array([len(fitted_params_A_ep_sorted[i]) for i in range(recursive_fit_n)])

# Compute MAE on parameters estimation
MAE_peak_number_A_ep,MAE_A_ep_5,MAE_A_ep_4,MAE_A_ep_3,MAE_A_ep_2,MAE_A_ep_1 = compute_MAE(peak_number_test[0:recursive_fit_n],
                                                                        peak_number_fit_A_ep,
                                                                        peak_params_test_sorted[0:recursive_fit_n],
                                                                        fitted_params_A_ep_sorted)

# Compute R2 error
SST_A_ep = np.squeeze(np.sum((spectra_test[0:recursive_fit_n]-np.mean(spectra_test[0:recursive_fit_n],1,keepdims=True))**2,1))
SSE_A_ep = np.squeeze(np.sum((spectra_test[0:recursive_fit_n]-fitted_spectra_A_ep)**2,1))
R2_A_ep = 1-SSE_A_ep/SST_A_ep


# Store number of sub-peaks according to fit
peak_number_fit_B = np.array([len(fitted_params_B_sorted[i]) for i in range(recursive_fit_n)])

# Compute MAE on parameters estimation
MAE_peak_number_B,MAE_B_5,MAE_B_4,MAE_B_3,MAE_B_2,MAE_B_1 = compute_MAE(peak_number_test[0:recursive_fit_n],
                                                                        peak_number_fit_B,
                                                                        peak_params_test_sorted[0:recursive_fit_n],
                                                                        fitted_params_B_sorted)

# Compute R2 error
SST_B = np.squeeze(np.sum((spectra_test[0:recursive_fit_n]-np.mean(spectra_test[0:recursive_fit_n],1,keepdims=True))**2,1))
SSE_B = np.squeeze(np.sum((spectra_test[0:recursive_fit_n]-fitted_spectra_B)**2,1))
R2_B = 1-SSE_B/SST_B

# Store number of sub-peaks according to fit
peak_number_fit_B_ip = np.array([len(fitted_params_B_ip_sorted[i]) for i in range(recursive_fit_n)])

# Compute MAE on parameters estimation
MAE_peak_number_B_ip,MAE_B_ip_5,MAE_B_ip_4,MAE_B_ip_3,MAE_B_ip_2,MAE_B_ip_1 = compute_MAE(peak_number_test[0:recursive_fit_n],
                                                                        peak_number_fit_B_ip,
                                                                        peak_params_test_sorted[0:recursive_fit_n],
                                                                        fitted_params_B_ip_sorted)

# Compute R2 error
SST_B_ip = np.squeeze(np.sum((spectra_test[0:recursive_fit_n]-np.mean(spectra_test[0:recursive_fit_n],1,keepdims=True))**2,1))
SSE_B_ip = np.squeeze(np.sum((spectra_test[0:recursive_fit_n]-fitted_spectra_B_ip)**2,1))
R2_B_ip = 1-SSE_B_ip/SST_B_ip

# Store number of sub-peaks according to fit
peak_number_fit_B_ep = np.array([len(params) for params in fitted_params_B_ep_sorted])
recursive_fit_n = len(peak_number_fit_B_ep)
print("recursive_fit_n:", recursive_fit_n)
print("fitted_spectra_B_ep shape:", fitted_spectra_B_ep.shape)
recursive_fit_n = min(len(spectra_test), fitted_spectra_B_ep.shape[0])
print("Adjusted recursive_fit_n:", recursive_fit_n)
# Compute MAE on parameters estimation
MAE_peak_number_B_ep,MAE_B_ep_5,MAE_B_ep_4,MAE_B_ep_3,MAE_B_ep_2,MAE_B_ep_1 = compute_MAE(peak_number_test[0:recursive_fit_n],
                                                                        peak_number_fit_B_ep,
                                                                        peak_params_test_sorted[0:recursive_fit_n],
                                                                        fitted_params_B_ep_sorted)

# Compute R2 error
SST_B_ep = np.squeeze(np.sum((spectra_test[:recursive_fit_n] - np.mean(spectra_test[:recursive_fit_n], axis=1, keepdims=True))**2, axis=1))
SSE_B_ep = np.squeeze(np.sum((spectra_test[:recursive_fit_n] - fitted_spectra_B_ep[:recursive_fit_n])**2, axis=1))
R2_B_ep = 1 - SSE_B_ep / SST_B_ep


#%% Plot distribution of SSE
bins = np.linspace(0.98,1,5)
weights = np.ones((recursive_fit_n,3))/recursive_fit_n

plt.figure(figsize=(10,5))
plt.hist(np.stack([R2_A,R2_A_ip,R2_A_ep],1),bins=bins,rwidth=0.9,weights=weights)
plt.grid()
plt.xlim((bins[0],bins[-1]))
plt.ylim((0,1))
plt.xlabel("R2")
plt.ylabel("Fraction of test database")
plt.legend(("A1","A2","A3"))
plt.title(model_name+" - A recursive fit")
plt.xticks(rotation = 25)
plt.tight_layout()
plt.savefig(figures_folder+"R2_hist_A.jpg",dpi=300)
plt.show()

plt.figure(figsize=(10,5))
plt.hist(np.stack([R2_B,R2_B_ip,R2_B_ep],1),bins=bins,rwidth=0.9,weights=weights)
plt.grid()
plt.xlim((bins[0],bins[-1]))
plt.ylim((0,1))
plt.xlabel("R2")
plt.ylabel("Fraction of test database")
plt.legend(("B1","B2","B3"))
plt.title(model_name+" - B recursive fit")
plt.xticks(rotation = 25)
plt.tight_layout()
plt.savefig(figures_folder+"R2_hist_B.jpg",dpi=300)
plt.show()


#%% Plot example with given value of R2 and number of peaks
peaks_number_value = 5
indices = np.where(peak_number_test[0:recursive_fit_n]==peaks_number_value)[0]
plot_i = indices[random.randint(0,len(indices))-1]
gs_kw = dict(height_ratios=[2,1])

#plot_i = 233

# Fit A
fig, axd = plt.subplot_mosaic([['upper'],['lower']],gridspec_kw=gs_kw, figsize=(10,7.5))
axd['upper'].fill_between(energy_range,np.squeeze(spectra_test[plot_i]),color='k',edgecolor=None,alpha=0.2,label="Test spectrum")
axd['upper'].plot(energy_range,fitted_spectra_A[plot_i],'k',label="Recursive fit")
for i in range(len(peak_params_test_sorted[plot_i])):
    axd['upper'].fill_between(energy_range,
                     pseudo_Voigt(energy_range,
                                  peak_params_test_sorted[plot_i][i][0],peak_params_test_sorted[plot_i][i][1],peak_params_test_sorted[plot_i][i][2]),
                     alpha=0.5,color=colors[i],edgecolor=None)
for i in range(len(fitted_params_A_sorted[plot_i])):
    axd['upper'].plot(energy_range,
             pseudo_Voigt(energy_range,
                          fitted_params_A_sorted[plot_i][i][0],fitted_params_A_sorted[plot_i][i][1],fitted_params_A_sorted[plot_i][i][2]),
             color=colors[i])
axd['upper'].grid()
axd['upper'].set_xlabel("Energy")
axd['upper'].set_ylabel("Intensity")
axd['upper'].legend()
axd['upper'].set_title(model_name+" - A recursive fit")

axd['lower'].plot(energy_range,np.squeeze(spectra_test[plot_i]-fitted_spectra_A[plot_i]))
axd['lower'].grid()
axd['lower'].set_ylim((-0.1,0.1))
axd['lower'].set_xlabel("Energy")
axd['lower'].set_ylabel("Error")

plt.tight_layout()
plt.savefig(figures_folder+"Fit_example_"+str(plot_i)+"_A1.jpg",dpi=300)
plt.show()

# Fit A + internal polish
fig, axd = plt.subplot_mosaic([['upper'],['lower']],gridspec_kw=gs_kw, figsize=(10,7.5))
axd['upper'].fill_between(energy_range,np.squeeze(spectra_test[plot_i]),color='k',edgecolor=None,alpha=0.2,label="Test spectrum")
axd['upper'].plot(energy_range,fitted_spectra_A_ip[plot_i],'k',label="Recursive fit + internal polish")
for i in range(len(peak_params_test_sorted[plot_i])):
    axd['upper'].fill_between(energy_range,
                     pseudo_Voigt(energy_range,
                                  peak_params_test_sorted[plot_i][i][0],peak_params_test_sorted[plot_i][i][1],peak_params_test_sorted[plot_i][i][2]),
                     alpha=0.5,color=colors[i],edgecolor=None)
# Upper plot
for i in range(len(fitted_params_A_ip_sorted[plot_i])):
    params = fitted_params_A_ip_sorted[plot_i][i]
    if len(params) < 3:
        print(f"Warning: Insufficient parameters for peak {i}")
        continue
    axd['upper'].plot(energy_range,
             pseudo_Voigt(energy_range, params[0], params[1], params[2]),
             color=colors[i % len(colors)])  # Use modulo to avoid index out of range for colors

axd['upper'].grid()
axd['upper'].set_xlabel("Energy")
axd['upper'].set_ylabel("Intensity")
axd['upper'].legend()
axd['upper'].set_title(f"{model_name} - A recursive fit + internal polish (plot {plot_i})")

# Lower plot
if plot_i < len(spectra_test) and plot_i < len(fitted_spectra_A_ip):
    axd['lower'].plot(energy_range, np.squeeze(spectra_test[plot_i] - fitted_spectra_A_ip[plot_i]))
    axd['lower'].grid()
    axd['lower'].set_ylim((-0.1, 0.1))
    axd['lower'].set_xlabel("Energy")
    axd['lower'].set_ylabel("Error")
else:
    print(f"Error: plot_i ({plot_i}) is out of range for spectra_test or fitted_spectra_A_ip")

plt.tight_layout()
plt.savefig(figures_folder + f"Fit_example_{plot_i}_A2.jpg", dpi=300)
plt.show()
# Fit A + external polish
fig, axd = plt.subplot_mosaic([['upper'],['lower']],gridspec_kw=gs_kw, figsize=(10,7.5))
axd['upper'].fill_between(energy_range,np.squeeze(spectra_test[plot_i]),color='k',edgecolor=None,alpha=0.2,label="Test spectrum")
axd['upper'].plot(energy_range,fitted_spectra_A_ep[plot_i],'k',label="Recursive fit + external polish")
for i in range(len(peak_params_test_sorted[plot_i])):
    axd['upper'].fill_between(energy_range,
                     pseudo_Voigt(energy_range,
                                  peak_params_test_sorted[plot_i][i][0],peak_params_test_sorted[plot_i][i][1],peak_params_test_sorted[plot_i][i][2]),
                     alpha=0.5,color=colors[i],edgecolor=None)
for i in range(len(fitted_params_A_ep_sorted[plot_i])):
    axd['upper'].plot(energy_range,
             pseudo_Voigt(energy_range,
                          fitted_params_A_ep_sorted[plot_i][i][0],fitted_params_A_ep_sorted[plot_i][i][1],fitted_params_A_ep_sorted[plot_i][i][2]),
             color=colors[i])
axd['upper'].grid()
axd['upper'].set_xlabel("Energy")
axd['upper'].set_ylabel("Intensity")
axd['upper'].legend()
axd['upper'].set_title(model_name+" - A recursive fit + external polish")

axd['lower'].plot(energy_range,np.squeeze(spectra_test[plot_i]-fitted_spectra_A_ep[plot_i]))
axd['lower'].grid()
axd['lower'].set_ylim((-0.1,0.1))
axd['lower'].set_xlabel("Energy")
axd['lower'].set_ylabel("Error")

plt.tight_layout()
plt.savefig(figures_folder+"Fit_example_"+str(plot_i)+"_A3.jpg",dpi=300)
plt.show()


peak_params_test_sorted_i = np.concatenate(peak_params_test_sorted[plot_i],1)
fitted_params_A_sorted_i = np.concatenate(fitted_params_A_sorted[plot_i],1)
fitted_params_A_ip_sorted_i = np.concatenate(fitted_params_A_ip_sorted[plot_i],1)
fitted_params_A_ep_sorted_i = np.concatenate(fitted_params_A_ep_sorted[plot_i],1)

# plt.figure(figsize=(10,5))
# plt.fill_between(energy_range,np.squeeze(spectra_test[plot_i]),color='k',edgecolor=None,alpha=0.2,label="Test spectrum")
# plt.plot(energy_range,fitted_spectra_B[plot_i],'k',label="Recursive fit")
# plt.plot(energy_range,fitted_spectra_B_ip[plot_i],'k--',label="Recursive fit + internal polish")
# plt.plot(energy_range,fitted_spectra_B_ep[plot_i],'k:',label="Recursive fit + end polish")
# for i in range(len(peak_params_test_sorted[plot_i])):
#     plt.fill_between(energy_range,
#                      pseudo_Voigt(energy_range,
#                                   peak_params_test_sorted[plot_i][i][0],peak_params_test_sorted[plot_i][i][1],peak_params_test_sorted[plot_i][i][2]),
#                      alpha=0.5,color=colors[i])
# for i in range(len(fitted_params_B_sorted[plot_i])):
#     plt.plot(energy_range,
#              pseudo_Voigt(energy_range,
#                           fitted_params_B_sorted[plot_i][i][0],fitted_params_B_sorted[plot_i][i][1],fitted_params_B_sorted[plot_i][i][2]),
#              color=colors[i])
# for i in range(len(fitted_params_B_ip_sorted[plot_i])):
#     plt.plot(energy_range,
#              pseudo_Voigt(energy_range,
#                           fitted_params_B_ip_sorted[plot_i][i][0],fitted_params_B_ip_sorted[plot_i][i][1],fitted_params_B_ip_sorted[plot_i][i][2]),
#              '--',color=colors[i])
# for i in range(len(fitted_params_B_ep_sorted[plot_i])):
#     plt.plot(energy_range,
#              pseudo_Voigt(energy_range,
#                           fitted_params_B_ep_sorted[plot_i][i][0],fitted_params_B_ep_sorted[plot_i][i][1],fitted_params_B_ep_sorted[plot_i][i][2]),
#              ':',color=colors[i])
# plt.grid()
# plt.xlabel("Energy")
# plt.ylabel("Intensity")
# plt.legend()
# plt.title(model_name+" - B recursive fit")
# plt.tight_layout()
# plt.savefig(figures_folder+"Fit_example_"+str(plot_i)+"_B.jpg",dpi=300)
# plt.show()



# #%% Plot MAE versus minimum R2
# R2_range_n = 10
# R2_range = np.linspace(0.97,0.99,R2_range_n)

# MAE_5_R2_sparse_densenet = np.zeros((3,5,R2_range_n))*np.nan
# MAE_4_R2_sparse_densenet = np.zeros((3,4,R2_range_n))*np.nan
# MAE_3_R2_sparse_densenet = np.zeros((3,3,R2_range_n))*np.nan
# MAE_2_R2_sparse_densenet = np.zeros((3,2,R2_range_n))*np.nan
# fits_n_5 = np.zeros(R2_range_n)*np.nan
# fits_n_4 = np.zeros(R2_range_n)*np.nan
# fits_n_3 = np.zeros(R2_range_n)*np.nan
# fits_n_2 = np.zeros(R2_range_n)*np.nan
# for R2_i in range(R2_range_n):
#     indices = np.where(np.array(R2_sparse_densenet>R2_range[R2_i]))[0]
#     peak_params_test_i = [peak_params_test_sorted[i] for i in indices]
#     peak_params_sparse_densenet_i = [peak_params_sparse_densenet[i] for i in indices]
    
#     MAE_5_R2_sparse_densenet[:,:,R2_i] = errors_5_sub_peaks(peak_number_test[indices],
#                                                peak_number_sparse_densenet[indices],
#                                                peak_params_test_i,peak_params_sparse_densenet_i)
#     fits_n_5[R2_i] = len(np.where(peak_number_test[indices] == 5)[0])/len(np.where(peak_number_test == 5)[0])
    
    
#     MAE_4_R2_sparse_densenet[:,:,R2_i] = errors_4_sub_peaks(peak_number_test[indices],
#                                                peak_number_sparse_densenet[indices],
#                                                peak_params_test_i,peak_params_sparse_densenet_i)
#     fits_n_4[R2_i] = len(np.where(peak_number_test[indices] == 4)[0])/len(np.where(peak_number_test == 4)[0])
    
#     MAE_3_R2_sparse_densenet[:,:,R2_i] = errors_3_sub_peaks(peak_number_test[indices],
#                                                peak_number_sparse_densenet[indices],
#                                                peak_params_test_i,peak_params_sparse_densenet_i)
#     fits_n_3[R2_i] = len(np.where(peak_number_test[indices] == 3)[0])/len(np.where(peak_number_test == 3)[0])
    
#     MAE_2_R2_sparse_densenet[:,:,R2_i] = errors_2_sub_peaks(peak_number_test[indices],
#                                                peak_number_sparse_densenet[indices],
#                                                peak_params_test_i,peak_params_sparse_densenet_i)
#     fits_n_2[R2_i] = len(np.where(peak_number_test[indices] == 2)[0])/len(np.where(peak_number_test == 2)[0])


# plt.figure(figsize=(10,10))
# plt.subplot(221)
# plt.plot(R2_range,np.squeeze(MAE_5_R2_sparse_densenet[0,:,:]).T)
# plt.grid()
# plt.xlabel("Min R2 value")
# plt.ylabel("MAE on position")
# plt.legend(["Sub-peak 1","Sub-peak 2","Sub-peak 3","Sub-peak 4","Sub-peak 5"])
# right_axis = plt.gca().twinx()
# right_axis.plot(R2_range,100*fits_n_5,'r--')
# plt.ylabel("Portion of valid fits (%)",color='r')
# plt.title("Five sub-peaks")

# plt.subplot(222)
# plt.plot(R2_range,np.squeeze(MAE_4_R2_sparse_densenet[0,:,:]).T)
# plt.grid()
# plt.xlabel("Min R2 value")
# plt.ylabel("MAE on position")
# right_axis = plt.gca().twinx()
# right_axis.plot(R2_range,100*fits_n_4,'r--')
# plt.ylabel("Portion of valid fits (%)",color='r')
# plt.title("Four sub-peaks")

# plt.subplot(223)
# plt.plot(R2_range,np.squeeze(MAE_3_R2_sparse_densenet[0,:,:]).T)
# plt.grid()
# plt.xlabel("Min R2 value")
# plt.ylabel("MAE on position")
# right_axis = plt.gca().twinx()
# right_axis.plot(R2_range,100*fits_n_3,'r--')
# plt.ylabel("Portion of valid fits (%)",color='r')
# plt.title("Three sub-peaks")

# plt.subplot(224)
# plt.plot(R2_range,np.squeeze(MAE_2_R2_sparse_densenet[0,:,:]).T)
# plt.grid()
# plt.xlabel("Min R2 value")
# plt.ylabel("MAE on position")
# right_axis = plt.gca().twinx()
# right_axis.plot(R2_range,100*fits_n_2,'r--')
# plt.ylabel("Portion of valid fits (%)",color='r')
# plt.title("Two sub-peaks")
# plt.tight_layout()
# plt.savefig(figures_folder+"MAE_stats_sparse_densenet_position.jpg",dpi=300)
# plt.show()


# plt.figure(figsize=(10,10))
# plt.subplot(221)
# plt.plot(R2_range,np.squeeze(MAE_5_R2_sparse_densenet[1,:,:]).T)
# plt.grid()
# plt.xlabel("Min R2 value")
# plt.ylabel("MAE on width")
# plt.legend(["Sub-peak 1","Sub-peak 2","Sub-peak 3","Sub-peak 4","Sub-peak 5"])
# right_axis = plt.gca().twinx()
# right_axis.plot(R2_range,100*fits_n_5,'r--')
# plt.ylabel("Portion of valid fits (%)",color='r')
# plt.title("Five sub-peaks")

# plt.subplot(222)
# plt.plot(R2_range,np.squeeze(MAE_4_R2_sparse_densenet[1,:,:]).T)
# plt.grid()
# plt.xlabel("Min R2 value")
# plt.ylabel("MAE on width")
# right_axis = plt.gca().twinx()
# right_axis.plot(R2_range,100*fits_n_4,'r--')
# plt.ylabel("Portion of valid fits (%)",color='r')
# plt.title("Four sub-peaks")

# plt.subplot(223)
# plt.plot(R2_range,np.squeeze(MAE_3_R2_sparse_densenet[1,:,:]).T)
# plt.grid()
# plt.xlabel("Min R2 value")
# plt.ylabel("MAE on width")
# right_axis = plt.gca().twinx()
# right_axis.plot(R2_range,100*fits_n_3,'r--')
# plt.ylabel("Portion of valid fits (%)",color='r')
# plt.title("Three sub-peaks")

# plt.subplot(224)
# plt.plot(R2_range,np.squeeze(MAE_2_R2_sparse_densenet[1,:,:]).T)
# plt.grid()
# plt.xlabel("Min R2 value")
# plt.ylabel("MAE on width")
# right_axis = plt.gca().twinx()
# right_axis.plot(R2_range,100*fits_n_2,'r--')
# plt.ylabel("Portion of valid fits (%)",color='r')
# plt.title("Two sub-peaks")
# plt.tight_layout()
# plt.savefig(figures_folder+"MAE_stats_sparse_densenet_width.jpg",dpi=300)
# plt.show()



# plt.figure(figsize=(10,10))
# plt.subplot(221)
# plt.plot(R2_range,np.squeeze(MAE_5_R2_sparse_densenet[2,:,:]).T)
# plt.grid()
# plt.xlabel("Min R2 value")
# plt.ylabel("MAE on amplitude")
# plt.legend(["Sub-peak 1","Sub-peak 2","Sub-peak 3","Sub-peak 4","Sub-peak 5"])
# right_axis = plt.gca().twinx()
# right_axis.plot(R2_range,100*fits_n_5,'r--')
# plt.ylabel("Portion of valid fits (%)",color='r')
# plt.title("Five sub-peaks")

# plt.subplot(222)
# plt.plot(R2_range,np.squeeze(MAE_4_R2_sparse_densenet[2,:,:]).T)
# plt.grid()
# plt.xlabel("Min R2 value")
# plt.ylabel("MAE on amplitude")
# right_axis = plt.gca().twinx()
# right_axis.plot(R2_range,100*fits_n_4,'r--')
# plt.ylabel("Portion of valid fits (%)",color='r')
# plt.title("Four sub-peaks")

# plt.subplot(223)
# plt.plot(R2_range,np.squeeze(MAE_3_R2_sparse_densenet[2,:,:]).T)
# plt.grid()
# plt.xlabel("Min R2 value")
# plt.ylabel("MAE on amplitude")
# right_axis = plt.gca().twinx()
# right_axis.plot(R2_range,100*fits_n_3,'r--')
# plt.ylabel("Portion of valid fits (%)",color='r')
# plt.title("Three sub-peaks")

# plt.subplot(224)
# plt.plot(R2_range,np.squeeze(MAE_2_R2_sparse_densenet[2,:,:]).T)
# plt.grid()
# plt.xlabel("Min R2 value")
# plt.ylabel("MAE on amplitude")
# right_axis = plt.gca().twinx()
# right_axis.plot(R2_range,100*fits_n_2,'r--')
# plt.ylabel("Portion of valid fits (%)",color='r')
# plt.title("Two sub-peaks")
# plt.tight_layout()
# plt.savefig(figures_folder+"MAE_stats_sparse_densenet_amplitude.jpg",dpi=300)
# plt.show()












