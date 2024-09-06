# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:02:06 2024

@author: ajulien
"""


#%% Importation of packages
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tqdm import tqdm
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
colors = list(mcolors.TABLEAU_COLORS)
import matplotlib as mpl
from scipy.interpolate import interp1d
import sys
sys.path.append('C:/Users/ajulien/Documents/Codes/GitHub/IA_spectro/ML_peak_fitting_Park_adapted/')
from Fit_functions import recursive_fit_A,polish_basinhopping,pseudo_Voigt

#%% Inputs
# Folder containing trained CNN
main_data_folder = "C:/Users/ajulien/Documents/General_modeling_data/XPS_ht_data/ML_data/"
session_name = "ML_models_large_database_V1"
model_name = "Sparse_densenet"

# Define energy range processed by CNN
ML_E_range_min = 0
ML_E_range_max = 15
ML_E_range_n = 401

# Folder containing experimental data
expe_data_folder = "C:/Users/ajulien/Documents/General_modeling_data/XPS_ht_data/TR_PES_fits_AJ/"
measurement_i = 3

R2_min = 0.995

mpl.rcParams['lines.linewidth'] = 2.5
mpl.rcParams['lines.markersize'] = 11
mpl.rcParams['font.size'] = 25
mpl.rcParams['legend.fontsize'] = 20


#%% Load CNN model
ML_data_folder = main_data_folder+session_name+"/"
model_folder = ML_data_folder+model_name+"/"
model_file = model_folder+model_name+".keras"
sparse_densenet_model = load_model(model_file)

# Define energy range processed by CNN
ML_E_range = np.linspace(ML_E_range_min,ML_E_range_max,ML_E_range_n)


#%% Load experimental data
summary_data = pd.read_excel(expe_data_folder+"Available_data_summary.xlsx",index_col=0)

# Identification of experiment
session_i = summary_data.loc[measurement_i,"Measurement Session"]
sample_i = summary_data.loc[measurement_i,"Sample"]
file_i = summary_data.loc[measurement_i,"File"]

# Data sub-folder
expe_data_folder_i = expe_data_folder+session_i+"/"+sample_i+"/"+file_i+"/"

# Number of neighbors to aggeragte
time_ag_n = int(summary_data.loc[measurement_i,"Time agregation number"])

# Window size for smoothing over time
time_mov_avg_win = int(summary_data.loc[measurement_i,"Time moving average window"])

# Parameters for Savitzky-Golay filter
E_SG_win = int(summary_data.loc[measurement_i,"Energy Savitzky-Golay window"])
E_SG_order = int(summary_data.loc[measurement_i,"Energy Savitzky-Golay order"])

# Load pre-processed experimental data
expe_data_file = "Cleaned_data_agg_"+str(time_ag_n)+"_win_"+str(time_mov_avg_win)+"_SG_"+str(E_SG_win)+"_"+str(E_SG_order)+"_cst_background"
with open(expe_data_folder_i+expe_data_file+".pkl", 'rb') as f:
    filtered_data_energy,expe_data,BE_range,time_range,summary_data_i = pickle.load(f)
spectra_n = np.size(expe_data,0)
E_range_n = np.size(expe_data,1)


#%% Normalise experimental data for ML processing
# Normalize peak maximum height to 1
intensity_norm = 1/max(expe_data)

# Normalize energy range to be between 0 and 15
BE_span = max(BE_range)-min(BE_range)
ML_E_span = ML_E_range_max - ML_E_range_min
BE_range_ML = np.linspace(min(BE_range)-(ML_E_span-BE_span)/2,max(BE_range)+(ML_E_span-BE_span)/2,ML_E_range_n)
E_shift = min(BE_range_ML)-min(ML_E_range)

expe_data_ML = np.zeros([spectra_n,ML_E_range_n])
for spectrum_i in range(spectra_n):
    interp1d_i = interp1d(BE_range,expe_data.iloc[spectrum_i,:].to_numpy()*intensity_norm,fill_value=0,bounds_error=False)
    expe_data_ML[spectrum_i,:] =  interp1d_i(BE_range_ML)


#%% Fit experimental spectra with Sparse densenet model
fitted_spectra_A = np.zeros([spectra_n,ML_E_range_n,1])
fitted_params_A = []
fit_time_A = np.zeros(spectra_n)

fitted_spectra_A_ep = np.zeros([spectra_n,ML_E_range_n,1])
fitted_params_A_ep = []
fit_time_A_ep = np.zeros(spectra_n)
polish_n_A = 0
for spectrum_i in tqdm(range(spectra_n),desc="Iterative fitting",smoothing=0.1):
    
    t0 = time.time()
    fitted_spectra_A[spectrum_i],fitted_params_i = recursive_fit_A(sparse_densenet_model,expe_data_ML[spectrum_i,:],ML_E_range)
    fitted_params_A.append(fitted_params_i)
    fit_time_A[spectrum_i] = time.time()-t0
    
    t0 = time.time()
    fitted_spectra_A_ep[spectrum_i],fitted_params_i,do_polish = polish_basinhopping(expe_data_ML[spectrum_i,:],ML_E_range,fitted_params_A[spectrum_i],fitted_spectra_A[spectrum_i],R2_min)
    fitted_params_A_ep.append(fitted_params_i)
    if do_polish: polish_n_A = polish_n_A+1
    fit_time_A_ep[spectrum_i] = time.time()-t0

avg_time_A = np.mean(fit_time_A)
avg_time_A_ep = np.sum(fit_time_A_ep)/spectra_n
print("Avg fit time (iterative fit A): "+format(avg_time_A,".2f")+" s per spectrum")
print("Avg end polish time after iterative fit A: "+format(avg_time_A_ep,".2f")+" s per spectrum, "+str(polish_n_A)+" spectra treated")


#%% Plot one example of fit
gs_kw = dict(height_ratios=[2,1])

# Spectrum to plot
i_to_plot = 40

# Fit A
fig, axd = plt.subplot_mosaic([['upper'],['lower']],gridspec_kw=gs_kw, figsize=(10,7.5))
axd['upper'].fill_between(BE_range_ML,expe_data_ML[i_to_plot,:],color='k',edgecolor=None,alpha=0.2,label="Experimental spectrum")
axd['upper'].plot(BE_range_ML,fitted_spectra_A[i_to_plot],'k',label="Recursive fit")
for i in range(len(fitted_params_A[i_to_plot])):
    axd['upper'].plot(BE_range_ML,
             pseudo_Voigt(ML_E_range,
                          fitted_params_A[i_to_plot][i][0],fitted_params_A[i_to_plot][i][1],fitted_params_A[i_to_plot][i][2]),
             color=colors[i])
axd['upper'].grid()
axd['upper'].set_xlim((min(BE_range),max(BE_range)))
axd['upper'].set_ylabel("Intensity")
axd['upper'].legend()
axd['upper'].set_title(model_name+" - A recursive fit")

axd['lower'].plot(BE_range_ML,expe_data_ML[i_to_plot,:]-np.squeeze(fitted_spectra_A[i_to_plot]))
axd['lower'].grid()
axd['lower'].set_xlim((min(BE_range),max(BE_range)))
axd['lower'].set_ylim((-0.1,0.1))
axd['lower'].set_xlabel("Energy")
axd['lower'].set_ylabel("Error")

plt.tight_layout()
plt.savefig(expe_data_folder_i+"Fit_example_"+str(i_to_plot)+"_A.jpg",dpi=300)
plt.show()

# Fit A + external polish
fig, axd = plt.subplot_mosaic([['upper'],['lower']],gridspec_kw=gs_kw, figsize=(10,7.5))
axd['upper'].fill_between(BE_range_ML,expe_data_ML[i_to_plot,:],color='k',edgecolor=None,alpha=0.2,label="Experimental spectrum")
axd['upper'].plot(BE_range_ML,fitted_spectra_A_ep[i_to_plot],'k',label="Recursive fit + external polish")
for i in range(len(fitted_params_A_ep[i_to_plot])):
    axd['upper'].plot(BE_range_ML,
             pseudo_Voigt(ML_E_range,
                          fitted_params_A_ep[i_to_plot][i][0],fitted_params_A_ep[i_to_plot][i][1],fitted_params_A_ep[i_to_plot][i][2]),
             color=colors[i])
axd['upper'].grid()
axd['upper'].set_xlim((min(BE_range),max(BE_range)))
axd['upper'].set_ylabel("Intensity")
axd['upper'].legend()
axd['upper'].set_title(model_name+" - A recursive fit + external polish")

axd['lower'].plot(BE_range_ML,expe_data_ML[i_to_plot,:]-np.squeeze(fitted_spectra_A_ep[i_to_plot]))
axd['lower'].grid()
axd['lower'].set_xlim((min(BE_range),max(BE_range)))
axd['lower'].set_ylim((-0.1,0.1))
axd['lower'].set_xlabel("Energy")
axd['lower'].set_ylabel("Error")

plt.tight_layout()
plt.savefig(expe_data_folder_i+"Fit_example_"+str(i_to_plot)+"_A_ep.jpg",dpi=300)
plt.show()



#%% Plot evolution of first contribution parameters
plt.figure(figsize=(10,10))
plt.plot(time_range,[fitted_params_A_ep[i][0][0] for i in range(spectra_n)]+E_shift,'+:')
#plt.plot(time_range,[fitted_params_A_ep[i][1][0] for i in range(spectra_n)]+E_shift,'+:')
plt.grid()
plt.ylim((619.9,620.1))
plt.xlabel("Time ($\mu s$)")
plt.ylabel("Peak position (eV)")
plt.legend(("First contribution","Second contribution"))
plt.tight_layout()
plt.savefig(expe_data_folder_i+"Peak_1_pos_time.jpg",dpi=300)
plt.show()

plt.figure(figsize=(10,10))
plt.plot(time_range,[fitted_params_A_ep[i][0][1] for i in range(spectra_n)],'+:')
plt.grid()
plt.xlabel("Time ($\mu s$)")
plt.ylabel("Peak 1 width (eV)")
plt.tight_layout()
plt.savefig(expe_data_folder_i+"Peak_1_width_time.jpg",dpi=300)
plt.show()

plt.figure(figsize=(10,10))
plt.plot(time_range,[fitted_params_A_ep[i][0][2] for i in range(spectra_n)],'+:')
plt.plot(time_range,[fitted_params_A_ep[i][1][2] for i in range(spectra_n)],'+:')
plt.grid()
plt.xlabel("Time ($\mu s$)")
plt.ylabel("Peak amplitude (a.u.)")
plt.legend(("First contribution","Second contribution"))
plt.tight_layout()
plt.savefig(expe_data_folder_i+"Peak_1_amp_time.jpg",dpi=300)
plt.show()


