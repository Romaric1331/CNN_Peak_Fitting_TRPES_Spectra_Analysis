# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:02:06 2024

@author: ajulien
"""


#
#
#
#%% Begin of inputs

# Measurement number
measurement_i = 49

# Spectrum to plot
i_to_plot = 400

# Folder containing all data
data_folder =  "C:/Users/rsallustre/Documents/Data_transfer_Arthur_24_05_2024/XPS_ht_data/TR_PES_fits_AJ/"
# Folder containing ML data
ML_data_folder = "C:/Users/rsallustre/Documents/Data_transfer_Arthur_24_05_2024/XPS_ht_data/ML_data/"


#%% End of inputs
#
#
#


#%% Importation of libraries
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
import os
from Fit_expe_functions import recursive_fit_expe_spectrum,polish_basinhopping,pseudo_Voigt,compute_spectrum

mpl.rcParams['lines.linewidth'] = 2.5
mpl.rcParams['lines.markersize'] = 11
mpl.rcParams['font.size'] = 25
mpl.rcParams['legend.fontsize'] = 20


#%% Import Excel summary file
summary_data = pd.read_excel(data_folder+"Available_data_summary.xlsx",index_col=0)

# Identification of experiment
session_i = summary_data.loc[measurement_i,"Measurement Session"]
sample_i = summary_data.loc[measurement_i,"Sample"]
file_i = summary_data.loc[measurement_i,"File"]
core_level = summary_data.loc[measurement_i,"Core level"]

# Parameters defining the pre-processing
time_ag_n = int(summary_data.loc[measurement_i,"Time agregation number"])
time_mov_avg_win_us = float(summary_data.loc[measurement_i,"Time moving average window (us)"])
E_SG_win_meV = int(summary_data.loc[measurement_i,"Energy Savitzky-Golay window (meV)"])
E_SG_order = int(summary_data.loc[measurement_i,"Energy Savitzky-Golay order"])

# Parameters defining the fit
session_name = summary_data.loc[measurement_i,"ML training session"]
model_name = summary_data.loc[measurement_i,"ML model"]
if summary_data.loc[measurement_i,"Number of peaks"] == "Auto":
    num_peaks = None
else:
    num_peaks = int(summary_data.loc[measurement_i,"Number of peaks"])

# Folder containing pre-treated experimental data
expe_data_folder_i = data_folder+session_i+"/"+sample_i+"/"+file_i+\
    "/data_agg_"+str(time_ag_n)+"_win_"+str(time_mov_avg_win_us)+"_SG_"+str(int(E_SG_win_meV))+"_"+str(E_SG_order)+"/"


#%% Load CNN model
ML_model = load_model(ML_data_folder+session_name+"/"+model_name+"/"+model_name+".keras")

with open(ML_data_folder+session_name+"/Database/Validation_database.pkl", 'rb') as f:
    validation_database = pickle.load(f)
ML_E_range = validation_database[0]
ML_E_range_min = min(ML_E_range)
ML_E_range_max = max(ML_E_range)
ML_E_range_n = len(ML_E_range)


#%% Load experimental data
# Data sub-folder
fit_folder = expe_data_folder_i+session_name+"_"+model_name+"/"
os.makedirs(fit_folder,exist_ok=True)

# Load pre-processed experimental data
with open(expe_data_folder_i+"Pre_processed_data.pkl", 'rb') as f:
    filtered_data_energy,expe_data,BE_range,time_range = pickle.load(f)
spectra_n = np.size(expe_data,0)
BE_range_n = np.size(expe_data,1)


#%% Normalise experimental data for ML processing
# Normalize peak maximum height to 1
intensity_norm = 1/np.max(expe_data.to_numpy()) # 1/(counts.s-1)

# Shift energy range to be between 0 and 15, no scale change
BE_span = max(BE_range)-min(BE_range)
ML_E_span = ML_E_range_max - ML_E_range_min
BE_range_ML = np.linspace(min(BE_range)-(ML_E_span-BE_span)/2,max(BE_range)+(ML_E_span-BE_span)/2,ML_E_range_n)
E_shift = min(BE_range_ML)-min(ML_E_range)

expe_data_ML = np.zeros([spectra_n,ML_E_range_n])
for spectrum_i in range(spectra_n):
    interp1d_i = interp1d(BE_range,expe_data.iloc[spectrum_i,:].to_numpy()*intensity_norm,fill_value=0,bounds_error=False)
    expe_data_ML[spectrum_i,:] =  interp1d_i(BE_range_ML)


#%% Fit experimental spectra with Sparse densenet model
fit_spectra = np.zeros([spectra_n,BE_range_n])
fit_params = np.zeros((spectra_n,5,4))*np.nan
fit_time = np.zeros(spectra_n)

fit_spectra_ep = np.zeros([spectra_n,BE_range_n])
fit_params_ep = np.zeros((spectra_n,5,4))*np.nan
fit_time_ep = np.zeros(spectra_n)
polish_n = 0
for spectrum_i in tqdm(range(spectra_n),desc="Iterative fitting",smoothing=0.1):
    
    # Iterative fit
    t0 = time.time()
    fit_spectrum_i,fit_params[spectrum_i,:,:] = \
        recursive_fit_expe_spectrum(ML_model,expe_data_ML[spectrum_i,:],ML_E_range,num_peaks)
    fit_time[spectrum_i] = time.time()-t0
    
    # External polish
    t0 = time.time()
    fit_spectrum_i_ep,fit_params_ep[spectrum_i,:,:],do_polish = \
        polish_basinhopping(expe_data_ML[spectrum_i,:],ML_E_range,fit_params[spectrum_i,:,:],fit_spectrum_i,1)
    if do_polish: polish_n = polish_n+1
    fit_time_ep[spectrum_i] = time.time()-t0
    
    # Re-order peaks by order of BE position
    order = np.argsort(fit_params[spectrum_i,:,0])
    fit_params[spectrum_i,:,3] = fit_params[spectrum_i,0,3]*np.ones(5)
    fit_params[spectrum_i,:,:] = fit_params[spectrum_i,order,:]
    
    order_ep = np.argsort(fit_params_ep[spectrum_i,:,0])
    fit_params_ep[spectrum_i,:,3] = fit_params_ep[spectrum_i,0,3]*np.ones(5)
    fit_params_ep[spectrum_i,:,:] = fit_params_ep[spectrum_i,order_ep,:]
    
    # Convert back fit params into physical quantities
    fit_params[spectrum_i,:,0] = fit_params[spectrum_i,:,0]+E_shift # eV
    fit_params[spectrum_i,:,1] = fit_params[spectrum_i,:,1] # eV
    fit_params[spectrum_i,:,2] = fit_params[spectrum_i,:,2]/intensity_norm # counts.s-1
    fit_spectra[spectrum_i] = compute_spectrum(BE_range,fit_params[spectrum_i,:,:])
    
    fit_params_ep[spectrum_i,:,0] = fit_params_ep[spectrum_i,:,0]+E_shift # eV
    fit_params_ep[spectrum_i,:,1] = fit_params_ep[spectrum_i,:,1] # eV
    fit_params_ep[spectrum_i,:,2] = fit_params_ep[spectrum_i,:,2]/intensity_norm # counts.s-1
    fit_spectra_ep[spectrum_i] = compute_spectrum(BE_range,fit_params_ep[spectrum_i,:,:])


avg_fit_time = np.mean(fit_time)
avg_fit_time_ep = np.sum(fit_time_ep)/spectra_n

fit_str = '\n'.join(["Session: "+session_name,"Model: "+model_name,
                     "Avg fit time (only iterative CNN): "+format(avg_fit_time,".2f")+" s per spectrum",
                     "Avg fit time (iterative CNN + end polish): "+format(avg_fit_time_ep,".2f")+" s per spectrum, "+str(polish_n)+"/"+str(spectra_n)+" spectra treated"])
print(fit_str)

# Print text summary
with open(fit_folder+"Fit_summary.txt", 'w') as f:
    f.writelines(fit_str)
    f.close()


#%% Plot one example of fit
gs_kw = dict(height_ratios=[2,1])

# Spectrum to plot
i_to_plot = 2

# Recursive CNN fit
fig, axd = plt.subplot_mosaic([['upper'],['lower']],gridspec_kw=gs_kw, figsize=(10,10))
axd['upper'].fill_between(BE_range,expe_data.iloc[i_to_plot,:],color='k',edgecolor=None,alpha=0.2,label="Experimental spectrum")
axd['upper'].plot(BE_range,fit_spectra[i_to_plot],'k',label=model_name+" recursive fit")
for i in range(round(fit_params[i_to_plot,0,3])):
    axd['upper'].plot(BE_range,
             pseudo_Voigt(BE_range,
                          fit_params[i_to_plot,i,0],fit_params[i_to_plot,i,1],fit_params[i_to_plot,i,2]),
             color=colors[i])
axd['upper'].grid()
axd['upper'].set_xlim((min(BE_range),max(BE_range)))
axd['upper'].set_ylabel("Intensity (couts.s-1)")
axd['upper'].legend(loc='upper left')

axd['lower'].plot(BE_range,expe_data.iloc[i_to_plot,:]-np.squeeze(fit_spectra[i_to_plot]))
axd['lower'].grid()
axd['lower'].set_xlim((min(BE_range),max(BE_range)))
axd['lower'].set_xlabel("Binding energy (eV)")
axd['lower'].set_ylabel("Error (couts.s-1)")

plt.tight_layout()
plt.savefig(fit_folder+"Fit_example_"+str(i_to_plot)+".jpg",dpi=300)
plt.show()

# Recursive CNN fit + external polish
fig, axd = plt.subplot_mosaic([['upper'],['lower']],gridspec_kw=gs_kw, figsize=(10,10))
axd['upper'].fill_between(BE_range,expe_data.iloc[i_to_plot,:],color='k',edgecolor=None,alpha=0.2,label="Experimental spectrum")
axd['upper'].plot(BE_range,fit_spectra_ep[i_to_plot],'k',label=model_name+" recursive fit + external polish")
for i in range(round(fit_params_ep[i_to_plot,0,3])):
    axd['upper'].plot(BE_range,
             pseudo_Voigt(BE_range,
                          fit_params_ep[i_to_plot,i,0],fit_params_ep[i_to_plot,i,1],fit_params_ep[i_to_plot,i,2]),
             color=colors[i])
axd['upper'].grid()
axd['upper'].set_xlim((min(BE_range),max(BE_range)))
axd['upper'].set_ylabel("Intensity (couts.s-1)")
axd['upper'].legend(loc='upper left')

axd['lower'].plot(BE_range,expe_data.iloc[i_to_plot,:]-np.squeeze(fit_spectra_ep[i_to_plot]))
axd['lower'].grid()
axd['lower'].set_xlim((min(BE_range),max(BE_range)))
axd['lower'].set_xlabel("Binding energy (eV)")
axd['lower'].set_ylabel("Error (couts.s-1)")

plt.tight_layout()
plt.savefig(fit_folder+"Fit_example_"+str(i_to_plot)+"_ep.jpg",dpi=300)
plt.show()


#%% Save fitting results
# One pickle file containing all outputs
with open(fit_folder+"Fit_results.pkl", 'wb') as f:
    pickle.dump([fit_spectra,fit_spectra_ep,fit_params,fit_params_ep,BE_range,time_range], f)

# Four csv files containing one output each
np.savetxt(fit_folder+"Fit_data.txt",fit_spectra,delimiter='\t')
np.savetxt(fit_folder+"Fit_data_ep.txt",fit_spectra_ep,delimiter='\t')
np.savetxt(fit_folder+"Binding_energy_range_eV.txt",np.reshape(BE_range,(1,len(BE_range))),delimiter='\t')
np.savetxt(fit_folder+"Time_range_us.txt",time_range,delimiter='\t')

# One csv file per peak contribution over time
header_str = "Time (us)\tPosition (eV)\tWidth (eV)\tAmplitude (counts.s-1)\tNumber of peaks"
np.savetxt(fit_folder+"Fit_params_0.txt",
           np.hstack((np.reshape(time_range,(len(time_range),1)),fit_params[:,0,:])),delimiter='\t',header=header_str)
np.savetxt(fit_folder+"Fit_params_1.txt"
           ,np.hstack((np.reshape(time_range,(len(time_range),1)),fit_params[:,1,:])),delimiter='\t',header=header_str)
np.savetxt(fit_folder+"Fit_params_2.txt",
           np.hstack((np.reshape(time_range,(len(time_range),1)),fit_params[:,2,:])),delimiter='\t',header=header_str)
np.savetxt(fit_folder+"Fit_params_3.txt",
           np.hstack((np.reshape(time_range,(len(time_range),1)),fit_params[:,3,:])),delimiter='\t',header=header_str)
np.savetxt(fit_folder+"Fit_params_4.txt",
           np.hstack((np.reshape(time_range,(len(time_range),1)),fit_params[:,4,:])),delimiter='\t',header=header_str)

np.savetxt(fit_folder+"Fit_params_0_ep.txt",
           np.hstack((np.reshape(time_range,(len(time_range),1)),fit_params_ep[:,0,:])),delimiter='\t',header=header_str)
np.savetxt(fit_folder+"Fit_params_1_ep.txt",
           np.hstack((np.reshape(time_range,(len(time_range),1)),fit_params_ep[:,1,:])),delimiter='\t',header=header_str)
np.savetxt(fit_folder+"Fit_params_2_ep.txt",
           np.hstack((np.reshape(time_range,(len(time_range),1)),fit_params_ep[:,2,:])),delimiter='\t',header=header_str)
np.savetxt(fit_folder+"Fit_params_3_ep.txt",
           np.hstack((np.reshape(time_range,(len(time_range),1)),fit_params_ep[:,3,:])),delimiter='\t',header=header_str)
np.savetxt(fit_folder+"Fit_params_4_ep.txt",
           np.hstack((np.reshape(time_range,(len(time_range),1)),fit_params_ep[:,4,:])),delimiter='\t',header=header_str)


#%% Plot evolution of first contribution parameters
peak_pos_avg = np.nanmedian(fit_params[:,:,0],0)
plt.figure(figsize=(10,10))
plt.plot(time_range,fit_params[:,0,0]-peak_pos_avg[0],'+:',label="1st peak (+ "+format(peak_pos_avg[0],'.2f')+" eV)")
plt.plot(time_range,fit_params[:,1,0]-peak_pos_avg[1],'+:',label="2nd peak (+ "+format(peak_pos_avg[1],'.2f')+" eV)")
plt.plot(time_range,fit_params[:,2,0]-peak_pos_avg[2],'+:',label="3rd peak (+ "+format(peak_pos_avg[2],'.2f')+" eV)")
plt.plot(time_range,fit_params[:,3,0]-peak_pos_avg[3],'+:',label="4th peak (+ "+format(peak_pos_avg[3],'.2f')+" eV)")
plt.plot(time_range,fit_params[:,4,0]-peak_pos_avg[4],'+:',label="5th peak (+ "+format(peak_pos_avg[4],'.2f')+" eV)")
plt.grid()
plt.ylim((-0.07,0.07))
plt.xlabel("Time ($\mu s$)")
plt.ylabel("Peak position (eV)")
plt.legend()
plt.tight_layout()
plt.savefig(fit_folder+"Peak_pos_fit.jpg",dpi=300)
plt.show()

peak_pos_avg = np.nanmedian(fit_params_ep[:,:,0],0)
plt.figure(figsize=(10,10))
plt.plot(time_range,fit_params_ep[:,0,0]-peak_pos_avg[0],'+:',label="1st peak (+ "+format(peak_pos_avg[0],'.2f')+" eV)")
plt.plot(time_range,fit_params_ep[:,1,0]-peak_pos_avg[1],'+:',label="2nd peak (+ "+format(peak_pos_avg[1],'.2f')+" eV)")
plt.plot(time_range,fit_params_ep[:,2,0]-peak_pos_avg[2],'+:',label="3rd peak (+ "+format(peak_pos_avg[2],'.2f')+" eV)")
plt.plot(time_range,fit_params_ep[:,3,0]-peak_pos_avg[3],'+:',label="4th peak (+ "+format(peak_pos_avg[3],'.2f')+" eV)")
plt.plot(time_range,fit_params_ep[:,4,0]-peak_pos_avg[4],'+:',label="5th peak (+ "+format(peak_pos_avg[4],'.2f')+" eV)")
plt.grid()
plt.ylim((-0.07,0.07))
plt.xlabel("Time ($\mu s$)")
plt.ylabel("Peak position (eV)")
plt.legend()
plt.tight_layout()
plt.savefig(fit_folder+"Peak_pos_fit_ep.jpg",dpi=300)
plt.show()

peak_width_avg = np.nanmedian(fit_params[:,:,1],0)
plt.figure(figsize=(10,10))
plt.plot(time_range,fit_params[:,0,1]-peak_width_avg[0],'+:',label="1st peak (+ "+format(peak_width_avg[0],'.2f')+" eV)")
plt.plot(time_range,fit_params[:,1,1]-peak_width_avg[1],'+:',label="2nd peak (+ "+format(peak_width_avg[1],'.2f')+" eV)")
plt.plot(time_range,fit_params[:,2,1]-peak_width_avg[2],'+:',label="3rd peak (+ "+format(peak_width_avg[2],'.2f')+" eV)")
plt.plot(time_range,fit_params[:,3,1]-peak_width_avg[3],'+:',label="4th peak (+ "+format(peak_width_avg[3],'.2f')+" eV)")
plt.plot(time_range,fit_params[:,4,1]-peak_width_avg[4],'+:',label="5th peak (+ "+format(peak_width_avg[4],'.2f')+" eV)")
plt.grid()
plt.ylim((-0.06,0.06))
plt.xlabel("Time ($\mu s$)")
plt.ylabel("Peak width (eV)")
plt.legend()
plt.tight_layout()
plt.savefig(fit_folder+"Peak_width_fit.jpg",dpi=300)
plt.show()

peak_width_avg = np.nanmedian(fit_params_ep[:,:,1],0)
plt.figure(figsize=(10,10))
plt.plot(time_range,fit_params_ep[:,0,1]-peak_width_avg[0],'+:',label="1st peak (+ "+format(peak_width_avg[0],'.2f')+" eV)")
plt.plot(time_range,fit_params_ep[:,1,1]-peak_width_avg[1],'+:',label="2nd peak (+ "+format(peak_width_avg[1],'.2f')+" eV)")
plt.plot(time_range,fit_params_ep[:,2,1]-peak_width_avg[2],'+:',label="3rd peak (+ "+format(peak_width_avg[2],'.2f')+" eV)")
plt.plot(time_range,fit_params_ep[:,3,1]-peak_width_avg[3],'+:',label="4th peak (+ "+format(peak_width_avg[3],'.2f')+" eV)")
plt.plot(time_range,fit_params_ep[:,4,1]-peak_width_avg[4],'+:',label="5th peak (+ "+format(peak_width_avg[4],'.2f')+" eV)")
plt.grid()
plt.ylim((-0.06,0.06))
plt.xlabel("Time ($\mu s$)")
plt.ylabel("Peak width (eV)")
plt.legend()
plt.tight_layout()
plt.savefig(fit_folder+"Peak_width_fit_ep.jpg",dpi=300)
plt.show()

peak_amp_avg = np.nanmedian(fit_params[:,:,2],0)
plt.figure(figsize=(10,10))
plt.plot(time_range,fit_params[:,0,2]/peak_amp_avg[0],'+:',label="1st peak (x "+format(peak_amp_avg[0],'.2e')+" couts.s-1)")
plt.plot(time_range,fit_params[:,1,2]/peak_amp_avg[1],'+:',label="2nd peak (x "+format(peak_amp_avg[1],'.2e')+" couts.s-1)")
plt.plot(time_range,fit_params[:,2,2]/peak_amp_avg[2],'+:',label="3rd peak (x "+format(peak_amp_avg[2],'.2e')+" couts.s-1)")
plt.plot(time_range,fit_params[:,3,2]/peak_amp_avg[3],'+:',label="4th peak (x "+format(peak_amp_avg[3],'.2e')+" couts.s-1)")
plt.plot(time_range,fit_params[:,4,2]/peak_amp_avg[4],'+:',label="5th peak (x "+format(peak_amp_avg[4],'.2e')+" couts.s-1)")
plt.grid()
plt.ylim((0.9,1.1))
plt.xlabel("Time ($\mu s$)")
plt.ylabel("Peak amplitude (norm)")
plt.legend()
plt.tight_layout()
plt.savefig(fit_folder+"Peak_amp_fit.jpg",dpi=300)
plt.show()

peak_amp_avg = np.nanmedian(fit_params_ep[:,:,2],0)
plt.figure(figsize=(10,10))
plt.plot(time_range,fit_params_ep[:,0,2]/peak_amp_avg[0],'+:',label="1st peak (x "+format(peak_amp_avg[0],'.2e')+" couts.s-1)")
plt.plot(time_range,fit_params_ep[:,1,2]/peak_amp_avg[1],'+:',label="2nd peak (x "+format(peak_amp_avg[1],'.2e')+" couts.s-1)")
plt.plot(time_range,fit_params_ep[:,2,2]/peak_amp_avg[2],'+:',label="3rd peak (x "+format(peak_amp_avg[2],'.2e')+" couts.s-1)")
plt.plot(time_range,fit_params_ep[:,3,2]/peak_amp_avg[3],'+:',label="4th peak (x "+format(peak_amp_avg[3],'.2e')+" couts.s-1)")
plt.plot(time_range,fit_params_ep[:,4,2]/peak_amp_avg[4],'+:',label="5th peak (x "+format(peak_amp_avg[4],'.2e')+" couts.s-1)")
plt.grid()
plt.ylim((0.9,1.1))
plt.xlabel("Time ($\mu s$)")
plt.ylabel("Peak amplitude (norm)")
plt.legend()
plt.tight_layout()
plt.savefig(fit_folder+"Peak_amp_fit_ep.jpg",dpi=300)
plt.show()

plt.figure(figsize=(10,10))
plt.plot(time_range,fit_params[:,0,3],'+:')
plt.grid()
plt.xlabel("Time ($\mu s$)")
plt.ylabel("Number of contributions")
plt.tight_layout()
plt.savefig(fit_folder+"Peak_num_fit.jpg",dpi=300)
plt.show()
