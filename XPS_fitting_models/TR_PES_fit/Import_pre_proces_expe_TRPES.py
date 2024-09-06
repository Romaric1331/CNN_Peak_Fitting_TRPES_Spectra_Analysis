# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:11:15 2024

@author: ajulien
"""

#%% Importation of libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import matplotlib as mpl
import pickle
import os


#%% Define filters
# Moving average
def mov_avg_filter(data,win_size):
    window = np.ones(win_size)/win_size
    return scipy.ndimage.convolve(data, window, mode='nearest')

# Savitzky-Golay filter in energy dimension
def savgol_filter_1(data, win_size, poly_order):
    filtered = scipy.signal.savgol_filter(data, win_size, poly_order, mode='nearest').T
    filtered[filtered<0]=0
    return filtered


#%% Define background
def remove_constant_background(data,background_begin,background_end):
    data_numpy = data.to_numpy()
    background_value = np.nanmean(data.loc[background_begin:background_end])

    clean_data = data_numpy-background_value
    clean_data[clean_data<0] = 0
    return clean_data


#%% Inputs
data_folder = "C:/Users/ajulien/Documents/General_modeling_data/XPS_ht_data/TR_PES_fits_AJ/"
measurement_i = 3

# Spectrum to plot
i_to_plot = 400

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

# Raw data matrix orientation
mat_or = summary_data.loc[measurement_i,"Matrix orientiation"]

# Energy indices sequence to extract
E_i_begin = int(summary_data.loc[measurement_i,"E i begin"])
E_i_end = int(summary_data.loc[measurement_i,"E i end"])

# Number of neighbors to aggeragte
time_ag_n = int(summary_data.loc[measurement_i,"Time agregation number"])

# Window size for smoothing over time
time_mov_avg_win = int(summary_data.loc[measurement_i,"Time moving average window"])

# Parameters for Savitzky-Golay filter
E_SG_win = int(summary_data.loc[measurement_i,"Energy Savitzky-Golay window"])
E_SG_order = int(summary_data.loc[measurement_i,"Energy Savitzky-Golay order"])

# Range to identify constant background range
background_begin = int(summary_data.loc[measurement_i,"Constant background begin"])
background_end = int(summary_data.loc[measurement_i,"Constant background end"])

# Energy range
BE_begin = int(summary_data.loc[measurement_i,"BE begin (eV)"])
BE_end = int(summary_data.loc[measurement_i,"BE end (eV)"])

# Time resolution
t_int = int(summary_data.loc[measurement_i,"Time resolution (ns)"])


#%% Import raw data file
raw_data_file = data_folder+session_i+"/"+sample_i+"/"+file_i
raw_data = pd.read_csv(raw_data_file+".txt", delimiter='\t', header=None)
if mat_or == "(E,time)":
    raw_data = raw_data.transpose()
raw_data_zero = raw_data > 0
raw_data = raw_data.iloc[np.where(raw_data_zero.any(axis=1))[0],:]

# Define energy range
E_range_n = np.size(raw_data,1)
BE_range_raw = np.linspace(BE_end,BE_begin,E_range_n) # eV

# Define time range
t_range_n = np.size(raw_data,0)
time_range_raw = np.linspace(0,(t_range_n-1)*t_int,t_range_n) * 1e-3 # us

# Define output folder
output_folder = data_folder+session_i+"/"+sample_i+"/"+file_i+"/"
os.makedirs(output_folder,exist_ok=True)

# Plot raw data
plt.figure(figsize=(10,10))
plt.imshow(raw_data, aspect='auto',extent=[BE_range_raw[0],BE_range_raw[-1],time_range_raw[0],time_range_raw[-1]],interpolation=None)
plt.xlabel("Binding energy (eV)")
plt.ylabel("Time ($\mu s$)")
plt.title(core_level)
plt.tight_layout()
plt.savefig(output_folder+"All_data.jpg",dpi=300)
plt.show()


#%% Aggregate spectra
agregate_data = raw_data.iloc[:,E_i_begin:E_i_end].groupby(raw_data.index // time_ag_n).sum()
BE_range = BE_range_raw[E_i_begin:E_i_end]
time_range = time_range_raw[0:-1:time_ag_n]


#%% Apply filters
# On aggregated data in time dimension
filtered_data_time = agregate_data.apply(mov_avg_filter, axis=0, args=(time_mov_avg_win,))

# In energy dimension
filtered_data_energy = filtered_data_time.apply(savgol_filter_1, axis=1, args=(E_SG_win,E_SG_order), result_type='broadcast')


#%% Remove constant background
clean_data = filtered_data_energy.apply(remove_constant_background, axis=1, args=(background_begin,background_end), result_type='broadcast',raw=False)


#%% Plot one example of spectrum
i_to_plot_ag = int(round((i_to_plot + time_ag_n/2)/time_ag_n))

plt.figure(figsize=(10,10))
plt.plot(raw_data.loc[i_to_plot_ag,:],'+:',label="Raw data")
plt.plot(agregate_data.loc[i_to_plot_ag,:]/time_ag_n,'+:',label="Aggregated ("+str(time_ag_n)+" neighbours)")
plt.plot(filtered_data_time.loc[i_to_plot_ag,:]/time_ag_n,'+:',label="Agg + time avg (win size="+str(time_mov_avg_win)+")")
plt.plot(filtered_data_energy.loc[i_to_plot_ag,:]/time_ag_n,'+:',label="Agg + time avg + SG over E (win size="+str(E_SG_win)+" order="+str(E_SG_order)+")")
plt.xlabel("Energy")
plt.ylabel("Counts")
plt.legend()
plt.title(core_level+": spectrum after "+str(time_range_raw[i_to_plot])+" $\mu s$")
plt.tight_layout()
plt.savefig(output_folder+"Spectrum_"+str(i_to_plot)+"_agg_"+str(time_ag_n)+"_win_"+str(time_mov_avg_win)+"_SG_"+str(E_SG_win)+"_"+str(E_SG_order)+".jpg",dpi=300)
plt.show()

plt.figure(figsize=(10,10))
plt.plot(BE_range,filtered_data_energy.loc[i_to_plot_ag,:]/time_ag_n,'+:',label="Filtered")
#plt.plot(BE_range,clean_data.loc[i_to_plot_ag,:]/time_ag_n,'+:',label="Background removed")
plt.xlim((max(BE_range),min(BE_range)))
plt.xlabel("Binding energy (eV)")
plt.ylabel("Counts")
plt.legend()
plt.title(core_level+": spectrum after "+str(time_range_raw[i_to_plot])+" $\mu s$")
plt.tight_layout()
plt.savefig(output_folder+"Spectrum_"+str(i_to_plot)+"_agg_"+str(time_ag_n)+"_win_"+str(time_mov_avg_win)+"_SG_"+str(E_SG_win)+"_"+str(E_SG_order)+"_cst_background.jpg",dpi=300)
plt.show()

plt.figure(figsize=(10,10))
plt.imshow(clean_data, aspect='auto',extent=[BE_range[0],BE_range[-1],time_range[0],time_range[-1]],interpolation=None)
plt.xlabel("Binding energy (eV)")
plt.ylabel("Time ($\mu s$)")
plt.title(core_level)
plt.tight_layout()
plt.savefig(output_folder+"All_data_agg_"+str(time_ag_n)+"_win_"+str(time_mov_avg_win)+"_SG_"+str(E_SG_win)+"_"+str(E_SG_order)+"_cst_background.jpg",dpi=300)
plt.show()


#%% Save data
data_file = "Cleaned_data_agg_"+str(time_ag_n)+"_win_"+str(time_mov_avg_win)+"_SG_"+str(E_SG_win)+"_"+str(E_SG_order)+"_cst_background"
summary_data_i = summary_data.loc[measurement_i,:]
with open(output_folder+data_file+".pkl", 'wb') as f:
    pickle.dump([filtered_data_energy,clean_data,BE_range,time_range,summary_data_i], f)








