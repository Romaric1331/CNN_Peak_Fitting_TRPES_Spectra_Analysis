# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:38:26 2024

@author: ajulien
"""

#%% Importation of libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import matplotlib as mpl
import pickle


#%% Define backgournds
def remove_constant_background(data,background_begin,background_end):
    data_numpy = data.to_numpy()
    background_value = np.nanmean(data.loc[background_begin:background_end])

    clean_data = data_numpy-background_value
    clean_data[clean_data<0] = 0
    return clean_data


#%% Inputs
data_folder = "C:/Users/ajulien/Documents/General_modeling_data/XPS_ht_data/TR_PES_fits_AJ/"
measurement_i = 9

# Spectrum to plot
i_to_plot = 800
peak_zoom = (450,750)

mpl.rcParams['lines.linewidth'] = 2.5
mpl.rcParams['lines.markersize'] = 11
mpl.rcParams['font.size'] = 15


#%% Import Excel summary file
summary_data = pd.read_excel(data_folder+"Available_data_summary.xlsx",index_col=0)

# Identification of experiment
session_i = summary_data.loc[measurement_i,"Measurement Session"]
sample_i = summary_data.loc[measurement_i,"Sample"]
file_i = summary_data.loc[measurement_i,"File"]

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


#%% Load filtered data
filtered_data_file = data_folder+session_i+"/"+sample_i+"/"+file_i+"_agg_"+str(time_ag_n)+"_win_"+str(time_mov_avg_win)+"_SG_"+str(E_SG_win)+"_"+str(E_SG_order)
with open(filtered_data_file+".pkl", 'rb') as f:
    filtered_data,summary_data_i = pickle.load(f)


#%% Remove constant background
clean_data = filtered_data.apply(remove_constant_background, axis=1, args=(background_begin,background_end), result_type='broadcast',raw=False)


#%% Save data
clean_data_file = filtered_data_file+"_background"

clean_data.to_csv(clean_data_file+".txt", sep='\t', header=False, index=False)

summary_data_i = summary_data.loc[measurement_i,:]
with open(clean_data_file+".pkl", 'wb') as f:
    pickle.dump([clean_data,summary_data_i], f)


#%% Plot example of spectrum
i_to_plot_ag = int(round((i_to_plot + time_ag_n/2)/time_ag_n))

plt.figure(figsize=(10,10))
plt.plot(filtered_data.loc[i_to_plot_ag,:],'+:')
plt.plot(clean_data.loc[i_to_plot_ag,:],'+:')
#plt.xlim((300,600))
#plt.ylim((0,0.7))
plt.xlabel("Energy")
plt.ylabel("Counts")
plt.title("Spectrum at "+str(i_to_plot)+"th measurement")
#plt.savefig(output_folder+"Spectrum_"+str(i_to_plot)+"_comparison_zoom_"+str(n_sum)+"_"+str(win_size_ag)+"_"+str(win_size_E)+"_"+str(poly_order_E)+".jpg",dpi=300)
plt.show()















