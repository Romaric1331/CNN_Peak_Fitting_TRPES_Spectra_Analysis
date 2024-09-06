# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:24:46 2024

@author: ajulien
"""


#
#
#
#%% Begin of inputs

# Measurement number
measurement_i = 9

# Spectrum to plot
i_to_plot = 400

# Folder containing all data
data_folder = "C:/Users/rsallustre/Documents/Data_transfer_Arthur_24_05_2024/XPS_ht_data/TR_PES_fits_AJ/"

# Energy difference between Pb 4f 7/2 and 5/2 peaks
Pb_4f_BE_delta = 4.8 # eV


#%% End of inputs
#
#
#


#%% Importation of libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import re
from Pre_process_functions import mov_avg_filter,savgol_filter_1,find_peak_max_KE_index

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
E_i_begin_end = [int(s) for s in re.findall(r'\d+',summary_data.loc[measurement_i,"Studied energy range (indexes)"])]
E_i_begin = E_i_begin_end[0]
E_i_end = E_i_begin_end[1]

# Number of neighbors to aggeragte
time_ag_n = int(summary_data.loc[measurement_i,"Time agregation number"])

# Window size for smoothing over time
time_mov_avg_win = int(summary_data.loc[measurement_i,"Time moving average window (us)"])

# Parameters for Savitzky-Golay filter
E_SG_win = int(summary_data.loc[measurement_i,"Energy Savitzky-Golay window (meV)"])
E_SG_order = int(summary_data.loc[measurement_i,"Energy Savitzky-Golay order"])

# Time resolution
t_int = int(summary_data.loc[measurement_i,"Time resolution (ns)"]) # ns

# Parameters for probed energy range
photon_E = float(summary_data.loc[measurement_i,"Photon energy (eV)"]) # eV
KE_center = float(summary_data.loc[measurement_i,"Kinetic energy center (eV)"]) # eV

# Parameters for Pb 4f 5/2 and 7/2 identification
Pb_4f_72_range = [int(s) for s in re.findall(r'\d+',summary_data.loc[measurement_i,"Pb 4f 7/2 range (KE indexes)"])]
Pb_4f_52_range= [int(s) for s in re.findall(r'\d+',summary_data.loc[measurement_i,"Pb 4f 5/2 range (KE indexes)"])]


#%% Import raw data file
raw_data_file = data_folder+session_i+"/"+sample_i+"/"+file_i
raw_data = pd.read_csv(raw_data_file+".txt", delimiter='\t', header=None)
if mat_or == "(E,time)":
    raw_data = raw_data.transpose()
elif mat_or != "(time,E)":
    print("Warning: no proper matrix orientation given")
    
raw_data_zero = raw_data > 0
raw_data = raw_data.iloc[np.where(raw_data_zero.any(axis=1))[0],:]

time_n_raw = raw_data.shape[0]
E_n_raw = raw_data.shape[1]


#%% Aggregate spectra
agregate_data = raw_data.iloc[:,E_i_begin:E_i_end].groupby(raw_data.index // time_ag_n).sum()


#%% Apply filters
# On aggregated data in time dimension
filtered_data_time = agregate_data.apply(mov_avg_filter, axis=0, args=(time_mov_avg_win,))

# In energy dimension
filtered_data_energy = filtered_data_time.apply(savgol_filter_1, axis=1, args=(E_SG_win,E_SG_order), result_type='broadcast')


#%% Polynomial fit to find max of Pb 4f 5/2 and 7/2 peaks
Pb_4f_72_pos = filtered_data_energy.apply(find_peak_max_KE_index, axis=1, args=(Pb_4f_72_range,))
Pb_4f_52_pos = filtered_data_energy.apply(find_peak_max_KE_index, axis=1, args=(Pb_4f_52_range,))


#%% Energy resolution from Pb 4f peak distance
E_resolution = Pb_4f_BE_delta/np.mean(Pb_4f_72_pos-Pb_4f_52_pos) # eV / point

print("measurement_i: "+str(measurement_i))
print("Energy resolution: "+str(E_resolution)+" eV/point")

BE_begin_raw = photon_E - (KE_center - E_resolution*E_n_raw/2)
BE_end_raw = photon_E - (KE_center + E_resolution*E_n_raw/2)
BE_range_raw = np.linspace(BE_begin_raw,BE_end_raw,E_n_raw)
BE_range = BE_range_raw[E_i_begin:E_i_end]


#%% Plot
i_to_plot_ag = int(round((i_to_plot + time_ag_n/2)/time_ag_n))

plt.figure(figsize=(10,10))
plt.plot(raw_data.loc[i_to_plot_ag,:],'+:',label="Raw data")
plt.plot(agregate_data.loc[i_to_plot_ag,:]/time_ag_n,'+:',label="Aggregated ("+str(time_ag_n)+" neighbours)")
plt.plot(filtered_data_time.loc[i_to_plot_ag,:]/time_ag_n,'+:',label="Agg + time avg (win size="+str(time_mov_avg_win)+")")
plt.plot(filtered_data_energy.loc[i_to_plot_ag,:]/time_ag_n,'+:',label="Agg + time avg + SG over E (win size="+str(E_SG_win)+" order="+str(E_SG_order)+")")
plt.axvline(Pb_4f_72_pos[i_to_plot_ag])
plt.axvline(Pb_4f_52_pos[i_to_plot_ag])
#plt.xlim([300,1100])
plt.xlabel("Kinetic energy index")
plt.ylabel("Counts")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,10))
plt.plot(BE_range_raw,raw_data.loc[i_to_plot_ag,:],'+:',label="Raw")
plt.plot(BE_range,filtered_data_energy.loc[i_to_plot_ag,:]/time_ag_n,'+:',label="Cleaned")
plt.axvline(BE_range_raw[Pb_4f_72_pos[i_to_plot_ag]])
plt.axvline(BE_range_raw[Pb_4f_52_pos[i_to_plot_ag]])
plt.xlim(BE_range_raw[[E_i_begin,E_i_end]])
plt.xlabel("Binding energy")
plt.ylabel("Counts")
plt.legend()
plt.tight_layout()
plt.show()

print("Pb 4f 7/2 peak: "+str(BE_range_raw[Pb_4f_72_pos[i_to_plot_ag]])+" eV")
print("Pb 4f 5/2 peak: "+str(BE_range_raw[Pb_4f_52_pos[i_to_plot_ag]])+" eV")






