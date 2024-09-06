# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:11:15 2024

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



#%% End of inputs
#
#
#


#%% Importation of libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import pickle
import os
import re
from Pre_process_functions import mov_avg_filter,savgol_filter_1,define_linear_background_Pb_4f,define_linear_background_I_3d,find_peak_max_BE_value

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

# Parameters for probed energy range
photon_E = float(summary_data.loc[measurement_i,"Photon energy (eV)"]) # eV
KE_center = float(summary_data.loc[measurement_i,"Kinetic energy center (eV)"]) # eV
E_resolution = float(summary_data.loc[measurement_i,"Energy resolution (eV)"]) # eV

# Time resolution
t_int = int(summary_data.loc[measurement_i,"Time resolution (ns)"])*1e-3 # us

# Energy indices sequence to extract
E_i_begin_end = [int(s) for s in re.findall(r'\d+',summary_data.loc[measurement_i,"Studied energy range (indexes)"])]
E_i_begin = E_i_begin_end[0]
E_i_end = E_i_begin_end[1]

# Number of neighbors to aggeragte
time_ag_n = int(summary_data.loc[measurement_i,"Time agregation number"])

# Window size for smoothing over time
time_mov_avg_win_us = float(summary_data.loc[measurement_i,"Time moving average window (us)"])

# Parameters for Savitzky-Golay filter
E_SG_win_meV = int(summary_data.loc[measurement_i,"Energy Savitzky-Golay window (meV)"]) # meV
E_SG_order = int(summary_data.loc[measurement_i,"Energy Savitzky-Golay order"])

# Background parameters
if core_level == "Pb 4f":
    Pb_4f_72_range = [float(s) for s in re.findall(r'\d*\.*\d+',summary_data.loc[measurement_i,"Pb 4f 7/2 background range (eV)"])] # eV
    Pb_4f_52_range = [float(s) for s in re.findall(r'\d*\.*\d+',summary_data.loc[measurement_i,"Pb 4f 5/2 background range (eV)"])] # eV
    
elif core_level == "I 3d 5/2":
    I_3d_52_range = [float(s) for s in re.findall(r'\d*\.*\d+',summary_data.loc[measurement_i,"I 3d 5/2 background range (eV)"])] # eV


#%% Import raw data file
raw_data_file = data_folder+session_i+"/"+sample_i+"/"+file_i
raw_data = pd.read_csv(raw_data_file+".txt", delimiter='\t', header=None)

if mat_or == "(E,time)":
    raw_data = raw_data.transpose()
elif mat_or != "(time,E)":
    print("Warning: no proper matrix orientation given")

# Remove last specrtra at 0
raw_data_zero = raw_data > 0
raw_data = raw_data.iloc[np.where(raw_data_zero.any(axis=1))[0],:]

# Convert into counts per second: aquisition time is considered equal to time resolution
raw_data = raw_data / (t_int*1e-9)

# Define energy range
E_range_raw_n = np.size(raw_data,1)
BE_begin_raw = photon_E - (KE_center - E_resolution*E_range_raw_n/2)
BE_end_raw = photon_E - (KE_center + E_resolution*E_range_raw_n/2)
BE_range_raw = np.linspace(BE_begin_raw,BE_end_raw,E_range_raw_n) # eV

# Define time range
t_range_n = np.size(raw_data,0)
time_range_raw = np.linspace(0,(t_range_n-1)*t_int,t_range_n) # us

# Define output folder
output_folder = data_folder+session_i+"/"+sample_i+"/"+file_i+\
    "/data_agg_"+str(time_ag_n)+"_win_"+str(time_mov_avg_win_us)+"_SG_"+str(int(E_SG_win_meV))+"_"+str(E_SG_order)+"/"
os.makedirs(output_folder,exist_ok=True)

# Plot raw data
plt.figure(figsize=(10,10))
plt.imshow(raw_data, aspect='auto',interpolation=None)
plt.xlabel("Kinetic energy index")
plt.ylabel("Time index")
plt.title(sample_i+": "+core_level)
plt.tight_layout()
plt.savefig(output_folder+"All_data_raw.jpg",dpi=300)
plt.show()


#%% Aggregate spectra
agregate_data = raw_data.iloc[:,E_i_begin:E_i_end].groupby(raw_data.index // time_ag_n).sum()/time_ag_n
BE_range = BE_range_raw[E_i_begin:E_i_end] # eV
time_range = time_range_raw[0:-1:time_ag_n] # us


#%% Apply filters
# On aggregated data in time dimension
time_mov_avg_win = int(time_mov_avg_win_us / t_int)
filtered_data_time = agregate_data.apply(mov_avg_filter, axis=0, args=(time_mov_avg_win,))

# In energy dimension
E_SG_win = int(E_SG_win_meV / (E_resolution*1e3))
filtered_data_energy = filtered_data_time.apply(savgol_filter_1, axis=1, args=(E_SG_win,E_SG_order), result_type='broadcast')


#%% Remove background
if core_level == "Pb 4f":
    linear_background = filtered_data_energy.apply(define_linear_background_Pb_4f, axis=1, args=(BE_range,Pb_4f_72_range,Pb_4f_52_range), result_type='broadcast',raw=False)
elif core_level == "I 3d 5/2":
    linear_background = filtered_data_energy.apply(define_linear_background_I_3d, axis=1, args=(BE_range,I_3d_52_range,), result_type='broadcast',raw=False)

clean_data = filtered_data_energy - linear_background
clean_data[clean_data<0] = 0


#%% Plot one example of spectrum
i_to_plot_ag = int(round((i_to_plot + time_ag_n/2)/time_ag_n))

plt.figure(figsize=(10,10))
plt.plot(raw_data.loc[i_to_plot_ag,:],'+:',label="Raw data")
plt.plot(agregate_data.loc[i_to_plot_ag,:],'+:',label="Aggregated ("+str(time_ag_n)+" neighbours)")
plt.plot(filtered_data_time.loc[i_to_plot_ag,:],'+:',label="Agg + time avg (win size="+str(time_mov_avg_win_us)+" us)")
plt.plot(filtered_data_energy.loc[i_to_plot_ag,:],'+:',label="Agg + time avg + SG over E (win size="+str(E_SG_win_meV)+" meV order="+str(E_SG_order)+")")
plt.xlabel("Kinetic energy index")
plt.ylabel("Intensity (counts.s-1)")
plt.legend()
plt.title(sample_i+": "+core_level+": spectrum after "+str(time_range_raw[i_to_plot])+" $\mu s$")
plt.tight_layout()
plt.savefig(output_folder+"Spectrum_"+str(i_to_plot)+"_smooth.jpg",dpi=300)
plt.show()

plt.figure(figsize=(10,10))
plt.plot(BE_range,filtered_data_energy.loc[i_to_plot_ag,:],'+:',label="Filtered")
plt.plot(BE_range,linear_background.loc[i_to_plot_ag,:],'+:',label="Linear background")
plt.plot(BE_range,clean_data.loc[i_to_plot_ag,:],'+:',label="Background removed")
plt.xlim(BE_range[[0,-1]])
plt.xlabel("Binding energy (eV)")
plt.ylabel("Intensity (counts.s-1)")
plt.legend()
plt.title(sample_i+": "+core_level+": spectrum after "+str(time_range_raw[i_to_plot])+" $\mu s$")
plt.tight_layout()
plt.savefig(output_folder+"Spectrum_"+str(i_to_plot)+"_background.jpg",dpi=300)
plt.show()

plt.figure(figsize=(10,10))
plt.imshow(clean_data, aspect='auto',extent=[BE_range[0],BE_range[-1],time_range[0],time_range[-1]],interpolation=None)
plt.xlabel("Binding energy (eV)")
plt.ylabel("Time ($\mu s$)")
plt.title(sample_i+": "+core_level)
plt.tight_layout()
plt.savefig(output_folder+"All_data_cleaned.jpg",dpi=300)
plt.show()


#%% Save data
# One pickle file containing all outputs
with open(output_folder+"Pre_processed_data.pkl", 'wb') as f:
    pickle.dump([filtered_data_energy,clean_data,BE_range,time_range], f)

# Four csv files containing one output each
np.savetxt(output_folder+"Filtered_data.txt",filtered_data_energy,delimiter='\t')
np.savetxt(output_folder+"Cleaned_data.txt",clean_data,delimiter='\t')
np.savetxt(output_folder+"Binding_energy_range_eV.txt",np.reshape(BE_range,(1,len(BE_range))),delimiter='\t')
np.savetxt(output_folder+"Time_range_us.txt",time_range,delimiter='\t')


#%% Follow peak position
if core_level == "Pb 4f":
    Pb_4f_72_pos = clean_data.apply(find_peak_max_BE_value, axis=1, args=(Pb_4f_72_range,BE_range))
    Pb_4f_52_pos = clean_data.apply(find_peak_max_BE_value, axis=1, args=(Pb_4f_52_range,BE_range))
    
    plt.figure(figsize=(10,10))
    plt.plot(time_range,Pb_4f_72_pos,'+:',color="tab:blue")
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(time_range,Pb_4f_52_pos,'+:',color="tab:orange")
    ax1.set_xlabel("Time ($\mu s$)")
    ax1.set_ylabel("Pb 4f 7/2 BE (eV)",color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    ax2.set_ylabel("Pb 4f 5/2 BE (eV)",color="tab:orange")
    ax2.tick_params(axis='y', labelcolor="tab:orange")
    plt.title(sample_i+": "+core_level)
    plt.tight_layout()
    plt.savefig(output_folder+"Peak_pos_simple.jpg",dpi=300)
    plt.show()

    plt.figure(figsize=(10,10))
    plt.plot(BE_range,clean_data.loc[i_to_plot_ag,:],'+:')
    plt.axvline(Pb_4f_72_pos[i_to_plot_ag])
    plt.axvline(Pb_4f_52_pos[i_to_plot_ag])
    plt.xlim(BE_range[[0,-1]])
    plt.xlabel("Binding energy (eV)")
    plt.ylabel("Intensity (counts.s-1)")
    plt.title(sample_i+": "+core_level+": spectrum after "+str(time_range_raw[i_to_plot])+" $\mu s$")
    plt.tight_layout()
    plt.savefig(output_folder+"Peak_id_simple_"+str(i_to_plot)+".jpg",dpi=300)
    plt.show()


elif core_level == "I 3d 5/2":
    I_3d_52_pos = clean_data.apply(find_peak_max_BE_value, axis=1, args=(I_3d_52_range,BE_range))
    
    plt.figure(figsize=(10,10))
    plt.plot(time_range,I_3d_52_pos,'+:')
    plt.xlabel("Time ($\mu s$)")
    plt.ylabel("I 3d 5/2 BE (eV)",color="tab:blue")
    plt.title(sample_i+": "+core_level)
    plt.tight_layout()
    plt.savefig(output_folder+"Peak_pos_simple.jpg",dpi=300)
    plt.show()

    plt.figure(figsize=(10,10))
    plt.plot(BE_range,clean_data.loc[i_to_plot_ag,:],'+:')
    plt.axvline(I_3d_52_pos[i_to_plot_ag])
    plt.xlim(BE_range[[0,-1]])
    plt.xlabel("Binding energy (eV)")
    plt.ylabel("Intensity (counts.s-1)")
    plt.title(sample_i+": "+core_level+": spectrum after "+str(time_range_raw[i_to_plot])+" $\mu s$")
    plt.tight_layout()
    plt.savefig(output_folder+"Peak_id_simple_"+str(i_to_plot)+".jpg",dpi=300)
    plt.show()













