# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 08:30:08 2024

@author: ajulien
"""

#%% Importation of libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import matplotlib as mpl


#%% Inputs
data_folder = "C:/Users/ajulien/Documents/General_modeling_data/XPS_ht_data/TR_PES/"
#file_name = "IWAN_TRPES_Pb4f_40mW_0001_ATR_00000_00000_"
file_name = "NREL1_TRPES_Pb4f_40mW_0003_ATR_00000_00000_"
#file_name = "NREL3_TRPES_Pb4f_4mW_0001_ATR_00000_00000_"
#file_name = "NREL3_TRPES_Pb4f_40mW_0003_ATR_00000_00000_"
#file_name = "NREL3_TRPES_Pb4f_40mW_0006_ATR_00000_00000_"
#file_name = "NREL4_TRPES_Pb4f_4mW_0004_ATR_00000_00000_"
#file_name = "NREL4_TRPES_Pb4f_40mW_0002_ATR_00000_00000_"

mpl.rcParams['lines.linewidth'] = 2.5
mpl.rcParams['lines.markersize'] = 11
mpl.rcParams['font.size'] = 15

output_folder = data_folder+file_name+"/"

# Number of neighbors to aggeragte
n_sum = 10
n_sum = 20
n_sum = 40

# Window size for smoothing over time
win_size_ag = 10
win_size_ag = 20
win_size_ag = 40

# Parameters for Savitzky-Golay filter
win_size_E = 15
poly_order_E = 5

# Spectrum to plot
i_to_plot = 100


#%% Load TR-PES data
raw_data = pd.read_csv(data_folder+file_name+".txt", delimiter='\t', header=None)


#%% Define filters
# Moving average
def mov_avg_filter(data,win_size):
    window = np.ones(win_size)/win_size
    return scipy.ndimage.convolve(data, window, mode='nearest') 

# Gaussian filter
def gauss_filter(data,sigma):
    return scipy.ndimage.gaussian_filter1d(data, sigma, mode='nearest')


#%% Aggregate spectra
agregate_data = raw_data.groupby(raw_data.index // n_sum).sum()


#%% Apply filters
win_size = win_size_ag*n_sum
sigma_ag = win_size_ag
sigma = sigma_ag*n_sum

# On raw data
mov_avg_data = raw_data.apply(mov_avg_filter, axis=0, args=(win_size,))
gauss_data = raw_data.apply(gauss_filter, axis=0, args=(sigma,))

# On aggregated data
agregate_mov_avg_data = agregate_data.apply(mov_avg_filter, axis=0, args=(win_size_ag,))
agregate_gauss_data = agregate_data.apply(gauss_filter, axis=0, args=(sigma_ag,))


#%% Plot on example of spectrum
i_to_plot_ag = int(round((i_to_plot + n_sum/2)/n_sum))

plt.figure(figsize=(10,10))
plt.plot(raw_data.loc[i_to_plot,:],':',label="Original")
plt.plot(mov_avg_data.loc[i_to_plot,:],':',label="Average (win size="+str(win_size)+")")
plt.plot(gauss_data.loc[i_to_plot,:],':',label="Gaussian (sigma="+str(sigma)+")")
plt.xlim((50,300))
plt.ylim((0,2))
plt.legend()
plt.xlabel("Energy")
plt.ylabel("Counts")
plt.title("Spectrum at "+str(i_to_plot)+"th measurement: direct filtering")
plt.savefig(output_folder+"Spectrum_"+str(i_to_plot)+"_"+str(n_sum)+"_"+str(win_size_ag)+".jpg",dpi=300)

plt.figure(figsize=(10,10))
plt.plot(agregate_data.loc[i_to_plot_ag,:],'--',label="Aggregated ("+str(n_sum)+" neighbours)")
plt.plot(agregate_mov_avg_data.loc[i_to_plot_ag,:],'--',label="Average (win size="+str(win_size_ag)+")")
plt.plot(agregate_gauss_data.loc[i_to_plot_ag,:],'--',label="Gaussian (sigma="+str(sigma_ag)+")")
plt.xlim((50,300))
plt.ylim((0,2*n_sum))
plt.legend()
plt.xlabel("Energy")
plt.ylabel("Counts")
plt.title("Spectrum at "+str(i_to_plot)+"th measurement: filtering after aggregating")
plt.savefig(output_folder+"Spectrum_"+str(i_to_plot)+"_aggregated_"+str(n_sum)+"_"+str(win_size_ag)+".jpg",dpi=300)

plt.figure(figsize=(10,10))
plt.plot(agregate_data.loc[i_to_plot_ag,:]/n_sum,'-',label="Aggregated ("+str(n_sum)+" neighbours)")
plt.plot(mov_avg_data.loc[i_to_plot,:],':',label="Average (win size="+str(win_size)+")",color="tab:orange")
plt.plot(gauss_data.loc[i_to_plot,:],':',label="Gaussian (sigma="+str(sigma)+")",color="tab:green")
plt.plot(agregate_mov_avg_data.loc[i_to_plot_ag,:]/n_sum,'--',label="Agg + avg (win size="+str(win_size_ag)+")",color="tab:orange")
plt.plot(agregate_gauss_data.loc[i_to_plot_ag,:]/n_sum,'--',label="Agg + gauss (win size="+str(sigma_ag)+")",color="tab:green")
plt.xlim((50,300))
#plt.ylim((0,0.7))
plt.legend()
plt.xlabel("Energy")
plt.ylabel("Counts")
plt.title("Spectrum at "+str(i_to_plot)+"th measurement: comparing results")
plt.savefig(output_folder+"Spectrum_"+str(i_to_plot)+"_comparison_"+str(n_sum)+"_"+str(win_size_ag)+".jpg",dpi=300)

plt.figure(figsize=(10,10))
plt.plot(agregate_data.loc[i_to_plot_ag,:]/n_sum,'-',label="Aggregated ("+str(n_sum)+" neighbours)")
plt.plot(mov_avg_data.loc[i_to_plot,:],':',label="Average (win size="+str(win_size)+")",color="tab:orange")
plt.plot(gauss_data.loc[i_to_plot,:],':',label="Gaussian (sigma="+str(sigma)+")",color="tab:green")
plt.plot(agregate_mov_avg_data.loc[i_to_plot_ag,:]/n_sum,'--',label="Agg + avg (win size="+str(win_size_ag)+")",color="tab:orange")
plt.plot(agregate_gauss_data.loc[i_to_plot_ag,:]/n_sum,'--',label="Agg + gauss (win size="+str(sigma_ag)+")",color="tab:green")
plt.xlim((150,250))
#plt.ylim((0,0.7))
plt.legend()
plt.xlabel("Energy")
plt.ylabel("Counts")
plt.title("Spectrum at "+str(i_to_plot)+"th measurement: comparing results")
plt.savefig(output_folder+"Spectrum_"+str(i_to_plot)+"_comparison_zoom_"+str(n_sum)+"_"+str(win_size_ag)+".jpg",dpi=300)


#%% Export data
agregate_mov_avg_data.to_csv(data_folder+file_name+"agg_"+str(n_sum)+"_win_"+str(win_size_ag)+".txt", sep='\t', header=False, index=False)


#%% Savitzky-Golay filter in energy dimension
def savgol_filter_1(data, win_size, poly_order):
    filtered = scipy.signal.savgol_filter(data, win_size, poly_order, mode='nearest').T
    filtered[filtered<0]=0
    return filtered


#%% Apply filter in energy dimension
agregate_mov_avg_SG_data = agregate_mov_avg_data.apply(savgol_filter_1, axis=1, args=(win_size_E,poly_order_E), result_type='broadcast')


#%% Plot on example of spectrum
plt.figure(figsize=(10,10))
plt.plot(agregate_data.loc[i_to_plot_ag,:]/n_sum,'-',label="Aggregated ("+str(n_sum)+" neighbours)")
plt.plot(agregate_mov_avg_data.loc[i_to_plot_ag,:]/n_sum,'--',label="Agg + avg (win size="+str(win_size_ag)+")",color="tab:orange")
plt.plot(agregate_mov_avg_SG_data.loc[i_to_plot_ag,:]/n_sum,'--',label="Agg + avg + SG over E (win size="+str(win_size_E)+" order="+str(poly_order_E)+")",color="tab:red")
plt.xlim((50,300))
#plt.ylim((0,0.7))
plt.legend()
plt.xlabel("Energy")
plt.ylabel("Counts")
plt.title("Spectrum at "+str(i_to_plot)+"th measurement: comparing results")
plt.savefig(output_folder+"Spectrum_"+str(i_to_plot)+"_comparison_"+str(n_sum)+"_"+str(win_size_ag)+"_"+str(win_size_E)+"_"+str(poly_order_E)+".jpg",dpi=300)


plt.figure(figsize=(10,10))
plt.plot(agregate_data.loc[i_to_plot_ag,:]/n_sum,'-',label="Aggregated ("+str(n_sum)+" neighbours)")
plt.plot(agregate_mov_avg_data.loc[i_to_plot_ag,:]/n_sum,'--',label="Agg + avg (win size="+str(win_size_ag)+")",color="tab:orange")
plt.plot(agregate_mov_avg_SG_data.loc[i_to_plot_ag,:]/n_sum,'--',label="Agg + avg + SG over E (win size="+str(win_size_E)+" order="+str(poly_order_E)+")",color="tab:red")
plt.xlim((150,250))
#plt.ylim((0,0.7))
plt.legend()
plt.xlabel("Energy")
plt.ylabel("Counts")
plt.title("Spectrum at "+str(i_to_plot)+"th measurement: comparing results")
plt.savefig(output_folder+"Spectrum_"+str(i_to_plot)+"_comparison_zoom_"+str(n_sum)+"_"+str(win_size_ag)+"_"+str(win_size_E)+"_"+str(poly_order_E)+".jpg",dpi=300)


#%% Export data
agregate_mov_avg_SG_data.to_csv(data_folder+file_name+"agg_"+str(n_sum)+"_win_"+str(win_size_ag)+"_SG_"+str(win_size_E)+"_"+str(poly_order_E)+".txt", sep='\t', header=False, index=False)


#%% PLot 2D
plt.figure(figsize=(10,10))
plt.imshow(agregate_data/n_sum, aspect='auto', vmax=0.5)


plt.figure(figsize=(10,10))
plt.imshow(agregate_mov_avg_SG_data/n_sum, aspect='auto', vmax=0.5)










