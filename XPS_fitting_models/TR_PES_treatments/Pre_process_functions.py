# -*- coding: utf-8 -*-
"""
Created on Thu May 23 08:56:57 2024

@author: ajulien
"""


#%% Importation of libraries
import numpy as np
import scipy


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

def define_linear_background_Pb_4f(data,BE_range,Pb_4f_72_range,Pb_4f_52_range):
    spectrum = data.to_numpy()
    
    Pb_4f_72_begin_i = np.argmin(abs(BE_range-Pb_4f_72_range[1]))
    Pb_4f_72_end_i = np.argmin(abs(BE_range-Pb_4f_72_range[0]))
    
    Pb_4f_52_begin_i = np.argmin(abs(BE_range-Pb_4f_52_range[1]))
    Pb_4f_52_end_i = np.argmin(abs(BE_range-Pb_4f_52_range[0]))
    
    if Pb_4f_72_end_i == Pb_4f_52_begin_i:
        Pb_4f_72_begin_i = Pb_4f_72_begin_i+1
    
    lin_background = BE_range*0
    lin_background[0:Pb_4f_52_begin_i] = spectrum[0:Pb_4f_52_begin_i]
    lin_background[Pb_4f_52_begin_i:Pb_4f_52_end_i] = np.linspace(spectrum[Pb_4f_52_begin_i],spectrum[Pb_4f_52_end_i],Pb_4f_52_end_i-Pb_4f_52_begin_i)
    lin_background[Pb_4f_52_end_i:Pb_4f_72_begin_i] = spectrum[Pb_4f_52_end_i:Pb_4f_72_begin_i]
    lin_background[Pb_4f_72_begin_i:Pb_4f_72_end_i] = np.linspace(spectrum[Pb_4f_72_begin_i],spectrum[Pb_4f_72_end_i],Pb_4f_72_end_i-Pb_4f_72_begin_i)
    lin_background[Pb_4f_72_end_i:] = spectrum[Pb_4f_72_end_i:]

    return lin_background

def define_linear_background_I_3d(data,BE_range,I_3d_52_range):
    spectrum = data.to_numpy()
    
    I_3d_52_begin_i = np.argmin(abs(BE_range-I_3d_52_range[1]))
    I_3d_52_end_i = np.argmin(abs(BE_range-I_3d_52_range[0]))
    
    lin_background = BE_range*0
    lin_background[0:I_3d_52_begin_i] = spectrum[0:I_3d_52_begin_i]
    lin_background[I_3d_52_begin_i:I_3d_52_end_i] = np.linspace(spectrum[I_3d_52_begin_i],spectrum[I_3d_52_end_i],I_3d_52_end_i-I_3d_52_begin_i)
    lin_background[I_3d_52_end_i:] = spectrum[I_3d_52_end_i:]

    return lin_background


#%% Find position of a peak through polynomial fit:
# Returns KE index at max
def find_peak_max_KE_index(data,peak_range):
    raw_indices = data.index.values
    raw_spectrum = data.to_numpy()
    peak_range_i = np.array(raw_indices > peak_range[0]) & np.array(raw_indices < peak_range[1])
    
    max_10_i = np.argsort(-raw_spectrum[peak_range_i])[0:20]
    focused_range = raw_indices[peak_range_i][np.sort(max_10_i)][[0,-1]]
    focused_range_i = np.array(raw_indices > focused_range[0]) & np.array(raw_indices < focused_range[1])
    
    my_poly_coefs = np.polyfit(raw_indices[focused_range_i],raw_spectrum[focused_range_i],3)
    my_poly = np.poly1d(my_poly_coefs)
    fitted_spectrum = my_poly(raw_indices[focused_range_i])
    max_i = np.argmax(fitted_spectrum)
    max_i_raw = np.arange(focused_range[0],focused_range[1],1)[max_i]
    return max_i_raw

# Returns BE value at max
def find_peak_max_BE_value(data,peak_range,BE_range):
    spectrum = data.to_numpy()
    
    peak_range_i = np.array(BE_range > peak_range[0]) & np.array(BE_range < peak_range[1])
    BE_range_peak = BE_range[peak_range_i]
    spectrum_peak = spectrum[peak_range_i]
    
    max_10_i = np.argsort(-spectrum_peak)[0:20]
    focused_BE_range = BE_range_peak[max_10_i]
    focused_spectrum = spectrum_peak[max_10_i]
    
    sort_i = np.argsort(focused_BE_range)
    focused_BE_range = focused_BE_range[sort_i]
    focused_spectrum = focused_spectrum[sort_i]
    
    my_poly_coefs = np.polyfit(focused_BE_range,focused_spectrum,3)
    my_poly = np.poly1d(my_poly_coefs)
    
    fitted_BE_range = np.linspace(focused_BE_range[0],focused_BE_range[-1],200)
    fitted_spectrum = my_poly(fitted_BE_range)
    
    return fitted_BE_range[np.argmax(fitted_spectrum)]
