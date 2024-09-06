# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:20:17 2024

@author: ajulien & Romaric
"""

#%% Importations
import numpy as np
from scipy.optimize import curve_fit,basinhopping


#%% Sort peaks per underlying area
def sort_sub_peaks(peak_params,energy_range):
    peak_params_sorted = []
    for i in range(len(peak_params)):
        sub_peaks_area = np.array([sum(pseudo_Voigt(energy_range,peak_params[i][j][0],peak_params[i][j][1],peak_params[i][j][2])) for j in range(len(peak_params[i]))])
        sub_peaks_order = np.argsort(sub_peaks_area)[::-1]
        peak_params_sorted.append([np.reshape(np.array(np.abs(peak_params[i][k][0:3])),(3,1)) for k in sub_peaks_order])
    return peak_params_sorted


#%% R2 eror
def compute_R2(true_spectrum,fitted_spectrum):
    SST = np.sum((true_spectrum-np.mean(true_spectrum))**2)
    SSE = np.sum((true_spectrum-fitted_spectrum)**2)
    return 1-SSE/SST

    
#%% Pseudo-Voigt profiles
def pseudo_Voigt(x,a,b,c):
    beta = 5.09791537e-01
    gamma = 4.41140472e-01
    y = c * ((0.7 * np.exp(-np.log(2) * (x-a)** 2 / (beta*b)** 2))
        + (0.3 / (1 + (x-a)** 2 / (gamma*b)** 2)))
    return y

def pseudo_Voigt_2(x,a1,b1,c1,a2,b2,c2):
    y1 = pseudo_Voigt(x,a1,b1,c1)
    y2 = pseudo_Voigt(x,a2,b2,c2)
    return y1+y2

def pseudo_Voigt_3(x,a1,b1,c1,a2,b2,c2,a3,b3,c3):
    y1 = pseudo_Voigt(x,a1,b1,c1)
    y2 = pseudo_Voigt(x,a2,b2,c2)
    y3 = pseudo_Voigt(x,a3,b3,c3)
    return y1+y2+y3

def pseudo_Voigt_4(x,a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4):
    y1 = pseudo_Voigt(x,a1,b1,c1)
    y2 = pseudo_Voigt(x,a2,b2,c2)
    y3 = pseudo_Voigt(x,a3,b3,c3)
    y4 = pseudo_Voigt(x,a4,b4,c4)
    return y1+y2+y3+y4

def pseudo_Voigt_5(x,a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5):
    y1 = pseudo_Voigt(x,a1,b1,c1)
    y2 = pseudo_Voigt(x,a2,b2,c2)
    y3 = pseudo_Voigt(x,a3,b3,c3)
    y4 = pseudo_Voigt(x,a4,b4,c4)
    y5 = pseudo_Voigt(x,a5,b5,c5)
    return y1+y2+y3+y4+y5


#%% Number of iterations and sub-peaks is based on first fit
def recursive_fit_A(model,spectrum,energy_range):
    # model: keras model
    # spectrum: numpy array (energy_range_n x 1) for spectrum to fit
    # energy_range: numpy array (energy_range_n x 1) for associated energy range
    
    energy_range_n = spectrum.shape[0]
    spectrum = spectrum.reshape(1,energy_range_n,1)
    energy_range = energy_range.reshape(1,energy_range_n,1)
    
    # First fit to extract first peak
    first_fit_0 = model.predict(spectrum,verbose=0)
    first_fit = np.concatenate(first_fit_0,0)
    fitted_params = [first_fit]
    peak_n_fit = int(np.round(first_fit[3,0]))
    
    # Substract first peak from spectrum
    first_correction = spectrum - pseudo_Voigt(energy_range,first_fit[0],first_fit[1],first_fit[2])
    # Avoid negative values
    first_correction[first_correction<0] = 0


    # Second fit to extract second peak
    if peak_n_fit > 1:
        second_fit_0 = model.predict(first_correction,verbose=0)
        second_fit = np.concatenate(second_fit_0,0)
        fitted_params.append(second_fit)
        
        # Substract second peak from remaining spectrum
        second_correction = first_correction - pseudo_Voigt(energy_range,second_fit[0],second_fit[1],second_fit[2])
        # Avoid negative values
        second_correction[second_correction<0] = 0


    # Third fit to extract third peak
    if peak_n_fit > 2:
        third_fit_0 = model.predict(second_correction,verbose=0)
        third_fit = np.concatenate(third_fit_0,0)
        fitted_params.append(third_fit)
        
        # Substract second peak from remaining spectrum
        third_correction = second_correction - pseudo_Voigt(energy_range,third_fit[0],third_fit[1],third_fit[2])
        # Avoid negative values
        third_correction[third_correction<0] = 0


    # Fourth fit to extract fourth peak
    if peak_n_fit > 3:
        fourth_fit_0 = model.predict(third_correction,verbose=0)
        fourth_fit = np.concatenate(fourth_fit_0,0)
        fitted_params.append(fourth_fit)
        
        # Substract second peak from remaining spectrum
        fourth_correction = third_correction - pseudo_Voigt(energy_range,fourth_fit[0],fourth_fit[1],fourth_fit[2])
        # Avoid negative values
        fourth_correction[fourth_correction<0] = 0
    

    # Fifth fit to extract fifth peak
    if peak_n_fit > 4:
        fifth_fit_0 = model.predict(fourth_correction,verbose=0)
        fifth_fit = np.concatenate(fifth_fit_0,0)
        fitted_params.append(fifth_fit)
        
        # Substract second peak from remaining spectrum
        fifth_correction = fourth_correction - pseudo_Voigt(energy_range,fifth_fit[0],fifth_fit[1],fifth_fit[2])
        # Avoid negative values
        fifth_correction[fifth_correction<0] = 0


    # Fitted total spectrum
    if peak_n_fit < 2:
        fitted_spectrum = pseudo_Voigt(energy_range,first_fit[0],first_fit[1],first_fit[2])
    
    elif peak_n_fit == 2:
        fitted_spectrum = pseudo_Voigt_2(energy_range,
                                         first_fit[0],first_fit[1],first_fit[2],
                                         second_fit[0],second_fit[1],second_fit[2])
    
    elif peak_n_fit == 3:
        fitted_spectrum = pseudo_Voigt_3(energy_range,
                                         first_fit[0],first_fit[1],first_fit[2],
                                         second_fit[0],second_fit[1],second_fit[2],
                                         third_fit[0],third_fit[1],third_fit[2])
    
    elif peak_n_fit == 4:
        fitted_spectrum = pseudo_Voigt_4(energy_range,
                                         first_fit[0],first_fit[1],first_fit[2],
                                         second_fit[0],second_fit[1],second_fit[2],
                                         third_fit[0],third_fit[1],third_fit[2],
                                         fourth_fit[0],fourth_fit[1],fourth_fit[2])
    
    elif peak_n_fit > 4:
        fitted_spectrum = pseudo_Voigt_5(energy_range,
                                         first_fit[0],first_fit[1],first_fit[2],
                                         second_fit[0],second_fit[1],second_fit[2],
                                         third_fit[0],third_fit[1],third_fit[2],
                                         fourth_fit[0],fourth_fit[1],fourth_fit[2],
                                         fifth_fit[0],fifth_fit[1],fifth_fit[2])

    return fitted_spectrum,fitted_params


#%% Number of iterations and sub-peaks is based on first fit + curve_fit polish at each intermediate step
def recursive_fit_A_internal_polish(model,spectrum,energy_range):
    # model: keras model
    # spectrum: numpy array (energy_range_n x 1) for spectrum to fit
    # energy_range: numpy array (energy_range_n x 1) for associated energy range
    
    energy_range_n = spectrum.shape[0]
    spectrum = spectrum.reshape(1,energy_range_n,1)
    energy_range = energy_range.reshape(1,energy_range_n,1)
    
    # First fit to extract first peak
    first_fit_0 = model.predict(spectrum,verbose=0)
    first_fit = np.concatenate(first_fit_0,0)
    
    # Polish first fit
    try:
        E_range_i = np.logical_and(energy_range > first_fit[0]-first_fit[1]*1.2 , energy_range < first_fit[0]+first_fit[1]*1.2)
        first_fit_polish,pcov = curve_fit(pseudo_Voigt,np.squeeze(energy_range[E_range_i]),np.squeeze(spectrum[E_range_i]),
                                       p0=first_fit[0:3,0],x_scale='jac',
                                       method='dogbox',xtol=1e-15)
    except:
        first_fit_polish = first_fit[0:3,0]
    first_fit[0:3,0] = first_fit_polish
    fitted_params = [first_fit]
    peak_n_fit = int(np.round(first_fit[3,0]))
    
    # Substract first peak from spectrum
    first_correction = spectrum - pseudo_Voigt(energy_range,first_fit[0],first_fit[1],first_fit[2])
    # Avoid negative values
    first_correction[first_correction<0] = 0


    # Second fit to extract second peak
    if peak_n_fit > 1:
        second_fit_0 = model.predict(first_correction,verbose=0)
        second_fit = np.concatenate(second_fit_0,0)
        
        # Polish second fit
        try:
            E_range_i = np.logical_and(energy_range > second_fit[0]-second_fit[1]*1.2 , energy_range < second_fit[0]+second_fit[1]*1.2)
            second_fit_polish,pcov = curve_fit(pseudo_Voigt,np.squeeze(energy_range[E_range_i]),np.squeeze(first_correction[E_range_i]),
                                           p0=second_fit[0:3,0],x_scale='jac',
                                           method='dogbox',xtol=1e-10)
        except:
            second_fit_polish = second_fit[0:3,0]
        second_fit[0:3,0] = second_fit_polish
        fitted_params.append(second_fit)
        
        # Substract second peak from remaining spectrum
        second_correction = first_correction - pseudo_Voigt(energy_range,second_fit[0],second_fit[1],second_fit[2])
        # Avoid negative values
        second_correction[second_correction<0] = 0


    # Third fit to extract third peak
    if peak_n_fit > 2:
        third_fit_0 = model.predict(second_correction,verbose=0)
        third_fit = np.concatenate(third_fit_0,0)
        
        # Polish third fit
        try:
            E_range_i = np.logical_and(energy_range > third_fit[0]-third_fit[1]*1.2 , energy_range < third_fit[0]+third_fit[1]*1.2)
            third_fit_polish,pcov = curve_fit(pseudo_Voigt,np.squeeze(energy_range[E_range_i]),np.squeeze(second_correction[E_range_i]),
                                           p0=third_fit[0:3,0],x_scale='jac',
                                           method='dogbox',xtol=1e-5)
        except:
            third_fit_polish = third_fit[0:3,0]
        third_fit[0:3,0] = third_fit_polish
        fitted_params.append(third_fit)
        
        # Substract third peak from remaining spectrum
        third_correction = second_correction - pseudo_Voigt(energy_range,third_fit[0],third_fit[1],third_fit[2])
        # Avoid negative values
        third_correction[third_correction<0] = 0


    # Fourth fit to extract fourth peak
    if peak_n_fit > 3:
        fourth_fit_0 = model.predict(third_correction,verbose=0)
        fourth_fit = np.concatenate(fourth_fit_0,0)
        
        # Polish fourth fit
        try:
            E_range_i = np.logical_and(energy_range > fourth_fit[0]-fourth_fit[1]*1.2 , energy_range < fourth_fit[0]+fourth_fit[1]*1.2)
            fourth_fit_polish,pcov = curve_fit(pseudo_Voigt,np.squeeze(energy_range[E_range_i]),np.squeeze(third_correction[E_range_i]),
                                           p0=fourth_fit[0:3,0],x_scale='jac',
                                           method='dogbox',xtol=1e-2)
        except:
            fourth_fit_polish = fourth_fit[0:3,0]
        fourth_fit[0:3,0] = fourth_fit_polish
        fitted_params.append(fourth_fit)
        
        # Substract fourth peak from remaining spectrum
        fourth_correction = third_correction - pseudo_Voigt(energy_range,fourth_fit[0],fourth_fit[1],fourth_fit[2])
        # Avoid negative values
        fourth_correction[fourth_correction<0] = 0
    

    # Fifth fit to extract fifth peak
    if peak_n_fit > 4:
        fifth_fit_0 = model.predict(fourth_correction,verbose=0)
        fifth_fit = np.concatenate(fifth_fit_0,0)
        
        # Polish fifth fit
        try:
            E_range_i = np.logical_and(energy_range > fifth_fit[0]-fifth_fit[1]*1.2 , energy_range < fifth_fit[0]+fifth_fit[1]*1.2)
            fifth_fit_polish,pcov = curve_fit(pseudo_Voigt,np.squeeze(energy_range[E_range_i]),np.squeeze(fourth_correction[E_range_i]),
                                           p0=fifth_fit[0:3,0],x_scale='jac',
                                           method='dogbox',xtol=1e-1)
        except:
            fifth_fit_polish = fifth_fit[0:3,0]
        fifth_fit[0:3,0] = fifth_fit_polish
        fitted_params.append(fifth_fit)
        
        # Substract fifth peak from remaining spectrum
        fifth_correction = fourth_correction - pseudo_Voigt(energy_range,fifth_fit[0],fifth_fit[1],fifth_fit[2])
        # Avoid negative values
        fifth_correction[fifth_correction<0] = 0


    # Fitted total spectrum
    if peak_n_fit < 2:
        fitted_spectrum = pseudo_Voigt(energy_range,first_fit[0],first_fit[1],first_fit[2])
    
    elif peak_n_fit == 2:
        fitted_spectrum = pseudo_Voigt_2(energy_range,
                                         first_fit[0],first_fit[1],first_fit[2],
                                         second_fit[0],second_fit[1],second_fit[2])
    
    elif peak_n_fit == 3:
        fitted_spectrum = pseudo_Voigt_3(energy_range,
                                         first_fit[0],first_fit[1],first_fit[2],
                                         second_fit[0],second_fit[1],second_fit[2],
                                         third_fit[0],third_fit[1],third_fit[2])
    
    elif peak_n_fit == 4:
        fitted_spectrum = pseudo_Voigt_4(energy_range,
                                         first_fit[0],first_fit[1],first_fit[2],
                                         second_fit[0],second_fit[1],second_fit[2],
                                         third_fit[0],third_fit[1],third_fit[2],
                                         fourth_fit[0],fourth_fit[1],fourth_fit[2])
    
    elif peak_n_fit > 4:
        fitted_spectrum = pseudo_Voigt_5(energy_range,
                                         first_fit[0],first_fit[1],first_fit[2],
                                         second_fit[0],second_fit[1],second_fit[2],
                                         third_fit[0],third_fit[1],third_fit[2],
                                         fourth_fit[0],fourth_fit[1],fourth_fit[2],
                                         fifth_fit[0],fifth_fit[1],fifth_fit[2])

    return fitted_spectrum,fitted_params


#%% Number of iterations and sub-peaks is based on max value of remainig spectrum
def recursive_fit_B(model,spectrum,energy_range):
    # model: keras model
    # spectrum: numpy array (energy_range_n x 1) for spectrum to fit
    # energy_range: numpy array (energy_range_n x 1) for associated energy range
    
    energy_range_n = spectrum.shape[0]
    spectrum = spectrum.reshape(1,energy_range_n,1)
    energy_range = energy_range.reshape(1,energy_range_n,1)
    
    second_correction = np.zeros((1,energy_range_n,1))*np.nan
    third_correction = np.zeros((1,energy_range_n,1))*np.nan
    fourth_correction = np.zeros((1,energy_range_n,1))*np.nan
    fifth_correction = np.zeros((1,energy_range_n,1))*np.nan
    
    peak_n_fit = 1
    
    # First fit to extract first peak
    first_fit_0 = model.predict(spectrum,verbose=0)
    first_fit = np.concatenate(first_fit_0,0)
    fitted_params = [first_fit]

    # Substract first peak from spectrum
    first_correction = spectrum - pseudo_Voigt(energy_range,first_fit[0],first_fit[1],first_fit[2])
    # Avoid negative values
    first_correction[first_correction<0] = 0
    
    if np.max(first_correction) > 0.1:
        peak_n_fit = 2
        
        # Second fit to extract second fit
        second_fit_0 = model.predict(first_correction,verbose=0)
        second_fit = np.concatenate(second_fit_0,0)
        fitted_params.append(second_fit)
        
        # Substract second peak from remaining spectrum
        second_correction = first_correction - pseudo_Voigt(energy_range,second_fit[0],second_fit[1],second_fit[2])
        # Avoid negative values
        second_correction[second_correction<0] = 0
    
    if np.max(second_correction) > 0.1:
        peak_n_fit = 3
        
        # Third fit to extract second fit
        third_fit_0 = model.predict(second_correction,verbose=0)
        third_fit = np.concatenate(third_fit_0,0)
        fitted_params.append(third_fit)
        
        # Substract second peak from remaining spectrum
        third_correction = second_correction - pseudo_Voigt(energy_range,third_fit[0],third_fit[1],third_fit[2])
        # Avoid negative values
        third_correction[third_correction<0] = 0
        
    if np.max(third_correction) > 0.1:
        peak_n_fit = 4
        
        fourth_fit_0 = model.predict(third_correction,verbose=0)
        fourth_fit = np.concatenate(fourth_fit_0,0)
        fitted_params.append(fourth_fit)
        
        # Substract second peak from remaining spectrum
        fourth_correction = third_correction - pseudo_Voigt(energy_range,fourth_fit[0],fourth_fit[1],fourth_fit[2])
        # Avoid negative values
        fourth_correction[fourth_correction<0] = 0

    if np.max(fourth_correction) > 0.1:
        peak_n_fit = 5
        
        fifth_fit_0 = model.predict(fourth_correction,verbose=0)
        fifth_fit = np.concatenate(fifth_fit_0,0)
        fitted_params.append(fifth_fit)
        
        # Substract second peak from remaining spectrum
        fifth_correction = fourth_correction - pseudo_Voigt(energy_range,fifth_fit[0],fifth_fit[1],fifth_fit[2])
        # Avoid negative values
        fifth_correction[fifth_correction<0] = 0
    
    # Fitted total spectrum
    if peak_n_fit < 2:
        fitted_spectrum = pseudo_Voigt(energy_range,first_fit[0],first_fit[1],first_fit[2])
    
    elif peak_n_fit == 2:
        fitted_spectrum = pseudo_Voigt_2(energy_range,
                                         first_fit[0],first_fit[1],first_fit[2],
                                         second_fit[0],second_fit[1],second_fit[2])
    
    elif peak_n_fit == 3:
        fitted_spectrum = pseudo_Voigt_3(energy_range,
                                         first_fit[0],first_fit[1],first_fit[2],
                                         second_fit[0],second_fit[1],second_fit[2],
                                         third_fit[0],third_fit[1],third_fit[2])
    
    elif peak_n_fit == 4:
        fitted_spectrum = pseudo_Voigt_4(energy_range,
                                         first_fit[0],first_fit[1],first_fit[2],
                                         second_fit[0],second_fit[1],second_fit[2],
                                         third_fit[0],third_fit[1],third_fit[2],
                                         fourth_fit[0],fourth_fit[1],fourth_fit[2])
    
    elif peak_n_fit > 4:
        fitted_spectrum = pseudo_Voigt_5(energy_range,
                                         first_fit[0],first_fit[1],first_fit[2],
                                         second_fit[0],second_fit[1],second_fit[2],
                                         third_fit[0],third_fit[1],third_fit[2],
                                         fourth_fit[0],fourth_fit[1],fourth_fit[2],
                                         fifth_fit[0],fifth_fit[1],fifth_fit[2])

    return fitted_spectrum,fitted_params


#%% Number of iterations and sub-peaks is based on max value of remainig spectrum + curve_fit polish at each intermediate step
def recursive_fit_B_internal_polish(model,spectrum,energy_range):
    # model: keras model
    # spectrum: numpy array (energy_range_n x 1) for spectrum to fit
    # energy_range: numpy array (energy_range_n x 1) for associated energy range
    
    energy_range_n = spectrum.shape[0]
    spectrum = spectrum.reshape(1,energy_range_n,1)
    energy_range = energy_range.reshape(1,energy_range_n,1)
    
    second_correction = np.zeros((1,energy_range_n,1))*np.nan
    third_correction = np.zeros((1,energy_range_n,1))*np.nan
    fourth_correction = np.zeros((1,energy_range_n,1))*np.nan
    fifth_correction = np.zeros((1,energy_range_n,1))*np.nan
    
    peak_n_fit = 1
    
    # First fit to extract first peak
    first_fit_0 = model.predict(spectrum,verbose=0)
    first_fit = np.concatenate(first_fit_0,0)
    
    # Polish first fit
    try:
        E_range_i = np.logical_and(energy_range > first_fit[0]-first_fit[1]*1.2 , energy_range < first_fit[0]+first_fit[1]*1.2)
        first_fit_polish,pcov = curve_fit(pseudo_Voigt,np.squeeze(energy_range[E_range_i]),np.squeeze(spectrum[E_range_i]),
                                       p0=first_fit[0:3,0],x_scale='jac',
                                       method='dogbox',xtol=1e-15)
    except:
        first_fit_polish = first_fit[0:3,0]
    first_fit[0:3,0] = first_fit_polish
    fitted_params = [first_fit]

    # Substract first peak from spectrum
    first_correction = spectrum - pseudo_Voigt(energy_range,first_fit[0],first_fit[1],first_fit[2])
    # Avoid negative values
    first_correction[first_correction<0] = 0
    
    if np.max(first_correction) > 0.1:
        peak_n_fit = 2
        
        # Second fit to extract second peak
        second_fit_0 = model.predict(first_correction,verbose=0)
        second_fit = np.concatenate(second_fit_0,0)
        
        # Polish second fit
        try:
            E_range_i = np.logical_and(energy_range > second_fit[0]-second_fit[1]*1.2 , energy_range < second_fit[0]+second_fit[1]*1.2)
            second_fit_polish,pcov = curve_fit(pseudo_Voigt,np.squeeze(energy_range[E_range_i]),np.squeeze(first_correction[E_range_i]),
                                           p0=second_fit[0:3,0],x_scale='jac',
                                           method='dogbox',xtol=1e-10)
        except:
            second_fit_polish = second_fit[0:3,0]
        second_fit[0:3,0] = second_fit_polish
        fitted_params.append(second_fit)
        
        # Substract second peak from remaining spectrum
        second_correction = first_correction - pseudo_Voigt(energy_range,second_fit[0],second_fit[1],second_fit[2])
        # Avoid negative values
        second_correction[second_correction<0] = 0
    
    if np.max(second_correction) > 0.1:
        peak_n_fit = 3
        
        # Third fit to extract third peak
        third_fit_0 = model.predict(second_correction,verbose=0)
        third_fit = np.concatenate(third_fit_0,0)
        
        # Polish third fit
        try:
            E_range_i = np.logical_and(energy_range > third_fit[0]-third_fit[1]*1.2 , energy_range < third_fit[0]+third_fit[1]*1.2)
            third_fit_polish,pcov = curve_fit(pseudo_Voigt,np.squeeze(energy_range[E_range_i]),np.squeeze(second_correction[E_range_i]),
                                           p0=third_fit[0:3,0],x_scale='jac',
                                           method='dogbox',xtol=1e-5)
        except:
            third_fit_polish = third_fit[0:3,0]
        third_fit[0:3,0] = third_fit_polish
        fitted_params.append(third_fit)
        
        # Substract third peak from remaining spectrum
        third_correction = second_correction - pseudo_Voigt(energy_range,third_fit[0],third_fit[1],third_fit[2])
        # Avoid negative values
        third_correction[third_correction<0] = 0
        
    if np.max(third_correction) > 0.1:
        peak_n_fit = 4
        
        # Fourth fit to extract fourth peak
        fourth_fit_0 = model.predict(third_correction,verbose=0)
        fourth_fit = np.concatenate(fourth_fit_0,0)
        
        # Polish fourth fit
        try:
            E_range_i = np.logical_and(energy_range > fourth_fit[0]-fourth_fit[1]*1.2 , energy_range < fourth_fit[0]+fourth_fit[1]*1.2)
            fourth_fit_polish,pcov = curve_fit(pseudo_Voigt,np.squeeze(energy_range[E_range_i]),np.squeeze(third_correction[E_range_i]),
                                           p0=fourth_fit[0:3,0],x_scale='jac',
                                           method='dogbox',xtol=1e-2)
        except:
            fourth_fit_polish = fourth_fit[0:3,0]
        fourth_fit[0:3,0] = fourth_fit_polish
        fitted_params.append(fourth_fit)
        
        # Substract fourth peak from remaining spectrum
        fourth_correction = third_correction - pseudo_Voigt(energy_range,fourth_fit[0],fourth_fit[1],fourth_fit[2])
        # Avoid negative values
        fourth_correction[fourth_correction<0] = 0

    if np.max(fourth_correction) > 0.1:
        peak_n_fit = 5
        
        # Fifth fit to extract fifth peak
        fifth_fit_0 = model.predict(fourth_correction,verbose=0)
        fifth_fit = np.concatenate(fifth_fit_0,0)
        
        # Polish fifth fit
        try:
            E_range_i = np.logical_and(energy_range > fifth_fit[0]-fifth_fit[1]*1.2 , energy_range < fifth_fit[0]+fifth_fit[1]*1.2)
            fifth_fit_polish,pcov = curve_fit(pseudo_Voigt,np.squeeze(energy_range[E_range_i]),np.squeeze(fourth_correction[E_range_i]),
                                           p0=fifth_fit[0:3,0],x_scale='jac',
                                           method='dogbox',xtol=1e-1)
        except:
            fifth_fit_polish = fifth_fit[0:3,0]
        fifth_fit[0:3,0] = fifth_fit_polish
        fitted_params.append(fifth_fit)
        
        # Substract fifth peak from remaining spectrum
        fifth_correction = fourth_correction - pseudo_Voigt(energy_range,fifth_fit[0],fifth_fit[1],fifth_fit[2])
        # Avoid negative values
        fifth_correction[fifth_correction<0] = 0
    
    
    # Fitted total spectrum
    if peak_n_fit < 2:
        fitted_spectrum = pseudo_Voigt(energy_range,first_fit[0],first_fit[1],first_fit[2])
    
    elif peak_n_fit == 2:
        fitted_spectrum = pseudo_Voigt_2(energy_range,
                                         first_fit[0],first_fit[1],first_fit[2],
                                         second_fit[0],second_fit[1],second_fit[2])
    
    elif peak_n_fit == 3:
        fitted_spectrum = pseudo_Voigt_3(energy_range,
                                         first_fit[0],first_fit[1],first_fit[2],
                                         second_fit[0],second_fit[1],second_fit[2],
                                         third_fit[0],third_fit[1],third_fit[2])
    
    elif peak_n_fit == 4:
        fitted_spectrum = pseudo_Voigt_4(energy_range,
                                         first_fit[0],first_fit[1],first_fit[2],
                                         second_fit[0],second_fit[1],second_fit[2],
                                         third_fit[0],third_fit[1],third_fit[2],
                                         fourth_fit[0],fourth_fit[1],fourth_fit[2])
    
    elif peak_n_fit > 4:
        fitted_spectrum = pseudo_Voigt_5(energy_range,
                                         first_fit[0],first_fit[1],first_fit[2],
                                         second_fit[0],second_fit[1],second_fit[2],
                                         third_fit[0],third_fit[1],third_fit[2],
                                         fourth_fit[0],fourth_fit[1],fourth_fit[2],
                                         fifth_fit[0],fifth_fit[1],fifth_fit[2])

    return fitted_spectrum,fitted_params


#%% Polishing functions
# Error function: one sub-peak
def error_func_1(p,x,y_expe):
    (a1,b1,c1) = p
    y_model = pseudo_Voigt(x,a1,b1,c1)
    return np.sum((y_model-y_expe)**2)

# Error function: two sub-peaks
def error_func_2(p,x,y_expe):
    (a1,b1,c1,a2,b2,c2) = p
    y_model = pseudo_Voigt_2(x,a1,b1,c1,a2,b2,c2)
    return np.sum((y_model-y_expe)**2)

# Error function: three sub-peaks
def error_func_3(p,x,y_expe):
    (a1,b1,c1,a2,b2,c2,a3,b3,c3) = p
    y_model = pseudo_Voigt_3(x,a1,b1,c1,a2,b2,c2,a3,b3,c3)
    return np.sum((y_model-y_expe)**2)

# Error function: four sub-peaks
def error_func_4(p,x,y_expe):
    (a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4) = p
    y_model = pseudo_Voigt_4(x,a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4)
    return np.sum((y_model-y_expe)**2)

# Error function: five sub-peaks
def error_func_5(p,x,y_expe):
    (a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5) = p
    y_model = pseudo_Voigt_5(x,a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5)
    return np.sum((y_model-y_expe)**2)


# Fit with basinhopping algorithm: one sub-peak
def polish_basinhopping_1(spectrum, energy_range, params, iterations_n):
    energy_range_n = spectrum.shape[0]
    spectrum = spectrum.reshape(1, energy_range_n, 1)
    energy_range = energy_range.reshape(1, energy_range_n, 1)
    
    x0 = (params[0][0,0], params[0][1,0], params[0][2,0])
    
    # Ensure bounds are always valid
    def create_bounds(value, min_value=0.01):
        lower = max(min_value, value * 0.9)
        upper = max(lower + 0.01, value * 1.1)  # Ensure upper > lower
        return (lower, upper)
    
    bounds = [create_bounds(x0[0]), create_bounds(x0[1]), create_bounds(x0[2])]

    
    optima = basinhopping(error_func_1, x0, niter=iterations_n, T=1, stepsize=0.5,
                          minimizer_kwargs={"args": (energy_range, spectrum),
                                            "bounds": bounds})
    
    fitted_params = [np.reshape(np.array([optima.x[0], optima.x[1], optima.x[2]]), (3,1))]
    fitted_spectrum = pseudo_Voigt(energy_range,
                                   fitted_params[0][0], fitted_params[0][1], fitted_params[0][2])
    return fitted_spectrum, fitted_params

def polish_basinhopping_2(spectrum, energy_range, params, iterations_n):
    # spectrum: numpy array (energy_range_n x 1) for spectrum to fit
    # energy_range: numpy array (energy_range_n x 1) for associated energy range
    
    energy_range_n = spectrum.shape[0]
    spectrum = spectrum.reshape(1, energy_range_n, 1)
    energy_range = energy_range.reshape(1, energy_range_n, 1)
    
    x0 = (params[0][0,0], params[0][1,0], params[0][2,0],
          params[1][0,0], params[1][1,0], params[1][2,0])
    
    # Define bounds ensuring lower bound is always less than upper bound
    bounds = []
    for i in range(6):
        lower = max(0.01, min(x0[i] * 0.9, x0[i] * 1.1))
        upper = max(lower + 0.01, max(x0[i] * 0.9, x0[i] * 1.1))
        bounds.append((lower, upper))
    optima = basinhopping(error_func_2, x0, niter=iterations_n, T=1, stepsize=0.5,
                          minimizer_kwargs={"method": "L-BFGS-B",
                                            "args": (energy_range, spectrum),
                                            "bounds": bounds})

    fitted_params = [np.reshape(np.array([optima.x[0], optima.x[1], optima.x[2]]), (3,1)),
                     np.reshape(np.array([optima.x[3], optima.x[4], optima.x[5]]), (3,1))]
    fitted_spectrum = pseudo_Voigt_2(energy_range,
                                     fitted_params[0][0], fitted_params[0][1], fitted_params[0][2],
                                     fitted_params[1][0], fitted_params[1][1], fitted_params[1][2])
    return fitted_spectrum, fitted_params

# Fit with basinhopping algorithm: three sub-peaks
def polish_basinhopping_3(spectrum,energy_range,params,iterations_n):
    # spectrum: numpy array (energy_range_n x 1) for spectrum to fit
    # energy_range: numpy array (energy_range_n x 1) for associated energy range
    
    energy_range_n = spectrum.shape[0]
    spectrum = spectrum.reshape(1,energy_range_n,1)
    energy_range = energy_range.reshape(1,energy_range_n,1)
    
    x0 = (params[0][0,0], params[0][1,0], params[0][2,0],
          params[1][0,0], params[1][1,0], params[1][2,0],
          params[2][0,0], params[2][1,0], params[2][2,0])
    
    bounds = []
    for i in range(9):
        lower = max(0.01, min(x0[i] * 0.9, x0[i] * 1.1))
        upper = max(lower + 0.01, max(x0[i] * 0.9, x0[i] * 1.1))
        bounds.append((lower, upper))
    optima = basinhopping(error_func_3,x0,niter=iterations_n,T=1,stepsize=0.5,
                          minimizer_kwargs={"args":(energy_range,spectrum),
                                            "bounds":bounds})

    fitted_params = [np.reshape(np.array([optima.x[0],optima.x[1],optima.x[2]]),(3,1)),
                     np.reshape(np.array([optima.x[3],optima.x[4],optima.x[5]]),(3,1)),
                     np.reshape(np.array([optima.x[6],optima.x[7],optima.x[8]]),(3,1))]
    fitted_spectrum = pseudo_Voigt_3(energy_range,
                                     fitted_params[0][0],fitted_params[0][1],fitted_params[0][2],
                                     fitted_params[1][0],fitted_params[1][1],fitted_params[1][2],
                                     fitted_params[2][0],fitted_params[2][1],fitted_params[2][2])
    return fitted_spectrum,fitted_params

# Fit with basinhopping algorithm: four sub-peaks
def polish_basinhopping_4(spectrum,energy_range,params,iterations_n):
    # spectrum: numpy array (energy_range_n x 1) for spectrum to fit
    # energy_range: numpy array (energy_range_n x 1) for associated energy range
    
    energy_range_n = spectrum.shape[0]
    spectrum = spectrum.reshape(1,energy_range_n,1)
    energy_range = energy_range.reshape(1,energy_range_n,1)
    
    x0 = (params[0][0,0], params[0][1,0], params[0][2,0],
          params[1][0,0], params[1][1,0], params[1][2,0],
          params[2][0,0], params[2][1,0], params[2][2,0],
          params[3][0,0], params[3][1,0], params[3][2,0])
    
    bounds = []
    for i in range(12):
        lower = max(0.01, min(x0[i] * 0.9, x0[i] * 1.1))
        upper = max(lower + 0.01, max(x0[i] * 0.9, x0[i] * 1.1))
        bounds.append((lower, upper))
    optima = basinhopping(error_func_4,x0,niter=iterations_n,T=1,stepsize=0.5,
                          minimizer_kwargs={"args":(energy_range,spectrum),
                                            "bounds":bounds})

    fitted_params = [np.reshape(np.array([optima.x[0],optima.x[1],optima.x[2]]),(3,1)),
                     np.reshape(np.array([optima.x[3],optima.x[4],optima.x[5]]),(3,1)),
                     np.reshape(np.array([optima.x[6],optima.x[7],optima.x[8]]),(3,1)),
                     np.reshape(np.array([optima.x[9],optima.x[10],optima.x[11]]),(3,1))]
    fitted_spectrum = pseudo_Voigt_4(energy_range,
                                     fitted_params[0][0],fitted_params[0][1],fitted_params[0][2],
                                     fitted_params[1][0],fitted_params[1][1],fitted_params[1][2],
                                     fitted_params[2][0],fitted_params[2][1],fitted_params[2][2],
                                     fitted_params[3][0],fitted_params[3][1],fitted_params[3][2])
    return fitted_spectrum,fitted_params

# Fit with basinhopping algorithm: five sub-peaks

def polish_basinhopping_5(spectrum, energy_range, params, iterations_n):
    energy_range_n = spectrum.shape[0]
    spectrum = spectrum.reshape(1, energy_range_n, 1)
    energy_range = energy_range.reshape(1, energy_range_n, 1)
    
    x0 = np.array([param[i, 0] for param in params for i in range(3)])
    
    bounds = []
    for i in range(15):
        lower = max(0.01, min(x0[i] * 0.9, x0[i] * 1.1))
        upper = max(lower + 0.01, max(x0[i] * 0.9, x0[i] * 1.1))
        bounds.append((lower, upper))
    
    optima = basinhopping(error_func_5, x0, niter=iterations_n, T=1, stepsize=0.5,
                          minimizer_kwargs={"method": "L-BFGS-B",
                                            "args": (energy_range, spectrum),
                                            "bounds": bounds})

    fitted_params = [np.reshape(optima.x[i:i+3], (3, 1)) for i in range(0, len(optima.x), 3)]
    fitted_spectrum = pseudo_Voigt_5(energy_range, 
                                     fitted_params[0][0, 0], fitted_params[0][1, 0], fitted_params[0][2, 0],
                                     fitted_params[1][0, 0], fitted_params[1][1, 0], fitted_params[1][2, 0],
                                     fitted_params[2][0, 0], fitted_params[2][1, 0], fitted_params[2][2, 0],
                                     fitted_params[3][0, 0], fitted_params[3][1, 0], fitted_params[3][2, 0],
                                     fitted_params[4][0, 0], fitted_params[4][1, 0], fitted_params[4][2, 0])
    return fitted_spectrum, fitted_params

def polish_basinhopping(spectrum,energy_range,fitted_params,fitted_spectrum,R2_min):
    R2 = compute_R2(spectrum,fitted_spectrum)
    do_polish = R2 < R2_min
    
    if do_polish:
        if len(fitted_params) < 2:
            polished_spectrum,polished_params = polish_basinhopping_1(spectrum,energy_range,fitted_params,50)
        
        elif len(fitted_params) == 2:
            polished_spectrum,polished_params = polish_basinhopping_2(spectrum,energy_range,fitted_params,50)
            
        elif len(fitted_params) == 3:
            polished_spectrum,polished_params = polish_basinhopping_3(spectrum,energy_range,fitted_params,50)
            
        elif len(fitted_params) == 4:
            polished_spectrum,polished_params = polish_basinhopping_4(spectrum,energy_range,fitted_params,50)
            
        elif len(fitted_params) > 4:
            polished_spectrum,polished_params = polish_basinhopping_5(spectrum,energy_range,fitted_params,50)
    
    else:
        polished_spectrum = fitted_spectrum
        polished_params = fitted_params
    
    return polished_spectrum,polished_params,do_polish








