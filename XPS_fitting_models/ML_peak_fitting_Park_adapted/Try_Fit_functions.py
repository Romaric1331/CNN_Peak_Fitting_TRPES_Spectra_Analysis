# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:20:17 2024

@author: ajulien
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
def pseudo_Voigt(x, a, b, c):
    x = np.array(x, dtype=float)  # Ensure x is a numpy array of floats
    a = float(a)
    b = float(b)
    c = float(c)
    
    beta = 5.09791537e-01
    gamma = 4.41140472e-01
    
    y = c * ((0.7 * np.exp(-np.log(2) * (x-a)**2 / (beta*b)**2))
             + (0.3 / (1 + (x-a)**2 / (gamma*b)**2)))
    
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


def compute_spectrum(energy_range,params):
    peak_n = np.size(params[np.isfinite(params[:,0]),0])
    
    # Fitted total spectrum
    if peak_n < 2:
        spectrum = pseudo_Voigt(energy_range,
                                params[0,0],params[0,1],params[0,2])
    
    elif peak_n == 2:
        spectrum = pseudo_Voigt_2(energy_range,
                                  params[0,0],params[0,1],params[0,2],
                                  params[1,0],params[1,1],params[1,2])
    
    elif peak_n == 3:
        spectrum = pseudo_Voigt_3(energy_range,
                                  params[0,0],params[0,1],params[0,2],
                                  params[1,0],params[1,1],params[1,2],
                                  params[2,0],params[2,1],params[2,2])
    
    elif peak_n == 4:
        spectrum = pseudo_Voigt_4(energy_range,
                                  params[0,0],params[0,1],params[0,2],
                                  params[1,0],params[1,1],params[1,2],
                                  params[2,0],params[2,1],params[2,2],
                                  params[3,0],params[3,1],params[3,2])
    
    elif peak_n > 4:
        spectrum = pseudo_Voigt_5(energy_range,
                                  params[0,0],params[0,1],params[0,2],
                                  params[1,0],params[1,1],params[1,2],
                                  params[2,0],params[2,1],params[2,2],
                                  params[3,0],params[3,1],params[3,2],
                                  params[4,0],params[4,1],params[4,2])
        
    return spectrum

#%% Number of iterations and sub-peaks is based on first fit
def recursive_fit_A(model,spectrum,energy_range,peak_n_fit=None):
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
    if peak_n_fit == None:
        peak_n_fit = int(np.round(first_fit[3,0]))
    else:
        first_fit[3,0] = peak_n_fit
    
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


#%% Number of iterations and sub-peaks is based on first fit and 
def recursive_fit_A_no_close_peak(model,spectrum,energy_range,peak_n_fit=None):
    # model: keras model
    # spectrum: numpy array (energy_range_n x 1) for spectrum to fit
    # energy_range: numpy array (energy_range_n x 1) for associated energy range
    
    attenuation = 0.01
    fitted_params = np.zeros((5,4))*np.nan
    
    energy_range_n = spectrum.shape[0]
    spectrum = spectrum.reshape(1,energy_range_n,1)
    energy_range = energy_range.reshape(1,energy_range_n,1)
    
    # First fit to extract first peak
    first_fit_0 = model.predict(spectrum,verbose=0)
    first_fit = np.concatenate(first_fit_0,1)
    
    # Number of peaks to find
    if peak_n_fit == None:
        peak_n_fit = int(np.round(first_fit[0,3]))
    first_fit[0,3] = peak_n_fit
    
    fitted_params[0,:] = first_fit
    
    # Substract first peak from spectrum
    first_correction = spectrum - pseudo_Voigt(energy_range,first_fit[0,0],first_fit[0,1],first_fit[0,2])
    # Avoid negative values
    first_correction[first_correction<0] = 0
    
    # Reduce amplitude around peak
    peak_range = np.array(energy_range > (first_fit[0,0]-first_fit[0,1])) & np.array(energy_range < (first_fit[0,0]+first_fit[0,1]))
    first_correction[peak_range] = first_correction[peak_range]*attenuation
    
    # Second fit to extract second peak
    if peak_n_fit > 1:
        second_fit_0 = model.predict(first_correction,verbose=0)
        second_fit = np.concatenate(second_fit_0,1)
        fitted_params[1,:] = second_fit
        
        # Substract second peak from remaining spectrum
        second_correction = first_correction - pseudo_Voigt(energy_range,second_fit[0,0],second_fit[0,1],second_fit[0,2])
        # Avoid negative values
        second_correction[second_correction<0] = 0
        
        # Reduce amplitude around peak
        peak_range = np.array(energy_range > (second_fit[0,0]-second_fit[0,1])) & np.array(energy_range < (second_fit[0,0]+second_fit[0,1]))
        second_correction[peak_range] = second_correction[peak_range]*attenuation


    # Third fit to extract third peak
    if peak_n_fit > 2:
        third_fit_0 = model.predict(second_correction,verbose=0)
        third_fit = np.concatenate(third_fit_0,1)
        fitted_params[2,:] = third_fit
        
        # Substract second peak from remaining spectrum
        third_correction = second_correction - pseudo_Voigt(energy_range,third_fit[0,0],third_fit[0,1],third_fit[0,2])
        # Avoid negative values
        third_correction[third_correction<0] = 0
        
        # Reduce amplitude around peak
        peak_range = np.array(energy_range > (third_fit[0,0]-third_fit[0,1])) & np.array(energy_range < (third_fit[0,0]+third_fit[0,1]))
        third_correction[peak_range] = third_correction[peak_range]*attenuation


    # Fourth fit to extract fourth peak
    if peak_n_fit > 3:
        fourth_fit_0 = model.predict(third_correction,verbose=0)
        fourth_fit = np.concatenate(fourth_fit_0,1)
        fitted_params[3,:] = fourth_fit
        
        # Substract second peak from remaining spectrum
        fourth_correction = third_correction - pseudo_Voigt(energy_range,fourth_fit[0,0],fourth_fit[0,1],fourth_fit[0,2])
        # Avoid negative values
        fourth_correction[fourth_correction<0] = 0
        
        # Reduce amplitude around peak
        peak_range = np.array(energy_range > (fourth_fit[0,0]-fourth_fit[0,1])) & np.array(energy_range < (fourth_fit[0,0]+fourth_fit[0,1]))
        fourth_correction[peak_range] = fourth_correction[peak_range]*attenuation
    

    # Fifth fit to extract fifth peak
    if peak_n_fit > 4:
        fifth_fit_0 = model.predict(fourth_correction,verbose=0)
        fifth_fit = np.concatenate(fifth_fit_0,1)
        fitted_params[4,:] = fifth_fit
        
        # Substract second peak from remaining spectrum
        fifth_correction = fourth_correction - pseudo_Voigt(energy_range,fifth_fit[0,0],fifth_fit[0,1],fifth_fit[0,2])
        # Avoid negative values
        fifth_correction[fifth_correction<0] = 0
        
        # Reduce amplitude around peak
        peak_range = np.array(energy_range > (fifth_fit[0,0]-fifth_fit[0,1])) & np.array(energy_range < (fifth_fit[0,0]+fifth_fit[0,1]))
        fifth_correction[peak_range] = fifth_correction[peak_range]*attenuation


    # Fitted total spectrum
    if peak_n_fit < 2:
        fitted_spectrum = pseudo_Voigt(energy_range,first_fit[0,0],first_fit[0,1],first_fit[0,2])
    
    elif peak_n_fit == 2:
        fitted_spectrum = pseudo_Voigt_2(energy_range,
                                         first_fit[0,0],first_fit[0,1],first_fit[0,2],
                                         second_fit[0,0],second_fit[0,1],second_fit[0,2])
    
    elif peak_n_fit == 3:
        fitted_spectrum = pseudo_Voigt_3(energy_range,
                                         first_fit[0,0],first_fit[0,1],first_fit[0,2],
                                         second_fit[0,0],second_fit[0,1],second_fit[0,2],
                                         third_fit[0,0],third_fit[0,1],third_fit[0,2])
    
    elif peak_n_fit == 4:
        fitted_spectrum = pseudo_Voigt_4(energy_range,
                                         first_fit[0,0],first_fit[0,1],first_fit[0,2],
                                         second_fit[0,0],second_fit[0,1],second_fit[0,2],
                                         third_fit[0,0],third_fit[0,1],third_fit[0,2],
                                         fourth_fit[0,0],fourth_fit[0,1],fourth_fit[0,2])
    
    elif peak_n_fit > 4:
        fitted_spectrum = pseudo_Voigt_5(energy_range,
                                         first_fit[0,0],first_fit[0,1],first_fit[0,2],
                                         second_fit[0,0],second_fit[0,1],second_fit[0,2],
                                         third_fit[0,0],third_fit[0,1],third_fit[0,2],
                                         fourth_fit[0,0],fourth_fit[0,1],fourth_fit[0,2],
                                         fifth_fit[0,0],fifth_fit[0,1],fifth_fit[0,2])

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



def polish_basinhopping_1(spectrum, energy_range, params, iterations_n):
    energy_range_n = spectrum.shape[0]
    spectrum = spectrum.reshape(1, energy_range_n, 1)
    energy_range = energy_range.reshape(1, energy_range_n, 1)
    
    # Ensure x0 is a 1-dimensional numpy array
    x0 = np.array([params[0][0], params[0][1], params[0][2]]).flatten()
    
    bounds = [(x0[0] * 0.9, x0[0] * 1.1), (x0[1] * 0.9, x0[1] * 1.1), (x0[2] * 0.9, x0[2] * 1.1)]
    
    optima = basinhopping(error_func_1, x0, niter=iterations_n, T=1, stepsize=0.5,
                          minimizer_kwargs={"args": (energy_range.squeeze(), spectrum.squeeze()),
                                            "bounds": bounds})
    
    fitted_params = np.copy(params)
    fitted_params[0][0:3] = optima.x.reshape(3, 1)

    fitted_spectrum = pseudo_Voigt(energy_range.squeeze(), fitted_params[0][0], fitted_params[0][1], fitted_params[0][2])
    
    return fitted_spectrum, fitted_params


# Fit with basinhopping algorithm: two sub-peaks
def polish_basinhopping_2(spectrum, energy_range, params, iterations_n):
    energy_range_n = spectrum.shape[0]
    spectrum = spectrum.reshape(1, energy_range_n, 1)
    energy_range = energy_range.reshape(1, energy_range_n, 1)
    
    # Convert params to a numpy array for easier handling
    params_array = np.array(params)
    
    x0 = np.array([params_array[i][j] for i in range(2) for j in range(3)]).flatten()
    
    bounds = [(x*0.9, x*1.1) for x in x0]
    
    optima = basinhopping(error_func_2, x0, niter=iterations_n, T=1, stepsize=0.5,
                          minimizer_kwargs={"args": (energy_range.squeeze(), spectrum.squeeze()),
                                            "bounds": bounds})
    
    fitted_params = np.copy(params_array)
    for i in range(2):
        fitted_params[i, 0:3] = optima.x[i*3:(i+1)*3].reshape(3, 1)
    
    fitted_spectrum = pseudo_Voigt_2(energy_range.squeeze(),
                                     *fitted_params.flatten()[:6])
    
    return fitted_spectrum, fitted_params.tolist()

# Fit with basinhopping algorithm: three sub-peaks
def polish_basinhopping_3(spectrum, energy_range, params, iterations_n):
    energy_range_n = spectrum.shape[0]
    spectrum = spectrum.reshape(1, energy_range_n, 1)
    energy_range = energy_range.reshape(1, energy_range_n, 1)
    
    # Ensure params is a numpy array
    params = np.array(params)
    
    # Flatten params into a 1D array
    x0 = np.array([params[i][0] for i in range(3) for j in range(3)]).flatten()

    bounds = [(x*0.9, x*1.1) for x in x0]
    
    optima = basinhopping(error_func_3, x0, niter=iterations_n, T=1, stepsize=0.5,
                          minimizer_kwargs={"args": (energy_range.squeeze(), spectrum.squeeze()),
                                            "bounds": bounds})
    
    fitted_params = np.copy(params)
    for i in range(3):
        fitted_params[i][0:3] = optima.x[i*3:(i+1)*3].reshape(3, 1)
    
    fitted_spectrum = pseudo_Voigt_3(energy_range.squeeze(),
                                     *fitted_params.flatten()[:9])
    
    return fitted_spectrum, fitted_params.tolist()




# Fit with basinhopping algorithm: four sub-peaks
def polish_basinhopping_4(spectrum, energy_range, params, iterations_n):
    energy_range_n = spectrum.shape[0]
    spectrum = spectrum.reshape(1, energy_range_n, 1)
    energy_range = energy_range.reshape(1, energy_range_n, 1)
    
    # Convert params to a numpy array for easier handling
    params_array = np.array(params)
    
    x0 = np.array([params_array[i][j] for i in range(4) for j in range(3)]).flatten()
    
    bounds = [(x*0.9, x*1.1) for x in x0]
    
    optima = basinhopping(error_func_4, x0, niter=iterations_n, T=1, stepsize=0.5,
                          minimizer_kwargs={"args": (energy_range.squeeze(), spectrum.squeeze()),
                                            "bounds": bounds})
    
    fitted_params = np.copy(params_array)
    for i in range(4):
        fitted_params[i, 0:3] = optima.x[i*3:(i+1)*3].reshape(3, 1)
    
    fitted_spectrum = pseudo_Voigt_4(energy_range.squeeze(),
                                     *fitted_params.flatten()[:12])
    
    return fitted_spectrum, fitted_params.tolist()


# Fit with basinhopping algorithm: five sub-peaks
def polish_basinhopping_5(spectrum, energy_range, params, iterations_n):
    energy_range_n = spectrum.shape[0]
    spectrum = spectrum.reshape(1, energy_range_n, 1)
    energy_range = energy_range.reshape(1, energy_range_n, 1)
    
    # Convert params to a numpy array for easier handling
    params_array = np.array(params)
    
    x0 = np.array([params_array[i][j] for i in range(5) for j in range(3)]).flatten()
    
    # Ensure bounds are always valid
    bounds = []
    for x in x0:
        if x != 0:
            bounds.append((x*0.9, x*1.1) if x > 0 else (x*1.1, x*0.9))
        else:
            bounds.append((-0.1, 0.1))  # Small range around 0 for zero values
    
    optima = basinhopping(error_func_5, x0, niter=iterations_n, T=1, stepsize=0.5,
                          minimizer_kwargs={"args": (energy_range.squeeze(), spectrum.squeeze()),
                                            "bounds": bounds})
    
    fitted_params = np.copy(params_array)
    for i in range(5):
        fitted_params[i, 0:3] = optima.x[i*3:(i+1)*3].reshape(3, 1)
    
    fitted_spectrum = pseudo_Voigt_5(energy_range.squeeze(),
                                     *fitted_params.flatten()[:15])
    
    return fitted_spectrum, fitted_params.tolist()
  # Return only the original number of peaks

def polish_basinhopping(spectrum, energy_range, fitted_params, fitted_spectrum, R2_min):
    # Ensure consistent shapes
    spectrum = np.array(spectrum).reshape(-1, 1)
    energy_range = np.array(energy_range).reshape(-1, 1)
    fitted_spectrum = np.array(fitted_spectrum).reshape(-1, 1)

    R2 = compute_R2(spectrum, fitted_spectrum)
    do_polish = R2 < R2_min
    if do_polish:
        num_peaks = len(fitted_params)
        if num_peaks == 1:
            polished_spectrum, polished_params = polish_basinhopping_1(spectrum, energy_range, fitted_params, 50)
        elif num_peaks == 2:
            polished_spectrum, polished_params = polish_basinhopping_2(spectrum, energy_range, fitted_params, 50)
        elif num_peaks == 3:
            polished_spectrum, polished_params = polish_basinhopping_3(spectrum, energy_range, fitted_params, 50)
        elif num_peaks == 4:
            polished_spectrum, polished_params = polish_basinhopping_4(spectrum, energy_range, fitted_params, 50)
        elif num_peaks >= 5:
            polished_spectrum, polished_params = polish_basinhopping_5(spectrum, energy_range, fitted_params, 50)
        else:
            print(f"Unexpected number of peaks: {num_peaks}. Skipping polish.")
            polished_spectrum, polished_params = fitted_spectrum, fitted_params
    else:
        polished_spectrum = fitted_spectrum
        polished_params = fitted_params

    # Ensure polished_spectrum has the correct shape
    polished_spectrum = np.array(polished_spectrum).reshape(spectrum.shape)

    return polished_spectrum, polished_params, do_polish













