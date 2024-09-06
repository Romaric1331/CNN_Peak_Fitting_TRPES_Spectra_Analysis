# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:01:11 2024

@author: ajulien
"""

#%% Importation of libraries
import numpy as np
from scipy.optimize import basinhopping


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


#%% Number of iterations and sub-peaks is based on first fit and 
def recursive_fit_expe_spectrum(model,spectrum,energy_range,peak_n_fit=None):
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


#%% Polishing functions
# R2 value between two spectra
def compute_R2(true_spectrum,fitted_spectrum):
    SST = np.sum((true_spectrum-np.mean(true_spectrum))**2)
    SSE = np.sum((true_spectrum-fitted_spectrum)**2)
    return 1-SSE/SST

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
def polish_basinhopping_1(spectrum,energy_range,params,iterations_n):
    # spectrum: numpy array (energy_range_n x 1) for spectrum to fit
    # energy_range: numpy array (energy_range_n x 1) for associated energy range
    
    energy_range_n = spectrum.shape[0]
    spectrum = spectrum.reshape(1,energy_range_n,1)
    energy_range = energy_range.reshape(1,energy_range_n,1)
    
    x0 = (params[0,0],params[0,1],params[0,2])
    bounds = [(x0[0]*0.9,x0[0]*1.1),(x0[1]*0.9,x0[1]*1.1),(x0[2]*0.9,x0[2]*1.1)]
    optima = basinhopping(error_func_1,x0,niter=iterations_n,T=1,stepsize=0.5,
                          minimizer_kwargs={"args":(energy_range,spectrum),
                                            "bounds":bounds})
    
    fitted_params = np.copy(params)
    fitted_params[0,0:3] = np.array([optima.x[0],optima.x[1],optima.x[2]])
    fitted_spectrum = pseudo_Voigt(energy_range,
                                   fitted_params[0,0],fitted_params[0,1],fitted_params[0,2])
    return fitted_spectrum,fitted_params

# Fit with basinhopping algorithm: two sub-peaks
def polish_basinhopping_2(spectrum,energy_range,params,iterations_n):
    # spectrum: numpy array (energy_range_n x 1) for spectrum to fit
    # energy_range: numpy array (energy_range_n x 1) for associated energy range
    
    energy_range_n = spectrum.shape[0]
    spectrum = spectrum.reshape(1,energy_range_n,1)
    energy_range = energy_range.reshape(1,energy_range_n,1)
    
    x0 = (params[0,0],params[0,1],params[0,2],
          params[1,0],params[1,1],params[1,2])
    bounds = [(x0[0]*0.9,x0[0]*1.1),(x0[1]*0.9,x0[1]*1.1),(x0[2]*0.9,x0[2]*1.1),
              (x0[3]*0.9,x0[3]*1.1),(x0[4]*0.9,x0[4]*1.1),(x0[5]*0.9,x0[5]*1.1)]
    optima = basinhopping(error_func_2,x0,niter=iterations_n,T=1,stepsize=0.5,
                          minimizer_kwargs={"args":(energy_range,spectrum),
                                            "bounds":bounds})
    
    fitted_params = np.copy(params)
    fitted_params[0,0:3] = np.array([optima.x[0],optima.x[1],optima.x[2]])
    fitted_params[1,0:3] = np.array([optima.x[3],optima.x[4],optima.x[5]])
    fitted_spectrum = pseudo_Voigt_2(energy_range,
                                     fitted_params[0,0],fitted_params[0,1],fitted_params[0,2],
                                     fitted_params[1,0],fitted_params[1,1],fitted_params[1,2])
    return fitted_spectrum,fitted_params

# Fit with basinhopping algorithm: three sub-peaks
def polish_basinhopping_3(spectrum,energy_range,params,iterations_n):
    # spectrum: numpy array (energy_range_n x 1) for spectrum to fit
    # energy_range: numpy array (energy_range_n x 1) for associated energy range
    
    energy_range_n = spectrum.shape[0]
    spectrum = spectrum.reshape(1,energy_range_n,1)
    energy_range = energy_range.reshape(1,energy_range_n,1)
    
    x0 = (params[0,0],params[0,1],params[0,2],
          params[1,0],params[1,1],params[1,2],
          params[2,0],params[2,1],params[2,2])
    bounds = [(x0[0]*0.9,x0[0]*1.1),(x0[1]*0.9,x0[1]*1.1),(x0[2]*0.9,x0[2]*1.1),
              (x0[3]*0.9,x0[3]*1.1),(x0[4]*0.9,x0[4]*1.1),(x0[5]*0.9,x0[5]*1.1),
              (x0[6]*0.9,x0[6]*1.1),(x0[7]*0.9,x0[7]*1.1),(x0[8]*0.9,x0[8]*1.1)]
    optima = basinhopping(error_func_3,x0,niter=iterations_n,T=1,stepsize=0.5,
                          minimizer_kwargs={"args":(energy_range,spectrum),
                                            "bounds":bounds})
    
    fitted_params = np.copy(params)
    fitted_params[0,0:3] = np.array([optima.x[0],optima.x[1],optima.x[2]])
    fitted_params[1,0:3] = np.array([optima.x[3],optima.x[4],optima.x[5]])
    fitted_params[2,0:3] = np.array([optima.x[6],optima.x[7],optima.x[8]])
    fitted_spectrum = pseudo_Voigt_3(energy_range,
                                     fitted_params[0,0],fitted_params[0,1],fitted_params[0,2],
                                     fitted_params[1,0],fitted_params[1,1],fitted_params[1,2],
                                     fitted_params[2,0],fitted_params[2,1],fitted_params[2,2])
    return fitted_spectrum,fitted_params

# Fit with basinhopping algorithm: four sub-peaks
def polish_basinhopping_4(spectrum,energy_range,params,iterations_n):
    # spectrum: numpy array (energy_range_n x 1) for spectrum to fit
    # energy_range: numpy array (energy_range_n x 1) for associated energy range
    
    energy_range_n = spectrum.shape[0]
    spectrum = spectrum.reshape(1,energy_range_n,1)
    energy_range = energy_range.reshape(1,energy_range_n,1)
    
    x0 = (params[0,0],params[0,1],params[0,2],
          params[1,0],params[1,1],params[1,2],
          params[2,0],params[2,1],params[2,2],
          params[3,0],params[3,1],params[3,2])
    bounds = [(x0[0]*0.9,x0[0]*1.1),(x0[1]*0.9,x0[1]*1.1),(x0[2]*0.9,x0[2]*1.1),
              (x0[3]*0.9,x0[3]*1.1),(x0[4]*0.9,x0[4]*1.1),(x0[5]*0.9,x0[5]*1.1),
              (x0[6]*0.9,x0[6]*1.1),(x0[7]*0.9,x0[7]*1.1),(x0[8]*0.9,x0[8]*1.1),
              (x0[9]*0.9,x0[9]*1.1),(x0[10]*0.9,x0[10]*1.1),(x0[11]*0.9,x0[11]*1.1)]
    optima = basinhopping(error_func_4,x0,niter=iterations_n,T=1,stepsize=0.5,
                          minimizer_kwargs={"args":(energy_range,spectrum),
                                            "bounds":bounds})
    
    fitted_params = np.copy(params)
    fitted_params[0,0:3] = np.array([optima.x[0],optima.x[1],optima.x[2]])
    fitted_params[1,0:3] = np.array([optima.x[3],optima.x[4],optima.x[5]])
    fitted_params[2,0:3] = np.array([optima.x[6],optima.x[7],optima.x[8]])
    fitted_params[3,0:3] = np.array([optima.x[9],optima.x[10],optima.x[11]])
    fitted_spectrum = pseudo_Voigt_4(energy_range,
                                     fitted_params[0,0],fitted_params[0,1],fitted_params[0,2],
                                     fitted_params[1,0],fitted_params[1,1],fitted_params[1,2],
                                     fitted_params[2,0],fitted_params[2,1],fitted_params[2,2],
                                     fitted_params[3,0],fitted_params[3,1],fitted_params[3,2])
    return fitted_spectrum,fitted_params

# Fit with basinhopping algorithm: five sub-peaks
def polish_basinhopping_5(spectrum,energy_range,params,iterations_n):
    # spectrum: numpy array (energy_range_n x 1) for spectrum to fit
    # energy_range: numpy array (energy_range_n x 1) for associated energy range
    
    energy_range_n = spectrum.shape[0]
    spectrum = spectrum.reshape(1,energy_range_n,1)
    energy_range = energy_range.reshape(1,energy_range_n,1)
    
    x0 = (params[0,0],params[0,1],params[0,2],
          params[1,0],params[1,1],params[1,2],
          params[2,0],params[2,1],params[2,2],
          params[3,0],params[3,1],params[3,2],
          params[4,0],params[4,1],params[4,2])
    bounds = [(x0[0]*0.9,x0[0]*1.1),(x0[1]*0.9,x0[1]*1.1),(x0[2]*0.9,x0[2]*1.1),
              (x0[3]*0.9,x0[3]*1.1),(x0[4]*0.9,x0[4]*1.1),(x0[5]*0.9,x0[5]*1.1),
              (x0[6]*0.9,x0[6]*1.1),(x0[7]*0.9,x0[7]*1.1),(x0[8]*0.9,x0[8]*1.1),
              (x0[9]*0.9,x0[9]*1.1),(x0[10]*0.9,x0[10]*1.1),(x0[11]*0.9,x0[11]*1.1),
              (x0[12]*0.9,x0[12]*1.1),(x0[13]*0.9,x0[13]*1.1),(x0[14]*0.9,x0[14]*1.1)]
    optima = basinhopping(error_func_5,x0,niter=iterations_n,T=1,stepsize=0.5,
                          minimizer_kwargs={"args":(energy_range,spectrum),
                                            "bounds":bounds})
    
    fitted_params = np.copy(params)
    fitted_params[0,0:3] = np.array([optima.x[0],optima.x[1],optima.x[2]])
    fitted_params[1,0:3] = np.array([optima.x[3],optima.x[4],optima.x[5]])
    fitted_params[2,0:3] = np.array([optima.x[6],optima.x[7],optima.x[8]])
    fitted_params[3,0:3] = np.array([optima.x[9],optima.x[10],optima.x[11]])
    fitted_params[3,0:3] = np.array([optima.x[12],optima.x[13],optima.x[14]])
    fitted_spectrum = pseudo_Voigt_5(energy_range,
                                     fitted_params[0,0],fitted_params[0,1],fitted_params[0,2],
                                     fitted_params[1,0],fitted_params[1,1],fitted_params[1,2],
                                     fitted_params[2,0],fitted_params[2,1],fitted_params[2,2],
                                     fitted_params[3,0],fitted_params[3,1],fitted_params[3,2],
                                     fitted_params[4,0],fitted_params[4,1],fitted_params[4,2])
    return fitted_spectrum,fitted_params


def polish_basinhopping(spectrum,energy_range,fitted_params,fitted_spectrum,R2_min):
    R2 = compute_R2(spectrum,fitted_spectrum)
    do_polish = R2 < R2_min
    
    if do_polish:
        if fitted_params[0,3] < 2:
            polished_spectrum,polished_params = polish_basinhopping_1(spectrum,energy_range,fitted_params,50)
        
        elif fitted_params[0,3] == 2:
            polished_spectrum,polished_params = polish_basinhopping_2(spectrum,energy_range,fitted_params,50)
            
        elif fitted_params[0,3] == 3:
            polished_spectrum,polished_params = polish_basinhopping_3(spectrum,energy_range,fitted_params,50)
            
        elif fitted_params[0,3] == 4:
            polished_spectrum,polished_params = polish_basinhopping_4(spectrum,energy_range,fitted_params,50)
            
        elif fitted_params[0,3] > 4:
            polished_spectrum,polished_params = polish_basinhopping_5(spectrum,energy_range,fitted_params,50)
    
    else:
        polished_spectrum = fitted_spectrum
        polished_params = fitted_params
    
    return polished_spectrum,polished_params,do_polish

