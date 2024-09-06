# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:46:54 2024

@author: ajulien
"""

#%% Importation of packages
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import scipy.optimize as sco
import time
import matplotlib.colors as mcolors

colors = list(mcolors.TABLEAU_COLORS)


#%% Inputs
main_data_folder = "C:/Users/rsallustre/Documents/XPS_fitting/"
session_name = "seventh_test_03-06-24"
data_folder = main_data_folder+session_name+"/"

figures_folder = data_folder+"Figures_polishing_step/"
os.makedirs(figures_folder,exist_ok=True)


#%% Definition of pseudo Voigt function
def pseudo_Voigt(a, b, c, x):
    beta = 5.09791537e-01
    gamma = 4.41140472e-01
    y = c * ((0.7 * np.exp(-np.log(2) * (x - a) ** 2 / (beta * b) ** 2))+ (0.3 / (1 + (x - a) ** 2 / (gamma * b) ** 2)))
    return y


#%% Load test database
database_folder = data_folder+"Database/"
with open(database_folder+"Test_database.pkl", 'rb') as f:
    energy_range, peak_label_test, spectra_test, peak_params_test = pickle.load(f)
test_n = len(peak_label_test)
energy_range_n = len(energy_range)

# Store number of sub-peaks in test database
peak_number_test = np.array([len(peak_params_test[i]) for i in range(test_n)])

# Sort peaks per underlying area
peak_params_test_sorted = []
for i in range(test_n):
    sub_peaks_area = np.array([sum(pseudo_Voigt(peak_params_test[i][j][0],peak_params_test[i][j][1],peak_params_test[i][j][2],energy_range)) for j in range(peak_number_test[i])])
    sub_peaks_order = np.argsort(sub_peaks_area)[::-1]
    peak_params_test_sorted.append([np.array(peak_params_test[i][k])[:, np.newaxis] for k in sub_peaks_order])

spectra_test_avg = np.mean(spectra_test,1,keepdims=True)
SST = np.squeeze(np.sum((spectra_test-spectra_test_avg)**2,1))


#%% Sparse_densenet
model_name = "Sparse_densenet"
model_folder = data_folder+model_name+"/"

with open(model_folder+model_name+"_recursive_fit_test.pkl", 'rb') as f:
    energy_range, spectra_sparse_densenet, peak_params_sparse_densenet = pickle.load(f)

# Store number of sub-peaks according to fit
peak_number_sparse_densenet = np.array([len(peak_params_sparse_densenet[i]) for i in range(test_n)])

SSE_sparse_densenet = np.squeeze(np.sum((spectra_test-spectra_sparse_densenet)**2,1))
R2_sparse_densenet = 1-SSE_sparse_densenet/SST


#%% Do polishing step for one example of CNN fit
test_i = 92
spectra_test_i = np.squeeze(spectra_test[test_i])
peak_params_model_i = peak_params_sparse_densenet[test_i]

def y2(p):
    beta = 5.09791537e-01
    gamma = 4.41140472e-01
    (a_1,b_1,c_1,a_2,b_2,c_2,a_3,b_3,c_3,a_4,b_4,c_4,a_5,b_5,c_5) = p
    total_y = sum(((c_1*((0.7*np.exp(-np.log(2) * (energy_range - a_1) ** 2 / (beta * b_1) ** 2))
                    +(0.3 / (1 + (energy_range - a_1) ** 2 / (gamma * b_1) ** 2))))
            +(c_2*((0.7*np.exp(-np.log(2) * (energy_range - a_2) ** 2 / (beta * b_2) ** 2))
                    +(0.3 / (1 + (energy_range - a_2) ** 2 / (gamma * b_2) ** 2))))
            +(c_3*((0.7*np.exp(-np.log(2) * (energy_range - a_3) ** 2 / (beta * b_3) ** 2))
                    +(0.3 / (1 + (energy_range - a_3) ** 2 / (gamma * b_3) ** 2))) )
            +(c_4*((0.7*np.exp(-np.log(2) * (energy_range - a_4) ** 2 / (beta * b_4) ** 2))
                    +(0.3 / (1 + (energy_range - a_4) ** 2 / (gamma * b_4) ** 2))))
            +(c_5*((0.7*np.exp(-np.log(2) * (energy_range - a_5) ** 2 / (beta * b_5) ** 2))
                    +(0.3 / (1 + (energy_range - a_5) ** 2 / (gamma * b_5) ** 2))))
            - spectra_test_i)**2)
    return total_y

x0 = (peak_params_model_i[0][0,0],peak_params_model_i[0][1,0],peak_params_model_i[0][2,0],
      peak_params_model_i[1][0,0],peak_params_model_i[1][1,0],peak_params_model_i[1][2,0],
      peak_params_model_i[2][0,0],peak_params_model_i[2][1,0],peak_params_model_i[2][2,0],
      peak_params_model_i[3][0,0],peak_params_model_i[3][1,0],peak_params_model_i[3][2,0],
      peak_params_model_i[4][0,0],peak_params_model_i[4][1,0],peak_params_model_i[4][2,0])
t0 = time.time()
optima = sco.basinhopping(y2,x0,niter=100,T=1,stepsize=0.5)
elpased_time = time.time()-t0
print("Elapsed time: "+format(elpased_time,".2e")+" s")

peak_params_model_i_basinhopping = [np.reshape(np.array([optima.x[0],optima.x[1],optima.x[2]]),(3,1)),
                                    np.reshape(np.array([optima.x[3],optima.x[4],optima.x[5]]),(3,1)),
                                    np.reshape(np.array([optima.x[6],optima.x[7],optima.x[8]]),(3,1)),
                                    np.reshape(np.array([optima.x[9],optima.x[10],optima.x[11]]),(3,1)),
                                    np.reshape(np.array([optima.x[12],optima.x[13],optima.x[14]]),(3,1))]

#%%
peak_params_test_i = peak_params_test_sorted[test_i]
peaks_number_value = peak_number_sparse_densenet[test_i]

spectra_polish_i = pseudo_Voigt(peak_params_model_i_basinhopping[0][0],peak_params_model_i_basinhopping[0][1],peak_params_model_i_basinhopping[0][2],energy_range)+ \
    pseudo_Voigt(peak_params_model_i_basinhopping[1][0],peak_params_model_i_basinhopping[1][1],peak_params_model_i_basinhopping[1][2],energy_range)+ \
        pseudo_Voigt(peak_params_model_i_basinhopping[2][0],peak_params_model_i_basinhopping[2][1],peak_params_model_i_basinhopping[2][2],energy_range)+ \
            pseudo_Voigt(peak_params_model_i_basinhopping[3][0],peak_params_model_i_basinhopping[3][1],peak_params_model_i_basinhopping[3][2],energy_range)+ \
                pseudo_Voigt(peak_params_model_i_basinhopping[4][0],peak_params_model_i_basinhopping[4][1],peak_params_model_i_basinhopping[4][2],energy_range)

SSE_i = np.squeeze(np.sum((spectra_test_i-spectra_polish_i)**2,0))
R2_i = 1-SSE_i/SST[test_i]

plt.figure(figsize=(10,5))
plt.fill_between(energy_range,spectra_test_i,color='k',edgecolor=None,alpha=0.2,label="Test spectrum")
plt.plot(energy_range,spectra_sparse_densenet[test_i],'k',label="Fit spectrum (R2="+format(R2_sparse_densenet[test_i],".4f")+")")
plt.plot(energy_range,spectra_polish_i,'k--',label="Fit spectrum + polish (R2="+format(R2_i,".4f")+")")
for i in range(peaks_number_value):
    plt.fill_between(energy_range,
                     pseudo_Voigt(peak_params_test_i[i][0],peak_params_test_i[i][1],peak_params_test_i[i][2],energy_range),
                     alpha=0.5,color=colors[i])
for i in range(peaks_number_value):
    plt.plot(energy_range,
             pseudo_Voigt(peak_params_model_i[i][0],peak_params_model_i[i][1],peak_params_model_i[i][2],energy_range),
             color=colors[i])
for i in range(peaks_number_value):
    plt.plot(energy_range,
             pseudo_Voigt(peak_params_model_i_basinhopping[i][0],peak_params_model_i_basinhopping[i][1],peak_params_model_i_basinhopping[i][2],energy_range),
             '--',color=colors[i])
plt.grid()
plt.xlabel("Energy")
plt.ylabel("Intensity")
plt.legend()
plt.title(model_name+" fit with 5 sub-peaks")
plt.tight_layout()
plt.savefig(figures_folder+"Fit_example_"+str(test_i)+"_"+model_name+"_polish.jpg",dpi=300)
plt.show()


header = ";Peak 1;Peak 2;Peak 3;Peak 4;Peak 5"

row_names = np.reshape(np.array(["Position","Width","Amplitude"]),(3,1))
np.savetxt(figures_folder+"Fit_example_"+str(test_i)+"_test.csv",
            np.concatenate([row_names,np.squeeze(np.array(peak_params_test_i)).T],1),header=header,delimiter=';',comments='',fmt='%s')
np.savetxt(figures_folder+"Fit_example_"+str(test_i)+"_"+model_name+"_polish.csv",
            np.concatenate([row_names,np.squeeze(np.array(peak_params_model_i_basinhopping)).T],1),header=header,delimiter=';',comments='',fmt='%s')

row_names = np.reshape(np.array(["Position","Width","Amplitude","Peak number"]),(4,1))
np.savetxt(figures_folder+"Fit_example_"+str(test_i)+"_"+model_name+".csv",
            np.concatenate([row_names,np.squeeze(np.array(peak_params_model_i)).T],1),header=header,delimiter=';',comments='',fmt='%s')






