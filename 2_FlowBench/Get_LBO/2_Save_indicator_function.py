# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:36:36 2025

@author: Chenl
"""

import numpy as np
import scipy.io as sio
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# from function import filed_plot, matrix_plot
def compute_fourier_series(nodes, modes, coeffs, is_sine):
    
    f = np.zeros_like(nodes[:,0])
    n_terms = modes.shape[0]
    
    for i in range(n_terms):
        kx = modes[i, 0]
        ky = modes[i, 1]
        phase = kx * nodes[:,0] + ky * nodes[:,1] 

        if is_sine[i]:
            f += coeffs[i] * np.sin(phase)
        else:
            f += coeffs[i] * np.cos(phase)
    
    return f

def generate_fourier_params(num_modes=5):

    modes_i = np.column_stack([
                                np.random.uniform(-1, 5, num_modes),
                                np.random.uniform(-1, 5, num_modes)
                              ])
    coeffs_i = np.random.uniform(-1, 1, num_modes)
    
    if np.random.rand() > 0.5:
        is_sine_i = np.random.randint(0, 2, num_modes, dtype=bool)
    else:
        if np.random.rand() > 0.5:
            is_sine_i = np.ones(num_modes, dtype=bool)  # 全是正弦
        else:
            is_sine_i = np.zeros(num_modes, dtype=bool) # 全是余弦
    return modes_i, coeffs_i, is_sine_i

if __name__ == '__main__':
    
    n = 1000
    
    Points_A = sio.loadmat('../Data/Geo_A/Nodes_LBO_basis.mat')['Points']
    Points_B = sio.loadmat('../Data/Geo_B/Nodes_LBO_basis.mat')['Points']
    
    Ind_A = []
    Ind_B = []

    for i in range(n):
        modes_i, coeffs_i, is_sine_i = generate_fourier_params(num_modes=6)
        Ind_A.append(compute_fourier_series(Points_A, modes_i/8, coeffs_i, is_sine_i))
        Ind_B.append(compute_fourier_series(Points_B, modes_i/8, coeffs_i, is_sine_i))

    Ind_A = np.array(Ind_A)
    Ind_B = np.array(Ind_B)
    
    sio.savemat('../Data/Geo_A/Indicator_function.mat', {'value': Ind_A})
    sio.savemat('../Data/Geo_B/Indicator_function.mat', {'value': Ind_B})
    
    
    