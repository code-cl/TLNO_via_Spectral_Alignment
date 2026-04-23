# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:36:36 2025

@author: Chenl
"""

# import pyvista as pv
import numpy as np
import scipy.io as sio
import os
import matplotlib.tri as mtri
# import matplotlib.pyplot as plt
# import scipy.sparse as sp
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# import pyvista as pv
# import numpy as np
# import matplotlib.colors as colors
# import matplotlib.ticker as ticker

from function import filed_plot, matrix_plot

if __name__ == '__main__':
    
    reference_shape = 'Petals4_data'
    coupling_shape  = 'Petals1_data'
    # diretory_result = 'Ref_' + reference_shape + '_Cou_' + coupling_shape
    # if not os.path.exists(diretory_result):
    #     os.makedirs(diretory_result)
        
    k = 64
    n = 1000
    
    # 读取参考几何信息
    lbo_data  = sio.loadmat('../Data/' + reference_shape + '/Nodes_LBO_basis.mat')
    point_gr  = lbo_data['Points']
    elem_gr   = lbo_data['Elements']
    lbo_gr    = lbo_data['Eigenvectors'][:,:k]
    evals_gr  = lbo_data['Eigenvalues'][0,:k]
    # W_gr      = lbo_data['Stiffness']
    A_gr      = lbo_data['Mass']
    # triangle_gr = mtri.Triangulation(point_gr[:,0], point_gr[:,1], elem_gr)
    
    # 读取需要耦合的几何信息
    lbo_data  = sio.loadmat('../Data/' + coupling_shape + '/Nodes_LBO_basis.mat')
    point_gc  = lbo_data['Points']
    elem_gc   = lbo_data['Elements']
    lbo_gc    = lbo_data['Eigenvectors'][:,:k]
    evals_gc  = lbo_data['Eigenvalues'][0,:k]
    # W_gc      = lbo_data['Stiffness']
    A_gc      = lbo_data['Mass']
    # triangle_gc = mtri.Triangulation(point_gc[:,0], point_gc[:,1], elem_gc)
    
    
    Ind_gr = sio.loadmat('../Data/' + reference_shape + '/Indicator_function.mat')["value"][:n].T
    Ind_gc = sio.loadmat('../Data/' + coupling_shape  + '/Indicator_function.mat')["value"][:n].T
    

    M = (Ind_gc.T @ A_gc @ lbo_gc).T @ (Ind_gr.T @ A_gr @ lbo_gr)
    S, sigma, Rt = np.linalg.svd(M, full_matrices=True)
    C = S @ Rt
    # sio.savemat(diretory_result + '/Diag_C.mat', {'C'  : C,
    #                                               'S' : S,
    #                                               'R' : Rt.T})
    
    lbo_diag = lbo_gc @ C
    sio.savemat('../Data/' + reference_shape + '/Coupled_basis_function.mat', {'Points': point_gr,
                                                                            'Elements': elem_gr,
                                                                            'Eigenvectors': lbo_gr,
                                                                            'Eigenvalues': evals_gr,
                                                                            'Mass': A_gr
                                                                            })
    
    sio.savemat('../Data/' + coupling_shape + '/Coupled_basis_function.mat', {'Points': point_gc,
                                                                            'Elements': elem_gc,
                                                                            'Eigenvectors': lbo_diag,
                                                                            'Eigenvalues': evals_gc,
                                                                            'Mass': A_gc})
    
    
    
    
    
    
    