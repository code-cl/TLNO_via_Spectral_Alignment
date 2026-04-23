import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import torch
import numpy as np
import scipy.io as sio

def read_data(file_path, n_floders, n_train, n_test):
    
    lbo_data = sio.loadmat(file_path +'Nodes_LBO_basis')
    nodes    = lbo_data['Points']
    
    da = sio.loadmat(file_path + '/data.mat')     
    x_data = da['input']
    y_data = da['output']
    # print(x_data.shape, y_data.shape)
    
    if len(x_data.shape)==2:
        x_data = x_data.reshape(x_data.shape[0],-1,1)
    y_data = y_data + nodes
        
    if n_train+n_test>x_data.shape[0]:
        raise ValueError("Please check 'num of data' !", n_train+n_test, x_data.shape[0])
        
    x_train = x_data[:n_train]
    y_train = y_data[:n_train]
    x_test  = x_data[-n_test:]
    y_test  = y_data[-n_test:]
    # print(x_train.shape, y_train.shape)
    # print(x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test


def read_lbo(file_path, num_lbos, name):
    
    lbo_path = file_path + name
    lbo_file = sio.loadmat(lbo_path)
    lbo_data = lbo_file['Eigenvectors'][:,:num_lbos]
    mass     = lbo_file['Mass' ] 
    
    if num_lbos > lbo_data.shape[1]:
        raise ValueError("Please check 'num of lbo' !")
    lbo_bases = torch.Tensor(lbo_data).cuda()
    mass      = torch.Tensor(mass).cuda()
    lbo_inver = mass @ lbo_bases
    print('lbo_bases:', lbo_bases.shape, 'mass:', mass.shape)
    
    return lbo_bases, lbo_inver