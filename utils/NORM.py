# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 17:09:22 2025

@author: Chenl
"""
from timm.models.layers import trunc_normal_
import torch
import scipy.io as sio
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.function import SpectralConv, ACTIVATION
# from utils.lapy import Solver, TriaMesh, TetMesh
# import torch_geometric.nn as gnn

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        
        self.placeholder = args['placeholder']
        n_layers  = args['n_layers']
        input_dim = args['x_dim']
        out_dim   = args['y_dim']
        num_channels = args['num_channels']
        num_lbos     = args['num_lbos']
        # mlp_ratio    = args['mlp_ratio']
        device       = args['device']
        lbo_data = sio.loadmat(args['lbo_path'])['Eigenvectors'][:,:num_lbos]
        if num_lbos > lbo_data.shape[1]:
            raise ValueError("Please check 'num of lbo' !")
        lbo_bases = torch.Tensor(lbo_data).to(device)
        lbo_bases = F.normalize( lbo_bases, p = 2, dim = -1 ) # L2 Norm
        lbo_inver = (lbo_bases.T @ lbo_bases).inverse() @ lbo_bases.T
        print('lbo_bases:', lbo_bases.shape, 'lbo_inver:', lbo_inver.shape)

        self.fc0 = nn.Linear(input_dim, num_channels) 
        self.fc1 = nn.Linear(num_channels, 128)
        self.fc2 = nn.Linear(128, out_dim)
 
        print('Model is ', args['model_type'])
        if args['model_type'] == 'NORM':
            self.blocks = nn.ModuleList([L_layer(num_channels = num_channels, 
                                                 num_modes   = num_lbos, 
                                                 LBO_MATRIX  = lbo_bases, 
                                                 LBO_INVERSE = lbo_inver,
                                                 act = 'gelu',
                                                 ) for _ in range(n_layers)])
        else:
            raise ValueError("Please check 'model_type' !", args['model_type'])
            
        if args['initialize_weights'] == True:
            self.initialize_weights()
        if self.placeholder == True:
            self.placeholder_para = nn.Parameter((1 / (num_channels)) * torch.rand(num_channels, dtype=torch.float))
        
            
    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        if self.placeholder == True:        
            x = self.fc0(x) + self.placeholder_para
        else:
            x = self.fc0(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x

class L_layer(nn.Module):
    def __init__ (self, num_channels, num_modes, LBO_MATRIX, LBO_INVERSE, act = 'gelu'):
        super(L_layer, self).__init__()
        
        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        self.Conv = SpectralConv(num_channels, num_modes)
        self.W    = nn.Conv1d(num_channels, num_channels, 1)
        self.LBO_MATRIX  = LBO_MATRIX
        self.LBO_INVERSE = LBO_INVERSE
        self.act = act()
        # self.SELayer = SELayer(num_channels)      
        # print('W参数有：', count_params(self.W)) 
    def forward(self, x):
        x_spect = self.Conv(x, self.LBO_MATRIX, self.LBO_INVERSE)
        x_w     = (self.W(x.permute(0, 2, 1))).permute(0, 2, 1)
        x = self.act(x_spect + x_w)
        return x  