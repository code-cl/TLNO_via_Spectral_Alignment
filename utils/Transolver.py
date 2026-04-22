# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 17:09:22 2025

@author: Chenl
"""
from timm.models.layers import trunc_normal_
import torch
# import scipy.io as sio
import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
from utils.function import MLP, Physics_Attention_Irregular_Mesh
                            

class Model(nn.Module):
    def __init__(self, 
                 args, 
                 dropout=0.0,
                 act='gelu',
                 mlp_ratio=1):
        super(Model, self).__init__()
        print('Model is Transolver!')
        self.placeholder = args['placeholder']
        n_layers  = args['n_layers']
        input_dim = args['x_dim']
        out_dim   = args['y_dim']
        num_channels = args['num_channels']
        num_heads    = args['num_heads']
        num_slices    = args['num_slices']

        # self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)  
        self.preprocess = MLP(input_dim, num_channels * 2, num_channels, n_layers=0, res=False, act=act)
        self.ln  = nn.LayerNorm(num_channels)
        self.mlp = nn.Linear(num_channels, out_dim)
        
        self.blocks = nn.ModuleList([Translover_block(
                                                        num_channels = num_channels, 
                                                        num_heads = num_heads, 
                                                        dropout   = dropout, 
                                                        mlp_ratio = mlp_ratio,
                                                        act = act,
                                                        slice_num = num_slices
                                                        ) for _ in range(n_layers)])
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
            x = self.preprocess(x) + self.placeholder_para
        else:
            x = self.preprocess(x)

        for block in self.blocks:
            x = block(x)
        x = self.mlp(self.ln(x))

        return x

class Translover_block(nn.Module):
    def __init__(
                    self,
                    num_heads: int,
                    num_channels: int,
                    dropout: float,
                    act='gelu',
                    mlp_ratio=4,
                    slice_num=32
                ):
        super(Translover_block, self).__init__()
        
        self.ln_1 = nn.LayerNorm(num_channels)
        self.Attn = Physics_Attention_Irregular_Mesh(num_channels, heads=num_heads, dim_head=num_channels // num_heads,
                                                     dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(num_channels)
        self.mlp = MLP(num_channels, num_channels * mlp_ratio, num_channels, n_layers = 0, res=False, act=act)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        return fx
    
