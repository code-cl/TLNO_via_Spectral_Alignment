# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 16:16:01 2025

@author: Chenl
"""
import torch
import torch.nn as nn
import re

def extract_letter(s):
    """
    提取字符串中字母前缀和对应数字的配对，并存入字典
    例如："ntr200" -> {'ntr': 200}
    """
    # 使用split先按"_"分割字符串
    parts = s.split('_')
    
    result_dict = {}
    
    for part in parts:
        # 对每个部分使用正则匹配
        match = re.match(r'^([a-zA-Z]+)(\d+)$', part)
        if match:
            letters, numbers = match.groups()
            # 将数字转换为整数，并存入字典
            result_dict[letters] = int(numbers)
    
    if 'NORM_plus' in s:
        result_dict['mt'] = 'NORM_plus'
    elif 'Transolver' in s:
        result_dict['mt'] = 'Transolver'
    elif 'NORM' in s:
        result_dict['mt'] = 'NORM'
    return result_dict

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
        # print(f"{name:<40} | {str(param.size()):<20} | {param_count:<10}")
    # print(f"Total Trainable Params: {total_params}")
    return total_params

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 
              'leaky_relu': nn.LeakyReLU(0.1), 'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}
              
class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) 
                                      for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x
    
class SpectralConv(nn.Module):
    def __init__ (self, num_channels, num_modes):
        super(SpectralConv, self).__init__()
        self.scale = (1 / (num_channels*num_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(num_channels, num_channels, 
                                                            num_modes, dtype=torch.float))
    def forward(self, x, LBO_MATRIX, LBO_INVERSE):
        # x = x.permute(0, 2, 1)
        # print(x.shape)
        x = LBO_INVERSE @ x  
        # print(x.shape)
        x = x.permute(0, 2, 1)
        x = torch.einsum("bix,iox->box", x[:, :], self.weights)
        # print(x.shape)
        x =  x @ LBO_MATRIX.T
        return x.permute(0, 2, 1)