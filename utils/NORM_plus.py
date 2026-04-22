
from timm.models.layers import trunc_normal_
import torch
import scipy.io as sio
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
# import torch_geometric.nn as gnn
from einops import rearrange
# from utils.lapy import Solver, TriaMesh, TetMesh

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        
        self.placeholder = args['placeholder']
        n_layers  = args['n_layers']
        input_dim = args['x_dim']
        out_dim   = args['y_dim']
        num_channels = args['num_channels']
        num_lbos     = args['num_lbos']
        # space_dim    = args['space_dim']
        num_heads    = args['num_heads']
        device       = args['device']
        model_type   = args['model_type']
        # print('Model is ', model_type)
        
        lbo_file = sio.loadmat(args['lbo_path'])
        lbo_data  = lbo_file['Eigenvectors'][:,:num_lbos]
        mass      = lbo_file['Mass' ] 
        
        if num_lbos > lbo_data.shape[1]:
            raise ValueError("Please check 'num of lbo' !")
        lbo_bases = torch.Tensor(lbo_data).to(device)
        mass      = torch.Tensor(mass).to(device)
        lbo_inver = mass @ lbo_bases
        print('lbo_bases:', lbo_bases.shape, 'mass:', mass.shape)
        
        self.fc0 = nn.Linear(input_dim, num_channels) 
        self.fc1 = nn.Linear(num_channels, 128)
        self.fc2 = nn.Linear(128, out_dim)
        
        # self.ln  = nn.LayerNorm(num_channels)
        # self.mlp = nn.Linear(num_channels, out_dim)
        
        model_type = 'NORM_plus'
        edge_index = None
        edge_weight = None

        self.blocks = nn.ModuleList([NO_Layer(
                                                    model_type = model_type,
                                                    lbo_bases = lbo_bases,
                                                    lbo_inver = lbo_inver,
                                                    num_channels = num_channels,
                                                    num_heads    = num_heads,
                                                    num_modes    = num_lbos,
                                                    dropout      = 0.0,
                                                    edge_index  = edge_index,
                                                    edge_weight = edge_weight,
                                                    act='gelu',
                                                    mlp_ratio=1,
                                                    kernelsize = 3
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
            x = self.fc0(x) + self.placeholder_para
        else:
            x = self.fc0(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        # x = self.mlp(self.ln(x))
        return x


class NO_Layer(nn.Module):
    def __init__(
                    self,
                    model_type,
                    lbo_bases,
                    lbo_inver,
                    num_channels: int,
                    num_heads: int,
                    num_modes: int,
                    dropout: float,
                    edge_index,
                    edge_weight,
                    act='gelu',
                    mlp_ratio=1,
                    kernelsize = 3
                ):
        super(NO_Layer, self).__init__()
        
        self.ln_1 = nn.LayerNorm(num_channels)
        self.Conv = Convolution_block(
                                        model_type   = model_type,
                                        num_channels = num_channels, 
                                        num_heads = num_heads,  
                                        num_modes = num_modes, 
                                        edge_index  = edge_index,
                                        edge_weight = edge_weight,
                                        kernelsize = kernelsize,
                                        dropout = dropout)
        
        self.ln_2 = nn.LayerNorm(num_channels)
        self.mlp  = MLP(num_channels, num_channels * mlp_ratio, num_channels, n_layers = 0, res=False, act=act)
        self.lbo_bases = lbo_bases
        # self.mass = mass
        self.lbo_inv = lbo_inver
    def forward(self, fx):
        fx = self.Conv(self.ln_1(fx), self.lbo_bases, self.lbo_inv) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        return fx


class Convolution_block(nn.Module):
    def __init__(self, 
                 model_type,
                 num_channels, 
                 num_heads = 8,  
                 num_modes = 64, 
                 edge_index  = None,
                 edge_weight = None,
                 kernelsize = 3,
                 dropout = 0.0): 
        super().__init__()
        
        self.SpectralConv = Spectral_Conv(  num_channels = num_channels, 
                                            num_heads = num_heads,  
                                            num_modes = num_modes, 
                                            kernelsize = kernelsize,
                                            dropout = dropout)
        
    def forward(self, x, LBO_MATRIX, LBO_INVER ):
        #  x : B N C
        x = self.SpectralConv(x, LBO_MATRIX, LBO_INVER) 
        
        return x
    

class Spectral_Conv(nn.Module):
    def __init__(self, 
                 num_channels, 
                 num_heads = 8,  
                 num_modes = 64, 
                 kernelsize = 3,
                 dropout = 0.0): 
        super().__init__()
        
        if num_channels % num_heads == 0:
            self.project = True
            
        dim_head = num_channels // num_heads
        inner_dim = dim_head * num_heads
        self.dim_head = dim_head
        self.heads = num_heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.temperature = nn.Parameter(torch.ones([1, num_heads, 1, 1]) * 0.5)
        self.in_project_fx     = nn.Conv1d(num_channels, inner_dim, kernelsize, 1, kernelsize//2 )
        self.mlp_trans_weights = nn.Parameter( torch.empty((dim_head, dim_head)) )
        torch.nn.init.kaiming_uniform_(self.mlp_trans_weights, a=math.sqrt(5))
        self.layernorm  = nn.LayerNorm( (num_modes, dim_head ) )
        self.to_out     = nn.Sequential( nn.Linear(inner_dim, num_channels), nn.Dropout(dropout) )
        
    def forward(self, x, LBO_MATRIX, LBO_INVER):
        
        B, N, C = x.shape
        x = x.permute(0, 2, 1).contiguous()  # B C N
        fx_mid = self.in_project_fx(x).permute(0, 2, 1).contiguous() \
                 .reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()  # B H N D, HD~C
        # spectral_feature = LBO_INVERSE @ fx_mid # B H K D
        spectral_feature = (fx_mid.permute(0, 1, 3, 2) @ LBO_INVER).permute(0, 1, 3, 2)
        
        bsize, hsize, ksize, dsize = spectral_feature.shape
        spectral_feature = self.layernorm(spectral_feature.reshape( -1, ksize, dsize )).reshape( bsize, hsize, ksize, dsize )
        out_spectral_feature = torch.einsum("bhgi,io->bhgo", spectral_feature, self.mlp_trans_weights)
        
        out_x = LBO_MATRIX @ out_spectral_feature
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        
        return self.to_out(out_x)


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