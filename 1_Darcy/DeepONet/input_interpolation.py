import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.tri as mtri
# import torch
import numpy as np
import scipy.io as sio
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import scipy.interpolate as interp
# from scipy.interpolate import NearestNDInterpolator

if __name__ == "__main__":
    
    name = 'Geo_B'
    data_path = '../Data/'+name
    save_path = 'logs/' + name + '/'
    lbo_path  = data_path + '/Nodes_LBO_basis'
    
    lbo_data = sio.loadmat(lbo_path)
    nodes  = lbo_data['Points'][:,0:2]
    elems  = lbo_data['Elements']
    
    # elems  = np.hstack((np.full((elems.shape[0], 1), elems.shape[1]), elems))
    # print('nodes:', nodes.shape, elems.shape, np.min(elems))
    # print('elems:', elems.shape, np.min(elems))
    
    data = sio.loadmat(data_path + '/data.mat')
    x_data = data["c_field"]
    
    x_min, x_max = nodes[:, 0].min(), nodes[:, 0].max()
    y_min, y_max = nodes[:, 1].min(), nodes[:, 1].max()
    grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    
    n_steps = x_data.shape[0]
    x_gridded = np.zeros((n_steps, 100, 100))
    
    for i in range(n_steps):
        print(i)
        interp_func = interp.NearestNDInterpolator(nodes, x_data[i])
        x_gridded[i] = interp_func(grid_points).reshape((100,100))
    
    sio.savemat(data_path + '/data_.mat',{'c_field':data["c_field"],
                                          'u_field':data["u_field"],
                                          'c_field_':x_gridded,
                                          'xx':data["xx"],
                                          'yy':data["yy"]})
    
    if 1:
        k = 23
        triangle = mtri.Triangulation(nodes[:,0], nodes[:,1], elems)
        fig, axs = plt.subplots(figsize=(8,6)) 
        plt.subplots_adjust(left=0.08, right=0.95,bottom=0.16,top=0.9,wspace=0.2)
        
        ax1 = plt.subplot(121)
        ax1.set_aspect(1)
        cs = plt.tricontourf(triangle, x_data[k], np.linspace(3, 15, 5), cmap='binary')
        cb = plt.colorbar(cs, location = 'bottom')
        cb.set_ticks([5, 10, 15])
        cb.formatter = ticker.FormatStrFormatter('%.2f')
        cb.update_ticks()
        plt.title('Input field', fontsize=12)
        
        ax2 = plt.subplot(122)
        ax2.set_aspect(1)
        cs2 = ax2.contourf(grid_x, grid_y, x_gridded[k], np.linspace(3, 15, 5), cmap='binary')
        cb2 = plt.colorbar(cs2, location='bottom')
        cb2.set_ticks([5, 10, 15])
        cb2.formatter = ticker.FormatStrFormatter('%.2f')
        cb2.update_ticks()
        plt.title('Interpolation result', fontsize=12)
        # plt.title('插值结果 (100×100规则网格)', fontsize=12)
        # plt.xlabel('X坐标')
        
        plt.show()
        