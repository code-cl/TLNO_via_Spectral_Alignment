import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import torch
import numpy as np
import scipy.io as sio
import warnings
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from utils.utils_run import Dataset, Train, Test, load_model
def train(args, current_directory, n_times, task_type = None):
    
    save_indexs = np.arange(0, n_times, 1)
    err_results = np.zeros((n_times, 4))
    
    for i, save_index in enumerate(save_indexs):
        
        args['save_path'] = current_directory + '/' + str(save_index)
        if not os.path.exists(args['save_path']):
            os.makedirs(args['save_path'])
        else:
            import shutil
            shutil.rmtree(args['save_path'])
            os.makedirs(args['save_path'])
            
        txt_path = args['save_path'] + "/args.txt"
        with open(txt_path, "w") as f:
            # f.write(f"{'n_floders'}: {n_floders}\n")
            for key, value in args.items():
                if key == 'xdata' or key == 'ydata' or key == 'nodes' or key == 'elems' or key == 'Data_modes':
                    continue  # 跳过 'data'
                f.write(f"{key}: {value}\n")
        
        if task_type is not None:
            data_array = Dataset(args, task_type='Transfer')
            if 1: # Test source model
                source_model = load_model(args, task_type='Transfer')
                Test(args, data_array, source_model, current_task = 'Source')
            model, source_model = Train(args, data_array, task_type='Transfer')
        else:
            data_array = Dataset(args)
            model = Train(args, data_array)
            
        print('\n')    
        print(args['save_path'])
        err_results[i] = Test(args, data_array, model)
        
    sio.savemat(current_directory + '/err_results.mat', {'value': err_results})
    mean_err = np.mean(err_results, axis = 0)
    txt_path =  current_directory + '/log.txt'
    with open(txt_path, "w") as f:
        f.write(f"{'Test_loss'} : {mean_err[0]}\n")
        f.write(f"{'Test_MAE'}  : {mean_err[1]}\n")
        f.write(f"{'Test_MMax'} : {mean_err[2]}\n")
        f.write(f"{'Test_Max'}  : {mean_err[3]}\n")
        f.write("\n")
        for i in range(n_times):
            f.write(f"{'Error'}  : {err_results[i]}\n")
    
    return err_results


if __name__ == "__main__":
    
    name = 'Geo_A'
    data_path = '../Data/'+ name
    save_path = '../logs/'+ name + '/'
    lbo_path  = data_path + '/Nodes_LBO_basis'
    
    # Read data
    print('data_path:', data_path)
    data = sio.loadmat(data_path + '/data.mat')
    x_data = data["c_field"]
    y_data = data["u_field"]
    if len(x_data.shape)==2:
        x_data = x_data.reshape(x_data.shape[0],-1,1)

    args = dict()
    # Set data
    args['xdata']  = x_data
    args['ydata']  = y_data
    args['n_test' ] = 200
    args['save_tedata_size'] = args['n_test' ]
    args['save_trdata_size'] = 100
    dim_space = 2 # Attention!!!
    n_trains = [800]
    n_times = 3
    
    # Set training parameter
    args['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args['loss']   = 'L2' # L2 or SSE
    args['batch_size']   = 16
    args['learn_rate']   = 0.001
    args['epoch'  ]      = 500
    args['max_grad_norm'] = 0.1
    args['placeholder']   = True 
    args['initialize_weights'] = True
    args['optimizer'] = 'AdamW' 
    args['scheduler'] = 'OneCycleLR' # OneCycleLR 
    if args['scheduler'] == 'StepLR':
        args['step_size'] = 50
        args['gamma']     = 0.7
        args['learn_rate']   = 0.001
        args['max_grad_norm'] = None
    args['weight_decay'] = 1e-5
    
    # Set model parameter
    args['model_type']   = 'NORM_plus' # NORM_plus, NORM, Transolver                      
    if args['model_type']   == 'NORM_plus':
        args['n_layers']     = 8
        args['num_lbos']     = 32
        args['num_channels'] = 64
        args['num_heads']    = 4
        args['mlp_ratio']    = 1
        args['lbo_path']  = lbo_path
        current_directory = '_nte' + str(args['n_test'])\
                            + '_' + args['loss'] + '_nl' + str(args['n_layers'])+ '_nm' + str(args['num_lbos'])\
                            + '_nc' + str(args['num_channels']) + '_nh' + str(args['num_heads']) + '_' + args['optimizer']+ '_' + args['scheduler']
        
    else:
        raise ValueError("Please check 'model_type' !")
        
    args['norm_type'] = 'coeff_norm'  # 'coeff_norm' or 'point_norm' or 'no_norm
    args['x_dim'] = args['xdata'].shape[-1]
    if len(y_data.shape)==2:
        args['y_dim'] = 1
    elif len(y_data.shape)==3:
        args['y_dim'] = y_data.shape[-1]
    
    
    for n_t in n_trains:
        args['n_train'] = n_t 
        current_path = save_path + '/' + args['model_type'] + '_ntr' + str(args['n_train']) + current_directory
        err_results = train(args, current_path, n_times)
        mean_err = np.mean(err_results, axis = 0)
        print('\n**********************')
        print('\nMean error of '+str(n_times)+' run:')
        print('\nTesting error: %.4f'%(mean_err[0]))
        print('Testing MAError: %.4e'%(mean_err[1]))
        print('Testing MeanMax: %.4f'%(mean_err[2]))
        print('Testing MaxError: %.4f'%(mean_err[3]))
        print('\n')
        for i in range(n_times):
            print(f"{'Error'}  : {err_results[i]}\n")