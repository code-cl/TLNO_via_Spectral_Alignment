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
        
        if task_type is not None:
            args['source_model'] = args['source_str'] + '/' + str(save_index)
            print('Source model: ', args['source_model'])
            data_array = Dataset(args, task_type='Transfer')
            if 1: # Test source model
                source_model = load_model(args, task_type='Transfer')
                Test(args, data_array, source_model, current_task = 'Source')
            model, source_model = Train(args, data_array, task_type='Transfer')
        else:
            data_array = Dataset(args)
            model = Train(args, data_array)
        
        txt_path = args['save_path'] + "/args.txt"
        with open(txt_path, "w") as f:
            # f.write(f"{'n_floders'}: {n_floders}\n")
            for key, value in args.items():
                if key == 'xdata' or key == 'ydata' or key == 'nodes' or key == 'elems' or key == 'Data_modes':
                    continue  # 跳过 'data'
                f.write(f"{key}: {value}\n")
                
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
    
    source_shape = 'Geo_A'
    target_shape = 'Geo_B'
    data_path = '../Data/' + target_shape
    save_path = '../logs/' + source_shape + '_' + target_shape
    lbo_path  = data_path + '/Nodes_LBO_basis'
    
    source_model_str = 'NORM_plus_ntr300_nte200_L2_nl8_nm64_nc128_nh8_AdamW_OneCycleLR'
    source_model_path = '../logs/' + source_shape + '/' + source_model_str
                        
    from utils.function import extract_letter
    para_dict = extract_letter(source_model_str)
    
    # Read data
    print('source_shape:', source_shape)
    print('data_path:', data_path)
    data = sio.loadmat(data_path + '/data.mat')
    x_data = data["T_field"]
    y_data = data["D_field"]
    if len(x_data.shape)==2:
        x_data = x_data.reshape(x_data.shape[0],-1,1)

    args = dict()
    # Set data
    args['xdata']  = x_data
    args['ydata']  = y_data
    args['n_test' ] = 200
    args['save_tedata_size'] = args['n_test' ]
    args['save_trdata_size'] = 100
    dim_space = 3  # Attention!!!
    n_trains = [30, 40, 50]
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
    args['model_type']   = para_dict['mt']
    if args['model_type']   == 'NORM':
        args['n_layers']     = para_dict['nl']
        args['num_lbos']     = para_dict['nm']
        args['num_channels'] = para_dict['nc']
        args['mlp_ratio']    = 1
        current_directory = '_nte' + str(args['n_test'])\
                            + '_' + args['loss'] + '_nl' + str(args['n_layers'])+ '_nm' + str(args['num_lbos'])\
                            + '_nc' + str(args['num_channels']) + '_' + args['optimizer']+ '_' + args['scheduler']
                             
    elif args['model_type']   == 'NORM_plus':
        args['n_layers']     = para_dict['nl']
        args['num_lbos']     = para_dict['nm']
        args['num_channels'] = para_dict['nc']
        args['num_heads']    = para_dict['nh']
        args['mlp_ratio']    = 1
        current_directory = '_nte' + str(args['n_test'])\
                            + '_' + args['loss'] + '_nl' + str(args['n_layers'])+ '_nm' + str(args['num_lbos'])\
                            + '_nc' + str(args['num_channels']) + '_nh' + str(args['num_heads']) + '_' + args['optimizer']+ '_' + args['scheduler']
        
    elif args['model_type']   == 'Transolver':
        args['n_layers']     = para_dict['nl']
        args['num_slices']   = para_dict['ns']
        args['num_channels'] = para_dict['nc']
        args['num_heads']    = para_dict['nh']
        args['mlp_ratio']    = 1
        
        lbo_data = sio.loadmat(lbo_path)
        nodes    = lbo_data['Points'][:,0:dim_space]
        Points_expanded = np.tile(nodes, (x_data.shape[0], 1, 1))
        args['xdata']   = np.concatenate([x_data, Points_expanded], axis=-1)
        current_directory = '_nte' + str(args['n_test']) + '_' + args['loss'] \
                            + '_nl' + str(args['n_layers'])+ '_ns' + str(args['num_slices'])+ '_nc' + str(args['num_channels']) \
                            + '_nh' + str(args['num_heads']) + '_' + args['optimizer']+ '_' + args['scheduler']
    else:
        raise ValueError("Please check 'model_type' !")
    
    args['source_str'] = source_model_path
    args['lbo_path']  = lbo_path
    args['norm_type'] = 'coeff_norm'  # 'coeff_norm' or 'point_norm' or 'no_norm
    args['x_dim'] = args['xdata'].shape[-1]
    if len(y_data.shape)==2:
        args['y_dim'] = 1
    elif len(y_data.shape)==3:
        args['y_dim'] = y_data.shape[-1]
    
    for n_t in n_trains:
        args['n_train'] = n_t 
        current_path = save_path + '/' + args['model_type'] + '_ntr' + str(args['n_train']) + current_directory
        err_results = train(args, current_path, n_times, task_type='Transfer')
        mean_err    = np.mean(err_results, axis = 0)
        print('\n**********************')
        print('\nMean error of '+str(n_times)+' run:')
        print('\nTesting error: %.4f'%(mean_err[0]))
        print('Testing MAError: %.4e'%(mean_err[1]))
        print('Testing MeanMax: %.4f'%(mean_err[2]))
        print('Testing MaxError: %.4f'%(mean_err[3]))
        print('\n')
        for i in range(n_times):
            print(f"{'Error'}  : {err_results[i]}\n")