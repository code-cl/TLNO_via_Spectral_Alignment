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

from utils_geo.utils_run import Dataset, Train, Test, load_model
from utils_geo.utils_data import read_data, read_lbo
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
            f.write(f"{'x_train'}: {args['x_train'].shape}\n")
            f.write(f"{'y_train'}: {args['y_train'].shape}\n")
            f.write(f"{'x_test'}: {args['y_test'].shape}\n")
            f.write(f"{'y_test'}: {args['y_test'].shape}\n")
            for key, value in args.items():
                if key == 'x_train' or key == 'y_train' or key == 'x_test' or key == 'y_test' or \
                   key == 'train_index' or key == 'test_index' or key == 'lbo_bases' or key == 'lbo_inver':
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
    
    data_path = '../Data/'
    save_path = '../logs_geos/'
    
    n_train = 5000
    n_test  = 2000
    
    max_nodes = 1082
    num_lbos = 64
    
    files = ['1', '2', '3'] # , '2', '3'
    x_trains = []
    x_tests  = []
    y_trains = []
    y_tests  = []
    train_index = []
    test_index = []
    LBO_Base = []
    LBO_Inver = []
    
    for i in range (len(files)):
        
        file = files[i]
        save_path = save_path + file
        file_path = data_path + 'Petals' + file + '_data/' 
        x_train, y_train, x_test, y_test = read_data(file_path, n_train, n_test)
        lbo_bases, lbo_inver             = read_lbo(file_path, num_lbos, 'Coupled_basis_function')
        
        LBO_Base.append(lbo_bases)
        LBO_Inver.append(lbo_inver)
        print('x_train:', x_train.shape, 'y_train:', y_train.shape)
        print('x_test:' , x_test.shape , 'y_test:' , y_test.shape)
        print('lbo_bases:' , lbo_bases.shape , 'lbo_inver:' , lbo_inver.shape)
        
        num_nodes = x_train.shape[1]
        pad_length = max_nodes - num_nodes
        pad_width = [
                        (0, 0),    # 第一维（batch）：前后都不填充
                        (0, pad_length),  # 第二维（序列）
                        (0, 0)     # 第三维（特征）：不填充
                    ]
        x_train = np.pad(x_train, pad_width=pad_width, mode='constant')
        y_train = np.pad(y_train, pad_width=pad_width, mode='constant')
        x_test = np.pad(x_test, pad_width=pad_width, mode='constant')
        y_test = np.pad(y_test, pad_width=pad_width, mode='constant')
        # print(x_train.shape, y_train.shape)
        # print(x_test.shape, y_test.shape)
        
        x_trains.append(x_train)
        y_trains.append(y_train)
        x_tests.append(x_test)
        y_tests.append(y_test)
        train_index.append(np.array([[num_nodes, i]]).repeat(n_train, axis=0))
        test_index.append(np.array([[num_nodes, i]]).repeat(n_test, axis=0))
        
    x_trains = np.concatenate(x_trains, axis=0)
    y_trains = np.concatenate(y_trains, axis=0)
    x_tests  = np.concatenate(x_tests, axis=0)
    y_tests  = np.concatenate(y_tests, axis=0)
    train_index = np.concatenate(train_index, axis=0)
    test_index  = np.concatenate(test_index, axis=0)
    
    print('Data information:')
    print('x_trains:', x_trains.shape, 'y_trains:', y_trains.shape)
    print('x_tests:' , x_tests.shape , 'y_tests:' , y_tests.shape)
    print('train_index:', train_index.shape, 'test_index:', test_index.shape)
    
    batch_size = 16
    # random_index = np.arange(x_trains.shape[0])
    # np.random.shuffle(random_index.reshape(-1, batch_size))
    # shuffled_index = random_index.reshape(-1, batch_size).flatten()
    
    args = dict()
    # Set data
    args['x_train']  = x_trains#[shuffled_index]
    args['y_train']  = y_trains#[shuffled_index]
    args['x_test']  = x_tests
    args['y_test']  = y_tests
    args['train_index'] = train_index
    args['test_index']  = test_index
    # args['n_train'] = 
    args['n_test' ] = x_tests.shape[0]
    args['save_tedata_size'] = args['n_test' ]
    args['save_trdata_size'] = 100
    dim_space = 3 # Attention!!!
    n_trains = [x_trains.shape[0]]
    n_times = 1
    
    # Set training parameter
    args['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args['loss']   = 'L2' # L2 or SSE
    args['batch_size']   = batch_size
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
    args['model_type']   = 'NORM_plus'
    if args['model_type']   == 'NORM_plus':
        args['n_layers']     = 8
        args['num_lbos']     = 64
        args['num_channels'] = 128
        args['num_heads']    = 8
        args['mlp_ratio']    = 1
        args['lbo_bases']  = LBO_Base
        args['lbo_inver']  = LBO_Inver
        current_directory = '_nte' + str(args['n_test'])\
                            + '_' + args['loss'] + '_nl' + str(args['n_layers'])+ '_nm' + str(args['num_lbos'])\
                            + '_nc' + str(args['num_channels']) + '_nh' + str(args['num_heads']) + '_' + args['optimizer']+ '_' + args['scheduler']
        
    else:
        raise ValueError("Please check 'model_type' !")
        
    args['norm_type'] = 'coeff_norm'  # 'coeff_norm' or 'point_norm' or 'no_norm
    args['x_dim'] = args['x_train'].shape[-1]
    if len(args['y_train'].shape)==2:
        args['y_dim'] = 1
    elif len(args['y_train'].shape)==3:
        args['y_dim'] = args['y_train'].shape[-1]
    
    
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