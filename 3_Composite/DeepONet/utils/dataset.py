'''
Authors: Katiana Kontolati, PhD Candidate, Johns Hopkins University
         Somdatta Goswami, Postdoctoral Researcher, Brown University
Tensorflow Version Required: TF1.15     
'''
import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io as io
import sys
# np.random.seed(1234)

class DataSet:
    def __init__(self, bs, args):

        self.bs = bs
        if args['task_type'] == 'Train_source':
            self.F_train, self.U_train, self.F_test, self.U_test, \
            self.X, self.u_mean, self.u_std = self.load_data(args)
            
            print('Source data...')
            print(self.F_train.shape, self.U_train.shape)
            print(self.F_test.shape, self.U_test.shape)
            
        else:
            # self.F_train, self.U_train, self.F_test, self.U_test, \
            # self.X, self.u_mean, self.u_std = self.load_data(args)
            
            self.F_train_t, self.U_train_t, self.F_test_t, self.U_test_t, \
            self.X_t, self.u_mean_t, self.u_std_t = self.load_data_target(args)
        
            # print('Source data...')
            # print(self.F_train.shape, self.U_train.shape)
            # print(self.F_test.shape, self.U_test.shape)
            print('Target data...')
            print(self.F_train_t.shape, self.U_train_t.shape)
            print(self.F_test_t.shape,  self.U_test_t.shape)
        
    def decoder(self, x):
        x = x*(self.u_std + 1.0e-9) + self.u_mean
        return x
    
    def decoder_target(self, x):
        x = x*(self.u_std_t + 1.0e-9) + self.u_mean_t
        return x

    def load_data(self, args):
    
        # h = args['source_h'] # mesh resolution
        # w = args['source_w'] # mesh resolution
        s = args['source_r'] # node number
        s_bc = args['source_r_bc'] # node number
        n_train = args['source_ntr']
        n_test  = args['source_nte']
        # n_c = args['n_channels']

        f_train = args['x_data'][0:n_train]
        u_train = args['y_data'][0:n_train]
        
        f_test = args['x_data'][-1*n_test:]
        u_test = args['y_data'][-1*n_test:]
        
        if n_train + n_test > args['x_data'].shape[0]:
            raise ValueError("Please check 'load_data_target' !")
            
        X = args['nodes']
        
        f_train_mean = np.mean(np.reshape(f_train, (-1, s_bc)), 0)
        f_train_std  = np.std(np.reshape(f_train, (-1, s_bc)), 0)
        u_train_mean = np.mean(np.reshape(u_train, (-1, s)), 0)
        u_train_std  = np.std(np.reshape(u_train, (-1, s)), 0)
        
        f_train_mean = np.reshape(f_train_mean, (-1, 1, s_bc))
        f_train_std = np.reshape(f_train_std, (-1, 1, s_bc))
        u_train_mean = np.reshape(u_train_mean, (-1, s, 1))
        u_train_std = np.reshape(u_train_std, (-1, s, 1))
        
        F_train = np.reshape(f_train, (-1, 1, s_bc))
        F_train = (F_train - f_train_mean)/(f_train_std + 1.0e-9) 
        U_train = np.reshape(u_train, (-1, s, 1))
        U_train = (U_train - u_train_mean)/(u_train_std + 1.0e-9)
       

        F_test = np.reshape(f_test, (-1, 1, s_bc))
        F_test = (F_test - f_train_mean)/(f_train_std + 1.0e-9) 
        U_test = np.reshape(u_test, (-1, s, 1))
        U_test = (U_test - u_train_mean)/(u_train_std + 1.0e-9)

        return F_train, U_train, F_test, U_test, X, u_train_mean, u_train_std


    def load_data_target(self, args):
        
        s = args['target_r'] # node number
        s_bc = args['target_r_bc'] # node number
        n_train = args['target_ntr']
        n_test  = args['target_nte']
        # n_c = args['n_channels']

        f_train = args['x_data'][0:n_train]
        u_train = args['y_data'][0:n_train]
        
        f_test = args['x_data'][-1*n_test:]
        u_test = args['y_data'][-1*n_test:]
        
        if n_train + n_test > args['x_data'].shape[0]:
            raise ValueError("Please check 'load_data_target' !")
            
        X_t = args['nodes']
        
        f_train_mean = np.mean(np.reshape(f_train, (-1, s_bc)), 0)
        f_train_std  = np.std(np.reshape(f_train, (-1, s_bc)), 0)
        u_train_mean = np.mean(np.reshape(u_train, (-1, s)), 0)
        u_train_std  = np.std(np.reshape(u_train, (-1, s)), 0)
        
        f_train_mean = np.reshape(f_train_mean, (-1, 1, s_bc))
        f_train_std = np.reshape(f_train_std, (-1, 1, s_bc))
        u_train_mean = np.reshape(u_train_mean, (-1, s, 1))
        u_train_std = np.reshape(u_train_std, (-1, s, 1))

        F_train_t = np.reshape(f_train, (-1, 1, s_bc))
        F_train_t = (F_train_t - f_train_mean)/(f_train_std + 1.0e-9) 
        U_train_t = np.reshape(u_train, (-1, s, 1))
        U_train_t = (U_train_t - u_train_mean)/(u_train_std + 1.0e-9)
       

        F_test_t = np.reshape(f_test, (-1, 1, s_bc))
        F_test_t = (F_test_t - f_train_mean)/(f_train_std + 1.0e-9) 
        U_test_t = np.reshape(u_test, (-1, s, 1))
        U_test_t = (U_test_t - u_train_mean)/(u_train_std + 1.0e-9)
        
        return F_train_t, U_train_t, F_test_t, U_test_t, X_t, u_train_mean, u_train_std

    # Source
    def minibatch(self, batch_id = None):
        # choose random indices - replace=False to avoid sampling same data
        if batch_id is  None:
            batch_id = np.random.choice(self.F_train.shape[0], self.bs, replace=False)
        f_train = self.F_train[batch_id]
        u_train = self.U_train[batch_id]
        x_train = self.X

        # Xmin = np.array([ -1., -1.]).reshape((-1, 2))
        # Xmax = np.array([ 1., 1.]).reshape((-1, 2))
        
        Xmin = np.min(x_train, axis=0, keepdims=True)  # 保持二维形状
        Xmax = np.max(x_train, axis=0, keepdims=True)  # 计算每列的最大值

        return x_train, f_train, u_train, Xmin, Xmax

    def testbatch(self, num_test, batch_id = None):
        if batch_id is None:
            batch_id = np.random.choice(self.F_test.shape[0], num_test, replace=False)
        f_test = self.F_test[batch_id]
        u_test = self.U_test[batch_id]
        x_test = self.X

        batch_id = np.reshape(batch_id, (-1, 1))

        return batch_id, x_test, f_test, u_test
    
    def testall(self):
        # batch_id = np.random.choice(self.F_test.shape[0], num_test, replace=False)
        f_test = self.F_test
        u_test = self.U_test
        x_test = self.X

        # batch_id = np.reshape(batch_id, (-1, 1))

        return x_test, f_test, u_test
    
    # Target
    def minibatch_target(self, batch_id = None):
        # choose random indices - replace=False to avoid sampling same data
        if batch_id is None:
            batch_id = np.random.choice(self.F_train_t.shape[0], self.bs, replace=False)
        f_train = self.F_train_t[batch_id]
        u_train = self.U_train_t[batch_id]
        x_train = self.X_t

        # Xmin = np.array([ -1., -1.]).reshape((-1, 2))
        # Xmax = np.array([ 1, 1.]).reshape((-1, 2))
        Xmin = np.min(x_train, axis=0, keepdims=True)  # 保持二维形状
        Xmax = np.max(x_train, axis=0, keepdims=True)  # 计算每列的最大值
        
        return x_train, f_train, u_train, Xmin, Xmax


    def testbatch_target(self, num_test, batch_id = None):
        if batch_id is None:
            batch_id = np.random.choice(self.F_test_t.shape[0], num_test, replace=False)
        f_test = self.F_test_t[batch_id]
        u_test = self.U_test_t[batch_id]
        x_test = self.X_t

        batch_id = np.reshape(batch_id, (-1, 1))

        return batch_id, x_test, f_test, u_test
    
    def testall_target(self):
        # batch_id = np.random.choice(self.F_test_t.shape[0], num_test, replace=False)
        f_test = self.F_test_t
        u_test = self.U_test_t
        x_test = self.X_t

        # batch_id = np.reshape(batch_id, (-1, 1))

        return x_test, f_test, u_test

