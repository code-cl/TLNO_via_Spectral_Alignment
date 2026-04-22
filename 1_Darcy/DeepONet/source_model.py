'''
Manuscript Associated: Deep transfer operator learning for partial differential equations under conditional shift
Authors: Katiana Kontolati, PhD Candidate, Johns Hopkins University
         Somdatta Goswami, Postdoctoral Researcher, Brown University
Tensorflow Version Required: TF1.15     
This should be used for sharp data    

This is the source model.
'''
import os
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as io
from utils.dataset import DataSet
from utils.fnn import FNN
from utils.conv import CNN
from utils.savedata import SaveData
import shutil
import sys 
from functools import reduce
from operator import mul
print("You are using TensorFlow version", tf.__version__)
import math
# np.random.seed(1234)
#output dimension of Branch/Trunk (latent dimension)
p = 200
#fnn in CNN
layer_B = [512, 256, p]
#trunk net
layer_T = [2, 128, 128, 128, p]
#resolution
h = 100
w = 100
#parameters in CNN
n_channels = 1
#n_out_channels = 16
filter_size_1 = 5
filter_size_2 = 5
filter_size_3 = 5
filter_size_4 = 5
stride = 1

#filter size for each convolutional layer
num_filters_1 = 16
num_filters_2 = 16
num_filters_3 = 16
num_filters_4 = 64

#batch_size
# bs = 100

#size of input for Trunk net
# x_num = 1628
def get_num_params():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params

def main(args):
    
    x_num = args['source_r']
    bs = args['batch_size'] 
    ntr = args['source_ntr']
    nte = args['source_nte']
    data = DataSet(bs, args)
    x_train, f_train, u_train, Xmin, Xmax = data.minibatch()

    x_pos = tf.constant(x_train, dtype=tf.float32)
    x = tf.tile(x_pos[None, :, :], [bs, 1, 1]) #[bs, x_num, x_dim]

    f_ph = tf.placeholder(shape=[None, h, w, n_channels], dtype=tf.float32) #[bs, 1, h, w, n_channels]
    u_ph = tf.placeholder(shape=[None, x_num, 1], dtype=tf.float32) #[bs, x_num, 1]
    learning_rate = tf.placeholder(tf.float32, shape=[])
    
    # Branch net
    conv_model = CNN()

    #conv_linear = conv_model.linear_layer(f_ph, n_out_channels)
    conv_1, W1, b1 = conv_model.conv_layer(f_ph, filter_size_1, num_filters_1, stride, actn=tf.nn.relu)  
    pool_1 = conv_model.avg_pool(conv_1, ksize=2, stride=2)  
    conv_2, W2, b2 = conv_model.conv_layer(pool_1, filter_size_2, num_filters_2, stride, actn=tf.nn.relu)
    pool_2 = conv_model.avg_pool(conv_2, ksize=2, stride=2) 
    conv_3, W3, b3 = conv_model.conv_layer(pool_2, filter_size_3, num_filters_3, stride, actn=tf.nn.relu)
    pool_3 = conv_model.avg_pool(conv_3, ksize=2, stride=2)
    conv_4, W4, b4 = conv_model.conv_layer(pool_3, filter_size_4, num_filters_4, stride, actn=tf.nn.relu)
    pool_4 = conv_model.avg_pool(conv_4, ksize=2, stride=2) 
    layer_flat = conv_model.flatten_layer(pool_4)

    fnn_layer_1, Wf1, bf1 = conv_model.fnn_layer(layer_flat, layer_B[0], actn=tf.tanh, use_actn=True)
    fnn_layer_2, Wf2, bf2 = conv_model.fnn_layer(fnn_layer_1, layer_B[1], actn=tf.nn.tanh, use_actn=True)
    out_B, Wf3, bf3 = conv_model.fnn_layer(fnn_layer_2, layer_B[-1], actn=tf.tanh, use_actn=False) #[bs, p]
    u_B = tf.tile(out_B[:, None, :], [1, x_num, 1]) #[bs, x_num, p]
    
    # Trunk net
    fnn_model = FNN()
    W, b = fnn_model.hyper_initial(layer_T)
    u_T = fnn_model.fnn(W, b, x, Xmin, Xmax)
    u_nn = u_B*u_T

    u_pred = tf.reduce_sum(u_nn, axis=-1, keepdims=True)

    #loss = tf.reduce_mean(tf.square(u_ph - u_pred))
    loss = tf.reduce_sum(tf.norm(u_pred - u_ph, 2, axis=1)/tf.norm(u_ph, 2, axis=1))
    train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    print('Num of paras : %d'%(get_num_params()))
    n = 0
    nmax = args['epochs']  # epochs
    start_time = time.perf_counter()
    time_step_0 = time.perf_counter()
    
    train_loss = np.zeros((nmax+1, 1))
    test_loss = np.zeros((nmax+1, 1))    
    nn = 0
    while n <= nmax:
        train_l2 = 0
        batch_ids = np.random.choice(ntr, ntr, replace=False)
        ni = math.ceil(len(batch_ids) / bs)
        for i in range(ni):
            if bs*(i+1) > len(batch_ids):
                batch_id = batch_ids[-1*bs:]
            else:
                batch_id = batch_ids[bs*i:bs*(i+1)]
            
            if nn <10000:
                lr = 0.0001
            elif (nn < 20000):
                lr = 0.00005
            elif (nn < 40000):
                lr = 0.00001
            else:
                lr = 0.000005
            nn += 1

            x_train, f_train, u_train, _, _ = data.minibatch(batch_id)
            train_dict = {f_ph: f_train, u_ph: u_train, learning_rate: lr}
            loss_, _, u_train_ = sess.run([loss, train, u_pred], feed_dict=train_dict)
            # Calculate train l2 error
            u_real  = data.decoder(u_train)
            u_real_ = data.decoder(u_train_)
            train_l2 += np.sum(np.linalg.norm(u_real_ - u_real, 2, axis=1)/np.linalg.norm(u_real, 2, axis=1))
       
        train_l2 /= (ni*bs)
        test_l2 = 0
        if n%1 == 0:
            batch_ids = np.arange(0, nte, 1)
            ni = math.ceil(len(batch_ids) / bs )
            for i in range(ni):
                if bs*(i+1) > len(batch_ids):
                    batch_id = batch_ids[-1*bs:]
                else:
                    batch_id = batch_ids[bs*i:bs*(i+1)]
                test_id, x_test, f_test, u_test = data.testbatch(bs, batch_id)
                u_test_ = sess.run(u_pred, feed_dict={f_ph: f_test})
                u_test  = data.decoder(u_test)
                u_test_ = data.decoder(u_test_)
                test_l2 += np.sum(np.linalg.norm(u_test_ - u_test, 2, axis=1)/np.linalg.norm(u_test, 2, axis=1))
            test_l2 /= (ni*bs)  
            time_step_1000 = time.perf_counter()
            T = time_step_1000 - time_step_0
            print('Step: %d, Loss: %.4f, Test L2 error: %.4f, Time (secs): %.4f'%(n, train_l2, test_l2, T))
            time_step_0 = time.perf_counter()
    
        train_loss[n,0] = train_l2
        test_loss[n,0] = test_l2
        n += 1
        
    stop_time = time.perf_counter()
    print('Elapsed time (secs): %.3f'%(stop_time - start_time))
    
    loss_dict = {'train_error':train_loss.flatten(),
                 'test_error' :test_loss.flatten()}
    io.savemat(save_results_to +'/DeepONetBST_loss.mat', mdict = loss_dict) 
    
    # Save variables  
    W1_,b1_,W2_,b2_,W3_,b3_,W4_,b4_,Wf1_,bf1_,Wf2_,bf2_,Wf3_,bf3_ = \
        sess.run([W1,b1,W2,b2,W3,b3,W4,b4,Wf1,bf1,Wf2,bf2,Wf3,bf3])
    
    savedict_cnn = {'W1':W1_,'b1':b1_,'W2':W2_,'b2':b2_,'W3':W3_,'b3':b3_,'W4':W4_,'b4':b4_,\
                'Wf1':Wf1_,'bf1':bf1_,'Wf2':Wf2_,'bf2':bf2_,'Wf3':Wf3_,'bf3':bf3_}    
    
    Wt1, bt1, Wt2, bt2, Wt3, bt3, Wt4, bt4,= sess.run([W[0], b[0], W[1], b[1], W[2], b[2], W[3], b[3]])
    
    savedict_fnn = {'W1':Wt1,'b1':bt1,'W2':Wt2,'b2':bt2,'W3':Wt3,'b3':bt3,'W4':Wt4,'b4':bt4}    
    
    # Save variables (weights + biases)
    io.savemat(save_variables_to+'/CNN_vars.mat', mdict=savedict_cnn)
    io.savemat(save_variables_to+'/FNN_vars.mat', mdict=savedict_fnn)
    
    print('source_path', args['source_path'])
    print('Num of paras : %d'%(get_num_params()))
    # print('target_path', args['target_path'])
    print('source_data', args['source_ntr'], args['source_nte'])
    # print('target_data', args['target_ntr'], args['target_nte'])
    data_save = SaveData()
    num_test = args['source_nte']
    err_array = data_save.save(sess, x_pos, fnn_model, W, b, Xmin, Xmax, u_B, f_ph, u_ph, data, num_test, save_results_to, domain='source')
    
    ## Plotting the loss history
    test_data  = test_loss[:,0]
    train_data = train_loss[:,0]
    
    nfont = 12
    lw = 1.8
    plt.figure(figsize=(4,2.5))   
     
    ax = plt.subplot()
    xx = np.linspace(0, len(test_data), len(test_data))
    
    plt.plot(xx, test_data, lw = lw, color = '#757575',ls='-', label = 'DeepONetBST test')
    plt.plot(xx, train_data, lw = lw, color = '#E2921B',ls='--', label = 'DeepONetBST train')
    plt.xlabel('Iteration', fontsize=nfont)
    plt.ylabel('Loss', fontsize=nfont)
    
    # if np.min(train_data) > np.min(test_data):
    #     min_value = np.min(test_data)
    # else:
    #     min_value = np.min(train_data)

    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    linewidth = 0.5
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)
    ax.tick_params(direction='out',  width=linewidth)
    
    plt.legend(loc='upper right', ncol=1, fontsize = 10, frameon=True, facecolor='white') #,frameon=False
    plt.show()
    # plt.savefig(figs_path+'/loss.svg',format='svg', bbox_inches='tight')
    plt.savefig(save_results_to+'/Loss.png', dpi=300, bbox_inches='tight')
    
    return err_array

if __name__ == "__main__":
    
    # data file
    geo_list = ['Geo_A', 'Geo_B']
    r_list = np.array([1299, 1300])
    source_geo = 1
    
    data_path = '../Data/' + geo_list[source_geo]
    lbo_path  = data_path + '/Nodes_LBO_basis'
    lbo_data = io.loadmat(lbo_path)
    nodes  = lbo_data['Points'][:,0:2]
    
    print('data_path:', data_path)
    data = io.loadmat(data_path + '/data_.mat')
    x_data = data["c_field_"]
    y_data = data["u_field"]
    
    args = dict()
    args['task_type']    = 'Train_source'
    args['source_path']  = data_path
    args['source_h']     = h # mesh resolution
    args['source_w']     = w # mesh resolution
    args['source_r']     = r_list[source_geo] # node number
    # args['source_ntr']   = 300
    args['source_nte']   = 200
    args['x_data']   = x_data
    args['y_data']   = y_data
    args['nodes']   = nodes
    args['batch_size'] = 16
    
    n_trains = [30, 50, 80, 1000]
    n_times = 3
    for n_train in n_trains:
        args['source_ntr'] = n_train 
        args['epochs']     = 500 # 50000
        current_directory = os.getcwd()    
        case = "../logs/"+geo_list[source_geo]+"/DeepONet_Ntr" + str(args['source_ntr'])\
               + '_Nte' + str(args['source_nte'])
               
        save_indexs = np.arange(0, n_times, 1)
        err_results = np.zeros((n_times, 4))
        
        for i, save_index in enumerate(save_indexs):
            
            save_results_to   = current_directory + "/" + case + '/' + str(save_index)
            save_variables_to = current_directory + "/" + case + '/' + str(save_index) +"/Variables"
            # Remove existing results
            if os.path.exists(save_results_to):
                shutil.rmtree(save_results_to)
                # shutil.rmtree(save_variables_to)
            os.makedirs(save_results_to) 
            os.makedirs(save_variables_to)
            tf.reset_default_graph()
            err_results[i] = main(args)
            
            txt_path = save_results_to + "/args.txt"
            with open(txt_path, "w") as f:
                # f.write(f"{'n_floders'}: {n_floders}\n")
                for key, value in args.items():
                    if key == 'x_data' or key == 'y_data' or key == 'nodes' or key == 'elems' or key == 'Data_modes':
                        continue  # 跳过 'data'
                    f.write(f"{key}: {value}\n")
        io.savemat(current_directory + "/" + case + '/err_results.mat', {'value': err_results})
        mean_err = np.mean(err_results, axis = 0)
        txt_path =  current_directory + "/" + case + '/log.txt'
        with open(txt_path, "w") as f:
            f.write(f"{'Test_loss'} : {mean_err[0]}\n")
            f.write(f"{'Test_MAE'}  : {mean_err[1]}\n")
            f.write(f"{'Test_MMax'} : {mean_err[2]}\n")
            f.write(f"{'Test_Max'}  : {mean_err[3]}\n")
            f.write("\n")
            for i in range(n_times):
                f.write(f"{'Error'}  : {err_results[i]}\n")

        print('\n**********************')
        print('\nMean error of '+str(n_times)+' run:')
        print('\nTesting error: %.4f'%(mean_err[0]))
        print('Testing MAError: %.4e'%(mean_err[1]))
        print('Testing MeanMax: %.4f'%(mean_err[2]))
        print('Testing MaxError: %.4f'%(mean_err[3]))
        print('\n')
        for i in range(n_times):
            print(f"{'Error'}  : {err_results[i]}\n")
    
   