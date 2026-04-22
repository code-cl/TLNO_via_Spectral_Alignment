'''
Manuscript Associated: Deep transfer operator learning for partial differential equations under conditional shift
Authors: Katiana Kontolati, PhD Candidate, Johns Hopkins University
         Somdatta Goswami, Postdoctoral Researcher, Brown University
Tensorflow Version Required: TF1.15     
This should be used for sharp data    

This is the target model. Run this after completing the simulation of the source model.
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
# from utils.loss import CEOD_loss
# import sys 
import math
print("You are using TensorFlow version", tf.__version__)
from functools import reduce
from operator import mul
def get_num_params():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params

# save_index = 1
# current_directory = os.getcwd()    
# case = "Case_"
# folder_index = str(save_index)
# results_dir = "/" + case + folder_index +"/Results"
# variable_dir = "/" + case + folder_index +"/Variables"
# save_results_to = current_directory + results_dir
# save_variables_to = current_directory + variable_dir

# np.random.seed(1234)
#tf.set_random_seed(1234)

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
# bs = 50

#size of input for Trunk net
# x_num = 2295 
##### For Rightangled triangle x_num = 1200
##### For equilateral triangle x_num = 2295
beta = 0.1
def main(args):
    x_num = args['target_r'] 
    bs = args['batch_size'] 
    ntr = args['target_ntr']
    nte = args['target_nte']
    # loss2 = CEOD_loss(x_num, bs)
    # Load data 
    data = DataSet(bs, args)
    x_train, f_train, u_train, Xmin, Xmax = data.minibatch_target()
    x_pos = tf.constant(x_train, dtype=tf.float32)
    x = tf.tile(x_pos[None, :, :], [bs, 1, 1]) #[bs, x_num, x_dim]
    
    # c1 = tf.Variable(tf.random_normal(shape = [1, 1], dtype = tf.float32), dtype = tf.float32)
    # c2 = tf.Variable(tf.random_normal(shape = [1, 1], dtype = tf.float32), dtype = tf.float32)
    
    # Load pre-trained variables (from source network)
    cnn_vars = io.loadmat(save_variables_to+'/CNN_vars.mat')
    fnn_vars = io.loadmat(save_variables_to+'/FNN_vars.mat')
    
    # Placeholders
    # fnn_layer_1_ph = tf.placeholder(shape=[None, layer_B[0]], dtype=tf.float32)
    # u_pred_ph_s    = tf.placeholder(shape=[None, x_num, 1], dtype=tf.float32)
    f_ph = tf.placeholder(shape=[None, h, w, n_channels], dtype=tf.float32) #[bs, 1, h, w, n_channels]
    u_ph = tf.placeholder(shape=[None, x_num, 1], dtype=tf.float32) #[bs, x_num, 1]
    learning_rate = tf.placeholder(tf.float32, shape=[])

    # Target branch net
    # CNN of branch net
    conv_model = CNN()

    #conv_linear = conv_model.linear_layer(f_ph, n_out_channels)
    conv_1 = conv_model.conv_layer_target(f_ph, cnn_vars['W1'], cnn_vars['b1'], stride, actn=tf.nn.relu)
    pool_1 = conv_model.avg_pool(conv_1, ksize=2, stride=2)   
    conv_2 = conv_model.conv_layer_target(pool_1, cnn_vars['W2'], cnn_vars['b2'], stride, actn=tf.nn.relu)
    pool_2 = conv_model.avg_pool(conv_2, ksize=2, stride=2) 
    conv_3 = conv_model.conv_layer_target(pool_2, cnn_vars['W3'], cnn_vars['b3'], stride, actn=tf.nn.relu)
    pool_3 = conv_model.avg_pool(conv_3, ksize=2, stride=2)
    conv_4 = conv_model.conv_layer_target(pool_3, cnn_vars['W4'], cnn_vars['b4'], stride, actn=tf.nn.relu)
    pool_4 = conv_model.avg_pool(conv_4, ksize=2, stride=2) 
    layer_flat = conv_model.flatten_layer(pool_4)

    # FNN of branch net
    fnn_layer_1, Wf1, bf1 = conv_model.fnn_layer_target(layer_flat, cnn_vars['Wf1'], cnn_vars['bf1'], actn=tf.tanh, use_actn=True)
    fnn_layer_2, Wf2, bf2 = conv_model.fnn_layer_target(fnn_layer_1, cnn_vars['Wf2'], cnn_vars['bf2'], actn=tf.nn.tanh, use_actn=True)
    out_B, Wf3, bf3 = conv_model.fnn_layer_target(fnn_layer_2, cnn_vars['Wf3'], cnn_vars['bf3'], actn=tf.tanh, use_actn=False) #[bs, p]
    u_B = tf.tile(out_B[:, None, :], [1, x_num, 1]) #[bs, x_num, p]
    
    # Trunk net
    Wt, bt = [fnn_vars['W1'], fnn_vars['W2'], fnn_vars['W3'], fnn_vars['W4']], [fnn_vars['b1'], fnn_vars['b2'], fnn_vars['b3'], fnn_vars['b4']]

    fnn_model = FNN()
    W, b = fnn_model.hyper_initial_target(layer_T, Wt, bt)
    u_T = fnn_model.fnn(W, b, x, Xmin, Xmax)

    u_nn = u_B*u_T
    u_pred = tf.reduce_sum(u_nn, axis=-1, keepdims=True)
    W_cnn_fnn = [Wf1] + [Wf2] + [Wf3] 
    regularizers = fnn_model.l2_regularizer(W_cnn_fnn)

    # c1_t = tf.math.sin(np.pi*c1*tf.exp(c1)) 
    #c2_t = tf.math.sin(np.pi*c2*tf.exp(c2))
    
    ############################
    # Loss function
    l2_loss = tf.reduce_sum(tf.norm(u_pred - u_ph, 2, axis=1)/tf.norm(u_ph, 2, axis=1)) +  beta*regularizers
    # ceod_loss = loss2.CEOD([fnn_layer_1], u_ph, [fnn_layer_1_ph], u_pred_ph_s)   
    
    hybrid_loss = l2_loss #+ 10*ceod_loss  # total loss
    optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1 = 0.99)
    # optimizer2 = tf.train.AdamOptimizer(learning_rate=5.0e-1, beta1 = 0.99, beta2 = 0.99)

    grads_W= optimizer1.compute_gradients(hybrid_loss, W[-1])
    grads_b = optimizer1.compute_gradients(hybrid_loss, b[-1])    
    grads_Wf1 = optimizer1.compute_gradients(hybrid_loss, Wf1)
    grads_bf1 = optimizer1.compute_gradients(hybrid_loss, bf1)
    grads_Wf2 = optimizer1.compute_gradients(hybrid_loss, Wf2)
    grads_bf2 = optimizer1.compute_gradients(hybrid_loss, bf2)
    grads_Wf3 = optimizer1.compute_gradients(hybrid_loss, Wf3)
    grads_bf3 = optimizer1.compute_gradients(hybrid_loss, bf3)
    # grads_c1 = optimizer2.compute_gradients(hybrid_loss, [c1])
    #grads_c2 = optimizer2.compute_gradients(hybrid_loss, [c2])
    
    # grads_c1_minus = [(-gv[0], gv[1]) for gv in grads_c1]
    #grads_c2_minus = [(-gv[0], gv[1]) for gv in grads_c2]
    
    op_W = optimizer1.apply_gradients(grads_W)
    op_b = optimizer1.apply_gradients(grads_b)
    op_Wf1 = optimizer1.apply_gradients(grads_Wf1)
    op_bf1 = optimizer1.apply_gradients(grads_bf1)
    op_Wf2 = optimizer1.apply_gradients(grads_Wf2)
    op_bf2 = optimizer1.apply_gradients(grads_bf2)
    op_Wf3 = optimizer1.apply_gradients(grads_Wf3)
    op_bf3 = optimizer1.apply_gradients(grads_bf3)    
    # op_c1 = optimizer2.apply_gradients(grads_c1_minus)
    # op_c2= optimizer2.apply_gradients(grads_c2_minus)

    # train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(hybrid_loss)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    print('Num of paras : %d'%(get_num_params()))
    data_save = SaveData()
    num_test = args['target_nte']
    data_save.save(sess, x_pos, fnn_model, W, b, Xmin, Xmax, u_B, f_ph, u_ph, data, num_test, save_results_to, domain='target_initial')
    
    n = 0
    nmax = args['epochs']  # epochs
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
            
            x_train, f_train, u_train, _, _ = data.minibatch_target(batch_id) # target data
            # If trained with CEOD uncomment the lines below
            # x_train_source, f_train_source, u_train_source, _, _ = data.minibatch() # source data
            # fnn_layer_1_s = sess.run(fnn_layer_1, feed_dict={f_ph:f_train_source}) 
            # u_pred_s = sess.run(u_pred, feed_dict={f_ph:f_train_source}) 
            # train_dict = {f_ph: f_train, fnn_layer_1_ph: fnn_layer_1_s, u_ph: u_train, u_pred_ph_s: u_pred_s, learning_rate: lr}
            train_dict = {f_ph: f_train, u_ph: u_train, learning_rate: lr}
            sess.run([op_W,op_b,op_Wf1,op_bf1,op_Wf2,op_bf2,op_Wf3,op_bf3], train_dict)
            #sess.run([op_c2], train_dict)
            loss_, u_train_ = sess.run([hybrid_loss, u_pred], feed_dict=train_dict)
            # Calculate train l2 error
            u_real  = data.decoder_target(u_train)
            u_real_ = data.decoder_target(u_train_)
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
            
                test_id, x_test, f_test, u_test = data.testbatch_target(bs, batch_id)
                u_test_ = sess.run(u_pred, feed_dict={f_ph: f_test})
                u_test  = data.decoder_target(u_test)
                u_test_ = data.decoder_target(u_test_)   
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
    print('Total run time: ', stop_time)
    
    loss_dict = {'train_error':train_loss.flatten(),
                 'test_error' :test_loss.flatten()}
    io.savemat(save_results_to +'/DeepONetBST_loss.mat', mdict = loss_dict)
    
    # Save data
    print('source_path', source_results_to)
    print('target_path', args['target_path'])
    # print('source_data', args['source_ntr'], args['source_nte'])
    print('target_data', args['target_ntr'], args['target_nte'])
    print('Num of paras : %d'%(get_num_params()))
    data_save = SaveData()
    num_test = args['target_nte']
    err_array = data_save.save(sess, x_pos, fnn_model, W, b, Xmin, Xmax, u_B, f_ph, u_ph, data, num_test, save_results_to, domain='target')

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
    
    geo_list = ['Geo_A', 'Geo_B']
    r_list = np.array([1299, 1300])
    source_geo = 0
    target_geo = 1
    
    data_path = '../Data/' + geo_list[target_geo]
    lbo_path  = data_path + '/Nodes_LBO_basis'
    lbo_data = io.loadmat(lbo_path)
    nodes  = lbo_data['Points'][:,0:2]
    
    print('data_path:', data_path)
    data = io.loadmat(data_path + '/data_.mat')
    x_data = data["c_field_"]
    y_data = data["u_field"]
        
    args = dict()
    args['task_type']    = 'Parameter_finetune'
    # args['source_path']  = '../Data/' + geo_list[source_geo] + '/data'
    # args['source_r_bc']  = s_bc # mesh resolution
    # args['source_r']     = r_list[source_geo] # node number
    # args['x_data']   = x_data
    # args['y_data']   = y_data
    # args['x_label']   = 'c_field_'
    # args['y_label']   = 'u_field'
    
    args['target_path']  = data_path
    # args['target_r_bc']  = s_bc # mesh resolution
    args['target_h']     = h # mesh resolution
    args['target_w']     = w # mesh resolution
    args['target_r']     = r_list[target_geo] # node number
    args['x_data']   = x_data
    args['y_data']   = y_data
    args['nodes']    = nodes
    # args['target_ntr']   = 30
    args['target_nte']   = 200
    args['batch_size'] = 16
    args['epochs']     = 500
    args['beta'] = beta
    
    source_save_index = 1
    current_directory = os.getcwd()    
    
    n_trains = [30,50,80]
    n_times = 3
    for n_train in n_trains:
        args['target_ntr'] = n_train 
    
        save_indexs = np.arange(0, n_times, 1)
        err_results = np.zeros((n_times, 4))
        target_save_path = "../logs/"+geo_list[source_geo]+'_'+geo_list[target_geo] + '/DeepONet_Ntr' + str(args['target_ntr'])\
                           + '_Nte' + str(args['target_nte'])

        for i, save_index in enumerate(save_indexs):
            source_folder = "../logs/"+geo_list[source_geo]+"/DeepONet_Ntr1000_Nte200/"+ str(save_index)
            variable_dir = source_folder +"/Variables"
            source_results_to = source_folder
            save_variables_to = variable_dir
            save_results_to   = target_save_path + '/' + str(save_index)
            args['source_model'] = source_results_to
            if os.path.exists(save_results_to):
                import shutil
                shutil.rmtree(save_results_to)
            os.makedirs(save_results_to) 
            tf.reset_default_graph()
            err_results[i] = main(args)
            txt_path = save_results_to + "/args.txt"
            with open(txt_path, "w") as f:
                # f.write(f"{'n_floders'}: {n_floders}\n")
                for key, value in args.items():
                    if key == 'x_data' or key == 'y_data' or key == 'nodes' or key == 'elems' or key == 'Data_modes':
                        continue  # 跳过 'data'
                    f.write(f"{key}: {value}\n")

        io.savemat(target_save_path + '/err_results.mat', {'value': err_results})
        mean_err = np.mean(err_results, axis = 0)
        txt_path =  target_save_path + '/log.txt'
        with open(txt_path, "w") as f:
            f.write(f"{'Test_loss'} : {mean_err[0]}\n")
            f.write(f"{'Test_MAE'}  : {mean_err[1]}\n")
            f.write(f"{'Test_MMax'} : {mean_err[2]}\n")
            f.write(f"{'Test_Max'}  : {mean_err[3]}\n")
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