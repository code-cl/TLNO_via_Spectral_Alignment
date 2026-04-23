'''
Authors: Katiana Kontolati, PhD Candidate, Johns Hopkins University
         Somdatta Goswami, Postdoctoral Researcher, Brown University
Tensorflow Version Required: TF1.15     
'''
import tensorflow.compat.v1 as tf
import numpy as np
import sys
# from utils.fnn import FNN
from utils.plotting import *
import os
import scipy
from sklearn.metrics import mean_absolute_error

class SaveData:
    def __init__(self):
        pass

    def save(self, sess, x_pos, fnn_model, W, b, Xmin, Xmax, u_B, f_ph, u_ph, data, num_test, save_results_to, domain):
        
        domain = domain
        # save_results_to = save_results_to +"/" + domain        
        if not os.path.exists(save_results_to):
            os.makedirs(save_results_to)
            
        if domain == 'source':
            # test_id, x_test, f_test, u_test = data.testbatch(num_test)
            x_test, f_test, u_test = data.testall()
        else:
            # test_id, x_test, f_test, u_test = data.testbatch_target(num_test)
            x_test, f_test, u_test = data.testall_target()
        
        print('Start testing...')
        print('x_test:', x_test.shape)
        print('f_test:', f_test.shape)
        print('u_test:', u_test.shape)
        
        x = tf.tile(x_pos[None, :, :], [num_test, 1, 1])
        u_T = fnn_model.fnn(W, b, x, Xmin, Xmax)
        test_dict = {f_ph: f_test, u_ph: u_test}
        u_nn = u_B*u_T        
        u_pred = tf.reduce_sum(u_nn, axis=-1, keepdims=True)

        u_pred_ = sess.run(u_pred, feed_dict=test_dict)
        if domain == 'source':
            u_test = data.decoder(u_test)
            u_pred_ = data.decoder(u_pred_)
        else:
            u_test = data.decoder_target(u_test)
            u_pred_ = data.decoder_target(u_pred_)
        
        # f_test  = np.reshape(f_test, (f_test.shape[0], -1))
        u_pred_ = np.reshape(u_pred_, (u_test.shape[0], u_test.shape[1]))

        u_ref = np.reshape(u_test, (u_test.shape[0], u_test.shape[1]))

        err_u = np.mean(np.linalg.norm(u_pred_ - u_ref, 2, axis=1)/np.linalg.norm(u_ref, 2, axis=1))            
        maxes = np.max(np.abs(u_pred_ - u_ref), axis=1)
        maxe  = np.max(maxes)
        mmaxe = np.mean(maxes)
        mae   = mean_absolute_error(u_pred_.ravel(), u_ref.ravel())
        if domain == 'source':
            print('Source model')
            
            err_u = np.reshape(err_u, (-1, 1))
            np.savetxt(save_results_to+'/err_l2', err_u, fmt='%e')       
            scipy.io.savemat(save_results_to+'/DeepONet_result.mat', 
                         mdict={'x_test': f_test,
                                'y_test': u_ref, 
                                'pre_test': u_pred_})
            
        elif domain == 'target_initial':
            print('Target result with source parameters')
            err_u = np.reshape(err_u, (-1, 1))
            np.savetxt(save_results_to+'/err_source', err_u, fmt='%e')       
            scipy.io.savemat(save_results_to+'/Source_DeepONet_result.mat', 
                         mdict={'x_test': f_test,
                                'y_test': u_ref, 
                                'pre_test': u_pred_})
        else:
            print('Target result after parameter finetuning')
            err_u = np.reshape(err_u, (-1, 1))
            np.savetxt(save_results_to+'/err_target', err_u, fmt='%e')       
            scipy.io.savemat(save_results_to+'/DeepONet_result.mat', 
                         mdict={'x_test': f_test,
                                'y_test': u_ref, 
                                'pre_test': u_pred_})
            
        print('Relative L2 Error: %.5f'%(err_u))
        print('MAError: %.4e'%(mae))
        print('Mean Maximum Error: %.5f'%(mmaxe))
        print('Maximum Error: %.5f'%(maxe))  
        
        return err_u, mae, mmaxe, maxe
