import numpy as np
from numpy.linalg import eig as eig

def get_pairwise_loss(model_weights, i, j, FLAGS):

    small = np.min([i, j])
    large = np.max([i, j])
    
    ## Compute the index of the dual variables 

    w_fc1  = model_weights["W_FC1"]
    b_fc1 = model_weights["B_FC1"]
    w_fc2 = model_weights["W_FC2"]
    b_fc2 = model_weights["B_FC2"]
    c = model_weights["dual"]

    w_i = w_fc2[small, :]
    w_j = w_fc2[large, :]

    w_ij = w_i - w_j;


    diag_w = np.diag(np.ravel(w_ij))
    matrix_w = np.matmul(diag_w, np.transpose(w_fc1))
    vec_w = np.sum(matrix_w, axis = 0)
    


    final_w1 = np.concatenate( [ np.zeros([ FLAGS.num_hidden, FLAGS.num_hidden]), matrix_w], 1);
    final_w2 = np.concatenate( [np.transpose(matrix_w), np.zeros([FLAGS.dimension, FLAGS.dimension])], 1);
    final_w_small = np.concatenate([final_w1, final_w2], 0)

    
    ### Getting the symmetric version

    col = np.concatenate( [np.zeros([FLAGS.num_hidden, 1]), np.reshape(vec_w, [FLAGS.dimension, 1])], 0)

    final_w1 = np.concatenate( [ np.reshape(col, [FLAGS.num_hidden + FLAGS.dimension, 1]), final_w_small], 1)
    col2 = np.concatenate( [ np.zeros([FLAGS.num_hidden + 1, 1]), np.reshape(vec_w, [FLAGS.dimension, 1])], 0)
    final_w = np.concatenate( [np.reshape(col2, [1, FLAGS.num_hidden + FLAGS.dimension + 1]), final_w1], 0)
                                                                        

    ### Retrieving the correct dual variables 
   
    index = int((2*FLAGS.num_classes - 1 - small)*small*0.5 + large - small) - 1

    c_ij = c[index, :]
    new_c_ij = FLAGS.scale_dual*c_ij
    dual_mat = final_w - np.diag(np.reshape(new_c_ij, [FLAGS.dimension + FLAGS.num_hidden + 1]))


    #computing top eig of dual_mat

    eig_val, eig_vec  = eig(dual_mat)
    top_eig_val = np.max(eig_val)
    dual_value = (1 + FLAGS.dimension + FLAGS.num_hidden)*np.max(top_eig_val, 0) + np.sum(np.maximum(new_c_ij, 0))
    
    return dual_value
