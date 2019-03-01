from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.framework import ops
import time

import numpy as np
# from numpy.linalg import eigh as eig
from scipy.sparse.linalg import eigsh as eig
from scipy.sparse.linalg import ArpackNoConvergence

import os
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import scipy.sparse as sparse
import argparse
import sys
import json
import scipy.io as sio


### Model definition
from utils.get_model import get_model 

## GD attack
from attacks.gd import gd, train_gd
from attacks.so_pgd import train_so_pgd

# Calculate sigma' for softplus
def sigma_prime (t, beta):
    return beta * tf.nn.sigmoid(beta * t)

# Calculate gradient at x
def grad_2 (x, a, beta):
    return (beta**2) * tf.exp(-beta*(a+x)) / (x * (1.0+tf.exp(-beta*(a+x)))**2) - (sigma_prime(x+a, beta)-sigma_prime(a, beta)) / (x**2)

# Hacky upper bound, x < 0
def hacky_ub (x):
    # For beta = 5 only
    a, b, c, d, e = [1.27167784, -0.15555676, 0.46238266, 5.64449646, 1.55300805]
    return a/(x*x-b*x+c) + d/(-x+e)

FULL_EIG =0


def my_py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncEig' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def fast_eig_vec(x):    
    if(FULL_EIG == 0):
        try:
            [u, s] = eig(x, k = 1, which = 'LA', maxiter = 50000, tol=1E-4)
            return s
        except ArpackNoConvergence as e:
            print("Computing full")
            [E, V] = np.linalg.eigh(x)
            ind = np.argmax(E)
            return V[ind, :]
        
    else:
        [E, V] = np.linalg.eigh(x)
        ind = np.argmax(E)
        return V[ind, :]
        

def fast_eig(x):
    if(FULL_EIG == 0):
        try :
            [u, s] = eig(x, k = 1, which = 'LA', tol=1E-4, maxiter = 50000)
            return u
        except ArpackNoConvergence as e:
            print("Computing full")
            [E, V] = np.linalg.eigh(x)
            return np.max(E)
    else:
        
        [E, V] = np.linalg.eigh(x)
        return np.max(E)
        

    
def my_max_eig(x, name=None):    
    with ops.op_scope([x], name, "Max_Eig") as name:
        max_eig = my_py_func(fast_eig,
                        [x],
                        [tf.float32],
                        name=name,
                        grad=_MyMaxEigGrad)  # <-- here's the call to the gradient
        return max_eig[0]
        

def _MyMaxEigGrad(op, grad):
    x_op = op.inputs[0]
    output = tf.py_func(fast_eig_vec, [x_op], [tf.float32])[0]
    res_output = tf.reshape(output, [tf.size(output), 1])
    grad_output= tf.matmul(res_output, tf.transpose(res_output))
    return grad*grad_output


def compute_loss(Y_test, Logits, Bounds_matrix, epsilon, FLAGS):
## For every datapoint, compute hinge loss for max
    num_points = np.shape(Logits)[0]
    num_classes = FLAGS.num_classes
    Fo_loss = np.zeros([num_points, 1])
    Hinge_loss = np.zeros([num_points, 1])
    New_labels = np.zeros([num_points, 1])

    for i in range(num_points):
        ## Compute the label 
        label = np.argmax(Y_test[i, :])
        ## Compute max f(j) - f(true) after perturbation

        new_values = np.ravel(Logits[i, :]) - np.ravel(np.ones([num_classes, 1])*Logits[i, label]) + epsilon*np.ravel(Bounds_matrix[label, :])
        
        max_new_val = np.max(new_values)
        new_label = np.argmax(new_values)
        New_labels[i] = new_label

        if(max_new_val > 0):
            Fo_loss[i] = 1
        
        Hinge_loss[i] = max(0, 1 + max_new_val)
    
            
    
    return Fo_loss, Hinge_loss, New_labels

def get_classification_loss(sess, x, y_, y_model , allvars,tvars, FLAGS):

    ## Classification loss part 
    if(FLAGS.loss == 'hinge'):
        H = tf.reduce_max(y_model * (1 - y_), 1)
        H = tf.reshape(H, [-1, 1])
        H = tf.tile(H, [1, FLAGS.num_classes])
        
        L = tf.nn.relu((1 - y_model + H) * y_)
        return tf.reduce_mean(tf.reduce_max(L, 1))

    elif (FLAGS.loss == 'cross-entropy'):
        softmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_model))     
        return softmax_loss

## Computes the pairwise loss given classes i and j 
def get_pairwise_loss(y_, y_model, i, j, allvars, tvars, weights, FLAGS):

    small = np.min([i, j])
    large = np.max([i, j])
    
    ## Compute the index of the dual variables 

    w_fc1  = tvars[0]
    b_fc1 = tvars[1]
    w_fc2 = tvars[2]
    b_fc2 = tvars[3]
    c = tvars[4]


    w_i = tf.slice(w_fc2, [0, small], [FLAGS.num_hidden, 1]) 
    w_j = tf.slice(w_fc2, [0, large], [FLAGS.num_hidden, 1]) 
    w_ij = tf.subtract(w_i, w_j)

    diag_w = tf.diag(tf.reshape(w_ij, [FLAGS.num_hidden]))
    matrix_w = tf.matmul( diag_w, tf.transpose(w_fc1))
    vec_w = tf.reduce_sum(matrix_w, 0)

    ### Getting the symmetric version
    final_w1 = tf.concat( [tf.zeros([ FLAGS.num_hidden,  FLAGS.num_hidden]), matrix_w], 1)
    ## Adding the vec portion
    final_w2 = tf.concat( [tf.transpose(matrix_w), tf.zeros([FLAGS.dimension, FLAGS.dimension])], 1)
    final_w_small = tf.concat( [final_w1, final_w2], 0)
    
    col = tf.concat( [tf.zeros([FLAGS.num_hidden, 1]), tf.reshape(vec_w, [FLAGS.dimension, 1])], 0)
    final_w1 = tf.concat( [tf.reshape(col, [FLAGS.num_hidden + FLAGS.dimension, 1]), final_w_small], 1)
    col2 = tf.concat( [tf.zeros([FLAGS.num_hidden+1, 1]), tf.reshape(vec_w, [FLAGS.dimension, 1])], 0)
    final_w = tf.concat( [ tf.reshape(col2, [1, FLAGS.num_hidden + FLAGS.dimension + 1]), final_w1], 0)

    ### Retrieving the correct dual variables 
   
    index = int((2*FLAGS.num_classes - 1 - small)*small*0.5 + large - small) - 1


    c_ij = tf.slice(c , [index, 0], [1, FLAGS.dimension + FLAGS.num_hidden + 1])
    new_c_ij = FLAGS.scale_dual*c_ij

    dual_mat = final_w - tf.diag(tf.reshape(new_c_ij, [FLAGS.dimension + FLAGS.num_hidden + 1]))
    
    
    dual_loss = (1 + FLAGS.dimension + FLAGS.num_hidden)*tf.maximum(my_max_eig(dual_mat), 0) + tf.reduce_sum(tf.maximum(new_c_ij, 0))
    
        
    ## New version based on different weights 
    logits_i = tf.slice(y_model, [0, small], [FLAGS.batch_size, 1])
    logits_j = tf.slice(y_model, [0, large], [FLAGS.batch_size, 1])
    
    label_i = tf.slice(y_, [0, small], [FLAGS.batch_size, 1])
    label_j = tf.slice(y_, [0, large], [FLAGS.batch_size, 1])
    
    w = tf.gather_nd(weights, [small, large])
    weighted_loss = tf.multiply(dual_loss, w)
    
    return dual_loss, weighted_loss, dual_mat

def get_so_pairwise_loss(x, y_, y_model, i, label, grad_fx, a_pos, a_neg, sig_pri_2_max, sig_pri_2_min, w_i2, allvars, tvars, weights, FLAGS):
    # grad_fx: ? k * 10 j * 784 u
    # label: ?
    # indexed_label: ? * 2 [[0, label_0], [1, label_1], ...]
    # grad_fx_j: ? * 784
    indexed = tf.range(tf.size(label))
    indexed = tf.reshape(indexed, [tf.size(label), 1])
    label = tf.reshape(label, [tf.size(label), 1])
    indexed_label = tf.concat([indexed, label], 1)
    grad_fx_j = tf.gather_nd(grad_fx, indexed_label)
    
    i_exp = tf.ones([tf.size(label), 1], dtype = tf.int32) * i
    indexed_i = tf.concat([indexed, i_exp], 1)
    grad_fx_i = tf.gather_nd(grad_fx, indexed_i)
    
    grad_fx_ij = tf.subtract(grad_fx_i, grad_fx_j)
    grad_fx_ij = tf.reshape(grad_fx_ij, [-1, FLAGS.dimension])
    # grad_fx_ij: ? * 784
    # sig_pri_2_max: ? i * 500 j
    # a_pos_ij: ? i * 500 j
    
    i_exp = tf.ones([tf.size(label), 1], dtype = tf.int32) * i
    indexed_ij = tf.concat([i_exp, label], 1)
    a_pos_ij = tf.gather_nd(a_pos, indexed_ij)
    a_neg_ij = tf.gather_nd(a_neg, indexed_ij)
    # w_i2: 784 k * 784 m * 500 j
    
    # Compute PSD_M: ? i * 784 k * 784 m
    psd_M = tf.add(tf.einsum('ij,kmj->ikm', tf.multiply(sig_pri_2_max, a_pos_ij), w_i2), tf.einsum('ij,kmj->ikm', tf.multiply(sig_pri_2_min, a_neg_ij), w_i2))
   
    # Do PGD on epsilon to find local max of upper bound
    per, loss = train_so_pgd (x, y_, grad_fx_ij, psd_M, FLAGS.train_epsilon, FLAGS)
    # per: ? * 784
    # loss: ?
    print('get_so_pairwise_loss', i)
   
    #w = tf.gather_nd(weights, [i, j])
    #weighted_loss = tf.multiply(dual_loss, w)
    
    return loss
    
def get_class_loss(sess, y_, y_model, i, allvars, tvars, weights, FLAGS):

    loss = tf.Variable(0.0, name= "reg_loss" + str(i), trainable = False)
    sess.run(loss.initializer)
    class_matrices = []
    class_losses = []
    for j in range(0, FLAGS.num_classes):
        if(j==i):
            continue 
        unweighted_loss, weighted_loss, pairwise_matrix = get_pairwise_loss(y_, y_model, i, j, allvars, tvars, weights, FLAGS)
        loss = tf.add(loss, weighted_loss)
        class_losses.insert(len(class_losses), unweighted_loss)
        class_matrices.insert(len(class_matrices), pairwise_matrix)

    return loss, class_losses, class_matrices

def get_so_class_loss(sess, x, y_, y_model, i,  grad_fx, a_pos, a_neg, sig_pri_2_max, sig_pri_2_min, w_i2, allvars, tvars, weights, FLAGS):
    
    # y_: ? * 10
    # label: ?
    label = tf.cast(tf.argmax(y_, axis = 1), dtype = tf.int32)
   
    return get_so_pairwise_loss(x, y_, y_model, i, label, grad_fx, a_pos, a_neg, sig_pri_2_max, sig_pri_2_min, w_i2, allvars, tvars, weights, FLAGS)

def get_regularization_loss(sess, x, y_, y_model, allvars, tvars, weights, FLAGS):

    if (FLAGS.reg_type == 'first_order'):
        
        print('Training with first order regularization')
        final_loss = []
        final_matrices = []
        unweighted_losses = []
        
        print(x.shape)

        for r in range(0, FLAGS.num_classes):
            class_loss, loss_list, matrix_list = get_class_loss(sess, y_, y_model, r,  allvars,tvars, weights, FLAGS)
            final_loss.insert(len(final_loss), class_loss)
            for m in matrix_list:
                final_matrices.insert(len(final_matrices), m)
            for l in loss_list:
                unweighted_losses.insert(len(unweighted_losses), l)

        return final_loss, unweighted_losses, final_matrices 
    
    if (FLAGS.reg_type == 'fro'):
        w_fc1  = tvars[0]
        b_fc1 = tvars[1]
        w_fc2 = tvars[2]
        b_fc2 = tvars[3]
        fro_loss = (tf.norm(w_fc1) + tf.norm(w_fc2))
        
        fro_list = []
        for i in range(10):
            fro_list.insert(len(fro_list), fro_loss)
        return fro_list, tf.zeros([10, 10]),  tf.zeros([10, 10]) 
        


    if (FLAGS.reg_type == 'spectral'):
        w_fc1  = tvars[0] 
        w_fc2 = tvars[2]
        matrix = tf.add(tf.matmul(tf.transpose(w_fc1), w_fc1), tf.random_uniform( [FLAGS.num_hidden, FLAGS.num_hidden], minval = -1*FLAGS.noise_param, maxval = FLAGS.noise_param, dtype = tf.float32) )
        m = my_max_eig(matrix)

        l2 = tf.norm(w_fc2)
        spectral_loss = (tf.norm(w_fc2) + m)
        spectral_list = []
        for i in range(10):
            spectral_list.insert(len(spectral_list), spectral_loss)
        return spectral_list, tf.zeros([10 , 10]), tf.zeros([10, 10]) 


    if(FLAGS.reg_type == 'FGSM'):
        original_loss = get_classification_loss(sess, x, y_, y_model, allvars, tvars,FLAGS)
        grad, = tf.gradients(original_loss, x);
        signed_grad = tf.sign(grad)*FLAGS.train_epsilon
        x_adv = x + signed_grad 
        with tf.variable_scope("model_weights") as scope:
            scope.reuse_variables()
            y_adv = get_model(x_adv, FLAGS)
            scope.reuse_variables()
        
    if(FLAGS.reg_type == 'gd'):
        x_adv = train_gd(x, y_, FLAGS.train_epsilon, FLAGS)
        with tf.variable_scope("model_weights") as scope:
            scope.reuse_variables()
            y_adv = get_model(x_adv, FLAGS)
            scope.reuse_variables()
        
        adv_loss = get_classification_loss(sess, x, y_, y_adv, allvars, tvars,FLAGS)
        adv_list = []
        for i in range(FLAGS.num_classes):
            adv_list.insert(len(adv_list), adv_loss)
        return adv_list, tf.zeros([FLAGS.num_classes, FLAGS.num_classes]),tf.zeros([FLAGS.num_classes, FLAGS.num_classes]) 

    if(FLAGS.reg_type == 'so_pgd'):
        print('Training with second-order pgd')
        
        beta = 5.0
        
        # perturb x to x_adv, do pgd on epsilon to find a local max of upper bound
        w_fc1  = tvars[0]
        b_fc1 = tvars[1]
        w_fc2 = tvars[2]
        b_fc2 = tvars[3]
        
        # Preliminary work for Hessian term
        # Calculate W_j W_j^T
        w_i2 = tf.einsum('ij,kj->ikj', w_fc1, w_fc1)
        # Calculate l1 norm of each row of W_1
        b_0 = tf.norm(w_fc1, ord = 1, axis = 0)
        b_0 = b_0 * FLAGS.train_epsilon
       
        # Compute gradient of f(x)
        # tf.reshape(tf.matmul(tf.reshape(Aijk,[i*j,k]),Bkl),[i,j,l])
        # x: ? * 784
        # w_fc1: 784 u * 500 i
        # b_fc1: 500
        # w_fc2: 500 i * 10 j
        # pre_act: ? * 500
        # sigma_p: ? k * 500 i
        # grad_fx: ? k * 10 j * 784 u
        pre_act = tf.add(tf.matmul(x, w_fc1), b_fc1)
        sigma_p = sigma_prime(pre_act, beta)
        #grad_fx = tf.einsum('ij,ki->kju', w_fc2, sigma_p)
        grad_fx = tf.einsum('ij,ki,ui->kju', w_fc2, sigma_p, w_fc1)
        
        pre_act_neg = -tf.abs(pre_act)
        
        # Max and min of integral of sigma''
        grad_b0 = tf.nn.relu(tf.sign(grad_2(b_0, pre_act_neg, beta))) # Indicator
        
        # Max go from pre_act_neg to opt unless grad_b0 > 0
        candidate_1 = (sigma_prime(pre_act_neg + b_0, beta) - sigma_prime(pre_act_neg, beta)) / b_0
        candidate_2 = hacky_ub(pre_act_neg)
        # sig_pri_2_max: ? * 500
        sig_pri_2_max = candidate_1 * grad_b0 + candidate_2 * (1 - grad_b0)
        
        # Min go from pre_act - b_0 to pre_act
        sig_pri_2_min = (sigma_prime(pre_act_neg, beta) - sigma_prime(pre_act_neg - b_0, beta)) / b_0
        temp_a = tf.zeros([0, FLAGS.num_classes, FLAGS.num_hidden], dtype=tf.float32)
        for label in range(FLAGS.num_classes):
            temp_a_2 = tf.zeros([1, 0, FLAGS.num_hidden], dtype=tf.float32)
            for k in range(FLAGS.num_classes):
                w_i = tf.slice(w_fc2, [0, k], [FLAGS.num_hidden, 1])
                w_j = tf.slice(w_fc2, [0, label], [FLAGS.num_hidden, 1])
                temp_a_2 = tf.concat([temp_a_2, tf.expand_dims(tf.transpose(tf.subtract(w_i, w_j)), 0)], 1)
            temp_a = tf.concat([temp_a, temp_a_2], 0)
        # 10 * 10 * 500
        a_pos = tf.nn.relu(temp_a)
        a_neg = -tf.nn.relu(-temp_a)
        
        final_loss = tf.zeros([tf.shape(x)[0], 0])

        final_matrices = []
        unweighted_losses = []

        # final loss: ? * num_classes
        for r in range(0, FLAGS.num_classes):
            # class loss: ?
            class_loss = get_so_class_loss(sess, x, y_, y_model, r, grad_fx, a_pos, a_neg, sig_pri_2_max, sig_pri_2_min, w_i2, allvars,tvars, weights, FLAGS)
            class_loss = tf.reshape(class_loss, [tf.size(class_loss), 1])
            final_loss = tf.concat([final_loss, class_loss], 1)

        return final_loss, tf.zeros([FLAGS.num_classes, FLAGS.num_classes]),tf.zeros([FLAGS.num_classes, FLAGS.num_classes])

"""
Train takes as input args that has the following:
1. args["dataset"] : Name of the dataset
2. args["num_classes"]: Number of classes to use from the dataset 
3. args["num_hidden"]: Number of hidden nodes
4. args["reg_type"]: 0 for no regularization, 1 for frobenius norm, 2 for L1 norm, 3 for FO norm
5. args["reg_param"]: Amount of regularization 
6. args["save_path"]: Path to save the model (tensorflow format) -- can be restored 
7. args["learning_rate"]: For Adam 
8. args["beta1"]: For Adam 
9. args["beta2"]: For Adam 
10. args["batch_size"]: Batch size for train5Bing
"""

def train(X_train, Y_train, X_test, Y_test, FLAGS):
    

    ## Y_train and Y_test are in one-hot encoding 
    ## X_train and X_test are in [ num_examples, dimension] format

    ## Defining the model 
    x = tf.placeholder(tf.float32, [None, FLAGS.dimension])
    y_= tf.placeholder(tf.float32, [None, FLAGS.num_classes])
    weights = tf.placeholder(tf.float32, [FLAGS.num_classes, FLAGS.num_classes])

    
    with tf.variable_scope("model_weights") as scope:
        y_model = get_model(x, FLAGS)
        scope.reuse_variables()

    # Defining all possible variables that could be used while training 
    with tf.variable_scope("dual") as scope: 
        num_dual_variables = int(FLAGS.num_classes * (FLAGS.num_classes -1 )* 0.5)
        init = np.random.normal(0, FLAGS.sd, [num_dual_variables, FLAGS.num_hidden + FLAGS.dimension + 1])
        init[np.where(init< 0)] = 0
        init= np.float32(init)
        if(FLAGS.reg_type == 'first_order' and FLAGS.reg_param != 0):
            c = tf.get_variable("dual", dtype = tf.float32, initializer = init)
        else :
            c = tf.get_variable("dual", dtype = tf.float32, initializer = init, trainable = False)

         
    ## Call appropriate loss function based on regularization 

    tvars = tf.trainable_variables()
    allvars = tf.all_variables()
   
    config=tf.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.allow_growth=True
    sess = tf.InteractiveSession(config=config)

    saver = tf.train.Saver(max_to_keep=0)
    classification_loss = get_classification_loss(sess, x, y_, y_model, allvars, tvars, FLAGS)
    
    sess.run(tf.global_variables_initializer())
    correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    ## Running the training -- Saving all the variables after every 100 batch updates
    num_batches = int(np.floor(np.max(np.shape(X_train))/FLAGS.batch_size))
    

    if(os.path.isfile(FLAGS.msave + "-weights" + ".meta") != 0):
        print("Found weights; not training")
        return 
    else :
        print("training")
        if not os.path.exists(FLAGS.results_dir):
            os.makedirs(FLAGS.results_dir)
            

        with open(os.path.join(FLAGS.results_dir, 'flags.json'), 'w') as f:

            print(json.dumps(tf.flags.FLAGS.__flags), end="", file=f)


    with sess.as_default():
        assert tf.get_default_session() is sess

        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        
        
        if(FLAGS.checkPoint!=None):
            checkPoint = FLAGS.checkPoint
            saver.restore(sess, checkPoint)
            print("Restore model at" + checkPoint)
            
            init = sess.run(tvars[4])
            init = init/FLAGS.scale_dual
            Val = tf.get_variable("val", dtype = tf.float32, initializer = init, trainable = False)        
            sess.run(Val.initializer)
            sess.run(tf.assign(tvars[4], Val))

        WEIGHTS = np.ones([FLAGS.num_classes, FLAGS.num_classes])    
                
        for j in range(FLAGS.restore_epoch, FLAGS.num_epochs):
            
            if(j == 0 or ((j == 20 or j == 40 or j == FLAGS.restore_epoch) and FLAGS.reg_type == 'first_order')):
  
                print("Set things up with lower learning rate")
                opt = tf.train.AdamOptimizer(learning_rate=FLAGS.alpha, beta1=FLAGS.beta1, beta2=FLAGS.beta2, epsilon = FLAGS.adam_epsilon)

                if (FLAGS.reg_type == 'so_pgd'):
                    print("Computing second-order regularized loss")
                    # first order plus second order loss
                    foso_loss, unweighted_list, regularization_matrices = get_regularization_loss(sess, x, y_, y_model,allvars, tvars, weights, FLAGS)
                    print("Taking softmax")
                    # Take softmax
                    regularization_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = tf.add(y_model, foso_loss)))
                    
                    train_loss = FLAGS.normal_param * classification_loss + FLAGS.reg_param * regularization_loss
                    train_step = opt.minimize(train_loss)
                    print('finished settup train_step')
                
                elif (FLAGS.reg_type == 'None' or FLAGS.reg_param == 0 ):
                    train_loss = classification_loss
                    train_step = opt.minimize(train_loss)
                else:
                    print("Computing regularized loss")
                    final_loss, unweighted_list, regularization_matrices = get_regularization_loss(sess, x, y_, y_model,allvars, tvars, weights, FLAGS)
                    if(FLAGS.loss == 'custom'):
                        train_loss = classification_loss
                        train_step = opt.minimize(train_loss)
                    else: 
                        Train_step = []
                        
                        for r in range(FLAGS.num_classes):
                            train_loss = FLAGS.normal_param*classification_loss + FLAGS.reg_param*final_loss[r]
                            Train_step.insert(len(Train_step), opt.minimize(train_loss))
                        
                adam_initializers = [var.initializer for var in tf.all_variables() if ('Adam' in var.name or 'beta' in var.name)]

                for a in adam_initializers:
                    sess.run(a)
                FLAGS.alpha = FLAGS.alpha *0.1 
                    
            print("EPOCH " + str(j))

            # Recomputing weights 
            if( (j %FLAGS.weights_change == 0 or (j == FLAGS.restore_epoch and FLAGS.weights_change!=1000)) and FLAGS.reg_type == "first_order"):
                print("Changing weights")
                Logits = sess.run(y_model, feed_dict={x:X_train, y_:Y_train})
                
                SDP_list = np.zeros(FLAGS.num_classes*(FLAGS.num_classes -1))
                for k_ in range(0, FLAGS.num_classes*(FLAGS.num_classes -1)):
                    SDP_list[k_] = sess.run(unweighted_list[k_])
                
               
                COUNTS = np.zeros([FLAGS.num_classes, FLAGS.num_classes]);
                VALUES = np.zeros([FLAGS.num_classes, FLAGS.num_classes]);
 
                for p in range(np.shape(X_train)[0]):
                    label = np.argmax(Y_train[p, :])
                    max_val = 0 
                    max_ind = -1
                    max_weights = -1
                    for l_ in range(FLAGS.num_classes):
                        if (l_ == label):
                            continue
                        f = np.abs(Logits[p, label] - Logits[p, l_])
                        small = np.min([l_, label])
                        large = np.max([l_, label])
                        ind = large*(FLAGS.num_classes - 1) + small
                        SDP_val = SDP_list[ind]
                        if (max_val <= SDP_list[ind]/f):
                            max_val = SDP_list[ind]/f
                            max_ind = l_
                            val = f

                    COUNTS[max_ind, label] = COUNTS[max_ind, label] + 1
                    COUNTS[label, max_ind] = COUNTS[label, max_ind] + 1
                    VALUES[max_ind, label] = VALUES[max_ind, label] + val 
                    VALUES[label, max_ind] = VALUES[label, max_ind] + val 

                
                #1. Weights option 1 : Simply fraction of times it's the maximising term 
                if(FLAGS.weights_option == 0):
                    WEIGHTS = np.ones([FLAGS.num_classes, FLAGS.num_classes])/np.sum(WEIGHTS)
                if(FLAGS.weights_option == 1):
                    WEIGHTS = COUNTS/60000;
                    WEIGHTS = WEIGHTS/np.sum(WEIGHTS)
                if(FLAGS.weights_option == 2):
                    WEIGHTS = np.divide(COUNTS, VALUES)
                    WEIGHTS[np.where(COUNTS == 0)] = 0
                    WEIGHTS = WEIGHTS/np.sum(WEIGHTS)
                if(FLAGS.weights_option == 3):
                    WEIGHTS = VALUES
                    WEIGHTS = np.divide(WEIGHTS, np.sum(WEIGHTS))
                
                """
                Since it's random weights to start with, don't want to use weights 
                """

                if(j == 0):
                    WEIGHTS = np.ones([FLAGS.num_classes, FLAGS.num_classes])/np.sum(WEIGHTS)
                    WEIGHTS = np.divide(WEIGHTS, np.sum(WEIGHTS))


            ## Shuffling the data in each epoch
            st = time.time()
            P = np.arange(np.max(np.shape(X_train)))
            np.random.shuffle(P)
            X_train = X_train[P]
            Y_train = Y_train[P]
            en = time.time()
            time_shuffle = (en - st)
               

            for i in range(num_batches):
                start = i*FLAGS.batch_size
                end = (i+1)*FLAGS.batch_size
                
                
                if i % 1 == 0:
                    Test_accuracy = accuracy.eval(feed_dict={
                        x: X_test, y_: Y_test})


                    if(FLAGS.reg_type != 'None' and FLAGS.reg_param!=0):
                        
                        Train_loss = train_loss.eval( feed_dict={x:X_train[0:FLAGS.batch_size, :], y_:Y_train[0:FLAGS.batch_size, :], weights:WEIGHTS})
                        Classification_loss = classification_loss.eval(feed_dict={x:X_train[0:FLAGS.batch_size, :], y_:Y_train[0:FLAGS.batch_size, :]})
                    
                        stats = {
                            'test accuracy': float(Test_accuracy),
                            'train loss': float(Train_loss),
                            'classification loss': float(Classification_loss)
                            }
                    
                        # class loss undefined for so_pgd
                        if (FLAGS.reg_type != 'so_pgd'):
                            Reg_loss = np.zeros([FLAGS.num_classes, 1])
                            for r in range(FLAGS.num_classes):
                                Reg_loss[r] = final_loss[r].eval( feed_dict = {x: X_train[0:FLAGS.batch_size, :], y_:Y_train[0:FLAGS.batch_size, :], weights:WEIGHTS})
                                stats['loss' + str(r)] = float(Reg_loss[r])
                        else:
                            Reg_loss = Train_loss - Classification_loss
                        
                        ### Saving: Classification loss, all ten reg losses 
                        
                        ### Fraction of test points that are misclassified at EPS = 0.1
                        
                        if(FLAGS.reg_type == "first_order"):
                            SDP_list = np.zeros(FLAGS.num_classes*(FLAGS.num_classes -1))
                            for k_ in range(0, FLAGS.num_classes*(FLAGS.num_classes -1)):
                                SDP_list[k_] = sess.run(unweighted_list[k_])
                            
                            SDP_matrix = np.zeros([FLAGS.num_classes, FLAGS.num_classes])
                            ### Creating the "FO" matrix 
                            for i_ in range(FLAGS.num_classes):
                                for j_ in range(i_):
                                    ind = i_*(FLAGS.num_classes - 1) + j_
                                    SDP_matrix[i_, j_] = SDP_list[ind]
                                    SDP_matrix[j_, i_] = SDP_list[ind]
                                    print(SDP_list[ind])
                                
                           
                            SDP_matrix = SDP_matrix*0.25;
                           


                            Logits = sess.run(y_model, feed_dict={x:X_test, y_:Y_test})
                           
                            
                            FO_loss_01, FO_loss_hinge, FO_new_labels = compute_loss(Y_test, Logits, SDP_matrix, FLAGS.train_epsilon, FLAGS)
                            
                            fo_loss = np.sum(FO_loss_01)/np.max(np.shape(Y_test))
                    
                            stats["fo_loss"] = float(fo_loss)

                            print('step %d, test accuracy %g, regularization loss %g, classification loss %g, FO loss %g' % (i, Test_accuracy, np.max(Reg_loss), Classification_loss, fo_loss))                        
                        elif (FLAGS.reg_type == "so_pgd"):
                            print('step %d, test accuracy %g, training loss %g, regularization loss %g, classification loss %g' % (i, Test_accuracy, Train_loss, Reg_loss, Classification_loss))
                        else: 
                            print('step %d, test accuracy %g, regularization loss %g, classification loss %g' % (i, Test_accuracy, np.max(Reg_loss), Classification_loss))

                        results_new = FLAGS.results_dir + "-epoch" + str(j) + "-step" + str(i)
                        if(i == 100):
                            if not os.path.exists(results_new):
                                os.makedirs(results_new)
                            with open(os.path.join(results_new, 'stats.json'), 'w') as f:
                                print(json.dumps(stats), end="", file=f)

                    else : 
                        print('step %d, test accuracy %g' % (i, Test_accuracy))
                    # Saving all the variables here (weight + dual)
                if(i == 100 or ( FLAGS.dataset=='har' and i == 10)):
                    checkPoint = FLAGS.msave + "-epoch" + str(j) + "-step" + str(i)
                    save_path = saver.save(sess, checkPoint)
                    print("Model saved in file: %s" % save_path)
                    
                 
                if ((FLAGS.reg_type != 'None') and (FLAGS.reg_type != 'so_pgd')):
                    # Running one of a list of train steps here
                    
                    sess.run(Train_step[i%FLAGS.num_classes], feed_dict={x: X_train[start:end, :], y_:Y_train[start:end, :], weights:WEIGHTS}, options = options, run_metadata=run_metadata)     

                else :
                    sess.run(train_step, feed_dict={x: X_train[start:end, :], y_:Y_train[start:end, :]}, options = options, run_metadata=run_metadata)



        final = FLAGS.msave + "-final"    
        save_path = saver.save(sess, final)
        print("Model saved in file: %s" % save_path)
        ## Saving the model weights 
        
        saver1 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model_weights'))
        weights_path = FLAGS.msave + "-weights"
        save_path = saver1.save(sess, weights_path)
        print("Weights saved in file: %s" % save_path)




