from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.framework import ops
import time

import numpy as np
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
#from attacks.so_pgd import train_so_pgd

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

def get_classification_loss(x, y_, y_model, FLAGS):

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

# Testing speed issue
#def get_so_class_loss(x, y_, y_model, i,  grad_fx, grad_fx_label, a_pos, a_neg, sig_pri_2_max, sig_pri_2_min, w_i2, FLAGS):
def get_so_class_loss(x, y_, y_model, i, grad_fx, grad_fx_label, FLAGS):
    
    # grad_fx: ? k * 10 j * 784 u
    # label: ?
    # indexed_label: ? * 2 [[0, label_0], [1, label_1], ...]
    # grad_fx_i: ? * 784
    indexed = tf.range(tf.shape(x)[0])
    indexed = tf.reshape(indexed, [tf.shape(x)[0], 1])
    
    i_exp = tf.ones([tf.shape(x)[0], 1], dtype = tf.int32) * i
    indexed_i = tf.concat([indexed, i_exp], 1)
    grad_fx_i = tf.gather_nd(grad_fx, indexed_i)
    
    grad_fx_ij = tf.subtract(grad_fx_i, grad_fx_label)
    grad_fx_ij = tf.reshape(grad_fx_ij, [-1, FLAGS.dimension])
    # grad_fx_ij: ? * 784
    # sig_pri_2_max: ? i * 500 j
    # a_pos_ij: ? i * 500 j
    """
    indexed_ij = tf.concat([i_exp, label], 1)
    a_pos_ij = tf.gather_nd(a_pos, indexed_ij)
    a_neg_ij = tf.gather_nd(a_neg, indexed_ij)
    """
    # w_i2: 784 k * 784 m * 500 j
    
    # Compute PSD_M: ? i * 784 k * 784 m
    ## psd_M = tf.add(tf.einsum('ij,kmj->ikm', tf.multiply(sig_pri_2_max, a_pos_ij), w_i2), tf.einsum('ij,kmj->ikm', tf.multiply(sig_pri_2_min, a_neg_ij), w_i2))
    
    #psd_M = tf.einsum('ij,kmj->ikm', 6.25 * a_pos_ij, w_i2)
    #psd_M = tf.expand_dims(6.25 * tf.reduce_sum(w_i2, axis = 2), axis = 0)
    #psd_M = tf.tile(psd_M, [tf.size(label), 1, 1])
    
    # Get perturbation variable value

    #with tf.variable_scope("perturbation", reuse = True):
    #    per = tf.get_variable("per")
    
    ## per_i = tf.gather_nd(per, indexed_i)
    # per_i: ? * 784
    ## loss = tf.add(tf.reduce_sum(tf.multiply(grad_fx_ij, per_i), 1), tf.einsum('ij,ijk,ik->i', per_i, psd_M, per_i))

    # Test speed issue... ignoring so term
    loss = tf.reduce_sum(grad_fx_ij, 1) * FLAGS.train_epsilon
    
    # per: ? * 10 * 784
    # loss: ?

    return loss

def get_regularization_loss(x, y_, y_model, FLAGS):
        
    if(FLAGS.reg_type == 'gd'):
        x_adv = train_gd(x, y_, FLAGS.train_epsilon, FLAGS)
        with tf.variable_scope("model_weights") as scope:
            scope.reuse_variables()
            y_adv = get_model(x_adv, FLAGS)
            scope.reuse_variables()
        
        adv_loss = get_classification_loss(x, y_, y_adv, FLAGS)
        return adv_loss, []

    if(FLAGS.reg_type == 'so_pgd'):
        print('Building graph with second-order pgd')
        
        beta = 5.0
        
        # perturb x to x_adv, do pgd on epsilon to find a local max of upper bound
        with tf.variable_scope("model_weights", reuse = True):
            w_fc1  = tf.get_variable("W_fc1")
            b_fc1 = tf.get_variable("b_fc1")
            w_fc2 = tf.get_variable("W_fc2")
            b_fc2 = tf.get_variable("b_fc2")
        
        # Preliminary work for Hessian term
        # Calculate W_j W_j^T
        ## w_i2 = tf.einsum('ij,kj->ikj', w_fc1, w_fc1)
        # Calculate l1 norm of each row of W_1
        ## b_0 = tf.norm(w_fc1, ord = 1, axis = 0) * FLAGS.train_epsilon
       
        # Compute gradient of f(x)
        # x: ? * 784
        # w_fc1: 784 u * 500 i
        # b_fc1: 500
        # w_fc2: 500 i * 10 j
        # pre_act: ? * 500
        # sigma_p: ? k * 500 i
        # grad_fx: ? k * 10 j * 784 u
        
        pre_act = tf.add(tf.matmul(x, w_fc1), b_fc1)
        sigma_p = sigma_prime(pre_act, beta)
        grad_fx = tf.einsum('ij,ki,ui->kju', w_fc2, sigma_p, w_fc1)
        
        # y_: ? * 10
        # label: ?
        label = tf.cast(tf.argmax(y_, axis = 1), dtype = tf.int32)
        # indexed_label: ? * 2 [[0, label_0], [1, label_1], ...]
        # grad_fx_label: ? * 784
        indexed = tf.range(tf.size(label))
        indexed = tf.reshape(indexed, [tf.size(label), 1])
        label = tf.reshape(label, [tf.size(label), 1])
        indexed_label = tf.concat([indexed, label], 1)
        grad_fx_label = tf.gather_nd(grad_fx, indexed_label)

        """
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
        """
        final_loss = tf.zeros([tf.shape(x)[0], 0])
        class_margin = []

        # final loss: ? * num_classes
        for r in range(0, FLAGS.num_classes):
            # class loss: ?
            # class_loss = get_so_class_loss(x, y_, y_model, r, grad_fx, a_pos, a_neg, sig_pri_2_max, sig_pri_2_min, w_i2, FLAGS)
            
            # Testing speed issue
            class_loss = get_so_class_loss(x, y_, y_model, r, grad_fx, grad_fx_label, FLAGS)
            class_loss = tf.reshape(class_loss, [tf.size(class_loss), 1])
            final_loss = tf.concat([final_loss, class_loss], 1)
            class_margin.append(class_loss)

        return final_loss, class_margin

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
    ## X_train and X_test are in [num_examples, dimension] format

    ## Defining the model 
    x = tf.placeholder(tf.float32, [None, FLAGS.dimension])
    y_= tf.placeholder(tf.float32, [None, FLAGS.num_classes])
    
    with tf.variable_scope("model_weights") as scope:
        y_model = get_model(x, FLAGS)
        scope.reuse_variables()

    # Defining all possible variables that could be used while training 
    with tf.variable_scope("perturbation") as scope:
        per = tf.get_variable("per", shape = [FLAGS.batch_size, FLAGS.num_classes,  FLAGS.dimension], initializer = tf.zeros_initializer())
        # per: 100 * 10 * 784
        scope.reuse_variables()
         
    ## Call appropriate loss function based on regularization 

    config=tf.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.allow_growth=True
    sess = tf.InteractiveSession(config=config)

    saver = tf.train.Saver(max_to_keep=0)
    classification_loss = get_classification_loss(x, y_, y_model, FLAGS)

    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.alpha, beta1=FLAGS.beta1, beta2=FLAGS.beta2, epsilon = FLAGS.adam_epsilon)

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

        print("Set things up with lower learning rate")

        if (FLAGS.reg_type == 'so_pgd'):
            print("Computing second-order regularized loss")
            
            # first order plus second order loss
            foso_margin, class_margin = get_regularization_loss(x, y_, y_model, FLAGS)
            
            # Take softmax
            regularization_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = tf.add(y_model, foso_margin)))
            
            # Test no classifcation_loss
            #train_loss = FLAGS.normal_param * classification_loss + FLAGS.reg_param * regularization_loss
            train_loss = regularization_loss
            
            weights_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "model_weights")
            train_step_weights = opt.minimize(train_loss, var_list = weights_train_vars)
            print('Defined operation train_step_weights')
            
            """
            Train_step_per = []
    
            for r in range(FLAGS.num_classes):
                Train_step_per.append(opt.minimize(-class_margin[r], var_list = [per]))
            print('Defined operation train_step_per')
            """

        elif (FLAGS.reg_type == 'None' or FLAGS.reg_param == 0 ):
            train_loss = classification_loss
            train_step = opt.minimize(train_loss)
        else:
            print("Computing gd regularized loss")
            regularization_loss, dump_loss = get_regularization_loss(x, y_, y_model, FLAGS)
            train_loss = FLAGS.normal_param * classification_loss + FLAGS.reg_param * regularization_loss
            train_step_weights = opt.minimize(train_loss)

        sess.run(tf.global_variables_initializer())

        for j in range(FLAGS.restore_epoch, FLAGS.num_epochs):
            print("EPOCH " + str(j))

            ## Shuffling the data in each epoch
            st = time.time()
            P = np.arange(np.max(np.shape(X_train)))
            np.random.shuffle(P)
            X_train = X_train[P]
            Y_train = Y_train[P]
            en = time.time()
            time_shuffle = (en - st)

            for i in range(num_batches):
                start = i * FLAGS.batch_size
                end = (i+1) * FLAGS.batch_size
                
                if i % 50 == 0:
                    Test_accuracy = accuracy.eval(feed_dict={
                        x: X_test, y_: Y_test})

                    if(FLAGS.reg_type != 'None' and FLAGS.reg_param!=0):
                        
                        Train_loss = train_loss.eval(feed_dict={x:X_train[0:FLAGS.batch_size, :], y_:Y_train[0:FLAGS.batch_size, :]})
                        Classification_loss = classification_loss.eval(feed_dict={x:X_train[0:FLAGS.batch_size, :], y_:Y_train[0:FLAGS.batch_size, :]})
                    
                        stats = {
                            'test accuracy': float(Test_accuracy),
                            'train loss': float(Train_loss),
                            'classification loss': float(Classification_loss)
                            }
                    
                        if (FLAGS.reg_type == 'gd'):
                            Reg_loss = Train_loss - Classification_loss
                        else:
                            Reg_loss = Train_loss
                        
                        print('step %d, test accuracy %g, training loss %g, regularization loss %g, classification loss %g' % (i, Test_accuracy, Train_loss, Reg_loss, Classification_loss))

                        results_new = FLAGS.results_dir + "-epoch" + str(j) + "-step" + str(i)
                        if(i == 100):
                            if not os.path.exists(results_new):
                                os.makedirs(results_new)
                            with open(os.path.join(results_new, 'stats.json'), 'w') as f:
                                print(json.dumps(stats), end="", file=f)

                    else : 
                        print('step %d, test accuracy %g' % (i, Test_accuracy))
                    # Saving all the variables here (weight + dual)
                
                if (FLAGS.reg_type == 'so_pgd'):
                    """
                    # Re-initialize pertubations to zeros
                    per = tf.zeros_like(per)
                    
                    #for i in range(0, FLAGS.num_gd_iter):
                    for i1 in range(FLAGS.num_classes):
                        print("Train perturbation class:", i1)
                        # Train perturbation step
                        # Running one of a list of train steps here
                        sess.run(Train_step_per[i1], feed_dict={x: X_train[start:end, :], y_:Y_train[start:end, :]}, options = options, run_metadata=run_metadata)
                        
                    # Project pertubation
                    per = tf.clip_by_value(per, -FLAGS.train_epsilon, FLAGS.train_epsilon)
                    """
                
                # Running train step for weights
                print("Train weights step:", i)
                sess.run(train_step_weights, feed_dict={x: X_train[start:end, :], y_:Y_train[start:end, :]}, options = options, run_metadata=run_metadata)

        final = FLAGS.msave + "-final"    
        save_path = saver.save(sess, final)
        print("Model saved in file: %s" % save_path)
        ## Saving the model weights 
        
        saver1 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model_weights'))
        weights_path = FLAGS.msave + "-weights"
        save_path = saver1.save(sess, weights_path)
        print("Weights saved in file: %s" % save_path)
