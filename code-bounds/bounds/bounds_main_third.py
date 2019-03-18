import tensorflow as tf
import numpy as np
import cvxpy as cp
import os
import sys

from utils.get_model import get_model

## Bounds
from bounds.bounds_sdp import get_pairwise_loss
from bounds.bounds_spectral import bounds_spectral 
from bounds.bounds_fro import bounds_fro
from bounds.bounds_so import qp_solver, qp_feasibility, sdp_solver, eq_solver

"""
Computes the loss for each bound
Takes as input: 
1. X_test --> To compute the current function values 
2. Y_test
3. FLAGS for defining the model 
"""

def get_classification_loss(x, y_, y_model, FLAGS):
    
    ## Classification loss part
    if(FLAGS.loss == 'hinge'):
        H = tf.reduce_max(y_model * (1 - y_), 1)
        H = tf.reshape(H, [-1, 1])
        H = tf.tile(H, [1, FLAGS.num_classes])
        
        L = tf.nn.relu((1 - y_model + H) * y_)
        return tf.reduce_mean(tf.reduce_max(L, 1))
    
    elif (FLAGS.loss == 'cross-entropy'):
        softmax_loss = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_model)
        return softmax_loss

def project(x_t, x, epsilon):
    e = tf.ones(tf.shape(x), tf.float32)*epsilon
    proj_x = tf.minimum(x_t, tf.add(x, e))
    proj_x = tf.maximum(proj_x, tf.subtract(x, e))
    return proj_x

"""
    Takes as input
    1. x: Input placeholder
    2. y_: Labels placeholder (one hot)
    3. args["model"]
    4. args["num_gd_iter"]
    5. args["gd_learning_rate"]
    """
def train_gd (x, y_, epsilon, FLAGS):
    tf.set_random_seed(42)
    per = tf.random_uniform(tf.shape(x), minval = -1*epsilon, maxval = epsilon)
    x_t = x + per
        
    # Computing the gradient and updating
    for i in range(0, FLAGS.num_gd_iter):

        with tf.variable_scope("model_weights") as scope:
            scope.reuse_variables()
            y_adv = get_model(x_t, FLAGS)
    
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_adv))
        
        grad, = tf.gradients(loss, x_t);
                          
        signed_grad = tf.sign(grad)
        scaled_signed_grad = FLAGS.gd_learning_rate * signed_grad
                          
        x_t = (x_t + scaled_signed_grad)
        x_t =  project(x_t, x, epsilon)
                              
    return x_t

""" 
Restore the weights, and then save the weights
separately as a dictionary (two_layer)
and call the appropriate two_layer bounds 
Takes as input: 
a. X_test 
b. Y_test 
c. args["fsave"] + weights that has the weights
d. args["model"] : Model to build
e. args["results_dir"]: Directory inside which to add the loss files 
f. args["Epsilon"]: Range of epsilon to compute bounds over
"""

def bounds_main_finer(X_test, Y_test, Epsilon, FLAGS):
    
    x = tf.placeholder(tf.float32, [None, FLAGS.dimension])
    x_1 = tf.placeholder(tf.float32, [FLAGS.dimension])
    y_ = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
    y_1 = tf.placeholder(tf.float32, [FLAGS.num_classes])

    with tf.variable_scope("model_weights") as scope:
        y_model = get_model(x, FLAGS)
        scope.reuse_variables()
    
    with tf.variable_scope("model_weights") as scope:
        scope.reuse_variables()
        y_1_model = get_model(tf.reshape(x_1, [1, FLAGS.dimension]), FLAGS)
    
    tvars = tf.trainable_variables()
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    ## Restoring the weights

    saver.restore(sess, FLAGS.msave + "-final")
    
    w_fc1  = tvars[0]
    b_fc1 = tvars[1]
    w_fc2 = tvars[2]
    b_fc2 = tvars[3]
    print(w_fc1.eval()[0,1])
    
    x_adv = train_gd(x, y_, FLAGS.train_epsilon, FLAGS)
    with tf.variable_scope("model_weights") as scope:
        scope.reuse_variables()
        y_adv = get_model(x_adv, FLAGS)

    cla_loss = get_classification_loss(x, y_, y_model, FLAGS)
    cla_1_loss = get_classification_loss(x_1, y_1, y_1_model, FLAGS)
    adv_loss = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_model)

    X_adv = x_adv.eval(feed_dict={x: X_test, y_: Y_test}) - X_test
    Y_adv = y_adv.eval()
    Adv_loss = adv_loss.eval(feed_dict={y_: Y_test, y_model: })

    grad_x = tf.gradients(cla_loss, [x])

    num_points = np.shape(Y_test)[0]
    hessian_1_x = tf.hessians(cla_1_loss, [x_1])
    
    l_x_adv = adv_loss.eval(feed_dict={x: X_test, y_: Y_test})
    l_x = cla_loss.eval(feed_dict={x: X_test, y_: Y_test})
    grad_x = sess.run(grad_x, feed_dict={x: X_test, y_: Y_test})[0]
    print(len(l_x_adv), len(grad_x))
    for i in range(num_points):
        hes = sess.run(hessian_1_x, feed_dict={x_1: X_test[i], y_1: Y_test[i]})[0]
        b = np.dot(grad_x[i], X_adv[i])
        c = np.matmul(np.matmul(np.reshape(X_adv[i],[1, FLAGS.dimension]), np.reshape(hes,[FLAGS.dimension, FLAGS.dimension])), np.reshape(X_adv[i],[FLAGS.dimension, 1]))[0, 0]*0.5
        print(l_x_adv[i], l_x[i], b, c)

        print(l_x_adv[i] - l_x[i] - b - c)

    sess.close()
