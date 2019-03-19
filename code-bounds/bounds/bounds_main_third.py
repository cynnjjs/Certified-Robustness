import tensorflow as tf
import numpy as np
import cvxpy as cp
import os
import sys

from utils.get_model import get_model

"""
    Compute the loss: y_ is true lable, y_model is model prediction
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

"""
    Madry's PGD - projection step
"""
def project(x_t, x, epsilon):
    e = tf.ones(tf.shape(x), tf.float32)*epsilon
    proj_x = tf.minimum(x_t, tf.add(x, e))
    proj_x = tf.maximum(proj_x, tf.subtract(x, e))
    return proj_x

"""
    Madry's PGD - main
"""
def train_gd (x, y_, epsilon, FLAGS):
    
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
    Run PGD on test examples to get perturbed X.
    Evaluate the relative magnitude of l(x+epsilon), l(x), delta(l) * eps, 0.5* eps^T hessian(l) * eps, and residual
"""

def bounds_main_third(X_test, Y_test, Epsilon, FLAGS):
    # Fix random seed for initial pgd perturbations
    tf.set_random_seed(42)
    
    # Placeholder for the actual test set
    x = tf.placeholder(tf.float32, [None, FLAGS.dimension])
    # Placeholder for 1 test example, because tf.Hessians only calculates one x
    x_1 = tf.placeholder(tf.float32, [FLAGS.dimension])
    # y_ are true labels of x
    y_ = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
    # y_1 is true label of x_1
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

    x_adv = train_gd(x, y_, FLAGS.test_epsilon, FLAGS)
    
    # Clean classification loss for entire test set
    cla_loss = get_classification_loss(x, y_, y_model, FLAGS)
    # Clean classification loss for 1 test example
    cla_1_loss = get_classification_loss(x_1, y_1, y_1_model, FLAGS)
    grad_x = tf.gradients(cla_loss, [x])
    hessian_1_x = tf.hessians(cla_1_loss, [x_1])

    # Adversarial perturbation of entire test set
    X_adv = x_adv.eval(feed_dict={x: X_test, y_: Y_test})
    X_per = X_adv - X_test
    
    num_points = np.shape(Y_test)[0]

    l_x_adv = cla_loss.eval(feed_dict={x: X_adv, y_: Y_test})
    l_x = cla_loss.eval(feed_dict={x: X_test, y_: Y_test})
    grad_x = sess.run(grad_x, feed_dict={x: X_test, y_: Y_test})[0]

    for i in range(num_points):
        hes = sess.run(hessian_1_x, feed_dict={x_1: X_test[i], y_1: Y_test[i]})[0]
        b = np.dot(grad_x[i], X_per[i])
        c = np.matmul(np.matmul(np.reshape(X_per[i],[1, FLAGS.dimension]), np.reshape(hes,[FLAGS.dimension, FLAGS.dimension])), np.reshape(X_per[i],[FLAGS.dimension, 1]))[0, 0]*0.5
        print(l_x_adv[i], l_x[i], b, c)
        print('Residual:', l_x_adv[i] - l_x[i] - b - c)

    sess.close()
