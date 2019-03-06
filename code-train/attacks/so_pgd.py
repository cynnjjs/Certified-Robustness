from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from utils.get_model import get_model 

"""
Takes as input 
1. x: Input placeholder
2. y_: Labels placeholder (one hot)
3. args["model"]
4. args["num_gd_iter"]
5. args["gd_learning_rate"]
"""

def train_so_pgd (x, y_, grad_fx, w_fc1, FLAGS):
    # grad_fx: ? * 784
    # x, per: ? i * 784 j
    # PSD_M: ? i * 784 j * 784 k
    per = tf.zeros_like(x)
    #per = tf.random_uniform(tf.shape(x), minval = -1*delta, maxval = delta)
    
    # Computing the gradient and updating
    for i in range(0, FLAGS.num_gd_iter):
        # loss: ?
        ##loss = tf.add(tf.reduce_sum(tf.multiply(grad_fx, per), 1), tf.einsum('ij,ijk,ik->i', per, psd_M, per))
        w_per = tf.matmul(per, w_fc1)
        loss = tf.add(tf.reduce_sum(tf.multiply(grad_fx, per), 1), tf.reduce_sum(tf.multiply(w_per,w_per), 1))

        grad, = tf.gradients(-loss, per);

        signed_grad = tf.sign(grad)
        scaled_signed_grad = FLAGS.gd_learning_rate * signed_grad
  
        per = per + scaled_signed_grad
        per = tf.clip_by_value(per, -FLAGS.train_epsilon, -FLAGS.train_epsilon)
  
  #loss = tf.add(tf.reduce_sum(tf.multiply(grad_fx, per), 1), tf.einsum('ij,ijk,ik->i', per, psd_M, per))
                
    return tf.stop_gradient(per)
    
