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

def get_regularization_loss(x, y_, y_model, FLAGS):
        
    if(FLAGS.reg_type == 'gd'):
        x_adv = train_gd(x, y_, FLAGS.train_epsilon, FLAGS)
        with tf.variable_scope("model_weights") as scope:
            scope.reuse_variables()
            y_adv = get_model(x_adv, FLAGS)
            scope.reuse_variables()
        
        adv_loss = get_classification_loss(x, y_, y_adv, FLAGS)
        return adv_loss

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

        if (FLAGS.reg_type == 'None' or FLAGS.reg_param == 0 ):
            train_loss = classification_loss
            train_step = opt.minimize(train_loss)
        else:
            print("Computing gd regularized loss")
            regularization_loss = get_regularization_loss(x, y_, y_model, FLAGS)
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
                
                if i % 100 == 0:
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
