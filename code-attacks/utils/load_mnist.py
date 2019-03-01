"""
Function that returns data of required number of classes
args 

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import sys 
import os
import warnings

def load_mnist(FLAGS):

    mnist = input_data.read_data_sets('mnist', one_hot=True, reshape=False)
    X_train = np.vstack((mnist.train.images, mnist.validation.images))
    Y_train = np.vstack((mnist.train.labels, mnist.validation.labels))

    X_test = mnist.test.images
    Y_test = mnist.test.labels
    

    ### Selecting only the required labels
    if (os.path.isfile(FLAGS.class_path) == 0):
        labels = np.arange(FLAGS.num_classes)
    else :
        
        labels = np.loadtxt(FLAGS.class_path)
        labels = labels.astype(int)
    
    print('Labels')
    print(labels)
    indices =[]
    indices = np.array(indices, dtype = int)

    for a in labels: 
        indices = np.concatenate((indices, np.ravel(np.where(np.argmax(Y_train, 1) == a))), axis = 0)
            
    X_train = X_train[indices, :, :, :]
    Y_train= Y_train[indices,:]
    ## Selecting columns 
    Y_train = Y_train[:, labels]
    
    indices =[]
    indices = np.array(indices, dtype = int)

    for a in labels: 
        indices = np.concatenate((indices, np.ravel(np.where(np.argmax(Y_test, 1) == a))), axis = 0)
            
    X_test = X_test[indices, :, :, :]
    Y_test = Y_test[indices,:]
    Y_test = Y_test[:, labels]
        
    ### Reshaping into [num_examples, dimension] format

    X_train = np.reshape(X_train, [np.shape(X_train)[0], 784])
    X_test = np.reshape(X_test, [np.shape(X_test)[0], 784])

    np.random.seed(1)
    A = np.random.permutation( np.shape(X_test)[0])
    # X_test = X_test[A, :]
    # Y_test = Y_test[A, :]

    ## Permuting X_test
    
    
    print('Labels:', labels)
    print ('Number of training examples:', X_train.shape[0])
    print ('Number of test examples:', X_test.shape[0])
    
    return X_train, Y_train, X_test, Y_test



        
    



