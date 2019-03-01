import numpy as np 
import math

def bounds_spectral(model_weights, FLAGS):
    Spectral_bounds = np.zeros([FLAGS.num_classes, FLAGS.num_classes])
    ##1. Scan every pair of classes and create the matrix + vector  
    W_FC1 = model_weights["W_FC1"]
    # Dimension is [num_hidden, classes]
    W_FC2 = model_weights["W_FC2"]
    # Dimension is [num_classes, num_hidden]
    B_FC1 = model_weights["B_FC1"]
    B_FC2 = model_weights["B_FC2"]
    
    w, v = np.linalg.eig(np.dot(np.transpose(W_FC1), W_FC1))
    spectral_value = math.sqrt(max(w)*FLAGS.dimension); 

    for i in range (FLAGS.num_classes):
        for j in range(i):

            w = W_FC2[i, :] - W_FC2[j, :]
            w = np.ravel(w)
            
            Spectral_bounds[i, j] = spectral_value*np.linalg.norm(w)
            Spectral_bounds[j, i] = spectral_value*np.linalg.norm(w)

    return Spectral_bounds
