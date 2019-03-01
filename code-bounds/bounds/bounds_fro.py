import numpy as np 
import math 

def bounds_fro(model_weights, FLAGS):
    Fro_bounds = np.zeros([FLAGS.num_classes, FLAGS.num_classes])
    ##1. Scan every pair of classes and create the matrix + vector  
    W_FC1 = model_weights["W_FC1"]
    # Dimension is [num_hidden, classes]
    W_FC2 = model_weights["W_FC2"]
    # Dimension is [num_classes, num_hidden]
    B_FC1 = model_weights["B_FC1"]
    B_FC2 = model_weights["B_FC2"]
    
    fro_norm = np.linalg.norm(W_FC1, 'fro')
    fro_value = math.sqrt(FLAGS.dimension)*fro_norm

    for i in range (FLAGS.num_classes):
        for j in range(i):

            w = W_FC2[i, :] - W_FC2[j, :]
            w = np.ravel(w)
            
            Fro_bounds[i, j] = fro_value*np.linalg.norm(w)
            Fro_bounds[j, i] = fro_value*np.linalg.norm(w)

    return Fro_bounds
