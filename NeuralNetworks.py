'''
Neural Networks
'''

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

def _initialize_parameters(layer_dims):
  '''
  Inputs:
  layer_dims -- ndarray, contains the dimensions of each layer in the network
  
  return:
  parameters -- dictionary, contains 'W1', 'b1',..., 'WL', 'bL'
                Wl -- sparse matrix, shape (layer_dims[l], layer_dims[l-1]
                bl -- ndarray, shape (layer_dims[l], 1)

  '''
  
  np.random.seed(3)
  parameters = {}
  L = len(layer_dims)  # number of layers in the network
  
  for l in range (1, L):
    parameters['W' + str(l)] = np.random.randn(layer_dim[l], layer_dims[l-1]) * 0.01
    parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    
  return parameters  


def _forward_propagation(X, parameters):
  '''
  Implement forward propagation 
  
  inputs:
  X -- tranning data, sparse matrix, shape (n_features, n_samples)
  parameters -- output of _initialize_parameters
  
  output
  AL -- last post-activation value
  caches -- list of caches containing:
            every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
  '''
  
  caches = []
  A = X
  L = len(parameters) // 2
  
  for l in range(1, L):
    A_prev = A
    W, b = parameters['W' + str(l)], parameters['b' + str(l)]
    
    





  
  
