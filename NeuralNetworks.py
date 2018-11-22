''' 2-layer model'''

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

def initialize_parameters(layer_dims):
  '''
  Input:
  layer_dims -- ndarray, contains the dimensions of each layer in the network
  
  output:
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





  
  
