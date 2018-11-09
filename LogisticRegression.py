"""Logistic Regression, 2-class version
   Author: Archer M
"""

import numpy as np
from scipy.special import expit


def _initialize(X):
   '''
   initializes w, b
   X: array-like, sparse matrix, shape(n_samples, n_features)
   w: ndarray, shape (n_features,1)
   b: float, the intercept
   '''
   n_samples, n_features = X.shape
   w = np.zeros(n_features)
   b = 0.
   
   return w, b

def _logistic_cost_and_grad(w, b, X, Y):
   '''
   computes logistic cost and gradients
   X: array-like, sparse matrix, shape(n_samples, n_features)
   Y: ndarray, shape(n_samples,1)
   b: float, the intercept
   '''
   
   w, b = _initialize(X)
   n_samples, n_features = X.shape
   
   z = np.dot(X, w)
   A = expit(z)
   
   cost = np.sum(-Y * np.log(A) - (1-Y) * np.log(1-A)) / n_samples
   
   dw = np.dot(X, (A-Y)) / n_samples
   db = np.sum(A-Y) / n_samples
   
   grads = {"dw": dw,
            "db": db}
   cost = np.squeeze(csot)
   
   return cost, grads

def _optimize(w, b, X, Y, alpha, max_iter):
   
   cost = []
   
   for i in range(max_iter):
      cost, grads = _logistic_cost_and_grad(w, b, X, Y)
      dw, db = grads["dw"], grad["db"]
      ### simultaneously update w and b
      w -= alpha * dw
      b -= alpha * db
      
      ### track the cost
      for i % 100 == 0:
         cost.append(cost)
         
   return w, b

def _predict(w, b, X):
   z = np.dot(X, w)
   A = expit(z)
   Y_pred = []
   
   
   
   
   
   
   
   
   


def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):

    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost = False)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    
    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    out = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return out

"""
      
  
  


    
    
            

  
  
  
  
  
  
 
                
  
