"""Logistic Regression, 2-class version
   Author: Archer M
"""

import numpy as np
from scipy.special import expit

class LogisticRegression:
   
   def __init__(self, w, X, Y, b, aplpha, num_iterations):
      self.w = w
      self.X = X
      self.Y = Y
      self.b = b
      self.alpha = alpha
      self.num_iterations = num_iterations
     
   def _logistic_cost_and_grads(self, w, b, X, Y):
      
       """ Computes the logistic loss and gradients
       """   
      
      z = np.dot(w.T, X) + b 
      A = expit(z)
      cost = np.sum(-Y * np.log(A) - (1-Y) * np.log(1-A)) / X.shape[1]
      dw = np.dot(X, (A-Y).T) / X.shape[1]
      db = np.sum(A-Y) / X.shape[1]
   
      cost = np.squeeze(cost)
      grads = {"dw": dw,
               "db": db}
   
      return cost, grads
   
   def _optimization(self, w, b, X, Y, num_iterations):
      
      """ Optimizes the weight w, intercept b
      """
      cost = []
      for i in range(num_iterations):
         cost, grads = _logistic_cost_and_grads(w, b, X, Y)
         dw, db = grads["dw"], grads["db"]
         ### simultaneouly update w, b
         w -= alpha * dw
         b -= alpha * db
         
         ### records cost
         if i % 100 == 0:
            cost.append(cost)
            
         params = {"w": w,
                   "b": b}
         grads = {"dw": dw,
                  "db": db}
       
      return params, grads, cost
   
   def _predict(self, w, b, X):
     
      Y_pred = np.zeros((1, X.shape[1]))
      
      params, grads, cost = _optimization(w, b, X, Y, num_iterations)
        
      w = params['w'].reshape(X.shape[0], 1)
      b = params['b']
      
      A = expit(np.dot(w.T, X) + b)
      
      for i in range(X.shape[1]):
          if A[0,i] > 0.5:
              Y_pred[0,i] = 1
          else:
              Y_pred[0,i] = 0
      
      return Y_pred
   
   def _score(self, Y):
      
      Y_pred = _predict(w, b, X)
      Score = 100 - np.mean(np.abs(Y_pred - Y)) * 100
      
      out = {"costs": costs,
      "Y_prediction" : Y_pred, 
      "Score": Score,       
      "w" : w, 
      "b" : b,
      }
    
      return out
   

"""
# Sigmoid function
def sigmoid(z):
  s = 1 / (1 + np.exp(-z))
  return s

def initialization(dim):
  w = np.zeros((dim,1))
  b = 0.
  return w, b

def propagate(w, b, X, Y):
  n_samples = X.shape[1]
  z = np.dot(w.T, X) + b
  
  # FORWARD PROPAGATION (FIND THE COST FROM X)
  A = sigmoid(z)
  loss = np.sum(-Y * np.log(A) - (1-y) * np.log(1-A))/ n_samples
  
  # BACKWARD PROPAGATION (COMPUTE THE GRADIENTS)
  dw = np.dot(X, (A-Y).T) / n_samples
  db = np.sum(A-Y) / n_samples
  
  loss = np.squeeze(loss)
  grads = {"dw": dw,
           "db": db}
  
  return grads, loss

def optimize(w, b, X, Y, num_iterations, alpha, print_cost = False):
  loss = []
  
  for i in range(num_iterations):
    grads, loss = propagate(w, b, X, Y)
    
    dw = grads["dw"]
    db = grads["db"]
    
    # SIMULTANEOUSLY UPDATE w, b
    w -= alpha * dw
    b -= alpha * db
    
    if i % 100 == 0:
      loss.append(loss)
      
   params = {"w": w,
             "b": b}
   grads = {"dw": dw,
            "db": db}
    
   return params, grads, loss

def predict(w, b, X):
  n_samples = X.shape[1]
  Y_pred = np.zeros((1,m))
  w = w.reshape(X.shape[0],1)
  A = sigmoid(np.dot(w.T, X) + b)
  
  for i in range(X.shape[1]):
    if A[0,i] > 0.5:
      Y_pred[0,i] = 1
    else:
      Y_pred[0,i] = 0
      
  return Y_pred


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
      
  
  


    
    
            

  
  
  
  
  
  
 
                
  
