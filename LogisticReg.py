"""
Logistic Regression (Binomial Case)
"""
# Author: Archer Mo
# Reference: scikit-learn/sklearn/linear_model/logistic.py
      

import numpy as np
from scipy import optimize, sparse
from scipy.special import expit


# Define an intercept in case there is one
def _intercept(w, X, y):
    c = 0.
    if w.size == X.shape[1] + 1:
        c = w[1]
        w = w[1:]
    
    z = np.dot(X, w) + c
    yz = y * z
    return w, c, yz

def _logistic_loss_and_grad(w, X, y, alpha, sample_weight = None):
    n_samples, n_features = X.shape
    grad = np.empty(w)
    w, c, z = _intercept(w, X, yz)
    
    if sample_weight is None:
        sample_weight = np.ones(n_samples)
    
    # Logistic loss with regularization
    loss = -np.sum(sample_weight * log_logistic(yz)) + .5 * alpha * np.dot(w, w)
    
    z = expit(yz)
    z0 = sample_weight * (z - 1) * y
    
    grad[:n_features] = np.dot(X.T, z0) + alpha * w
    
    if grad.shape[0] > n_features:
        grad[1] = z0.sum()
    return loss, grad


        
    
    
    
    
"""        

class LogisticRegression:
    def__init__(self, alpha, iterations, fit_intercept = True):
        self.alpha = alpha
        self.ierations = iterations
        self.fit_intercept = fit_intercept
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0],1))
        return np.concatenate((intercept, X), axis = 1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(z))
    
    def __logistic_loss_and_grad(self, X, y, alpha, theta):
        n_sample, n_features = X.shape
        self.theta = np.zeros(X.shape[1])
        for i in range(self.iterations):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / n_sample
            self.theta = self.theta - self.alpha * gradient
            
            
"""
            
        
        
        
        
        
      


    
           
    

    
