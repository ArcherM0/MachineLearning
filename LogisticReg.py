# Import the packages we need for the computation
import numpy as np
import random
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets

# load the dataset
iris = dataset.load_iris()

# only look at the first two features
X = iris.data
Y = iris.target



theta = np.zeros(X.shape[1])
m = X.shape[0]
n_iterations = 2000
alpha = 0.5
cost_history = np.zeros(n_iterations)
theta_history = np.zeros(n_iterations)

for i in range(n_iterations):
    theta_history[i,:] = gradient_Descent(theta, alpha, X, y)
    cost_history[i] = cost(X, y, theta, m)

fig,ax = plt.subplots(figsize=(12,8))

ax.set_ylabel('J(Theta)')
ax.set_xlabel('Iterations')
_=ax.plot(range(n_iter),cost_history,'b.')

"""
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_loss_and_grad(X, y, theta, m):
    h = sigmoid(np.dot(X, theta))
    return (-y * np.log(h) - (1 - y)* np.log(1 - h)) / m

def gradient_Descent(theta, alpha, x , y):
    h = sigmoid(np.matmul(x, theta))
    gradient = np.dot(X.T, (h - y)) / m;
    theta = theta - alpha * gradient
    return theta 
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
            
        
        
        
        
        
      


    
           
    

    
