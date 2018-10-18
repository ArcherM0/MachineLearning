# Import the packages we need for the computation
import numpy as np
import random
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets

# load the dataset
iris = dataset.load_iris()

# only look at the first two features
X = iris.data[:, :2]
Y = iris.target

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(x, y, theta, m):
    h = sigmoid(np.dot(X, theta))
    cost = (np.dot(-y, np.log(h)) - np.dot((1 -y), np.log(1 - h)))/m
    return cost

def gradient_Descent(theta, alpha, x , y):
    m = x.shape[0]
    h = sigmoid(np.matmul(x, theta))
    grad = np.dot(X.T, (h - y)) / m;
    theta = theta - alpha * grad
    return theta


theta = np.zeros(X.shape[1])
m = X.shape[0]
n_iterations = 2000
alpha = 0.5
cost_history = np.zeros(n_iterations)

for i in range(n_iterations):
    theta = gradient_Descent(theta, alpha, X, y)
    cost_history[i] = cost(X, y, theta, m)

              

fig,ax = plt.subplots(figsize=(12,8))

ax.set_ylabel('J(Theta)')
ax.set_xlabel('Iterations')
_=ax.plot(range(n_iter),cost_history,'b.')



    


    
           
    

    
