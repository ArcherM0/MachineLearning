# Import the packages we need for the computation
import numpy as np
imporr random
import matplotlib.pyplot as plt 

# Define the hypothesis function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the Gradient Descent
def Grandient(x, y, alpha, theta, m):
    # hypothesis
    h = sigmoid(np.dot(x, theta))
    # cost function
    cost = (np.dot(-y.T, np.log(h)) - np.dot((1-y).T, np.log(1-h))) / m
    # gradient
    gradient = np.dot(x.T, (h - y)) / m
    # update theta
    theta = theta - alpha * gradient


    
           
    

    
