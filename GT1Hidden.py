#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from scipy import ndimage, misc
import numpy as np

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


# In[2]:


import pickle
with open('trainX.pkl','rb') as f:
    trainX = pickle.load(f)
with open('trainY.pkl','rb') as f:
    trainY = pickle.load(f)
with open('testX.pkl','rb') as f:
    testX = pickle.load(f)
with open('testY.pkl','rb') as f:
    testY = pickle.load(f)


# In[3]:


print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)


# In[4]:


testX = trainX[:,-5000:]
testY = trainY[:,-5000:]
trainX = trainX[:,:-5000]
trainY = trainY[:,:-5000]

print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)


# In[5]:


def minmaxscale(X):
    X = (X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
    return X


# In[6]:


trainX = minmaxscale(trainX)
testX = minmaxscale(testX)


# In[7]:


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    cache = z
    return s,z


# In[8]:


def relu(z):
    s = np.maximum(0,z)
    cache = z
    return s,z


# In[9]:


def initialize(l_dims):
    print("INITIALIZATION")
    np.random.seed(3)
    parameters = {}
    L = len(l_dims)-1            # number of layers in the network
    
    for l in range(1, L+1):
        parameters['W' + str(l)] = np.random.randn(l_dims[l], l_dims[l - 1]) * 0.1
        parameters['b' + str(l)] = np.zeros((l_dims[l], 1))

        
    return parameters


# In[10]:


def linear_activation_forward(A_prev, W, b, activation):
    
    if activation == "sigmoid":
        Z = np.dot(W, A_prev) + b
        linear_cache = (A_prev,W,b)
        A, activation_cache = sigmoid(Z)

    
    elif activation == "relu":
        Z = np.dot(W, A_prev) + b
        linear_cache = (A_prev,W,b)        
        A, activation_cache = relu(Z)
        
    cache = (linear_cache, activation_cache)

    return A, cache


# In[11]:


def computeError(AL,Y):
    #Y is of shape 1*(number_of_examples)
    m = Y.shape[1]
    # Compute loss from aL and y.
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL))) 
    cost = np.squeeze(cost)   
    return cost


# In[12]:


def computeWeightChange():
    print()


# In[13]:


def feedforward(X, parameters):
    
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Hidden layers will have ReLU activation to prevent vanishing gradient
    for l in range(1, L):
    
        A_prev = A 
        #print(l)
        #print(A_prev.shape)
        #print(parameters['W' + str(l)].shape)
        #print(parameters['b' + str(l)].shape)
        A, cache = linear_activation_forward(A_prev, 
                                             parameters['W' + str(l)], 
                                             parameters['b' + str(l)], 
                                             activation='relu')
        caches.append(cache)
            
    # Final layer will have sigmoid activation
    AL, cache = linear_activation_forward(A, 
                                          parameters['W' + str(L)], 
                                          parameters['b' + str(L)], 
                                          activation='sigmoid')
    caches.append(cache)            
    return AL, caches


# In[14]:


def linear_backward(dZ, cache):
    #Take in dZ and return dA_prev, dW and dZ
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, cache[0].T) / m
    db = np.squeeze(np.sum(dZ, axis=1, keepdims=True)) / m
    dA_prev = np.dot(cache[1].T, dZ)


    return dA_prev, dW, db


# In[15]:


def linear_activation_backward(dA, cache, activation):
    
    linear_cache, activation_cache = cache  
    
    if activation == "sigmoid":
        #Apply formula for Sigmoid
        #Take in dA and return dZ
        Z = activation_cache
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)      
        
    elif activation == "relu":
        #Simple enough for ReLU
        #Take in dA and return dZ
        Z = activation_cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0        
        
    # Shorten the code
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


# In[16]:


def backpropagate(AL, Y, caches):
    
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1] #The number of examples
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    #Find dAL
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    dA_prev_temp = grads["dA" + str(L)]
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev_temp, current_cache, "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# In[17]:


def update_parameters(parameters, grads, learning_rate):   
    
    L = len(parameters) // 2 # number of layers in the neural network
    # Update rule for each parameter. Use a for loop.

    for l in range(L):        
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


# In[18]:


def predict(w, b, X):
    
    m = X.shape[0]
    Y_prediction = np.zeros((m, 1))
    

    A = sigmoid(np.dot(X,w) + b)
    
    for i in range(A.shape[0]):
        # Convert probabilities a[0,i] to actual predictions p[0,i]
        ### START CODE HERE ### (≈ 4 lines of code)
        Y_prediction[i, 0] = 1 if A[i, 0] > 0.5 else 0
    
    
    return Y_prediction


# In[19]:


def L_layer_model(X, Y, layers_dims, learning_rate=0.01, num_iterations=3000, print_cost=False):

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization.
    
    parameters = initialize(layers_dims)

    # Loop (gradient descent)
    
    for i in range(0, num_iterations):
        print(i)
        #print("BEGIN FEED FORWARD")
        # Forward propagation
        AL, caches = feedforward(X, parameters)

        # Compute cost.
        #print("COMPUTING COST")
        cost = computeError(AL, Y)

        # Backward propagation.
        #print("BEGIN BACKPROP")
        grads = backpropagate(AL, Y, caches)
 
            
        # Update parameters.

        for key,val in grads.items():
            if('b' in key):
                grads[key] = val.reshape(-1,1)
            #print(key,grads[key].shape)
            
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 1 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 1 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


# In[20]:


parameters = L_layer_model(trainX, trainY, np.array([2500,100,20,5,1]), learning_rate = 0.8, num_iterations=10000, print_cost=True)


# In[ ]:


np.random.seed(1)
costs = []                         # keep track of cost

# Parameters initialization.
X = trainX
Y = trainY
layers_dims = np.array([2500,100,20,5,1])
learning_rate = 0.2
num_iterations=1000
print_cost=True

parameters = initialize(layers_dims)

# Loop (gradient descent)

for i in range(0, num_iterations):
    print(i)
    #print("BEGIN FEED FORWARD")
    # Forward propagation
    AL, caches = feedforward(X, parameters)

    # Compute cost.
    #print("COMPUTING COST")
    cost = computeError(AL, Y)

    # Backward propagation.
    #print("BEGIN BACKPROP")
    grads = backpropagate(AL, Y, caches)


    # Update parameters.

    for key,val in grads.items():
        if('b' in key):
            grads[key] = val.reshape(-1,1)
        #print(key,grads[key].shape)

    parameters = update_parameters(parameters, grads, learning_rate)

    # Print the cost every 100 training example
    if print_cost and i % 1 == 0:
        print ("Cost after iteration %i: %f" % (i, cost))
    if print_cost and i % 1 == 0:
        costs.append(cost)

# plot the cost
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate =" + str(learning_rate))
plt.show()


# In[67]:


from sklearn.metrics import confusion_matrix

m = trainX.shape[1]
Y_prediction_train = np.zeros((1, m))
m = testX.shape[1]
Y_prediction_test = np.zeros((1, m))

A,cache = feedforward(trainX,parameters)  
for i in range(A.shape[1]):
    Y_prediction_train[0, i] = 1 if A[0, i] > 0.5 else 0

A,cache = feedforward(testX,parameters)        
for i in range(A.shape[1]):
    Y_prediction_test[0, i] = 1 if A[0, i] > 0.5 else 0

print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - trainY)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - testY)) * 100))

print(trainY.shape)
print(Y_prediction_train.shape)

trainY = trainY.reshape(trainY.shape[1],trainY.shape[0])
Y_prediction_train = Y_prediction_train.reshape(Y_prediction_train.shape[1],Y_prediction_train.shape[0])
testY = testY.reshape(testY.shape[1],testY.shape[0])
Y_prediction_test = Y_prediction_test.reshape(Y_prediction_test.shape[1],Y_prediction_test.shape[0])

print(trainY.shape)
print(Y_prediction_train.shape)
'''
print(confusion_matrix(trainY, Y_prediction_train))
print(confusion_matrix(testY, Y_prediction_test))
'''
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate =" + str(learning_rate))
plt.show()


# In[61]:


AL.shape


# In[33]:


trainY.shape

