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


# In[6]:


file_list = os.listdir('train')


# In[4]:


#Make the TRAIN dataset
trainX=[]
trainY=[]
count=0
for f in file_list:
            if(count%100)==0:
                print(count)                
            count+=1
            
            #Read train image
            image = ndimage.imread('train/'+f, mode="RGB")
            #Resize it
            image_resized = misc.imresize(image, (50, 50))
            #Grayscale it
            gray = rgb2gray(image_resized)
            #Flatten to ease input
            gray = np.array(gray).flatten()
            #Make X
            trainX.append(gray)
            
            #Make corresponding Y
            if('cat' in f):
                trainY.append(0)
            else:
                trainY.append(1)
    


# In[ ]:


#Make the TEST dataset
testX=[]
testY=[]
count=0
file_list = os.listdir('test')
for f in file_list:
            if(count%100)==0:
                print(count)                
            count+=1
            
            #Read train image
            image = ndimage.imread('test/'+f, mode="RGB")
            #Resize it
            image_resized = misc.imresize(image, (50, 50))
            #Grayscale it
            gray = rgb2gray(image_resized)
            #Flatten to ease input
            gray = np.array(gray).flatten()
            #Make X
            testX.append(gray)
            
            #Make corresponding Y
            if('cat' in f):
                testY.append(0)
            else:
                testY.append(1)
testX = np.array(testX)
testY = np.array(testY)


# In[ ]:


testX.shape


# In[ ]:


#Dump the datasets into the hard-disk for later use
import pickle
with open('trainX.pkl','wb') as f:
    pickle.dump(trainX,f)
with open('trainY.pkl','wb') as f:
    pickle.dump(trainY,f)
with open('testX.pkl','wb') as f:
    pickle.dump(testX,f)
with open('testY.pkl','wb') as f:
    pickle.dump(testY,f)


# In[24]:


import pickle
with open('trainX.pkl','rb') as f:
    trainX = pickle.load(f)
with open('trainY.pkl','rb') as f:
    trainY = pickle.load(f)
with open('testX.pkl','rb') as f:
    testX = pickle.load(f)
with open('testY.pkl','rb') as f:
    testY = pickle.load(f)


# In[25]:


print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)


# In[26]:


trainX = trainX.reshape(trainX.shape[1],trainX.shape[0])
trainY = trainY.reshape(trainY.shape[1],trainY.shape[0])
testX = testX.reshape(testX.shape[1],testX.shape[0])
testY = testY.reshape(testY.shape[1],testY.shape[0])


# In[27]:


testX = trainX[-5000:]
testY = trainY[-5000:]
trainX = trainX[:-5000]
trainY = trainY[:-5000]


# In[28]:


print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)


# In[29]:


def minmaxscale(X):
    X = (X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
    return X


# In[30]:


trainX = minmaxscale(trainX)
testX = minmaxscale(testX)


# In[31]:


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))   
    return s


# In[32]:


def initialize_with_zeros(dim):
    

    w = np.zeros(shape=(dim, 1))
    b = 0
    
    return w, b


# In[33]:


def propagate(w, b, X, Y):
    
    m = X.shape[0]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HERE ### (≈ 2 lines of code)
    
    A = sigmoid(np.dot(X, w) + b)  # compute activation

    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost
    ### END CODE HERE ###
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### (≈ 2 lines of code)
    dw = (1 / m) * np.dot(X.T, (A - Y))
    db = (1 / m) * np.sum(A - Y)
    ### END CODE HERE ###
    cost = np.squeeze(cost)
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost


# In[34]:


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):

    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ### 
        grads, cost = propagate(w, b, X, Y)
        ### END CODE HERE ###
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w - learning_rate * dw  # need to broadcast
        b = b - learning_rate * db
        ### END CODE HERE ###
        
        # Record the costs
        if i % 1 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and i % 1 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


# In[38]:


from sklearn.metrics import confusion_matrix
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[1])
    print("INITIAL PARAMETERS")
    print(w)
    print(w.shape)
    print(b)    
    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    w = parameters["w"]
    b = parameters["b"]
    
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    print(confusion_matrix(Y_train, Y_prediction_train))
    print(confusion_matrix(Y_test, Y_prediction_test))
    
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


# In[39]:


def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[0]
    Y_prediction = np.zeros((m, 1))
    

    A = sigmoid(np.dot(X,w) + b)
    
    for i in range(A.shape[0]):
        # Convert probabilities a[0,i] to actual predictions p[0,i]
        ### START CODE HERE ### (≈ 4 lines of code)
        Y_prediction[i, 0] = 1 if A[i, 0] > 0.5 else 0
    
    
    return Y_prediction


# In[40]:


d = model(trainX, trainY, testX, testY, num_iterations = 1000, learning_rate = 0.01, print_cost = True)


# In[ ]:





# In[ ]:





# In[ ]:




