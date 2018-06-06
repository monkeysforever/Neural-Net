# -*- coding: utf-8 -*-
"""
The program performs classification on datasets generated using sklearn as well as a image dataset provided via kaggle through a neural network.
The user can define the layers of the neural network with respect to various activations and layer sizes. 

@author: Randeep
"""

import numpy as np
from imageio import imread 
import os
from skimage.transform import resize
import re
from random import shuffle
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.datasets import make_moons, make_circles, make_blobs

#Ratio to split data into training and test sets
SPLIT_RATIO = 90

def sigmoid(Z):
    #Sigmoid activation function
    return (1/(1 + np.exp(-Z)))

def tanh(Z):
    #tanh activation function
    return np.tanh(Z)

def relu(Z):
    #relu activation function
    A = {}
    A = Z
    A[A > 0] = A[A > 0]
    A[A <= 0] = 0
    return A

def initialize_parameters(dimensions):
    #Function to initialize the parameters
    #Dimensions refers to the sizes of the layers of the neural net
    Layers = len(dimensions)
    W = {}
    b = {}
    for l in range(1, Layers):
        W[l] = np.random.randn(dimensions[l], dimensions[l-1])
        b[l] = np.zeros((dimensions[l], 1))
    parameters = {'Weights' : W,
                  'Biasses' : b}
    return parameters

def linear_forward(X, W, b):
    #Function to compute the linear component of a layer during forward propagation
    Z = np.matmul(W, X) + b   
    return Z

def activation_forward(Z, activation_type):
    #Function to compute the activation component of a layer during forward propagation
    if activation_type == 'sigmoid':
        A = sigmoid(Z)        
    elif activation_type == 'relu':
        A = relu(Z)       
    elif activation_type == 'tanh':
        A = tanh(Z)       
    return A

def linear_backward(dZ, A_prev, W):
    #Function to compute gradients going from the linear component of a layer to the activation  component of the previous layer in backward propagation
    #m refers the number of samples
    m = dZ.shape[1]
    dW = np.matmul(dZ, A_prev.T)/m
    db = np.sum(dZ, axis = 1, keepdims = True)/m
    dA_prev = np.matmul(W.T, dZ)
    return dW, db, dA_prev

def activation_backward(dA, A, activation_type):
    #Function to compute gradients going from the activation component of a layer to the linear component of the same layer in backward propagation
    if activation_type == 'relu':
        A = (A > 0).astype(int) 
    elif activation_type == 'sigmoid':
        A = np.multiply(A, 1-A)             
    elif activation_type == 'tanh':
        A = 1 - np.square(A)    
    return np.multiply(dA, A)

def model_forward(X, parameters, activation_types):
    #Function to complete a single forward propagation
    W = parameters['Weights']
    b = parameters['Biasses']
    Z = {}
    A = {}
    A[0] = X
    Layers = len(W)
    for i in range(1, Layers+1):
        Z[i] = linear_forward(A[i-1], W[i], b[i])       
        A[i] = activation_forward(Z[i], activation_types[i-1])    
    
    return A, Z
    
def model_backward(Y, activation_types, A, Z, W):
    #Function to complete a single backward propagation
    Layers = len(activation_types)
    dW = {}
    db = {}
    dZ = {}
    dA_prev = -(np.divide(Y, A[Layers]) - np.divide(1 - Y, 1 - A[Layers]))
    
    for i in reversed(range(1, Layers+1)):        
        dZ[i] = activation_backward(dA_prev, A[i], activation_types[i-1])        
        dW[i], db[i], dA_prev = linear_backward(dZ[i], A[i-1], W[i])
        
    gradients = {'dW' : dW,
                 'db' : db}
    return gradients

def update_parameters(parameters, gradients, learning_rate):
    #Function to update Weights and Biasses using gradients computed during backward propagation
    Layers = len(parameters['Weights'])    
    for i in range(1, Layers + 1):         
        parameters['Weights'][i] = parameters['Weights'][i] - learning_rate * gradients['dW'][i]
        parameters['Biasses'][i] = parameters['Biasses'][i] - learning_rate * gradients['db'][i]        
    return parameters

def compute_cost(A, Y):
    #Compute the cost from the predicted values and values from the dataset
    m = Y.shape[1]
    Cost = -(np.matmul(Y, np.log(A).T) + np.matmul((1 - Y), np.log(1 - A).T))/m 
    Cost = np.squeeze(Cost)
    return Cost

def predict(X, model):
    #Function to make predictions based on the model argument and input data X
    parameters = model['Parameters']
    layers = len(parameters['Weights'])   
    activation_types = model['Activations']
    A, Z = model_forward(X, parameters, activation_types)    
    predictions = A[layers]    
    predictions = (predictions>0.5).astype(int)        
    return predictions

def calculate_accuracy(Y, Y_predictions):
    #Function to calculate accuracy of predictions
    return 100 - np.mean(np.abs(Y_predictions - Y))*100

def train(X_train, Y_train, X_test, Y_test, learning_rate, num_iterations, print_iteration, activation_types, layer_dimensions):
    #Function to train a model using hyperparameters, learning rate, number of iterations, activationa types and layer dimensions
    #Activation types defines the activation functions of each layer
    #Layer dimensions defines the sizes of the layers
    #Shape of X - (features, samples)
    #Shape of Y - (1, samples)
    parameters = initialize_parameters(layer_dimensions)            
    Costs = []
    for i in range(num_iterations):
        A, Z = model_forward(X_train, parameters, activation_types)
        Cost = compute_cost(A[len(layer_dimensions)-1], Y_train)
        if i != 0 and i % print_iteration == 0:
            Costs.append(Cost)
            print ("Cost after iteration %i: %f" %(i, Cost))     
        gradients = model_backward(Y_train, activation_types, A, Z, parameters['Weights'])
        parameters = update_parameters(parameters, gradients, learning_rate)
    model = {'Learning Rate' : learning_rate,
             'Activations' : activation_types,
             'Layer Sizes' : layer_dimensions,
             'Iterations' : num_iterations,
             'Parameters' : parameters,
             'Costs' : Costs}
    Y_train_predictions = predict(X_train, model)
    Y_test_predictions = predict(X_test, model)
    train_accuracy = calculate_accuracy(Y_train, Y_train_predictions)
    test_accuracy = calculate_accuracy(Y_test, Y_test_predictions)
    model['Test Accuracy'] = test_accuracy
    model['Training Accuracy'] = train_accuracy
    model['Training Predictions'] = Y_train_predictions
    model['Test Predictions'] = Y_test_predictions
    
    return model

def load_images(file_path, file_count, image_size):
    #Function to load images into training and test sets    
    dirs = os.listdir(file_path)
    dirs = np.array(dirs)
    shuffle(dirs)    
    split_index = file_count * SPLIT_RATIO // 100
    indices = np.random.permutation(file_count)
    training_idx, test_idx = indices[:split_index], indices[split_index:]    
    image_training_paths = dirs[training_idx]
    image_test_paths = dirs[test_idx]
    Y_training = [re.match('cat', s) != None for s in image_training_paths]
    Y_training = np.array(Y_training).astype(int)
    Y_test = [re.match('cat', s) != None for s in image_test_paths]
    Y_test = np.array(Y_test).astype(int)   
    Y_training = Y_training.reshape((1, Y_training.shape[0]))
    Y_test = Y_test.reshape((1, Y_test.shape[0]))    
    image_training_paths  = [file_path + '/' + s for s in image_training_paths]
    image_test_paths  = [file_path + '/' + s for s in image_test_paths]
    images_training = [resize(imread(s), (image_size, image_size)) for s in image_training_paths]
    images_test = [resize(imread(s), (image_size, image_size)) for s in image_test_paths]    
    X_training = np.array(images_training)
    X_test = np.array(images_test)   
    X_training = np.reshape(X_training,(-1 ,X_training.shape[1] * X_training.shape[2] * X_training.shape[3])).T
    X_test = np.reshape(X_test,(-1 ,X_test.shape[1] * X_test.shape[2] * X_test.shape[3])).T
    X_training = X_training/255
    X_test = X_test/255 
    print('iamgees loaded')    
    return X_training, Y_training, X_test, Y_test

def plot_cost(Costs):
    #Function to plot Costs and iterations
    pyplot.plot(Costs)
    pyplot.show()

def plot_dataset(X, Y):
    #This function plots the dataset        
    df = DataFrame(dict(x=X[0,:], y=X[1,:], label=Y[0, :]))
    colors = {0:'red', 1:'blue'}
    fig, ax = pyplot.subplots()    
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    pyplot.show()
    
def generate_dataset(examples_count, dataset_type = 'moons'):
    #This function generates different types of data sets and divides the dataset into training and test sets
    #The datasets supported are blobs, moons and circles
    #Shape of X for test and training is (number of features, examples)
    #Shape of Y for test and training is (1, examples)    
    if dataset_type == 'blobs':
        X, y = make_blobs(n_samples=examples_count, centers=2, n_features=2)        
    elif dataset_type == 'moons':
        X, y = make_moons(n_samples=examples_count, noise=0.1)
    elif dataset_type == 'circles':
        X, y = make_circles(n_samples=examples_count, noise=0.05)
            
    X = X.T
    y = np.reshape(y, (1, y.shape[0]))   
    split_index = X.shape[1]*SPLIT_RATIO//100    
    indices = np.random.permutation(X.shape[1])
    training_idx, test_idx = indices[:split_index], indices[split_index:]    
    X_training, X_test, Y_training, Y_test = X[:, training_idx], X[:, test_idx], y[:, training_idx], y[:, test_idx]    
    return X_training, X_test, Y_training, Y_test
#$teps to classify
#1.Generate dataset using generate_dataset or load_images
#X, X_test, Y, Y_test = generate_dataset(10000, 'circles')
#2.Set the layer sizes and activations
#activations = ['relu', 'relu', 'relu', 'sigmoid']
#dimensions = [2, 20, 7, 5, 1]
#dimensions[0] should be the number of input features
#dimensions[-1] should be 1 to signify the output layer
#3.Train your model
#model = train(X, Y, X_test, Y_test, 0.005, 10000, 200)