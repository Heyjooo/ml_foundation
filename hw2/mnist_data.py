# data 출처 : https://github.com/WegraLee/deep-learning-from-scratch

# Importing necessary libraries
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist

def loadData():
    # Loading MNIST data
    (x_train, y_train), (x_test, y_test) = load_mnist(flatten = True, normalize = False)

    print("The shape of the training set feature matrix is:", x_train.shape)
    print("The shape of the training label vector is:", y_train.shape)
    print("The shape of the test set feature matrix is:", x_test.shape)
    print("The shape of the test label vector is:", y_test.shape)

    #if model == "RL":
    #    x_train = x_train.reshape(60000, 28 * 28).astype('float32')/255.0 # nomalize
    #    y_train = y_train.astype('int32')
    #    x_test = x_test.reshape(10000, 28 * 28).astype('float32')/255.0   # nomalize
    #    y_test = y_test.astype('int32')     
    #    return x_train, y_train, x_test, y_test

    #else :
    return x_train, y_train, x_test, y_test
    
    
# Creating new training label vectors for each digit for the one-vs-all methode
'''
y_train_0=(y_train==0).astype(int)
y_train_1=(y_train==1).astype(int)
y_train_2=(y_train==2).astype(int)
y_train_3=(y_train==3).astype(int)
y_train_4=(y_train==4).astype(int)
y_train_5=(y_train==5).astype(int)
y_train_6=(y_train==6).astype(int)
y_train_7=(y_train==7).astype(int)
y_train_8=(y_train==8).astype(int)
y_train_9=(y_train==9).astype(int)
'''
# Creating new testing label vectors for each digit for the one-vs-all methode
'''
y_test_0=(y_test==0).astype(int)
y_test_1=(y_test==1).astype(int)
y_test_2=(y_test==2).astype(int)
y_test_3=(y_test==3).astype(int)
y_test_4=(y_test==4).astype(int)
y_test_5=(y_test==5).astype(int)
y_test_6=(y_test==6).astype(int)
y_test_7=(y_test==7).astype(int)
y_test_8=(y_test==8).astype(int)
y_test_9=(y_test==9).astype(int)
'''