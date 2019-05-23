#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Intelligent and Interactive Systems 
Spring 2019
Lab 2: Software for Machine Learning 
"""
# Import datasets, classifiers and performance metrics

import numpy as np
import cv2

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


'''
Global Variables 
'''
SIMPLE_EMBEDDING=False
MANUAL_SPLIT    =True
DATA_2D         =True


'''
Functions 
'''
def displayImages(digitsIm):
    '''
    This fuction allows viweing the data
    '''
    i = 0
    for image in digitsIm:
        if(i < 10):
            # TO-DO
            # Visualize your data
        else:
             break

    
def holdOut(fnDigits, fnData, nSamples, percentSplit=0.8):
    '''
    This function splits the data into training and test sets
    '''
    if(MANUAL_SPLIT):
            # TO-DO
    else:
        trainData, testData, trainLabels, expectedLabels = train_test_split(fnData, fnDigits.target,
                                                                            test_size=(1.0-percentSplit), random_state=0)

    return trainData, trainLabels, testData, expectedLabels


def plotData(X):
    '''
    This function plots either 2D data or 3D data
    '''
    if(DATA_2D):
        plt.scatter(X[:,0], X[:,1])
        plt.show()
    else:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(X[:,0], X[:,1], X[:,2])
        
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        
        plt.show()
        

'''
Main Program 
'''
# Data Visualization 
### TO-DO
### Function to display the data
displayImages(digits.images)


# Transforming the data in a (samples, features) matrix:


# Data Dimension
if(DATA_2D):
    nComp = 2
else:
    nComp = 3

# Data Reduction 
if(SIMPLE_EMBEDDING):
    #  Dimensionality Reduction PCA 
else:
    # Manifold embedding with tSNE

    
# Ploting Data
plotData(X_trans)

# Manually Split your data
X_train, X_labels, X_test, X_trueLabels = holdOut(digits, data, n_samples)
print(data.shape)
print(X_labels.shape)
print(X_trueLabels.shape)


# k-NearestNeighbour Classifier 


# Training the model


# Display classifier results


# Cross Validation 


# Support Vector Machines


# Training 


# Prediction


# Cross Validation
