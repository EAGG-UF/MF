#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:43:28 2022

@author: joseph.melville
"""
import numpy as np


      
def gradientDescent2(x, y, theta, alpha, m, numIterations):
    for i in range(0, numIterations):
        hypothesis = np.matmul(x, theta)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        
        tmp = np.matmul(x[0].T, loss[0])
        for i in range(1,len(loss)):
            tmp += np.matmul(x[i].T, loss[i])
        gradient = tmp/m
        
        # update
        theta = theta - alpha * gradient
    return theta


def genData2(numPoints):
    
    #alter this so that the argmin is returned
    
    x = np.random.rand(numPoints,3,3)
    w = np.array([.1, .5, 0.8])
    y = np.matmul(x,w)
    
    v = np.min(y, axis=1)[:,None]
    yy = (y==v).astype(float)
    
    return x, yy


# gen 100 points with a bias of 25 and 10 variance as a bit of noise
x, y = genData2(100)
m, n, _ = np.shape(x)
numIterations= 100000
alpha = 0.0005
theta = np.ones(n)
theta = gradientDescent2(x, y, theta, alpha, m, numIterations)
print(theta)


#SGD is not the solution to getting past the argmin
#maybe an iterative approach would work (expectation maximization kind of)
#how would I solve for p if I have f set (then I can solve for f given p)

#histogram approach


















