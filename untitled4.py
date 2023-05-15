#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 11:21:36 2022

@author: joseph.melville
"""

















#create an initial condition with all random ids
#run it through a mode filter
#track the total number of different neighbors after each step
#verify that it never goes up


import numpy as np
import torch
import matplotlib.pyplot as plt
import functions as fs
from tqdm import tqdm

im = torch.randint(64,(2**10,2**10))


nn = fs.num_diff_neighbors(im[None,None], window_size=3, pad_mode='circular').sum()



plt.imshow(im)







coords, samples, index = find_sample_coords(im, cov_mat=torch.Tensor([[25,0],[0,25]]), num_samples=64, bcs=['p','p'])

imf = im.flatten()
arrays = imf[index]
arrays_e = torch.zeros(imf.shape)
imf0 = torch.zeros(imf.shape)


for i in tqdm(range(arrays.shape[1])):
    uid, counts = torch.unique(arrays[:,i], return_counts=True)
    
    j = torch.where(uid==imf[i])[0]
    if len(j)==0: arrays_e[i] = 128
    else: arrays_e[i] = 128-counts[j]

    j0 = torch.argmax(counts)
    imf0[i] = uid[j0]


arrays0 = imf0[index]
arrays0_e = torch.zeros(imf0.shape)


for i in tqdm(range(arrays.shape[1])):
    uid, counts = torch.unique(arrays0[:,i], return_counts=True)
    
    j = torch.where(uid==imf0[i])[0]
    if len(j)==0: arrays0_e[i] = 128
    else: arrays0_e[i] = 128-counts[j]



torch.sum((arrays_e-arrays0_e)<0)
torch.sum((arrays_e-arrays0_e)==0)



torch.sum(arrays_e)>torch.sum(arrays0_e)







def count_occurance(arrays):
    counts = (arrays[:,None,:]==arrays[None,:,:]).sum(0)
    return counts


def count_energy(arrays, miso_matrix, cut):
    
    # Cutoff and normalize misorientation matrix (for cubic symetry, cut=63 degrees effectively normalizes)
    if cut==0: cut = 1e-10
    cut_rad = cut/180*np.pi
    miso_mat_norm = miso_matrix/cut_rad
    miso_mat_norm[miso_mat_norm>1] = 1
    
    #Mark where neighbor IDs do not match
    diff_matricies = (arrays[:,None,:]!=arrays[None,:,:]).float()

    #Find the indicies of ones
    i, j, k = torch.where(diff_matricies)
    
    #Find the ids of each of those indices
    i2 = arrays[i,k].long()
    j2 = arrays[j,k].long()
    
    #Find the misorientations of the id pairs
    f = miso_mat_norm[i2,j2].float()
    
    #Place misorientations in place of all the ones
    diff_matricies[i,j,k] = f
    
    #Invert and sum
    energy = torch.sum(1-diff_matricies, dim=0)

    return energy


def find_mode(arrays, miso_matrix, cut):
    #Takes the mode of the array using torch.Tensor.cuda
    
    if cut==0: counts = count_occurance(arrays) #counts the number of occurances for each value
    else: counts = count_energy(arrays, miso_matrix, cut) #counts the energy value
    i = torch.argmax(counts, dim=0)[None,] #find the indices of the max
    mode = torch.gather(arrays, dim=0, index=i)[0] #selects those indices
    return mode





