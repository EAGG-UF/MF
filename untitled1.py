#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 16:39:05 2022

@author: joseph.melville
"""

### Write up anisotropic MF
#Get isotropic MF running again
#Replace the count function with GPU version (eqivalency matrix, sum)
#H5, statistics, compare spparks isotropic
#Add anisotropy
#H5, statistics, compare with spparks anisotropic

### Estimate grain pair factors
#Create spparks training set (consistant misorientaions)
#Cacluate estimates - compare to actual

### Next question?
#What is the structure of the function I am fitting (assumed and estimated)?
#Are there features beside pair factors that may be important? Can I match experimental data statistics alread?
#If not, what other features would be useful?
#How would I incorperate those into this model?
#Is a two-point correlation one of those features, or is it redundant? Can I control the two point correlations through pair factors?






#Pull in the mode filter code and make sure it's working







import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import numpy as np
import functions as fs
import dropbox as db
import os
import numpy as np
from PIL import Image
import imageio
import torch
import matplotlib.pyplot as plt




# fp = '../PRIMME_all/PRIMME/data/spparks_sz(1024x1024)_ng(4096)_nsteps(100)_freq(1)_kt(0.66)_cut(0).h5'
# with h5py.File(fp, 'r') as f:
#     print(f['sim0'].keys())
#     ims = torch.from_numpy(f['sim0/ims_id'][:5].astype(float))
#     miso_matrix = torch.from_numpy(f['sim0/miso_matrix'][:])
# miso_matrices = miso_matrix[None,].repeat(5,1,1)

# fs.neighborhood_miso(ims, miso_matrices, window_size=3, pad_mode='circular')



# IPdb [3]: ims_unfold.shape
# torch.Size([5, 9, 1048576])

# IPdb [7]: miso_matrices.shape
# torch.Size([5, 4096, 4096])



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


def sample_cumsum(arrays):
    #"array" - shape=(array elements, number of arrays)
    #Chooses an index from each column in "array" by sampling from it's cumsum
    arrays_cumsum = torch.cumsum(arrays.T, dim=1)/torch.sum(arrays, dim=0)[:,None]
    sample_values = torch.rand(arrays_cumsum.shape[0]).to(arrays.device)
    sample_indices = torch.argmax((arrays_cumsum>sample_values.unsqueeze(1)).float(), dim=1)
    return sample_indices


def sample_counts(arrays, miso_matrix, cut):
    if cut==0: counts = count_occurance(arrays) #counts the number of occurances for each value
    else: counts = count_energy(arrays, miso_matrix, cut) #counts the energy value
    index = sample_cumsum(counts)[None,] #use this if you want to sample from the counts instead of choosing the max
    return torch.gather(arrays, dim=0, index=index)[0] #selects those indices


def normal_mode_filter(im, miso_matrix, cut=0, cov_mat=torch.Tensor([[25,0],[0,25]]), num_samples=64, bcs=['p','p'], memory_limit=1e9):
    
    #Create sampler
    cov_mat = cov_mat.to(im.device)
    mean_arr = torch.zeros(2).to(im.device)
    mvn = torch.distributions.MultivariateNormal(mean_arr, cov_mat)
    
    #Calculate the index coords
    arr0 = torch.arange(im.shape[0]).to(im.device)
    arr1 = torch.arange(im.shape[1]).to(im.device)
    coords = torch.cartesian_prod(arr0, arr1).float().transpose(0,1).reshape(1, 2, -1)
    
    num_batches = 2*num_samples*2*num_samples*im.numel()*64/8/memory_limit
    batch_size = int(im.numel()/num_batches)
    # batch_size = int(im.numel()/10)
    l = []
    coords_split = torch.split(coords, batch_size, dim=2)
    for c in coords_split: 
        
        #Sample neighborhoods
        samples = mvn.sample((num_samples, c.shape[2])).int().transpose(1,2) #sample separately for each site
        samples = torch.cat([samples, samples*-1], dim=0).to(im.device) #mirror the samples to keep a zero mean
        c = samples + c
    
        #Set bounds for the indices (wrap or clamp - add a reflect)
        if bcs[1]=='p': c[:,0,:] = c[:,0,:]%im.shape[0] #periodic or wrap
        else: c[:,0,:] = torch.clamp(c[:,0,:], min=0, max=im.shape[0]-1)
        if bcs[0]=='p': c[:,1,:] = c[:,1,:]%im.shape[1] #periodic or wrap
        else: c[:,1,:] = torch.clamp(c[:,1,:], min=0, max=im.shape[1]-1)
    
        #Flatten indices
        index = (c[:,1,:]+im.shape[1]*c[:,0,:]).long()
            
        #Gather the coord values and take the mode for each pixel (replace this with the matrix approach)   
        im_expand = im.reshape(-1,1).expand(-1, batch_size)
        im_sampled = torch.gather(im_expand, dim=0, index=index)
        # im_part = sample_counts(im_sampled, miso_matrix, cut)
        im_part = find_mode(im_sampled, miso_matrix, cut)
        l.append(im_part)
        
    im_next = torch.hstack(l).reshape(im.shape)
    
    return im_next


def run_mf(ic, ea, nsteps, cut, cov_mat=torch.Tensor([[25,0],[0,25]]), num_samples=64, miso_array=None, if_plot=False, bcs=['p','p'], memory_limit=1e9, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    
    # Setup
    im = torch.Tensor(ic).float().to(device)
    size = ic.shape
    ngrain = len(torch.unique(im))
    tmp = np.array([8,16,32], dtype='uint64')
    dtype = 'uint' + str(tmp[np.sum(ngrain>2**tmp)])
    if np.all(miso_array==None): miso_array = torch.Tensor(fs.find_misorientation(ea, mem_max=1) )
    miso_matrix = fs.miso_conversion(miso_array[None,]).to(device)[0]
    fp_save = './data/mf_sz(%dx%d)_ng(%d)_nsteps(%d)_cov(%d-%d-%d)_numnei(%d)_cut(%d).h5'%(size[0],size[1],ngrain,nsteps,cov_mat[0,0],cov_mat[1,1],cov_mat[0,1], num_samples, cut)
    
    # Run simulation
    log = [im.clone()]
    for i in tqdm(range(nsteps), 'Running MF simulation:'): 
        im = normal_mode_filter(im, miso_matrix, cut, cov_mat, num_samples, bcs, memory_limit=memory_limit)
        log.append(im.clone())
        if if_plot: plt.imshow(im[0,0,].cpu()); plt.show()
    
    ims_id = torch.stack(log)[:,None,].cpu().numpy()
    
    # Save Simulation
    with h5py.File(fp_save, 'a') as f:
        
        # If file already exists, create another group in the file for this simulaiton
        num_groups = len(f.keys())
        hp_save = 'sim%d'%num_groups
        g = f.create_group(hp_save)
        
        # Save data
        dset = g.create_dataset("ims_id", shape=ims_id.shape, dtype=dtype)
        dset2 = g.create_dataset("euler_angles", shape=ea.shape)
        dset3 = g.create_dataset("miso_array", shape=miso_array.shape)
        dset4 = g.create_dataset("miso_matrix", shape=miso_matrix.shape)
        dset[:] = ims_id
        dset2[:] = ea
        dset3[:] = miso_array #radians (does not save the exact "Miso.txt" file values, which are degrees divided by the cutoff angle)
        dset4[:] = miso_matrix.cpu() #same values as mis0_array, different format

    return ims_id


def image_covariance_matrix(im, min_max=[-200, 200], num_samples=8, bounds=['wrap','wrap']):
    #Sample and calculate the index coords
    mvn = torch.distributions.Uniform(torch.Tensor([min_max[0]]).to(im.device), torch.Tensor([min_max[1]]).to(im.device))
    samples = mvn.sample((num_samples, 2, im.numel()))[...,0].int()
    
    arr0 = torch.arange(im.shape[0]).to(im.device)
    arr1 = torch.arange(im.shape[1]).to(im.device)
    coords = torch.cartesian_prod(arr0, arr1).float().transpose(0,1).reshape(1, 2, -1)
    coords = samples+coords
    
    #Set bounds for the indices
    if bounds[1]=='wrap': coords[:,0,:] = coords[:,0,:]%im.shape[0]
    else: coords[:,0,:] = torch.clamp(coords[:,0,:], min=0, max=im.shape[0]-1)
    if bounds[0]=='wrap': coords[:,1,:] = coords[:,1,:]%im.shape[1]
    else: coords[:,1,:] = torch.clamp(coords[:,1,:], min=0, max=im.shape[1]-1)
    
    #Flatten indices
    index = (coords[:,1,:]+im.shape[1]*coords[:,0,:]).long()
        
    # #Gather the coord values and take the mode for each pixel      
    # im_expand = im.reshape(-1,1).expand(-1, im.numel())
    # v = torch.gather(im_expand, dim=0, index=index)
    # im_next = torch.mode(v.cpu(), dim=0).values.reshape(im.shape)
    # im_next = im_next.to(device)
    # # im_next = fs.rand_mode(v).reshape(im.shape)
    
    #Find the covariance matrix of just the samples that equal the mode of the samples
    index_mode = im.reshape(-1)[index]==im.reshape(-1)
    samples_mode = samples.transpose(1,2)[index_mode].transpose(1,0).cpu().numpy()
    # plt.plot(samples_mode[0,:1000], samples_mode[1,:1000], '.'); plt.show()
    
    return np.cov(samples_mode)


def find_sample_coords(im, cov_mat=torch.Tensor([[25,0],[0,25]]), num_samples=64, bcs=['p','p']):
    
    #Create sampler
    cov_mat = cov_mat.to(im.device)
    mean_arr = torch.zeros(2).to(im.device)
    mvn = torch.distributions.MultivariateNormal(mean_arr, cov_mat)
    
    #Calculate the index coords
    arr0 = torch.arange(im.shape[0]).to(im.device)
    arr1 = torch.arange(im.shape[1]).to(im.device)
    coords = torch.cartesian_prod(arr0, arr1).float().transpose(0,1).reshape(1, 2, -1)
    
    samples = mvn.sample((num_samples, coords.shape[2])).int().transpose(1,2) #sample separately for each site
    samples = torch.cat([samples, samples*-1], dim=0).to(im.device) #mirror the samples to keep a zero mean
    c = samples + coords #shifted samples
    
    #Set bounds for the indices (wrap or clamp - add a reflect)
    if bcs[1]=='p': c[:,0,:] = c[:,0,:]%im.shape[0] #periodic or wrap
    else: c[:,0,:] = torch.clamp(c[:,0,:], min=0, max=im.shape[0]-1)
    if bcs[0]=='p': c[:,1,:] = c[:,1,:]%im.shape[1] #periodic or wrap
    else: c[:,1,:] = torch.clamp(c[:,1,:], min=0, max=im.shape[1]-1)
    
    #Flatten indices
    index = (c[:,1,:]+im.shape[1]*c[:,0,:]).long()
    
    return coords, samples, index


def find_pf_matrix(id_ratios):
    #finds pair-factor (pf) matrix
    #pf matrix relates ID pair-factors to ID probability of adoption using ID ratios
    
    #Find reference indicies to create the matrix
    num_ids = len(id_ratios)
    num_pf = int(num_ids*(num_ids-1)/2) #number of pair-factors (combinations of IDs)
    i, j = torch.where(torch.triu(torch.ones(num_ids, num_ids), 1))
    k = torch.arange(num_pf)

    #Place ID rations in matrix
    A = torch.zeros(num_ids, num_pf)
    A[i,k] = id_ratios[j]
    A[j,k] = id_ratios[i]
    
    #Record order of pair-factor location indices
    pf_loc = torch.stack([i,j])
    
    return A, pf_loc





#Run mode fitler and compute grain stats for a large microstructure
ic, ea, _ = fs.voronoi2image(size=[1024, 1024], ngrain=4096) 

ims_id = run_mf(ic, ea, nsteps=2000, cut=25, cov_mat=torch.Tensor([[10,0],[0,10]]), num_samples=64, memory_limit=1e10)

hps = ['./data/mf_sz(1024x1024)_ng(4096)_nsteps(2000)_cov(10-10-0)_numnei(64)_cut(25).h5']
gps = ['sim0']
fs.compute_grain_stats(hps, gps)


#Calculate stats and plots for small microstructure (have to create seperately with about code)
hps = ['./data/mf_sz(1024x1024)_ng(4096)_nsteps(1000)_cov(25-25-0)_numnei(64)_cut(0).h5',
       './data/mf_sz(1024x1024)_ng(4096)_nsteps(1000)_cov(25-25-0)_numnei(64)_cut(25).h5',
       './data/mf_sz(1024x1024)_ng(4096)_nsteps(2000)_cov(10-10-0)_numnei(64)_cut(0).h5',
       './data/mf_sz(1024x1024)_ng(4096)_nsteps(2000)_cov(10-10-0)_numnei(64)_cut(25).h5']
gps = ['sim0', 'sim0', 'sim0', 'sim0']
fs.compute_grain_stats(hps, gps)
fs.make_time_plots(hps, gps, scale_ngrains_ratio=0.20)












# Estimation grain pair energy factors (should be equal to misorientations in this case)


#Create a small isotropic simulation with MF to work with


ic, ea, _ = fs.voronoi2image(size=[64, 64], ngrain=64) 
ims_id = run_mf(ic, ea, nsteps=100, cut=25, cov_mat=torch.Tensor([[25,0],[0,25]]), num_samples=64, memory_limit=1e10)

fp = './data/mf_sz(64x64)_ng(64)_nsteps(100)_cov(25-25-0)_numnei(64)_cut(0).h5'
with h5py.File(fp) as f:
    ims_id = f['sim0/ims_id'][:]
    miso_matrix = f['sim0/miso_matrix'][:] #cut off angle was zero, so this isn't ground truth (should actually get all ones)


im = torch.from_numpy(ims_id[0,0])
im2 = torch.from_numpy(ims_id[1,0])

coords, samples, index = find_sample_coords(im, cov_mat=torch.Tensor([[25,0],[0,25]]), num_samples=64)
 
im_expand = im.reshape(-1,1).expand(-1, im.numel())
im_sampled = torch.gather(im_expand, dim=0, index=index)




pf = torch.zeros(miso_matrix.shape) #hold pair factors
c = torch.zeros(miso_matrix.shape) #count how many times a pair factor is seen

#find the actual factors I am solving for
# if cut==0: cut = 1e-10
# cut_rad = cut/180*np.pi
# miso_mat_norm = miso_matrix/cut_rad
# miso_mat_norm[miso_mat_norm>1] = 1
#since cut=0, I should actually find all ones

for ii in range(im_sampled.shape[1]):
    s = im_sampled[:,ii] #samples
    n = im2.reshape(-1)[ii] #ID switched to
    u = torch.unique(s) #IDs present
    y = (u==n)
    r = (s[:,None]==u[None,]).sum(0)/len(s)
    A = torch.Tensor([[r[1],r[2],0],[r[0],0,r[2]],[0,r[0],r[1]]])
    x = torch.linalg.inv(A)[:,y==True][:,0]
    
    tmp = torch.arange(len(r))
    i, j = torch.meshgrid(tmp, tmp)
    kkk = torch.stack([i.flatten(), j.flatten()])
    kk = torch.unique(torch.sort(kkk, 0)[0], dim=1)
    k = kk[:,kk[0]!=kk[1]]
    
    for i in range(k.shape[1]):  
        j0, j1 = k[:,i]
        k0, k1 = u[j0].long(), u[j1].long()
        pf[k0, k1] += x[i]
        pf[k1, k0] += x[i]
        c[k0, k1] += 1
        c[k1, k0] += 1









# Make a test case with three grains

r_tmp = torch.arange(1,20)
r0, r1, r2 = torch.meshgrid(r_tmp,r_tmp,r_tmp)
tot = r0+r1+r2

r0 = (r0/tot).flatten()
r1 = (r1/tot).flatten()
r2 = (r2/tot).flatten()

bins = (torch.arange(11))/10
plt.plot(np.histogram(r0, bins=bins)[0])
plt.plot(np.histogram(r1, bins=bins)[0])
plt.plot(np.histogram(r2, bins=bins)[0])
plt.show()

rr = torch.stack([r0,r1,r2])


#If I sample all combinations of the same proportions, I get the perfect ratio for the pair factors
#If I sample all combinations of half, I get the exact values (??? - nope, just exact ratio)
#Tomorrow: 
    #Try just half ratios with no symmetry (same result?)
    #Try on different misorientations (get right ratios? need to change sampling proportions?)
    #Work through average of the symetrric sampled ratios above (pattern? use pattern to correct for not syymetric sampling? correct for no equal sampling?)


# I would like to see if my scaling helps at all
# To test it, I will will break it down to a very small problem



# If I have the actual values, I can estimate them directly
# Even if I know that they were probabilities I could estimate them
# However, what I have is a argmin of the probabilities




import scipy.optimize as so




r0=0.2;r1=0.3;r2=1-r1-r0
# r0=1/3;r1=1/3;r2=1/3+0.000001
# r0=1/2;r1=r0;r2=r0+0.000001
rr = torch.Tensor([[r0,r1,r2],[r0,r2,r1],[r1,r0,r2],[r1,r2,r0],[r2,r0,r1],[r2,r1,r0]]).T


# f0=.11;f1=.5;f2=.789
f0=1;f1=1;f2=1
mm_t = torch.Tensor([[0,f0,f1],[f0,0,f2],[f1,f2,0]])
# mm_t = torch.Tensor([[0,.9,.5],[.9,0,.1],[.5,.1,0]])

ff = 2*torch.Tensor([f0/(f0+f2), f1/(f1+f2), f0/(f0+f1), f2/(f1+f2), f1/(f0+f1), f2/(f0+f2)])
pf = torch.zeros(mm_t.shape) #hold pair factors
c = torch.zeros(mm_t.shape) #count how many times a pair factor is seen

# a = 2/(f0+f1)
# b = 2/(f0+f2)
# c = 2/(f1+f2)
# A = torch.Tensor([[r[1],r[2],0],[0,-r[2],r[0]],[-r[1],0,-r[0]]])

a = f0+f1+f2

ff[0]*r[1]+ff[1]*r[2]
ff[2]*r[0]+ff[3]*r[2]
ff[4]*r[0]+ff[5]*r[1]

1-(ff[0]*r[1]+ff[1]*r[2])
1-(ff[2]*r[0]+ff[3]*r[2])
1-(ff[4]*r[0]+ff[5]*r[1])



f0=.5;f1=.5;f2=.5
r0=0.2;r1=0.3;r2=1-r1-r0

r = torch.Tensor([r0, r1, r2])
f = torch.Tensor([3*f0/(f0+f1+f2), 3*f1/(f0+f1+f2), 3*f2/(f0+f1+f2)])

r[1]*f[0] + r[2]*f[1] + r[0]*f[0] + r[2]*f[2] + r[0]*f[1] + r[1]*f[2]


(r0+r1)*f1 + (r0+r2)*f1 + (r1+r2)*f2


f0=.2;f1=.4;f2=1
ff = 2*torch.Tensor([f0/(f0+f2), f1/(f1+f2), f0/(f0+f1), f2/(f1+f2), f1/(f0+f1), f2/(f0+f2)])

r0=0.2;r1=0.3;r2=1-r1-r0
    
p0 = 1 - ff[0]*r[1] - ff[1]*r[2]
p1 = 1 - ff[2]*r[0] - ff[3]*r[2]
p2 = 1 - ff[4]*r[0] - ff[5]*r[1]

print('%.4f + %.4f + %.4f = %.4f'%(p0,p1,p2,p0+p1+p2))


f0=.8;f1=.7;f2=1
ff = 2*torch.Tensor([f0/(f0+f2), f1/(f1+f2), f0/(f0+f1), f2/(f1+f2), f1/(f0+f1), f2/(f0+f2)])


log = []
log0 = []

log_A = []
log_tmp = []
log_b = []
for ii in tqdm(range(rr.shape[1])): 

    r = rr[:,ii]
    A = torch.Tensor([[r[1],r[2],0],[0,-r[2],r[0]],[-r[1],0,-r[0]]])
    tmp = torch.Tensor([0,2*r[2],2*(r[0]+r[1])])
    
    log_A.append(A)
    log_tmp.append(tmp)
    
    x_t = ff[:3]
    b_t = torch.matmul(A,x_t) + tmp
    b = (torch.min(b_t)!=b_t).float()
    log_b.append(b)
 
    
A = torch.cat(log_A) 
tmp = torch.cat(log_tmp) 
b = torch.cat(log_b) 

b_t = torch.matmul(A,x_t) + tmp
x = torch.linalg.lstsq(A, b-tmp).solution


# log_b2 = []
for ii in tqdm(range(rr.shape[1])): 

    r = rr[:,ii]
#     A = torch.Tensor([[r[1],r[2],0],[0,-r[2],r[0]],[-r[1],0,-r[0]]])
#     tmp = torch.Tensor([0,2*r[2],2*(r[0]+r[1])])
#     b_t = torch.matmul(A,x) + tmp
#     b = (torch.min(b_t)!=b_t).float()
#     log_b2.append(b)
 
# b2 = torch.cat(log_b2)    
 
# b = torch.cat(log_b) 

# print(torch.sum(b!=b2))

# torch.where(b!=b2)

# b2[b!=b2]
 

    # a0 = torch.Tensor([r[0]*(f0+f1), r[1]*(f0+f2), r[2]*(f1+f2)])
    # a1 = torch.Tensor([f0*(r[0]+r[1]), f1*(r[0]+r[2]), f2*(r[1]+r[2])])
    
    # if torch.argmin(a0)!=torch.argmin(a1): break
    
    A, pf_loc = find_pf_matrix(r)
    
    # A = torch.Tensor([[r[1],r[2],0],[0,-r[2],r[0]],[-r[1],0,-r[0]]])
    # A0 = torch.Tensor([[-r[2],0,-r[1]],[r[2],-r[0],0],[0,r[0],r[1]]])
    # A1 = torch.Tensor([[-r[2],0,-r[1]],[r[2],-r[0],0],[0,r[0],r[1]]])
    
    # A0 = torch.Tensor([[r[1],r[2],0],[0,-r[2],r[0]],[-r[1],0,-r[0]]])
    # A1 = torch.Tensor([[r[1],r[2],0],[0,-r[2],r[0]],[-r[1],0,-r[0]]])
    # A = torch.cat([A0,A1])
    # print(torch.linalg.cond(A))
    
    

    # x_t = ff[:3]
    # tmp0 = torch.Tensor([0,2*r[2],2*(r[0]+r[1])])
    # tmp1 = torch.Tensor([0,2*r[2],2*(r[0]+r[1])])
    # tmp = torch.cat([tmp0,tmp1])
    # b_t = torch.matmul(A,x_t) + tmp
    # b = (torch.min(b_t)!=b_t).float()
    
    # #finds the right answer
    # #adds up to 1
    # #but has negative probabilities
    
    # b = torch.Tensor([0,1,1,0,1,1])

    # x = torch.linalg.lstsq(A, b-tmp).solution
    
    # bb = torch.matmul(A,x) + tmp
    
    
    # log.append(x)
    
     
    x_t = torch.Tensor([f0,f1,f2])[:,None]
    tmp2 = torch.matmul(A,x_t)
    tmp3 = torch.min(tmp2)
    # b = tmp2[:,0]#(tmp2>tmp3)[:,0].float()
    b = (tmp2>tmp3)[:,0].float()
    b0 = (tmp2<=tmp3)[:,0].float()
    
    log0.append(tmp2[:,0])
    # log0.append(b0)
    
    # i = torch.argmin(tmp2)
    
    
    # if i==0: pf[0]+=r[0]; pf[2]+=r[2]; pf[1]+=r[0]; pf[2]+=r[1]
    # elif i==1: pf[0]+=r[1]; pf[1]+=r[2]; pf[1]+=r[0]; pf[2]+=r[1]
    # elif i==2: pf[0]+=r[1]; pf[1]+=r[2]; pf[0]+=r[0]; pf[2]+=r[2]
    # else: print('error')
    
    
    
    # log.append(b)
    # b = torch.stack(log)[:,0]
    
    
    
    
    # A = torch.stack((rr[1,:], rr[2,:])).T
    # x_t = ff[:2,None]
    
   
    
    # tmp = torch.linalg.pinv(A)
    
    # torch.matmul(tmp,b)
    
    
    x = torch.linalg.lstsq(A, b).solution
    
    # x = torch.from_numpy(so.nnls(A, b)[0])
    
    # x = so.lsq_linear(A.numpy().astype('double'), b.numpy().astype('double'), bounds=(0, 1)).x
    
    log.append(torch.Tensor(x))




l = torch.stack(log)
plt.plot(l); plt.show()
mn = torch.mean(l, dim=0)
print(mn)





p = torch.stack(log0)
p0 = p[:,0]
p1 = p[:,1]
p2 = p[:,2]

bd = np.linspace(0,1,11)
bb = bd[:-1]+0.5/(len(bd)-1)
a = np.histogramdd(rr.T, bins=[bd,bd,bd])[0]
a0 = np.histogramdd(rr.T, bins=[bd,bd,bd], weights=p0)[0]
a1 = np.histogramdd(rr.T, bins=[bd,bd,bd], weights=p1)[0]
a2 = np.histogramdd(rr.T, bins=[bd,bd,bd], weights=p2)[0]

a0[a!=0] = a0[a!=0]/a[a!=0]
a1[a!=0] = a1[a!=0]/a[a!=0]
a2[a!=0] = a2[a!=0]/a[a!=0]





j,i = np.where(a0[0,:,:]>0)
plt.plot(bb[j], a0[i,j,0], '.')
AA = torch.from_numpy(np.stack([np.ones(len(j)),bb[j]]).T).float()
BB = torch.from_numpy(a0[i,j,0]).float()
print(torch.linalg.lstsq(AA, BB).solution[1])

plt.plot(bb[j], a0[i,0,j])
AA = torch.from_numpy(np.stack([np.ones(len(j)),bb[j]]).T).float()
BB = torch.from_numpy(a0[i,0,j]).float()
print(torch.linalg.lstsq(AA, BB).solution[1])


j,i = np.where(a1[:,0,:]>0)
plt.plot(bb[j], a1[j,i,0])
AA = torch.from_numpy(np.stack([np.ones(len(j)),bb[j]]).T).float()
BB = torch.from_numpy(a1[j,i,0]).float()
print(torch.linalg.lstsq(AA, BB).solution[1])

plt.plot(bb[j], a1[0,i,j])
AA = torch.from_numpy(np.stack([np.ones(len(j)),bb[j]]).T).float()
BB = torch.from_numpy(a1[0,i,j]).float()
print(torch.linalg.lstsq(AA, BB).solution[1])


j,i = np.where(a2[:,:,0]>0)
plt.plot(bb[j], a2[j,0,i])
AA = torch.from_numpy(np.stack([np.ones(len(j)),bb[j]]).T).float()
BB = torch.from_numpy(a2[j,0,i]).float()
print(torch.linalg.lstsq(AA, BB).solution[1])

plt.plot(bb[j], a2[0,j,i])
AA = torch.from_numpy(np.stack([np.ones(len(j)),bb[j]]).T).float()
BB = torch.from_numpy(a2[0,j,i]).float()
print(torch.linalg.lstsq(AA, BB).solution[1])





i,j,k = np.where(a0>0)
c0 = a0[i,j,k]
i0,j0,k0 = bb[i], bb[j], bb[k]
i,j,k = np.where(a1>0)
c1 = a1[i,j,k]
i1,j1,k1 = bb[i], bb[j], bb[k]
i,j,k = np.where(a2>0)
c2 = a2[i,j,k]
i2,j2,k2 = bb[i], bb[j], bb[k]

fig = plt.figure()

ax = plt.axes(projection='3d')
# ax.view_init(0, 90)


# f0=0.2;f1=0.4;f2=1

tmp = np.linspace(0,1,2)
X,Y,Z = np.meshgrid(tmp,tmp,tmp)

AA = torch.from_numpy(np.stack([np.ones(len(i0)),i0,j0,k0]).T).float()
BB = torch.from_numpy(c0).float()
x = torch.linalg.lstsq(AA, BB).solution.numpy()
r = np.array([[1,0,0],[0,1,0],[0,0,1]])
print(x[0]+x[1]*r[0]+x[2]*r[1]+x[3]*r[2])

AA = torch.from_numpy(np.stack([np.ones(len(i1)),i1,j1,k1]).T).float()
BB = torch.from_numpy(c1).float()
x = torch.linalg.lstsq(AA, BB).solution.numpy()
r = np.array([[1,0,0],[0,1,0],[0,0,1]])
print(x[0]+x[1]*r[0]+x[2]*r[1]+x[3]*r[2])

AA = torch.from_numpy(np.stack([np.ones(len(i2)),i2,j2,k2]).T).float()
BB = torch.from_numpy(c2).float()
x = torch.linalg.lstsq(AA, BB).solution.numpy()
r = np.array([[1,0,0],[0,1,0],[0,0,1]])
print(x[0]+x[1]*r[0]+x[2]*r[1]+x[3]*r[2])



# ax.scatter3D(i0,j0,k0)
# ax.scatter3D(i1,j1,k1)
# ax.scatter3D(i2,j2,k2)
# ax.scatter3D(i0,j0,k0,c=1-c0, depthshade=0, cmap='Blues')
# ax.scatter3D(i1,j1,k1,c=1-c1, depthshade=0, cmap='Oranges')
# ax.scatter3D(i2,j2,k2,c=1-c2, depthshade=0, cmap='Greens')
ax.scatter3D(i0,j0,1-c0)
ax.scatter3D(i1,j1,1-c1)
ax.scatter3D(i2,j2,1-c2)
ax.set_xlim([1,0])
ax.set_ylim([0,1])
# ax.set_zlim([1,0])

ax.set_xlabel('r0')
ax.set_ylabel('r1')
ax.set_zlabel('r2')

#find those lines, which gives you the inequalities, which gives you the pair_factors







#find the locations where blue and yellow overlap


b0 = (a0!=0)*(a0!=1)*(a1!=0)*(a1!=1)
b1 = (a0!=0)*(a0!=1)*(a2!=0)*(a2!=1)
b2 = (a1!=0)*(a1!=1)*(a2!=0)*(a2!=1)

i,j,k = np.where(b0)
i0,j0,k0 = bb[i], bb[j], bb[k]
i,j,k = np.where(b1)
i1,j1,k1 = bb[i], bb[j], bb[k]
i,j,k = np.where(b2)
i2,j2,k2 = bb[i], bb[j], bb[k]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(i0,j0,k0)
ax.scatter3D(i1,j1,k1)
ax.scatter3D(i2,j2,k2)
ax.set_xlim([1,0])
ax.set_ylim([0,1])
ax.set_zlim([1,0])

ax.set_xlabel('r0')
ax.set_ylabel('r1')
ax.set_zlabel('r2')



#fit a line to those coordinates



AA = torch.from_numpy(np.stack([np.ones(len(i0)),j0,k0]).T).float()
BB = torch.from_numpy(i0).float()
print(torch.linalg.lstsq(AA, BB).solution)

AA = torch.from_numpy(np.stack([np.ones(len(i1)),j1,k1]).T).float()
BB = torch.from_numpy(i1).float()
print(torch.linalg.lstsq(AA, BB).solution)

AA = torch.from_numpy(np.stack([np.ones(len(i2)),j2,k2]).T).float()
BB = torch.from_numpy(i2).float()
print(torch.linalg.lstsq(AA, BB).solution)









#!!!!!
#I could try the approach where there an underlying equation all of them are using
#!!!!!





#The bigger the difference between them, the higher the estimate
r0=.2;r1=.3;r2=.5 #1.333 (4/3)
r0=.1;r1=.2;r2=.7 #2.3810
r0=.2;r1=.2;r2=.6 #1.666
r0=.1;r1=.1;r2=.8 #3.333 (10/3)
r0=.05;r1=.05;r2=.9 #6.666 (20/3)
r0=.005;r1=.005;r2=.99 #66.666 (200/3)

r0=.3;r1=.3;r2=.4 #1.111
r0=1/3;r1=1/3;r2=.1/3.00001 #1
r0=0;r1=1/2;r2=1/2 #0.666

r0=0.01;r1=0.495;r2=0.495 #2/3




r0=.2;r1=.3;r2=.5 #1.333 (4/3)


(r1-r2)/(r0*td), (r0-r1)/(r2*td), (r0-r2)/(r1*td)

(r1-r2)/(r0*td)

r1/r0/td - r2/r0/td


td = (r0-r1)+(r0-r2)+(r1-r2)
td = 2*(r0-r2)

aaa = np.mean([(r1-r2)/(r0*td), (r0-r1)/(r2*td), (r0-r2)/(r1*td)])
aaa = np.mean([(r1-r2)/(r0*td), (r0-r1)/(r2*td), 1/(r1*2)])

td = (r1-r0)+(r0-r2)+(r1-r2)
td = 2*(r1-r2)

aaa = np.mean([(r1-r2)/(r0*td), (r1-r0)/(r2*td), (r0-r2)/(r1*td)])



td = (r0-r1)+(r0-r2)+(r1-r2)
a0 = np.mean([(r1-r2)/(r0*td), (r0-r1)/(r2*td), (r0-r2)/(r1*td)])

td = (r1-r0)+(r0-r2)+(r1-r2)
a1 = np.mean([(r1-r2)/(r0*td), (r1-r0)/(r2*td), (r0-r2)/(r1*td)])

td = (r1-r0)+(r2-r0)+(r1-r2)
a2 = np.mean([(r1-r2)/(r0*td), (r1-r0)/(r2*td), (r2-r0)/(r1*td)])

td = (r1-r0)+(r2-r0)+(r2-r1)
a3 = np.mean([(r2-r1)/(r0*td), (r1-r0)/(r2*td), (r2-r0)/(r1*td)])

td = (r0-r1)+(r2-r0)+(r2-r1)
a4 = np.mean([(r2-r1)/(r0*td), (r0-r1)/(r2*td), (r2-r0)/(r1*td)])

td = (r0-r1)+(r0-r2)+(r2-r1)
a5 = np.mean([(r2-r1)/(r0*td), (r0-r1)/(r2*td), (r0-r2)/(r1*td)])

np.mean([a0,a1,a2,a3,a4,a5])



aaa = np.mean([(r1-r2)/(r0*td), (r0-r1)/(r2*td), (r0-r2)/(r1*td)])
print(aaa)



tn = r0+r1+r2

r0/((r1-r2)*tn)


td0 = (r0-r1)+(r0-r2)
td0/(r0*td)

td1 = (r0-r1)+(r1-r2)
td1/(r1*td)

td2 = (r0-r2)+(r1-r2)
td2/(r2*td)



(r0-r1)/(r0*td)



r2-r0

(r0+r1+r2)
2/3


.7


r0=1/2;r1=0.00001;r2=0.000001 = 4/3


1/(r0/(r1+r2)+r1/(r0+r2)+r2/(r0+r1))

1/(r0/(r0+r1) + r1/(r0+r1) + r0/(r0+r2) + r2/(r0+r2) + r1/(r1+r2) + r2/(r1+r2) + r2/(r0+r1) + r1/(r0+r2) + r0/(r1+r2))

(r0-r1)/r0 + (r0-r1)/r1 + (r0-r2)/r0 + (r0-r2)/r2 + (r1-r2)/r1 + (r1-r2)/r2 


(r0-r1)/(r0-r2) + (r0-r2)/(r0-r1) + (r1-r0)/(r1-r2) + (r1-r2)/(r1-r0)



4/3
3/4



1/(r0+r1+r2)




np.mean([2*f0/(f0+f1), 2*f1/(f0+f1), 2*f0/(f0+f2), 2*f2/(f0+f2), 2*f1/(f1+f2), 2*f2/(f1+f2)])




#The ratio are what are actually estimated
#Those are only correct is there is a symmetric sampling of the ratios.
#How do I scalethings when it's not symmetric to still get the right ratios?





ic, ea, _ = fs.voronoi2image(size=[64, 64], ngrain=64) 
ims_id = run_mf(ic, ea, nsteps=100, cut=25, cov_mat=torch.Tensor([[25,0],[0,25]]), num_samples=64, memory_limit=1e10)

fp = './data/mf_sz(64x64)_ng(64)_nsteps(100)_cov(25-25-0)_numnei(64)_cut(0).h5'
with h5py.File(fp) as f:
    ims_id = f['sim0/ims_id'][:]
    miso_matrix = f['sim0/miso_matrix'][:]



im = torch.from_numpy(ims_id[0,0])
im2 = torch.from_numpy(ims_id[1,0])

coords, samples, index = find_sample_coords(im, cov_mat=torch.Tensor([[25,0],[0,25]]), num_samples=64)
 
im_expand = im.reshape(-1,1).expand(-1, im.numel())
im_sampled = torch.gather(im_expand, dim=0, index=index)





mm_t = torch.Tensor((miso_matrix>0))
# mm_t = torch.Tensor(miso_matrix)

# mm_t = torch.rand(miso_matrix.shape)

log = []
log1 = []

pf = torch.zeros(miso_matrix.shape) #hold pair factors
c = torch.zeros(miso_matrix.shape) #count how many times a pair factor is seen

for ii in range(im_sampled.shape[1]):

    s = im_sampled[:,ii] #samples
    n = im2.reshape(-1)[ii] #ID switched to
    u = torch.unique(s) #IDs present
    
    if len(u)>1:
    
        b = (u!=n).float()
        r = (s[:,None]==u[None,]).sum(0)/len(s)
        A, pf_loc = find_pf_matrix(r)
        
        # tmp = torch.matmul(A,A.T)
        # tmp = torch.linalg.inv(tmp)
        # tmp = torch.matmul(A.T,tmp)
        # x = torch.matmul(tmp,b)
        
        x = torch.linalg.lstsq(A, b).solution
        
        i = u[pf_loc[0]].long()
        j = u[pf_loc[1]].long()
        
        pf[i,j] += x
        c[i,j] += 1
        
        ij = c>0
        mm = pf.clone()
        mm[ij] = mm[ij]/c[ij]
        mm += mm.T
        e = torch.mean(torch.abs(mm-mm_t))
        log.append(e)
        log1.append(torch.linalg.cond(A))


plt.plot(log1); plt.show()

plt.plot(log); plt.show()
aaa = torch.abs(mm-mm_t).flatten()
bbb = c.flatten()
plt.plot(bbb, aaa, '.'); plt.show()


print(torch.min(mm))
print(torch.max(mm))
print(log[-1])


#run the simulation witt mm as the new misorientations
#are all the same decisions made?



log = []
for ii in range(im_sampled.shape[1]):

    s = im_sampled[:,ii] #samples
    n = im2.reshape(-1)[ii] #ID switched to
    u = torch.unique(s) #IDs present
    
    if len(u)>1:
    
        b = (u==n).float()
        r = (s[:,None]==u[None,]).sum(0)/len(s)
        A, pf_loc = find_pf_matrix(r)
        
        i = u[pf_loc[0]].long()
        j = u[pf_loc[1]].long()
        
        x = mm[i,j]
        
        tmp0 = torch.matmul(A,x)
        tmp1 = (torch.min(tmp0)==tmp0).float()
        
        log.append(torch.all(b==tmp1))
        


tmp = torch.stack(log).float()
torch.mean(tmp)

#Is the sampling symmetric?
#If I do more training, will it get better? (more r combos, or ID combos)











#TOMORROW
#Finish verifying Lin's SPPARKS runs
#Estimate misorientation
#Estimate growth speed
#Troublshoot PRIMME ("gid_to_msio"?)
#Clean up files - put on git and maybe PyPI
#Get PRIMME working on hipergator









#Show that MF is consistent

ic, ea, _ = fs.voronoi2image(size=[512, 512], ngrain=1024) 
for _ in range(10):
    ims_id = run_mf(ic, ea, nsteps=200, cut=0, cov_mat=torch.Tensor([[25,0],[0,25]]), num_samples=64, memory_limit=1e10)
    ims_id = run_mf(ic, ea, nsteps=200, cut=25, cov_mat=torch.Tensor([[25,0],[0,25]]), num_samples=64, memory_limit=1e10)

for _ in range(10):
    ic, ea, _ = fs.voronoi2image(size=[512, 512], ngrain=1024) 
    ims_id = run_mf(ic, ea, nsteps=200, cut=0, cov_mat=torch.Tensor([[25,0],[0,25]]), num_samples=64, memory_limit=1e10)
    ims_id = run_mf(ic, ea, nsteps=200, cut=25, cov_mat=torch.Tensor([[25,0],[0,25]]), num_samples=64, memory_limit=1e10)
    


hps = 20*['./data/mf_sz(512x512)_ng(1024)_nsteps(200)_cov(25-25-0)_numnei(64)_cut(0).h5',]+20*['./data/mf_sz(512x512)_ng(1024)_nsteps(200)_cov(25-25-0)_numnei(64)_cut(25).h5',]
gps = ['sim%d'%i for i in range(20)]+['sim%d'%i for i in range(20)]
fs.compute_grain_stats(hps, gps)
fs.make_time_plots(hps, gps, scale_ngrains_ratio=0.20)


with h5py.File(hps[0], 'a') as f:
    print(f['sim0'].keys())
    ga0 = f['sim0/grain_areas'][:]

with h5py.File(hps[1]) as f:
    print(f['sim0'].keys())
    ga1 = f['sim0/grain_areas'][:]

n0 = np.sum(ga0>0, axis=1)
n1 = np.sum(ga1>0, axis=1)


#plot all
hps = 10*['./data/mf_sz(512x512)_ng(1024)_nsteps(200)_cov(25-25-0)_numnei(64)_cut(0).h5',]+10*['./data/mf_sz(512x512)_ng(1024)_nsteps(200)_cov(25-25-0)_numnei(64)_cut(25).h5',]
gps = ['sim%d'%i for i in range(10)]+['sim%d'%i for i in range(10)]
cr = [10,10]
fs.make_time_plots(hps, gps, scale_ngrains_ratio=0.20, cr=cr, legend=False)


















#Run example initial conditions to show MF works
     
    
ic, ea, _ = fs.voronoi2image(size=[512, 512], ngrain=512)  

ims_id = run_mf(ic, ea, 100, 360, cov_mat=torch.Tensor([[25,0],[0,25]]), num_samples=64)
    
imageio.mimsave('../GrainGrowth/mf_360.mp4', ims_id[:,0].astype(np.uint8))
plt.imshow(ims_id[-1,0])
plt.savefig('../GrainGrowth/mf_360.jpg')



ic, ea = fs.generate_3grainIC(size=[256,256], h=150)
miso_array = torch.Tensor([45,43,5])/180*np.pi

ims_id = run_mf(ic, ea, 100, 360, cov_mat=torch.Tensor([[50,0],[0,50]]), num_samples=64, miso_array=miso_array, bcs=['p','c'])
    
imageio.mimsave('../GrainGrowth/mf_3g_360.mp4', ims_id[:,0].astype(np.uint8))
plt.imshow(ims_id[-1,0])
plt.savefig('../GrainGrowth/mf_3g_360.jpg')




cc, sz = fs.generate_hex_grain_centers(dim=512, dim_ngrain=8)
ic, ea, _ = fs.voronoi2image(sz, ngrain=64, center_coords0=cc) 
   
ims_id = run_mf(ic, ea, 1000, 25, cov_mat=torch.Tensor([[50,0],[0,50]]), num_samples=64)

imageio.mimsave('../GrainGrowth/mf_hex_25.mp4', (ims_id[:,0]*8).astype(np.uint8))
plt.imshow(ims_id[-1,0])
plt.savefig('../GrainGrowth/mf_hex_25.jpg')







