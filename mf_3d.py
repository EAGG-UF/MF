#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 13:58:08 2023

@author: joseph.melville
"""


import numpy as np
import torch
import matplotlib.pyplot as plt
import functions_primme as fs
import h5py
from tqdm import tqdm


# Functions




### MAIN


# fp = '../primme_share/PRIMME/data/primme_sz(128x128x128)_ng(8192)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
# with h5py.File(fp, 'r') as f:
#     ic = f['sim0/ims_id'][0,0].astype('int')
#     ea = f['sim0/euler_angles'][:].astype('int')

ic, ea, _ = fs.voronoi2image(size=[128,128,128], ngrain=4096)

ims, fp_save = run_mf(ic, ea, nsteps=1000, cut=0, cov=3, num_samples=64)
fs.compute_grain_stats(fp_save)
fs.make_time_plots(fp_save)




hps = ['./data/mf_sz(128x128x128)_ng(4096)_nsteps(1000)_cov(3)_numnei(64)_cut(0).h5',
       './data/mf_sz(128x128x128)_ng(4096)_nsteps(1000)_cov(25)_numnei(64)_cut(25).h5']








fs.plotly_micro(im)



with h5py.File(hps[0], 'r') as f:
    print(f.keys())
    print(f['sim0'].keys())
    im = f['sim0/ims_id'][0,0,].astype('int')
    ic = f['sim0/ims_id'][0,0,].astype('int')
    ea = f['sim0/euler_angles'][:]


ic, ea, _ = fs.voronoi2image(size=[128,]*3, ngrain=4096)    
ims = run_mf(ic, ea, nsteps=1000, cut=0, cov=25, num_samples=64)
ims = run_mf(ic, ea, nsteps=1000, cut=25, cov=25, num_samples=64)

hps = ['./data/mf_sz(128x128x128)_ng(4096)_nsteps(1000)_cov(25)_numnei(64)_cut(0).h5',
       './data/mf_sz(128x128x128)_ng(4096)_nsteps(1000)_cov(25)_numnei(64)_cut(25).h5']
fs.compute_grain_stats(hps)
# fs.make_videos(hps) 
fs.make_time_plots(hps)


fp = './data/mf_sz(128x128x128)_ng(8192)_nsteps(1000)_cov(3)_numnei(64)_cut(0).h5'
fs.make_time_plots(fp)



ic, ea, _ = fs.voronoi2image(size=[1024,]*2, ngrain=4096)  

ims = run_mf(ic, ea, nsteps=10, cut=0, cov=25, num_samples=64)


plt.imshow(ic)

hps = ['./data/mf_sz(1024x1024)_ng(4096)_nsteps(1000)_cov(25)_numnei(64)_cut(0).h5']
fs.compute_grain_stats(hps)
fs.make_videos(hps) 
fs.make_time_plots(hps)

#12
#37
#26
#65




fs.create_3D_paraview_vtr(ic)


















