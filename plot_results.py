#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 16:03:12 2023

@author: joseph.melville
"""



import functions as fs
import h5py 
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas
# from scipy.stats import multivariate_normal
# import scipy as sp
import shutil
# import torch
# import scipy.io



w=2; h=2
sw=3; sh=2
if_leg = True

def export_legend(legend, filename="legend.png", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)



### 2D isotropic <R>^2 vs time, number of grains vs time (MF, SPPARKS, PF) #!!!

mat = scipy.io.loadmat('./data/CSV.mat')
grain_areas = mat['areas'].T
ng = (grain_areas!=0).sum(1)
si = np.argmin(np.abs(ng-600))
grain_radii = np.sqrt(grain_areas/np.pi)
grain_radii_avg = grain_radii.sum(1)/ng #find mean without zeros
r2_pf = (grain_radii_avg**2)[:si] #square after the mean

t_pf = mat['data'][:si,0]
p_pf = np.polyfit(t_pf, r2_pf, deg=1)[0]
ng_pf = ng[:si]

with h5py.File('./data/mf_sz(2400x2400)_ng(20000)_nsteps(1000)_cov(25)_numnei(64)_cut(0).h5', 'r') as f:
    grain_areas = f['sim2/grain_areas'][:]
ng = (grain_areas!=0).sum(1)
si = np.argmin(np.abs(ng-600))
grain_radii = np.sqrt(grain_areas/np.pi)
grain_radii_avg = grain_radii.sum(1)/ng #find mean without zeros
r2_mf = (grain_radii_avg**2)[:si] #square after the mean
p = np.polyfit(np.arange(si), r2_mf, deg=1)[0]
scale = p/p_pf
t_mf = np.arange(si)*scale
ng_mf = ng[:si]

with h5py.File('./data/spparks_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1.0)_kt(0.66)_cut(0).h5', 'r') as f:
    grain_areas = f['sim0/grain_areas'][:]
ng = (grain_areas!=0).sum(1)
si = np.argmin(np.abs(ng-600))
grain_radii = np.sqrt(grain_areas/np.pi)
grain_radii_avg = grain_radii.sum(1)/ng #find mean without zeros
r2_mcp = (grain_radii_avg**2)[:si] #square after the mean
p = np.polyfit(np.arange(si), r2_mcp, deg=1)[0]
scale = p/p_pf
t_mcp = np.arange(si)*scale
ng_mcp = ng[:si]

plt.figure(figsize=[sw,sh], dpi=600)
plt.rcParams['font.size'] = 8
plt.plot(t_mf, r2_mf*1e-12, '-')
plt.plot(t_mcp, r2_mcp*1e-12, '--')
plt.plot(t_pf ,r2_pf*1e-12, '-.')
plt.xlabel('Time (s)')
plt.ylabel('$<R>^2$ ($m^2$)')
if if_leg: plt.legend(['MF','MCP','PF'])
plt.savefig('/blue/joel.harley/joseph.melville/tmp/2d_r2_vs_time.png', bbox_inches='tight', dpi=600)
plt.show()



### 2D isotropic average number of sides through time (MF, SPPARKS, PF) #!!!

with h5py.File('./data/mf_sz(2400x2400)_ng(20000)_nsteps(1000)_cov(25)_numnei(64)_cut(0).h5', 'r') as f:
    gsa_mf = f['sim2/grain_sides_avg'][:]

with h5py.File('./data/spparks_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1.0)_kt(0.66)_cut(0).h5', 'r') as f:
    gsa_mcp = f['sim0/grain_sides_avg'][:]

gsa_pf = pandas.read_csv('./data/pf/Kristien Everett - grain_neighbors_stats Case4_hd.csv').values[:,0]

plt.figure(figsize=[sw,sh], dpi=600)
plt.rcParams['font.size'] = 8
plt.plot(t_mf, gsa_mf[:len(t_mf)], '-', linewidth=1, alpha=1)
plt.plot(t_mcp, gsa_mcp[:len(t_mcp)], '--', linewidth=1, alpha=1)
plt.plot(t_pf, gsa_pf[:len(t_pf)], '-.', linewidth=1, alpha=1)
plt.xlabel('Time (s)')
plt.ylabel('Avg Number \nof Sides')
if if_leg: plt.legend(['MF','MCP','PF'])
plt.savefig('/blue/joel.harley/joseph.melville/tmp/2d_num_sides_vs_time.png', bbox_inches='tight', dpi=600)
plt.show()



### 2D isotropic normalized radius distribution (MF, SPPARKS, PF, Yadav, Zollinger) #!!!

for num_grains in [4000, 3000, 2000]: #4000, 3000, 2000

    with h5py.File('./data/mf_sz(2400x2400)_ng(20000)_nsteps(1000)_cov(25)_numnei(64)_cut(0).h5', 'r') as f:
        grain_areas = f['sim2/grain_areas'][:]
    n = (grain_areas!=0).sum(1)
    j = np.argmin(np.abs(n-num_grains))
    a = grain_areas[j]
    r = np.sqrt(a[a!=0]/np.pi)
    rn = r/np.mean(r)
    h_mf, x_edges = np.histogram(rn, bins='auto', density=True)
    x_mf = x_edges[:-1]+np.diff(x_edges)/2
    n_mf = len(rn)
    
    mat = scipy.io.loadmat('./data/previous_figures/Case4GSizeMCPG%d.mat'%num_grains)
    rn = mat['rnorm'][:,0]
    h_mcp, x_edges = np.histogram(rn, bins='auto', density=True)
    x_mcp = x_edges[:-1]+np.diff(x_edges)/2
    n_mcp = len(rn)
    
    mat = scipy.io.loadmat('./data/previous_figures/Case4GSizePFG%d.mat'%num_grains)
    rn = mat['rnorm'][0]
    h_pf, x_edges = np.histogram(rn, bins='auto', density=True)
    x_pf = x_edges[:-1]+np.diff(x_edges)/2
    n_pf = len(rn)
    
    mat = scipy.io.loadmat('./data/previous_figures/RadiusDistPureTP2DNew.mat')
    x_yad = mat['y1'][0]
    h_yad = mat['RtotalFHD5'][0]
    
    x_zol, h_zol = np.loadtxt('./data/previous_figures/Results.csv', delimiter=',',skiprows=1).T
    
    plt.figure(figsize=[sw,sh], dpi=600)
    plt.rcParams['font.size'] = 8
    plt.plot(x_mf, h_mf, '-')
    plt.plot(x_mcp, h_mcp, '--')
    plt.plot(x_pf, h_pf, '-.')
    plt.plot(x_yad, h_yad, '*', ms = 3)
    plt.plot(x_zol, h_zol, 'd', ms = 3)
    plt.xlabel('$R/<R>$ - Normalized Radius')
    plt.ylabel('Frequency')
    plt.xlim([0,3])
    plt.ylim([0,1.2])
    if if_leg: plt.legend(['MF, $N_G$ - %d'%n_mf, 'MCP, $N_G$ - %d'%n_mcp, 'PF, $N_G$ - %d'%n_pf, 'Yadav 2018', 'Zollner 2016'], fontsize=7)
    plt.savefig('/blue/joel.harley/joseph.melville/tmp/2d_r_dist%d.png'%num_grains, bbox_inches='tight', dpi=600)
    plt.show()



### 2D isotropic number of sides distribution (MF, SPPARKS, PF, Yadav, Mason) #!!!

for num_grains in [4000, 3000, 2000]: #4000, 3000, 2000
    
    with h5py.File('./data/mf_sz(2400x2400)_ng(20000)_nsteps(1000)_cov(25)_numnei(64)_cut(0).h5', 'r') as f:
        grain_sides = f['sim2/grain_sides'][:]
    n = (grain_sides!=0).sum(1)
    i = np.argmin(np.abs(n-num_grains))
    s = grain_sides[i][grain_sides[i]!=0]
    bins = np.arange(1,20)-0.5
    h_mf, x_edges = np.histogram(s, bins=bins, density=True)
    x_mf = x_edges[:-1]+np.diff(x_edges)/2
    n_mf = len(s)
    
    mat = scipy.io.loadmat('./data/previous_figures_sides/Case4SidesMCPG%d.mat'%num_grains)
    s = mat['the_sides'][0]
    h_mcp, x_edges = np.histogram(s, bins=bins, density=True)
    x_mcp = x_edges[:-1]+np.diff(x_edges)/2
    n_mcp = len(s)
    
    mat = scipy.io.loadmat('./data/previous_figures_sides/Case4SidesPFG%d.mat'%num_grains)
    s = mat['the_sides'][0]
    h_pf, x_edges = np.histogram(s, bins=bins, density=True)
    x_pf = x_edges[:-1]+np.diff(x_edges)/2
    n_pf = len(s)
    
    mat = scipy.io.loadmat('./data/previous_figures_sides/FaceDistPureTP2DNew.mat')
    x_yad = mat['y1'][0]
    h_yad = mat['FtotalFHD5'][0]
    
    x_mas, h_mas = np.loadtxt('./data/previous_figures_sides/ResultsMasonLazar2DGTD.txt').T
    
    plt.figure(figsize=[sw,sh], dpi=600)
    plt.rcParams['font.size'] = 8
    plt.plot(x_mf, h_mf, '-')
    plt.plot(x_mcp, h_mcp, '--')
    plt.plot(x_pf, h_pf, '-.')
    plt.plot(x_yad, h_yad, '*', ms=4)
    plt.plot(x_mas, h_mas, '^C6', ms=3)
    plt.xlabel('Number of Sides')
    plt.ylabel('Frequency')
    plt.xlim([0,15])
    plt.ylim([0,0.4])
    if if_leg: plt.legend(['MF, $N_G$ - %d'%n_mf, 'MCP, $N_G$ - %d'%n_mcp, 'PF, $N_G$ - %d'%n_pf, 'Yadav 2018', 'Masson 2015'], fontsize=7)
    plt.savefig('/blue/joel.harley/joseph.melville/tmp/2d_num_sides_dist%d.png'%num_grains, bbox_inches='tight', dpi=600)
    plt.show()

plt.figure(figsize=[2,1], dpi=600)
plt.rcParams['font.size'] = 8
plt.plot(x_mf, h_mf, '-')
plt.plot(x_mcp, h_mcp, '--')
plt.plot(x_pf, h_pf, '-.')
plt.plot(x_yad, h_yad, '*', ms = 5)
plt.plot(x_zol, h_zol, 'd', ms = 5)
plt.plot(x_mas, h_mas, '^C6', ms = 5)
legend = plt.legend(['MF', 'MCP', 'PF', 'Yadav 2018', 'Zollner 2016', 'Masson 2015'], bbox_to_anchor=[1,1,1,1])
export_legend(legend, filename='../tmp/2d_stats_legend.png')
plt.show()
    
    
    
### 2D isotropic microstructure comparisons (MF, MCP, PF) #!!!

num_grains = [512, 300, 150, 50]

with h5py.File('./data/mf_sz(512x512)_ng(512)_nsteps(1000)_cov(25)_numnei(64)_cut(0).h5', 'r') as f:
    ims_mf = f['sim0/ims_id'][:]
ng_mf = np.array([len(np.unique(im))for im in ims_mf])

with h5py.File('./data/32c512grs512stsPkT066_img.hdf5', 'r') as f:
    ims_mcp = f['images'][:]
ng_mcp = np.array([len(np.unique(im))for im in ims_mcp])

plt.figure(figsize=[6.5,3], dpi=600)
plt.rcParams['font.size'] = 8
for i in range(len(num_grains)):
    
    j = np.argmin(np.abs(ng_mf-num_grains[i]))
    im_mf = ims_mf[j,0]
    
    j = np.argmin(np.abs(ng_mcp-num_grains[i]))
    im_mcp = ims_mcp[j,]
    
    # import imageio.v3 as iio
    # im = iio.imread('./data/pf/Case3PeriodicUniqueGrains.0000.png')
    # im_pf = im[2:-1,6:-3,0]
    
    # with h5py.File('./data/pf/Case3ImagesHighRes.hdf5', 'r') as f:
    #     ims = f['images'][:]
    # im_pf = ims[1]
    
    plt.subplot(2,4,1+i)
    plt.imshow(im_mf, interpolation='none')
    plt.title('$N_G$=%d'%len(np.unique(im_mf)), fontsize=8)
    plt.tick_params(bottom=False, left=False,labelleft=False, labelbottom=False)
    if i==0: plt.ylabel('MF', fontsize=8)
    plt.subplot(2,4,5+i)
    plt.imshow(im_mcp, interpolation='none')
    # plt.title('$N_G$=%d'%len(np.unique(im_mcp)))
    plt.tick_params(bottom=False, left=False,labelleft=False, labelbottom=False)
    if i==0: plt.ylabel('MCP', fontsize=8)
    # plt.imshow(np.fliplr(im_pf), interpolation='none')
    plt.savefig('/blue/joel.harley/joseph.melville/tmp/2d_comp.png', bbox_inches='tight', dpi=600)
plt.show()



### 2D Dihedral STD and Avg Miso through time (iso vs ani) #!!!

with h5py.File('./data/mf_sz(1024x1024)_ng(4096)_nsteps(1000)_cov(25)_numnei(64)_cut(0).h5', 'r') as f:
    print(f['sim0'].keys())
    msa0 = f['sim0/ims_miso_spparks_avg'][:]
    das0 = f['sim0/dihedral_std'][:]
    ng0 = (f['sim0/grain_areas'][:]!=0).sum(1)
    
with h5py.File('./data/mf_sz(1024x1024)_ng(4096)_nsteps(1000)_cov(25)_numnei(64)_cut(25).h5', 'r') as f:
    print(f['sim0'].keys())
    msa25 = f['sim0/ims_miso_spparks_avg'][:]
    das25 = f['sim0/dihedral_std'][:]
    ng25 = (f['sim0/grain_areas'][:]!=0).sum(1)
    
# np.argmin(np.abs(ng25-4096*0.05))
x_cut = 420

plt.figure(figsize=[sw,sh], dpi=600)
plt.rcParams['font.size'] = 8
plt.plot(msa0[:x_cut])
plt.plot(msa25[:x_cut],'--',linewidth=1)
plt.title('')
plt.xlabel('Number of Frames')
plt.ylabel('Avg Boundary \nMisorientation')
if if_leg: plt.legend(['Isotropic','Anisotropic'])
plt.savefig('/blue/joel.harley/joseph.melville/tmp/2d_avg_miso.png', bbox_inches='tight', dpi=600)
plt.show()

plt.figure(figsize=[sw,sh], dpi=600)
plt.rcParams['font.size'] = 8
plt.plot(das0[:x_cut])
plt.plot(das25[:x_cut],'--',linewidth=1)
plt.title('')
plt.xlabel('Number of Frames')
plt.ylabel('Dihedral \nAngle STD')
if if_leg: plt.legend(['Isotropic','Anisotropic'])
plt.savefig('/blue/joel.harley/joseph.melville/tmp/2d_dihedral.png', bbox_inches='tight', dpi=600)
plt.show()

plt.figure(figsize=[3,2], dpi=600)
plt.rcParams['font.size'] = 8
plt.plot(das0[:x_cut])
plt.plot(das25[:x_cut],'--')
legend = plt.legend(['Isotropic','Anisotropic'], bbox_to_anchor=[1,1,1,1])
if if_leg: legend = plt.legend(['MF', 'MCP', 'PF', 'Yadav 2018', 'Masson 2015'], bbox_to_anchor=[1,1,1,1])
export_legend(legend, filename='../tmp/2d_ani_stats_legend.png')
plt.show()



### 3D isotropic <R>^2vs time, number of grains vs time (MF, SPPARKS) #!!!

with h5py.File('./data/mf_sz(128x128x128)_ng(8192)_nsteps(1000)_cov(4)_numnei(64)_cut(0).h5', 'r') as f:
    grain_areas = f['sim0/grain_areas'][:]
ng = (grain_areas!=0).sum(1)
si = np.argmin(np.abs(ng-410))
grain_radii = np.cbrt(grain_areas*3/4/np.pi)
grain_radii_avg = grain_radii.sum(1)/ng #find mean without zeros
r2_mf = (grain_radii_avg**2)[:si] #square after the mean
t_mf = np.arange(si)
p_mf = np.polyfit(t_mf, r2_mf, deg=1)[0]
ng_mf = ng[:si]

grain_areas = np.load('./data/spparks_grain_areas_128p3_8192.npy')
ng = (grain_areas!=0).sum(1)
si = np.argmin(np.abs(ng-410))
grain_radii = np.cbrt(grain_areas*3/4/np.pi)
grain_radii_avg = grain_radii.sum(1)/ng #find mean without zeros
r2_mcp = (grain_radii_avg**2) #square after the mean
p = np.polyfit(np.arange(si), r2_mcp[:si], deg=1)[0]
scale = p/p_mf
t_mcp = np.arange(len(r2_mcp))*scale
ng_mcp = ng

plt.figure(figsize=[sw,sh], dpi=600)
plt.rcParams['font.size'] = 8
plt.plot(t_mf, r2_mf)
plt.plot(t_mcp, r2_mcp,'--')
plt.xlabel('Time (Unitless)')
plt.ylabel('$<R>^2$ (Pixels)')
plt.xlim([0,t_mf[-1]])
plt.ylim([0,100])
if if_leg: plt.legend(['MF','MCP'])
plt.savefig('/blue/joel.harley/joseph.melville/tmp/3d_r2_vs_time.png', bbox_inches='tight', dpi=600)
plt.show()



### 3D isotropic average number of sides through time (MF, SPPARKS) #!!!

with h5py.File('./data/mf_sz(128x128x128)_ng(8192)_nsteps(1000)_cov(4)_numnei(64)_cut(0).h5', 'r') as f:
    gsa_mf = f['sim0/grain_sides_avg'][:]
    
gsa_mcp = np.load('./data/spparks_grain_sides_avg_128p3_8192.npy')

plt.figure(figsize=[sw,sh], dpi=600)
plt.rcParams['font.size'] = 8
plt.plot(t_mf, gsa_mf[:len(t_mf)])
plt.plot(t_mcp, gsa_mcp,'--')
plt.xlim([0, t_mf[-1]])
plt.xlabel('Time (Unitless)')
plt.ylabel('Avg Number \nof Sides')
if if_leg: plt.legend(['MF','MCP'])
plt.savefig('/blue/joel.harley/joseph.melville/tmp/3d_num_sides_vs_time.png', bbox_inches='tight', dpi=600)
plt.show()



### 3D isotropic normalized radius distribution (MF, SPPARKS) #!!!

for num_grains in [2000, 1500, 1000]:
    
    with h5py.File('./data/mf_sz(128x128x128)_ng(8192)_nsteps(1000)_cov(4)_numnei(64)_cut(0).h5', 'r') as f:
        grain_areas = f['sim0/grain_areas'][:]
    n = (grain_areas!=0).sum(1)
    i = np.argmin(np.abs(n-num_grains))
    a = grain_areas[i]
    r = np.cbrt(a[a!=0]*3/4/np.pi)
    rn = r/np.mean(r)
    h_mf, x_edges = np.histogram(rn, bins='auto', density=True)
    x_mf = x_edges[:-1]+np.diff(x_edges)/2
    n_mf = len(rn)
    
    grain_areas = np.load('./data/spparks_grain_areas_128p3_8192.npy')
    n = (grain_areas!=0).sum(1)
    i = np.argmin(np.abs(n-num_grains))
    a = grain_areas[i]
    r = np.cbrt(a[a!=0]*3/4/np.pi)
    rn = r/np.mean(r)
    h_mcp, x_edges = np.histogram(rn, bins='auto', density=True)
    x_mcp = x_edges[:-1]+np.diff(x_edges)/2
    n_mcp = len(rn)
    
    plt.figure(figsize=[sw,sh], dpi=600)
    plt.rcParams['font.size'] = 8
    plt.plot(x_mf, h_mf, '-')
    plt.plot(x_mcp, h_mcp, '--')
    plt.xlabel('$R/<R>$ - Normalized Radius')
    plt.ylabel('Frequency')
    if if_leg: plt.legend(['MF, $N_G$ - %d'%n_mf, 'MCP, $N_G$ - %d'%n_mcp], fontsize=7)
    plt.savefig('/blue/joel.harley/joseph.melville/tmp/3d_r_dist%d.png'%num_grains, bbox_inches='tight', dpi=600)
    plt.show()



### 3D isotropic number of sides distribution (MF, SPPARKS) #!!!

for num_grains in [2000, 1500, 1000]:

    with h5py.File('./data/mf_sz(128x128x128)_ng(8192)_nsteps(1000)_cov(4)_numnei(64)_cut(0).h5', 'r') as f:
        grain_sides = f['sim0/grain_sides'][:]
    n = (grain_sides!=0).sum(1)
    i = np.argmin(np.abs(n-num_grains))
    s = grain_sides[i][grain_sides[i]!=0]
    bins = np.arange(1,35)-0.5
    h_mf, x_edges = np.histogram(s, bins=bins, density=True)
    x_mf = x_edges[:-1]+np.diff(x_edges)/2
    n_mf = len(s)
    
    grain_sides = np.load('./data/spparks_grain_sides_128p3_8192.npy')
    n = (grain_sides!=0).sum(1)
    i = np.argmin(np.abs(n-num_grains))
    s = grain_sides[i][grain_sides[i]!=0]
    bins = np.arange(1,35)-0.5
    h_mcp, x_edges = np.histogram(s, bins=bins, density=True)
    x_mcp = x_edges[:-1]+np.diff(x_edges)/2
    n_mcp = len(s)
    
    plt.figure(figsize=[sw,sh], dpi=600)
    plt.rcParams['font.size'] = 8
    plt.plot(x_mf, h_mf, '-')
    plt.plot(x_mcp, h_mcp, '--')
    plt.xlabel('Number of Sides')
    plt.ylabel('Frequency')
    if if_leg: plt.legend(['MF, $N_G$ - %d'%n_mf, 'MCP, $N_G$ - %d'%n_mcp], fontsize=7)
    plt.savefig('/blue/joel.harley/joseph.melville/tmp/3d_num_sides_dist%d.png'%num_grains, bbox_inches='tight', dpi=600)
    plt.show()



### 3D isotropic microstructure comparisons (MF, MCP) #!!!

num_grains = [8192, 5000, 2500, 1000]
si_mf = [0,19,44,94]
si_mcp = [0,22, 48, 96]

with h5py.File('./data/mf_sz(128x128x128)_ng(8192)_nsteps(1000)_cov(4)_numnei(64)_cut(0).h5', 'r') as f:
    grain_areas = f['sim0/grain_areas'][:]
    ng_mf = (grain_areas!=0).sum(1)[si_mf]
    ims_mf = f['sim0/ims_id'][si_mf,0,64]
    
grain_sides = np.load('./data/spparks_grain_sides_128p3_8192.npy')
ng_mcp = (grain_sides!=0).sum(1)[si_mcp]
ims_mcp = np.load('./data/spparks_ims_id_128p3_8192.npy')[:,64]

plt.figure(figsize=[6.5,3], dpi=600)
plt.rcParams['font.size'] = 8
for i in range(len(num_grains)):
    plt.subplot(2, 4, 1+i)
    plt.imshow(ims_mf[i], interpolation='none')
    plt.title('$N_G$=%d'%ng_mf[i], fontsize=8)
    plt.tick_params(bottom=False, left=False,labelleft=False, labelbottom=False)
    if i==0: plt.ylabel('MF', fontsize=8)
    plt.subplot(2, 4, 5+i)
    plt.imshow(ims_mcp[i], interpolation='none')
    plt.tick_params(bottom=False, left=False,labelleft=False, labelbottom=False)
    if i==0: plt.ylabel('MCP', fontsize=8)
plt.savefig('/blue/joel.harley/joseph.melville/tmp/3d_comp.png', bbox_inches='tight', dpi=600)
plt.show()



### 3D Dihedral STD and Avg Miso through time (iso vs ani) #!!!

with h5py.File('./data/mf_sz(128x128x128)_ng(8192)_nsteps(1000)_cov(4)_numnei(64)_cut(0).h5', 'r') as f:
    # Hits 410 grains at frame 186
    # print(f['sim0'].keys())
    msa0 = f['sim0/ims_miso_spparks_avg'][:]
    das0 = f['sim0/dihedral_std'][:]
    ng0 = (f['sim0/grain_areas'][:]!=0).sum(1)

with h5py.File('./data/mf_sz(128x128x128)_ng(8192)_nsteps(1000)_cov(4)_numnei(64)_cut(25).h5', 'r') as f:
    # Hits 410 grains at frame 149
    # print(f['sim0'].keys())
    msa25 = f['sim0/ims_miso_spparks_avg'][:]
    das25 = f['sim0/dihedral_std'][:]
    ng25 = (f['sim0/grain_areas'][:]!=0).sum(1)
x_cut = 150

plt.figure(figsize=[sw,sh], dpi=600)
plt.rcParams['font.size'] = 8
plt.plot(msa0[:x_cut])
plt.plot(msa25[:x_cut],'--')
plt.title('')
plt.xlabel('Number of Frames')
plt.ylabel('Avg Boundary \nMisorientation')
if if_leg: plt.legend(['Isotropic','Anisotropic'])
plt.savefig('/blue/joel.harley/joseph.melville/tmp/3d_avg_miso.png', bbox_inches='tight', dpi=600)
plt.show()

plt.figure(figsize=[sw,sh], dpi=600)
plt.rcParams['font.size'] = 8
plt.plot(das0[:x_cut])
plt.plot(das25[:x_cut],'--')
plt.title('')
plt.xlabel('Number of Frames')
plt.ylabel('Dihedral \nAngle STD')
if if_leg: plt.legend(['Isotropic','Anisotropic'])
plt.savefig('/blue/joel.harley/joseph.melville/tmp/3d_dihedral.png', bbox_inches='tight', dpi=600)
plt.show()



### Variance vs slope of <R>^2 #!!!

nss = [4,8,16,32,64]
stds = np.arange(2,11)
covs = stds**2

# ng = 2**14
# ic, ea, _ = fs.voronoi2image([2048, 2048], ng)
# ma = fs.find_misorientation(ea, mem_max=10)

# for ns in nss: #(10 hrs runtime)
#     log = []
#     for cov in covs:
#         print(cov)
#         log_p = []
#         for i in range(30):
#             ims = fs.run_mf(ic, ea, nsteps=300, cut=0, cov=int(cov), num_samples=ns, miso_array=ma, if_save=False)
#             a = fs.iterate_function(ims, fs.find_grain_areas, [ng-1])
#             r = np.sqrt(a/np.pi)
#             n = (r!=0).sum(1)
#             ra = r.sum(1)/n #find mean without zeros
#             ra2 = (ra**2)[100:] #square after the mean
#             if i==0: plt.plot(ra2); plt.show()
#             x = np.arange(len(ra2))
#             p = np.polyfit(x, ra2, 1)
#             log_p.append(p[0])
#         print(np.mean(log_p))
#         log.append(log_p)
        
#         s = np.block(log).mean(1)
#         e = np.block(log).std(1)
#         plt.errorbar(covs[:len(s)], s, e, marker='.', ms=20, capsize=3) 
    
    # np.save('./data/Slope and error of <R>2 vs var - 2048x2048 16384grains %dns 30rep'%ns, np.stack([s,e]))

log_s = []
log_e = []
for ns in nss:
    s, e = np.load('./data/Slope and error of <R>2 vs var - 2048x2048 16384grains %dns 30rep.npy'%ns)
    log_s.append(s)
    log_e.append(e)
    # log.append((s[-1]-s[0])/(covs[-1]-covs[0]))
    # plt.errorbar(covs[:len(s)], s, 2*e, marker='.', ms=2, capsize=3, linestyle='-') 
s_new = np.stack(log_s).mean(0) #find the mean for all nss
e_all = np.stack(log_e)
e_new = np.sqrt((e_all**2*29).sum(0)/(30*e_all.shape[1]-1)) #find the std for all nss

plt.figure(figsize=[w,h], dpi=600)
plt.rcParams['font.size'] = 8
# plt.errorbar(covs[:len(s)], s, 2*e, marker='.', ms=2, capsize=3, linestyle='-') 
plt.errorbar(covs[:len(s)], s_new, 2*e_new, marker='.', ms=2, capsize=3, linestyle='-')
plt.ylabel('Slope of $<R>^2$')
plt.xlabel('Variance')
plt.savefig('/blue/joel.harley/joseph.melville/tmp/var_vs_growth_speed.png', bbox_inches='tight', dpi=600)
plt.show()



### Runtime vs nss #!!!
# ic, ea, _ = fs.voronoi2image([1024,1024], 4096)
# ma = fs.find_misorientation(ea, mem_max=10)
    
nss = np.array([3,4,8,16,32,64,128,256,512])
# itr = [100,100,100,100,100,10,10,1,1]
# log = []
# for i, ns in enumerate(nss): 
#     ims, runtime = fs.run_mf(ic, ea, nsteps=itr[i], cut=0, cov=25, num_samples=ns, miso_array=ma, if_save=False, if_time=True)
#     log.append(runtime/itr[i])
# runtime = np.stack(log)
# np.save('./data/runtime_vs_nss', runtime)
runtime = np.load('./data/runtime_vs_nss.npy')

# plt.figure(figsize=[2,2], dpi=600)
# plt.rcParams['font.size'] = 8
# plt.plot(nss, 1/runtime, '*-')
# plt.ylabel('FPS')
# plt.xlabel('$N_S$')
# plt.annotate('$N_S$: %d\nFPS: %.1f'%(nss[0], 1/runtime[0]), (nss[0]+50, 1/runtime[0]-80))
# plt.annotate('$N_S$: %d\nFPS: %.1f'%(nss[-1], 1/runtime[-1]), (nss[-1]-150, 1/runtime[-1]+40))
# plt.savefig('/blue/joel.harley/joseph.melville/tmp/fps_vs_nss.png', bbox_inches='tight', dpi=600)
# plt.show()

p = np.sum(np.linalg.pinv(np.array(nss**2)[:,None])*runtime)
runtime_fit = p*nss**2
r2 = 1-((runtime-runtime_fit)**2).sum()/((runtime-runtime.mean())**2).sum()

plt.figure(figsize=[w,h], dpi=600)
plt.rcParams['font.size'] = 8
plt.plot(nss, runtime, '*-')
# plt.plot(nss, runtime_fit, '*-')
plt.ylabel('Runtime (s)')
plt.xlabel('$N_S$')
# plt.legend(['Fit $y = ax^2$ \n$1-R^2$: %1.1e'%(1-r2)], fontsize=7, loc=2)
plt.legend(['Fit $y = (%1.1e)x^2$ \n1-$R^2$: %1.1e'%(p, 1-r2)], fontsize=7, loc=2)
plt.savefig('/blue/joel.harley/joseph.melville/tmp/runtime_vs_nss.png', bbox_inches='tight', dpi=600)
plt.show()



### Single step error vs nss #!!!
var = [25]#[4,9,16,25,36] #all proportional along x axis
nss = np.array([3,4,8,16,32,64,128,256,512])#.astype(float)

# log = []
# for v in var:
#     log0 = []
#     for ns in nss:
#         ims0 = fs.run_mf(ic, ea, nsteps=1, cut=0, cov=v, num_samples=ns, miso_array=ma, if_save=False)
#         ims1 = fs.run_mf(ic, ea, nsteps=1, cut=0, cov=v, num_samples=ns, miso_array=ma, if_save=False)
#         log0.append(1-(ims0[1]==ims1[1]).sum()/(1024**2))
#     log.append(log0)
# loss = np.block(log).squeeze()
# np.save('./data/l0loss', loss)
loss = np.load('./data/l0loss.npy')

plt.figure(figsize=[w,h], dpi=600)
plt.rcParams['font.size'] = 8
plt.plot(nss, loss, '*-')
plt.ylabel('Mean $L0$ Loss')
plt.xlabel('$N_S$')
plt.ylim([0,.5])
plt.savefig('/blue/joel.harley/joseph.melville/tmp/l0loss_vs_nss.png', bbox_inches='tight', dpi=600)
plt.show()



### Images vs nss #!!!
nss = [3,8,16]
# log = []
# for ns in nss:
#     ims = fs.run_mf(ic, ea, nsteps=100, cut=0, cov=25, num_samples=ns, miso_array=ma, if_save=False)
#     ng = np.array([len(np.unique(im)) for im in ims])
#     i = np.argmin(np.abs(ng-1000))
#     log.append(ims[i,0])
# ims = np.stack(log)
# np.save('./data/ims_vs_nss', ims)
ims = np.load('./data/ims_vs_nss.npy')
    
for i in range(len(nss)):    
    plt.figure(figsize=[w,h], dpi=600)
    plt.rcParams['font.size'] = 8
    plt.imshow(ims[i], interpolation='none')
    plt.title('$N_S$=%d'%nss[i])
    plt.tick_params(bottom=False, left=False,labelleft=False, labelbottom=False)
    plt.savefig('/blue/joel.harley/joseph.melville/tmp/structure_ns%d.png'%nss[i], bbox_inches='tight', dpi=600)
    plt.show()



### Distribution quality - statistical error vs nss #!!!

var=25

# Normalized radius
nss = np.array([3,4,8,16,32,64,128])

log = []
log_h = []
log_n = []
for ns in nss:
    
    log0 = []
    log_h0 = []
    log_n0 = []
    for num_grains in [4000, 3000, 2000]: #4000, 3000, 2000
    
        with h5py.File('./data/mf_sz(2400x2400)_ng(20000)_nsteps(1000)_cov(%d)_numnei(%d)_cut(0).h5'%(var,ns), 'r') as f:
            grain_areas = f['sim0/grain_areas'][:]
            if ns==64 and var==25:  grain_areas = f['sim2/grain_areas'][:]
        n = (grain_areas!=0).sum(1)
        j = np.argmin(np.abs(n-num_grains))
        a = grain_areas[j]
        r = np.sqrt(a[a!=0]/np.pi)
        rn = r/np.mean(r)
        
        mat = scipy.io.loadmat('./data/previous_figures/RadiusDistPureTP2DNew.mat')
        x_yad = mat['y1'][0]
        h_yad = mat['RtotalFHD5'][0]
        
        x_zol, h_zol = np.loadtxt('./data/previous_figures/Results.csv', delimiter=',',skiprows=1).T
        
        x_match = x_yad
        h_match = h_yad
        df = np.diff(x_match)
        e0 = np.array([x_match[0]-df[0]/2])
        e1 = np.array([x_match[-1]+df[-1]/2])
        tmp = x_match[:-1]+df/2
        bins = np.concatenate([e0, tmp, e1])
        h_mf, x_edges = np.histogram(rn, bins=bins, density=True)
        x_mf = x_match
        n_mf = len(rn)
        
        log_h0.append(h_mf)
        log_n0.append(n_mf)
        rmse = np.sqrt(np.sum((h_match-h_mf)**2))
        log0.append(rmse)
    
    log.append(log0)
    log_h.append(log_h0)
    log_n.append(log_n0)
        
rmse = np.block(log)

rmse_r = rmse
log_r = log
log_h_r = log_h
log_n_r = log_n
x_yad_r = x_yad
h_yad_r = h_yad
x_mf_r = x_mf


# Number of sides
log = []
log_h = []
log_n = []
for ns in nss:
    
    log0 = []
    log_h0 = []
    log_n0 = []
    for num_grains in [4000, 3000, 2000]: #4000, 3000, 2000
    
        with h5py.File('./data/mf_sz(2400x2400)_ng(20000)_nsteps(1000)_cov(%d)_numnei(%d)_cut(0).h5'%(var, ns), 'r') as f:
            grain_sides = f['sim0/grain_sides'][:]
            if ns==64 and var==25: grain_sides = f['sim2/grain_sides'][:]
        n = (grain_sides!=0).sum(1)
        i = np.argmin(np.abs(n-num_grains))
        s = grain_sides[i][grain_sides[i]!=0]
        
        mat = scipy.io.loadmat('./data/previous_figures_sides/FaceDistPureTP2DNew.mat')
        x_yad = mat['y1'][0]
        h_yad = mat['FtotalFHD5'][0]
        
        x_mas, h_mas = np.loadtxt('./data/previous_figures_sides/ResultsMasonLazar2DGTD.txt').T
        
        x_match = x_yad
        h_match = h_yad
        df = np.diff(x_match)
        e0 = np.array([x_match[0]-df[0]/2])
        e1 = np.array([x_match[-1]+df[-1]/2])
        tmp = x_match[:-1]+df/2
        bins = np.concatenate([e0, tmp, e1])
        h_mf, x_edges = np.histogram(s, bins=bins, density=True)
        x_mf = x_match
        n_mf = len(s)
        
        log_h0.append(h_mf)
        log_n0.append(n_mf)
        rmse = np.sqrt(np.sum((h_match-h_mf)**2))
        log0.append(rmse)
    
    log.append(log0)
    log_h.append(log_h0)
    log_n.append(log_n0)
    
rmse = np.block(log)

# Plot rmse vs nss
plt.figure(figsize=[w,h], dpi=600)
plt.rcParams['font.size'] = 8
plt.plot(nss, rmse_r.mean(1), '*-')
plt.ylabel('RMSE - Normalized Radius')
plt.xlabel('$N_S$')
if if_leg: plt.legend(['Normalized Radius', 'Number of Sides'], fontsize=7)
plt.savefig('/blue/joel.harley/joseph.melville/tmp/rmse_r_vs_nss.png', bbox_inches='tight', dpi=600)
plt.show()

# Plot rmse vs nss
plt.figure(figsize=[w,h], dpi=600)
plt.rcParams['font.size'] = 8
plt.plot(nss, rmse.mean(1), '*-')
plt.ylabel('RMSE - Number of Sides')
plt.xlabel('$N_S$')
if if_leg: plt.legend(['Normalized Radius', 'Number of Sides'], fontsize=7)
plt.savefig('/blue/joel.harley/joseph.melville/tmp/rmse_sides_vs_nss.png', bbox_inches='tight', dpi=600)
plt.show()


# Plot r distributions
for i in [0,2,3]:
    plt.figure(figsize=[2,1], dpi=600)
    plt.rcParams['font.size'] = 8
    plt.plot(x_mf_r, log_h_r[i][0], '-')
    plt.plot(x_yad_r, h_yad_r, '*', ms=5)
    plt.tick_params(bottom=False, left=False,labelleft=False, labelbottom=False)
    plt.xlabel('Normalized Radius')
    plt.ylabel('Frequency')
    plt.xlim([0,3])
    plt.ylim([0,1.2])
    plt.savefig('/blue/joel.harley/joseph.melville/tmp/r_dist_comp_ns%d.png'%nss[i], bbox_inches='tight', dpi=600)
    plt.show()
   
   
# Plot s distributions
for i in [0,2,3]:
    plt.figure(figsize=[2,1], dpi=600)
    plt.rcParams['font.size'] = 8
    plt.plot(x_mf, log_h[i][0], '-')
    plt.plot(x_yad, h_yad, '*', ms=5)
    plt.tick_params(bottom=False, left=False,labelleft=False, labelbottom=False)
    plt.xlabel('Number of Sides')
    plt.ylabel('Frequency')
    plt.xlim([0,15])
    plt.ylim([0,0.4])
    plt.savefig('/blue/joel.harley/joseph.melville/tmp/s_dist_comp_ns%d.png'%nss[i], bbox_inches='tight', dpi=600)
    plt.show()



### 2D MF run time multiple vs ns (for iso and ani MF, include PF and SPPARKS) #!!!

nss = [4,8,16,32,64]
ng_bounds = [10000, 5000]
ra_bounds = [225, 375]
ra_bounds = [200, 400]

def find_runtime(grain_areas, runtime_tot, ng_bounds):
    ng = (grain_areas!=0).sum(1)
    si = np.argmin(np.abs(ng-ng_bounds[0]))
    sf = np.argmin(np.abs(ng-ng_bounds[1]))
    ns = sf-si
    runtime = runtime_tot*ns/len(ng)
    print('Runtime Total: %.2f, Start Step: %d, Stop Step: %d'%(runtime_tot, si, sf))
    return runtime

def find_runtime2(grain_areas, d, runtime_tot, ra_bounds):
    
    if d==2: r = np.sqrt(grain_areas/np.pi) #for 2 dimensions
    elif d==3: r = np.cbrt(grain_areas*3/4/np.pi) #for 3 dimensions
    n = (r!=0).sum(1)
    ra = r.sum(1)/n #find mean without zeros
    ra2 = ra**2 #square after the mean
    si = np.argmin(np.abs(ra2-ra_bounds[0]))
    sf = np.argmin(np.abs(ra2-ra_bounds[1]))
    nf = sf-si
    runtime = runtime_tot*nf/len(n)
    
    plt.figure()
    plt.plot(ra2)
    plt.plot(si, ra2[si], '.')
    plt.plot(sf, ra2[sf], '.')
    plt.show()
    
    # print('Runtime Total: %.2f, Start Step: %d, Stop Step: %d'%(runtime_tot, si, sf))
    return runtime

mat = scipy.io.loadmat('./data/CSV.mat')
grain_areas = mat['areas'].T[:300]
runtime_tot = 40590
# runtime_pf = find_runtime(grain_areas, runtime_tot, ng_bounds)
runtime_pf = find_runtime2(grain_areas, 2, runtime_tot, ra_bounds)


with h5py.File('./data/spparks_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1.0)_kt(0.66)_cut(0).h5', 'r') as f:
    grain_areas = f['sim0/grain_areas'][:1000,]
runtime_tot = 7776
# runtime_mcp = find_runtime(grain_areas, runtime_tot, ng_bounds)
runtime_mcp = find_runtime2(grain_areas, 2, runtime_tot, ra_bounds)


# with h5py.File('./data/spparks_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1.0)_kt(0.66)_cut(0).h5', 'r') as f:
#     ic = f['sim0/ims_id'][0,0].astype(float)
# ea = np.load('./data/ea_20000.npy')
# ma = np.load('./data/miso_array_20000.npy')

# log = []
# for ns in nss:
#     ims, runtime_tot = fs.run_mf(ic, ea, 100, cut=0, cov=25, num_samples=ns, miso_array=ma, if_time=True, if_save=False)
#     grain_areas = fs.iterate_function(ims, fs.find_grain_areas, [19999])
#     # log.append(find_runtime(grain_areas, runtime_tot, ng_bounds))
#     log.append(find_runtime2(grain_areas, 2, runtime_tot, ra_bounds))
# runtime_mf_iso = np.stack(log)
# # np.save('./data/runtime_mf_iso_var%d.npy'%covs[i], runtime_mf_iso)
# np.save('./data/runtime_mf_iso.npy', runtime_mf_iso)
runtime_mf_iso = np.load('./data/runtime_mf_iso.npy')

# log = []
# for ns in nss:
#     ims, runtime_tot = fs.run_mf(ic, ea, nsteps=100, cut=25, cov=25, num_samples=ns, miso_array=ma, if_time=True, if_save=False)
#     grain_areas = fs.iterate_function(ims, fs.find_grain_areas, [19999])
#     # log.append(find_runtime(grain_areas, runtime_tot, ng_bounds))
#     log.append(find_runtime2(grain_areas, 2, runtime_tot, ra_bounds))
# runtime_mf_ani = np.stack(log)
# np.save('./data/runtime_mf_ani.npy', runtime_mf_ani)
runtime_mf_ani = np.load('./data/runtime_mf_ani.npy')


#Confirm the speed multiple is the same for sample density
#I should probably do error bars for this, but I would need to have the exact same sample density (not today)
# log0 = []
# covs = [4,9,16,25,36] 
# nsteps = [500, 200, 100, 100, 50]
# for i in range(len(covs)):
#     log = []
#     for ns in nss:
#         ims, runtime_tot = fs.run_mf(ic, ea, nsteps=nsteps[i], cut=0, cov=covs[i], num_samples=ns, miso_array=ma, if_time=True, if_save=False)
#         a = fs.iterate_function(ims, fs.find_grain_areas, [19999])
#         r = np.sqrt(a/np.pi)
#         n = (r!=0).sum(1)
#         ra = r.sum(1)/n #find mean without zeros
#         ra2 = ra**2 #square after the mean
        
#         si = np.argmin(np.abs(ra2-ra_bounds[0]))
#         sf = np.argmin(np.abs(ra2-ra_bounds[1]))
#         nf = sf-si
#         runtime = runtime_tot*nf/len(n)
        
#         print('Runtime Total: %.2f, Start Step: %d, Stop Step: %d'%(runtime_tot, si, sf))
        
        
#         log.append(runtime)
        
#     log0.append(log)
    

# aaa = np.block(log0)

# covs = [4,9,16,25,36] 
# for i, var in enumerate(covs):
#     plt.plot(np.array(nss)/var, runtime_mcp/aaa[i,])


# 3D
ng_bounds = [4000, 2000]
ra_bounds = [9.3, 19.8]
ra_bounds = [20, 30]

# ic = np.load('./data/ic_128p3_8192.npy').astype(float)
# ea = np.load('./data/ea_128p3_8192.npy')
# ma = np.load('./data/miso_array_128p3_8192.npy')

# log = []
# for i in range(len(nss)):
#     ims, runtime_tot = fs.run_mf(ic, ea, nsteps=60, cut=0, cov=4, num_samples=nss[i], miso_array=ma, if_time=True, if_save=False)
#     grain_areas = fs.iterate_function(ims, fs.find_grain_areas, [8191])
#     # log.append(find_runtime(grain_areas, runtime_tot, ng_bounds))
#     log.append(find_runtime2(grain_areas, 3, runtime_tot, ra_bounds))
# runtime_mf_iso_3D = np.stack(log)
# np.save('./data/runtime_mf_iso_3D.npy', runtime_mf_iso_3D)
runtime_mf_iso_3D = np.load('./data/runtime_mf_iso_3D.npy')

# log = []
# for ns in nss:
#     ims, runtime_tot = fs.run_mf(ic, ea, nsteps=60, cut=25, cov=4, num_samples=ns, miso_array=ma, if_time=True, if_save=False)
#     grain_areas = fs.iterate_function(ims, fs.find_grain_areas, [8191])
#     # log.append(find_runtime(grain_areas, runtime_tot, ng_bounds))
#     log.append(find_runtime2(grain_areas, 3, runtime_tot, ra_bounds))
# runtime_mf_ani_3D = np.stack(log)
# np.save('./data/runtime_mf_ani_3D.npy', runtime_mf_ani_3D)
runtime_mf_ani_3D = np.load('./data/runtime_mf_ani_3D.npy')


grain_areas = np.load('./data/spparks_grain_areas_128p3_8192.npy')
runtime_tot = 2720
runtime_mcp_3D = find_runtime2(grain_areas, 3, runtime_tot, ra_bounds)


# Plot
iso_2D = runtime_mcp/runtime_mf_iso
ani_2D = runtime_mcp/runtime_mf_ani
iso_3D = runtime_mcp_3D/runtime_mf_iso_3D
ani_3D = runtime_mcp_3D/runtime_mf_ani_3D

plt.figure(figsize=[3,2], dpi=600)
plt.rcParams['font.size'] = 8
plt.plot(nss, np.log10(iso_2D), '*-', ms=5)
plt.plot(nss, np.log10(ani_2D), '^-', ms=5)
plt.plot(nss, np.log10(iso_3D), '*-.', ms=5)
plt.plot(nss, np.log10(ani_3D), '^-.', ms=5)
plt.xlabel('$N_S$')
plt.ylabel('$Log_{10}$ Times Taster')
plt.legend(['2D Iso','2D Ani', '3D Iso', '3D Ani'], fontsize=7)
plt.savefig('/blue/joel.harley/joseph.melville/tmp/comp_speed.png', bbox_inches='tight', dpi=600)
plt.show()



### Zip all figures for easy download
shutil.make_archive('../tmp', 'zip', '../tmp')













### Apendix  #!!!

import torch

with h5py.File('./data/mf_sz(2400x2400)_ng(20000)_nsteps(1000)_cov(25)_numnei(64)_cut(0).h5', 'r') as f:
    ic = torch.from_numpy(f['sim2/ims_id'][0,0].astype(float))
    ea = torch.from_numpy(f['sim2/euler_angles'][:].astype(float))
    ma = torch.from_numpy(f['sim2/miso_array'][:].astype(float))


#nss = [ 2,  3,  4,  8, 16, 32, 64, 128, 256, 512]
#covs = [4, 9, 16, 25, 36]
        

nss = [ 2,  3,  4,  8, 16, 32, 64, 128]
covs = [4]

for ns in nss:
    for var in covs:
        print('Running: ns=%d, var=%d'%(ns, var))
        ims, fp = fs.run_mf(ic, ea, nsteps=1500, cut=0, cov=var, num_samples=ns, miso_array=ma, if_save=True)
        fs.compute_grain_stats(fp)
        
nss = [ 2,  3,  4,  8, 16, 32, 64, 128]
covs = [9]

for ns in nss:
    for var in covs:
        print('Running: ns=%d, var=%d'%(ns, var))
        ims, fp = fs.run_mf(ic, ea, nsteps=1500, cut=0, cov=var, num_samples=ns, miso_array=ma, if_save=True)
        fs.compute_grain_stats(fp)
        
    
nss = [ 2,  3,  4,  8, 16, 32, 64, 128]
covs = [25]

for ns in nss:
    for var in covs:
        print('Running: ns=%d, var=%d'%(ns, var))
        ims, fp = fs.run_mf(ic, ea, nsteps=1000, cut=0, cov=var, num_samples=ns, miso_array=ma, if_save=True)
        fs.compute_grain_stats(fp)




with h5py.File('./data/mf_sz(2400x2400)_ng(20000)_nsteps(1000)_cov(16)_numnei(128)_cut(0).h5', 'r') as f:
    im = torch.from_numpy(f['sim0/ims_id'][-1,0].astype(float))

len(np.unique(im))


# ### Percentage edge pixels at 1000 grains steps vs nss

# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ic, ea, _ = fs.voronoi2image([1024,1024], 4096)
# ma = fs.find_misorientation(ea, mem_max=10)

# nss = [2,3,8,16,32,64]
# log = []
# for ns in nss:
#     ims = fs.run_mf(ic, ea, nsteps=100, cut=0, cov=25, num_samples=ns, miso_array=ma, if_save=False)
    
#     ng = np.array([len(np.unique(im)) for im in ims])
#     i = np.argmin(np.abs(ng-1000))
    
#     im = ims[i]
#     imt = torch.from_numpy(im[None,].astype(float)).to(device)
#     ime = fs.num_diff_neighbors(imt, window_size=3, pad_mode='circular')
#     log.append((ime>0).sum().cpu().numpy()/(1024**2))
    
#     # plt.imshow(ims[i,0]); plt.show()

# plt.figure(figsize=[3,2], dpi=300)
# plt.rcParams['font.size'] = 8
# plt.plot(nss, log)
# plt.ylabel('Ratio of pixels on boundary')
# plt.xlabel('Number of samples')
# plt.savefig('/blue/joel.harley/joseph.melville/tmp/mf_boundary_vs_nss.png', bbox_inches='tight', dpi=300)
# plt.show()


# ### The STD of the grain growth speed goes down with larger number of samples (if you normalize the first sample to 1)

# log = []
# for ns in nss:
#     s, e = np.load('./data/Slope and error of <R>2 vs var - 2048x2048 16384grains %dns 30rep.npy'%ns)
#     log.append(e)


# tmp = np.stack(log)
# anstd = (tmp/tmp.max(0)).mean(1) #avg normalized std
# anstd = (tmp/tmp[0]).mean(1) #avg normalized std

# plt.plot(nss, anstd)
# plt.title('')
# plt.ylabel('STD of slope of $<R>^2$')
# plt.xlabel('Number of samples')

# plt.plot((tmp/tmp.max(0)))


# #Plot
# plt.figure()
# legend = []
# log = []
# loge = []
# logp = []
# for ns in nss:
#     s, e = np.load('./data/Slope and error of <R>2 vs var - 2048x2048 16384grains %dns 30rep.npy'%ns)
#     log.append(s)
#     loge.append(e)
#     x = np.arange(len(s))
#     # p = np.polyfit(covs, s, 1)
    
#     p = np.sum(np.linalg.pinv(np.array(covs)[:,None])*s)
#     logp.append(p)
    
#     # print('Slope: %f \nIntercept: %f'%(p[0], p[1]))
#     # plt.errorbar(covs[:len(s)], s, e, marker='.', ms=20, capsize=3) 
    
#     plt.errorbar(covs[:len(s)], s, 2*e, marker='.', ms=2, capsize=3, linestyle='') 
#     # legend.append('Ns: %d, Slope: %.3f, Var=0: %.3f'%(ns,p[0],p[1]))
#     legend.append('Ns: %d, Slope: %.3f'%(ns,p))
# plt.ylabel('Slope of $<R>^2$')
# plt.xlabel('Variance')
# plt.legend(legend)
# # plt.legend(['Slope: %f'%p[0]])
# plt.show()

# bbb = np.stack(loge)
# ccc = (bbb/bbb.max(0)).mean(1)

# plt.plot(nss, ccc)
# plt.title('Neighborhood variance of 4-100\nAvg of Normalized')
# plt.ylabel('STD of slope of $<R>^2$')
# plt.xlabel('Number of samples')




# ## Can step a low sample rate witha  high sample one to clean it up
# with h5py.File('./data/mf_sz(2400x2400)_ng(20000)_nsteps(1000)_cov(25)_numnei(3)_cut(0).h5', 'r') as f:
#     ic = torch.from_numpy(f['sim0/ims_id'][148,0].astype(float))
#     ea = torch.from_numpy(f['sim0/euler_angles'][:].astype(float))
#     ma = torch.from_numpy(f['sim0/miso_array'][:].astype(float))
    
# ims = fs.run_mf(ic, ea, nsteps=3, cut=0, cov=4, num_samples=128, miso_array=ma, if_save=False)

# len(np.unique(ims[-1]))

# plt.imshow(ic)
# plt.imshow(ims[-1,0])

# d = ims[0:1]
# d = ims[-1][None]
# grain_areas = fs.iterate_function(d, fs.find_grain_areas, args=[19999])
# grain_sides = fs.iterate_function(d, fs.find_grain_num_neighbors, args=[19999])



#Look at the microstructures at different sample rates
# #What makes the lower sampling rates unsuitable - quantify that


# ic, ea, _ = fs.voronoi2image([1024,1024], 4096)
# ma = fs.find_misorientation(ea, mem_max=10)


# ims, runtime_tot = fs.run_mf(ic, ea, nsteps=200, cut=0, cov=25, num_samples=2, miso_array=ma, if_time=True, if_save=False)
# plt.imshow(ims[-1,0])



# for ns in [1,2,3,4,8,16,32,64]:
#     ims, runtime_tot = fs.run_mf(ic, ea, nsteps=200, cut=0, cov=25, num_samples=ns, miso_array=ma, if_time=True, if_save=False)
#     # plt.imshow(ims[-1,0]); plt.show()
    
#     pc = (ims[:-1]!=ims[1:]).sum(1).sum(1).sum(1)/(1024**2)
    
#     device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     log = []
#     for im in ims: 
#         imt = torch.from_numpy(im[None,].astype(float)).to(device)
#         ime = fs.num_diff_neighbors(imt, window_size=3, pad_mode='circular')
#         log.append((ime[0,0]>0).sum()/(1024**2))
#     pe = torch.stack(log).cpu().numpy()[:-1]
    
#     # plt.plot(pc)
#     # plt.plot(pe)
#     plt.plot(pc/pe)

# plt.show()




# ### Performance vs ns

# #plot the distributions for MF simulations of different ns



# nss = [4,8,16,32,64]

# # with h5py.File('./data/spparks_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1.0)_kt(0.66)_cut(0).h5', 'r') as f:
# #     ic = f['sim0/ims_id'][0,0].astype(float)
# # ea = np.load('./data/ea_20000.npy')
# # ma = np.load('./data/miso_array_20000.npy')

# # log = []
# # for i in range(len(nss)):
# #     ims = fs.run_mf(ic, ea, nsteps=250, cut=0, cov=25, num_samples=nss[i], miso_array=ma, if_save=False)
# #     grain_areas = fs.iterate_function(ims, fs.find_grain_areas, [19999])  
# #     log.append(grain_areas)
# # grain_areas = np.stack(log)
# # np.save('./data/grain_areas_2400p2_20000.npy', grain_areas)
# grain_areas_many = np.load('./data/grain_areas_2400p2_20000.npy')



# num_grains = [2000]

# for i in range(len(num_grains)):
#     mat = scipy.io.loadmat('./data/previous_figures/Case4GSizeMCPG%d.mat'%num_grains[i])
#     rn = mat['rnorm'][:,0]
#     h_mcp, x_edges = np.histogram(rn, bins='auto', density=True)
#     x_mcp = x_edges[:-1]+np.diff(x_edges)/2
#     n_mcp = len(rn)
    
#     mat = scipy.io.loadmat('./data/previous_figures/Case4GSizePFG%d.mat'%num_grains[i])
#     rn = mat['rnorm'][0]
#     h_pf, x_edges = np.histogram(rn, bins='auto', density=True)
#     x_pf = x_edges[:-1]+np.diff(x_edges)/2
#     n_pf = len(rn)
    
#     mat = scipy.io.loadmat('./data/previous_figures/RadiusDistPureTP2DNew.mat')
#     x_yad = mat['y1'][0]
#     h_yad = mat['RtotalFHD5'][0]
    
#     x_zol, h_zol = np.loadtxt('./data/previous_figures/Results.csv', delimiter=',',skiprows=1).T
    
#     plt.figure()
#     legs = []
#     # for j in range(len(nss)):
#     j=4
#     grain_areas = grain_areas_many[j]
#     n = (grain_areas!=0).sum(1)
#     k = np.argmin(np.abs(n-num_grains[i]))
#     a = grain_areas[k]
#     r = np.sqrt(a[a!=0]/np.pi)
#     rn = r/np.mean(r)
#     h_mf, x_edges = np.histogram(rn, bins='auto', density=True)
#     x_mf = x_edges[:-1]+np.diff(x_edges)/2
#     n_mf = len(rn)
    
#     plt.plot(x_mf, h_mf, '-')
#     legs.append('MF, $N_G$: %d, ns: %d'%(n_mf, nss[j]))
    
#     plt.plot(x_mcp, h_mcp, '-')
#     plt.plot(x_pf, h_pf, '-')
#     plt.plot(x_yad, h_yad, '*')
#     plt.plot(x_zol, h_zol, 'd')
#     plt.xlabel('$R/<R>$ - Normalized Radius')
#     plt.ylabel('Frequency')
#     plt.xlim([0,3])
#     plt.ylim([0,1.2])
#     plt.legend(legs+['MCP, $N_G$ - %d'%n_mcp, 'PF, $N_G$ - %d'%n_pf, 'Yadav 2018', 'Zollner 2016'])
#     plt.show()






# ### Number of samples vs slope of <R>^2 

# nss = [4,8,16,32,64,128,256,512] #number of samples
# var = 25
# num_grains = 2**12 #number of grains
# # ic, ea, _ = fs.voronoi2image([1024,1024], num_grains)
# # ma = fs.find_misorientation(ea, mem_max=10)

# # log_ng = []
# # log = []
# # for ns in nss:
# #     log_r2 = []
# #     log_ng0 = []
# #     for i in range(30):
# #         ims = fs.run_mf(ic, ea, nsteps=300, cut=0, cov=var, num_samples=ns, miso_array=ma, if_save=False)
# #         a = fs.iterate_function(ims, fs.find_grain_areas, [num_grains-1])
# #         r = np.sqrt(a/np.pi)
# #         ng = (r!=0).sum(1)
# #         ra = r.sum(1)/ng #find mean without zeros
# #         r2 = (ra**2) #square after the mean
# #         log_r2.append(r2)
# #         log_ng0.append(ng)
# #     log.append(log_r2)
# #     log_ng.append(log_ng0)

# # r2_many = np.block(log).reshape(8, 30, -1)
# # np.save('./data/r2_for_ns_vs_r2_slope.npy', r2_many)
# # r2_many = np.load('./data/r2_for_ns_vs_r2_slope.npy')

# # ng_many = np.block(log_ng).reshape(8, 30, -1)
# # np.save('./data/ng_for_ns_vs_r2_slope.npy', ng_many)
# # ng_many = np.load('./data/ng_for_ns_vs_r2_slope.npy')



# plt.plot(r2_many[0,0])

# r2_many = np.load('./data/r2_for_ns_vs_r2_slope.npy')

# log = []
# for i in range(len(nss)):
#     log0 = []
#     for j in range(30):
#         r2 = r2_many[i,j,:]
#         x = np.arange(len(r2))
#         p = np.sum(np.linalg.pinv(np.array(x)[:,None])*r2)
#         log0.append(p)
#     log.append(log0)
    
# s = np.block(log).mean(1)
# e = np.block(log).std(1)

# plt.errorbar(nss[:len(s)], s, 2*e, marker='.', ms=20, capsize=3) 

# x = np.array(nss[:len(s)])
# y = s

# def model_func(x, A, B, C):
#     return A * x**B + C

# def fit_nonlinear(t, y):
#     opt_parms, parm_cov = sp.optimize.curve_fit(model_func, t, y, maxfev=10000)
#     A, B, C = opt_parms
#     return A, B, C

# A, B, C = fit_nonlinear(x, y)
# x_new = np.arange(2028)
# yf = model_func(x_new, A, B, C)

# plt.plot(x_new, yf,'--')

# plt.title("Fit model: $y = A*x^B + C$")
# plt.legend(['Data','y = $%1.2f*x**{%1.2f} + %1.2f$'%(A,B,C)])
# plt.ylim([2,5])


# # def model_func(x, A, B, C, D):
# #     return (A*x+B)/(C*x+D) 

# # def fit_nonlinear(t, y):
# #     opt_parms, parm_cov = sp.optimize.curve_fit(model_func, t, y, maxfev=1000)
# #     A, B, C, D = opt_parms
# #     return A, B, C, D

# # A, B, C, D = fit_nonlinear(x, y)
# # x_new = np.arange(2028)
# # yf = model_func(x_new, A, B, C, D)

# # plt.figure()
# # plt.errorbar(nss[:len(s)], s, 2*e, marker='.', ms=20, capsize=3) 
# # plt.plot(x_new, yf,'--')
# # plt.title("Fit model: $y = (A*x + B)/(C*x+D)$")
# # plt.legend(['Data','y = $y = (%1.2f*x + %1.2f)/(%1.2f*x+%1.2f)$'%(A,B,C,D)])
# # plt.show()





# ### Discrete gaussian and resulting square grains 

# s = 16 #variance

# mu = [0, 0]
# sigma = np.array([[s,0],[0,s]])
# w, h = 10, 10
# std = [np.sqrt(sigma[0, 0]), np.sqrt(sigma[1, 1])]
# x = np.arange(-w, w+1)
# y = np.arange(-h, h+1)
# x, y = np.meshgrid(x, y)
# x_ = x.flatten()
# y_ = y.flatten()
# xy = np.vstack((x_, y_)).T
# normal_rv = multivariate_normal(mu, sigma)
# z = normal_rv.pdf(xy)
# z = z.reshape(2*w+1, 2*h+1, order='F')

# plt.imshow(z, extent=[-w, w,-h,h])
# plt.title('Var: %d'%s)


# ic, ea, _ = fs.voronoi2image([512,512], 1024)
# ims = fs.run_mf(ic, ea, nsteps=ns, cut=1000, cov=9, num_samples=64, if_save=False)
# a = fs.iterate_function(ims, fs.find_grain_areas, [2**14-1])
# r = np.sqrt(a/np.pi)
# n = (r!=0).sum(1)
# ra = r.sum(1)/n #find mean without zeros
# ra2 = (ra**2) #square after the mean
# plt.plot(ra2) 
# plt.show()

# plt.imshow(ims[-1,0])
# plt.title('var=%d'%s)







# ### Plot for counts variance
# var = 16
# s3 = np.sqrt(var)*3
# c = np.array([256,256]) #center

# ic, ea, _ = fs.voronoi2image([512,512], 1024)
# im = ic[int(c[0]-s3):int(c[0]+s3)+1, int(c[1]-s3):int(c[1]+s3)+1]
# plt.imshow(im)

# w, h = im.shape
# mu = [0, 0]
# sigma = np.array([[var,0],[0,var]])
# x = np.arange(w)-int(w/2)
# y = np.arange(h)-int(h/2) 
# x, y = np.meshgrid(x, y)
# x_ = x.flatten()
# y_ = y.flatten()
# xy = np.vstack((x_, y_)).T
# normal_rv = multivariate_normal(mu, sigma)
# z = normal_rv.pdf(xy)
# z = z.reshape(w, h, order='F')

# plt.imshow(z, extent=[-w/2, w/2, -h/2, h/2])
# plt.title('Var: %d'%var)


# u = np.unique(im)

# log = []
# for uu in u: 
#     log.append(z[im==uu].sum())





# np.sum(log)
# np.sum(z)


# ns = 64
# s = np.fix(normal_rv.rvs((ns)))
# print(np.min(s))
# print(np.max(s))

# ic[c[0]+s[0], c[1]+s[1]]



# ### Plot for number of pixel changes vs number of samples used
# nss = [4,8,16,32,64,128]
# ng = 1024
# sz = [512,512]

# ic, ea, _ = fs.voronoi2image(sz, ng)
# log = []
# for ns in nss:
#     ims = fs.run_mf(ic, ea, nsteps=50, cut=0, cov=36, num_samples=ns, if_save=False)
#     a = fs.iterate_function(ims, fs.find_grain_areas, [ng-1])
#     r = np.sqrt(a/np.pi)
#     n = (r!=0).sum(1)[:-1]
    
#     # c = np.sqrt((ims[1:,0]!=ims[:-1,0]).sum(1).sum(1)/np.product(sz))
#     c = (ims[1:,0]!=ims[:-1,0]).sum(1).sum(1)/np.product(sz)
#     plt.plot(n, c)
    
    
#     i = np.argmin(np.abs(n-600))
#     log.append(c[i])
    
# plt.plot(nss, log)




# slopes = []
# legend = []
# plt.figure()
# for ns in nss:
#     ims = fs.run_mf(ic, ea, nsteps=50, cut=0, cov=36, num_samples=ns, if_save=False)
#     a = fs.iterate_function(ims, fs.find_grain_areas, [ng-1])
#     r = np.sqrt(a/np.pi)
#     n = (r!=0).sum(1)[:-1]
    
#     # c = np.sqrt((ims[1:,0]!=ims[:-1,0]).sum(1).sum(1)/np.product(sz))
#     c = (ims[1:,0]!=ims[:-1,0]).sum(1).sum(1)/np.product(sz)
#     plt.plot(n, c)
    
    
#     i = np.argmin(np.abs(n-600))
#     c[i]
    
    
    
    
#     p = np.polyfit(ng-n, c, 1)
#     slopes.append(p[0])
#     legend.append('N samples: %d'%ns)

# plt.ylabel('SQRT of % pixels changed, single step')
# plt.xlabel('Number of grains')
# plt.xlim([1100, 200])
# plt.legend(legend)
# plt.show()


# plt.plot(nss, )




# #Example of plotting phase field average radius squared and average number of sides through time
# mat = scipy.io.loadmat('./data/Case4Data.mat')

# ng = mat['NumGrainsCase4'][:,0]
# plt.plot(ng)

# r = mat['RadiusCase4'][:,0]
# plt.plot(r**2)

# t = mat['TimeCase4'][:,0]
# plt.plot(t, r**2)

# ns_avg = pandas.read_csv('./data/pf/Kristien Everett - grain_neighbors_stats Case4_hd.csv').values[:,0]
# plt.plot(t[1:], ns_avg)

# gs_avg = pandas.read_csv('./data/pf/Kristien Everett - grain_size_stats Case4_hd.csv').values[:,0]
# plt.plot(t[1:], gs_avg**2)


# #Examples of plotting normalized radius distribution
# mat = scipy.io.loadmat('./data/previous_figures/Case4GSizeMCPG2000.mat')
# r = mat['rnorm'][:,0]
# plt.plot(r)
# plt.hist(r)

# mat = scipy.io.loadmat('./data/previous_figures/RadiusDistPureTP2DNew.mat')
# y1 = mat['y1'][0]
# RtotalFHD5 = mat['RtotalFHD5'][0]
# plt.plot(y1, RtotalFHD5)

# x, y = np.loadtxt('./data/previous_figures/Results.csv', delimiter=',',skiprows=1).T
# plt.plot(x,y)


# #Examples of plotting number of sides distribution
# mat = scipy.io.loadmat('./data/previous_figures_sides/Case4SidesMCPG2000.mat')
# s = mat['the_sides'][0]
# plt.plot(s)
# plt.hist(s)

# mat = scipy.io.loadmat('./data/previous_figures_sides/FaceDistPureTP2DNew.mat')
# y1 = mat['y1'][0]
# FtotalFHD5 = mat['FtotalFHD5'][0]
# plt.plot(y1, FtotalFHD5)

# x, y = np.loadtxt('./data/previous_figures_sides/ResultsMasonLazar2DGTD.txt').T
# plt.plot(x,y)




#Hipergator does not have the fonts
# plt.rcParams.update({'font.size': 10})
# plt.rcParams["font.family"] = "Times New Roman"
# plt.figure(figsize=(3.25, 6), dpi=300)


# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = "Times New Roman"

# cs = {'fontname':'Comic Sans MS'}
# plt.savefig('results/Fig10_Histograms of AUC Values.png', bbox_inches='tight', dpi=300)





# with h5py.File('./data/32c512grs512stsPkT066_img.hdf5', 'r') as f:
#     print(f['images'].shape)
#     ic = f['images'][0].astype(float)
#     ea = np.random.rand(512, 3)

# plt.imshow(ic)

# ims, fp_save = fs.run_mf(ic, ea, nsteps=1000, cut=0, cov=25, num_samples=64)
# fs.compute_grain_stats(fp_save)
# fs.make_time_plots(fp_save)








    
# len(np.unique(ic))
# plt.imshow(ic)

# with h5py.File('./data/mf_sz(2400x2400)_ng(20000)_nsteps(1000)_cov(25)_numnei(64)_cut(0).h5', 'r') as f:
#     im_first = f['sim0/ims_id'][0,0]
#     im = f['sim0/ims_id'][500,0]
#     im_last = f['sim0/ims_id'][-1,0]
#     print(f['sim0'].keys())

# len(np.unique(im_last))
# plt.imshow(im_last)








# fs.compute_grain_stats('./data/mf_sz(128x128x128)_ng(8192)_nsteps(1000)_cov(4)_numnei(64)_cut(0).h5')
# fs.compute_grain_stats('./data/mf_sz(128x128x128)_ng(8192)_nsteps(1000)_cov(4)_numnei(64)_cut(25).h5')



# ic = np.load('./data/ic_128p3_8192.npy').astype(float)[0,0]
# ea = np.load('./data/ea_128p3_8192.npy').astype(float)
# ma = np.load('./data/miso_array_128p3_8192.npy').astype(float)

# ic, ea, _ = fs.voronoi2image([256,]*3, 20000, memory_limit=10e9)

# np.save('./data/ic_256p3_20000.npy', ic)

# ea = np.load('./data/ea_20000.npy')

# miso_array = fs.find_misorientation(ea, mem_max=10)
# np.save('./data/miso_array_20000.npy', miso_array)

# ims, _ = fs.run_mf(ic, ea, nsteps=3, cut=25, cov=25, num_samples=64, miso_array=miso_array, if_save=False)





# ### Code to run MF and calculate stats
# with h5py.File('./data/spparks_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1.0)_kt(0.66)_cut(0).h5', 'r') as f:
#     ic = f['sim0/ims_id'][0,0].astype(float)
#     scale = np.pi*np.array([2, 0.5, 2])[None,]
#     ea = np.random.rand(20000, 3)*scale
    
# ims, _ = fs.run_mf(ic, ea, nsteps=1000, cut=25, cov=25, num_samples=64, if_save=True)
# fs.compute_grain_stats('./data/mf_sz(2400x2400)_ng(20000)_nsteps(1000)_cov(25)_numnei(2)_cut(25).h5')












# ic, ea, _ = fs.voronoi2image([256,]*3, 20000, memory_limit=10e9)

# ic = np.load('./data/ic_180p3_20000.npy')
# ea = np.load('./data/ea_180p3_20000.npy')
# ma = np.load('./data/miso_array_180p3_20000.npy')

# ims = fs.run_mf(ic, ea, nsteps=5, cut=0, cov=9, num_samples=64, miso_array=ma, if_save=True)
# ims = fs.run_mf(ic, ea, nsteps=1000, cut=25, cov=9, num_samples=64, if_save=True)

# fs.compute_grain_stats('./data/mf_sz(128x128x128)_ng(4096)_nsteps(1000)_cov(9)_numnei(64)_cut(0).h5')
# fs.compute_grain_stats('./data/mf_sz(128x128x128)_ng(4096)_nsteps(1000)_cov(9)_numnei(64)_cut(25).h5')








# ic = np.load('./data/ic_128p3_8192.npy').astype(float)[0,0]
# ea = np.load('./data/ea_128p3_8192.npy').astype(float)
# ma = np.load('./data/miso_array_128p3_8192.npy').astype(float)

# ims = fs.run_mf(ic, ea, nsteps=1000, cut=0, cov=4, num_samples=64, miso_array=ma, if_save=True)
# ims = fs.run_mf(ic, ea, nsteps=1000, cut=25, cov=4, num_samples=64, miso_array=ma, if_save=True)

# fs.compute_grain_stats('./data/mf_sz(128x128x128)_ng(8192)_nsteps(1000)_cov(4)_numnei(64)_cut(0).h5')
# fs.compute_grain_stats('./data/mf_sz(128x128x128)_ng(8192)_nsteps(1000)_cov(4)_numnei(64)_cut(25).h5')











# with h5py.File('./data/mf_sz(128x128x128)_ng(4096)_nsteps(1000)_cov(9)_numnei(64)_cut(0).h5', 'r') as f:
#     print(f['sim0'].keys())
#     ga = f['sim0/grain_areas'][:]
    
# ng = (ga!=0).sum(1)
# plt.plot(ng)    
# ims = f['sim0/ims_id'][:]
# scale = np.pi*np.array([2, 0.5, 2])[None,]
# ea = np.random.rand(20000, 3)*scale


# im = ims[400,0]

# np.unique(im)


# plt.imshow(im[0])





