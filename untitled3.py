#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 14:36:25 2023

@author: joseph.melville
"""


import numpy as np
from uvw import RectilinearGrid, DataArray

# Creating coordinates
x = np.linspace(-0.5, 0.5, 10)
y = np.linspace(-0.5, 0.5, 20)
z = np.linspace(-0.9, 0.9, 30)

# Creating the file (with possible data compression)
grid = RectilinearGrid('grid2.vtr', (x, y, z), compression=True)

# # A centered ball
# x, y, z = np.meshgrid(x, y, z, indexing='ij')
# r = np.sqrt(x**2 + y**2 + z**2)
# ball = (r < 0.3).astype(int)

# # Some multi-component multi-dimensional data
# data = np.zeros([10, 20, 30, 3, 3])
# data[ball, ...] = np.array([[0, 1, 0],
#                             [1, 0, 0],
#                             [0, 1, 1]])



# Some cell data
cell_data = np.zeros([9, 19, 29])
# cell_data[0::2, 0::2, 0::2] = 1
cell_data[0:2, 0:2, 0:2] = 1
cell_data[2:6, 10:12, 20:22] = 2

# Adding the point data (see help(DataArray) for more info)
# grid.addPointData(DataArray(data, range(3), 'ball'))
# Adding the cell data
grid.addCellData(DataArray(cell_data, range(3), 'checkers'))
grid.write()




with h5py.File('./data/mf_sz(128x128x128)_ng(4096)_nsteps(1000)_cov(25)_numnei(64)_cut(0).h5', 'r') as f:
    ims = f['sim0/ims_id'][0,0]




create_3D_paraview_vtr(ims)



