# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IF THIS CODE IS USED FOR A RESEARCH PUBLICATION, please cite:
    Melville, J., Yadav, V., Yang, L., Krause, A. R., Tonks, M. R., & Harley, J. B. (2024). A new efficient grain growth model using a random Gaussian-sampled mode filter. Materials & Design, 237, 112604.
"""

import functions as fs



#Two Dimensions
#Initial condition
sz = [2048, 2048] #size of initial microstructure
ng = 2**14 #number of grains initial microstructure
ic, ea, _ = fs.voronoi2image(sz, ng)
ma = fs.find_misorientation(ea, mem_max=10)

#Mode filter
ns = 64 #number of samples per neighborhood
cm = [[25,0],[0,25]] #neighborhood covariance matrix
c = 0 #Read-Shockley cut off angle (increase for anisotropic growth)
ims = fs.run_mf(ic, ea, nsteps=10, cut=c, cov=cm, num_samples=ns, miso_array=ma, if_save=False)
            


#Three Dimensions
#Initial condition
sz = [64, 64, 64] #size of initial microstructure
ng = 2**12 #number of grains initial microstructure
ic, ea, _ = fs.voronoi2image(sz, ng)
ma = fs.find_misorientation(ea, mem_max=10)

#Mode filter
ns = 64 #number of samples per neighborhood
cm = [[4,0,0],[0,4,0],[0,0,4]] #neighborhood covariance matrix
c = 0 #Read-Shockley cut off angle (increase for anisotropic growth)
ims = fs.run_mf(ic, ea, nsteps=10, cut=c, cov=cm, num_samples=ns, miso_array=ma, if_save=False)         