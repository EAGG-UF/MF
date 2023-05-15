# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 11:48:09 2021

@author: josep
"""


import numpy as np
import h5py 
from tqdm import tqdm
import matplotlib.pyplot as plt


def dump_to_hdf5(path_to_dump="Circle_512by512", path_to_hdf5="Circle_512by512", num_steps=None):
    #A more general purpose extract dump file - reads lines directly to an hdf5 file and saves header names
    #The lines can then be extracted one-by-one from the hdf5 file and converted to an image
    #"num_steps" is a guess at how many entries there are in the dump file to report how long it will take
    
    with open(path_to_dump+".dump") as file:
        bounds = np.zeros([3,2])
        for i, line in enumerate(file): #extract the number of atoms, bounds, and variable names from the first few lines
            if i==3: num_atoms = int(line) 
            if i==5: bounds[0,:] = np.array(line.split(), dtype=float)
            if i==6: bounds[1,:] = np.array(line.split(), dtype=float)
            if i==7: bounds[2,:] = np.array(line.split(), dtype=float)
            if i==8: var_names = line.split()[2:]
            if i>8: break
    bounds = np.ceil(bounds).astype(int) #reformat bounds
    entry_length = num_atoms+9 #there are 9 header lines in each entry
    
    if num_steps!=None: total_lines = num_steps*entry_length
    else: total_lines=None
    
    time_steps = []
    with h5py.File(path_to_hdf5+".hdf5", 'w') as f:
        f["bounds"] = bounds #metadata
        f["variable_names"] = [x.encode() for x in var_names] #metadata
        dset = f.create_dataset("dump_extract", shape=(1,num_atoms,len(var_names)), maxshape=(None,num_atoms,len(var_names)))#, chunks=True)
        with open(path_to_dump+".dump") as file:
            for i, line in tqdm(enumerate(file), "EXTRACTING SPPARKS DUMP (%s.dump)"%path_to_dump, total=total_lines):
                [entry_num, line_num] = np.divmod(i,entry_length) #what entry number and entry line number does this line number indicate
                if line_num==0: entry = np.zeros([num_atoms, len(var_names)]) #reset the entry values at the beginning of each entry
                if line_num==1: time_steps.append(int(line.split()[-1])) #log the time step
                atom_num = line_num-9 #track which atom line we're on
                if atom_num>0 and atom_num<num_atoms: entry[atom_num,] = np.array(line.split(), dtype=float) #record valid atom lines
                if line_num==entry_length-1: 
                    dset[-1,:,:] = entry #save this entry before going to the next
                    dset.resize(dset.shape[0]+1, axis=0) #make more room in the hdf5 dataset
        dset.resize(dset.shape[0]-1, axis=0) #remove the extra room that wasn't used
        time_steps = np.array(time_steps) #reformat time_steps
        f["time_steps"] = time_steps #metadata
        
    return var_names, time_steps, bounds


def dump_extract_to_images(path_to_hdf5="Hex_443by512", new_path="Hex_443by512_2", vi=1, xi=2, yi=3):
    #Convert hdf5 dataset "dump_extract" to "images" in a new hdf5 file
    #This function is intended to be used to process the raw extracted data from a SPPARKS dump (e.g. extract images)
    #"vi" is the index in the third dimension of "f[("dump_extract")]" that corresponds to the image pixels 
    #"xi" and "yi" are the indices in the third dimension of "f[("dump_extract")]" that correspond position of each pixel value

    with h5py.File(path_to_hdf5+".hdf5", 'a') as f:
        b = f[("bounds")]
        e = f[("dump_extract")]
        s = e.shape
        ii = (e[0,:,xi]*b[1,1]+e[0,:,yi]).astype(int) #calculate indicies for remapping the pixels
        
        #Find the smallest data type that can be used without overflowing
        m = np.max(e[0,:,vi])
        tmp = np.array([8,16,32], dtype='uint64')
        dtype = 'uint' + str(tmp[np.sum(m>2**tmp)])
        
        with h5py.File(new_path+".hdf5", 'w') as ff: #open new hdf5 to write to
            ims = ff.create_dataset("images", shape=(tuple([s[0]])+tuple(b[0:2,1].tolist())), dtype=dtype)
            
            for i in tqdm(range(s[0]), "EXTRACT ID IMAGES FROM HDF5 FILE (%s.hdf5)"%path_to_hdf5):
                ee = np.zeros([s[1]])
                ee[ii] = e[i,:,vi]
                ims[i,] = ee.reshape(b[0:2,1])
        
            plt.imshow(ims[0,]) #show the first image as a sample



#File names just for easy access:
# Circle_512by512
# Hex_443by512
# 512grsPoly_512by512_Periodic
# 512grsPoly_512by512_NonPeriodic
# 32c20000grs2400sts

path_to_dump = "32c64grs443sts"
num_steps=100 #just for calculating how much time it will take (doesn't have to be exactly correct)
var_names, time_step, bounds = dump_to_hdf5(path_to_dump, path_to_dump, num_steps) #copies raw dump extract to hdf5 file
dump_extract_to_images(path_to_dump, path_to_dump+"_2") #processes raw dump extract in hdf5 file to get images then deletes raw data


# instructions to work with hdf5 file
# f = h5py.File(path_to_dump+".hdf5", 'a')
# ims =f[('images')]
# #do processing on "ims" here
# f.close()

# with h5py.File(path_to_hdf5+".hdf5", 'a') as f:
#     ims =f[('images')]
#     #do processing on "ims" here  
    
    
    
    