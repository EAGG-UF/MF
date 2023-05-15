"""
Created on Thu Jan 14 09:59:32 2021

@author: joseph.melville
"""



'''
File Description:
    The file contains functions for working with spparks simulations. 
    Write SPPARKS parameter files to SPPARKS environment (e.g. ".init", ".in", ".sh")
    Read simulation results from SPPARKS environment (e.g. ".dump", ".init", ".cluster")
    Process simulation data (e.g. "num_diff_neighbors")
    Assumes: 
        path_to_sim_2D = "../examples/agg/2d_sim/"
        path_to_sim_3D = "../examples/agg/3d_sim/"
        path_to_data = "../../GrainGrowth/data/"
        current_folder = "PRIMME"
    
Example 1: Running and capturing simulations in an HDF5 file

import sPkY as fs

size = [64, 32, 128] #2D or 3D, rectangels work
ngrain = 64 #number of grains
nsteps = 50 #SPPARKS steps to run
freq_dump = 1 #how offten to dump an image (record)
freq_stat = 1 #how often to report stats on the simulation
rseed = 4951 #change to get different growth from teh same initial condition
hf_name = "test" #name of hdf5 file to write to

img, EulerAngles, center_coords0 = fs.voronoi2image(size, ngrain) #generate initial condition
fs.image2init(img, EulerAngles) #write initial condition
path_sim = fs.run_spparks(size, ngrain, nsteps, freq_dump, freq_stat, rseed) #run simulation
euler_angle_images, sim_steps, grain_euler_angles, grain_ID_images, energy_images = fs.extract_spparks_dump(dim=len(size)) #extract SPPARKS dump data to python
fs.capture_sim(hf_name, dim) #write dump file to an hdf5 file for storage/reference in python 
'''



#IMPORT LIBRARIES
import numpy as np
import math
from tqdm import tqdm
import os
import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from unfoldNd import unfoldNd #pip install unfoldNd
import pynvml #pip install pynvml
from scipy.special import gammainc
from scipy.stats import skew, kurtosis
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import dropbox #pip install dropbox



def show_gpus():
    pynvml.nvmlInit()
    count = torch.cuda.device_count()
    for i in range(count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_free = mem_info.free / 1024 ** 3
        device_name = torch.cuda.get_device_name(device=i)
        print("%d: Memory Free - %2.2f \tGB (%s)"%(i, mem_free, device_name))
        #torch.cuda.empty_cache()
        
        

def count_tags(fp = r"./edit_files/agg_poly_edit.in"):
    '''
    Returns and print ths number of tags (##<counting numbers>##) found in a given file.

    Parameters
    ----------
    fp : string, optional
        Path to text file to be read in. The default is r"./output/agg_poly_edit.in".

    Returns
    -------
    num_tags : int
        Number of tags found in the file.

    '''    

    # Read the file into a string
    with open(fp, 'r') as f:
        f_str = f.read()

    # Count the number of tags
    num_tags = 0;
    while 1: 
        if "##%d##"%(num_tags+1) in f_str:
            num_tags += 1
        else: 
            print("There are %d tags in '%s'"%(num_tags, fp))
            return num_tags
        
        

def replace_tags(fp_in = r"./edit_files/spparks_2d.in", 
                      replacement_text = ['45684', '512', '511.5', '511.5', '1', '10', '50', '500', "agg"], 
                      fp_out = r"../examples/agg/2d_sim/agg_poly.in",
                      print_chars = [0,0]):
    '''
    This function takes the txt file at file_path, replaces markers in the
    text (##<counting numbers>##) with the strings provided in
    replacement_text. (Markers need to be placed in the target txt file by
    the user ahead of time.) 
    
    Variables:
        fp_in (*.txt): path to text file to be read in
        replacement_text (list of strings): text to replace each marker with
        fp_out (*.txt): path to text file to be written to
        print_chars (list of two integers): the first and last character to print in the file
    '''
    
    # Read the file into a string
    with open(fp_in, 'r') as f:
        f_str = f.read()
    
    # Print some lines from the file before substitution
    if sum(print_chars) != 0: print(f_str[int(print_chars[0]):int(print_chars[1])])
    
    # Replace tags with the text replacement_text
    for i, rt in enumerate(replacement_text):
        f_str = f_str.replace("##%d##"%(i+1),rt)
        
    # Print some lines from the file after substitution
    if sum(print_chars) != 0: print(f_str[int(print_chars[0]):int(print_chars[1])])
    
    # Write string to a file
    with open(fp_out, 'w') as f:
        f.write(f_str)
        
    # Completion Message
    print("TAGS REPLACED - CREATED: %s"%fp_out)



# def voronoi2image(size=512, ngrain=128, dim=2):
#     '''
#     Creates an image of grain IDs (and euler angles for each ID) for the initial condition file for SPPARKS simulations 

#     Parameters
#     ----------
#     size : int, optional
#         The square dimensions of the microstructure. The default is 512.
#     ngrain : int, optional
#         Number of grains. The default is 64.
#     dim : int, optional
#         Number of dimensions for the output image. The default is 2.
        
#     Returns
#     -------
#     grain_ID_image (numpy, shape=[size,size,*size]): pixels indicate the grain ID of the grain it belongs to
#     grain_euler_angles (numpy, shape=[ngrain,3]): number of grains by three Euler angles
    
#     '''
    
#     #Randomly generate grain centers
#     if dim==3: img = np.zeros((size,size,size))
#     else: img = np.zeros((size,size))
#     GCords = np.zeros((ngrain,3))
#     for i in range(0,ngrain):
#         GCords[i,0],GCords[i,1],GCords[i,2]= np.random.randint(size),np.random.randint(size),np.random.randint(size)
    
#     #Paint each domain site according to which grain center is closest
#     print("\n"); time.sleep(0.2)
#     if dim==3: 
#         for i in tqdm(range(0,size), "CREATING NEW %dD IC (size: %d, ngrain: %d)"%(dim, size, ngrain)):
#             for j in range(0,size):
#                 for h in range(0,size):
#                     SiteID = 0
#                     MinDist = math.sqrt((GCords[0,0]-h)**2+(GCords[0,1]-j)**2+(GCords[0,2]-i)**2)
#                     for k in range(1,ngrain):
#                         dist = math.sqrt((GCords[k,0]-h)**2+(GCords[k,1]-j)**2+(GCords[k,2]-i)**2) 
#                         if dist < MinDist:
#                             SiteID = k
#                             MinDist = dist
#                     img[h,j,i] = SiteID
                
#     else:
#         for i in tqdm(range(0,size), "CREATING NEW %dD IC (size: %d, ngrain: %d)"%(dim, size, ngrain)):
#             for j in range(0,size):
#                 SiteID = 0
#                 MinDist = math.sqrt((GCords[0,0]-j)**2+(GCords[0,1]-i)**2)
#                 for k in range(1,ngrain):
#                     dist = math.sqrt((GCords[k,0]-j)**2+(GCords[k,1]-i)**2) 
#                     if dist < MinDist:
#                         SiteID = k
#                         MinDist = dist
#                 img[j,i] = SiteID
    
#     #Randomly generate Euler Angles for each grain
#     EulerAngles = np.zeros((ngrain,3))
#     for i in range(0,ngrain):
#        EulerAngles[i,0] = 2*math.pi*np.random.uniform(0,1)
#        EulerAngles[i,1] = 0.5*math.pi*np.random.uniform(0,1)
#        EulerAngles[i,2] = 2*math.pi*np.random.uniform(0,1)
    
#     return img, EulerAngles





# def distance(x0, x1, dimensions): #periodic, but not parallel - I can probably do this matrices...?
#     delta = numpy.abs(x0 - x1)
#     delta = numpy.where(delta > 0.5 * dimensions, delta - dimensions, delta)
#     return numpy.sqrt((delta ** 2).sum(axis=-1))



def generate_circleIC(size=[512,512], r=64, c=None):
    if c==None: c = ((torch.Tensor(size)-1)/2)
    else: c = torch.Tensor(c)
    r = torch.Tensor([r])
    a1 = torch.arange(size[0]).unsqueeze(1).repeat(1, size[1])
    a2 = torch.arange(size[1]).unsqueeze(0).repeat(size[0], 1)
    img = (torch.sqrt((c[0]-a1)**2+(c[1]-a2)**2)<r).float()
    euler_angles = math.pi*torch.rand((2,3))*torch.Tensor([2,0.5,2])
    return img, euler_angles


def generate_3grainIC(size=[512,512], h=350):
    img = torch.ones(512, 512)
    img[size[0]-h:,256:] = 0
    img[size[1]-h:,:256] = 2
    euler_angles = math.pi*torch.rand((3,3))*torch.Tensor([2,0.5,2])
    return img.numpy(), euler_angles


def generate_hex_grain_centers(dim=512, dim_ngrain=8):
    #Generates grain centers that can be used to generate a voronoi tesselation of hexagonal grains
    #"dim" is the dimension of one side length, the other is calculated to fit the same number of grains in that direction
    #"dim_ngrain" is the number of grains along one one dimension, it is the same for both dimensions
    mid_length = dim/dim_ngrain #length between two flat sides of the hexagon
    side_length = mid_length/np.sqrt(3) #side length of hexagon
    size = [int(dim*np.sqrt(3)/2), dim] #image size
    
    r1 = torch.arange(1.5*side_length, size[0], 3*side_length).float() #row coordinates of first column
    r2 = torch.arange(0, size[0], 3*side_length).float() #row coordinates of second column
    c1 = torch.arange(0, size[1], mid_length).float() #column coordinates of first row
    c2 = torch.arange(mid_length/2, size[1], mid_length).float() #column coordinates of second row
    
    centers1 = torch.cartesian_prod(r1, c1) #take all combinations of first row and column coordinates
    centers2 = torch.cartesian_prod(r2, c2) #take all combinations of second row and column coordinates
    grain_centers = torch.cat([centers1,centers2], dim=0)[torch.randperm(dim_ngrain**2)]
    return grain_centers


def generate_random_grain_centers(size=[128, 64, 32], ngrain=512):
    grain_centers = torch.rand(ngrain, len(size)).to(size.device)*size
    return grain_centers


def write_grain_centers_txt(center_coords, fp="grains"):
    #Generate a "grains.txt" of grain center locations for use in MOOSE simulations (https://github.com/idaholab/moose/blob/next/modules/phase_field/test/tests/initial_conditions/grains.txt)
    #The file is written to the current directory and can be used for 2D or 3D "size" inputs
    
    if center_coords.shape[1]==2: header = "x y\n"
    else: header = "x y z\n"

    np.savetxt("%s.txt"%fp, center_coords, delimiter=' ', fmt='%.5f')
    
    with open("%s.txt"%fp, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(header + content)

def read_grain_centers_txt(fp="Case2_grains_centers"):
    with open("%s.txt"%fp) as file:
        lines = file.readlines()
        lines = [x.split() for x in lines]
        grain_centers = torch.Tensor(np.array(lines[1:]).astype(float))
    return grain_centers


def voronoi2image(size=[128, 64, 32], ngrain=512, memory_limit=1e9, p=2, center_coords0=None):          
    
    #SETUP AND EDIT LOCAL VARIABLES
    if type(size)!=torch.Tensor:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        size = torch.Tensor(size).to(device)
    device = size.device
    dim = len(size)
    if center_coords0!=None: center_coords0 = center_coords0.to(device)
    
    #GENERATE RANDOM GRAIN CENTERS
    # center_coords = torch.cat([torch.randint(0, size[i], (ngrain,1)) for i in range(dim)], dim=1).float().to(device)
    # center_coords0 = torch.cat([torch.randint(0, size[i], (ngrain,1)) for i in range(dim)], dim=1).float()
    if center_coords0==None: center_coords0 = generate_random_grain_centers(size, ngrain).to(device)
    else: ngrain = center_coords0.shape[0]
    center_coords = torch.Tensor([]).to(device)
    for i in range(3): #include all combinations of dimension shifts to calculate periodic distances
        for j in range(3):
            if len(size)==2:
                center_coords = torch.cat([center_coords, center_coords0 + size*(torch.Tensor([i,j]).to(device)-1)])
            else: 
                for k in range(3):
                    center_coords = torch.cat([center_coords, center_coords0 + torch.Tensor(size)*(torch.Tensor([i,j,k]).to(device)-1)])
    center_coords = center_coords.float().to(device)
    
    #CALCULATE THE MEMORY NEEDED TO THE LARGE VARIABLES
    mem_center_coords = float(64*dim*center_coords.shape[0])
    mem_cords = 64*torch.prod(size)*dim
    mem_dist = 64*torch.prod(size)*center_coords.shape[0]
    mem_ids = 64*torch.prod(size)
    available_memory = memory_limit - mem_center_coords - mem_ids
    batch_memory = mem_cords + mem_dist
    
    #CALCULATE THE NUMBER OF BATCHES NEEDED TO STAY UNDER THE "memory_limit"
    num_batches = torch.ceil(batch_memory/available_memory).int()
    num_dim_batch = torch.ceil(num_batches**(1/dim)).int() #how many batches per dimension
    dim_batch_size = torch.ceil(size/num_dim_batch).int() #what's the size of each of the batches (per dimension)
    num_dim_batch = torch.ceil(size/dim_batch_size).int() #the actual number of batches per dimension (needed because of rouning error)
    
    if available_memory>0: #if there is avaiable memory
        #CALCULATE THE ID IMAGE
        all_ids = torch.zeros(tuple(size.int().cpu().numpy())).type(torch.int16)           
        ref = [torch.arange(size.cpu().numpy()[i]).int() for i in range(dim)] #aranges of each dimension length
        tmp = tuple([torch.arange(i).int() for i in num_dim_batch]) #make a tuple to iterate with number of batches for dimension
        for itr in tqdm(torch.cartesian_prod(*tmp).to(device)): #asterisk allows variable number of inputs as a tuple
            
            start = itr*dim_batch_size #sample start for each dimension
            stop = (itr+1)*dim_batch_size #sample end for each dimension
            stop[stop>=size] = size[stop>=size].int() #reset the stop value to the end of the dimension if it went over
            indicies = [ref[i][start[i]:stop[i]] for i in range(dim)] #sample indicies for each dimension
            
            coords = torch.cartesian_prod(*indicies).float().to(device) #coordinates for each pixel
            dist = torch.cdist(center_coords, coords, p=p) #distance between each pixel and the "center_coords" (grain centers)
            ids = (torch.argmin(dist, dim=0).reshape(tuple(stop-start))%ngrain).int() #a batch of the final ID image (use modulo/remainder quotient to account for periodic grain centers)
            
            if dim==2: all_ids[start[0]:stop[0], start[1]:stop[1]] = ids
            else: all_ids[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]] = ids
            
        total_memory = batch_memory + mem_center_coords + mem_ids #total memory used by this function
        print("Total Memory: %3.3f GB, Batches: %d"%(total_memory/1e9, num_batches))
        
        #GENERATE RANDOM EULER ANGLES FOR EACH ID
        euler_angles = torch.stack([2*math.pi*torch.rand((ngrain)), \
                              0.5*math.pi*torch.rand((ngrain)), \
                              2*math.pi*torch.rand((ngrain))], 1)
            
        return all_ids, euler_angles.cpu().numpy(), center_coords0.cpu().numpy()
            
    else: 
        print("Available Memory: %d - Increase memory limit"%available_memory)
        return None, None, None
    
    
    
def image2init(img, EulerAngles, fp=None):
    '''
    Takes an image of grain IDs (and euler angles assigned to each ID) and writes it to an init file for a SPPARKS simulation
    The initial condition file is written to the 2D or 3D file based on the dimension of 'img'
    
    Inputs:
        img (numpy, integers): pixels indicate the grain ID of the grain it belongs to
        EulerAngles (numpy): number of grains by three Euler angles
    '''
    # Set local variables
    size = img.shape
    dim = len(img.shape)
    if fp==None: fp = r"../examples/agg/%sd_sim/PolyIC.init"%str(dim)
    IC = [0]*(np.product(size)+3)
    
    # Write the information in the SPPARKS format and save the file
    IC[0] = '# This line is ignored\n'
    IC[1] = 'Values\n'
    IC[2] = '\n'
    k=0
    
    if dim==3: 
        for i in range(0,size[2]):
            for j in range(0,size[1]):
                for h in range(0,size[0]):
                    SiteID = int(img[h,j,i])
                    IC[k+3] = str(k+1) + ' ' + str(int(SiteID+1)) + ' ' + str(EulerAngles[SiteID,0]) + ' ' + str(EulerAngles[SiteID,1]) + ' ' + str(EulerAngles[SiteID,2]) + '\n'
                    k = k + 1
    
    else:
        for i in range(0,size[1]):
            for j in range(0,size[0]):
                SiteID = int(img[j,i])
                IC[k+3] = str(k+1) + ' ' + str(int(SiteID+1)) + ' ' + str(EulerAngles[SiteID,0]) + ' ' + str(EulerAngles[SiteID,1]) + ' ' + str(EulerAngles[SiteID,2]) + '\n'
                k = k + 1
            
    with open(fp, 'w') as file:
        file.writelines(IC)
        
    # Completion message
    print("NEW IC WRITTEN TO FILE: %s"%fp)
    


def image2init_rnd(size=64, dim=2, fp=None):
    '''
    Takes an image of grain IDs (and euler angles assigned to each ID) and writes it to an init file for a SPPARKS simulation
    The initial condition file is written to the 2D or 3D file based on the dimension of 'img'
    
    Inputs:
        img (numpy, integers): pixels indicate the grain ID of the grain it belongs to
        EulerAngles (numpy): number of grains by three Euler angles
    '''
    # Set local variables
    if fp==None: fp = r"../examples/agg/%sd_sim/PolyIC.init"%str(dim)
    
    # Write the information in the SPPARKS format and save the file
    if dim==3: IC = [0]*(size*size*size+3)
    else: IC = [0]*(size*size+3)
    IC[0] = '# This line is ignored\n'
    IC[1] = 'Values\n'
    IC[2] = '\n'
    k=0
    for i in range(0,size):
        for j in range(0,size):
            if dim==3: 
                for h in range(0,size):
                    SiteID = h+(j+i*size)*size
                    eulers = [2*math.pi*np.random.uniform(0,1), 0.5*math.pi*np.random.uniform(0,1), 2*math.pi*np.random.uniform(0,1)]
                    IC[k+3] = str(k+1) + ' ' + str(int(SiteID+1)) + ' ' + str(eulers[0]) + ' ' + str(eulers[1]) + ' ' + str(eulers[2]) + '\n'
                    k = k + 1
            else:
                SiteID = j+i*size
                eulers = [2*math.pi*np.random.uniform(0,1), 0.5*math.pi*np.random.uniform(0,1), 2*math.pi*np.random.uniform(0,1)]
                IC[k+3] = str(k+1) + ' ' + str(int(SiteID+1)) + ' ' + str(eulers[0]) + ' ' + str(eulers[1]) + ' ' + str(eulers[2]) + '\n'
                k = k + 1
            
    with open(fp, 'w') as file:
        file.writelines(IC)
        
    # Completion message
    print("NEW IC WRITTEN TO FILE: %s"%fp)
        
     
        
def init2image(dim=2, just_params=True): #OBSOLETE SINCE ALLOWING IMAGES OF DIFFERENT SIZE - NEEDS SIZE INPUT NOW
    '''
    Reads an inital condition file and converts it to an image of grain IDs (an euler angles for ID), or just returns the size and dimension parameters
    
    Inputs:
        dim (int): the dimension of the image to be read in
        just_params (bool): if ture, only returns the parameters, does not create images
    
    Outputs: 
        img (numpy, shape=[size,size,*size]): pixels indicate the grain ID of the grain it belongs to
        EulerAngles (numpy, shape=[ngrain,3]): number of grains by three Euler angles
            OR
        size (int): the size of one dimensions in the img
        ngrain (int): number of grains
        
    '''
    
    # Read in init file
    fp = r"../examples/agg/%sd_sim/PolyIC.init"%str(dim)
    with open(fp, 'r') as f:   
        f_lines = f.readlines()
    f_lines = f_lines[3:] #These lines are just comments
    
    # Find parameters
    if dim==3: size = int(np.cbrt(len(f_lines)))
    else: size = int(np.sqrt(len(f_lines)))
    ngrain = 1
    for l in f_lines:
        grain_ID = int(l.split()[1])
        if grain_ID > ngrain: 
            ngrain = grain_ID
    
    if not just_params:
        # Create images from file
        if dim==3: 
            img = np.zeros((size,size,size))
            EulerAngles = np.zeros((ngrain,3))
        else:
            img = np.zeros((size,size))
            EulerAngles = np.zeros((ngrain,3))
        
        # Populate from file
        for i in range(0,size):
            for j in range(0,size):
                if dim==3: 
                    for h in range(0,size):
                        l = f_lines[h+(j+i*size)*size]
                        ls = l.split()
                        grain_ID = int(ls[1])
                        euler_angles = np.array(ls[2:], dtype=float)
                        img[h,j,i] = grain_ID
                        EulerAngles[grain_ID-1] = euler_angles
                
                else:
                    l = f_lines[j+i*size]
                    ls = l.split()
                    grain_ID = int(ls[1])
                    euler_angles = np.array(ls[2:], dtype=float)
                    img[j,i] = grain_ID
                    EulerAngles[grain_ID-1] = euler_angles
                    
        return img, EulerAngles
    return size, ngrain



def calc_MisoEnergy(fp=r"../examples/agg/2d_sim/"):
    # Caclulates and writes MisoEnergy.txt from Miso.txt in the given file path 'fp'
    with open(fp + "Miso.txt", 'r') as f: f_str = f.read()
    miso = np.asarray(f_str.split('\n')[0:-1], dtype=float)
    
    theta = miso;
    theta = theta*(theta<1)+(theta>1);
    gamma = theta*(1-np.log(theta));
    
    tmp =[]
    for i in gamma: tmp.append('%1.6f'%i); tmp.append('\n')
    with open(fp + "MisoEnergy.txt", 'w') as file: file.writelines(tmp)
    
    
    
# def num_diff_neighbors(ims_unfold): 
#     #ims_unfold - torch tensor of shape = [N, product(kernel_size), dim1*dim2] from [N, 1, dim1, dim2] using "torch.nn.Unfold" object
#     #Addtiional dimensions to ims_unfold could be included at the end
#     center_pxl_ind = int(ims_unfold.shape[1]/2)
#     return torch.sum(ims_unfold[:,center_pxl_ind,] != ims_unfold.transpose(0,1), dim=0) #shape = [N, dim1*dim2]  


def pad_mixed(ims, pad, pad_mode="reflect"):
    #Allows for padding of "ims" with different padding modes per dimension
    #ims: shape = (num images, num channels, dim1, dim2, dim3(optional))
    #pad: e.g. pad = (1, 1, 2, 2) - pad last dim by (1, 1) and 2nd to last by (2, 2)
    #pad_mode: Same pad modes as "F.pad", but can be a list to pad each dimension differently
    #e.g. pad_mode = ["circular", "reflect"] - periodic boundary condition on last dimension, Neumann (zero flux) on 2nd to last
    
    if type(pad_mode)==list: 
        dims = len(ims.shape)-2
        pad_mode = pad_mode + [pad_mode[-1]]*(dims-len(pad_mode)) #copy last dimension if needed
        ims_padded = ims
        for i in range(dims):
            pad_1d = (0,)*i*2 + pad[i*2:(i+1)*2] + (0,)*(dims-i-1)*2
            ims_padded = F.pad(ims_padded.float(), pad_1d, pad_mode[i])
    else:
        ims_padded = F.pad(ims.float(), pad, pad_mode) #if "pad_mode"!=list, pad dimensions simultaneously
    return ims_padded


def my_unfoldNd(ims, kernel_size=3, pad_mode='circular'):
    #Pads "ims" before unfolding
    #ims.shape = (number of images, number of channels, dim1, dim2, optional dim3)
    dims = len(ims.shape)-2
    if type(kernel_size)!=list: kernel_size = [kernel_size] #convert to "list" if it isn't
    kernel_size = tuple(kernel_size + [kernel_size[-1]]*(dims-len(kernel_size))) #copy last dimension if needed
    pad = tuple((torch.Tensor(kernel_size).repeat_interleave(2)/2).int().numpy()) #calculate padding needed based on kernel_size
    ims_padded = pad_mixed(ims, pad, pad_mode) #pad "ims" to maintain dimensions after unfolding
    ims_unfold = unfoldNd(ims_padded, kernel_size=kernel_size) #shape = [N, product(kernel_size), dim1*dim2*dim3]
    return ims_unfold


    # if type(kernel_size)==int: kernel_size = [kernel_size] #convert to "list" if "int" is given
    # kernel_size = kernel_size + [kernel_size[-1]]*(len(ims.shape)-2-len(kernel_size)) #copy last dimension of kernel_size if needed
    # pad = tuple((torch.Tensor(kernel_size).repeat_interleave(2)/2).int().numpy()) #calculate padding needed based in kernel_size
    # ims_unfold = unfoldNd(F.pad(ims, pad, pad_mode), kernel_size=kernel_size) #shape = [N, product(kernel_size), dim1*dim2*dim3]
    # return ims_unfold


def view_unfoldNd(ims, kernel_size=3, pad_mode='circular'):
    #Pads "ims" before unfolding
    #ims.shape = (number of images, number of channels, dim1, dim2, optional dim3)
    #Does the same thing as "my_unfoldNd", but returns a view of the original image (less space, different final shape)
    #ims_padded.storage().data_ptr() == ims_unfold.storage().data_ptr()
    dims = len(ims.shape)-2
    if type(kernel_size)!=list: kernel_size = [kernel_size] #convert to "list" if it isn't
    kernel_size = tuple(kernel_size + [kernel_size[-1]]*(dims-len(kernel_size))) #copy last dimension if needed
    pad = tuple((torch.Tensor(kernel_size).repeat_interleave(2)/2).int().numpy()) #calculate padding needed based on kernel_size
    ims_padded = pad_mixed(ims, pad, pad_mode) #pad "ims" to maintain dimensions after unfolding
    
    ims_unfold = ims_padded.view(ims_padded.shape)
    for i in range(len(kernel_size)):
        ims_unfold = ims_unfold.unfold(2+i,kernel_size[i],1) #end result is (ims.shape, kernel.shape)
        
    return ims_unfold

    
def num_diff_neighbors(ims, window_size=3, pad_mode='circular'): 
    #ims - torch.Tensor of shape [# of images, 1, dim1, dim2, dim3(optional)]
    #window_size - the patch around each pixel that constitutes its neighbors
    #May need to add memory management through batches for large tensors in the future
    
    if type(window_size)==int: window_size = [window_size] #convert to "list" if "int" is given
    window_size = tuple(window_size + [window_size[-1]]*(len(ims.shape)-2-len(window_size))) #copy last dimension of window_size if needed
    pad = tuple((torch.Tensor(window_size).repeat_interleave(2)/2).int().numpy()) #calculate padding needed based in window_size
    ims_unfold = unfoldNd(F.pad(ims, pad, pad_mode), kernel_size=window_size) #shape = [N, product(window_size), dim1*dim2*dim3]
    center_pxl_ind = int(ims_unfold.shape[1]/2)
    ims_diff_unfold = torch.sum(ims_unfold[:,center_pxl_ind,] != ims_unfold.transpose(0,1), dim=0) #shape = [N, dim1*dim2*dim3]
    return ims_diff_unfold.reshape(ims.shape) #reshape to orignal image shape



def num_diff_neighbors_inline(ims_unfold): 
    #ims_unfold - torch tensor of shape = [N, product(kernel_size), dim1*dim2] from [N, 1, dim1, dim2] using "torch.nn.Unfold" object
    #Addtiional dimensions to ims_unfold could be included at the end
    center_pxl_ind = int(ims_unfold.shape[1]/2)
    return torch.sum(ims_unfold[:,center_pxl_ind,] != ims_unfold.transpose(0,1), dim=0) #shape = [N, dim1*dim2]



def run_spparks(size=[512,512], ngrain=512, nsteps=500, freq_dump=1, freq_stat=1, rseed=45684, which_sim='agg', del_files=False):
    '''
    Runs one simulation and returns the file path where the simulation was run
    
    Input:
        rseed: random seed for the simulation (the same rseed and IC will grow the same)
        freq_stat: how many steps between printing stats
        freq_dump: how many steps between recording the structure
        nsteps: number of simulation steps
        dims: square dimension of the structure
        ngrain: number of grains
        which_sim ('agg' or 'eng'): dictates which simulator to use where eng is the latest and allows the use of multiple cores 
        del_files: if True, deletes Miso.txt and Energy.txt. files and allows agg to calculate new files
    Output:
        path_sim
    '''

    # Set and edit local variables
    num_processors = 1 #does not affect agg, agg can only do 1
    dim = len(size)
    path_sim = r"../examples/agg/%sd_sim/"%str(dim)
    path_home = r"../../../PRIMME/"
    path_edit_in = r"./edit_files/spparks_%sd.in"%str(dim)
    path_edit_sh = r"./edit_files/spparks.sh"
    
    # Setup simulation file parameters
    size = size.copy()
    if len(size)==2: size.append(1)
    size[:dim] = (np.array(size[:dim]) - 0.5).tolist()
    if which_sim=='eng': #run agg only once if eng is the simulator we are using (to calculate Miso.txt and Energy.txt files)
        replacement_text_agg_in = [str(rseed), str(ngrain), str(size[0]), str(size[1]), str(size[2]), str(freq_stat), str(freq_dump), str(0), 'agg']
    else: 
        replacement_text_agg_in = [str(rseed), str(ngrain), str(size[0]), str(size[1]), str(size[2]), str(freq_stat), str(freq_dump), str(nsteps), 'agg']
    replacement_text_agg_sh = [str(1), 'agg']
    replacement_text_eng_in = [str(rseed), str(ngrain), str(size[0]), str(size[1]), str(size[2]), str(freq_stat), str(freq_dump), str(nsteps), 'eng']
    replacement_text_eng_sh = [str(num_processors), 'eng']
    
    # Write simulation files 'spparks.in and spparks.sh files'
    replace_tags(path_edit_in, replacement_text_agg_in, path_sim + "agg.in")
    replace_tags(path_edit_sh, replacement_text_agg_sh, path_sim + "agg.sh")
    replace_tags(path_edit_in, replacement_text_eng_in, path_sim + "eng.in")
    replace_tags(path_edit_sh, replacement_text_eng_sh, path_sim + "eng.sh")
    
    # Clean up some files (if they are there)
    if del_files:
        os.system("rm " + path_sim + "Miso.txt") 
        os.system("rm " + path_sim + "Energy.txt") 
    
    # Run simulation
    print("\nRUNNING SIMULATION \n")
    os.chdir(path_sim)
    os.system('chmod +x agg.sh')
    os.system('chmod +x eng.sh')
    os.system('./agg.sh')
    if which_sim=='eng': 
        calc_MisoEnergy(r"./")
        os.system('./eng.sh')
    os.chdir(path_home)
    print("\nSIMULATION COMPLETE \nSIMULATION PATH: %s\n"%path_sim)
    
    return path_sim



def extract_spparks_dump(dim=2):
    '''
    Extracts the information from a spparks.dump file containing euler angles (dump  1 text 1.0 ${fileBase}.dump id site d1 d2 d3)
    Placed information in Numpy variables.
    Works for both 2D and 3D dump files
    
    Parameters
    ----------
    dim : int
        relative path to spparks.dump file
    Returns
    -------
    euler_angle_images : numpy array
        dimensions of [number of images, euler angles in 3 channels, dimx, dimy, dimz (0 for 2D)]
    sim_steps : numpy array
        the monte carlo step for the image of the same index in euler_angle_images
    grain_euler_angles : numpy array
        the euler angles for each grain ID
    grain_ID_images: numpy array
    energy_images: numpy array or site energy
    '''

    sim_steps = []
    euler_angle_images = []
    grain_ID_images = []
    energy_images = []
    num_grains = 0
    path_to_dump = r"../examples/agg/%sd_sim/spparks.dump"%str(dim)
    
    with  open(path_to_dump) as file: 
        print('Loaded')
        for i, l in enumerate(file.readlines()):
            
            t = l.split(' ')
            
            #First time through
            if i == 1:
                sim_step = int(t[-11]) #capture simulation step
                print('Capture sim step: %d'%sim_step)
            elif i == 5: dimx = int(np.ceil(float(t[1]))) #find image dimensions 
            elif i == 6: dimy = int(np.ceil(float(t[1])))
            elif i == 7: 
                dimz = int(np.ceil(float(t[1])))
                num_elements = dimx*dimy*dimz
                image = np.zeros([3, num_elements]) #create image to hold element orientations at this growth step
                ID_image = np.zeros([1, num_elements]) #create image to hold element grain IDs at this growth step
                energy_image = np.zeros([1, num_elements]) #create image to hold element energy at this growth step
                print('Dimensions: [%d, %d, %d]'%(dimx, dimy, dimz))
            
            elif i > 7: 
                [q, r] = np.divmod(i,num_elements+9) #what line are we on in this simulation step
            
                if q==0: #find highest labeled grain
                    if i > 8: 
                        if int(t[1]) > num_grains: num_grains = int(t[1])
                        if i==num_elements+8: 
                            grain_euler_angles = np.zeros([num_grains, 3])
                            print('Number of grains: %d'%num_grains)
                            
                if q==1: #record euler angles for each grain on second pass
                    if r > 8: grain_euler_angles[int(t[1])-1, :] = [float(t[2]), float(t[3]), float(t[4])] 
                
                if r == 0: 
                    image = np.zeros([3, num_elements]) #create image to hold element orientations at this growth step
                    ID_image = np.zeros([1, num_elements]) #create image to hold element grain IDs at this growth step
                    energy_image = np.zeros([1, num_elements]) #create image to hold element energy at this growth step
                elif r == 1:
                    sim_step = int(float(t[-1]))  #capture simulation step
                    print('Capture sim step: %d'%sim_step)
                elif r > 8: 
                    image[:,int(t[0])-1] =  [float(t[2]), float(t[3]), float(t[4])] #record this element's orientation
                    ID_image[:,int(t[0])-1] = [int(t[1])-1] #'-1' to start from 0 instead of 1
                    energy_image[:,int(t[0])-1] = [float(t[5])] #record this element's energy
                
                if r==num_elements+8: #add sim_step and euler_angle_image to the master lists
                    sim_steps.append(sim_step)
                    if dimz==1: 
                        euler_angle_images.append(image.reshape([3, dimy, dimx]).transpose([0,2,1]))
                        grain_ID_images.append(ID_image.reshape([1, dimy, dimx]).transpose([0,2,1]))
                        energy_images.append(energy_image.reshape([1, dimy, dimx]).transpose([0,2,1]))
                    else: 
                        euler_angle_images.append(image.reshape([3, dimz, dimy, dimx]).transpose(0,3,2,1))
                        grain_ID_images.append(ID_image.reshape([1, dimz, dimy, dimx]).transpose(0,3,2,1))
                        energy_images.append(energy_image.reshape([1, dimz, dimy, dimx]).transpose(0,3,2,1))
    
    #Convert to numpy
    sim_steps = np.array(sim_steps)     
    euler_angle_images = np.array(euler_angle_images)  
    grain_ID_images = np.array(grain_ID_images)    
    energy_images = np.array(energy_images)  
    
    return euler_angle_images, sim_steps, grain_euler_angles, grain_ID_images, energy_images



def dump_to_hdf5(path_to_dump="Circle_512by512.dump", path_to_hdf5="Circle_512by512.hdf5", num_steps=None):
    #A more general purpose extract dump file - reads lines directly to an hdf5 file and saves header names
    #The lines can then be extracted one-by-one from the hdf5 file and converted to an image
    #"num_steps" is a guess at how many entries there are in the dump file to report how long it will take
    
    with open(path_to_dump) as file:
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
    with h5py.File(path_to_hdf5, 'w') as f:
        f["bounds"] = bounds #metadata
        f["variable_names"] = [x.encode() for x in var_names] #metadata
        dset = f.create_dataset("dump_extract", shape=(1,num_atoms,len(var_names)), maxshape=(None,num_atoms,len(var_names)))#, chunks=True)
        with open(path_to_dump) as file:
            for i, line in tqdm(enumerate(file), "EXTRACTING SPPARKS DUMP (%s)"%path_to_dump, total=total_lines):
                [entry_num, line_num] = np.divmod(i,entry_length) #what entry number and entry line number does this line number indicate
                if line_num==0: entry = np.zeros([num_atoms, len(var_names)]) #reset the entry values at the beginning of each entry
                if line_num==1: time_steps.append(int(line.split()[-1])) #log the time step
                atom_num = line_num-9 #track which atom line we're on
                if atom_num>=0 and atom_num<num_atoms: entry[atom_num,] = np.array(line.split(), dtype=float) #record valid atom lines
                if line_num==entry_length-1: 
                    dset[-1,:,:] = entry #save this entry before going to the next
                    dset.resize(dset.shape[0]+1, axis=0) #make more room in the hdf5 dataset
        dset.resize(dset.shape[0]-1, axis=0) #remove the extra room that wasn't used
        time_steps = np.array(time_steps) #reformat time_steps
        f["time_steps"] = time_steps #metadata
        
    return var_names, time_steps, bounds



def dump_extract_to_images(path_to_hdf5="Hex_443by512.hdf5", new_path="Hex_443by512_ims.hdf5", vi=1, xi=2, yi=3):
    #Convert hdf5 dataset "dump_extract" to "images" in a new hdf5 file
    #This function is intended to be used to process the raw extracted data from a SPPARKS dump (e.g. extract images)
    #"vi" is the index in the third dimension of "f[("dump_extract")]" that corresponds to the image pixels 
    #"xi" and "yi" are the indices in the third dimension of "f[("dump_extract")]" that correspond position of each pixel value

    with h5py.File(path_to_hdf5, 'r') as f:
        b = f[("bounds")]
        e = f[("dump_extract")]
        s = e.shape
        ii = (e[0,:,xi]*b[1,1]+e[0,:,yi]).astype(int) #calculate indicies for remapping the pixels
        
        #Find the smallest data type that can be used without overflowing
        m = np.max(e[0,:,vi])
        tmp = np.array([8,16,32], dtype='uint64')
        dtype = 'uint' + str(tmp[np.sum(m>2**tmp)])
        
        with h5py.File(new_path, 'w') as ff: #open new hdf5 to write to
            ims = ff.create_dataset("images", shape=(tuple([s[0]])+tuple(b[0:2,1].tolist())), dtype=dtype)
            
            for i in tqdm(range(s[0]), "EXTRACT ID IMAGES FROM HDF5 FILE (%s)"%path_to_hdf5):
                ee = np.zeros([s[1]])
                ee[ii] = e[i,:,vi]
                ims[i,] = ee.reshape(b[0:2,1])
        
            plt.imshow(ims[0,]) #show the first image as a sample
    
    # instructions to work with hdf5 file
    # f = h5py.File(path_to_hdf5+".hdf5", 'r')
    # ims =f[('images')]
    # #do processing on "ims" here
    # f.close()
    
    # with h5py.File(path_to_hdf5+".hdf5", 'a') as f:
    #     ims =f[('images')]
    #     #do processing on "ims" here

    



def hdf5_to_npy_ims(path_to_hdf5="Hex_443by512", path_to_npy="Hex_443by512", vi=1, xi=2, yi=3):
    #Convert hdf5 (which was in turn converted from a dump file using "dump_to_hdf5") to a npy file 
    #This functions is intended to be used to process the raw extracted data from a SPPARKS dump (e.g. extract images)
    #"vi" is the index in the third dimension of "f[("values")]" that corresponds to the image pixels 
    #"xi" and "yi" are the indices in the third dimension of "f[("values")]" that correspond position of each pixel value

    ims_list = []
    with h5py.File(path_to_hdf5+".hdf5", 'r') as f:
        b = f[("bounds")]
        v = f[("values")]
        s = v.shape
        ii = (v[0,:,xi]*b[1,1]+v[0,:,yi]).astype(int)
        
        for i in tqdm(range(s[0]), "EXTRACT ID IMAGES FROM HDF5 FILE (%s.hdf5)"%path_to_hdf5):
            vv = np.zeros([s[1]])
            vv[ii] = v[i,:,vi]
            ims_list.append(vv.reshape(b[0:2,1]))
    
    plt.imshow(ims_list[0])
    ims = np.array(ims_list)   
    np.save("%s.npy"%path_to_npy, ims.astype(int)) 
    return ims



def str_np_code(var_in):
    '''
    Encodes and decodes strings to ascii representation in a numpy array to store in HDF5 datatsets 
    '''
    if type(var_in) == str: 
        var_out = np.asarray([ord(c) for c in var_in])
    else: #assume it is the numpy encoding
        var_out = ''
        for c in var_in: var_out=var_out+chr(c)
    return var_out



def capture_sim(hf_name = "test", dim = 2):
    '''
    Captures predetermined data from either the preset 2D or 3D simulation folders and places it in the indicated hdf5 file in a predetermined structure and data folder. 

    Parameters
    ----------
    hf_name : string, optional
        name of the hdf5 file to which data to this simulation will be saved. The default is "test".
    dim : int, optional
        2 or 3. set the simulation folder reference to the 2D or 3D simulation folder. The default is 2.

    Returns
    -------
    first_image : numpy
        The first image from euler_angle_images.
    last_image : numpy
        The last image from euler_angle_images.
    '''

    # Local variables
    path_to_sim = r"../examples/agg/%sd_sim/"%str(dim)
    print("\nCAPTURING A %sD SIMULATION\n"%str(dim))
    path_to_data = r"../../GrainGrowth/data/"
      
    # Capture simulation data  
    euler_angle_images, sim_steps, grain_euler_angles, grain_ID_images, energy_images = extract_spparks_dump(dim)
    with open(path_to_sim + "spparks.cluster","r") as f: clustertxt = f.read()
    with open(path_to_sim + "PolyIC.init","r") as f: inittxt = f.read()
    with open(path_to_sim + "Miso.txt","r") as f: misotxt = f.read()
    with open(path_to_sim + "Energy.txt","r") as f: energytxt = f.read()
    
    # Store simulation data
    hf = h5py.File(path_to_data + hf_name + ".hdf5", "a") 
    try: 
        sim_name = "sim%d"%(len(hf .keys())+1) #plus one to however many simulations are already saved in this file
        print("\nSIMULATION ADDED TO FILE: " + sim_name)
        
        hf[sim_name + "/euler_angle_images"] = euler_angle_images
        hf[sim_name + "/sim_steps"] = sim_steps
        hf[sim_name + "/grain_euler_angles"] = grain_euler_angles
        hf[sim_name + "/grain_ID_images"] = grain_ID_images
        hf[sim_name + "/energy_images"] = energy_images
        hf[sim_name + "/clustertxt"] = str_np_code(clustertxt) #encode strings as ascii in a numpy
        hf[sim_name + "/inittxt"] = str_np_code(inittxt)
        hf[sim_name + "/misotxt"] = str_np_code(misotxt)
        hf[sim_name + "/energytxt"] = str_np_code(energytxt)
        
        hf.close()
    except ValueError:
        print(ValueError)
        hf.close()
    
    print("\nCAPTURE COMPLETE\n")
    
    # Return first and last image for troubleshooting
    #first_image = euler_angle_images[0]
    #last_image = euler_angle_images[-1]
    #return first_image, last_image
    
    
    
def capture_sim2(hf_name = "test", dim = 2):
    '''
    Same as capture_sim but only captures the euler angles
    '''

    # Local variables
    print("\nCAPTURING A %sD SIMULATION\n"%str(dim))
    path_to_data = r"../../GrainGrowth/data/"
    path_to_sim = r"../examples/agg/%sd_sim/"%str(dim)
      
    # Capture simulation data  
    euler_angle_images, sim_steps, grain_euler_angles, grain_ID_images, energy_images = extract_spparks_dump(dim)
    
    # Store simulation data
    hf = h5py.File(path_to_data + hf_name + ".hdf5", "a") 
    try: 
        sim_name = "sim%d"%(len(hf .keys())+1) #plus one to however many simulations are already saved in this file
        #print("\nSIMULATION ADDED TO FILE: " + sim_name)
        
        hf[sim_name + "/euler_angle_images"] = euler_angle_images
        hf[sim_name + "/energy_images"] = energy_images
        
        with open(path_to_sim + "spparks.cluster","r") as f: clustertxt = f.read()
        hf[sim_name + "/clustertxt"] = str_np_code(clustertxt) #encode strings as ascii in a numpy
        
        hf.close()
    except ValueError:
        print(ValueError)
        hf.close()
    
    #print("\nCAPTURE COMPLETE\n")



def extract_sim(path_to_data=r"../../GrainGrowth/data/", hf_name=None, sim_name=None):
    '''
    Interactive method for extracting data from HDF5 files created using capture_sim()
    '''
    
    # Find the desired file
    if hf_name==None: #then ask for user input
        _, _, lst_files = next(os.walk(path_to_data))
        for i in range(len(lst_files)): print('%d: %s'%(i, lst_files[i]))
        tmp = input('Which data file do you want?   (input integer): ')
        hf_name = lst_files[int(tmp)]
    
    # Open the file and find the desired simulaiton
    hf = h5py.File(path_to_data + hf_name, "r") 
    try:
        if sim_name==None:
            lst_sims = list(hf.keys())
            for i in range(len(lst_sims)): print('%d: %s'%(i, lst_sims[i]))
            tmp = input('Which simulation do you want?   (input integer): ')
            sim_name = lst_sims[int(tmp)]
        
        # Extract the data and close the file
        euler_angle_images = hf[sim_name + "/euler_angle_images"][:]
        sim_steps = hf[sim_name + "/sim_steps"][:]
        grain_euler_angles = hf[sim_name + "/grain_euler_angles"][:]
        grain_ID_images = hf[sim_name + "/grain_ID_images"][:]
        energy_images = hf[sim_name + "/energy_images"][:]
        clustertxt = str_np_code(hf[sim_name + "/clustertxt"][:]) # decode strings from numpy
        inittxt = str_np_code(hf[sim_name + "/inittxt"][:])
        misotxt = str_np_code(hf[sim_name + "/misotxt"][:])
        energytxt = str_np_code(hf[sim_name + "/energytxt"][:])
        
        hf.close()
    except ValueError:
        print(ValueError)
        hf.close()
    
    return euler_angle_images, sim_steps, grain_euler_angles, grain_ID_images, energy_images, clustertxt, inittxt, misotxt, energytxt



def extract_var(path_to_data=r"../../GrainGrowth/data/", hf_name=None, var_name=None):
    '''
    Pull a specific variable from all the sims in a specified file and return in a list   
    '''
    
    var_list = []
    
    # Find the desired file
    if hf_name==None: #then ask for user input
        _, _, lst_files = next(os.walk(path_to_data))
        for i in range(len(lst_files)): print('%d: %s'%(i, lst_files[i]))
        tmp = input('Which data file do you want?   (input integer): ')
        hf_name = lst_files[int(tmp)]
    
    # Open the file and find the desired simulaiton
    hf = h5py.File(path_to_data + hf_name, "r") 
    try:
        if var_name==None:
            lst_vars = list(hf['sim1'].keys())
            for i in range(len(lst_vars)): print('%d: %s'%(i, lst_vars[i]))
            tmp = input('Which variable do you want?   (input integer): ')
            var_name = lst_vars[int(tmp)]
        
        for sim_name in list(hf.keys()):
            if 'txt' in var_name:
                var_list.append(str_np_code(hf[sim_name + "/%s"%var_name][:]))
            else:
                var_list.append(hf[sim_name + "/%s"%var_name][:]) 
        
        hf.close()
    except ValueError:
        print(ValueError)
        hf.close()    

    return var_list



def analyze_clusters(clustertxt, nclusters=None, if_plot=False):
    '''
    Extract descriptions about microstructure clusters (grains/spins) from the spparks.cluster file
    Input:
        clustertxt (String): spparks.cluster file loaded into a string 
    '''

    #Split lines
    clustertxt_lines = clustertxt.splitlines()
    
    #Find the number of timesteps and starting number of clusters
    time_steps = 0
    for i, l in enumerate(clustertxt_lines):
        temp = l.split(' ')
        if i == 7: 
            if nclusters==None: nclusters = int(temp[2])
        if '--------------------------------------------------' in l: time_steps +=1
    
    # Setup variables
    sim_step = [] #monte carlo time
    ncluster = [] #number of clusters
    avg_size = [] #average cluster size in voxels
    avg_radius = [] #average cluster radius in voxels
    cluster_size = np.zeros((time_steps, nclusters)) #all cluster sizes in voxels
    
    #Extract variables 
    loc = 0 #Track how far into each growth time step we are
    for i, l in enumerate(clustertxt_lines):
        temp = l.split(' ')
        
        if '--------------------------------------------------' in temp[0]: loc = 0
        elif 'Time' in temp[0]: sim_step.append(int(float(temp[2])))
        elif 'ncluster' in temp[0]: ncluster.append(int(temp[2]))
        elif '<N>' in temp[0]: avg_size.append(float(temp[2]))
        elif '<R>' in temp[0]: avg_radius.append(float(temp[2]))
        elif loc>=6 and len(temp)==8: 
            cluster_size[len(sim_step)-1, int(temp[2])-1] = int(temp[4])
            
        loc += 1
    
    #Calculate cluster variance
    cluster_var = [] #cluster size variance
    for i, a in enumerate(sim_step):
        cluster_var.append(np.var(cluster_size[i, cluster_size[i,:] != 0]))
    
    # Plot all variables as a function of sim_step
    if if_plot: 
        plt.subplot(5,1,1); plt.plot(sim_step,ncluster); plt.title('Number of Clusters'); plt.ylabel('Number of Clusters');
        plt.subplot(5,1,2); plt.plot(sim_step,avg_size); plt.title('Average Cluster Size'); plt.ylabel('Voxels');
        plt.subplot(5,1,3);plt.plot(sim_step,avg_radius); plt.title('Average Cluster Radius'); plt.ylabel('Voxels');
        plt.subplot(5,1,4);plt.plot(sim_step,cluster_var); plt.title('Cluster Size Variance'); plt.ylabel('Variance');
        plt.subplot(5,1,5);plt.plot(sim_step, cluster_size); plt.title('Cluster Sizes'); plt.xlabel('Time'); plt.ylabel('Voxels');
    
    sim_step, ncluster, avg_size, avg_radius, cluster_var = np.asarray([sim_step, ncluster, avg_size, avg_radius, cluster_var])
    
    return sim_step, ncluster, avg_size, avg_radius, cluster_var, cluster_size



def extract_cluster_var(clustertxt_list, var_name='cluster_size', nclusters=None):
    '''
    Extract a specific variable ('var_name') from each 'clustertxt' in 'clustertxt_list'
    '''
    if type(clustertxt_list) is not list: 
        sim_step, ncluster, avg_size, avg_radius, cluster_var, cluster_size = analyze_clusters(clustertxt_list, nclusters=nclusters, if_plot=False)
        return eval(var_name)
    else:
        var_list = []
        for clustertxt in clustertxt_list:
            sim_step, ncluster, avg_size, avg_radius, cluster_var, cluster_size = analyze_clusters(clustertxt, nclusters=nclusters, if_plot=False)
            var_list.append(eval(var_name))
        return var_list



def get_line(i,j): 
    '''
    i and j are counting numbers, 1 and higher, that reference items (such as grains) - i and j cannot be the same
    Given a pair matrix where index [1,2] compares item 1 with item 2 (misorientation, Energy, distance...)
    If you compress that down into an array that does not have redundant information
    This function tells what line in the array each pair value would be found (not counting numbers)
    In other words, this function finds the line numbers for 'pairMat2array'
    '''
    if i>j: #i must be bigger than j
        i_old = i
        i = j
        j = i_old
    ln = int(i-1+(j-2)*(j-1)/2)
    return ln



def get_line_elem(i, j, h, size):
    '''
    Find the line in a numpy array that have been flattened from the original indicies
    i, j, h are indices of the matrix before flattening
    size is a dimenion of the original square or cube matrix
    '''
    if h is not None: ln = h+(j+i*size)*size
    else: ln = h+(j+i*size)*size
    
    return ln



def pairMat2txt(pair_mat = np.zeros([64,64]), fp_out = '../examples/agg/2d_sim/Miso.txt'):
    '''
    Input a pair matrix such as garin-to-grain misorientation or energy (symmetric and diagonal is zero)
    Converts to an array that does not include redundant data (diagonal or symmetric data)
    Uses 'get_line' to place the data in the array (keeps top triangle and appends one entry at a time)
    Saves to 'fp_out'
    Can be used to create 'Miso.txt' or 'Energy.txt' files
    '''
    
    # Create array of correct length
    num_grains = pair_mat.shape[0]
    temp_ls = [0]*int((num_grains**2-num_grains)/2) 
    
    # Transfer entries from the metrix to the array
    for i in range(num_grains):
        for j in range(num_grains):
            if i>j:
                loc = get_line(i+1,j+1)
                temp_ls[loc] = str(pair_mat[i,j]) + '\n'
    
    with open(fp_out, 'w') as file:
        file.writelines(temp_ls)
        
        
        
def txt2PairMat(txt_path=r"../examples/agg/2d_sim/Miso.txt", ngrain=128):
    with open(txt_path,"r") as f: txt = f.read()
    arr = np.asarray(txt.split('\n')[0:-1], dtype='float')
    pair_mat = np.zeros((ngrain, ngrain))
    for i, j in np.ndindex(pair_mat.shape): 
        if i>j:
            m = arr[get_line(i+1,j+1)]
            pair_mat[i,j] = m
            pair_mat[j,i] = m
    return pair_mat



def elemMat2txt(elem_mat = np.zeros([64,64]), fp_out = '../examples/agg/2d_sim/SiteEnergy.txt'):
    '''
    Flatten the input matrix and save to a txt file for use with spparks simulations
    Can be used for 'EnergyElem.txt' the contains the energy at every pixel in the structure

    '''
    elem_mat = elem_mat.flatten() #flattens in row order (in order of dimensions)
    temp_ls = [0]*elem_mat.size
    
    # Transfer entries from the metrix to the array
    for i in range(elem_mat.size):
        temp_ls[i] = str(elem_mat[i]) + '\n'
    
    with open(fp_out, 'w') as file:
        file.writelines(temp_ls)



def plot_spk(im):
    '''
    im - 1D array or 2d array image or 3d or more [channels, xdim, ydim, ...more...]
    '''
    #Remove any dimensions of 1
    im = np.squeeze(im)
    
    if im.ndim==1: 
        plt.plot(im)
    elif im.ndim==2: 
        im = (im-np.min(im))/np.max(im-np.min(im)) #normalize image to 1
        plt.imshow(im, interpolation='none'); plt.axis('off') 
    else:
        #Cut out everything beyond 3 channels
        if im.shape[0]>=3: im = im[0:3,] 
    
        #Cut all dimensions beyond 3 in half
        if im.ndim>3: 
            bi = tuple([int(x/2) for x in im.shape[3:]])
            im = im[:,:,:,bi]
            for _ in range(3,im.ndim): im = np.squeeze(im, axis=3)
            
        im = (im-np.min(im))/np.max(im-np.min(im)) #normalize image to 1
        im = np.transpose(im, axes=[1,2,0])
        plt.imshow(im, interpolation='none'); plt.axis('off') 
        
    

def set_abnormal_energy(grain_IDs=None, energy_ratio=0.7, num_grains=128, num_abnormal=16, dim=2, use_miso=False):
    '''
    Write an Energy.txt file (to be read by spparks) that gives lower energy to the grain_IDs by a factor of energy_ratio
    The grains with lower energy will growth abnormally large
    If no grains_IDs are provided, num_abnormal grains are randomly picked
    An energy_ratio of 0.7 works well
    If use_miso=True, the current Miso.txt file (run a single step of the simulation using "del_files = True" to ensure this file is created for the current scenario) is used as a basis for creating Energy.txt, before it is set to zero
    When using this function, "run_spparks" should be set to "del_files = False" to ensure the Energy.txt is not deleted before the run, making the growth normal again
    '''
    
    # Set which grains to be abnormal, randomly
    if grain_IDs is None: 
        grain_IDs = np.random.choice(np.arange(0, num_grains), size=num_abnormal, replace=False)
    
    # Set all the energy pairs to 1, then lower the select grain_IDs to the energy_ratio
    pair_mat = np.ones([num_grains,num_grains]) 
    if use_miso:
        pair_mat = txt2PairMat(txt_path=r"../examples/agg/%dd_sim/Miso.txt"%dim, ngrain=num_grains)
    pair_mat[grain_IDs,:] *= energy_ratio
    pair_mat[:,grain_IDs] *= energy_ratio
    pairMat2txt(pair_mat=pair_mat, fp_out = '../examples/agg/%dd_sim/Energy.txt'%dim)
    
    # Set the misorientation matrix to zero
    pair_mat = np.zeros([128,128])
    pairMat2txt(pair_mat=pair_mat, fp_out = '../examples/agg/%dd_sim/Miso.txt'%dim)
    
    return grain_IDs



# def generate_grain_centers_txt(size=[64,32,128], ngrain=64, fp="grains.txt"):
#     #Generate a "grains.txt" of grain center locations for use in MOOSE simulations (https://github.com/idaholab/moose/blob/next/modules/phase_field/test/tests/initial_conditions/grains.txt)
#     #The file is written to the current directory and can be used for 2D or 3D "size" inputs
    
#     #For uniform sampled centers
#     center_coords = np.random.rand(ngrain, len(size))*np.array(size)
    
#     #For Poisson Disk sampled centers
#     # size_n = size/np.max(size)
#     # r = np.sqrt(np.product(size_n)*1.93/np.pi/ngrain)
#     # center_coords = Bridson_sampling(dims=size_n, radius=r, k=30)*np.max(size)
#     print("Number centers: %d"%center_coords.shape[0])
    
#     if len(size)==2: header = "x y\n"
#     else: header = "x y z\n"

#     np.savetxt(fp, center_coords, delimiter=' ', fmt='%.5f')
    
#     with open(fp, 'r+') as f:
#             content = f.read()
#             f.seek(0, 0)
#             f.write(header + content)
            
#     return center_coords





#From: https://github.com/diregoblin/poisson_disc_sampling/blob/main/poisson_disc.py
####################################################################################
def hypersphere_volume_sample(center,radius,k=1):
    # Uniform sampling in a hyperspere
    # Based on Matlab implementation by Roger Stafford
    # Can be optimized for Bridson algorithm by excluding all points within the r/2 sphere
    ndim = center.size
    x = np.random.normal(size=(k, ndim))
    ssq = np.sum(x**2,axis=1)
    fr = radius*gammainc(ndim/2,ssq/2)**(1/ndim)/np.sqrt(ssq)
    frtiled = np.tile(fr.reshape(k,1),(1,ndim))
    p = center + np.multiply(x,frtiled)
    return p


def hypersphere_surface_sample(center,radius,k=1):
    # Uniform sampling on the sphere's surface
    ndim = center.size
    vec = np.random.standard_normal(size=(k, ndim))
    vec /= np.linalg.norm(vec, axis=1)[:,None]
    p = center + np.multiply(vec, radius)
    return p


def squared_distance(p0, p1):
    return np.sum(np.square(p0-p1))


def Bridson_sampling(dims=np.array([1.0,1.0]), radius=0.05, k=30, hypersphere_sample=hypersphere_volume_sample):
    # References: Fast Poisson Disk Sampling in Arbitrary Dimensions
    #             Robert Bridson, SIGGRAPH, 2007

    ndim=len(dims)

    # size of the sphere from which the samples are drawn relative to the size of a disc (radius)
    sample_factor = 2
    if hypersphere_sample == hypersphere_volume_sample:
        sample_factor = 2
        
    # for the surface sampler, all new points are almost exactly 1 radius away from at least one existing sample
    # eps to avoid rejection
    if hypersphere_sample == hypersphere_surface_sample:
        eps = 0.001
        sample_factor = 1 + eps
    
    def in_limits(p):
        return np.all(np.zeros(ndim) <= p) and np.all(p < dims)

    # Check if there are samples closer than "squared_radius" to the candidate "p"
    def in_neighborhood(p, n=2):
        indices = (p / cellsize).astype(int)
        indmin = np.maximum(indices - n, np.zeros(ndim, dtype=int))
        indmax = np.minimum(indices + n + 1, gridsize)
        
        # Check if the center cell is empty
        if not np.isnan(P[tuple(indices)][0]):
            return True
        a = []
        for i in range(ndim):
            a.append(slice(indmin[i], indmax[i]))
        with np.errstate(invalid='ignore'):
            if np.any(np.sum(np.square(p - P[tuple(a)]), axis=ndim) < squared_radius):
                return True

    def add_point(p):
        points.append(p)
        indices = (p/cellsize).astype(int)
        P[tuple(indices)] = p

    cellsize = radius/np.sqrt(ndim)
    gridsize = (np.ceil(dims/cellsize)).astype(int)

    # Squared radius because we'll compare squared distance
    squared_radius = radius*radius

    # Positions of cells
    P = np.empty(np.append(gridsize, ndim), dtype=np.float32) #n-dim value for each grid cell
    # Initialise empty cells with NaNs
    P.fill(np.nan)

    points = []
    add_point(np.random.uniform(np.zeros(ndim), dims))
    while len(points):
        i = np.random.randint(len(points))
        p = points[i]
        del points[i]
        Q = hypersphere_sample(np.array(p), radius * sample_factor, k)
        for q in Q:
            if in_limits(q) and not in_neighborhood(q):
                add_point(q)
    return P[~np.isnan(P).any(axis=ndim)]

####################################################################################



def cumsum_sample(arrays):
    #"array" - shape=(number of arrays, array elements)
    #Chooses an index from each row in "array" by sampling from it's cumsum
    arrays_cumsum = torch.cumsum(arrays, dim=1)/torch.sum(arrays, dim=1).unsqueeze(1)
    sample_values = torch.rand(arrays_cumsum.shape[0]).to(arrays.device)
    sample_indices = torch.argmax((arrays_cumsum>sample_values.unsqueeze(1)).float(), dim=1)
    return sample_indices

def rand_argmax(arrays):
    #"array" - shape=(number of arrays, array elements)
    #Chooses an index from each row in "array" that is the max value or a random max value indicie if there are multiples
    arrays_max, _ = torch.max(arrays, dim=1)
    arrays_marked = arrays==arrays_max.unsqueeze(1)
    samples_indices = cumsum_sample(arrays_marked)
    return samples_indices

# def count_occurance(arrays): #too much memory
#     #Counts the number of occuances of each value in the array and replaces the value with that count
#     #Operates on the first dimension
#     num_samples = arrays.shape[0]
#     comparison = arrays.reshape(1,num_samples,-1)==arrays.reshape(num_samples,1,-1) #boolean
#     counts = torch.sum(comparison, dim=1) #each value replaced with the count
#     return counts

def count_occurance(arrays):
    #Counts the number of occuances of each value in the array and replaces the value with that count
    #Operates on the first dimension
    l = []
    for i in range(arrays.shape[0]):
        l.append(torch.sum(arrays[i:i+1]==arrays, dim=0))
    counts = torch.stack(l)
    return counts


def rand_mode(arrays):
    #Takes the mode of the array using torch.Tensor.cuda
    counts = count_occurance(arrays) #counts the number of occurances for each value
    index = rand_argmax(counts.transpose(1,0)).reshape(1,-1) #finds the index for the max value (choosing randomly when there are ties)
    # index = cumsum_sample(counts.transpose(1,0)).reshape(1,-1) #use this if you want to sample from the counts instead of choosing the max
    rand_mode = torch.gather(arrays, dim=0, index=index)[0] #selects those indices
    return rand_mode


# def rand_mode2(arrays): #not actually rand_mode, just a different way to do mode
#     #Takes the mode of the array using torch.Tensor.cuda
#     arrays_mode = torch.mode(arrays.cpu(), dim=0).values
    
#     arrays_marked = arrays==arrays_mode.unsqueeze(0).to(arrays.device)
#     index = cumsum_sample(arrays_marked.transpose(1,0)).unsqueeze(0)
#     rand_mode = torch.gather(arrays, dim=0, index=index)[0] #select those indices
#     return rand_mode






def create_SPPARKS_dataset(fp, size=[512,512], ngrains_range=[256, 512], nsets=1000, future_steps=4, max_steps=20, offset_steps=1):
    
    # DETERMINE THE SMALLEST POSSIBLE DATA TYPE POSSIBLE
    m = np.max(ngrains_range)
    tmp = np.array([8,16,32], dtype='uint64')
    dtype = 'uint' + str(tmp[np.sum(m>2**tmp)])

    with h5py.File(fp, 'w') as f:
        dset = f.create_dataset("dataset", shape=(1, future_steps+1, 1, size[0], size[1]) , maxshape=(None, future_steps+1, 1, size[0], size[1]), dtype=dtype)
        for _ in tqdm(range(nsets)):
            
            # SET PARAMETERS
            ngrains = np.random.randint(ngrains_range[0], ngrains_range[1]+1) #number of grains
            nsteps = np.random.randint(offset_steps+future_steps, max_steps) #SPPARKS steps to run
            freq_dump = 1 #how offten to dump an image (record)
            freq_stat = 1 #how often to report stats on the simulation
            rseed = 4951 #change to get different growth from teh same initial condition
            
            # RUN SIMULATION
            img, EulerAngles, center_coords0 = voronoi2image(size, ngrains) #generate initial condition
            image2init(img, EulerAngles) #write initial condition
            _ = run_spparks(size, ngrains, nsteps, freq_dump, freq_stat, rseed) #run simulation
            _, _, _, grain_ID_images, _ = extract_spparks_dump(dim=len(size)) #extract SPPARKS dump data to python
                
            # WRITE TO THE DATASET
            dset[-1,:,:] = grain_ID_images[-(future_steps+1):,] 
            dset.resize(dset.shape[0]+1, axis=0) 
        dset.resize(dset.shape[0]-1, axis=0) #Remove the final dimension that has nothing written to it



def compute_action_energy_change(im, im_next, obs_dim=9, act_dim=9):
    #Calculate the energy change introduced by actions in each "im" action window
    #Energy is calculated as the number of different neighbors for each observation window
    #Find the current energy at each site in "im" observational windows
    #Finds the energy of "im_next" using observational windows with center pixels replaced with possible actions
    #The difference is the energy change
    #FUTURE WORK -> If I change how the num-neighbors function works, I could probably use expand instead of repeat
    
    # SETUP UNFOLD PARAMETERS
    pad_mode = "reflect"
    pad_obs = tuple([int(np.floor(obs_dim/2))]*4)
    pad_act = tuple([int(np.floor(act_dim/2))]*4)
    unfold_obs = torch.nn.Unfold(kernel_size=obs_dim)
    unfold_act = torch.nn.Unfold(kernel_size=act_dim)
    
    # CALCULATE CURRENT ENERGY
    windows_curr_obs = unfold_obs(F.pad(im, pad_obs, pad_mode))
    current_energy = num_diff_neighbors_inline(windows_curr_obs)
    
    # CACLULATE ENERGY AFTER TAKING EACH POSSIBLE ACTION AT EACH LOCATION
    windows_curr_act = unfold_act(F.pad(im, pad_act, pad_mode))
    windows_next_obs = unfold_obs(F.pad(im_next, pad_obs, pad_mode))
    windows_next_obs_ex = windows_next_obs.unsqueeze(dim=1).repeat(1, act_dim*act_dim, 1, 1) #copy the matrix for the number of actions needed
    windows_next_obs_ex[:,:,int(obs_dim**2/2),:] = windows_curr_act #place one action into each matrix copy
    windows_next_obs_ex = windows_next_obs_ex.unsqueeze(4).transpose(0,4)[0] #reshape for the below function
    action_energy = num_diff_neighbors_inline(windows_next_obs_ex) #find the energy for each copy, each having a different action
    
    # CALCULATE ENERGY CHANGE, SCALED BY THE NUMBER OF TOTAL NUMBER OF OBSERVATIONAL NEIGHBORS
    energy_change = (current_energy.transpose(0,1)-action_energy)/(obs_dim**2-1)
    return energy_change



def compute_energy_labels(im_seq, obs_dim=9, act_dim=9):
    #Compute the action energy change between the each image and the one immediately following
    #MAYBE CHANGE IT TO THIS IN THE FUTURE -> Compute the action energy change between the first image and all following
    #The total energy label is a decay sum of those action energy changes
    
    # CALCULATE ALL THE ACTION ENERGY CHANGES
    size = im_seq.shape[1:]
    energy_changes = []
    for i in range(im_seq.shape[0]-1):
        ims_curr = im_seq[i].reshape(1,1,im_seq.shape[2],im_seq.shape[3])
        ims_next = im_seq[i+1].reshape(1,1,im_seq.shape[2],im_seq.shape[3])
        energy_change = compute_action_energy_change(ims_curr, ims_next, obs_dim=obs_dim, act_dim=act_dim)
        energy_changes.append(energy_change)
    
    # COMBINE THEM USING A DECAY SUM
    energy_change = torch.cat(energy_changes, dim=2)
    decay_rate = 1/2
    decay = decay_rate**torch.arange(1,im_seq.shape[0]).reshape(1,1,-1).to(im_seq.device)
    energy_labels = torch.sum(energy_change*decay, dim=2).transpose(0,1).reshape(np.product(size), act_dim, act_dim)
    
    return energy_labels


def compute_energy_labels2(im_seq, obs_dim=9, act_dim=9):
    #Compute the action energy change between the each image and the one immediately following
    #MAYBE CHANGE IT TO THIS IN THE FUTURE -> Compute the action energy change between the first image and all following
    #The total energy label is a decay sum of those action energy changes
    
    
    
    #expand all of the futre images with an observation window
    
    
    
    
    # CALCULATE ALL THE ACTION ENERGY CHANGES
    size = im_seq.shape[1:]
    energy_changes = []
    for i in range(im_seq.shape[0]-1):
        ims_curr = im_seq[i].reshape(1,1,im_seq.shape[2],im_seq.shape[3])
        ims_next = im_seq[i+1].reshape(1,1,im_seq.shape[2],im_seq.shape[3])
        energy_change = compute_action_energy_change(ims_curr, ims_next, obs_dim=obs_dim, act_dim=act_dim)
        energy_changes.append(energy_change)
    
    # COMBINE THEM USING A DECAY SUM
    energy_change = torch.cat(energy_changes, dim=2)
    decay_rate = 1/2
    decay = decay_rate**torch.arange(1,im_seq.shape[0]).reshape(1,1,-1).to(im_seq.device)
    energy_labels = torch.sum(energy_change*decay, dim=2).transpose(0,1).reshape(np.product(size), act_dim, act_dim)
    
    return energy_labels



def compute_action_labels(im_seq, act_dim=9):
    #Label which actions in each action window were actually taken between the first image and all following
    #The total energy label is a decay sum of those action labels
    
    size = im_seq.shape[1:]
    im = im_seq[0:1,]
    ims_next = im_seq[1:]
    
    # SETUP UNFOLD PARAMETERS
    pad_mode = "reflect"
    pad_act = tuple([int(np.floor(act_dim/2))]*4)
    unfold_act = torch.nn.Unfold(kernel_size=act_dim)
    
    # CALCULATE ACTION LABELS
    window_act = unfold_act(F.pad(im, pad_act, pad_mode))[0] #Unfold to get windows of possible actions for each site
    ims_next_flat = ims_next.view(ims_next.shape[0], -1)
    
    actions_marked = window_act.unsqueeze(0).expand(4,-1,-1)==ims_next_flat.unsqueeze(1) #Mark the actions that matches each future image (the "action taken")
    decay_rate = 1/2
    decay = decay_rate**torch.arange(1,im_seq.shape[0]).reshape(-1,1,1).to(im.device)
    action_labels = torch.sum(actions_marked*decay, dim=0).transpose(0,1).reshape(np.product(size), act_dim, act_dim)
    
    return action_labels



def compute_labels(im_seq, obs_dim=9, act_dim=9):
    energy_labels = compute_energy_labels(im_seq, obs_dim=obs_dim, act_dim=act_dim)
    action_labels = compute_action_labels(im_seq, act_dim=act_dim)
    labels = action_labels + energy_labels
    return labels



def compute_features(im, obs_dim=9):
    size = im.shape[1:]
    local_energy = num_diff_neighbors(im, window_size=7, pad_mode='reflect')
    features = my_unfoldNd(local_energy.float(), obs_dim).T.reshape(np.product(size),obs_dim,obs_dim)
    return features



# class CustomImageDataset(Dataset): #way too much overhead, way to slow
#     def __init__(self, ims, dim=0):
#         self.ims = ims
#         self.dim = dim
#         self.device = ims.device

#     def __len__(self):
#         return self.ims.shape[self.dim]

#     def __getitem__(self, i):
#         im = self.ims.select(self.dim, i)
#         return im

# def iter_data(arrays, dim=0, batch_size=64, shuffle=False): #way too much overhead, way to slow
#     #Creates and iterator over the first dimension of "arrays"
#     dset = CustomImageDataset(arrays, dim)
#     return DataLoader(dset, batch_size=batch_size, shuffle=shuffle)

















###Kristien's code edited by me - put back into its own file eventually
def grain_size(im, max_id=19999): 
    #"im" is a torch.Tensor grain id image of shape=(1,1,dim1,dim2) (only one image at a time)
    #'max_id' defines which grain id neighbors should be returned -> range(0,max_id+1)
    #Outputs are of length 'max_id'+1 where each element corresponds to the respective grain id
    
    search_ids = torch.arange(max_id+1).to(im.device) #these are the ids being serach, should include every id possibly in the image
    im2 = torch.hstack([im.flatten(), search_ids]) #ensures the torch.unique results has a count for every id
    areas = torch.unique(im2, return_counts=True)[1]-1 #minus 1 to counteract the above concatenation
    sizes = 2*torch.sqrt(areas/np.pi) #assumes grain size equals the diameter of a circular area - i.e. d = 2 * (A/pi)^(1/2)

    return sizes


def grain_num_neighbors(im, max_id=19999, if_AW=False):
    #"im" is a torch.Tensor grain id image of shape=(1,1,dim1,dim2) (only one image at a time)
    #'max_id' defines which grain id neighbors should be returned -> range(0,max_id+1)
    #Outputs are of length 'max_id'+1 where each element corresponds to the respective grain id
    
    #Pad the images to define how pairs are made along the edges
    im_pad = pad_mixed(im, [1,1,1,1], pad_mode="circular")
    
    #Find all the unique id nieghbors pairs in the image
    pl = torch.stack([im[0,0,].flatten(), im_pad[0,0,1:-1,0:-2].flatten()]) #left pairs
    pr = torch.stack([im[0,0,].flatten(), im_pad[0,0,1:-1,2:].flatten()]) #right pairs
    pu = torch.stack([im[0,0,].flatten(), im_pad[0,0,0:-2,1:-1].flatten()]) #up pairs
    pd = torch.stack([im[0,0,].flatten(), im_pad[0,0,2:,1:-1].flatten()]) #down pairs
    pairs = torch.hstack([pl,pr,pu,pd]) #list of all possible four neighbor pixel pairs in the image
    pairs_sort, _ = torch.sort(pairs, dim=0) #makes pair order not matter
    pairs_unique = torch.unique(pairs_sort, dim=1) #these pairs define every grain boundary uniquely (plus self pairs like [0,0]
    
    #Find how many pairs are associated with each grain id
    search_ids = torch.arange(max_id+1).to(im.device) #these are the ids being serach, should include every id possibly in the image
    pairs_unique2 = torch.hstack([pairs_unique.flatten(), search_ids]) #ensures the torch.unique results has a count for every id
    num_neighbors = torch.unique(pairs_unique2, return_counts=True)[1]-3 #minus 1 to counteract the above concatenation and 2 for the self pairs (e.g. [0,0])
    
    if if_AW==True:
        l = []
        for ids in tqdm(range(max_id+1)):
            if num_neighbors[ids]==0: l.append(0)
            else:
                i = (torch.sum(pairs_unique[:,torch.sum(pairs_unique==ids, dim=0)==1], dim=0)-ids).long()
                l.append(torch.mean(num_neighbors[i].float()))
        AW = torch.Tensor(l).to(im.device)
        return num_neighbors, AW
    else: 
        return num_neighbors


def metric_stats(array):
    #'array' is a 1d numpy array
    
    array = array[array!=0] #remove all zero values
    mn = np.mean(array)
    std = np.std(array)
    skw = skew(array)
    kurt = kurtosis(array, fisher=True)
    stats = np.array([mn, std, skw, kurt])

    return stats


def apply_grain_func(h5_path, func, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    #'h5_path' is the full path from current folder to h5 file (assumes a single dataset called 'images')
    #'func' is a function that inputs a torch.Tensor of shape=(1,1,dim1,dim2) and outputs a constant sized 1d tensor array
    
    l = []
    l_stats = []
    with h5py.File(h5_path, "r") as f:
        num_images = f['images'].shape[0]
        max_id = np.max(f['images'][0])
        for i in tqdm(range(num_images)): #for all the images
            im = torch.from_numpy(f['images'][i].astype(np.int16)).unsqueeze(0).unsqueeze(0).to(device) #convert to tensor of correct shape
            array = func(im, max_id=max_id).cpu().numpy() #rungiven function
            l.append(array) #store results
            l_stats.append(metric_stats(array)) #run and store stats
         
    arrays = np.stack(l)
    array_stats = np.stack(l_stats)
    
    return arrays, array_stats


# fp = 'data/sims dqn2_AGG_grain256_structure257_episode200_9observ_9action_kt0.5'
# h5_path = '%s/Case4_2400p.hdf5'%fp
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# arrays, array_stats = apply_grain_func(h5_path, func=grain_size, device=device)
# np.savetxt("%s/grain_sizes.csv"%fp, arrays, delimiter=",")
# np.savetxt("%s/grain_size_stats.csv"%fp, array_stats, delimiter=",")

# arrays, array_stats = apply_grain_func(h5_path, func=grain_num_neighbors, device=device)
# np.savetxt("%s/grain_num_neighbors.csv"%fp, arrays, delimiter=",")
# np.savetxt("%s/grain_num_neighbor_stats.csv"%fp, array_stats, delimiter=",")




