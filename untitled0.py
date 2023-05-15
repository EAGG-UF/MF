# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""











import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import numpy as np
import sPkY as fs
import dropbox as db
import os
import numpy as np
from PIL import Image
import imageio
import torch



























#Download EBSD data - https://www.dropbox.com/home/Abnormal/Experimental%20Microstructures
#Delete "Old Samples"

#Find all file that end in ".dream3d" with their paths
from glob2 import glob
fps = glob('./**/*.dream3d', recursive=True)

#Copy all of the IDs to a new h5 file for better access
for fp in fps:
    with h5py.File(fp, 'r') as f:
        image = f['DataContainers/ImageDataContainer/CellData/FeatureIds'][0,:,:,0]
    
    fp_split = fp.split('\\')
    with h5py.File('Experimental_EBSD_data.h5', 'a') as f:
        f[fp_split[3]+'/'+fp_split[5]+'/'+fp_split[6].split('.')[0]] = image

#List the shape of all the images
with h5py.File('Experimental_EBSD_data.h5', 'r') as f:
    for i, l1 in enumerate(list(f.keys())):
        for j, l2 in enumerate(list(f[l1].keys())):
            l = l1+'/'+l2
            print('(%d, %d) - '%(i,j)+l)
            for k, l3 in enumerate(list(f[l].keys())):
                print('(%d) - '%k+str(f[l+'/'+l3].shape))
                
#View a whole folder of images
im_list = []
i = 0
j = 3
with h5py.File('Experimental_EBSD_data.h5', 'r') as f:
    l1 = list(f.keys())[i]
    l2 = list(f[l1].keys())[j]
    print(f.keys())
    print(f[l1].keys())
    for k, l3 in enumerate(list(f[l1+'/'+l2].keys())):
        im = f[l1+'/'+l2+'/'+l3][:]
        # plt.imshow(im)
        # plt.title(str(k))
        # plt.show()
        im_list.append(im)

#Select a specific image
i = 0
j = 1
k = 0
with h5py.File('Experimental_EBSD_data.h5', 'r') as f:
    l1 = list(f.keys())[i]
    l2 = list(f[l1].keys())[j]
    l3 = list(f[l1+'/'+l2].keys())[k]
    print(f.keys())
    print(f[l1].keys())
    print(f[l1+'/'+l2].keys())
    im = f[l1+'/'+l2+'/'+l3][:]
    plt.imshow(im)
    plt.title(l1+'/'+l2+'/'+str(k))
    plt.show()






#Lets look at the covariance matrix of the experimental data
#Magnet textured
im_list_0 = im_list #(0,1) #as sintered
im_list_8 = im_list #(0,4) #8 hrs
im_list_16 = im_list #(0,0) #16 hrs
im_list_32 = im_list #(0,2) #32 hrs
im_list_64 = im_list #(0,3) #64 hrs





i=0
im_list = [im_list_0[i], im_list_8[i], im_list_16[i], im_list_32[i], im_list_64[i]]








aa = np.array([0,8,16,32,64])

mean_list = []
std_list = []

im_lists = [im_list_0, im_list_8, im_list_16, im_list_32, im_list_64]
for im_list in im_lists:
    l=[]
    for im in im_list:
        tmp = np.product(im.shape)/(np.unique(im).shape)
        l.append(tmp)
        
    mean_list.append(np.mean(l))
    std_list.append(np.std(l))
        
plt.plot(aa, np.array([mean_list, std_list]).transpose(), '*-')

tmp = np.array(mean_list)
gs_slope, _ = np.polyfit(aa, tmp, 1)


# plt.plot(aa, np.array(l), '*-')
# gs_slope, _ = np.polyfit(aa, np.array(l), 1)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


cmv_mean_list = []
cmv_std_list = []

im_lists = [im_list_0, im_list_8, im_list_16, im_list_32, im_list_64]
for im_list in im_lists:
    l = []
    for im in im_list:
        im = torch.from_numpy(im.astype(float)).short().to(device)
        cvm = image_covariance_matrix(im, min_max=[-100, 100], num_samples=16)
        l.append(cvm)


    #Do the covariance matricies change in a single step?
    cmv = np.array(l)
    # cmv = cmv/cmv[:,1:2,1:2]
    cmv_mean = np.mean(cmv, axis=0)
    cmv_std = np.std(cmv, axis=0)
    print(cmv_mean)
    print(cmv_std)
    
    cmv_mean_list.append(cmv_mean)
    cmv_std_list.append(cmv_std)
    






a = np.array(cmv_mean_list)
b = np.array(cmv_std_list)

#plot the slop of the means and variances

plt.plot(aa, a.reshape(5,-1), '*-'); plt.show()
plt.plot(aa, b.reshape(5,-1), '*-'); plt.show()





dd = np.array(cmv_mean_list)
var = dd[:,np.array([0,1,0]), np.array([0,1,1])]
plt.plot(aa, var, '*-')

xvar, _ = np.polyfit(aa, var[:,0], 1)
yvar, _ = np.polyfit(aa, var[:,1], 1)
xycov, _ = np.polyfit(aa, var[:,2], 1)

print('Variances (x, y, cov): %.2f, %.2f, %.2f'%(xvar, yvar, xycov))
print('Ratios (y/x, cov/x): %.2f, %.2f'%(yvar/xvar, xycov/xvar))








#Nonmagenet - Ratios (y/x, cov/x): 0.91, 0.07
#gs_slope - 33.940429422957

#Sim nonmag - Ratios (y/x, cov/x): 0.85, 0.08
#gs_slope - 46.88913001003387

#Mag - Ratios (y/x, cov/x): 1.71, 0.39
# 59.67657240268154

#Sim mag - 2.19, 0.53
# 56.73996085787561





#now create the same image lists with just the mode filter and show its variation






device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ll = [[],[],[],[],[]]



for i in range(len(ll)):

    for _ in range(10):
        s = im_list_0[i].shape
        img, _, _ = fs.voronoi2image(size=s, ngrain=int(np.product(s)/131), memory_limit=1e9, p=2)
        im = img.short().to(device)
        cov_mat = torch.Tensor([[1,0.39],[0.39,1.71]])*59.6
        num_samples = 32
        bounds=['wrap','wrap']

        for _ in tqdm(range(i)):
            im = normal_mode_filter2(im, cov_mat, num_samples, bounds=bounds)
        
        ll[i].append(im.cpu().numpy())
        


im_list_0 = ll[0] 
im_list_8 = ll[1]
im_list_16 = ll[2]
im_list_32 = ll[3] 
im_list_64 = ll[4]



plt.imshow(im_list_64[0])



#Ratios (y/x, cov/x): 11.17, 2.50
#Avg grain area: 6.75812716


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
im = torch.from_numpy(im_list[0]).short().to(device)
cov_mat = torch.Tensor([[1,2.50],[02.50,11.17]])*6.76
num_samples = 32
bounds=['nwrap','nwrap']

l = [im.cpu().numpy()]
for i in tqdm(range(64)):
    im = normal_mode_filter2(im, cov_mat, num_samples, bounds=bounds)
    l.append(im.cpu().numpy())
    # print(np.sum(l[-1]!=l[-2]))
    # print(np.unique(l[-1]).shape[0])
plt.imshow(l[-1]); plt.show()






aa = np.array(l).astype(np.uint16)
with h5py.File("./case experiments/%s.hdf5"%f, 'w') as fl:
    fl["images"] = aa
    
    
imageio.mimsave('./experimental.gif'%f, aa.astype(np.uint8))
    
plt.imshow(im_list[4])
    
    
import moviepy.editor as mp

clip = mp.VideoFileClip('./case experiments/%s.gif'%f)
clip.write_videofile('./case experiments/%s.mp4'%f)






ll=[]
for im in l:
    tmp = np.product(im.shape)/(np.unique(im).shape)
    ll.append(tmp)














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
    
    
    
    
    
    
    
    
    
dd = np.array(l_cov)
var = dd[:,np.array([0,1,0]), np.array([0,1,1])]
plt.plot(var)

xvar, _ = np.polyfit(np.arange(len(var[:,0])), var[:,0], 1)
yvar, _ = np.polyfit(np.arange(len(var[:,0])), var[:,1], 1)
xycov, _ = np.polyfit(np.arange(len(var[:,0])), var[:,2], 1)

print('Variances (x, y, cov): %.2f, %.2f, %.2f'%(xvar, yvar, xycov))
print('Ratios (y/x, cov/x): %.2f, %.2f'%(yvar/xvar, xycov/xvar))




























def normal_mode_filter(im, cov_mat=torch.Tensor([[25,0],[0,25]]), num_samples=64, bounds=['wrap','wrap']):
    
    cov_mat = cov_mat.to(im.device)
    mean_arr = torch.zeros(2).to(im.device)
    
    #Sample and calculate the index coords
    mvn = torch.distributions.MultivariateNormal(mean_arr, cov_mat)
    samples = mvn.sample((num_samples, im.numel())).int().transpose(1,2) #sample separately for each site
    # samples = mvn.sample((num_samples,)).int().reshape(-1,2,1) #use a single sample for all sites
    samples = torch.cat([samples, samples*-1], dim=0).to(im.device) #mirror the samples to keep a zero mean

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
        
    #Gather the coord values and take the mode for each pixel      
    im_expand = im.reshape(-1,1).expand(-1, im.numel())
    im_sampled = torch.gather(im_expand, dim=0, index=index)
    del samples
    del coords
    del index
    
    # im_next = torch.mode(im_sampled.cpu(), dim=0).values.reshape(im.shape)
    # im_next = im_next.to(im.device)
    
    
    im_next = fs.rand_mode(im_sampled).reshape(im.shape)
    
    return im_next


#using torch dataset and dataloader are far too slow
def normal_mode_filter2(im, cov_mat=torch.Tensor([[25,0],[0,25]]), num_samples=64, bounds=['wrap','wrap']):
    
    cov_mat = cov_mat.to(im.device)
    mean_arr = torch.zeros(2).to(im.device)
    
    #Sample and calculate the index coords
    mvn = torch.distributions.MultivariateNormal(mean_arr, cov_mat)
    
    arr0 = torch.arange(im.shape[0]).to(im.device)
    arr1 = torch.arange(im.shape[1]).to(im.device)
    coords = torch.cartesian_prod(arr0, arr1).float().transpose(0,1).reshape(1, 2, -1)
    
    batch_size = int(im.numel()/10)
    l = []
    coords_split = torch.split(coords, batch_size, dim=2)
    for c in coords_split: 
        
        samples = mvn.sample((num_samples, c.shape[2])).int().transpose(1,2) #sample separately for each site
        samples = torch.cat([samples, samples*-1], dim=0).to(im.device) #mirror the samples to keep a zero mean
        c = samples + c
    
        #Set bounds for the indices
        if bounds[1]=='wrap': c[:,0,:] = c[:,0,:]%im.shape[0]
        else: c[:,0,:] = torch.clamp(c[:,0,:], min=0, max=im.shape[0]-1)
        if bounds[0]=='wrap': c[:,1,:] = c[:,1,:]%im.shape[1]
        else: c[:,1,:] = torch.clamp(c[:,1,:], min=0, max=im.shape[1]-1)
    
        #Flatten indices
        index = (c[:,1,:]+im.shape[1]*c[:,0,:]).long()
            
        #Gather the coord values and take the mode for each pixel      
        im_expand = im.reshape(-1,1).expand(-1, batch_size)
        im_sampled = torch.gather(im_expand, dim=0, index=index)
        
        l.append(fs.rand_mode(im_sampled))
        
    im_next = torch.hstack(l).reshape(im.shape)
    
    return im_next






f = "Case4_2400p_rand200_balanced" #file name


#Let's make sure the three tests still work for the gaussian

# im_p, _, _ = fs.voronoi2image(size=[2400, 2400], ngrain=20000)
# im_p, _, _ = fs.voronoi2image(size=[1024, 1024], ngrain=1024)
# im_p, _, _ = fs.voronoi2image(size=[1024, 1024], ngrain=2**13)
# im_p, _ = fs.generate_circleIC(size=[512,512], r=200, c=None)
# im_p, _ = fs.generate_3grainIC(size=[512,512], h=350)
# gc = fs.generate_hex_grain_centers(dim=512, dim_ngrain=8)
# size = torch.Tensor([432, 512]).to(device)
# im_p, _, _ = fs.voronoi2image(size, ngrain=None, center_coords0=gc)




# # Polycrystaline initial condition
# # img, EulerAngles, center_coords = fs.voronoi2image(size=[2400,2400], ngrain=20000, memory_limit=1e9, p=2)
# grain_centers = fs.read_grain_centers_txt(fp="Case4_grains_centers")
# img, EulerAngles, center_coords = fs.voronoi2image(size=[2400,2400], ngrain=grain_centers.shape[0], center_coords0=grain_centers)
# # fs.write_grain_centers_txt(center_coords, fp="../../GrainGrowth/%s_grains_centers.txt"%f)
# plt.imshow(img)

# np.save("Case4.npy", img)
img = np.load("Case4.npy")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
im = torch.from_numpy(img).short().to(device)
# im = im_p.short().to(device)#torch.from_numpy(im_p.astype(float)).short().to(device)
cov_mat = torch.Tensor([[50,0],[0,50]])
num_samples = 32
bounds=['wrap','wrap']

l = [im.cpu().numpy()]
for i in tqdm(range(200)):
    im = normal_mode_filter2(im, cov_mat, num_samples, bounds=bounds)
    l.append(im.cpu().numpy())
    # print(np.sum(l[-1]!=l[-2]))
    # print(np.unique(l[-1]).shape[0])
plt.imshow(l[-1]); plt.show()

np.unique(l[-1]).shape


aa = np.array(l).astype(np.uint16)
with h5py.File("./case experiments/%s.hdf5"%f, 'w') as fl:
    fl["images"] = aa
    
    
imageio.mimsave('./case experiments/%s.gif'%f, aa.astype(np.uint8))
    
    
    
import moviepy.editor as mp

clip = mp.VideoFileClip('./case experiments/%s.gif'%f)
clip.write_videofile('./case experiments/%s.mp4'%f)

























fp = 'data/sims dqn2_AGG_grain256_structure257_episode1000_17observ_17action_kt0.5_setrot'
h5_path = '%s/Case4_2400p.hdf5'%fp
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

arrays, array_stats = fs.apply_grain_func(h5_path, func=fs.grain_size, device=device)
np.savetxt("%s/grain_sizes.csv"%fp, arrays, delimiter=",")
np.savetxt("%s/grain_size_stats.csv"%fp, array_stats, delimiter=",")

arrays, array_stats = fs.apply_grain_func(h5_path, func=fs.grain_num_neighbors, device=device)
np.savetxt("%s/grain_num_neighbors.csv"%fp, arrays, delimiter=",")
np.savetxt("%s/grain_num_neighbor_stats.csv"%fp, array_stats, delimiter=",")














with h5py.File(h5_path, "r") as f:
    im = torch.from_numpy(f['images'][0].astype(np.int16)).unsqueeze(0).unsqueeze(0).to(device) #convert to tensor of correct shape
       




plt.plot()














#find the pixels that border two grains, then three grains (a juntion)


im_p, _, _ = fs.voronoi2image(size=[512, 512], ngrain=512)
im = im_p.unsqueeze(0).unsqueeze(0).to(device) #convert to tensor of correct shape
   

a = fs.view_unfoldNd(im, kernel_size=3, pad_mode='circular').squeeze()

# pairs_unique[:,torch.sum(pairs_unique==0, dim=0)!=0]
#neighbors to [0] - [34.,  97., 128., 157., 206., 311., 426.]

b = torch.sum(torch.sum(a==0, dim=2), dim=2)>0
c = torch.sum(torch.sum(a==97, dim=2), dim=2)>0
d = torch.sum(torch.sum(a==34, dim=2), dim=2)>0


e = torch.logical_and(torch.logical_and(b, c), d)


plt.imshow(e.cpu())

torch.sum(e)














grain_size - mean only not norm
neighbors - nothing norm

#stats
#cols - mean (not norm), std, skew, kurt
#rows - steps


grain_size - normed
neighbors - nothing norm

#other
#rows - steps
#cols - all the arrays
























device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
im_p, _, _ = fs.voronoi2image(size=[2400, 2400], ngrain=20000)
im = im_p.short().to(device)#torch.from_numpy(im_p.astype(float)).short().to(device)
cov_mat = torch.Tensor([[50,0],[0,50]])
num_samples = 16
bounds=['wrap','wrap']











aaa, bbb = fs.view_unfoldNd(ims, kernel_size=3, pad_mode='circular')



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    l_cov = []
    
    for i in tqdm(range(len(l)-1)):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        im = torch.from_numpy(l[i].astype(float)).to(device)
        im_next = torch.from_numpy(l[i+1].astype(float)).to(device)
        
        
        # cov_mat=torch.Tensor([[300,0],[0,300]])
        min_max = [-200, 200]
        num_samples=8
        bounds=['wrap','wrap']
        
        
        
        # cov_mat = cov_mat.to(device)
        mean_arr = torch.zeros(2).to(device)
        
        #Sample and calculate the index coords
        # mvn = torch.distributions.MultivariateNormal(mean_arr, cov_mat)
        # samples = mvn.sample((num_samples, im.numel())).int().transpose(1,2)
        # #samples = mvn.sample((num_samples,)).int().reshape(-1,2,1)
        # #samples = torch.cat([samples, samples*-1], dim=0) #mirror the samples to keep a zero mean
        mvn = torch.distributions.Uniform(torch.Tensor([min_max[0]]).to(device), torch.Tensor([min_max[1]]).to(device))
        samples = mvn.sample((num_samples, 2, im.numel()))[...,0].int()
        
        arr0 = torch.arange(im.shape[0]).to(device)
        arr1 = torch.arange(im.shape[1]).to(device)
        coords = torch.cartesian_prod(arr0, arr1).float().transpose(0,1).reshape(1, 2, -1)
        coords = samples+coords
        
        #Set bounds for the indices
        if bounds[1]=='wrap': coords[:,0,:] = coords[:,0,:]%im.shape[0]
        else: coords[:,0,:] = torch.clamp(coords[:,0,:], min=0, max=im.shape[0]-1)
        if bounds[0]=='wrap': coords[:,1,:] = coords[:,1,:]%im.shape[1]
        else: coords[:,1,:] = torch.clamp(coords[:,1,:], min=0, max=im.shape[1]-1)
        
        #Flatten indices
        index = (coords[:,1,:]+im.shape[1]*coords[:,0,:]).long()
        
        
        
        
        ttt = im.reshape(-1)[index]==im_next.reshape(-1)
        c = samples.transpose(1,2)[ttt].transpose(1,0).cpu().numpy()
        # plt.plot(c[0,:1000], c[1,:1000], '.')
        
        
        d = np.cov(c)
        l_cov.append(d)
        
        
        
    dd = np.array(l_cov)
    
    # cut = 15
    # plt.plot(dd[:,0,0], dd[:,1,1]); plt.show()
    # vr, _ = np.polyfit(np.sqrt(dd[cut:,0,0]), np.sqrt(dd[cut:,1,1]), 1)
    # print(vr)
    
    # cut = 35
    
    var = dd[:,np.array([0,1,0]), np.array([0,1,1])]
    plt.plot(var)
    
    xvar, _ = np.polyfit(np.arange(len(var[:,0])), var[:,0], 1)
    yvar, _ = np.polyfit(np.arange(len(var[:,0])), var[:,1], 1)
    xycov, _ = np.polyfit(np.arange(len(var[:,0])), var[:,2], 1)
    
    print('Variances (x, y, cov): %.2f, %.2f, %.2f'%(xvar, yvar, xycov))
    print('Ratios (y/x, cov/x): %.2f, %.2f'%(yvar/xvar, xycov/xvar))
    
    
    
    # plt.plot(var[:,2]/var[:,1])
    # plt.plot(dd[:,1,0]/dd[:,0,0]); plt.show()
    # cvr = np.mean(dd[cut:,1,0]/dd[cut:,1,1])
    
    # cvr_arr = dd[cut:,1,0]/dd[cut:,1,1]
    # mi = np.argmax(np.abs(cvr_arr))
    # cvr = cvr_arr[mi]
    # print(cvr)
    
    lll.append(np.array([cov_mat[0,0]/xvar, cov_mat[1,1]/yvar, cov_mat[1,0]/xycov]))
    llll.append(np.array([yvar/xvar, xycov/xvar]))
    print(len(lll))
    
    
np.mean(np.array(llll), axis=0)








#I think I can now look at a structure, characterize it, and grow towards it by using it to create a kernel (which may have more than covariance data)
#I can also just extract the covariance data from one image and the next and grow toward that ratio


#Now:
    #Check other ratios to see if they work
    #Look into covariances specifically, can those be tracked, are they specific ratios?
    #Try to mimic experimental data









imageio.mimsave('gauss hex variance 50 sample size 256 all sample.gif', (6*np.array(l)).astype(np.uint8))
imageio.mimsave('gauss 3 grain variance 200.gif', (200*np.array(l)).astype(np.uint8))
plt.imshow(im_p)






with h5py.File('Case4_2400p.hdf5', 'r') as f: 
    im_p = f['images'][0,]
with h5py.File('Case4_mode_filter.hdf5', 'w') as f: 
    ims = f.create_dataset("images", shape=(201, 2400, 2400), dtype='uint16')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
im = torch.from_numpy(im_p.astype(float)).to(device)

with h5py.File('Case4_mode_filter.hdf5', 'a') as f:
    f['images'][0,] = im.cpu().numpy()
    for i in tqdm(range(200)):
        im = normal_mode_filter(im, device=device)
        f['images'][i+1,] = im.cpu().numpy()






    

with h5py.File('Case4_mode_filter.hdf5', 'r') as f: 
    a = f['images'][:]


imageio.mimsave('Case4_mode_filter.gif', a.astype(np.uint8))





c = [np.unique(x).shape[0] for x in a]
plt.plot(c)









import time

start = time.time()


end = time.time()
print(end - start)

































f = open("Case3Periodic.e") 


print(f.readline())



f.close()




import matplotlib.pyplot as plt
import cv2
vidcap = cv2.VideoCapture('Case4UniqueGrains.mov')
success,image = vidcap.read()
ims = np.expand_dims(image, axis=0)

while(1):
    success,image = vidcap.read()
    if success!=True: break
    else: ims = np.concatenate([ims,np.expand_dims(image, axis=0)], axis=0)
    
    
















import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import numpy as np
import sPkY as fs
import dropbox as db
import os
import numpy as np
from PIL import Image
import imageio
import torch



#Load and concatenate all of the images
l = []
for i in tqdm(range(598)): #358, 598

    #Load image
    # im = np.array(imageio.imread('Case3PeriodicUniqueGrains/case3 (%d).png'%(i+1)))
    im = np.array(imageio.imread('Case4LowResUniqueGrains/case4 (%d).png'%(i+1)))
    l1 = np.all(im[int(im.shape[0]/2),:,:]!=np.array([255,255,255]), axis=1)
    l2 = np.all(im[:,int(im.shape[1]/2),:]!=np.array([255,255,255]), axis=1)
    a = np.where(l1==True)[0]
    b = np.where(l2==True)[0]
    im1 = im[b[0]:b[-1]+1, a[0]:a[-1]+1,:3]
    
    #Reshape and rotate
    pil_image=Image.fromarray(im1)
    im2 = np.array(pil_image.resize((2400,2400)))
    im3 = np.rot90(im2, 3)
    
    
    # #Flatten channels 
    # im4 = im3[...,0] + im3[...,0]*512 + im3[...,0]*512**2
    
    # #Mode filter
    # im_old = torch.from_numpy(im4.astype('float')).reshape(1,1,512,512)
    # while(1):
    #     tmp = fs.my_unfoldNd(im_old, kernel_size=5)
    #     im_new = torch.mode(tmp, dim=1).values[0].reshape(1,1,512,512)
    #     if torch.all(im_old==im_new): break
    #     im_old = im_new
    
    # im5 = im_new[0,0,]
    # plt.imshow(im5)
    
    l.append(im3)
ll = np.array(l)


#Load grain centers and ID values
# centers  = np.round(np.genfromtxt('Case3_grains_centers.txt', delimiter=' ')[1:,:]).astype(int)
# with h5py.File('Case3_512p.hdf5', 'r') as f: 
#     im_p = f['images'][0,]
#     ids = im_p[centers[:,0], centers[:,1]]

centers = np.genfromtxt('Case4_grains_centers.txt', delimiter=' ')[1:,:].astype(int)
with h5py.File('Case4_2400p.hdf5', 'r') as f: 
    im_p = f['images'][0,]
    ids = im_p[centers[:,0], centers[:,1]]





#Start with the unique_id initial condition
#Add one image at a time
#For each additional image:
    #Section each grain and reassign the most common value for each to the grain_id
    #Then mode filter
    
    
  

# ims = torch.from_numpy(ll.copy()) #ll[...,0] + ll[...,0]*512 + ll[...,2]*512**2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

ims = torch.from_numpy(im_p.reshape(1,2400,2400).astype(float).copy())
lll = ll[...,0] + ll[...,0]*2400 + ll[...,2]*2400**2
# ims_new = torch.from_numpy((ll[...,0] + ll[...,0]*2400 + ll[...,2]*2400**2).astype(float).copy())


for j in tqdm(range(lll.shape[0]-1)):

    im = ims[j,].reshape(-1).to(device)
    # im_new = ims_new[j+1,].reshape(-1).to(device)
    im_new = torch.from_numpy(lll[j+1,].astype(float).copy()).reshape(-1).to(device)
    
    #Reassign mode of each grain area to correct ID
    for i in range(20000):
        if torch.sum(im==i)>0: #if the grain still exists
            v = torch.mode(im_new[im==i], dim=0).values
            bool_arr = ((im_new==v)*(im==i))==1
            im_new[bool_arr] = i
    
    #Mode filter
    tmp0 = fs.my_unfoldNd(im.reshape(1,1,2400,2400), kernel_size=3) #add the values of the image before 
    im_old = im_new.reshape(1,1,2400,2400)
    while(1):
        tmp1 = fs.my_unfoldNd(im_old, kernel_size=3)
        tmp = torch.cat([tmp0, tmp1], dim=1)
        im_new = torch.mode(tmp.cpu(), dim=1).values[0].reshape(1,1,2400,2400).to(device)
        if torch.all(im_old==im_new): break
        im_old = im_new
        print("Iteration")
    
    ims = torch.cat([ims, im_new.reshape(1, 2400, 2400).cpu()], dim=0)
    
    plt.imshow(ims[-1,].cpu()); plt.show() #look at it
    print('\n')
    print('Number of Grains: %d'%torch.unique(ims[-1,]).shape)
    print('Number of differences from step 1: %d\n'%torch.sum(ims[-1,]!=ims[0,]))
    










def load_moose_image(fp='Case4LowResUniqueGrains/case4 (1).png', crop_color=np.array([255,255,255]), size=[2400,2400]):

    #Load
    im = np.array(imageio.imread(fp))
    
    #Crop out crop color (assumes it is square)
    is_not_crop_color = np.all(im!=crop_color, axis=2)
    a, b = np.where(is_not_crop_color)
    im_cropped = im[a[0]:a[-1]+1, b[0]:b[-1]+1,:3]
    
    #Reshape 
    pil_image=Image.fromarray(im_cropped)
    im_reshape = np.array(pil_image.resize(size))
    
    #Rotate image (to match SPPARKS and PRIMME data)
    im_rot = np.rot90(im_reshape, 3)
    
    #Flatten to one dimensions (for processing)
    im_final = im_rot[...,0] + im_rot[...,0]*2400 + im_rot[...,2]*2400**2
    
    return im_final



def find_neighbor_coords(coords, lim=[512, 512]):
    tmp = [np.array([1,0]), np.array([-1,0]), np.array([0,1]), np.array([0,-1])]
    all_coords = np.concatenate([coords, coords+tmp[0], coords+tmp[1], coords+tmp[2], coords+tmp[3]])
    
    #Wrap values outside the allowed limits 
    all_coords[:,0] = all_coords[:,0] % lim[0]
    all_coords[:,1] = all_coords[:,1] % lim[1]
    
    neigh_coords = np.unique(all_coords, axis=0) 
    return neigh_coords





def assign_ids_to_grain_modes(im_before, im, num_grains): 
    print('Assigning IDs to the modes for each segmented grain color...')
    device = torch.device("cuda:%d"%im_before.get_device() if torch.cuda.is_available() else "cpu")
    size=torch.tensor((im_before.shape)).to(device)
    for i in range(num_grains):
        if torch.sum(im_before==i)>0: #if the grain still exists
            color_mode = torch.mode(im[im_before==i], dim=0).values
            
            
            c = torch.stack(torch.where(im_before==i)).transpose(1,0).cpu().numpy()
            for i in range(1): c = find_neighbor_coords(c, lim=[512, 512])
            img = np.zeros([512,512])
            img[c[:,0], c[:,1]] = 1
            img = torch.from_numpy(img).to(device)
            
            # #find a circle around the grain
            # x, y = torch.where(im_before==i)
            # r = int(np.ceil(float(torch.sqrt(((x[-1]-x[0])/2)**2+((y[-1]-y[0])/2)**2)))) + 2
            # c = [int(x[0]+(x[-1]-x[0])/2), int(y[0]+(y[-1]-y[0])/2)]
            # img, _ = fs.generate_circleIC(size=size, r=r, c=c)
            
            
    
            bool_arr = ((im==color_mode)*img)==1
            im[bool_arr] = i
    return im



def iterative_mode_filtering(im, kernel_size, ims_for_smoothing=[]):
    device = torch.device("cuda:%d"%im.get_device() if torch.cuda.is_available() else "cpu")
    print('Performing iterative mode filtering...')
    size = tuple(im.shape)
    
    windows_for_smoothing = []
    for i in ims_for_smoothing:
        windows_for_smoothing.append(fs.my_unfoldNd(i.reshape((1,1,)+size), kernel_size=kernel_size))
        
    im_old = im.reshape((1,1,)+size)
    while(1):
        windows = fs.my_unfoldNd(im_old, kernel_size=kernel_size)
        
        for i in windows_for_smoothing:
            windows = torch.cat([windows, i], dim=1)
            
        im_new = torch.mode(windows.cpu(), dim=1).values[0].reshape((1,1,)+size)
        if device!=None: im_new = im_new.to(device)
        if torch.all(im_old==im_new): 
            break
        im_old = im_new
    return im_new[0,0,]



def moose_png_to_hdf5(path_to_IC_hdf5='Case4_2400p.hdf5', path_to_moose_hdf5='Case4_MOOSE.hdf5', path_to_moose_ims='Case4LowResUniqueGrains/case4 (%d).png', num_files=598):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with h5py.File(path_to_IC_hdf5, 'r') as f: 
        im0 = torch.from_numpy(f['images'][0,].astype(float)).to(device) #first image should look like this
    num_grains = torch.unique(im0).shape[0]
    size = tuple(im0.shape)
    tmp = np.array([8, 16, 32, 64], dtype=float)
    dtype = 'uint' + str(int(tmp[np.sum(num_grains>2**tmp)]))
    
    im_before = im0 
    im = torch.from_numpy(load_moose_image(fp=path_to_moose_ims%(1), size=size).astype(float)).to(device)
    
    with h5py.File(path_to_moose_hdf5, 'w') as f:
        ims = f.create_dataset("images", shape=((num_files+1,)+size), dtype=dtype)
        ims[0,] = im0.reshape((1,)+size).cpu().numpy()
        for j in tqdm(range(2,num_files+1)):
            im_next = torch.from_numpy(load_moose_image(fp=path_to_moose_ims%(j), size=size).astype(float)).to(device)
            im_ids = assign_ids_to_grain_modes(im_before, im, num_grains)
            im_ids_next = assign_ids_to_grain_modes(im_before, im_next, num_grains)
            
            im_filtered = iterative_mode_filtering(im_ids, 5, [im_before, im_ids_next])
            ims[j-1,] = im_filtered.reshape((1,)+size).cpu().numpy()
            im_before = im_filtered
            im = im_next
            
            plt.imshow(im_filtered.cpu()); plt.show() #look at it
            print('\nNumber of Grains: %d'%torch.unique(im_filtered).shape)
            print('Number of differences from step 1: %d\n'%torch.sum(im_filtered!=im0))







path_to_IC_hdf5 = 'Case3_512p.hdf5'
path_to_moose_hdf5 = 'Case3p_MOOSE_2.hdf5'
path_to_moose_ims = 'Case3PeriodicUniqueGrains/case3 (%d).png'
num_files = 358
moose_png_to_hdf5(path_to_IC_hdf5, path_to_moose_hdf5, path_to_moose_ims, num_files)
   


plt.imshow(im_before.cpu())
plt.imshow(im_ids.cpu())

plt.plot(torch.unique(im_ids).cpu())

torch.unique(im_ids)[:511]







windows = fs.my_unfoldNd(im_filtered.reshape(1,1,512,512), kernel_size=3)[0]

#range mode filter



import numpy as np
from scipy import stats

a = np.array([[1, 3, 4, 2, 2, 7],
              [5, 2, 2, 1, 4, 1],
              [3, 3, 2, 2, 1, 1]])

m, c = stats.mode(windows, axis=0)
print(m)

im_new = torch.mode(windows.cpu(), dim=1).values[0].reshape((1,1,)+size)






#flatten each window, remove anything that isn't in range, then take th emode using pytorch


imm = 

for i in range(512*512): 
    ii = windows[...,i]<512
    if torch.sum(ii)>0: window_mode = torch.mode(windows[ii,i]).values
    











with h5py.File('Case3p_MOOSE_2.hdf5', 'r') as f:
    a = f['images'][:]
    print(f['images'].shape)


imageio.mimsave('test2.gif', a.astype(np.uint8))

















k=25


# im_p, _, _ = fs.voronoi2image(size=[512, 512], ngrain=512)
# im_p, _ = fs.generate_circleIC(size=[512,512], r=200, c=None)
# im_p, _ = fs.generate_3grainIC(size=[512,512], h=350)
gc = fs.generate_hex_grain_centers(dim=512, dim_ngrain=8)
im_p, _, _ = fs.voronoi2image(size=[432, 512], ngrain=None, center_coords0=gc)
a = torch.from_numpy(im_p.reshape(1,1,512,512).astype(float))


# img= torch.ones(k,k)
# img[1:-1,1:-1]=0

# c = torch.Tensor([int(k/2), int(k/2)]).reshape(1,2).float()
# xy = torch.cartesian_prod(torch.arange(k), torch.arange(k)).float()
# # img = torch.cdist(xy, c, p=2)[:,0]<=int(k/2)
# d = torch.cdist(xy, c, p=2)[:,0]
# img = ((d<=int(k/2))*1).reshape(k,k)#(d>int(k/2)-1)).reshape(k,k)

# indxs = torch.stack(torch.where(img.reshape(-1)==True))

g = [a[0,0,].numpy()]
l = []
ll = []
for i in tqdm(range(100)):
    
    
    
    
    cvm = torch.Tensor([[50,0],[0,10]]) 
    m = torch.distributions.MultivariateNormal(torch.zeros(2), cvm)
    b = m.sample((64,)).int()
    b = torch.cat([b,b*-1], dim=0)
    v = (torch.max(torch.abs(b))*2+1)
    mn = float(int(v/2))
    img = torch.zeros(v,v)
    img[(b[:,0]+mn).long(), (b[:,1]+mn).long()] = 1
    plt.imshow(img)
    indxs = torch.stack(torch.where(img.reshape(-1)==True))
    
    
    windows = fs.my_unfoldNd(a, kernel_size=v)
    
    
    
    
    w = windows[0,indxs,]

    a_new = torch.mode(w, dim=1).values[0].reshape((1,1,512,512))
    l.append(torch.sum(a_new!=a))
    print(l[-1])
    a = a_new
    plt.imshow(a.reshape((512,512))); plt.show()
    ll.append(torch.unique(a).shape)
    print(ll[-1])
    g.append(a[0,0,].numpy())



plt.plot(np.diff(ll)); plt.show()
plt.plot(ll); plt.show()
plt.imshow(img)
imageio.mimsave('long xy gauss mode poly.gif', np.array(g).astype(np.uint8))




















with h5py.File('Case4_MOOSE_MATLAB.hdf5', 'w') as f: 
    ims = f.create_dataset("images", shape=(592, 2387, 2388), dtype='uint16')
        


with h5py.File('Case4_MOOSE_MATLAB.hdf5', 'a') as f:
    with h5py.File('Case4_4.h5', 'r') as ff:
        for i in tqdm(range(592,593)): #print(i)
            f['images'][i-1,] = ff["image%d"%i][:]


with h5py.File('Case4_4.h5', 'r') as f: 
    print(f.keys())



#1 - 1 through 200
#1_1 - 200 through 223
#2 - 224 through 323
#3 - 323 through 435
#4 - 436 through 592





a.shape
plt.imshow(a)



with h5py.File('Case4_MOOSE_MATLAB.hdf5', 'r') as f:
    a = f['images'][:]

imageio.mimsave('Case4_MOOSE_MATLAB.gif', a.astype(np.uint8))


























#try to make this faster
for i in range(num_grains):
    if torch.sum(im_grains==i)>0: #if the grain still exists
        color_mode = torch.mode(im_new[im_grains==i], dim=0).values
        bool_arr = ((im_new==color_mode)*(im_grains==i))==1
        im_new[bool_arr] = i



a = im_grains.reshape(1, -1).repeat(10, 1)

aa = (a==torch.arange(10).reshape(-1,1))





b = im_new.reshape(1, -1).repeat(10, 1)


b[aa]




mask = x!=0
y_mean = (y*mask).sum(dim=0)/mask.sum(dim=0)



c = torch.masked_select(b, aa)

torch.mode(b[0]*aa[0])
plt.imshow((b[0]*aa[0]).reshape(size))
plt.imshow((b[0]*aa[0]).reshape(size))


b[0]*aa[0]

torch.nan(1)



torch.mode(b*aa, dim=1).shape


b[aa==False] = float(torch.empty(1))
torch.mode(b, dim=1)



tmp_input[range(len(input)), target] = float("-Inf")



plt.imshow(aa[2,].reshape(size))
torch.sum(aa[1,])




















#Get the color for each id and a coord to grow from

im3 = ll[0,]

l = []
pp = []
for i in range(len(ids)):
    a, b = np.unique(im3.reshape(-1,3)[im_p.reshape(-1)==i], axis=0, return_counts=True)
    c = a[np.argmax(b),]
    l.append(c)
    
    aaa = (im_p==i)*np.all(im3==c, axis=2)
    xx, yy = np.where(aaa==True)
    j = int(len(xx)/2)
    p = np.array([xx[j], yy[j]])
    pp.append(p)

l = np.array(l)
p = np.array(pp)









for i in range(512):
    im_flat = im.reshape(-1,3)
    a, b = np.unique(im.reshape(-1,3)[im_p.reshape(-1)==i], axis=0, return_counts=True)
    c = a[np.argmax(b),]
    bool_arr = np.all(np.concatenate([im_flat==c, im_new.reshape(-1,1)==i], axis=1), axis=1)
    im_flat[bool_arr] = i
    
    
    
iim = bp.concatenate([im_new, im_flat.])    


plt.imshow(bool_arr.reshape(512,512))





for j in range(512):

    
j=0
im[im==torch.mode(torch.from_numpy(im[ims_new==j].astype('float'))).values.numpy()]=j

plt.imshow(im)


plt.imshow(im[ims_new==j])















#get the actual first image and reassign ids ot colors and set is as the first image


for i in range(512):
    im_p[im_p==i] = l[i,0] + l[i,1]*512 + l[i,2]*512**2







np.unique(l, axis=0).shape








#let's see if it works for 1 image first
im = ll[...,0] + ll[...,0]*512 + ll[...,2]*512**2

im[0,] = im_p

im_old = torch.from_numpy(im.astype('float')).reshape(1,1,512,512)
while(1):
    tmp = fs.my_unfoldNd(im_old, kernel_size=3)
    im_new = torch.mode(tmp, dim=1).values[0].reshape(1,1,512,512)
    if torch.all(im_old==im_new): break
    im_old = im_new

im = im_new[0,0,]
plt.imshow(im)


np.unique(im).shape




im_flooded = np.zeros(im.shape)-1

for i in tqdm(range(p.shape[0])):
    coord = tuple(p[i,])
    im_filled = flood(im, coord)
    x, y = np.where(im_filled==True)
    im_flooded[x, y] = ids[i]
    # plt.imshow(im_filled); plt.show()
    # print(np.sum(im_filled))

plt.imshow(im_flooded)

np.unique(im_flooded).shape


plt.imshow(im_flooded==-1)


#Find the coords for all 512 grains
im = ll[...,0] + ll[...,0]*512 + ll[...,2]*512**2
im_flooded = np.zeros(im.shape)-1

for i in tqdm(range(p.shape[0])):
    coord = tuple(np.concatenate([np.array([0]), p[i,]]))
    im_filled = flood(im, coord)
    z, x, y = np.where(im_filled==True)
    im_flooded[z, x, y] = ids[i]
    plt.imshow(im_filled[0,])
    print(np.sum(im_filled))




plt.imshow(im_flooded[0,])


np.unique(im_flooded[0,]).shape








im = ll
a = np.zeros(im.shape[:-1])-1

for i in tqdm(range(p.shape[0])):

    coord = np.concatenate([np.array([0]), p[i,]])
    coords = find_value_region(im, coord)
    print(coords.shape)
    a[coords[:,0], coords[:,1], coords[:,2]] = ids[i]




i=0
plt.imshow(a[i,])
i+=1






#Smooth the -1 values with a mode filter
#in 3d
im_old = torch.from_numpy(im4.astype('float')).reshape(1,1,512,512)
while(1):
    tmp = fs.my_unfoldNd(im_old, kernel_size=3)
    im_new = torch.mode(tmp, dim=1).values[0].reshape(1,1,512,512)
    if torch.all(im_old==im_new): break
    im_old = im_new

im5 = im_new[0,0,]
plt.imshow(im5)




























def find_value_region(im, coord):
    coords = coord.astype('int').reshape(1,-1)
    is_checked = np.array([0,])
    
    #Check all neighbors and keep those of the same value
    while np.any(is_checked==0):
        i = np.where(is_checked==0)[0][0] #find the index of the next coordinate with an unchecked neighbor
        
        #Extract the unchecked coordinates
        c = coords[i,]
        value = im[c[0],c[1],c[2]]
        z = np.array([1, -1, 0, 0, 0, 0]) + c[0]
        x = np.array([0, 0, 1, -1, 0, 0]) + c[1]
        y = np.array([0, 0, 0, 0, 1, -1]) + c[2]
        
        #Wrap values outside the allowed limits 
        z = np.clip(z, 0, im.shape[0]-1)
        x = x % im.shape[1]
        y = y % im.shape[2]
        
        #Extract coordinate value (allow for 1 or 3 color channels)
        #Check if the 
        if len(im.shape)==3: same_value = im[z,x,y,]==value 
        else: same_value = np.all(im[z,x,y,]==value, axis=1)
        
        #If they are equal and not already logged, log them
        for j in range(len(same_value)):
            coord = np.array([z[j], x[j], y[j]]).reshape(1,-1)
            if same_value[j]:
                already_logged = np.any(np.all(coord==coords, axis=1))
                if not already_logged:
                    coords = np.concatenate([coords,coord], axis=0)
                    is_checked = np.concatenate([is_checked, np.array([0])])
        is_checked[i] = 1 #mark as checked
    return coords















































#Load image
im = np.array(imageio.imread('hi.png'))
l1 = np.all(im[int(im.shape[0]/2),:,:]!=np.array([255,255,255]), axis=1)
l2 = np.all(im[:,int(im.shape[1]/2),:]!=np.array([255,255,255]), axis=1)
a = np.where(l1==True)[0]
b = np.where(l2==True)[0]
im1 = im[b[0]:b[-1]+1, a[0]:a[-1]+1,:3]

#Reshape and rotate
pil_image=Image.fromarray(im1)
im2 = np.array(pil_image.resize((512,512)))
im3 = np.rot90(im2, 3)

#Flatten channels 
im4 = im3[...,0] + im3[...,0]*512 + im3[...,0]*512**2

#Mode filter
im_old = torch.from_numpy(im4.astype('float')).reshape(1,1,512,512)
while(1):
    tmp = fs.my_unfoldNd(im_old, kernel_size=5)
    im_new = torch.mode(tmp, dim=1).values[0].reshape(1,1,512,512)
    if torch.all(im_old==im_new): break
    im_old = im_new

im5 = im_new[0,0,]
plt.imshow(im5)



#go pixel to pixel, if there is not already an id assigned there, then grow it and assign all those to the next id



ids = np.zeros([512,512])-1
next_id = 0

for i in tqdm(range(512)):
    for j in range(512):
        if ids[i, j]==-1:
            coord = np.array([i,j])
            coords = find_value_region(im5, coord)
            x = coords[:,0]
            y = coords[:,1]
            ids[x,y,] = next_id
            next_id += 1
print(next_id)
plt.imshow(ids)














#Load grain centers and ID values
centers  = np.round(np.genfromtxt('Case3_grains_centers.txt', delimiter=' ')[1:,:]).astype(int)
with h5py.File('Case3_512p.hdf5', 'r') as f: 
    im_p = f['images'][0,]
    ids = tmp[centers[:,0], centers[:,1]]




#Find value for each MOOSE grain

l = []
for i in range(len(ids)):
    a, b = np.unique(im3.reshape(-1,3)[im_p.reshape(-1)==i], axis=0, return_counts=True)
    c = a[np.argmax(b),]
    l.append(c)

l = np.array(l)

np.unique(l, axis=0).shape




from skimage.filters import gaussian


im = im3.reshape(-1,3)
a = np.zeros(512*512)
for ll in l: 
    tt = np.all(im==ll, axis=1).reshape(512, 512)
    tt = (gaussian(tt, sigma=1, mode='wrap')>0.4).reshape(-1)
    a[tt==True] = 1


plt.imshow(a.reshape(512, 512))



plt.imshow(tt.reshape(512, 512))




from skimage.filters import gaussian
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture



i=0
tt = np.all(im3.reshape(-1,3)==l[i,], axis=1).reshape(512, 512)
tt = gaussian(tt, sigma=1, mode='wrap')>0.5

x, y = np.where(tt==True)
xy = np.stack([x,y]).transpose()

kmeans = KMeans(n_clusters=2, random_state=0).fit(xy)
cc = np.round(kmeans.cluster_centers_).astype(int)

# gm = GaussianMixture(n_components=2, random_state=0).fit(xy)
# cc = np.round(gm.means_).astype(int)

plt.imshow(tt)
plt.plot(cc[:,1], cc[:,0], ".")

l[1,]
im3[cc[:,0], cc[:,1]]
i+=1








with h5py.File('myfile.h5', 'r') as f:
    print(f.keys())
    a = f['image0'][:]

plt.imshow(a)
np.unique(a).shape


#given the values of each grain, can I find the 




plt.imshow(np.all(im3.reshape(-1,3)==c, axis=1).reshape(512,512))


plt.imshow(im5)




aaa, bbb = np.unique(im3.reshape(-1,3), axis=0, return_counts=True)



np.sum(bbb>200)

plt.plot(np.ones(len(bbb)), bbb, '.')
plt.plot(1, 200, '.')
# plt.plot(np.zeros(bbb.shape)+200)




























#Use the initial IC to find the values of each grain (even though there are duplicates)
#For each image:
    #Step through the values to assign grain ids (use clustering for multiple grains with the same value)














np.sum(ids==-1)




def find_value_region (im, coord):
    coords = coord.astype('int').reshape(1,-1)
    is_checked = np.array([0,])
    
    #Check all neighbors and keep those of the same value
    while np.any(is_checked==0):
        i = np.where(is_checked==0)[0][0] #find the index of the next coordinate with an unchecked neighbor
        
        #Extract the unchecked coordinates
        c = coords[i,]
        value = im[c[0],c[1],]
        x = np.array([1, -1, 0, 0]) + c[0]
        y = np.array([0, 0, 1, -1]) + c[1]
        
        #Wrap values outside the allowed limits 
        x = x % im.shape[0]
        y = y % im.shape[1]
        
        #Extract coordinate value
        if len(im.shape)==2: same_value = im[x,y,]==value
        else: same_value = np.all(im[x,y,]==value, axis=1)
        
        #If they are equal and not already logged, make log them
        for j in range(len(same_value)):
            coord = np.array([x[j], y[j]]).reshape(1,-1)
            if same_value[j]:
                already_logged = np.any(np.all(coord==coords, axis=1))
                if not already_logged:
                    coords = np.concatenate([coords,coord], axis=0)
                    is_checked = np.concatenate([is_checked, np.array([0])])
        is_checked[i] = 1 #mark as checked
    return coords


def find_neighbor_coords(coords, lim=[512, 512]):
    tmp = [np.array([1,0]), np.array([-1,0]), np.array([0,1]), np.array([0,-1])]
    all_coords = np.concatenate([coords, coords+tmp[0], coords+tmp[1], coords+tmp[2], coords+tmp[3]])
    
    #Clip values outside the allowed limits 
    all_coords[:,0] = all_coords[:,0] % lim[0]
    all_coords[:,1] = all_coords[:,1] % lim[1]
    
    neigh_coords = np.unique(all_coords, axis=0) 
    return neigh_coords





#For all grain centers, find the matching coords and set those to the correct ID


a = np.zeros([512,512])-1
im = im5
for i, coord in enumerate(tqdm(centers)): 

    coords = find_value_region (im, coord)
    for _ in range(2): coords = find_neighbor_coords(coords)
    a[coords[:,0], coords[:,1]] = ids[i]







plt.imshow(a)
plt.imshow(tmp)


plt.imshow(a[160:250, 160:250])

plt.imshow(im[160:250, 160:250])



    




#now I could either do a mode smoothing just of -1
#or expand each grain by 1 nieghbor at a time



im = im3.transpose(2,0,1).reshape(1,3,512,512).copy()
ims = torch.from_numpy(im)
a = fs.my_unfoldNd(ims, kernel_size=3).reshape(9,3,-1)
imm = torch.mode(a, dim=0).values[0].reshape(512, 512)



torch.mode()












#raster through all the pixels, whichever grain center color it's closest to, assign that pixel to that value



####CONVERT unfoldND window sizes to tuple!!!!!!!!!!!



from scipy.spatial import distance 

im4 = np.zeros([512*512])
for i, e in enumerate(tqdm(im3.reshape(-1,3))): 
    im4[i] = int(np.argmin(distance.cdist(e.reshape(1,3),l)))



im5 = im4.reshape(512, 512)
plt.imshow(im5[210:230, 210:230])


import torch
ims = torch.from_numpy(im5).reshape(1,1,512,512)
a = fs.num_diff_neighbors(ims, window_size=3)




a = fs.my_unfoldNd(ims, kernel_size=5)
imm = torch.mode(a, dim=1).values[0].reshape(512, 512)

a = fs.my_unfoldNd(imm.reshape(1,1,512,512), kernel_size=3)
imm = torch.mode(a, dim=1).values[0].reshape(512, 512)



plt.imshow(imm)


torch.unique(imm).shape







#load, crop, and shape image



#assign ID's 



#go through each grain center and expand as far as it can be



























257



plt.imshow(b)













im = np.array(imageio.imread('Case3PeriodicUniqueGrains.0000.png'))
i = im[0,0,]


plt.imshow(im1)
im2 = im.reshape(-1,3)[im1.reshape(-1)==True,:]
side_length = int(np.sqrt(im2.shape[0]))
im3 = im2.reshape(side_length,side_length, 3)
pil_image=Image.fromarray(im3)
im4 = np.array(pil_image.resize((512,512)))
plt.imshow(im4)




print(im.shape)

plt.imshow(im)



im[0,0]

im1 = np.sum(im, axis=2)!=(255*3)

plt.imshow(im1)
im2 = im.reshape(-1,3)[im1.reshape(-1)==True,:]
side_length = int(np.sqrt(im2.shape[0]))
im3 = im2.reshape(side_length,side_length, 3)
pil_image=Image.fromarray(im3)
im4 = np.array(pil_image.resize((512,512)))
plt.imshow(im4)





plt.imshow(np_array[250:300,250:300])
plt.plot(np_array[:,-1,])
im5 = np.sum(im4.reshape(-1,3), axis=1)
im5 = im4.reshape(-1,3)
a, b = np.unique(im5, axis=0, return_counts=True)





c1 = a[b>18]


cc = np.all(im5==c1[13], axis=1).reshape(512,512)


plt.imshow(cc)

i = im[0,0]

ii = np.array(im==i, dtype='int8')

plt.imshow(im)



plt.imshow((im5==np.array([51, 65, 148])).reshape(512,512,3))



c1[13]








im5_1 = im5
a_1 = a


im5_2 = im5
a_2 = a




#I want all of the pixels to stay the same or go away
#are all of 2 found in 1

for a in a_2:
    print(np.sum(np.all(a==a_1, axis=1)))














im6 = np.zeros([512*512])
for i in tqdm(range(len(a))): 
    im6[np.all(im5==a[i], axis=1)==True] = i


plt.imshow(im6.reshape(512,512))


im6.shape



a1, b1 = np.unique(im6, return_counts=True)


plt.plot(b1, ',')
plt.ylim([0,1000])


np.sum(b1>=19)

np.sum(im6==0)



i=0
print(np.sum((im6==1)))
plt.imshow((im6==1).reshape(512,512))
i+=1

#those that are consistent from image to image
#those that are coniguous
#kernels?






with h5py.File('Case3_512p.hdf5', 'r') as f:
    a = f['images'][0,]




im7 = im6.reshape(512,512)




a1, b1 = np.unique(a, return_counts=True)
print(a1.shape)
plt.plot(b1,','); plt.show()

a2, b2 = np.unique(im7, return_counts=True)
print(a2.shape)
plt.plot(b2,','); plt.show()



c = np.where(b2>10)[0][0]


plt.imshow(im7==c)




#find the centroids using the normal IC
#register the colors in the blured Ic from the centroids
#register the other colors by what centroid they are closest to



np.where(a==0)[0].shape


c = 



a2, b2 = np.unique(im7, return_counts=True)


a2.shape
c1 = a2[b2>18]

plt.imshow(im7==c1[11])





































a2, b2 = np.unique(b1, return_counts=True)
e, _ = np.histogram(b1, bins=a1.shape[0])


plt.plot(b2, ',')

plt.xlim([0,30])
plt.xlim([0,300])



np.sum(np.all(im7, axis=1))

c, _ = np.histogram(im5, bins=1000)
plt.plot(c, ',')
plt.ylim([0,1])


np.sum(c>=1)


a, b = np.unique(im4.reshape(-1,3), axis=0, return_counts=True)


plt.plot(b,',')

np.sum(b>50)
np.histog


np.sum(im[400,:,0]!=82)
plt.imshow(im)


im1 = im.reshape(-1, 3)
a, b = np.unique(im1, axis=0, return_counts=True)


b[np.argmax(b)]=0








from skimage.filters import unsharp_mask
result_1 = unsharp_mask(im4, radius=1, amount=1)




plt.imshow(im4[250:300,250:300,])
plt.imshow(result_1[250:300,250:300,])



plt.imshow(im5, )


























    
    
def download_dropbox_file(access_token, file_from='/UFdata/SPPARKS/poly/dump_file/32c512grs512stsP.dump', file_to='32c512grs512stsP.dump', chunk_size=int(1e9)):
    #https://www.dropbox.com/developers/documentation/python#tutorial
    dbx = db.Dropbox(access_token)
    md, res = dbx.files_download(path=file_from)
    total_chunks = int(md.size/chunk_size)
    with open(file_to, 'wb') as f:
        for i, chunk in tqdm(enumerate(res.iter_content(chunk_size=chunk_size)), 'Writting to: %s'%file_to, total=total_chunks):
            f.write(chunk)

def upload_dropbox_file(access_token, file_from='32c512grs512stsP_new.hdf5', file_to='/UFdata/SPPARKS/poly/32c512grs512stsP_new.hdf5', chunk_size=int(1e8)):
    #https://www.dropbox.com/developers/documentation/python#tutorial
    dbx = db.Dropbox(access_token)
    file_size = os.path.getsize(file_from)
    num_it = int(np.ceil(file_size/chunk_size))
    with open(file_from, 'rb') as f:
        if file_size <= chunk_size: 
            dbx.files_upload(f.read(), file_to) #just upload the whole thing
        else:
            upload_session_start_result = dbx.files_upload_session_start(f.read(chunk_size))
            cursor = db.files.UploadSessionCursor(session_id=upload_session_start_result.session_id,offset=f.tell())
            commit = db.files.CommitInfo(path=file_to)
            for i in tqdm(range(num_it-2), 'Writting to: %s'%file_to):
                dbx.files_upload_session_append(f.read(chunk_size),cursor.session_id,cursor.offset)
                cursor.offset = f.tell()
            dbx.files_upload_session_finish(f.read(chunk_size),cursor,commit)

#also copy over all of the functions being used below to the main sPkY





#'/UFdata/SPPARKS/circle/dump_file/32cCase1_1.dump'
# '/UFdata/SPPARKS/circle/dump_file/32cCase1_2.dump'
'/UFdata/SPPARKS/circle/dump_file/32cCase1_3.dump'
'/UFdata/SPPARKS/circle/dump_file/32cCase1_4.dump'

'/UFdata/SPPARKS/hex/dump_file/32c64grs443sts.dump'





#Setup paths
file_from = '/UFdata/SPPARKS/circle/dump_file/32cCase1_2.dump'
file_to = file_from.split('/')[-1] #'32c512grs512stsNP.dump'

#download file
access_token='xZx9HlRyOZ0AAAAAAAAAAQjGulMiaUbhK5w8sMMtNZ0uQrY-ouWlQkqjb-bFgj4b' #access token
download_dropbox_file(access_token, file_from, file_to)

#convert to hdf5
path_to_hdf5 = file_to.split('.')[0]+".hdf5"
var_names, time_steps, bounds = fs.dump_to_hdf5(path_to_dump=file_to, path_to_hdf5=path_to_hdf5, num_steps=5000)

#convert to images
new_path = file_to.split('.')[0]+"_new.hdf5"
fs.dump_extract_to_images(path_to_hdf5=path_to_hdf5, new_path=new_path, vi=1, xi=2, yi=3)

#validate
with h5py.File(new_path, 'r') as f:
    print(f.keys())
    print(f['images'].shape)
    a = f['images'][:]
print(np.sum(a[1000,]==0))
print(a[100,0,0])
plt.imshow(a[2000]); plt.show()

#upload to dropbox
# file_to2 = file_from.split('.')[0].replace('/dump_file', '') + "_new.hdf5"
file_to2 = file_from.replace('/dump_file', '').replace('.dump', "_new.hdf5")
upload_dropbox_file(access_token, file_from=new_path, file_to=file_to2, chunk_size=int(1e8))
    
#delete files
os.system("del %s"%file_to) 
os.system("del %s"%path_to_hdf5) 
os.system("del %s"%new_path) 



#redownload and check
download_dropbox_file(access_token, file_to2, new_path)

with h5py.File(new_path, 'r') as f:
    print(f.keys())
    print(f['images'].shape)
    a = f['images'][:]

print(np.sum(a[1000,]==0))
print(a[100,0,0])
plt.imshow(a[2000]); plt.show()

os.system("del %s"%new_path) 


































a = np.loadtxt("dataset_0000002.csv", dtype='int', delimiter=',', skiprows=1)
b = np.loadtxt("dataset_000000.csv", dtype='int', delimiter=',', skiprows=1)


a3 = a[:,3] #a3 is ids
a5 = a[:,5] #a5 is unique ids
b0 = b[:,0] #b0 is x
b1 = b[:,1] #b1 is y
b16 = b[:,-1] #b16 is ids


c = np.zeros([513,513])

for i in tqdm(range(len(a3))):
    x = b0[b16==a3[i]]
    y = b1[b16==a3[i]]
    c[x, y] = a5[i]
    
    

plt.imshow(c)




bb = np.diff(np.sort(x+y*513))
plt.plot(bb)

l = []
for i in tqdm(range(len(a3))):
    l.append(np.where(b16==a3[i])[0])
t = np.array(l)


x = b0[t][:,0]
y = b1[t][:,0]
aa = a5[t]

for i in range(len(t)):
    c[x[i], y[i]] = aa[i]



l = []
for i in tqdm(range(len(a3))):
    l.append(np.where(b16==a3[i])[0])
t = np.array(l)




aa = a5[t]

plt.plot(np.diff(np.unique(x)))


plt.plot(x,y,',')

plt.plot(y)


plt.plot(aa)

plt.plot(b16)
















#validate
with h5py.File('Case3_512p.hdf5', 'r') as f:
    print(f.keys())
    print(f['images'].shape)
    a = f['images'][:]

print(np.sum(a[1000,]==0))
print(a[100,0,0])
plt.imshow(a[0]); plt.show()


im = a[0,]



a1, b1 = np.unique(a5, return_counts=True)
a2, b2 = np.unique(im, return_counts=True)



np.sum(b1==b2[2])


a, b = np.sort(abs(b1-b2[0]))


a = np.diff(a5[:512])>0
b = np.diff(im[-1,:])>0

plt.plot(a)
plt.plot(np.flip(b))
plt.show()




np.min(im)













