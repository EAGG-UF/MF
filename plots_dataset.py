# -*- coding: utf-8 -*-
"""
@author: Joseph Melville
Code for SmallSat 2022 article "Methods for Data-centric Small Satellite Anomaly Detection and Fault Prediction"
Section: "DATASET"
Runtime: ~2 min
"""



### IMPORT LIBRARIES 
import functions as sf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



### Statistics for written portion
#Open the Pandas dataframes and find the mean and STD of the data rates
with pd.HDFStore('./dataset/telemetry_data_pd') as f:
    
    #How many packets are there? Save their names.
    packet_names = list(f.keys())
    print("Number of packets: %d"%len(packet_names))
    
    #How many items does each have? And how many time steps each
    #Also gather utc_times and sample rates
    num_items = []
    num_entries = []
    utc_times = []
    rates = []
    for p in tqdm(packet_names): #for each packet
        num_entry, num_item = f[p].shape  #find the number of items and entries
        num_items.append(num_item)
        num_entries.append(num_entry)
        d = f[p]['utc_time'].values 
        
        #Remove all entries outside a two year time frame
        t1 = sf.form_to_unix('2019-01-01 00:00:00')
        t2 = sf.form_to_unix('2021-01-01 00:00:00')
        d = d[np.logical_and(d>t1, d<t2)]
        
        
        utc_times.append(d)
        
        if len(d)>0:
            #Count the frequency of each utc_time
            a = np.sort(d)
            a_set = np.sort(np.array(list(set(d))))
            edges = np.hstack([a_set[0]-0.5, a_set+0.5])
            h, _ = np.histogram(a, bins=edges)
    
            #Step through h and add rates one at a time
            r = []
            for i in range(len(a_set)-1):
                v1 = a_set[i]
                dv = 1/h[i]
                for j in range(h[i]):
                    if j+1==h[i]: v2 = a_set[i+1]
                    else: v2 = a_set[i] + dv*(j+1)
                    r.append(1/(v2-v1))
                    v1 = v2
            rates.append(r)
        else: rates.append([])
     
np.save('dataset/num_items.npy', num_items)
np.save('dataset/num_entires.npy', num_entries)

# print("Range of items per packet: %d-%d"%(np.min(num_items), np.max(num_items)))
print("Mean items per packet: %f"%(np.mean(num_items)))
print("Median items per packet: %d"%(np.median(num_items)))
# print("Range of entries per packet: %d-%d"%(np.min(num_entries), np.max(num_entries)))
print("Mean entries per packet: %f"%(np.mean(num_entries)))
print("Median entries per packet: %d"%(np.median(num_entries)))

total_points = np.sum(np.array(num_items)*np.array(num_entries))
print("Total points: %d\nTotal memory: 3.9 GB"%total_points)

r = np.hstack(rates)
print("Mean sample rate: %f"%(np.mean(r)))
print("Median sample rate: %f"%(np.median(r)))
    


### Figure 1: Telemetry UTC Histogram
ut = np.hstack(utc_times)
t1 = sf.form_to_unix('2020-01-01 00:00:00')
t2 = sf.form_to_unix('2020-10-01 00:00:00')
edges = np.linspace(t1,t2,9*4)
hh, _ = np.histogram(ut, bins=edges)

plt.rcParams.update({'font.size': 10})
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(3.25, 2), dpi=300)
plt.bar(np.arange(0,35), hh/np.sum(hh))
plt.xlabel('Weeks')
plt.ylabel('% of total packets')
plt.savefig('dataset/Fig1_Telemetry UTC Histogram.png', bbox_inches='tight', dpi=300)



### Figure 2: Item Statistics
with pd.HDFStore('./dataset/telemetry_data_pd') as f:
    p = 'xact_app_module_attitude_overview_cosmos'
    item_names = list(f[p].keys())
    num_entries = len(f[p][item_names[0]])
    print('Packet: %s'%p)
    print('Number of items: %d'%len(item_names))
    print('Number of entries: %d'%num_entries)
    
    # item_range = []
    item_stats = []
    
    for i in item_names:
        v = f[p][i].values
        # item_range.append([np.max(v), np.min(v)])
        iqr = np.percentile(v,0.75)-np.percentile(v,0.25)
        tmp = np.array([np.mean(v), np.std(v), np.median(v), iqr])
        item_stats.append(tmp)
        
a3 = np.array(item_stats)

plt.rcParams.update({'font.size': 10})
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(3.25, 2), dpi=300)

#Plot mean/median log10 scale
bbb = np.log10(np.abs(a3))
bbb[a3==0] = 0
plt.plot(bbb[:,0], '-')
plt.plot(bbb[:,2], '-')

#Plot STD/IQR log10 scale
plt.plot(bbb[:,1], '.c')
plt.plot(bbb[:,3], '.m')
plt.xticks([])
plt.ylim(-5, 10)
plt.legend(['Mean','Median','STD','IQR'], bbox_to_anchor=(0, -0.5, 1, 1), loc='lower center', ncol=2)
plt.ylabel('Absolute Magnitude (log10)')
plt.xlabel('Items (30 total)')
plt.savefig('dataset/Fig2_Item Statistics.png', bbox_inches='tight', dpi=300)



### Figure 3: Item Correlation Matrix
with pd.HDFStore('./dataset/telemetry_data_pd') as f:
    p = 'xact_app_module_attitude_overview_cosmos'
    c = np.corrcoef(f[p].values.T)
cc = np.sort(c[np.isnan(c)==False])[:-30] #remove nan values and the 30 diagonal ones
rms = np.sqrt(np.mean(cc**2)) #average absolute magnitude or item pair correlations
print('RMS: %1.3f'%rms)

plt.rcParams.update({'font.size': 10})
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(3.25, 2), dpi=300)
plt.imshow(c)
plt.colorbar()
plt.yticks([])
plt.xticks([])
plt.clim([-1,1])
plt.savefig('dataset/Fig3_Item Correlation Matrix.png', bbox_inches='tight', dpi=300)



### Figure 4: Principal Component Analysis (PCA)
with pd.HDFStore('./dataset/telemetry_data_pd') as f:
    p = 'xact_app_module_attitude_overview_cosmos'
    vs = f[p].values

#Find eigen-decomposition of coariance matrix
import numpy as np
from sklearn.decomposition import PCA
vv = 1e10
X = np.clip(vs, -vv, vv)
pca = PCA(n_components=X.shape[1])
y = pca.fit_transform(X)

#Find condition numbers when keeping different numbers of eigenvectors
eig_vals = pca.explained_variance_
cond_nums = []
for e in eig_vals:
    cond_nums.append(eig_vals[0]/e)

plt.rcParams.update({'font.size': 10})
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(3.25, 2), dpi=300)
plt.bar(np.arange(30)+1, np.log10(cond_nums), color='C0')
plt.ylabel('Covariance Eigenvalue Ratios', color='C1', fontweight='bold')
plt.xlabel('Covariance Eigenvectors')
plt.twinx()
plt.bar(np.arange(30)+1, pca.explained_variance_ratio_, color='C1')
plt.ylabel('Condition Number (log10)', color='C0', fontweight='bold')
plt.xlim([0.3,30.7])
plt.savefig('dataset/Fig4_Principal Component Analysis.png', bbox_inches='tight', dpi=300)



### Figure 5: Item Vizualization
with pd.HDFStore('./dataset/telemetry_data_pd') as f:
    p = 'xact_app_module_attitude_overview_cosmos'
    i = 'wheel_speed_rpm0'
    v = f[p][i].values
    t = f[p]['utc_time'].values

plt.rcParams.update({'font.size': 10})
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(3.25, 4), dpi=300)

#Plot the 
plt.subplot(2,1,1)
[t, v] = sf.my_sort([t, v])
plt.plot(t,v, '.C0') #sorted
plt.xticks([])

#Plot vertical lines ot inticate the cut timeframe
ii = 1920950/2 + 10000
i = np.arange(ii,ii+1000).astype(int)
plt.axvline(t[i[0]], color='C1')
plt.axvline(t[i[-1]], color='C1')

#Plot the cutout timeframe
plt.subplot(2,1,2)
plt.plot(t[i],v[i], '.C1') #zoomed in
plt.xticks([])
plt.xlabel('UTC timestamps')
plt.savefig('dataset/Fig5_Item Vizualization.png', bbox_inches='tight', dpi=300)