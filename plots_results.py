# -*- coding: utf-8 -*-
"""
@author: Joseph Melville
Code for SmallSat 2022 article "Methods for Data-centric Small Satellite Anomaly Detection and Fault Prediction"
Section: "RESULTS"
Runtime: ~10 min
"""



### IMPORT LIBRARIES 
import functions as sf
import functions_dataset as sfd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



### SETUP
results1 = np.load('results/results_reboot.npy')
results2 = np.load('results/results_sunsafe.npy')
results3 = np.load('results/results_momentum.npy')
results = [results1, results2, results3]

t_new = sf.time_arr(date_start='2020-02-01 00:00:00', date_end='2020-09-28 01:00:00', dt=60) #seconds
faults = sfd.load_obj('./dataset/faults')
outliers = sfd.load_obj('./dataset/oultiers')



### Figure 10: Histograms of F1 Scores
plt.rcParams.update({'font.size': 10})
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(3.25, 6), dpi=300)

edges = np.linspace(0,1,10)

plt.subplot(3,1,1)
plt.hist(results[0][:,:,6], bins=edges)
plt.title('Reboot')
plt.ylabel('Number in bin')
plt.xlim(0,1)
plt.ylim(0,65)
plt.xticks([])
plt.legend(['Blip','IF','IF-PCA','LOF'])

plt.subplot(3,1,2)
plt.hist(results[1][:,:,6], bins=edges)
plt.title('Demote to SunSafe')
plt.ylabel('Number in bin')
plt.xlim(0,1)
plt.ylim(0,65)
plt.xticks([])

plt.subplot(3,1,3)
plt.hist(results[2][:,:,6], bins=edges)
plt.title('Momentum Error')
plt.ylabel('Number in bin')
plt.xlim(0,1)
plt.ylim(0,65)
plt.xlabel('Area Under ROC Curve')

# plt.legend(['Blip','IF','IF-PCA','LOF'], bbox_to_anchor=(0, -0.8, 1, 1), loc='lower center', ncol=2)
plt.savefig('results/Fig10_Histograms of AUC Values.png', bbox_inches='tight', dpi=300)
    
    
    
### Table 1: Top five predictability values for each label
#Export top 10 results from each label type to create a table (trimmed down to 5 each in table to remove redundancy)
with pd.HDFStore('./dataset/telemetry_data_pd') as df: pks = list(df.keys()) #for extracting the packet name used
ll = ['Blip', 'IF', 'IF-PCA', 'LOF'] #for extracting the outlier method used

for k, r in enumerate(results): 

    r = results[k]
    l = []
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            tmp = np.concatenate([r[i,j], np.array([i,j])])
            l.append(tmp)
    r = np.stack(l)
    r1 = pd.DataFrame(r).sort_values(by=6, ascending=False, na_position='last').values
    r2 = pd.DataFrame(r).sort_values(by=4, ascending=False, na_position='last').values
    
    methods = list(np.array(ll)[r1[:10,22].astype(int)]) + list(np.array(ll)[r2[:10,22].astype(int)])
    items = list(np.array(pks)[r1[:10,21].astype(int)]) + list(np.array(pks)[r2[:10,21].astype(int)])
    items = [m.replace('/', '') for m in items]
    auc = list(r1[:10,6]) + list(r2[:10,6])
    f1 = list(r1[:10,4]) + list(r2[:10,4])
    prec = list(r1[:10,10]) + list(r2[:10,10])
    rec = list(r1[:10,11]) + list(r2[:10,11])
    tp_mean = list(r1[:10,13]) + list(r2[:10,13])
    tp_std = list(r1[:10,14]) + list(r2[:10,14])
   
    df = pd.DataFrame(list(zip(methods, items, auc, prec, rec, tp_mean, tp_std))) #place in panda dateframe
    df.to_csv('results/statistics%d.csv'%(k+1), index=False) #out1, out2, out3 #save

    

### Figure 11: Probability of fault per timestep

#Reboot label best F1
t_faults = faults['reboot']
t_outliers = outliers['/xact_app_module_attitude_overview/blip']
x1, c1, frs1 = sf.predictability_metrics(t_faults, t_outliers, t_new, run_type=1)

#Demote to sunsafe label best F1
t_faults = faults['demote_to_sunsafe']
t_outliers = outliers['/beacon_aggregator_beacon_status_set/blip']
x2, c2, frs2 = sf.predictability_metrics(t_faults, t_outliers, t_new, run_type=1)

#Momentum error label best F1
t_faults = faults['momentum_errors']
t_outliers = outliers['/xact_app_module_sensor_detailed_cosmos/LOF']
x3, c3, frs3 = sf.predictability_metrics(t_faults, t_outliers, t_new, run_type=1)

#Plot 
plt.rcParams.update({'font.size': 10})
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(3.25, 6), dpi=300)

plt.subplot(3,1,1)
plt.plot(x1[frs1], c1[frs1], '.', color='C1')
plt.plot(x1, c1, color='C0')
plt.plot(x1[frs1], c1[frs1], '.', color='C1')
plt.title('Reboot')
plt.ylabel('Probability of fault')
plt.xticks([])
plt.legend(['Fault'])

plt.subplot(3,1,2)
plt.plot(x2[frs2], c2[frs2], '.', color='C1')
plt.plot(x2, c2, color='C0')
plt.plot(x2[frs2], c2[frs2], '.', color='C1')
plt.title('Demote to SunSafe')
plt.ylabel('Probability of fault')
plt.xticks([])

plt.subplot(3,1,3)
plt.plot(x3[frs3], c3[frs3], '.', color='C1')
plt.plot(x3, c3, color='C0')
plt.plot(x3[frs3], c3[frs3], '.', color='C1')
plt.title('Momentum Error')
plt.xlabel('Weeks')
plt.ylabel('Probability of fault')
plt.savefig('results/Fig11_Probability of fault per timestep.png', bbox_inches='tight', dpi=300)



### Figure 12: Relationship between STD and AUC
num_entries = np.load('dataset/num_entires.npy')
ii = np.where(np.array(num_entries)>10000)[0] #cut out small packets to limit noise 
r = results[2]

plt.rcParams.update({'font.size': 10})
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(3.25, 2), dpi=300)
plt.plot(r[:,:,6], r[:,:,4], '.')
plt.ylabel('F1 Score')
plt.xlabel('Area Under ROC Curve')
plt.xlim([0.48,1])
plt.ylim([0.64, 1])
plt.legend(['Blip', 'IF','IF-PCA','LOF'], bbox_to_anchor=(0, -0.7, 1, 1), loc='lower center', ncol=2)
plt.savefig('results/Fig12_Relationship between AUC and F1.png', bbox_inches='tight', dpi=300)



# ### For text: Compare predictability given a range a sampling rates
# #Takes 8 minutes to run
# t_faults = faults['momentum_errors']
# dts = [10, 30, 60, 120, 300, 600, 1200, 2400] #seconds

# tmp0 = []
# for dt in tqdm(dts):
#     t_new = sf.time_arr(date_start='2020-02-01 00:00:00', date_end='2020-09-28 01:00:00', dt=dt) #seconds
#     tmp1 = []
#     for om in ['blip', 'IF', 'IF-PCA', 'LOF']:
#         t_outliers = outliers['/xact_app_module_attitude_overview_cosmos/'+om]
#         pred = sf.predictability_metrics(t_faults, t_outliers, t_new)
#         tmp1.append(pred)
#     tmp0.append(np.array(tmp1))
# tmp = np.array(tmp0)

# # #AUC
# # tmp[:,:,6] = np.array([[0.44935186, 0.53871159, 0.76585314, 0.65919154],
# #        [0.44928649, 0.54318892, 0.76517215, 0.80050925],
# #        [0.44809717, 0.5482041 , 0.76338663, 0.88843624],
# #        [0.44886542, 0.55793356, 0.76735637, 0.94891164],
# #        [0.44979482, 0.58268705, 0.77623517, 0.97324052],
# #        [0.4501609 , 0.61675968, 0.78636824, 0.97473059],
# #        [0.45131794, 0.66828479, 0.80256425, 0.97515551],
# #        [0.45297889, 0.73767445, 0.82593917, 0.97537838]])

# # #F1 score
# # tmp[:,:,4] = np.array([[0.63164188, 0.63164188, 0.69575323, 0.63164188],
# #        [0.64089309, 0.64089309, 0.69474032, 0.75201414],
# #        [0.65230455, 0.65230455, 0.69189181, 0.87513189],
# #        [0.65780419, 0.65780419, 0.69871452, 0.94643419],
# #        [0.66678882, 0.66678882, 0.71368557, 0.97272915],
# #        [0.67676168, 0.67676168, 0.73031299, 0.9744638 ],
# #        [0.6919729 , 0.6919729 , 0.75609096, 0.97518362],
# #        [0.71537947, 0.71537947, 0.79145321, 0.97578513]])