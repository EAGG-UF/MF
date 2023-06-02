# -*- coding: utf-8 -*-
"""
@author: Joseph Melville
Code for SmallSat 2022 article "Methods for Data-centric Small Satellite Anomaly Detection and Fault Prediction"
Section: "METHODS"
Runtime: ~10 sec
"""



### IMPORT LIBRARIES 
import functions as sf
import functions_dataset as sfd
import numpy as np
import matplotlib.pyplot as plt



### Setup
t_new = sf.time_arr(date_start='2020-02-01 00:00:00', date_end='2020-09-28 01:00:00', dt=60) #seconds
faults = sfd.load_obj('./dataset/faults')
outliers = sfd.load_obj('./dataset/oultiers')
t_faults = faults['reboot']
t_outliers = outliers['/xact_app_module_attitude_overview/blip']


#Calculate outliers, and faults and results at the end of the dataset creation!!!


### Figure 6: Seconds to reboot 
x, faults_rs_dist, outliers_rs2  = sf.predictability_metrics(t_faults, t_outliers, t_new, run_type=2)

plt.rcParams.update({'font.size': 10})
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(3.25, 2), dpi=300)
plt.plot(x[outliers_rs2], faults_rs_dist[outliers_rs2], '.', color='C1')
plt.plot(x, faults_rs_dist, color='C0')
plt.plot(x[outliers_rs2], faults_rs_dist[outliers_rs2], '.', color='C1')
plt.legend(['Outliers'], loc=1)
plt.xlabel('Weeks')
plt.ylabel('Seconds to reboot')
plt.savefig('methods/Fig6_Seconds to reboot.png', bbox_inches='tight', dpi=300)



### Figure 7: Outlier-to-fault and fault-to-oultier densities
x, o2f, f2o, outlier_to_fault, fault_to_outlier  = sf.predictability_metrics(t_faults, t_outliers, t_new, run_type=3)
   
plt.rcParams.update({'font.size': 10})
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(3.25, 2), dpi=300)
plt.plot(x, np.cumsum(o2f), color='C0')
plt.plot(x, np.cumsum(f2o), '--', color='C1')
plt.xlabel('Hours')
plt.ylabel('Distibution')
plt.legend(['outlier-to-fault CDF','fault-to-outlier CDF'])
plt.savefig('methods/Fig7_Outlier-to-fault and fault-from-outlier CDF.png', bbox_inches='tight', dpi=300)

print('Hours to fault given an outlier: %1.1f (mean), %1.1f (STD)'%(np.mean(outlier_to_fault)/60/60, np.std(outlier_to_fault)/60/60))
print('Hours from outlier given a fault: %1.1f (mean), %1.1f (STD)'%(np.mean(fault_to_outlier)/60/60, np.std(fault_to_outlier)/60/60))



### Figure 8: Probability of fault per time step
x, c, faults_rs = sf.predictability_metrics(t_faults, t_outliers, t_new, run_type=1)

plt.rcParams.update({'font.size': 10})
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(3.25, 2), dpi=300)
plt.plot(x[faults_rs], c[faults_rs], '.', color='C1')
plt.plot(x, c, color='C0')
plt.plot(x[faults_rs], c[faults_rs], '.', color='C1')
plt.xlabel('Weeks')
plt.ylabel('Probability of fault')
plt.legend(['Reboot'])
plt.savefig('methods/Fig8_Probability of fault per time step.png', bbox_inches='tight', dpi=300)



#Figure 9: Fault classification ROC curve
fprs, tprs = sf.predictability_metrics(t_faults, t_outliers, t_new, run_type=4)

plt.rcParams.update({'font.size': 10})
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(3.25, 2), dpi=300)
plt.plot(fprs, tprs, '.-', color='C0')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('methods/Fig9_Fault classification ROC curve.png', bbox_inches='tight', dpi=300)

p = sf.predictability_metrics(t_faults, t_outliers, t_new, run_type=0)
print('Area under curve (AUC): %1.2f'%p[6])
print('Highest F1 value: %1.4f'%p[4])
print('Best threshold for F1: %1.2f'%p[5])