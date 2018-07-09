# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 09:08:41 2018

@author: aims
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

def file_name(file_dir):   
    root_tmp=[]
    dirs_tmp=[]
    files_tmp=[]
    for root, dirs, files in os.walk(file_dir):  
        root_tmp.append(root)
        dirs_tmp.append(dirs)
        files_tmp.append(files)        
    return root_tmp, dirs_tmp, files_tmp



path = 'D:/AIMS/phm_data_challenge_2018/data'
root_tmp, dirs_tmp, files_tmp = file_name(path)

fault_files = [root_tmp[1] +'/'+ file for file in files_tmp[1]]
train_files = [root_tmp[2] +'/'+ file for file in files_tmp[2]]
ttf_files = [root_tmp[3] +'/'+ file for file in files_tmp[3]]

df = pd.read_pickle('data/train/01_M01_DC_train')
fault = pd.read_pickle('data/fault/01_M01_train_fault_data')

df=df.replace([np.inf, -np.inf], np.nan).dropna()
df=df.reset_index(drop=True)

idx = []
for time_current in fault.time:
    idx_tmp = np.argmax(df.time[(time_current-df.time)>=0])
    idx.append(idx_tmp)    
    
df0 = df[(df.stage==0) & (df.recipe==0)]

rms_parameter = {}
for col in df0.columns[7:16]:
    counter = 0
    flag = 0
    rms = pd.DataFrame(columns = ['value','start','end'])
    for tmp in df0[col].index:
        if (flag == 0) & (df0['ETCHPBNGASREADBACK'][tmp] > -1):
            counter += 1
            flag = 1
            tmp_start = tmp
        if (flag == 1) & (df0['ETCHPBNGASREADBACK'][tmp] < -1):
            flag = 0
            tmp_end = tmp
            tmp_period = df0[col].loc[tmp_start:tmp_end]
            rms_tmp = np.sqrt(np.mean(tmp_period**2))
            rms.loc[counter-1] = [rms_tmp, tmp_start, tmp_end]
    rms_parameter[col] = rms
    print(counter)
    
# =============================================================================
# # Seperate with recipe_step
# rms_parameter = {}
# for col in df0.columns[7:16]:
#     counter = 0
#     flag = 0
#     rms = []
#     for tmp in df0[col].index:
#         if (flag == 0) & (df0['recipe_step'][tmp] == 2):
#             counter += 1
#             flag = 1
#             tmp_start = tmp
#         if (flag == 1) & (df0['recipe_step'][tmp] == 1):
#             flag = 0
#             tmp_end = tmp
#             tmp_period = df0[col][tmp_start:tmp_end]
#             rms_tmp = np.sqrt(np.mean(tmp_period**2))
#             rms.append(rms_tmp)
#     rms_parameter[col] = rms
#     print(counter)
# =============================================================================
# =============================================================================
# unit_name = []
# for i in range(20):
#     unit_name.append(files_tmp[1][i][0:6])
# 
# fault_name_1 = 'FlowCool Pressure Dropped Below Limit'
# fault_FPDBL = {}
# fault_FPDBL_time = []
# 
# for fault_file, train_file, ttf_file, unit in zip(fault_files, train_files, ttf_files, unit_name):
#     df = pd.read_pickle(train_file)
#     fault = pd.read_pickle(fault_file)
#     ttf = pd.read_pickle(ttf_file)
#     
#     df=df.replace([np.inf, -np.inf], np.nan).dropna()
#     df=df.reset_index(drop=True)
# 
#     idx = []
#     FPDBL = fault[fault.fault_name == fault_name_1]
#     FPDBL = FPDBL.reset_index(drop=True)
#     for time_current in FPDBL.time:
#         idx_tmp = np.argmax(df.time[(time_current-df.time)>=0])
#         idx.append(idx_tmp)    
#         
#     fault_FPDBL[unit+'_idx'] = idx
#     for i in range(len(idx)):
#         if i == 0:
#             time_tmp = FPDBL.time[i] - df.time[0]
#         else:
#             time_tmp = FPDBL.time[i] - FPDBL.time[i-1]
#         fault_FPDBL_time.append(time_tmp)
# 
# =============================================================================

#
#plt.figure()
#plt.title('All parameter straight plot')
#for i in range(1,18):
#    ax = plt.subplot(6,3,i)
#    ax.set_title(df0.columns[i+6])
#    plt.plot(df0.time[:10000], df0.iloc[:10000,i+6])
#     
plt.figure()
plt.title('All parameter straight plot')
for i in range(1,10):
    ax = plt.subplot(3,3,i)
    var = df.columns[i+6]
    ax.set_title(var)
    plt.plot(df0.loc[rms_parameter[var].end].time, rms_parameter[var].value,'r')
    for i in idx:
        plt.axvline(df.time[i])
# 
# =============================================================================
var = df.columns[1+6]
plt.figure()
plt.title(var)
ax.set_title(var)
plt.plot(df0.loc[rms_parameter[var].end].time, rms_parameter[var].value,'r')
for i in idx:
    plt.axvline(df.time[i])

