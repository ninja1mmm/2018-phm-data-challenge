#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 13:37:10 2018

@author: ninja1mmm
"""
import os
import numpy as np
import pandas as pd
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
        
        
root = '/home/ninja1mmm/Desktop/phm/data'
root_tmp, dirs_tmp, files_tmp = file_name(root)

combined_all = {}
feature_all = pd.DataFrame(columns = ['mean', 'std','root amplitude',
                                      'rms','max','skewness','kurtosis',
                                      'peak factor','margin','waveform',
                                      'pulse','start_time', 'end_time',
                                      'recipe', 'stage', 'Lot'])
#df_check = pd.DataFrame()

# read the first file to test here
file_tmp = files_tmp[2][0]
# iterate through the files if needed
#for file_tmp in files_tmp[2]:
    
path_tmp = root_tmp[2]+'/'+file_tmp
df = pd.read_pickle(path_tmp)
#df_tmp = df[df['Lot']==28113]
#if len(df_tmp)>0:
#    df_tmp = df_tmp.iloc[0,:]
#    df_check = df_check.append(df_tmp)
    #------------------------------------------------------------------------------
# Crucial step
df=df.replace([np.inf, -np.inf], np.nan).dropna()
df=df.reset_index(drop=True)
df_scaler = preprocessing.MinMaxScaler(feature_range = (0,1))

#------------------------------------------------------------------------------
    



lot_list = list(set(df.Lot))
# Check if Lot already existed
for key in lot_list:
    if key in combined_all.keys():
        print('The Lot %d in %s already existed in %s' % (key, file_tmp, 
                                                          combined_all[key]))
        
#    for tmp in lot_list:
#        combined_all[tmp] = file_tmp
# Select and save all the wafer processing cycles
list_tmp = []
lot_last = df.Lot[0]
counter = 0
idx = 0
# Specify the range. Here set to 100000 for the ease of test
for row_tmp in df.index:
    lot_tmp = df.iloc[row_tmp,:].Lot
    if lot_tmp == lot_last:
        list_tmp.append(df.iloc[row_tmp,:])
        counter += 1
    else:
        df_tmp = pd.concat(list_tmp, axis = 1)
        # lot_last serves as the key, can be changed 
#        combined_all[lot_last] = df_tmp.T
        combined_all[df_tmp.T.time.iloc[-1]] = df_tmp.T
        # Calculate mean and save in feature dictionary as an example
        # Normalize the data again because for some parameters we need the local (within cycle) feature
        feature_tmp = df_tmp.T.iloc[:,7:] # Not a correct way, because shutter position also need to be excluded
        feature_tmp = df_scaler.fit_transform(feature_tmp)
# ------------------------------------------------------------------
        # Add features here. Remember to add new columns when initialzing df
        t1 = np.mean(feature_tmp) 
        t2 = np.std(feature_tmp)
        t3 = np.mean(np.sqrt(np.abs(feature_tmp)))**2
        t4 = np.sqrt(np.mean(feature_tmp**2))
        t5 = np.max(feature_tmp)
        t6 = np.sum((feature_tmp-t1)**3)/((len(feature_tmp)-1)*(t2**3))
        t7 = np.sum((feature_tmp-t1)**4)/((len(feature_tmp)-1)*(t2**4))
        t8 = t5/t4
        t9 = t5/t3
        t10 = t4/(np.sum(np.abs(feature_tmp))/len(feature_tmp))
        t11 = t5/(np.sum(np.abs(feature_tmp))/(len(feature_tmp)))
        # Newly added
        
        
        # First order difference
        
# ---------------------------------------------------------------------
        feature_all.loc[idx,:] = [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,
                                   df_tmp.T.time.iloc[0],df_tmp.T.time.iloc[-1],
                                   df_tmp.T.recipe.iloc[0],df_tmp.T.stage.iloc[0],
                                   lot_last]
        
        list_tmp = []
        idx += 1
        counter = 0
    lot_last = lot_tmp
    print(row_tmp)
    
    
    
#------------------------------------------------------------------------------




