# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 11:02:53 2018

@author: aims
"""
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def file_name(file_dir):   
    root_tmp=[]
    dirs_tmp=[]
    files_tmp=[]
    for root, dirs, files in os.walk(file_dir):  
        root_tmp.append(root)
        dirs_tmp.append(dirs)
        files_tmp.append(files)
    
    train_set_name = []
    fault_set_name = []
    ttf_set_name = []
    for i in range(20):
        train_set_name.append(root_tmp[1]+'/'+files_tmp[0][i])
        fault_set_name.append(root_tmp[1]+'/'+files_tmp[1][i])
        ttf_set_name.append(root_tmp[1]+'/'+files_tmp[1][i])
        
    return  train_set_name,fault_set_name, ttf_set_name, files_tmp
if __name__ == '__main__':
    path = '/home/ninja1mmm/Desktop/phm'
    train_set_name,fault_set_name, ttf_set_name, files_tmp = file_name(path)
    root, dirs, files = file_name(path)
    
    for i in range(20):
        train_set = pd.read_csv(train_set_name[i])
        fault_set = pd.read_csv(fault_set_name[i])
        train_set.to_pickle(files_tmp[0][i][0:-4])
        fault_set.to_pickle(files_tmp[1][i][0:-4])        
        
        
    
    