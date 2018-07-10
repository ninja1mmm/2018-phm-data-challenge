# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import math
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn import svm
from keras import backend as K

def Error(y_pred, y_real):
    y_pred = np.nan_to_num(y_pred, copy = True)
    y_real = np.nan_to_num(y_real, copy = True)
    temp = np.exp(-0.001 * y_real) * np.abs(y_real - y_pred)
    error = np.sum(temp)
    return error

def customLoss(y_pred, y_real):
    return K.sum(K.exp(-0.001 * y_real) * K.abs(y_real - y_pred))
    
#------------------------------------------------------------------------------
# Read in Data
sensor_data = pd.read_csv('C:\\Users\\chen.zc\\Desktop\\phm_data_challenge_2018.tar\\train\\01_M01_DC_train.csv')
faults_data = pd.read_csv('C:\\Users\\chen.zc\\Desktop\\phm_data_challenge_2018.tar\\train\\train_faults\\01_M01_train_fault_data.csv')
ttf_data = pd.read_csv('C:\\Users\\chen.zc\\Desktop\\phm_data_challenge_2018.tar\\train\\train_ttf\\01_M01_DC_train.csv')

sensor_data = sensor_data.drop(['Tool'], axis = 1)
sensor_data = sensor_data.drop(['Lot'], axis = 1)

# =============================================================================
# sensor_data = sensor_data.loc[sensor_data.index %10 == 0]
# ttf_data = ttf_data.loc[ttf_data.index %10 == 0]
# =============================================================================
sensor_data.index = range(0,len(sensor_data))
ttf_data.index = range(0,len(ttf_data))

def cutoff(sensor_data, faults_data, ttf_data, column):
    # cut off the tail of the data set that with NaN ttf
    temp = faults_data[faults_data['fault_name'] == column]
    last_failure = temp['time'].values[-1]
    array = np.asarray(sensor_data['time'])
    closest_ind = (np.abs(array - last_failure)).argmin()
    if ((array[closest_ind] - last_failure) != np.abs(array[closest_ind] - last_failure)):
        ind = closest_ind + 1
    elif ((array[closest_ind] - last_failure) == 0):
        ind = closest_ind + 1
    else:
        ind = closest_ind
    sensor_data = sensor_data[:ind]
    ttf_data = ttf_data[:ind]
    faults_data = faults_data[faults_data['fault_name'] == column]
    return sensor_data, ttf_data, faults_data

sensor_fault1, ttf_fault1, faults_fault1 = cutoff(sensor_data, faults_data, \
                    ttf_data, 'FlowCool Pressure Dropped Below Limit')    

sensor_fault1 = sensor_fault1.fillna(method = 'ffill')
sensor_fault1['recipe'] = sensor_fault1['recipe'] + 200
label = ttf_fault1['TTF_FlowCool Pressure Dropped Below Limit']

#------------------------------------------------------------------------------
# Capture the trends
temp = ttf_fault1.shift(1)
diff = ttf_fault1['TTF_FlowCool Pressure Dropped Below Limit'] - \
        temp['TTF_FlowCool Pressure Dropped Below Limit']
idx = diff[diff > 0].index
trend_start_time = idx.values
trend_start_time = np.insert(trend_start_time, 0, 0)   

# =============================================================================
# #------------------------------------------------------------------------------
# # One hot encoding
# catagory = ['recipe', 'recipe_step']
# for cat in catagory:
#     one_hot = pd.get_dummies(sensor_fault1[cat])
#     sensor_fault1 = sensor_fault1.drop([cat], axis = 1)
#     sensor_fault1 = sensor_fault1.join(one_hot)
# 
# =============================================================================

#------------------------------------------------------------------------------
# Select data points
def Select_tail(df, y, start_time, num):
    col = []
    for t in range(1, len(start_time)):
        if start_time[t] - start_time[t-1] > 2*num:
            col.append(df[start_time[t] - num: start_time[t]])
        else:
            col.append(df[start_time[t-1]: start_time[t]])
    df_result = pd.concat(col, axis = 0)
    y_result = [0] * len(df_result)
    return df_result, y_result

def Select_head(df, y, start_time, num):
    col = []
    for t in range(0, len(start_time)-1):
        if start_time[t + 1] - start_time[t] > 2*num:
            col.append(df[start_time[t]: start_time[t] + num])
        else:
            col.append(df[start_time[t]: start_time[t + 1]])
    df_result = pd.concat(col, axis = 0)
    y_result = [1] * len(df_result)
    return df_result, y_result
    
df_tail, y_tail = Select_tail(sensor_fault1, label, trend_start_time, 1000)
df_head, y_head = Select_head(sensor_fault1, label, trend_start_time, 1000)
df = df_head.append(df_tail)
y_head = pd.DataFrame(y_head)
y_tail = pd.DataFrame(y_tail)
y = y_head.append(y_tail)
df = df.drop(['time','runnum'], axis = 1)
df = df.reset_index(drop = True)
y = y.reset_index(drop = True)
df['label'] = y
#------------------------------------------------------------------------------
# Normalization & split dataset
df_scaler = preprocessing.StandardScaler()
feature = df_scaler.fit_transform(df)
X_train, X_valid = feature[:8000], feature[8000:]
df = df.sample(frac=1).reset_index(drop=True)
X_train, X_valid = df.iloc[:8000, :-1], df.iloc[8000:, :-1]
y_train, y_valid = df.iloc[:8000, -1], df.iloc[8000:, -1]
#------------------------------------------------------------------------------
# SVM
clf = svm.SVC(C = 1e3, gamma = 1)
clf.fit(X_train, y_train)
print('Training error is %f' %clf.score(X_train, y_train))
print('Test error is %f' %clf.score(X_valid, y_valid))
