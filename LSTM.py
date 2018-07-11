import random
random.seed(1234)
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
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
import keras.callbacks
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
def Select(df, y, start_time, num):
    col = []
    y_result = pd.Series()
    for t in range(1, len(start_time)):
        if start_time[t] - start_time[t-1] > num:
            col.append(df[start_time[t] - num: start_time[t]])
            y_result = y_result.append(y[start_time[t] - num: start_time[t]])
        else:
            col.append(df[start_time[t-1]: start_time[t]])
            y_result = y_result.append(y[start_time[t-1]: start_time[t]])
    df_result = pd.concat(col, axis = 0)
    return df_result, y_result
    
df_select, y_select = Select(sensor_fault1, label, trend_start_time, 2000)

#------------------------------------------------------------------------------
# Shift dataset
def series_to_supervised(data, y, n_in=50, dropnan=True):
    data_col = []
    y_col = []
    for i in range (0, n_in):
        data_col.append(data.shift(i))
        y_col.append(y.shift(i))
    result = pd.concat(data_col, axis = 1)
    label = pd.concat(y_col, axis = 1)
    if dropnan:
        result = result[n_in:]
        label = label[n_in:]
    return result, label

df, y = series_to_supervised(df_select, y_select, 10, True)
df_scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
y_scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
feature = df_scaler.fit_transform(df)
label = y_scaler.fit_transform(y)
y_train, y_valid, y_test = label[0:3900], label[16000:], label
X_train, X_valid, y_test = feature[0:3900], feature[16000:], feature


#------------------------------------------------------------------------------
# LSTM
X_train = X_train.reshape((X_train.shape[0], 10, 22))
X_valid = X_valid.reshape((X_valid.shape[0], 10, 22))
model = Sequential()
model.add(LSTM(10, return_sequences=True,  input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10))
model.add(Dense(10))
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss=customLoss, optimizer='adam')
# Early stopping
es = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')
history = model.fit(X_train, y_train, epochs=500, batch_size=256, \
                    validation_data=(X_valid, y_valid), verbose=2, shuffle=False)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# scale back the outputs
yhat = model.predict(X_train)
y_pred = y_scaler.inverse_transform(yhat)
y_real = y_scaler.inverse_transform(y_train)
plt.figure()
plt.plot(y_real[:,0])
plt.plot(y_pred[:,0])
plt.legend()
plt.show()
# =============================================================================
# #------------------------------------------------------------------------------
# # Check correlation between features and labels
# def spearman(frame, features):
#     spr = pd.DataFrame()
#     spr['feature'] = features
#     spr['spearman'] = [frame[f].corr(frame['TTF_FlowCool Pressure Dropped Below Limit'], 'spearman') for f in features]
#     spr = spr.sort_values('spearman')
#     plt.figure(figsize=(6, 0.25*len(features)))
#     sns.barplot(data=spr, y='feature', x='spearman', orient='h')
# features = df.columns[0:18]
# spearman(df, features)
# =============================================================================

