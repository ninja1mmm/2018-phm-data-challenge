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
from keras.layers import Dense
from keras.layers import LSTM
import keras.callbacks


#------------------------------------------------------------------------------
# Read in Data
#sensor_data = pd.read_csv('C:\\Users\\chen.zc\\Desktop\\phm_data_challenge_2018.tar\\train\\01_M01_DC_train.csv')
#faults_data = pd.read_csv('C:\\Users\\chen.zc\\Desktop\\phm_data_challenge_2018.tar\\train\\train_faults\\01_M01_train_fault_data.csv')
#ttf_data = pd.read_csv('C:\\Users\\chen.zc\\Desktop\\phm_data_challenge_2018.tar\\train\\train_ttf\\01_M01_DC_train.csv')

sensor_data = pd.read_pickle('/home/ninja1mmm/Desktop/phm/data/train/01_M01_DC_train')
faults_data = pd.read_pickle('/home/ninja1mmm/Desktop/phm/data/fault/01_M01_train_fault_data')
ttf_data = pd.read_pickle('/home/ninja1mmm/Desktop/phm/data/ttf/01_M01_DC_train_ttf')


sensor_data = sensor_data.drop(['Tool'], axis = 1)
sensor_data = sensor_data.drop(['Lot'], axis = 1)

sensor_data = sensor_data.loc[sensor_data.index %10 == 0]
ttf_data = ttf_data.loc[ttf_data.index %10 == 0]
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
# Shift dataset
def series_to_supervised(data, y, n_in=1, dropnan=True):
    for i in range (0, n_in):
        temp = data.shift(i)
        data = pd.concat([data, temp], axis = 1)
    if dropnan:
        data = data[n_in:]
        y = y[n_in:]
    return data, y

label = ttf_fault1['TTF_FlowCool Pressure Dropped Below Limit']
df, y = series_to_supervised(sensor_fault1, label, 1, True)
y = y.values.reshape(-1, 1)
df_scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
y_scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
feature = df_scaler.fit_transform(df)
label = y_scaler.fit_transform(y)
X_train, X_valid = feature[:250000], feature[250000:]
y_train, y_valid = label[:250000], label[250000:]

#------------------------------------------------------------------------------
# LSTM
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_valid = X_valid.reshape((X_valid.shape[0], 1, X_valid.shape[1]))
model = Sequential()
model.add(LSTM(64, return_sequences = True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(64))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# Early stopping
es = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')
history = model.fit(X_train, y_train, epochs=50, batch_size=64, \
                    validation_data=(X_valid, y_valid), verbose=2, shuffle=False\
                    , callbacks=[es])

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# scale back the outputs
yhat = model.predict(X_train)
y_pred = y_scaler.inverse_transform(yhat)
y_real = y_scaler.inverse_transform(y_train)
plt.figure()
plt.plot(y_real)
plt.plot(y_pred)
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

