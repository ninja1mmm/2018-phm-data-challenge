import random
random.seed(1234)
# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scipy.stats
from scipy.fftpack import fft, fftfreq
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
import keras.callbacks
import tensorflow as tf
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
sensor_data = pd.read_csv('D:\\phm_data_challenge_2018.tar\\train\\01_M01_DC_train.csv')
faults_data = pd.read_csv('D:\\phm_data_challenge_2018.tar\\train\\train_faults\\01_M01_train_fault_data.csv')
ttf_data = pd.read_csv('D:\\phm_data_challenge_2018.tar\\train\\train_ttf\\01_M01_DC_train.csv')

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

#------------------------------------------------------------------------------
# Separate data frame
window_size = 500
pdf = matplotlib.backends.backend_pdf.PdfPages("Feature Extraction.pdf")
for col in sensor_fault1.columns:
    df = sensor_fault1.iloc[0:354984][col]
    df = df.iloc[df.index %10 == 0]
    mean = []
    rms = []
    kurtosis = []
    for i in range(0,len(df)-window_size):
        temp = df.iloc[i:i+window_size]
        m = np.mean(temp)
        r = np.sqrt(np.mean(temp**2))
        k = scipy.stats.kurtosis(temp)
        mean.append(m)
        rms.append(r)
        kurtosis.append(k)
        FT = fft(temp)
    plt.figure()
    plt.plot(mean)
    plt.title('%s mean' %col)
    pdf.savefig()
    plt.figure()
    plt.plot(rms)
    plt.title('%s rms' %col)
    pdf.savefig()
    plt.figure()
    plt.plot(kurtosis)
    plt.title('%s kurtosis' %col)
    pdf.savefig()
pdf.close()