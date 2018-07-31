#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 15:33:44 2018

@author: ninja1mmm
"""

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import math
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
import keras.callbacks
import keras.backend as K
import tensorflow as tf
#import DataGenerator

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

fault_name_1 = 'FlowCool Pressure Dropped Below Limit'

# =============================================================================
# df_tmp = pd.read_pickle('/home/ninja1mmm/Desktop/phm/data/train/01_M01_DC_train')
# 
# df_scaler = pd.DataFrame(index=['min','max'],columns=df_tmp.columns[4:])
# y_scaler = pd.DataFrame(index = ['min','max'], columns = [fault_name_1])
# 
# for file_tmp in files_tmp[2]:    
#     path_tmp = root_tmp[2]+'/'+file_tmp
#     df = pd.read_pickle(path_tmp)
#     ttf_tmp = root_tmp[3]+'/'+file_tmp+'_ttf'
#     ttf = pd.read_pickle(ttf_tmp)
#     # 
#     ttf_1 = ttf['TTF_'+fault_name_1]
#     #
#     for col in df_scaler.columns:
#         max_tmp = np.max(df[col])
#         min_tmp = np.min(df[col])
#         if np.isnan(df_scaler[col]['max']) | (df_scaler[col]['max'] < max_tmp):
#             df_scaler[col]['max'] = max_tmp
#         if np.isnan(df_scaler[col]['min']) | (df_scaler[col]['min'] > min_tmp):
#             df_scaler[col]['min'] = min_tmp
#     
#     max_label = np.max(ttf_1)
#     min_label = np.min(ttf_1)
#     if np.isnan(y_scaler[fault_name_1]['max']) | (y_scaler[fault_name_1]['max'] < max_label):
#         y_scaler[fault_name_1]['max'] = max_label
#     if np.isnan(y_scaler[fault_name_1]['min']) | (y_scaler[fault_name_1]['min'] > min_label):
#         y_scaler[fault_name_1]['min'] = min_label
#         
# df_path = [root_tmp[2] + '/' + x for x in files_tmp[2]]
# ttf_path = [root_tmp[3] + '/'+ x + '_ttf' for x in files_tmp[2]]
# 
# =============================================================================
y_scaler = pd.read_pickle('y_scaler')
df_scaler = pd.read_pickle('df_scaler')

def cutoff(sensor_data, faults_data, ttf_data, column):
        # cut off the tail of the data set that with NaN ttf
        temp = faults_data[faults_data['fault_name'] == column]
        if len(temp) == 0:
            return pd.DataFrame(columns=sensor_data.columns),\
                    pd.DataFrame(columns=ttf_data.columns),\
                    pd.DataFrame(columns=faults_data.columns)
        last_failure = temp['time'].values[-1]
        array = np.asarray(sensor_data['time'])
        closest_ind = (array[(last_failure-array) > 0]).argmax()
        sensor_data = sensor_data[:closest_ind+1]
        ttf_data = ttf_data[:closest_ind+1]
        faults_data = faults_data[faults_data['fault_name'] == column]
        return sensor_data, ttf_data, faults_data

def generate_data(roots, unitname, df_scaler, y_scaler, batch_size):
#    counter = 0
#    unitname = files[counter][:6]
#    print(unitname)
#    counter = (counter + 1) % len(files)
    
    sensor_data = pd.read_pickle(roots[2] + '/' + unitname + '_DC_train')
    faults_data = pd.read_pickle(roots[1] + '/' + unitname + '_train_fault_data')
    ttf_data = pd.read_pickle(roots[3] + '/' + unitname + '_DC_train_ttf')

    fault_name_1 = 'FlowCool Pressure Dropped Below Limit'
    sensor_data = sensor_data.drop(['Tool'], axis = 1)
    sensor_data = sensor_data.drop(['Lot'], axis = 1)

    sensor_data.index = range(0,len(sensor_data))
    ttf_data.index = range(0,len(ttf_data))
    sensor_fault1, ttf_fault1, faults_fault1 = cutoff(sensor_data, faults_data, \
                                                      ttf_data, fault_name_1)    

    sensor_fault1 = sensor_fault1.fillna(method = 'ffill') 
    label = ttf_fault1['TTF_'+fault_name_1]
    X_train = sensor_fault1.iloc[:,2:]
    X_train = (X_train - df_scaler.loc['min'])/(df_scaler.loc['max'] - df_scaler.loc['min'])
    
    y = label
    y_train = y.values.reshape(-1,1)
    y_train = (y_train - y_scaler.loc['min'][0])/(y_scaler.loc['max'][0] - y_scaler.loc['min'][0])
    
    X_train = X_train.astype('float32').values
    y_train = y_train.astype('float32')
    if len(X_train)%batch_size==0:
        batch_num = len(X_train)//batch_size
    else:
        batch_num = len(X_train)//batch_size + 1
    
    return(X_train, y_train, batch_num)    
        
#        y_train = y_train.flatten()
#        for cbatch in range(0, X_train.shape[0], batch_size):
#            yield (X_train.iloc[cbatch:(cbatch + batch_size),:], y_train[cbatch:(cbatch + batch_size)])

# -----------------------------------------------------------------------------
tf.reset_default_graph()
X = tf.placeholder(tf.float32,[None,20])
y = tf.placeholder(tf.float32,[None,1])
is_training = tf.placeholder(tf.bool)

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def FCNN(X, y, is_training):
    W_fc1 = weight_variable([20,10])
    b_fc1 = bias_variable([10])
    W_fc2 = weight_variable([10,10])
    b_fc2 = bias_variable([10])
    W_fc3 = weight_variable([10,1])
    b_fc3 = bias_variable([1])
    
    h_fc1 = tf.nn.relu(tf.matmul(X, W_fc1) + b_fc1)
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    y_out = tf.matmul(h_fc2, W_fc3) + b_fc3
    
    return y_out

y_out = FCNN(X, y, is_training)


def customLoss(yTrue,yPred):
    return K.sum(K.exp(-0.001*yTrue)*K.abs(yTrue-yPred))

# -----------------------------------------------------------------------------
total_loss = tf.multiply(tf.exp(tf.multiply(-0.001,y)),tf.abs(y_out-y))
sum_loss = tf.reduce_sum(total_loss)

# define our optimizer
optimizer = tf.train.AdamOptimizer(5e-4) # select optimizer and set learning rate
train_step = optimizer.minimize(sum_loss)


# -----------------------------------------------------------------------------
# Feed a random batch into the model and make sure the ouput is the right size
# =============================================================================
# x = np.random.randn(128,20)
# with tf.Session() as sess:
#     with tf.device("/gpu:0"): #"/cpu:0" or "/gpu:0"
#         tf.global_variables_initializer().run()
# 
#         ans = sess.run(y_out,feed_dict={X:x,is_training:True})
#         %timeit sess.run(y_out,feed_dict={X:x,is_training:True})
#         print(ans.shape)
#         print(np.array_equal(ans.shape, np.array([128, 1])))
# =============================================================================
# -----------------------------------------------------------------------------
def run_model(session, predict, loss_val, root_tmp, train_files, df_scaler,y_scaler,\
              batch_size=128, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
#    correct_prediction = tf.equal(tf.argmax(predict,1), y)
#    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # shuffle indicies
    

    training_now = training is not None
    
    epochs = len(train_files)
    # counter 
    batch_left = 0
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        losses = []
        
        if batch_left <= 0:
            unitname = train_files[e][:6]
            Xd, yd, batch_left = generate_data(root_tmp, unitname, df_scaler, y_scaler, batch_size)
        
        train_indicies = np.arange(Xd.shape[0])
        np.random.shuffle(train_indicies)
            
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         is_training: training_now }
            # get batch size
            actual_batch_size = yd[idx].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            if training_now:
                _,loss = session.run([training, loss_val],feed_dict=feed_dict)
            else:
                loss = session.run([loss_val],feed_dict=feed_dict)
                loss = loss[0]
            # aggregate performance stats
            losses.append(loss)
            batch_left -= 1
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1}"\
                      .format(iter_cnt,loss))
            iter_cnt += 1
            
#        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {1}, Overall loss = {0}"\
              .format(np.sum(losses)/i,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss

train_files = files_tmp[2][:15]
valid_files = files_tmp[2][15:]
saver = tf.train.Saver()
with tf.Session() as sess:
    with tf.device("/gpu:0"): #"/cpu:0" or "/gpu:0" 
        
        sess.run(tf.global_variables_initializer())
        print('Training')
        counter = 0
        run_model(sess,y_out,sum_loss,root_tmp, train_files,df_scaler,y_scaler,\
                  128,1000,train_step, True)        
        print('Validation')
        run_model(sess,y_out,sum_loss,root_tmp, valid_files,df_scaler,y_scaler,\
                  128)
        save_path = saver.save(sess,'model1.ckpt')

# -----------------------------------------------------------------------------

# Parameters
# =============================================================================
# params = {'dim': (32,32,32),
#           'batch_size': 64,
#           'n_classes': 6,
#           'n_channels': 1,
#           'shuffle': True}
# 
# # Datasets
# partition = # IDs
# labels = # Labels
# 
# # Generators
# training_generator = DataGenerator(partition['train'], labels, **params)
# validation_generator = DataGenerator(partition['validation'], labels, **params)
# 
# # Design model
# model = Sequential()
# [...] # Architecture
# model.compile()
# 
# # Train model on dataset
# model.fit_generator(generator=training_generator,
#                     validation_data=validation_generator,
#                     use_multiprocessing=True,
#                     workers=6)
# 
# 
# =============================================================================

#------------------------------------------------------------------------------
# Read in Data
#sensor_data = pd.read_csv('C:\\Users\\chen.zc\\Desktop\\phm_data_challenge_2018.tar\\train\\01_M01_DC_train.csv')
#faults_data = pd.read_csv('C:\\Users\\chen.zc\\Desktop\\phm_data_challenge_2018.tar\\train\\train_faults\\01_M01_train_fault_data.csv')
#ttf_data = pd.read_csv('C:\\Users\\chen.zc\\Desktop\\phm_data_challenge_2018.tar\\train\\train_ttf\\01_M01_DC_train.csv')
# =============================================================================
# 
# sensor_data = pd.read_pickle('/home/ninja1mmm/Desktop/phm/data/train/01_M01_DC_train')
# faults_data = pd.read_pickle('/home/ninja1mmm/Desktop/phm/data/fault/01_M01_train_fault_data')
# ttf_data = pd.read_pickle('/home/ninja1mmm/Desktop/phm/data/ttf/01_M01_DC_train_ttf')
# 
# fault_name_1 = 'FlowCool Pressure Dropped Below Limit'
# sensor_data = sensor_data.drop(['Tool'], axis = 1)
# sensor_data = sensor_data.drop(['Lot'], axis = 1)
# 
# #sensor_data = sensor_data.loc[sensor_data.index %10 == 0]
# #ttf_data = ttf_data.loc[ttf_data.index %10 == 0]
# sensor_data.index = range(0,len(sensor_data))
# ttf_data.index = range(0,len(ttf_data))
# 
# sensor_fault1, ttf_fault1, faults_fault1 = cutoff(sensor_data, faults_data, \
#                     ttf_data, fault_name_1)    
# 
# sensor_fault1 = sensor_fault1.fillna(method = 'ffill')
# 
# 
# label = ttf_fault1['TTF_'+fault_name_1]
# df = sensor_fault1.iloc[:,2:]
# y = label
# y = y.values.reshape(-1,1)
# 
# feature = (df-np.min(df))/(np.max(df)-np.min(df))
# label = (y-np.min(y))/(np.max(y)-np.min(y))
# 
# X_train, X_valid = feature[:1830000], feature[1830000:]
# y_train, y_valid = label[:1830000], label[1830000:]
# 
# 
# 
# #------------------------------------------------------------------------------
# 
# # LSTM
# #X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
# #X_valid = X_valid.reshape((X_valid.shape[0], 1, X_valid.shape[1]))
# 
# model = Sequential()
# model.add(Dense(10, input_shape=(20,)))
# model.add(Dense(10))
# #model.add(LSTM(64, return_sequences = True, input_shape=(X_train.shape[1], X_train.shape[2])))
# #model.add(LSTM(64))
# model.add(Dense(1))
# model.compile(loss = customLoss, optimizer='adam')
# 
# # =============================================================================
# 
# # Early stopping
# #es = keras.callbacks.EarlyStopping(monitor='val_loss',
# #                              min_delta=0,
# #                              patience=4,
# #                              verbose=0, mode='auto')
# history = model.fit(X_train, y_train, epochs=100, batch_size=128, \
#                     validation_data=(X_valid, y_valid), verbose=2, shuffle=True)
# 
# # =============================================================================
# # training_generator = generate_batches(files_tmp[2][:15], root_tmp, df_scaler, y_scaler, 128)
# # validation_generator = generate_batches(files_tmp[2][15:], root_tmp, df_scaler, y_scaler, 128)
# # 
# # 
# # history = model.fit_generator(generator=training_generator,
# #                      validation_data=validation_generator,
# # #                     steps_per_epoch = 1000,
# # #                     validation_steps = 1000,
# #                      use_multiprocessing=True,
# #                      workers=6)
# # 
# # =============================================================================
# # =============================================================================
# plt.figure()
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# plt.show()
# 
# # scale back the outputs
# yhat = model.predict(X_train)
# #y_pred = y_scaler.inverse_transform(yhat)
# #y_real = y_scaler.inverse_transform(y_train)
# plt.figure()
# plt.plot(y_train)
# plt.plot(yhat)
# plt.legend()
# plt.show()
# 
# 
# =============================================================================
