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
from sklearn import tree
from sklearn import ensemble
from keras import backend as K
from sklearn.tree import export_graphviz
import pydot
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

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
# =============================================================================
# sensor_data = pd.read_csv('C:\\Users\\chen.zc\\Desktop\\phm_data_challenge_2018.tar\\train\\01_M01_DC_train.csv')
# faults_data = pd.read_csv('C:\\Users\\chen.zc\\Desktop\\phm_data_challenge_2018.tar\\train\\train_faults\\01_M01_train_fault_data.csv')
# ttf_data = pd.read_csv('C:\\Users\\chen.zc\\Desktop\\phm_data_challenge_2018.tar\\train\\train_ttf\\01_M01_DC_train.csv')
# 
# =============================================================================
sensor_data = pd.read_pickle('/home/ninja1mmm/Desktop/phm/data/train/01_M01_DC_train')
faults_data = pd.read_pickle('/home/ninja1mmm/Desktop/phm/data/fault/01_M01_train_fault_data')
ttf_data = pd.read_pickle('/home/ninja1mmm/Desktop/phm/data/ttf/01_M01_DC_train_ttf')

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
#sensor_fault1['recipe'] = sensor_fault1['recipe'] + 200
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
# One hot encoding with "rare" values
sensor_fault1.loc[sensor_fault1['recipe'].value_counts()[sensor_fault1['recipe']].values < 250000, 'recipe'] = "RARE_VALUE"
sensor_fault1.loc[sensor_fault1['stage'].value_counts()[sensor_fault1['stage']].values < 250000, 'stage'] = "RARE_VALUE"
enc = OneHotEncoder(handle_unknown='ignore')
names = ['recipe','stage']
tmp = sensor_fault1[names]
for i in range(tmp.shape[1]):
    tmp.iloc[:,i] = tmp.iloc[:,i].astype('str')
    le = preprocessing.LabelEncoder()
    tmp.iloc[:,i] = le.fit_transform(tmp.iloc[:,i])
tmp = pd.DataFrame(enc.fit_transform(tmp).toarray())

sensor_fault1 = sensor_fault1.join(tmp)
sensor_fault1 = sensor_fault1.drop(['stage'],axis=1)
sensor_fault1 = sensor_fault1.drop(['recipe'],axis=1)
#------------------------------------------------------------------------------
#sensor_fault1 = sensor_fault1[sensor_fault1['recipe_step']==3]
#sensor_fault1 = sensor_fault1.drop(['recipe_step'],axis=1)
# -----------------------------------------------------------------------------
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
#df = df.drop(['time','runnum','ETCHSOURCEUSAGE','ETCHAUXSOURCETIMER','ETCHAUX2SOURCETIMER'], axis = 1)

df = df.reset_index(drop = True)
y = y.reset_index(drop = True)
df['label'] = y
#------------------------------------------------------------------------------
# Normalization & split dataset
df_scaler = preprocessing.StandardScaler()
feature = df_scaler.fit_transform(df)
#X_train, X_valid = feature[:8000], feature[8000:]
df_new = df.sample(frac=1).reset_index(drop=True)
X_train, X_valid = df_new.iloc[:10000, :-1], df_new.iloc[10000:, :-1]
y_train, y_valid = df_new.iloc[:10000, -1], df_new.iloc[10000:, -1]
#------------------------------------------------------------------------------
# SVM
#clf = svm.SVC(C = 1, gamma = 1)
# Decision Tree
#clf = tree.DecisionTreeClassifier()
#clf = tree.DecisionTreeClassifier(criterion = 'entropy')
#clf = ensemble.RandomForestClassifier(criterion = 'entropy')
clf = LogisticRegression()


clf.fit(X_train, y_train)
print('Training error is %f' %clf.score(X_train, y_train))
print('Test error is %f' %clf.score(X_valid, y_valid))



n = 2000
X_pred,_ = Select_tail(sensor_fault1, label, trend_start_time, n)
X_pred = X_pred.drop(['time','runnum'], axis = 1)
y_pred = clf.predict(X_pred)
print('Predicition error: %f' %(np.sum(y_pred)/len(y_pred)))

plt.figure()
for i in range(1,len(trend_start_time)):
    plt.subplot(4,3,i)        
    test_fault_period = sensor_fault1.iloc[trend_start_time[i-1]:trend_start_time[i],:]
    test_pred = test_fault_period.drop(['time','runnum'], axis = 1)
#    test_y_pred = clf.predict(test_pred)
    test_y_pred = clf.predict_proba(test_pred)[:,0]
    plt.scatter(test_fault_period['time'],test_y_pred,s=2)
#    plt.plot(test_y_pred)
#    plt.plot(test_y_pred[-8000:])

# Write dot file. Can be translated to pdf with line commands
#with open("Fault1.dot", "w") as f:
#    f = tree.export_graphviz(clf, out_file=f)


# =============================================================================
# # View the graph directly
# tree = clf.estimators_[5]
# # Export the image to a dot file
# export_graphviz(tree, out_file = 'tree.dot', feature_names = X_train.columns, rounded = True, precision = 1)
# # Use dot file to create a graph
# (graph, ) = pydot.graph_from_dot_file('tree.dot')
# # Write graph to a png file
# graph.write_png('tree.png')
# =============================================================================
