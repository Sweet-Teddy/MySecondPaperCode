from pickle import dump
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from pandas import *
from math import sqrt
from numpy import *
import statsmodels.api as sm
import os
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import math

## import data
# 此处针对数据集jts_jun_1m.csv,如果其他数据集需要更改parse_daes列名
# df = pd.read_csv('./datasets/CSI.csv', parse_dates=['Date'])
# #df = pd.read_csv('./datasets/MSFT.csv', parse_dates=['Date'])
# #df = pd.read_csv('./datasets/CSI.csv', parse_dates=['Date'])

# data = df
# # 对数据 0 的特殊处理，我们将0替换为na
# data = data.replace(0, np.nan)
# data = data.dropna()
# # 目标值Close在第三列
# target_index = 4
# # Calculate technical indicators


# def get_technical_indicators(data):
#     # Create 7 and 21 days Moving Average
#     data['MA7'] = data.iloc[:, target_index].rolling(window=7).mean()
#     data['MA21'] = data.iloc[:, target_index].rolling(window=21).mean()

#     # Create Bollinger Bands
#     data['20SD'] = data.iloc[:, target_index].rolling(20).std()
#     data['upper_band'] = data['MA21'] + (data['20SD'] * 2)
#     data['lower_band'] = data['MA21'] - (data['20SD'] * 2)

#     # Create Exponential moving average
#     data['EMA'] = data.iloc[:, target_index].ewm(com=0.5).mean()

#     return data


# T_df = get_technical_indicators(data)
# # Drop the first 21 rows
# dataset = T_df.iloc[20:, :].reset_index(drop=True)

# # 根据收盘价的增加或减少，我们增加一列分类，如果 大于当前的请求 1  小于 -1  等于就是0
# # 目标字符串
# st = 'Close'
# dataset['CS'] = 0
# for i in range(1, dataset.shape[0]):
#     if(dataset[st][i] > dataset[st][i-1] + 1):
#         dataset['CS'][i] = 1
#     elif(dataset[st][i] + 0.9 < dataset[st][i-1]):
#         dataset['CS'][i] = -1
#     else:
#         dataset['CS'][i] = 0
# # 统计 跌 涨 平 出现的次数
# print(dataset['CS'].value_counts())
# # 将处理好的数据存储成csv
# dataset.to_csv("./saveData/CSI/dataset_classification.csv", index=False)


# %% --------------------------------------- Load Data  -----------------------------------------------------------------
# dataset = pd.read_csv('./datasets/Finaldata_with_Fourier.csv', parse_dates=['Date'])
# news = pd.read_csv("./datasets/News.csv", parse_dates=["Date"])
df = pd.read_csv('./saveData/CSI/dataset_classification.csv',
                 parse_dates=['Date'])
#df =pd.read_csv('./saveData/MSFT/dataset_classification.csv',parse_dates=['Date'])
# 我们当前只考虑请求数单变量，因此去除其他列，如果需要对多变量考虑，自行调整
dataset = pd.DataFrame()
dataset = df

# Set the date to datetime data
datetime_series = pd.to_datetime(dataset['Date'])
datetime_index = pd.DatetimeIndex(datetime_series.values)
dataset = dataset.set_index(datetime_index)
dataset = dataset.sort_values(by='Date')
dataset = dataset.drop(columns='Date')
# Get features and target
X_value = pd.DataFrame(dataset.iloc[:, :-1])
y_value = pd.DataFrame(dataset.iloc[:, -1])


# # Autocorrelation Check
# sm.graphics.tsa.plot_acf(y_value.squeeze(), lags=100)
# plt.show()

# Normalized the data
X_scaler = MinMaxScaler(feature_range=(-1, 1))
y_scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaler.fit(X_value)
y_scaler.fit(y_value)

X_scale_dataset = X_scaler.fit_transform(X_value)
y_scale_dataset = y_scaler.fit_transform(y_value)

dump(X_scaler, open('X_scaler.pkl', 'wb'))
dump(y_scaler, open('y_scaler.pkl', 'wb'))
# Reshape the data
'''Set the data input steps and output steps,
    we use 30 days data to predict 1 day price here,
    reshape it to (None, input_step, number of features) used for LSTM input'''
# 输入步长
n_steps_in = 30
n_features = X_value.shape[1]
# 输出步长
n_steps_out = 1

# Get X/y dataset


def get_X_y(X_data, y_data):
    X = list()
    y = list()
    yc = list()

    length = len(X_data)
    for i in range(0, length, 1):
        X_value = X_data[i: i + n_steps_in][:, :]
        y_value = y_data[i + n_steps_in: i + (n_steps_in + n_steps_out)][:, 0]
        yc_value = y_data[i: i + n_steps_in][:, :]
        if len(X_value) == n_steps_in and len(y_value) == 1:
            X.append(X_value)
            y.append(y_value)
            yc.append(yc_value)

    return np.array(X), np.array(y), np.array(yc)

# get the train test predict index


def predict_index(dataset, X_train, n_steps_in, n_steps_out):

    # get the predict data (remove the in_steps days)
    train_predict_index = dataset.iloc[n_steps_in: X_train.shape[0] +
                                       n_steps_in + n_steps_out - 1, :].index
    test_predict_index = dataset.iloc[X_train.shape[0] + n_steps_in:, :].index

    return train_predict_index, test_predict_index

# Split train/test dataset
# 前90%做训练集，后10%做测试集，在前90%中找出与测试集相同长度的作为验证集


def split_train_test(data):
    train_size = round(len(X) * 0.85)
    data_train = data[0:train_size]
    data_test = data[train_size:]
    return data_train, data_test


# Get data and check shape
X, y, yc = get_X_y(X_scale_dataset, y_scale_dataset)


X_train, X_test, = split_train_test(X)
y_train, y_test, = split_train_test(y)
yc_train, yc_test, = split_train_test(yc)
index_train, index_test, = predict_index(
    dataset, X_train, n_steps_in, n_steps_out)

# 将yc_train 和yc_test转化成三维
# 这里做三分类的时候需要将y_train  yc_train y_test 转化成3分类标签[label_1,label_2,label_3]
# 如果为1 表示涨 [0,0,1]
# 如果为0 表示平 [0,1,0]
# 如果为-1 表示跌[1,0,0]
l1 = []
for i in range(yc_train.shape[0]):
    temp = []
    for j in range(yc_train.shape[1]):
        if(yc_train[i][j] == 1):  # 涨
            temp.append(np.array([0, 0, 1]))
        elif(yc_train[i][j] == 0):  # 平
            temp.append(np.array([0, 1, 0]))
        else:
            temp.append(np.array([1, 0, 0]))
    l1.append(temp)
yc_train_CS = np.array(l1)


l1 = []
for i in range(yc_test.shape[0]):
    temp = []
    for j in range(yc_test.shape[1]):
        if(yc_test[i][j] == 1):  # 涨
            temp.append(np.array([0, 0, 1]))
        elif(yc_test[i][j] == 0):  # 平
            temp.append(np.array([0, 1, 0]))
        else:
            temp.append(np.array([1, 0, 0]))
    l1.append(temp)
yc_test_CS = np.array(l1)

l1 = []
for i in range(y_train.shape[0]):
    if(y_train[i] == 1):  # 涨
        l1.append(np.array([0, 0, 1]))
    elif(y_train[i] == 0):  # 平
        l1.append(np.array([0, 1, 0]))
    else:
        l1.append(np.array([1, 0, 0]))
y_train_CS = np.array(l1)

l1 = []
for i in range(y_test.shape[0]):
    if(y_test[i] == 1):  # 涨
        l1.append(np.array([0, 0, 1]))
    elif(y_test[i] == 0):  # 平
        l1.append(np.array([0, 1, 0]))
    else:
        l1.append(np.array([1, 0, 0]))
y_test_CS = np.array(l1)


# 将X_train_CS ,y_train_CS划分为训练集和验证集
X_train_raw = X_train
y_train_raw = y_train_CS
X_train = []
y_train = []
X_val = []
y_val = []

X_train = X_train_raw[0:len(X_train_raw)-int(0.5*len(X_test))]
y_train = y_train_raw[0:len(X_train_raw)-int(0.5*len(X_test))]
X_val = X_train_raw[len(X_train_raw)-int(0.5*len(X_test)):]
y_val = y_train_CS[len(X_train_raw)-int(0.5*len(X_test)):]

# 处理下，yc_train 保证其长度与X_train长度一样
# yc_test 长度 X_test长度一样
yc_train_CS = yc_train_CS[:X_train.shape[0]]

yc_train_1 = []
for i in range(yc_train_CS.shape[0]):
    yc_train_1.append(yc_train_CS[i][0])
yc_train_1 = np.array(yc_train_1)

yc_test_1 = []
for i in range(yc_test_CS.shape[0]):
    yc_test_1.append(yc_test_CS[i][0])
yc_test_1 = np.array(yc_test_1)

yc_test_1 = yc_test_1.reshape(yc_test_1.shape[0], yc_test_1.shape[1], 1)
yc_train_1 = yc_train_1.reshape(yc_train_1.shape[0], yc_train_1.shape[1], 1)

# %% --------------------------------------- Save dataset Classification-----------------------------------------------------------------
print('X shape: ', X.shape)
print('y shape: ', y.shape)
print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('X_val shape: ', X_val.shape)
print('y_val shape: ', y_val.shape)
print('X_test shape: ', X_test.shape)
print('y_test_CS shape:', y_test_CS.shape)
# S&P500

np.save("./saveData/CSI/X_train.npy", X_train)
np.save("./saveData/CSI/y_train.npy", y_train)
np.save("./saveData/CSI/X_test.npy", X_test)
np.save("./saveData/CSI/y_test.npy", y_test_CS)
np.save("./saveData/CSI/X_val.npy", X_val)
np.save("./saveData/CSI/y_val.npy", y_val)
np.save("./saveData/CSI/yc_train.npy", yc_train_1)
np.save("./saveData/CSI/yc_test.npy", yc_test_1)
# np.save("./saveData/MSFT/X_train.npy", X_train)
# np.save("./saveData/MSFT/y_train.npy", y_train)
# np.save("./saveData/MSFT/X_test.npy", X_test)
# np.save("./saveData/MSFT/y_test.npy", y_test_CS)
# np.save("./saveData/MSFT/X_val.npy", X_val)
# np.save("./saveData/MSFT/y_val.npy", y_val)

# np.save("./saveData/S&P500/yc_train.npy", yc_train_1)
# np.save("./saveData/S&P500/yc_test.npy", yc_test_1)
# np.save('./saveData/index_train.npy', index_train)
# np.save('./saveData/index_test.npy', index_test)
# np.save('./saveData/train_predict_index.npy',index_train)
# np.save('./saveData/test_predict_index.npy',index_test)


# # %% --------------------------------------- Save dataset  prediction-----------------------------------------------------------------
# print('X shape: ', X.shape)
# print('y shape: ', y.shape)
# print('X_train shape: ', X_train.shape)
# print('y_train shape: ', y_train.shape)
# print('y_c_train shape: ', yc_train.shape)
# print('X_test shape: ', X_test.shape)
# print('y_test shape: ', y_test.shape)
# print('y_c_test shape: ', yc_test.shape)
# print('index_train shape:', index_train.shape)
# print('index_test shape:', index_test.shape)

# np.save("./saveData/X_train.npy", X_train)
# np.save("./saveData/y_train.npy", y_train)
# np.save("./saveData/X_test.npy", X_test)
# np.save("./saveData/y_test.npy", y_test)
# np.save("./saveData/yc_train.npy", yc_train)
# np.save("./saveData/yc_test.npy", yc_test)
# np.save('./saveData/index_train.npy', index_train)
# np.save('./saveData/index_test.npy', index_test)
# np.save('./saveData/train_predict_index.npy',index_train)
# np.save('./saveData/test_predict_index.npy',index_test)
