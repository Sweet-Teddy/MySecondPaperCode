import os
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import math

## import data
# 此处针对数据集jts_jun_1m.csv,如果其他数据集需要更改parse_daes列名
df = pd.read_csv('./datasets/CSI.csv', parse_dates=['Date'])
#df = pd.read_csv('./datasets/MSFT.csv', parse_dates=['Date'])
#df = pd.read_csv('./datasets/CSI.csv', parse_dates=['Date'])

data = df
# 对数据 0 的特殊处理，我们将0替换为na
data = data.replace(0, np.nan)
data = data.dropna()
# 目标值Close在第三列
target_index = 4
# Calculate technical indicators


def get_technical_indicators(data):
    # Create 7 and 21 days Moving Average
    data['MA7'] = data.iloc[:, target_index].rolling(window=7).mean()
    data['MA21'] = data.iloc[:, target_index].rolling(window=21).mean()

    # Create Bollinger Bands
    data['20SD'] = data.iloc[:, target_index].rolling(20).std()
    data['upper_band'] = data['MA21'] + (data['20SD'] * 2)
    data['lower_band'] = data['MA21'] - (data['20SD'] * 2)

    # Create Exponential moving average
    data['EMA'] = data.iloc[:, target_index].ewm(com=0.5).mean()

    return data


T_df = get_technical_indicators(data)
# Drop the first 21 rows
dataset = T_df.iloc[20:, :].reset_index(drop=True)

# 根据收盘价的增加或减少，我们增加一列分类，如果 大于当前的请求 1  小于 -1  等于就是0
# 目标字符串
st = 'Close'
dataset['CS'] = 0
for i in range(1, dataset.shape[0]):
    if(dataset[st][i] > dataset[st][i-1] + 3):
        dataset['CS'][i] = 1
    elif(dataset[st][i] + 0.9 < dataset[st][i-1]):
        dataset['CS'][i] = -1
    else:
        dataset['CS'][i] = 0
# 统计 跌 涨 平 出现的次数
print(dataset['CS'].value_counts())
# 将处理好的数据存储成csv
dataset.to_csv("./saveData/CSI/dataset_classification.csv", index=False)
