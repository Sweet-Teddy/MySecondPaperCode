# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 19:07:01 2021

@author: Administrator
"""

import tensorflow as tf
from scipy import interp
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn import svm, datasets
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow

from tensorflow import keras
from numpy import *
from math import sqrt
from pandas import *
from datetime import datetime, timedelta

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, Activation
from tensorflow.keras.layers import BatchNormalization, Embedding, TimeDistributed, LeakyReLU
from tensorflow.keras.layers import LSTM, GRU, Conv1D, Flatten, RepeatVector, MaxPooling1D, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

import seaborn as sns
from matplotlib import pyplot
from pickle import load
from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score, precision_score, f1_score, recall_score
tf.config.experimental_run_functions_eagerly(True)

X_train = np.load("./saveData/MSFT/X_train.npy", allow_pickle=True)
y_train = np.load("./saveData/MSFT/y_train.npy", allow_pickle=True)
X_test = np.load("./saveData/MSFT/X_test.npy", allow_pickle=True)
y_test = np.load("./saveData/MSFT/y_test.npy", allow_pickle=True)
X_val = np.load("./saveData/MSFT/X_val.npy", allow_pickle=True)
y_val = np.load("./saveData/MSFT/y_val.npy", allow_pickle=True)
yc_train = np.load("./saveData/MSFT/yc_train.npy", allow_pickle=True)
yc_test = np.load("./saveData/MSFT/yc_test.npy", allow_pickle=True)

# X_train = np.load("./saveData/X_train.npy", allow_pickle=True)
# y_train = np.load("./saveData/y_train.npy", allow_pickle=True)
# X_test = np.load("./saveData/X_test.npy", allow_pickle=True)
# y_test = np.load("./saveData/y_test.npy", allow_pickle=True)
# X_val = np.load("./saveData/X_val.npy", allow_pickle=True)
# y_val = np.load("./saveData/y_val.npy", allow_pickle=True)
# X_train = np.load("./saveData/S&P500_v_0.1/X_train.npy", allow_pickle=True)
# y_train = np.load("./saveData/S&P500_v_0.1/y_train.npy", allow_pickle=True)
# X_test = np.load("./saveData/S&P500_v_0.1/X_test.npy", allow_pickle=True)
# y_test = np.load("./saveData/S&P500_v_0.1/y_test.npy", allow_pickle=True)
# X_val = np.load("./saveData/S&P500_v_0.1/X_val.npy", allow_pickle=True)
# y_val = np.load("./saveData/S&P500_v_0.1/y_val.npy", allow_pickle=True)
# yc_train = np.load("./saveData/S&P500_v_0.1/yc_train.npy", allow_pickle=True)
# yc_test = np.load("./saveData/S&P500_v_0.1/yc_test.npy", allow_pickle=True)

# X_train = np.load("./saveData/CSI/X_train.npy", allow_pickle=True)
# y_train = np.load("./saveData/CSI/y_train.npy", allow_pickle=True)
# X_test = np.load("./saveData/CSI/X_test.npy", allow_pickle=True)
# y_test = np.load("./saveData/CSI/y_test.npy", allow_pickle=True)
# X_val = np.load("./saveData/CSI/X_val.npy", allow_pickle=True)
# y_val = np.load("./saveData/CSI/y_val.npy", allow_pickle=True)
# yc_train = np.load("./saveData/CSI/yc_train.npy", allow_pickle=True)
# yc_test = np.load("./saveData/CSI/yc_test.npy", allow_pickle=True)


# X_train = np.load("./saveData/PAICC/X_train.npy", allow_pickle=True)
# y_train = np.load("./saveData/PAICC/y_train.npy", allow_pickle=True)
# X_test = np.load("./saveData/PAICC/X_test.npy", allow_pickle=True)
# y_test = np.load("./saveData/PAICC/y_test.npy", allow_pickle=True)
# X_val = np.load("./saveData/PAICC/X_val.npy", allow_pickle=True)
# y_val = np.load("./saveData/PAICC/y_val.npy", allow_pickle=True)
# yc_train = np.load("./saveData/PAICC/yc_train.npy", allow_pickle=True)
# yc_test = np.load("./saveData/PAICC/yc_test.npy", allow_pickle=True)

# yc_train = np.load("./saveData/yc_train.npy", allow_pickle=True)
# yc_test = np.load("./saveData/yc_test.npy", allow_pickle=True)
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)

X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)

X_val = X_val.astype(np.float32)
y_val = y_val.astype(np.float32)

# Parameters
LR = 0.001
BATCH_SIZE = 64
N_EPOCH = 500
dropout = 0.2
input_dim = X_train.shape[1]
feature_size = X_train.shape[2]
# 做3分类
output_dim = y_train.shape[1]

#output_dim = 3

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.CategoricalAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]


def basic_cnn_lstm(input_dim, feature_size):

    # model = Sequential()
    # model.add(Conv1D(filters=64, kernel_size=2,
    #                  input_shape=(input_dim, feature_size)
    #                  ))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Flatten())
    # model.add(RepeatVector(30))
    # model.add(LSTM(units=100, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(units=100, return_sequences=True,
    #                ))
    # model.add(Dropout(dropout))
    # model.add(LSTM(64))
    # model.add(Dense(16))
    # model.add(Dense(output_dim, activation='softmax'))
    # model.add(Activation('softmax'))
    model = Sequential()
    # ,kernel_regularizer = regularizers.l2(0.01)
    model.add(Conv1D(filters=256, kernel_size=2,
                     input_shape=(input_dim, feature_size), activation='elu'))
    model.add(Conv1D(filters=128, kernel_size=2,  activation='elu'
                     ))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(20))
    model.add(LSTM(units=100, return_sequences=True, activation='elu'
                   ))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True, activation='elu'))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(10))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=LR), metrics=METRICS)
    model.summary()
    history = model.fit(X_train, y_train, epochs=N_EPOCH, batch_size=BATCH_SIZE, validation_data=(X_val, y_val),
                        verbose=2, shuffle=False)
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='validation')
    pyplot.legend()
    pyplot.show()
    return model, history


model, history = basic_cnn_lstm(input_dim, feature_size)
model.save('./saveModel/CNN_LSTM_classification.h5')
# print(model.summary())
history_dict = history.history
history_dict.keys()


acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# “bo”代表 "蓝点"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b代表“蓝色实线”
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./saveImage/CNN_LSTM/train_loss.png', dpi=300)

# plt.show()
plt.clf()   # 清除数字
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('./saveImage/CNN_LSTM/acc.png', dpi=300)
# plt.show()

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_metrics(history):
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n+1)
        plt.plot(history.epoch,
                 history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()


plot_metrics(history)


def get_metrics(y_true, y_predicted):
    if(tf.is_tensor(y_predicted)):
        y_predicted = y_predicted.numpy()

        # 需要把预测的结果转化，即按概率大小把y_predicted转化成0,1
    for i in range(y_predicted.shape[0]):
        max_index = 0
        if(y_predicted[i][max_index] < y_predicted[i][1]):
            max_index = 1
        if(y_predicted[i][max_index] < y_predicted[i][2]):
            max_index = 2
        list_temp = [0, 1, 2]
        list_temp.pop(max_index)
        y_predicted[i][max_index] = 1
        y_predicted[i][list_temp[0]] = 0
        y_predicted[i][list_temp[1]] = 0

    y_predicted = y_predicted.astype(np.int32)
    y_true_test = y_true.astype(np.int32)
    y_pred_test = y_predicted

    # 将y_true和y_pred转化成-1,0,1三类，便于计算混淆矩阵
    y_true = []
    y_pred = []
    for i in range(y_true_test.shape[0]):
        if(str(y_true_test[i]) == str(np.array([1, 0, 0]))):
            y_true.append(-1)
        elif(str(y_true_test[i]) == str(np.array([0, 1, 0]))):
            y_true.append(0)
        else:
            y_true.append(1)

    for i in range(y_pred_test.shape[0]):
        if(str(y_pred_test[i]) == str(np.array([1, 0, 0]))):
            y_pred.append(-1)
        elif(str(y_pred_test[i]) == str(np.array([0, 1, 0]))):
            y_pred.append(0)
        else:
            y_pred.append(1)

    # 得到混合矩阵
    METRIC = {'Weighted precision': 0, 'Weighted recall': 0, 'Weighted f1-score': 0,
              'Macro precision': 0, 'Macro recall': 0, 'Macro f1-score': 0,
              'Micro precision': 0, 'Micro recall': 0, 'Micro f1-score': 0}

    METRIC['Weighted precision'] = precision_score(
        y_true, y_pred, average='weighted')
    METRIC['Weighted recall'] = recall_score(
        y_true, y_pred, average='weighted')
    METRIC['Weighted f1-score'] = f1_score(y_true, y_pred, average='weighted')
    METRIC['Macro precision'] = precision_score(
        y_true, y_pred, average='macro')
    METRIC['Macro recall'] = recall_score(y_true, y_pred, average='macro')
    METRIC['Macro f1-score'] = f1_score(y_true, y_pred, average='macro')
    METRIC['Micro precision'] = precision_score(
        y_true, y_pred, average='micro')
    METRIC['Micro recall'] = recall_score(y_true, y_pred, average='micro')
    METRIC['Micro f1-score'] = f1_score(y_true, y_pred, average='micro')
    return METRIC


train_predictions_baseline = model.predict(X_train, batch_size=BATCH_SIZE)
test_predictions_baseline = model.predict(X_test, batch_size=BATCH_SIZE)

baseline_results = model.evaluate(X_test, y_test,
                                  batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(model.metrics_names, baseline_results):
    print(name, ': ', value)
print()

#plot_cm(y_test, test_predictions_baseline)


# 同时也是分类得分y_score
# 概率矩阵P和标签矩阵L分别对应代码中的y_score和y_one_hot：
yhat = model.predict(X_test, verbose=0)
result = {}
result = get_metrics(y_test, yhat)
# 计算属于各个类别的概率，返回值的shape = [n_samples, n_classes]
y_score = model.predict(X_test)
# 1、调用函数计算micro类型的AUC
print('调用函数auc：', metrics.roc_auc_score(y_test, y_score, average='micro'))
# 2、手动计算micro类型的AUC
# 首先将矩阵y_one_hot和y_score展开，然后计算假正例率FPR和真正例率TPR
fpr, tpr, thresholds = metrics.roc_curve(y_test.ravel(), y_score.ravel())
auc = metrics.auc(fpr.astype(np.float32), tpr.astype(np.float32))
print('手动计算auc：', auc)
# 绘图
mpl.rcParams['font.sans-serif'] = u'SimHei'
mpl.rcParams['axes.unicode_minus'] = False
# FPR就是横坐标,TPR就是纵坐标
plt.plot(fpr, tpr, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % auc)
plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
plt.xlim((-0.01, 1.02))
plt.ylim((-0.01, 1.02))
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.grid(b=True, ls=':')
plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
plt.title(u'mtj_june数据集分类后的ROC和AUC', fontsize=17)
plt.savefig('./saveImage/CNN_LSTM/ROC.png', dpi=300)
# plt.show()

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 3
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(
        y_test[:, i], y_score[:, i])
    fpr[i] = fpr[i].astype(np.float32)
    tpr[i] = tpr[i].astype(np.float32)
    roc_auc[i] = metrics.auc(fpr[i],
                             tpr[i])
# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = metrics.auc(fpr["micro"],
                               tpr["micro"])
# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = metrics.auc(fpr["macro"].astype(np.float32),
                               tpr["macro"].astype(np.float32))

# Plot all ROC curves
lw = 2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.savefig('./saveImage/CNN_LSTM/Result.png', dpi=300)
# plt.show()
for key in result:
    print(str(key) + '： ' + str(result[key]))


def result_to_0_1(y):
    if(tf.is_tensor(y)):
        y = y.numpy()
        for i in range(y.shape[0]):
            max_index = 0
            if(y[i][max_index] < y[i][1]):
                max_index = 1
            if(y[i][max_index] < y[i][2]):
                max_index = 2
            list_temp = [0, 1, 2]
            list_temp.pop(max_index)
            y[i][max_index] = 1
            y[i][list_temp[0]] = 0
            y[i][list_temp[1]] = 0
    else:
        for i in range(y.shape[0]):
            max_index = 0
            if(y[i][max_index] < y[i][1]):
                max_index = 1
            if(y[i][max_index] < y[i][2]):
                max_index = 2
            list_temp = [0, 1, 2]
            list_temp.pop(max_index)
            y[i][max_index] = 1
            y[i][list_temp[0]] = 0
            y[i][list_temp[1]] = 0
    return y


def to_0_1(y):
    res = []
    for i in range(y.shape[0]):
        if(y[i][0] > y[i][1] and y[i][0] > y[i][2]):
            res.append(-1)  # 跌
        if(y[i][1] > y[i][0] and y[i][1] > y[i][2]):
            res.append(0)  # 平
        if(y[i][2] > y[i][0] and y[i][2] > y[i][1]):
            res.append(1)  # 涨
    return np.array(res)


y = result_to_0_1(yhat)
y_pred = to_0_1(yhat)
y_true = to_0_1(y_test)

cm = confusion_matrix(y_true, y_pred)
confusion_matrix = pd.DataFrame(
    cm, index=['Up', 'O', 'Down'], columns=['Up', 'O', 'Down'])
print(cm)
fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='.20g',
            annot_kws={"size": 19}, cmap="Blues")
plt.ylabel("True label", fontsize=18)
plt.xlabel("Predicted label", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig('./saveImage/CNN_LSTM/confusion_matrix')
# plt.show()

# 将y_score转化成y  通过one-hot

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 3
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])
# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw = 2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.4f})'
         ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.4f})'
         ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.savefig('./saveImage/CNN_LSTM/Result_0_1.png', dpi=300)
