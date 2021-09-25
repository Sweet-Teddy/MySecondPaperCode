# coding=utf-8
from scipy import interp
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import svm, datasets
from itertools import cycle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import warnings
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
# from main.feature import get_all_features
from tensorflow.keras.layers import GRU, LSTM, GlobalAveragePooling1D, RepeatVector, Bidirectional, Dense, Flatten, Conv1D, BatchNormalization, LeakyReLU, Dropout, Activation, MaxPooling1D, Flatten
from tensorflow.keras import Sequential, Input, Model
from pickle import load
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import regularizers
from sklearn import metrics as ms

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score, precision_score, f1_score, recall_score
tf.config.experimental_run_functions_eagerly(True)

warnings.filterwarnings('ignore')

# X_train = np.load("./saveData/X_train.npy", allow_pickle=True)
# y_train = np.load("./saveData/y_train.npy", allow_pickle=True)
# X_test = np.load("./saveData/X_test.npy", allow_pickle=True)
# y_test = np.load("./saveData/y_test.npy", allow_pickle=True)
# X_val = np.load("./saveData/X_val.npy", allow_pickle=True)
# y_val = np.load("./saveData/y_val.npy", allow_pickle=True)
# yc_train = np.load("./saveData/yc_train.npy", allow_pickle=True)
# yc_test = np.load("./saveData/yc_test.npy", allow_pickle=True)

# X_train = np.load("./saveData/S&P500/X_train.npy", allow_pickle=True)
# y_train = np.load("./saveData/S&P500/y_train.npy", allow_pickle=True)
# X_test = np.load("./saveData/S&P500/X_test.npy", allow_pickle=True)
# y_test = np.load("./saveData/S&P500/y_test.npy", allow_pickle=True)
# X_val = np.load("./saveData/S&P500/X_val.npy", allow_pickle=True)
# y_val = np.load("./saveData/S&P500/y_val.npy", allow_pickle=True)
# yc_train = np.load("./saveData/S&P500/yc_train.npy", allow_pickle=True)
# yc_test = np.load("./saveData/S&P500/yc_test.npy", allow_pickle=True)

X_train = np.load("./saveData/PAICC/X_train.npy", allow_pickle=True)
y_train = np.load("./saveData/PAICC/y_train.npy", allow_pickle=True)
X_test = np.load("./saveData/PAICC/X_test.npy", allow_pickle=True)
y_test = np.load("./saveData/PAICC/y_test.npy", allow_pickle=True)
X_val = np.load("./saveData/PAICC/X_val.npy", allow_pickle=True)
y_val = np.load("./saveData/PAICC/y_val.npy", allow_pickle=True)
yc_train = np.load("./saveData/PAICC/yc_train.npy", allow_pickle=True)
yc_test = np.load("./saveData/PAICC/yc_test.npy", allow_pickle=True)

# X_train = np.load("./saveData/CSI/X_train.npy", allow_pickle=True)
# y_train = np.load("./saveData/CSI/y_train.npy", allow_pickle=True)
# X_test = np.load("./saveData/CSI/X_test.npy", allow_pickle=True)
# y_test = np.load("./saveData/CSI/y_test.npy", allow_pickle=True)
# X_val = np.load("./saveData/CSI/X_val.npy", allow_pickle=True)
# y_val = np.load("./saveData/CSI/y_val.npy", allow_pickle=True)
# yc_train = np.load("./saveData/CSI/yc_train.npy", allow_pickle=True)
# yc_test = np.load("./saveData/CSI/yc_test.npy", allow_pickle=True)

# X_train = np.load("./saveData/MSFT/X_train.npy", allow_pickle=True)
# y_train = np.load("./saveData/MSFT/y_train.npy", allow_pickle=True)
# X_test = np.load("./saveData/MSFT/X_test.npy", allow_pickle=True)
# y_test = np.load("./saveData/MSFT/y_test.npy", allow_pickle=True)
# X_val = np.load("./saveData/MSFT/X_val.npy", allow_pickle=True)
# y_val = np.load("./saveData/MSFT/y_val.npy", allow_pickle=True)
# yc_train = np.load("./saveData/MSFT/yc_train.npy", allow_pickle=True)
# yc_test = np.load("./saveData/MSFT/yc_test.npy", allow_pickle=True)


# X_train = np.load("./saveData/IBM/X_train.npy", allow_pickle=True)
# y_train = np.load("./saveData/IBM/y_train.npy", allow_pickle=True)
# X_test = np.load("./saveData/IBM/X_test.npy", allow_pickle=True)
# y_test = np.load("./saveData/IBM/y_test.npy", allow_pickle=True)
# X_val = np.load("./saveData/IBM/X_val.npy", allow_pickle=True)
# y_val = np.load("./saveData/IBM/y_val.npy", allow_pickle=True)
# yc_train = np.load("./saveData/IBM/yc_train.npy", allow_pickle=True)
# yc_test = np.load("./saveData/IBM/yc_test.npy", allow_pickle=True)

# X_train = np.load("./saveData/TSLA/X_train.npy", allow_pickle=True)
# y_train = np.load("./saveData/TSLA/y_train.npy", allow_pickle=True)
# X_test = np.load("./saveData/TSLA/X_test.npy", allow_pickle=True)
# y_test = np.load("./saveData/TSLA/y_test.npy", allow_pickle=True)
# X_val = np.load("./saveData/TSLA/X_val.npy", allow_pickle=True)
# y_val = np.load("./saveData/TSLA/y_val.npy", allow_pickle=True)
# yc_train = np.load("./saveData/TSLA/yc_train.npy", allow_pickle=True)
# yc_test = np.load("./saveData/TSLA/yc_test.npy", allow_pickle=True)

# X_train = np.load("./saveData/SSE/X_train.npy", allow_pickle=True)
# y_train = np.load("./saveData/SSE/y_train.npy", allow_pickle=True)
# X_test = np.load("./saveData/SSE/X_test.npy", allow_pickle=True)
# y_test = np.load("./saveData/SSE/y_test.npy", allow_pickle=True)
# X_val = np.load("./saveData/SSE/X_val.npy", allow_pickle=True)
# y_val = np.load("./saveData/SSE/y_val.npy", allow_pickle=True)
# yc_train = np.load("./saveData/SSE/yc_train.npy", allow_pickle=True)
# yc_test = np.load("./saveData/SSE/yc_test.npy", allow_pickle=True)
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)

X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)

X_val = X_val.astype(np.float32)
y_val = y_val.astype(np.float32)
yc_train = yc_train.astype(np.float32)
yc_test = yc_test.astype(np.float32)
#
# def make_generator_model(input_dim, output_dim, feature_size) -> tf.keras.models.Model:
n_sequence = X_train.shape[1]
n_features = X_train.shape[2]
g_output_dim = y_train.shape[1]

# Parameters
LR = 0.001
lr2 = 1.2
# BATCH_SIZE = 64
# N_EPOCH = 20
dropout = 0.2


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
#
# def make_generator_model(input_dim, output_dim, feature_size) -> tf.keras.models.Model:
n_sequence = X_train.shape[1]
n_features = X_train.shape[2]
g_output_dim = y_train.shape[1]


# METRICS = [
#     keras.metrics.TruePositives(name='tp'),
#     keras.metrics.FalsePositives(name='fp'),
#     keras.metrics.TrueNegatives(name='tn'),
#     keras.metrics.FalseNegatives(name='fn'),
#     keras.metrics.BinaryAccuracy(name='accuracy'),
#     keras.metrics.Precision(name='precision'),
#     keras.metrics.Recall(name='recall'),
#     keras.metrics.AUC(name='auc'),
#     keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
# ]


def make_generator_model():
    # 指定变量的scope
    # with tf.variable_creator_scope("generator", reuse=reuse_g):
    model = Sequential()
    # ,kernel_regularizer = regularizers.l2(0.01)
    model.add(Conv1D(filters=256, kernel_size=2,
                     input_shape=(n_sequence, n_features), activation='elu'))
    model.add(Conv1D(filters=128, kernel_size=2,  activation='elu'
                     ))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(20))
    model.add(LSTM(units=100, return_sequences=True, activation='elu'
                   ))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True, activation='elu'
                   ))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True, activation='elu'))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(100))
    model.add(Dense(g_output_dim, activation='softmax'))
    min_delta_val = 0.0001
    lr_cb = ReduceLROnPlateau(monitor='recall',
                              factor=0.5, min_delta=min_delta_val, patience=10, verbose=1)
    es_cb = EarlyStopping(monitor='recall',
                          min_delta=min_delta_val, patience=10, verbose=1, restore_best_weights=True)

    callbacks_model = [lr_cb, es_cb]
    model.compile(loss='categorical_crossentropy',
                  callbacks=callbacks_model, optimizer=Adam(lr=LR))
    return model

    #  =============================================================================
    # print("model dim: ", input_dim, output_dim)
    # model = Sequential()
    # model.add(LSTM(256, return_sequences=True, input_shape=(
    #     n_sequence, n_features)))
    # model.add(Dropout(dropout))
    # model.add(LSTM(128, return_sequences=True, activation='elu'))
    # model.add(Dropout(dropout))
    # model.add(LSTM(100, activation='elu'))
    # model.add(Dropout(dropout))
    # # model.add(Dense(128))
    # # model.add(Dense(64))
    # model.add(Dense(g_output_dim))
    # # , activation='softmax'
    # model.add(Activation('softmax'))
    # return model
    # =============================================================================
    # # =============================================================================
    # model = Sequential()
    # model.add(Conv1D(filters=1024, kernel_size=5,
    #                  input_shape=(n_sequence, n_features)))
    # model.add(Conv1D(filters=512, kernel_size=5))
    # model.add(MaxPooling1D(pool_size=5))
    # model.add(Conv1D(filters=256, kernel_size=5))
    # model.add(Flatten())
    # model.add(Dense(20))
    # model.add(Dense(g_output_dim, activation='softmax'))
    # # model.add(Dense(16))
    # #model.add(Dense(units=g_output_dim, activation='softmax'))
    # return model
    # model = Sequential()
    # model.add(Conv1D(624, 3, activation='relu',
    #                  input_shape=(n_sequence, n_features)))
    # model.add(Conv1D(624, 3, activation='relu'))
    # model.add(MaxPooling1D(1))
    # model.add(Conv1D(312, 1, activation='relu'))
    # model.add(Conv1D(321, 1, activation='relu'))
    # model.add(GlobalAveragePooling1D())
    # model.add(Dropout(0.5))
    # model.add(Dense(g_output_dim, activation='softmax'))
    # model.summary()
    # return model

#


def make_discriminator_model():

    # # youhua  LSTM当判别器
    # model = Sequential()
    # # model.add(Flatten())
    # # model.add(Dense(units=128, input_shape=(
    # #     (n_sequence+1) * n_features,), activation=None))
    # model.add(LSTM(128, return_sequences=True, input_shape=(3, 1)
    #                ))
    # model.add(LSTM(128, return_sequences=True))
    # model.add(Dropout(dropout))
    # model.add(LSTM(100, return_sequences=True))
    # model.add(Dropout(dropout))
    # model.add(Flatten())
    # model.add(Dense(output_dim, activation='softmax'))
    # return model

    # CNN当判别器
    # model = Sequential()
    # model.add(Conv1D(64, 3, activation='relu',
    #                  input_shape=(3, 1)))
    # model.add(Conv1D(64, 1, activation='relu'))
    # model.add(MaxPooling1D(1))
    # model.add(Conv1D(128, 1, activation='relu'))
    # #model.add(Conv1D(128, 1, activation='relu'))
    # # model.add(GlobalAveragePooling1D())
    # model.add(Dropout(0.5))
    # model.add(Flatten())
    # model.add(Dense(g_output_dim, activation='softmax'))
    # model.summary()
    # return model

    # 多层感知机当判别器
    # with tf.variable_creator_scope("discriminator", reuse=reuse_d):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(units=256, input_shape=(
        (n_sequence) * n_features,), activation=None))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    # model.add(tf.keras.layers.GaussianNoise(stddev=0.2))
    model.add(Dropout(0.5))
    model.add(Dense(units=128, activation=None))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    # model.add(Dropout(0.5)
    # model.add(Dense(units=100, activation=None, kernel_initializer='random_normal'))
    model.add(Dense(units=100, activation=None))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    # model.add(Dense(units=10, activation=None, kernel_initializer='random_normal'))
    model.add(Dense(units=10, activation=None))
    # model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    # model.add(Dense(1 ,activation='softmax'))
    model.add(Dense(g_output_dim))
    model.add(Activation('softmax'))

    return model


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


# 将yc转化为float64
yc_train = yc_train.astype(np.float32)
yc_test = yc_test.astype(np.float32)


class GAN:
    # opt为参数，比如学习率等
    # generator生成器
    # discriminator判别器
    def __init__(self, generator, discriminator, opt):
        self.opt = opt
        self.lr = opt["lr"]
        self.generator = generator
        self.discriminator = discriminator
        # 二进制交叉熵
        # self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # self.cross_entropy = tf.keras.backend.categorical_crossentropy(from_logits=True)
        # 生成器即模型的训练优化器 adam
        self.generator_optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        # early stopping
        # self.generator_callbacks = callbacks
        # 判别器的优化器
        self.discriminator_optimizer = tf.keras.optimizers.Adam(lr=self.lr*lr2)
        self.batch_size = self.opt['bs']
        self.checkpoint_dir = '../training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,

                                              generator=self.generator,
                                              discriminator=self.discriminator)

    # 判别器损失函数
    # 多分类器，需要使用softmax_cross_entropy_with_logits,计算logits和lable之间的softmax交叉熵
    def discriminator_loss(self, y_true, real_output, fake_output):
        # real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        # fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        real_loss = tf.reduce_mean(tf.keras.backend.categorical_crossentropy(
            y_true, real_output, from_logits=False))
        fake_loss = tf.reduce_mean(tf.keras.backend.categorical_crossentropy(
            tf.zeros_like(fake_output), fake_output, from_logits=False))
        # total_loss = tf.add(real_loss, fake_loss)
        # 生成一个负样本，与y_true相反的
        # y_fake = []
        # list_index = [i for i in range(len(y_true))]
        # for i in range(len(y_true)):
        #     if(y_true[i][0] == 1.):
        #         list_index[i] = 0
        #     if(y_true[i][1] == 1.):
        #         list_index[i] = 1
        #     if(y_true[i][2] == 1.):
        #         list_index[i] = 2
        #     if(list_index[i] == 2):
        #         y_fake.append([1., 0., 0.])
        #     else:
        #         y_fake.append([0., 0., 1.])
        # y_fake = np.array(y_fake, dtype=np.float32)
        # # 将y_fake转化为tensor
        # y_fake = tf.convert_to_tensor(y_fake)
        # fake_loss = tf.reduce_mean(
        #     tf.keras.backend.categorical_crossentropy(y_fake, fake_output))
        total_loss = tf.add(real_loss, fake_loss)
        return total_loss
    # 生成器损失函数

    def generator_loss(self, y_true, fake_output):
        # return self.cross_entropy(tf.ones_like(fake_output), fake_output)
        return tf.reduce_mean(tf.keras.backend.categorical_crossentropy(y_true, fake_output, from_logits=False))

    # 获取评价指标

    # %%----------generator生成器的获取评价指标方法----------
    def get_metrics_gen(self, y_true, y_predicted):
        # y_true = y_true.numpy()
        # y_predicted = y_predicted.numpy()
        # print(y_true.shape)
        # print(y_predicted.shape)
        # print(y_true)
        # print(y_predicted)
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
        # y_true = y_true.numpy()
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
        cm = confusion_matrix(y_true, y_pred)
        conf_matrix = pd.DataFrame(
            cm, index=['跌', '平', '涨'], columns=['跌', '平', '涨'])

        # # plot size setting
        # fig, ax = plt.subplots(figsize=(12, 9))
        # sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 19}, cmap="Blues")
        # plt.ylabel('True label', fontsize=18)
        # plt.xlabel('Predicted label', fontsize=18)
        # plt.xticks(fontsize=18)
        # plt.yticks(fontsize=18)
        # plt.savefig('./saveImage/confusion.png',dpi=300)
        # plt.show()

        METRIC = {'Weighted precision': 0, 'Weighted recall': 0, 'Weighted f1-score': 0,
                  'Macro precision': 0, 'Macro recall': 0, 'Macro f1-score': 0,
                  'Micro precision': 0, 'Micro recall': 0, 'Micro f1-score': 0}
        # tp = tf.keras.metrics.TruePositives()
        # tp.update_state(y_true, y_predicted)
        # METRIC['tp'] = tp.result().numpy()
        METRIC['Weighted precision'] = precision_score(
            y_true, y_pred, average='weighted')

        METRIC['Weighted recall'] = recall_score(
            y_true, y_pred, average='weighted')
        METRIC['Weighted f1-score'] = f1_score(y_true,
                                               y_pred, average='weighted')
        METRIC['Macro precision'] = precision_score(
            y_true, y_pred, average='macro')
        METRIC['Macro recall'] = recall_score(y_true, y_pred, average='macro')
        METRIC['Macro f1-score'] = f1_score(y_true, y_pred, average='macro')
        METRIC['Micro precision'] = precision_score(
            y_true, y_pred, average='micro')
        METRIC['Micro recall'] = recall_score(y_true, y_pred, average='micro')
        METRIC['Micro f1-score'] = f1_score(y_true, y_pred, average='micro')

        return METRIC

# %%--------------------------生成器指标获取------------------------
    def get_metrics_disc(self, y_true, y_predicted):
        if(tf.is_tensor(y_predicted)):
            y_predicted = y_predicted.numpy()
        # y_true = y_true.numpy()
        # y_predicted = y_predicted.astype(np.int32)
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
        if(tf.is_tensor(y_true)):
            y_true = y_true.numpy()
        # y_true = y_true.numpy()
        # y_true = y_true.astype(np.int32)
            # 需要把预测的结果转化，即按概率大小把y_predicted转化成0,1

        for i in range(y_true.shape[0]):
            max_index = 0
            if(y_true[i][max_index] < y_true[i][1]):
                max_index = 1
            if(y_true[i][max_index] < y_true[i][2]):
                max_index = 2
            list_temp = [0, 1, 2]
            list_temp.pop(max_index)
            y_true[i][max_index] = 1
            y_true[i][list_temp[0]] = 0
            y_true[i][list_temp[1]] = 0
        # 将y_true和y_pred转化成-1,0,1三类，便于计算混淆矩阵
        y_predicted = y_predicted.astype(np.int32)
        y_true_test = y_true
        y_pred_test = y_predicted
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
        # tp = tf.keras.metrics.TruePositives()
        # tp.update_state(y_true, y_predicted)
        # METRIC['tp'] = tp.result().numpy()
        METRIC['Weighted precision'] = precision_score(
            y_true, y_pred, average='weighted')
        METRIC['Weighted recall'] = recall_score(
            y_true, y_pred, average='weighted')
        METRIC['Weighted f1-score'] = f1_score(y_true,
                                               y_pred, average='weighted')
        METRIC['Macro precision'] = precision_score(
            y_true, y_pred, average='macro')
        METRIC['Macro recall'] = recall_score(y_true, y_pred, average='macro')
        METRIC['Macro f1-score'] = f1_score(y_true, y_pred, average='macro')
        METRIC['Micro precision'] = precision_score(
            y_true, y_pred, average='micro')
        METRIC['Micro recall'] = recall_score(y_true, y_pred, average='micro')
        METRIC['Micro f1-score'] = f1_score(y_true, y_pred, average='micro')

        return METRIC

    @ tf.function
    def train_step(self, real_x, real_y, yc):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # 缺少一个打乱数据real_x的操作，即同时打乱real_x,real_y,yc
            # indices = np.random.permutation(real_x.shape[0])
            # real_x = real_x[indices]
            # real_y = real_y[indices]
            # yc = yc[indices]
            generated_data = self.generator(real_x, training=True)
            generated_data_reshape = tf.reshape(
                generated_data, [generated_data.shape[0], generated_data.shape[1], 1])

            d_fake_input = generated_data_reshape
            d_real_input = tf.reshape(
                real_y, [real_y.shape[0], real_y.shape[1], 1])

            # d_fake_input = tf.concat(
            #     [tf.cast(generated_data_reshape, tf.float32), yc], axis=1)
            # d_fake_input = generated_data_reshape
            # real_y_reshape = tf.reshape(
            #     real_y, [real_y.shape[0], real_y.shape[1], 1])

            # d_real_input = tf.concat(
            #     [tf.cast(real_y_reshape, tf.float32), yc], axis=1)
            # d_real_input = real_y_reshape

            # Reshape for MLP
            # d_fake_input = tf.reshape(d_fake_input, [d_fake_input.shape[0], d_fake_input.shape[1]])
            # d_real_input = tf.reshape(d_real_input, [d_real_input.shape[0], d_real_input.shape[1]])

            real_output = self.discriminator(d_real_input, training=True)
            fake_output = self.discriminator(d_fake_input, training=True)

            # 用LSTM或CNN做判别器
            # fake_output = self.discriminator(tf.reshape(
            #     d_fake_input, [d_fake_input.shape[0], d_fake_input.shape[1], d_fake_input.shape[2]]))
            # real_output = self.discriminator(tf.reshape(
            #     d_real_input, [d_real_input.shape[0], d_real_input.shape[1], d_real_input.shape[2]]))
            # #将结果转化成0,1
            # real_output = result_to_0_1(real_output)
            # fake_output = result_to_0_1(fake_output)

            # y_fake = []
            # y_true = real_y
            # list_index = [i for i in range(len(y_true))]
            # for i in range(len(y_true)):
            #     if(y_true[i][0] == 1.):
            #         list_index[i] = 0
            #     if(y_true[i][1] == 1.):
            #         list_index[i] = 1
            #     if(y_true[i][2] == 1.):
            #         list_index[i] = 2
            #     if(list_index[i] == 2):  # 涨的负类,就是跌
            #         y_fake.append([1., 0., 0.])
            #     else:
            #         y_fake.append([0., 0., 1.])
            # y_fake = np.array(y_fake, dtype=np.float32)
            # # 将y_fake转化为tensor
            # y_fake = tf.convert_to_tensor(y_fake)

            # gen_loss = self.generator_loss(fake_output)
            gen_loss = self.generator_loss(
                real_y.astype(np.float32), generated_data)
            disc_loss = self.discriminator_loss(
                real_y.astype(np.float32), real_output, fake_output)

           # gen_metric,y_true_gen = self.get_metrics_gen(real_y,generated_data)
            gen_metric = self.get_metrics_gen(real_y, generated_data)
            disc_metric1 = self.get_metrics_disc(real_y, real_output)
            y_fake = tf.zeros_like(fake_output)
            disc_metric2 = self.get_metrics_disc(y_fake, fake_output)
            disc_metric = {}

            for key, value in disc_metric2.items():
                disc_metric[key] = (disc_metric1[key]+disc_metric2[key])/2
            # disc_metric2 = self.get_metrics_disc(,fake_output)
            # disc_metric = self.get_metrics_disc(y_true_gen,fake_output)
            print(
                "--------------------------------------------------------------------------")

        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return real_y, generated_data, {'d_loss': disc_loss, 'g_loss': gen_loss}, gen_metric, disc_metric

    def train(self, real_x, real_y, yc, opt):

        train_hist = {}
        train_hist['D_losses'] = []
        train_hist['G_losses'] = []
        train_hist['per_epoch_times'] = []
        train_hist['total_ptime'] = []
        train_hist['D_Weighted precision'] = []
        train_hist['G_Weighted precision'] = []
        train_hist['D_Weighted recall'] = []
        train_hist['G_Weighted recall'] = []

        train_hist['D_Weighted f1-score'] = []
        train_hist['G_Weighted f1-score'] = []

        train_hist['D_Macro precision'] = []
        train_hist['G_Macro precision'] = []

        train_hist['D_Macro recall'] = []
        train_hist['G_Macro recall'] = []

        train_hist['D_Macro f1-score'] = []
        train_hist['G_Macro f1-score'] = []

        train_hist['D_Micro precision'] = []
        train_hist['G_Micro precision'] = []

        train_hist['D_Micro recall'] = []
        train_hist['G_Micro recall'] = []
        train_hist['D_Micro f1-score'] = []
        train_hist['G_Micro f1-score'] = []

        # 加入分类任务指标
        # 1.tp
        # 2.fp
        # 3.tn
        # 4.fn
        # 5.accuracy
        # 6.precision
        # 7.recall
        # 8.auc
        # 9.prc

        epochs = opt["epoch"]
        for epoch in range(epochs):
            start = time.time()

            real_price, fake_price, loss, gen_metric, disc_metric = self.train_step(
                real_x, real_y, yc)

            G_losses = []
            D_losses = []

            Real_price = []
            Predicted_price = []

            D_losses.append(loss['d_loss'].numpy())
            G_losses.append(loss['g_loss'].numpy())

            Predicted_price.append(fake_price.numpy())
            # Real_price.append(real_price.numpy())
            Real_price.append(real_price)
            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                tf.keras.models.save_model(
                    generator, './saveModel/gen_model_3_1_%d.h5' % epoch)
                # self.checkpoint.save(file_prefix=self.checkpoint_prefix + f'-{epoch}')
                # print('epoch', epoch + 1, 'd_loss:', loss['d_loss'].numpy(), 'g_loss', loss['g_loss'].numpy())
            print('epoch:', epoch + 1, '-d_loss:', loss['d_loss'].numpy(), '-d_w_pre:', disc_metric['Weighted precision'], '-d_w_recall:', disc_metric['Weighted recall'],
                  '-d_w_f1-score:', disc_metric['Weighted f1-score'], '-d_Mac_pre:', disc_metric[
                      'Macro precision'], '-d_Mac_recall:', disc_metric['Macro recall'], '-d_Mac_f1-score:', disc_metric['Macro f1-score'],
                  '-d_Mic_pre:', disc_metric['Micro precision'], '-d_Mic_recall:', disc_metric[
                      'Micro recall'], '-d_Mic_f1-score:', disc_metric['Micro f1-score'],

                  '-g_loss:', loss['g_loss'].numpy(
            ), '-g_w_pre:', gen_metric['Weighted precision'], '-g_w_recall:', gen_metric['Weighted recall'],
                '-g_w_f1-score:', gen_metric['Weighted f1-score'], '-g_Mac_pre:', gen_metric[
                      'Macro precision'], '-g_Mac_recall:', gen_metric['Macro recall'], '-g_Mac_f1-score:', gen_metric['Macro f1-score'],
                '-g_Mic_pre:', gen_metric['Micro precision'], '-g_Mic_recall:', gen_metric['Micro recall'], '-g_Mic_f1-score:', gen_metric['Micro f1-score'],)

            # print('------Weighted------')
            # print('Weighted precision', precision_score(
            #     y_true, y_pred, average='weighted'))
            # print('Weighted recall', recall_score(y_true, y_pred, average='weighted'))
            # print('Weighted f1-score', f1_score(y_true, y_pred, average='weighted'))
            # print('------Macro------')
            # print('Macro precision', precision_score(y_true, y_pred, average='macro'))
            # print('Macro recall', recall_score(y_true, y_pred, average='macro'))
            # print('Macro f1-score', f1_score(y_true, y_pred, average='macro'))
            # print('------Micro------')
            # print('Micro precision', precision_score(y_true, y_pred, average='micro'))
            # print('Micro recall', recall_score(y_true, y_pred, average='micro'))
            # print('Micro f1-score', f1_score(y_true, y_pred, average='micro'))
            # print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
            # For printing loss
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - start
            train_hist['D_losses'].append(D_losses)
            train_hist['G_losses'].append(G_losses)
            train_hist['per_epoch_times'].append(per_epoch_ptime)

            train_hist['D_Weighted precision'].append(
                disc_metric['Weighted precision'])
            train_hist['G_Weighted precision'].append(
                gen_metric['Weighted precision'])
            train_hist['D_Weighted recall'].append(
                disc_metric['Weighted recall'])
            train_hist['G_Weighted recall'].append(
                gen_metric['Weighted recall'])

            train_hist['D_Weighted f1-score'].append(
                disc_metric['Weighted f1-score'])
            train_hist['G_Weighted f1-score'].append(
                gen_metric['Weighted f1-score'])

            train_hist['D_Macro precision'].append(
                disc_metric['Macro precision'])
            train_hist['G_Macro precision'].append(
                gen_metric['Macro precision'])

            train_hist['D_Macro recall'].append(disc_metric['Macro recall'])
            train_hist['G_Macro recall'].append(gen_metric['Macro recall'])

            train_hist['D_Macro f1-score'].append(
                disc_metric['Macro f1-score'])
            train_hist['G_Macro f1-score'].append(gen_metric['Macro f1-score'])

            train_hist['D_Micro precision'].append(
                disc_metric['Micro precision'])
            train_hist['G_Micro precision'].append(
                gen_metric['Micro precision'])
            train_hist['D_Micro recall'].append(disc_metric['Micro recall'])
            train_hist['G_Micro recall'].append(gen_metric['Micro recall'])
            train_hist['D_Micro f1-score'].append(
                disc_metric['Micro f1-score'])
            train_hist['G_Micro f1-score'].append(gen_metric['Micro f1-score'])

        # Reshape the predicted result & real
        Predicted_price = np.array(Predicted_price)
        Predicted_price = Predicted_price.reshape(
            Predicted_price.shape[1], Predicted_price.shape[2])
        Real_price = np.array(Real_price)
        Real_price = Real_price.reshape(
            Real_price.shape[1], Real_price.shape[2])

        plt.figure()
        plt.plot(train_hist['D_losses'], label='D_loss')
        plt.plot(train_hist['G_losses'], label='G_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('./saveImage/train_G_D_loss.png', dpi=300)
        # plt.show()

        return Predicted_price, Real_price, np.sqrt(mean_squared_error(Real_price, Predicted_price)) / np.mean(
            Real_price), train_hist


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


# 主函数部分
if __name__ == '__main__':
    input_dim = X_train.shape[1]
    feature_size = X_train.shape[2]
    output_dim = y_train.shape[1]

    # For Bayesian
    # opt = {"lr": 0.00016, "epoch": 500, 'bs': 500}
    opt = {"lr": 0.001, "epoch": 500, 'bs': 30}

    # generator = make_generator_model(X_train.shape[1], output_dim, X_train.shape[2])
    generator = make_generator_model()
    #reuse_g = False
    discriminator = make_discriminator_model()
    gan = GAN(generator, discriminator, opt)
    Predicted_price, Real_price, RMSPE, history = gan.train(
        X_train, y_train, yc_train, opt)

    history_dict = history
    history_dict.keys()

    G_acc = history_dict['G_Micro f1-score']
    D_acc = history_dict['D_Micro f1-score']
    G_loss = history_dict['G_losses']
    D_loss = history_dict['D_losses']

    epochs = range(1, len(G_acc) + 1)

    # “bo”代表 "蓝点"
    plt.figure()
    plt.plot(epochs, G_loss, 'bo', label='GAN_Training loss')
    # b代表“蓝色实线”
    plt.plot(epochs, D_loss, 'b',  label='Dis_Training loss')
    plt.title('Gen and Disc Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./saveImage/G_D_loss.png', dpi=300)
    # plt.show()
    # 在测试集上进行测试
    # G_model = tf.keras.models.load_model('./saveModel/gen_model_3_1_194.h5')

    G_model = generator
    y_predicted = G_model(X_test)

    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn import metrics
    from sklearn.preprocessing import label_binarize

    # y_hat = G_model(X_test)
    y_hat = y_predicted
    result = {}
    result = get_metrics(y_test, y_hat)
    # 计算属于各个类别的概率，返回值的shape = [n_samples, n_classes]
    y_score = G_model(X_test).numpy()
    # y_score = discriminator(generator(X_test)).numpy()
    # 1、调用函数计算micro类型的AUC
    print('调用函数auc：', metrics.roc_auc_score(y_test, y_score, average='micro'))
    # 2、手动计算micro类型的AUC
    # 首先将矩阵y_one_hot和y_score展开，然后计算假正例率FPR和真正例率TPR
    fpr, tpr, thresholds = metrics.roc_curve(y_test.ravel(), y_score.ravel())
    auc = metrics.auc(fpr, tpr)
    print('手动计算auc：', auc)
    # 绘图
    mpl.rcParams['font.sans-serif'] = u'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    # FPR就是横坐标,TPR就是纵坐标
    plt.figure()
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
    plt.savefig('./saveImage/ROC_AUC.png', dpi=300)
    # plt.show()


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


y = result_to_0_1(y_hat)

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 3
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])
# Compute micro-average ROC curve and ROC area（方法二）
fpr["weighted"], tpr["weighted"], _ = roc_curve(
    y_test.ravel(), y_score.ravel())
roc_auc["weighted"] = metrics.auc(fpr["weighted"], tpr["weighted"])


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
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.4f})'
#          ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.4f})'
         ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
plt.plot(fpr["weighted"], tpr["weighted"],
         label='weighted-average ROC curve (area = {0:0.4f})'
         ''.format(roc_auc["weighted"]),
         color='deeppink', linestyle=':', linewidth=4)

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
plt.savefig('./saveImage/Result.png', dpi=300)
# plt.show()
for key in result:
    print(str(key) + '： ' + str(result[key]))

y_pred = to_0_1(y_hat)
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
plt.savefig('./saveImage/confusion_matrix')
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
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.4f})'
#          ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)

# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (area = {0:0.4f})'
#          ''.format(roc_auc["macro"]),
#          color='navy', linestyle=':', linewidth=4)

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
plt.savefig('./saveImage/Result_0_1.png', dpi=300)
