import math
from sklearn.metrics import mean_squared_error as mse_two
import numpy as np
import pandas as pd
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout, LeakyReLU, Flatten, Conv1D, RepeatVector, MaxPooling1D, Bidirectional

import time
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.layers import Lambda
#from keras.backend import slice

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Ploting
import matplotlib.pyplot as plt


data_names = ["S&P500", "SSE", "IBM", "MSFT", "PAICC"]
data_name = data_names[0]
# data_path="stock_market_GAN/datasets/"+data_name+".csv"
data_path = "./datasets/"+data_name+".csv"
print(data_path)
dataframe = pd.read_csv(data_path)

# 计算以步长为5的移动平均指数


def add_Ma(dataframe):
    Ma_window = 5
    for i in range(0, dataframe.shape[0]-Ma_window):
        dataframe.loc[dataframe.index[i+Ma_window], 'Ma'] = np.round(
            ((dataframe.iloc[i, 4] + dataframe.iloc[i+1, 4] + dataframe.iloc[i+2, 4] + dataframe.iloc[i+3, 4] + dataframe.iloc[i+4, 4])/5), 6)
    return dataframe[5:-5]


dataframe = add_Ma(dataframe)
# 时间序列生成器


class Standarized_TimeseriesGenerator(tf.keras.preprocessing.sequence.TimeseriesGenerator):
    def __getitem__(self, index):
        samples, targets = super(
            Standarized_TimeseriesGenerator, self).__getitem__(index)
        # shape : (n_batch, n_sequence, n_features)
        mean = samples.mean(axis=1)
        std = samples.std(axis=1)
        # standarize along each feature
        samples = (samples - mean[:, None, :])/std[:, None, :]
        # targets = (targets - mean[..., 3])/std[..., 3] # The close value is our target
        targets = (targets - mean)/std  # The close value is our target
        return samples, targets

# TimeseriesGenerator Check


data = np.array([[i, i**2, i**3, i**4] for i in range(11)])
#targets = np.array([[i**4] for i in range(11)])
targets = data

print(targets.shape)

mean = data[:-1].mean(axis=0)[None, :]
std = data[:-1].std(axis=0)[None, :]

data_gen = Standarized_TimeseriesGenerator(data, targets,
                                           length=10, sampling_rate=1,
                                           batch_size=2)


n_sequence = 5
n_features = 7
n_batch = 32


def get_gen_train_test(dataframe):
    data = dataframe.drop(columns='Date').to_numpy()
    # targets = data[:,3, None] #add none to have same number of dimensions as data
    targets = data
    n_samples = data.shape[0]
    train_test_split = int(n_samples*0.9)

    data_gen_train = Standarized_TimeseriesGenerator(data, targets,
                                                     length=n_sequence, sampling_rate=1,
                                                     stride=1, batch_size=n_batch,
                                                     start_index=0,
                                                     end_index=train_test_split,
                                                     shuffle=True)
    data_gen_test = Standarized_TimeseriesGenerator(data, targets,
                                                    length=n_sequence, sampling_rate=1,
                                                    stride=1, batch_size=n_batch,
                                                    start_index=train_test_split,
                                                    end_index=n_samples-1)

    return data_gen_train, data_gen_test


data_gen_train, data_gen_test = get_gen_train_test(dataframe)

# baseline: use previous day as estimation for the next
squared_error = 0


def mean_squared_error(X, lenght=5):
    squared_error = 0
    for i in range(0, X.shape[0] - lenght):
        x = X[i:i+lenght]
        mean = x.mean()
        std = x.std()
        x = (x - mean)/std
        y = (X[i+lenght] - mean)/std
        squared_error += np.square(x[-1]-y)
    return squared_error/X.shape[0]


baseline_error = mean_squared_error(data[:, 3])


data_gen = Standarized_TimeseriesGenerator(data, targets,
                                           length=5, sampling_rate=1,
                                           stride=1, batch_size=32)

# baseline: use previous day as estimation for the next


def mean_squared_error(dataset):
    mse = 0
    for X_batch, y_batch in dataset:
        # X_batch.shape : (n_batch, n_sequence, n_features)
        #mse += np.mean(np.square(X_batch[:, -1, 3:4]-y_batch))
        mse += np.mean(np.square(X_batch[:, -1, 3:4]-y_batch[:, 3:4]))
    mse /= len(dataset)
    return mse


baseline_error = mean_squared_error(data_gen)

# Metrics not included in keras

# For some reason keras mape is different so it has to be customly defined.
# Paper definition isn't percentual, hence the difference. (x100)


def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true[:, 3]-y_pred[:, 3]))


def mae(y_true, y_pred):
    return tf.reduce_mean(tf.keras.backend.abs((y_true[:, 3]-y_pred[:, 3])))
    # ***The absolute is over the whole thing as y_true can be negative


def mape(y_true, y_pred):
    return tf.reduce_mean(tf.keras.backend.abs((y_true[:, 3]-y_pred[:, 3])/y_true[:, 3]))
    # ***The absolute is over the whole thing as y_true can be negative


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true[:, 3]-y_pred[:, 3])))


def ar(y_true, y_pred):
    mask = tf.cast(y_pred[1:, 3] > y_true[:-1, 3], tf.float32)
    return tf.reduce_mean((y_true[1:, 3]-y_true[:-1, 3])*mask)


def discriminator_loss(real_output, fake_output):
    # 二进制交叉熵
    real_loss = tf.keras.losses.binary_crossentropy(
        tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.binary_crossentropy(
        tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(x, y, fake_output):
    a1 = 0.01
    g_loss = tf.keras.losses.binary_crossentropy(
        tf.ones_like(fake_output), fake_output)
    g_mse = tf.keras.losses.MSE(x, y)
    return a1*g_mse + (1-a1)*g_loss, g_mse


def make_generator_model():

    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=2, activation='relu',
                     input_shape=(n_sequence, n_features),
                     kernel_regularizer=regularizers.l2(0.01)))
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(30))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True,
                   kernel_regularizer=(regularizers.l2(0.01))))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Bidirectional(LSTM(128, activation='relu')))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_features))
    opt2 = optimizers.Adam(lr=0.0001, decay=0.001)
    model.compile(loss=None, metrics=[
                  mse, mae, mape, rmse, ar], optimizer=opt2)
    model.summary()
    return model


def make_generator_model_beifen():

    inputs = Input(shape=(n_sequence, n_features,))
    lstm_1 = LSTM(units=10, return_sequences=True, activation=None,
                  kernel_initializer='random_normal')(inputs)
    batch_norm1 = tf.keras.layers.BatchNormalization()(lstm_1)
    lstm_1_LRelu = LeakyReLU(alpha=0.3)(batch_norm1)
    lstm_1_droput = Dropout(0.3)(lstm_1_LRelu)
    lstm_2 = LSTM(units=10, return_sequences=False, activation=None,
                  kernel_initializer='random_normal')(lstm_1_droput)
    batch_norm2 = tf.keras.layers.BatchNormalization()(lstm_2)
    lstm_2_LRelu = LeakyReLU(alpha=0.3)(batch_norm2)
    lstm_2_droput = Dropout(0.3)(lstm_2_LRelu)
    #lstm_3 = LSTM(units=10, return_sequences = False, activation=None, kernel_initializer='random_normal')(lstm_2_droput)
    # batch_norm3=tf.keras.layers.BatchNormalization()(lstm_3)
    #lstm_3_LRelu = LeakyReLU(alpha=0.3)(batch_norm3)
    #lstm_3_droput = Dropout(0.3)(lstm_3_LRelu)
    #lstm_4 = LSTM(units=100, return_sequences = False, activation=None, kernel_initializer='random_normal')(lstm_3_droput)
    # batch_norm4=tf.keras.layers.BatchNormalization()(lstm_4)
    #lstm_4_LRelu = LeakyReLU(alpha=0.3)(batch_norm4)
    #lstm_4_droput = Dropout(0.5)(lstm_4_LRelu)
    output_dense = Dense(n_features, activation=None)(lstm_2_droput)
    output = LeakyReLU(alpha=0.3)(output_dense)

    #prediction = Lambda( lambda x: x[..., 3:4])(output)
    #slice_model = Model(inputs = inputs, outputs = prediction)
    #slice_model.compile(loss='mse', metrics = ['mse', 'mae', 'mape', rmse, ar])
    # slice_model.summary()

    model = Model(inputs=inputs, outputs=output)
    # model.compile(loss=generator_loss)
    model.compile(loss=None, metrics=[mse, mae, mape, rmse, ar])
    #model.compile(loss=None, metrics = [mse , mae, mape, rmse])
    model.summary()

    # return model, slice_model
    return model

    # history = model.fit(data_gen_train, validation_data=data_gen_test, epochs = 100,
    #                   callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5))


#generator_test = make_generator_model_beifen()
generator = make_generator_model()


def make_discriminator_model():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(units=72, input_shape=((n_sequence+1) * n_features,),
                    activation=None, kernel_initializer='random_normal'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(tf.keras.layers.GaussianNoise(stddev=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(units=100, activation=None,
                    kernel_initializer='random_normal'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(Dropout(0.3))
    model.add(Dense(units=10, activation=None,
                    kernel_initializer='random_normal'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    opt2 = optimizers.Adam(lr=0.0001, decay=0.001)
    model.compile(loss='discriminator_loss', optimizer=opt2)
    # model.compile(loss=discriminator_loss)
    # model.summary()
    # history = model.fit(data_gen_train, validation_data=data_gen_test, epochs = 100,
    #                   callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5))

    return model


discriminator = make_discriminator_model()
learning_rate = 1e-4
generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".


def train_step_def(sequences, sequences_end):
    # sequences is the real output

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_prediction = generator(sequences, training=True)

        sequences_true = tf.concat(
            (sequences, sequences_end[:, None, :]), axis=1)
        sequences_fake = tf.concat(
            (sequences, generated_prediction[:, None, :]), axis=1)

        real_output = discriminator(sequences_true, training=True)
        fake_output = discriminator(sequences_fake, training=True)

        gen_loss, gen_mse_loss = generator_loss(generated_prediction,
                                                sequences_end,
                                                fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))

    return tf.reduce_mean(gen_loss), tf.reduce_mean(disc_loss), tf.reduce_mean(gen_mse_loss)


def test_step_def(sequences, sequences_end):
    generated_prediction = generator(sequences, training=False)

    sequences_true = tf.concat((sequences, sequences_end[:, None, :]), axis=1)
    sequences_fake = tf.concat(
        (sequences, generated_prediction[:, None, :]), axis=1)

    real_output = discriminator(sequences_true, training=False)
    fake_output = discriminator(sequences_fake, training=False)

    gen_loss, gen_mse_loss = generator_loss(
        generated_prediction, sequences_end, fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)
    return tf.reduce_mean(gen_loss), tf.reduce_mean(disc_loss), tf.reduce_mean(gen_mse_loss)


def train(dataset, dataset_val, epochs):
    history = np.empty(shape=(8, epochs))
    history_val = np.empty(shape=(8, epochs))
    len_dataset = len(dataset)
    len_dataset_val = len(dataset_val)
    for epoch in range(epochs):
        start = time.time()

        cur_dis_loss = 0
        cur_gen_loss = 0
        cur_gen_mse_loss = 0
        for sequence_batch, sequence_end_batch in dataset:
            aux_cur_losses = train_step(tf.cast(sequence_batch, tf.float32),
                                        tf.cast(sequence_end_batch, tf.float32))
            cur_gen_loss += aux_cur_losses[0]/len_dataset
            cur_dis_loss += aux_cur_losses[1]/len_dataset
            cur_gen_mse_loss += aux_cur_losses[2]/len_dataset

        #cur_gen_loss = generator.evaluate(dataset,verbose=False)
        cur_gen_metrics = generator.evaluate(dataset, verbose=False)[1:]
        #cur_dis_loss = discriminator.evaluate(dataset,verbose=False)

        history[:, epoch] = cur_gen_loss, cur_dis_loss, cur_gen_mse_loss, * \
            cur_gen_metrics
        #history[:, epoch] = cur_gen_loss, *cur_gen_slice_metrics

        #cur_gen_loss_val = generator.evaluate(dataset_val,verbose=False)
        cur_gen_metrics_val = generator.evaluate(
            dataset_val, verbose=False)[1:]

        #cur_dis_loss_val = discriminator.evaluate(dataset_val,verbose=False)

        cur_gen_loss_val = 0
        cur_dis_loss_val = 0
        cur_gen_mse_loss_val = 0
        for sequence_batch, sequence_end_batch in dataset_val:
            aux_cur_losses_val = test_step(tf.cast(sequence_batch, tf.float32),
                                           tf.cast(sequence_end_batch, tf.float32))
            cur_gen_loss_val += aux_cur_losses_val[0]/len_dataset_val
            cur_dis_loss_val += aux_cur_losses_val[1]/len_dataset_val
            cur_gen_mse_loss_val += aux_cur_losses_val[2]/len_dataset_val

        history_val[:, epoch] = cur_gen_loss_val, cur_dis_loss_val, cur_gen_mse_loss_val, * \
            cur_gen_metrics_val

        print('Time for epoch {} is {} sec Generator Loss: {},  Discriminator_loss: {}'
              .format(epoch + 1, time.time()-start, cur_gen_loss, cur_dis_loss))

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

    return history, history_val


metrics = ["gen_loss", "dis_loss", "gen_mse_loss",
           'mse', 'mae', 'mape', 'rmse', 'ar']


def plot_history(history, history_val):
    for i, metric_name in enumerate(metrics):
        plt.figure()
        plt.title(metric_name)
        plt.plot(history[i], label='train')
        plt.plot(history_val[i], label='test')
        plt.legend()
    plt.savefig('./image/history.png')
    # plt.show()

#plot_history(history, history_val)


def return_MAPE(test, predicted):
    MAPE1 = 0
    for i in range(len(test)):
        MAPE1 += abs((test[i]-predicted[i])/test[i])
    MAPE1 = MAPE1 / len(test)
    print("The Mean Absolute Percentage Erro is {}.".format(MAPE1))


def return_rmse(test, predicted):
    rmse = math.sqrt(mse_two(test, predicted))
    print("The root mean squared error is {}.".format(rmse))


def plot_frame(sequence, target, model):
    sequence, target = data_gen_test[0]
    y_pred = model.predict(sequence)[..., 3]
    y_true = target[..., 3]

    plt.figure()
    plt.title("closing price")
    plt.plot(y_true, label="true")
    plt.plot(y_pred, label="prediction")
    plt.legend()
    # plt.show()
    plt.savefig('./image/predicted.png')
    # 此处输出模型预测结果的MSE 与MAEP
    return_rmse(y_true, y_pred)
    return_MAPE(y_true, y_pred)

#plot_frame(*data_gen_test[0], generator)


def get_best_results(history):
    # get best mse
    min_index = np.argmin(history[3, :])

    return history[:, min_index]

# get_best_results(history_val)


results = np.zeros((6, 8))
# ----------------------------训练-------------------------------
i = 0
EPOCHS = 10
data_name = data_names[i]
data_path = "./datasets/"+data_name+".csv"
data_path = "./datasets/"+data_name+".csv"
print(data_path)
dataframe = pd.read_csv(data_path).dropna()

dataframe = add_Ma(dataframe)

# plot_dataframe(dataframe)

data_gen_train, data_gen_test = get_gen_train_test(dataframe)

generator = make_generator_model()
discriminator = make_discriminator_model()

learning_rate = 1e-4
generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)


@tf.function
def train_step(sequences, sequences_end):
    return train_step_def(sequences, sequences_end)


@tf.function
def test_step(sequences, sequences_end):
    return test_step_def(sequences, sequences_end)


checkpoint_dir = './training_checkpoints'+data_name
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

history, history_val = train(data_gen_train, data_gen_test, EPOCHS)

plot_history(history, history_val)
plot_frame(*data_gen_test[0], generator)

print("[MSE Baseline] train:", mean_squared_error(
    data_gen_train), " test:", mean_squared_error(data_gen_test))

results[i] = get_best_results(history_val)
print(metrics, "=\n", results[i])


# =============================================================================
# #SSE数据集上
# i = 1
# data_name = data_names[i]
# data_path="./datasets/"+data_name+".csv"
# print(data_path)
# dataframe = pd.read_csv(data_path)[1768:].dropna()
#
# dataframe = add_Ma(dataframe)
#
# #plot_dataframe(dataframe)
#
# data_gen_train, data_gen_test = get_gen_train_test(dataframe)
#
# generator = make_generator_model()
# discriminator=make_discriminator_model()
#
# learning_rate=1e-4
# generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
# discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
#
# @tf.function
# def train_step(sequences, sequences_end):
#   return train_step_def(sequences, sequences_end)
#
# @tf.function
# def test_step(sequences, sequences_end):
#   return test_step_def(sequences, sequences_end)
#
# checkpoint_dir = './training_checkpoints'+data_name
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)
#
# history, history_val = train(data_gen_train, data_gen_test, EPOCHS)
#
# plot_history(history, history_val)
# plot_frame(*data_gen_test[0], generator)
#
# print("[MSE Baseline] train:",mean_squared_error(data_gen_train)," test:", mean_squared_error(data_gen_test))
#
# results[i] = get_best_results(history_val)
# print(metrics,"=\n",results[i])
# =============================================================================


# =============================================================================
# #IBM数据集
# i = 2
# data_name = data_names[i]
# data_path="stock_market_GAN/datasets/"+data_name+".csv"
# print(data_path)
# dataframe = pd.read_csv(data_path).dropna()
#
# dataframe = add_Ma(dataframe)
#
# #plot_dataframe(dataframe)
#
# data_gen_train, data_gen_test = get_gen_train_test(dataframe)
#
# generator = make_generator_model()
# discriminator=make_discriminator_model()
#
# learning_rate=1e-4
# generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
# discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
#
# @tf.function
# def train_step(sequences, sequences_end):
#   return train_step_def(sequences, sequences_end)
#
# @tf.function
# def test_step(sequences, sequences_end):
#   return test_step_def(sequences, sequences_end)
#
# checkpoint_dir = './training_checkpoints'+data_name
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)
#
# history, history_val = train(data_gen_train, data_gen_test, EPOCHS)
#
# plot_history(history, history_val)
# plot_frame(*data_gen_test[0], generator)
#
# print("[MSE Baseline] train:",mean_squared_error(data_gen_train)," test:", mean_squared_error(data_gen_test))
#
# results[i] = get_best_results(history_val)
# print(metrics,"=\n",results[i])
# =============================================================================


# =============================================================================
# #MSFT数据集
# i = 3
# data_name = data_names[i]
# data_path="./datasets/"+data_name+".csv"
# print(data_path)
# dataframe = pd.read_csv(data_path).dropna()
#
# dataframe = add_Ma(dataframe)
#
# #plot_dataframe(dataframe)
#
# data_gen_train, data_gen_test = get_gen_train_test(dataframe)
#
# generator = make_generator_model()
# discriminator=make_discriminator_model()
#
# learning_rate=1e-4
# generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
# discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
#
# @tf.function
# def train_step(sequences, sequences_end):
#   return train_step_def(sequences, sequences_end)
#
# @tf.function
# def test_step(sequences, sequences_end):
#   return test_step_def(sequences, sequences_end)
#
# checkpoint_dir = './training_checkpoints'+data_name
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)
#
# history, history_val = train(data_gen_train, data_gen_test, EPOCHS)
#
# plot_history(history, history_val)
# plot_frame(*data_gen_test[0], generator)
#
# print("[MSE Baseline] train:",mean_squared_error(data_gen_train)," test:", mean_squared_error(data_gen_test))
#
# results[i] = get_best_results(history_val)
# print(metrics,"=\n",results[i])
# =============================================================================


# =============================================================================
# #PAICC数据集
# i = 4
# data_name = data_names[i]
# data_path="./datasets/"+data_name+".csv"
# print(data_path)
#
# #datos estaticos en dataframe[1479:1524]
# dataframe = pd.read_csv(data_path)[1534:].dropna()
#
# dataframe = add_Ma(dataframe)
#
# #plot_dataframe(dataframe)
#
# data_gen_train, data_gen_test = get_gen_train_test(dataframe)
#
# generator = make_generator_model()
# discriminator=make_discriminator_model()
#
# learning_rate=1e-4
# generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
# discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
#
# @tf.function
# def train_step(sequences, sequences_end):
#   return train_step_def(sequences, sequences_end)
#
# @tf.function
# def test_step(sequences, sequences_end):
#   return test_step_def(sequences, sequences_end)
#
# checkpoint_dir = './training_checkpoints'+data_name
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)
#
# history, history_val = train(data_gen_train, data_gen_test, EPOCHS)
#
# plot_history(history, history_val)
# plot_frame(*data_gen_test[0], generator)
#
# print("[MSE Baseline] train:",mean_squared_error(data_gen_train)," test:", mean_squared_error(data_gen_test))
#
# results[i] = get_best_results(history_val)
# print(metrics,"=\n",results[i])
# =============================================================================


# =============================================================================
# #CSI数据集
# i = 5
# data_name = data_names[i]
# data_path="./datasets/"+data_name+".csv"
# print(data_path)
#
# #datos estaticos en dataframe[1479:1524]
# dataframe = pd.read_csv(data_path).dropna()
#
# dataframe = add_Ma(dataframe)
#
# #plot_dataframe(dataframe)
#
# data_gen_train, data_gen_test = get_gen_train_test(dataframe)
#
# generator = make_generator_model()
# discriminator=make_discriminator_model()
#
# learning_rate=1e-4
# generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
# discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
#
# @tf.function
# def train_step(sequences, sequences_end):
#   return train_step_def(sequences, sequences_end)
#
# @tf.function
# def test_step(sequences, sequences_end):
#   return test_step_def(sequences, sequences_end)
#
# checkpoint_dir = './training_checkpoints'+data_name
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)
#
# history, history_val = train(data_gen_train, data_gen_test, EPOCHS)
#
# plot_history(history, history_val)
# plot_frame(*data_gen_test[0], generator)
#
# print("[MSE Baseline] train:",mean_squared_error(data_gen_train)," test:", mean_squared_error(data_gen_test))
#
# results[i] = get_best_results(history_val)
# print(metrics,"=\n",results[i])
# =============================================================================

# =============================================================================
# #结果展示
# df = pd.DataFrame(data = results, index = data_names, columns = metrics)
# df.loc['mean'] = results.mean(axis=0)
# df.head(6)
# =============================================================================
