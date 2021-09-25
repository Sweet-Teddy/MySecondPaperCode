# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 10:41:26 2021

@author: Administrator
"""
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Dense, Flatten, Conv1D, BatchNormalization, LeakyReLU, Dropout, Activation
from sklearn.metrics import mean_squared_error
from pickle import load
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
File_Path = []
# 获取到当前文件的目录，并检查是否有对应文件夹，如果不存在则自动新建文件
File_Path.append(os.getcwd()[:] + '\\saveModel\\')
File_Path.append(os.getcwd()[:] + '\\saveData\\')
File_Path.append(os.getcwd()[:] + '\\saveData\\oneInput\\')
File_Path.append(os.getcwd()[:] + '\\saveImage\\')
for str in File_Path:
    if not os.path.exists(str):
        os.makedirs(str)
    else:
        print(str + "文件夹已存在，无需创建")

#from main.feature import get_all_features

X_train = np.load("./saveData/X_train.npy", allow_pickle=True)
y_train = np.load("./saveData/y_train.npy", allow_pickle=True)
X_test = np.load("./saveData/X_test.npy", allow_pickle=True)
y_test = np.load("./saveData/y_test.npy", allow_pickle=True)
X_val = np.load("./saveData/X_val.npy", allow_pickle=True)
y_val = np.load("./saveData/y_val.npy", allow_pickle=True)
yc_train = np.load("./saveData/yc_train.npy", allow_pickle=True)
yc_test = np.load("./saveData/yc_test.npy", allow_pickle=True)

# 生成器
# def make_generator_model(input_dim, output_dim, feature_size) -> tf.keras.models.Model:
n_sequence = X_train.shape[1]
n_features = X_train.shape[2]
g_output_dim = y_train.shape[1]


def make_generator_model():
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
    #output_dense = Dense(n_features, activation='softmax')(lstm_2_droput)
    output_dense = Dense(g_output_dim, activation='softmax')(lstm_2_droput)
    #output = LeakyReLU(alpha=0.3)(output_dense)
    output = Activation('softmax')(output_dense)

    #prediction = Lambda( lambda x: x[..., 3:4])(output)
    #slice_model = Model(inputs = inputs, outputs = prediction)
    #slice_model.compile(loss='mse', metrics = ['mse', 'mae', 'mape', rmse, ar])
    # slice_model.summary()

    model = Model(inputs=inputs, outputs=output)
    # model.compile(loss=generator_loss)
    model.compile(loss=None, metrics='mse')
    #model.compile(loss=None, metrics = [mse , mae, mape, rmse])
    model.summary()

    # return model, slice_model
    return model


#     return model


# 判别器
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
    #model.add(Dense(1 ,activation='softmax'))
    model.add(Dense(g_output_dim, activation='softmax'))
    model.compile(loss='discriminator_loss')
    # model.compile(loss=discriminator_loss)
    # model.summary()
    # history = model.fit(data_gen_train, validation_data=data_gen_test, epochs = 100,
    #                   callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5))
    return model


# 将yc转化为float64
yc_train = yc_train.astype(float)
yc_test = yc_test.astype(float)


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
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(
            from_logits=True)
        # 生成器即模型的训练优化器 adam
        self.generator_optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        # 判别器的优化器
        self.discriminator_optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        self.batch_size = self.opt['bs']
        self.checkpoint_dir = '../training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

    # 判别器损失函数
    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    # 生成器损失函数

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, real_x, real_y, yc):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # 此处的real_x也就是X_train
            # (31068,3,2)->generator->(31068,3)
            generated_data = self.generator(real_x, training=True)
            generated_data_reshape = tf.reshape(
                generated_data, [generated_data.shape[0], generated_data.shape[1], 1])
            d_fake_input = tf.concat(
                [tf.cast(generated_data_reshape, tf.float64), yc], axis=1)
            real_y_reshape = tf.reshape(
                real_y, [real_y.shape[0], real_y.shape[1], 1])
            #d_real_input = tf.concat([real_y_reshape, yc], axis=1)
            d_real_input = tf.concat(
                [tf.cast(real_y_reshape, tf.float64), yc], axis=1)

            # Reshape for MLP
            # d_fake_input = tf.reshape(d_fake_input, [d_fake_input.shape[0], d_fake_input.shape[1]])
            # d_real_input = tf.reshape(d_real_input, [d_real_input.shape[0], d_real_input.shape[1]])

            real_output = self.discriminator(d_real_input, training=True)
            fake_output = self.discriminator(d_fake_input, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return real_y, generated_data, {'d_loss': disc_loss, 'g_loss': gen_loss}

    def train(self, real_x, real_y, yc, opt):
        train_hist = {}
        train_hist['D_losses'] = []
        train_hist['G_losses'] = []
        train_hist['per_epoch_times'] = []
        train_hist['total_ptime'] = []

        epochs = opt["epoch"]
        for epoch in range(epochs):
            start = time.time()

            real_price, fake_price, loss = self.train_step(real_x, real_y, yc)

            G_losses = []
            D_losses = []

            Real_price = []
            Predicted_price = []

            D_losses.append(loss['d_loss'].numpy())
            G_losses.append(loss['g_loss'].numpy())

            Predicted_price.append(fake_price.numpy())
            Real_price.append(real_price.numpy())

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                tf.keras.models.save_model(
                    generator, './saveModel/gen_model_3_1_%d.h5' % epoch)
                self.checkpoint.save(
                    file_prefix=self.checkpoint_prefix + f'-{epoch}')
                print('epoch', epoch + 1, 'd_loss',
                      loss['d_loss'].numpy(), 'g_loss', loss['g_loss'].numpy())
            # print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
            # For printing loss
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - start
            train_hist['D_losses'].append(D_losses)
            train_hist['G_losses'].append(G_losses)
            train_hist['per_epoch_times'].append(per_epoch_ptime)

        # Reshape the predicted result & real
        Predicted_price = np.array(Predicted_price)
        Predicted_price = Predicted_price.reshape(
            Predicted_price.shape[1], Predicted_price.shape[2])
        Real_price = np.array(Real_price)
        Real_price = Real_price.reshape(
            Real_price.shape[1], Real_price.shape[2])

        plt.plot(train_hist['D_losses'], label='D_loss')
        plt.plot(train_hist['G_losses'], label='G_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('./saveImage/train_G_D_loss.png', dpi=300)
        # plt.show()

        return Predicted_price, Real_price, np.sqrt(mean_squared_error(Real_price, Predicted_price)) / np.mean(
            Real_price)


# 主函数部分
if __name__ == '__main__':
    input_dim = X_train.shape[1]
    feature_size = X_train.shape[2]
    output_dim = y_train.shape[1]

    # For Bayesian
    opt = {"lr": 0.0001, "epoch": 15, 'bs': 128}

    #generator = make_generator_model(X_train.shape[1], output_dim, X_train.shape[2])
    generator = make_generator_model()
    discriminator = make_discriminator_model()
    gan = GAN(generator, discriminator, opt)
    Predicted_price, Real_price, RMSPE = gan.train(
        X_train, y_train, yc_train, opt)

# %% --------------------------------------- Plot the result  -----------------------------------------------------------------

# Rescale back the real dataset
X_scaler = load(open('X_scaler.pkl', 'rb'))
y_scaler = load(open('y_scaler.pkl', 'rb'))
train_predict_index = np.load("train_predict_index.npy", allow_pickle=True)
test_predict_index = np.load("test_predict_index.npy", allow_pickle=True)
#dataset_train = pd.read_csv('dataset_train.csv', index_col=0)


print("----- predicted price -----", Predicted_price)

rescaled_Real_price = y_scaler.inverse_transform(Real_price)
rescaled_Predicted_price = y_scaler.inverse_transform(Predicted_price)

print("----- rescaled predicted price -----", rescaled_Predicted_price)
print("----- SHAPE rescaled predicted price -----",
      rescaled_Predicted_price.shape)

predict_result = pd.DataFrame()
for i in range(rescaled_Predicted_price.shape[0]):
    y_predict = pd.DataFrame(rescaled_Predicted_price[i], columns=[
                             "predicted_price"], index=train_predict_index[i:i+output_dim])
    predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)
#
real_price = pd.DataFrame()
for i in range(rescaled_Real_price.shape[0]):
    y_train = pd.DataFrame(rescaled_Real_price[i], columns=[
                           "real_price"], index=train_predict_index[i:i+output_dim])
    real_price = pd.concat([real_price, y_train], axis=1, sort=False)

predict_result['predicted_mean'] = predict_result.mean(axis=1)
real_price['real_mean'] = real_price.mean(axis=1)

# Plot the predicted result
plt.figure(figsize=(16, 8))
plt.plot(real_price["real_mean"])
plt.plot(predict_result["predicted_mean"], color='r')
plt.xlabel("Date")
plt.ylabel("Stock price")
plt.legend(("Real price", "Predicted price"), loc="upper left", fontsize=16)
plt.title("The result of Training", fontsize=20)
plt.show()

# Calculate RMSE
predicted = predict_result["predicted_mean"]
real = real_price["real_mean"]
For_MSE = pd.concat([predicted, real], axis=1)
RMSE = np.sqrt(mean_squared_error(predicted, real))
print('-- Train RMSE -- ', RMSE)
