#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/13 15:11
# @Author  : hyy
# @Site    : /Users/hyy/Desktop/yuhuangshan/
# @File    : lstm.py

import os
import pandas as pd 
import numpy as np 
import time
import math
import matplotlib.pyplot as plt

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping, History
from keras.layers.advanced_activations import PReLU, LeakyReLU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error,r2_score

from features import cross_feature, data_trend, volume_trend
from model_evaluation import model_evaluation_multi_step
from model_visualization import model_visulaization_multi_step, plot_loss


# 输入数据
INPUT_DATA = "shuju1.csv"
# 需要加入训练的特征长度
# FEATURE_LEN = 5
# 时间序列长度
SEQ_LEN = 20
# 预测步长
STEP_LEN = 20
# epochs大小
EPOCHS = 500
# 批大小
BATCH_SIZE = 128
# 测试训练集比例
TRAIN_SAMPLES_RATE = 0.8
# 网络形状
LAYERS = [100, SEQ_LEN, 100, STEP_LEN, 41]


def min_max_normal(train,test):
    # 对数据进行归一化
    scaler = MinMaxScaler()
    scalerModel = scaler.fit(train)
    _train = scalerModel.transform(train)
    _test  = scalerModel.transform(test)
    return _train, _test, scalerModel

def reshape_data(data, SEQ_LEN):
    reshape_data = []
    for i in range(len(data) - SEQ_LEN):
        reshape_data.append(data[i: i + SEQ_LEN])
    return np.array(reshape_data)


# In[546]:

def build_model():
    """
    定义模型
    """
    model = Sequential()

    model.add(LSTM(units=LAYERS[1], input_shape=(LAYERS[1], LAYERS[4]), return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(LAYERS[2], return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=LAYERS[3]))
    model.add(BatchNormalization(weights=None, epsilon=1e-06, momentum=0.9))
    #model.add(Activation("tanh"))
    #act = PReLU(alpha_initializer='zeros', weights=None)
    act = LeakyReLU(alpha=0.3)
    model.add(act)

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model


def load_data():
    raw_data = pd.read_csv(INPUT_DATA)
    #print(len(raw_data))
    raw_data.columns = ["date", "time", "open", "close", "high", "low", "volume",
                     "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10",
                     "p11", "p12", "p13", "p14", "p15", "p16", "p17", "p18", "p19", "p20",
                     "p21", "p22", "p23", "p24", "p25", "p26", "p27", "p28", "p29", "p30",
                     "p31", "p32", "p33", "p34", "p35", "p36", "p37"]

    #raw_data = raw_data[['price_date'] + Conf.FIELDS].dropna()
    #print(len(raw_data))
    train_samples_num = int((len(raw_data) - STEP_LEN) * TRAIN_SAMPLES_RATE)
    
    # Divide features and labels
    labels = raw_data[["date", "time","high"]]
    labels_tag = labels["high"]
    data =  raw_data.drop(['high'],axis=1) # 删除需要预测的数据行
    features = data.drop(["date", "time"],axis=1)
    
    _x_train = np.array(features[:train_samples_num])
    _x_test = np.array(features[train_samples_num:-STEP_LEN])
    _y_train = np.array(labels_tag).T[STEP_LEN:(train_samples_num + STEP_LEN)]
    _y_test = np.array(labels_tag).T[train_samples_num + STEP_LEN:]
    #_y_train = _y_train[:, np.newaxis] # 行向量转化为列向量
    #_y_test = _y_test[:, np.newaxis]
    print _x_train.shape, _x_test.shape, _y_train.shape, _y_test.shape,
    print train_samples_num
    _x_train, _x_test, x_scaler = min_max_normal(_x_train, _x_test)
    _y_train, _y_test, y_scaler = min_max_normal(_y_train, _y_test)
    
    x_train = reshape_data(_x_train, SEQ_LEN)
    x_test = reshape_data(_x_test, SEQ_LEN)
    y_train = reshape_data(_y_train, STEP_LEN)
    y_test = reshape_data(_y_test, STEP_LEN)

    print x_train.shape, x_test.shape, y_train.shape, y_test.shape

    return [x_train, x_test, y_train, y_test, labels, labels_tag, data, features, x_scaler, y_scaler]


def predict_by_day(model, data):
    """
    按天预测
    """
    predict = model.predict(data)
    print(predict.shape)
    # predict = np.reshape(predict, (len(predict),))
    # print(predict.shape)
    return predict


def predict_by_days(model, data):
    """
    预测未来所有价格，这种方法仅对特征只有一个价格时有效，因为SQL_LEN+1天的非价格特征无法提前知道）
    """
    # 用于保存预测结果
    predict_seq = []
    current_predict = None
    for i in range(len(data)):
        # 当前用于预测的样本
        current_x = data[i]
        if i > 0:
            current_x[-1, -1] = current_predict
        current_predict = model.predict(current_x[np.newaxis, :, :])[0, 0]
        predict_seq.append(current_predict)
    return predict_seq


def inverse_normalise_y(scaler, scalerd_y):
    return scaler.inverse_transform(scalerd_y)



def main():
    global_start_time = time.time()
    print('> Loading data... ')
    x_train, x_test, y_train, y_test, labels, labels_tag, data, features, x_scaler, y_scaler = load_data()
    model = build_model()
    print(model.summary())
    # keras.callbacks.History记录每个epochs的loss及val_loss
    hist = History()
    earlystopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0)
    model.fit(np.array(x_train), np.array(y_train), batch_size=500, epochs=100, shuffle=True,\
    validation_split=0.05, callbacks=[hist, earlystopping])

    # 控制台打印历史loss及val_loss
    print(hist.history['loss'])
    print(hist.history['val_loss'])

    # 可视化历史loss及val_loss
    plot_loss(hist.history['loss'], hist.history['val_loss'])
    # predicted = predict_by_days(model, X_test, 20)
    predicted = predict_by_day(model, x_test)
    predicted = inverse_normalise_y(y_scaler, predicted)
    y_test = inverse_normalise_y(y_scaler, y_test)
    print y_test.shape, predicted.shape
    print('Training duration (s) : ', time.time() - global_start_time)
    print predicted[0]
    print y_test[0]
    # 模型评估
    print u"模型评估\n"
    model_evaluation_multi_step(pd.DataFrame(predicted),pd.DataFrame(y_test))
    # 预测结果可视化
    print u"预测结果可视化\n"
    model_visulaization_multi_step(y_test, predicted)


if __name__ == '__main__':
    main()









