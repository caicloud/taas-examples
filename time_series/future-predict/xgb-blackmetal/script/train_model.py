#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/14 11:11
# @Author  : hyy
# @Site    : /Users/hyy/Desktop/yuhuangshan/script/
# @File    : save_model.py

import sys
import pandas as pd
import numpy as np
import talib
import xgboost as xgb
from sklearn import metrics
import math
import time

import feature

INTPUT_DATA = sys.argv[1]
SAVE_MODEL_LONG = sys.argv[2]
SAVE_MODEL_SHORT = sys.argv[3]


# 多头标签
def get_long_labels(pd_data, bar_long=12, p_long=1.8):
    labels = []
    row = pd_data.shape[0]
    start_price = pd_data['open_price'].values
    for i in range(row-bar_long):
        interval = pd_data['high_price'].values[i:(i+bar_long)]
        highest_price = interval.max()
        lowest_price = interval.min()
        label = 0
        if highest_price - start_price[i] > p_long and start_price[i] < lowest_price: #< 1.5 * p_long:
            label = 1
        labels.append(label)
    labels = pd.DataFrame(labels)
    return labels


# 空头标签
def get_short_labels(pd_data, bar_short=10, p_short=1.6):
    labels = []
    row = pd_data.shape[0]
    start_price = pd_data['open_price'].values
    for i in range(row-bar_short):
        interval = pd_data['low_price'].values[i:(i+bar_short)]
        highest_price = interval.max()
        lowest_price = interval.min()
        label = 0
        if start_price[i] - lowest_price > p_short and highest_price < start_price[i]: #< 1.5 * p_short:
            label = 1
        labels.append(label)
    labels = pd.DataFrame(labels)
    return labels
# 数据集划分
def train_test_split(feature, labels, ratio=0.8):
    s = int(len(labels) * ratio)
    train_x = feature.iloc[:s, :]
    test_x = feature.iloc[s:, :]
    train_y = labels.iloc[:s]
    test_y = labels.iloc[s:]
    return train_x, train_y, test_x, test_y


# 模型评价
def Y_count(Y):
    #统计每一类的数量
    label_value = np.unique(Y)
    label_count = []
    for i in label_value:
        label_count.append((Y[:]==i).sum())
    return label_value, label_count

def data_describe(X,Y):
    #统计有多少数据有多少行多少列，目标值有几类，每一类有多少个
    raw, col = X.shape
    Y_label_value, Y_label_count = Y_count(Y)
    print '-' * 50
    print 'data.shape: %s, %s' % (raw, col )
    print 'Y value is :'
    print Y_label_value
    print  'Y count is : '
    print  Y_label_count
    print '-' * 50
    
def model_evaluation(model, X, Y):
    from sklearn import metrics
    from sklearn.metrics import confusion_matrix
    # import matplotlib.pyplot as plt
    data_describe(X, Y)
    print '-'*50
    predict = model.predict(X)
    precision = metrics.precision_score(Y, predict)
    recall = metrics.recall_score(Y, predict)
    accuracy = metrics.accuracy_score(Y, predict)
    print '准确率: %.2f%%, 精确率: %.2f%%, 召回率: %.2f%%' % (100 * accuracy, 100 * precision, 100 * recall)
    f1_score = metrics.f1_score(Y, predict)
    print 'f1_score: %.2f%%' % (100 * f1_score)
    print '混淆矩阵：'
    print(metrics.confusion_matrix(Y, predict))
    print '分类报告：'
    print(metrics.classification_report(Y, predict))
    print '-'*50


bar_short = 10
bar_long = 10
p_short = 10
p_long = 10
SIZE = 0.8

def main():
    data = pd.read_csv(INTPUT_DATA, header = None)
    data.columns = ["date", "time", "open_price", "close_price", "high_price", "low_price", "trade_volume", 
                 "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10",
                 "p11", "p12", "p13", "p14", "p15", "p16", "p17", "p18", "p19", "p20", 
                 "p21", "p22", "p23", "p24", "p25", "p26", "p27", "p28", "p29", "p30",
                 "p31", "p32", "p33", "p34", "p35", "p36", "p37"]
    print "data readed"
    
    label_long = get_long_labels(data, bar_long, p_long)
    label_short = get_short_labels(data, bar_short, p_short)
    print 'labels are readed'
    
    data = data.iloc[:,:7]
    # 特征工程
    data = feature.cross_feature(data)
   # data = feature.data_trend(data)
    # trends, cross = feature.volume_trend(data)
    # D = data
    # D["trends"] = np.array(trends)
    # D["cross"] = np.array(cross)
    # D = feature.data_trend(D).fillna(0)
    #data = feature.calculate_macd(data)
    D = feature.calculate_change_aver(data)
    feature_long = D.iloc[:-bar_long,:]
    feature_short = D.iloc[:-bar_short,:]
    
   # label_long=label_long.shift(-1).fillna(0)
    #label_short=label_short.shift(-1).fillna(0)

    feature_long.index = range(len(feature_long))
    label_long.index = range(len(label_long))
    feature_short.index = range(len(feature_short))
    label_short.index = range(len(label_short))

    train_longx, train_longy, test_longx, test_longy = train_test_split(feature_long, label_long, ratio=SIZE)
    train_long_x, test_long_x = train_longx.iloc[:,2:], test_longx.iloc[:, 2:]
    train_shortx, train_shorty, test_shortx, test_shorty = train_test_split(feature_short, label_short, ratio=SIZE)
    train_short_x, test_short_x = train_shortx.iloc[:,2:], test_shortx.iloc[:, 2:]

    # model
    clf_short = xgb.XGBClassifier(learning_rate =0.1, base_score=0.5, colsample_bylevel=1, colsample_bytree=0.8, 
                            max_delta_step=1, max_depth=5, 
                             min_child_weight=5, missing=None, n_estimators=100, 
                             nthread=-1, objective='binary:logistic', reg_alpha=10e-3, reg_lambda=10e-5, 
                             scale_pos_weight=1, seed=0, silent=False, subsample=0.8)
    clf_long = xgb.XGBClassifier(learning_rate =0.1, base_score=0.5, colsample_bylevel=1, colsample_bytree=0.8, 
                            max_delta_step=1, max_depth=5, 
                             min_child_weight=5, missing=None, n_estimators=100, 
                             nthread=-1, objective='binary:logistic', reg_alpha=10e-3, reg_lambda=10e-5, 
                             scale_pos_weight=1, seed=0, silent=False, subsample=0.8)
    clf_short.fit(train_short_x, train_shorty)
    clf_long.fit(train_long_x, train_longy)


    model_evaluation(clf_long, test_long_x, test_longy)
   # model_evaluation(clf_long, train_long_x, train_longy)

    model_evaluation(clf_short, test_short_x, test_shorty)
   # model_evaluation(clf_short, train_short_x, train_shorty)
   # model_evaluation(clf_short, train_short_x, train_shorty)   
    clf_long.booster().save_model(SAVE_MODEL_LONG)
    clf_short.booster().save_model(SAVE_MODEL_SHORT)


if __name__ == '__main__':
    main()



