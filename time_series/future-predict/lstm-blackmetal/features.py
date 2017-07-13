#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/13 14:11
# @Author  : hyy
# @Site    : /Users/hyy/Desktop/yuhuangshan/
# @File    : feature.py

import sys
import MySQLdb
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import metrics
import time
from sklearn import preprocessing
from string import Template
import math
import talib 


# 交叉特征
def cross_feature(feature_data, N=20):
    open_price, close_price, high_price, low_price, trade_volume = feature_data['open_price'].values, feature_data['close_price'].values, feature_data['high_price'].values, feature_data['low_price'].values, feature_data['trade_volume'].values
    feature_data['o_c'] = open_price - close_price  # 阴线实体
    feature_data['h_c'] = high_price - close_price  # 阳线上影线长度
    feature_data['h_o'] = high_price - open_price  # 阴线上影
    feature_data['h_l'] = high_price - low_price
    feature_data['l_c'] = close_price - low_price  # 阴线下影
    feature_data['l_o'] = open_price - low_price  # 阳线下影
    open_high = np.zeros(len(feature_data))
    open_low = np.zeros(len(feature_data))
    close_high = np.zeros(len(feature_data))
    close_low = np.zeros(len(feature_data))
    for i in range(0, len(feature_data)):
        start = max(0, i-N)
        end = min(len(feature_data)-1, i+1)
        highest_price = high_price[start:end].max()
        lowest_price = low_price[start:end].min()
        first_open_price = open_price[start]
        last_close_price = close_price[end]
        open_high[i] = first_open_price - highest_price
        open_low[i] = first_open_price - lowest_price
        close_high[i] = last_close_price - highest_price
        close_low[i] = last_close_price - lowest_price
    feature_data['open_high'] = open_high
    feature_data['open_low'] = open_low
    feature_data['close_high'] = close_high
    feature_data['close_low'] = close_low
    return feature_data

# 趋势特征
def data_trend(data):
    open_prices, close_prices, high_prices, low_prices, volumes = np.asarray(data['open_price']), np.asarray(data['close_price']), np.asarray(data['high_price']), np.asarray(data['low_price']), np.asarray(data['trade_volume'])
    ma5 = talib.MA(close_prices, timeperiod=5)
    ma10 = talib.MA(close_prices, timeperiod=10)
    ma20 = talib.MA(close_prices, timeperiod=20)
    ma30 = talib.MA(close_prices, timeperiod=30)
    ma60 = talib.MA(close_prices, timeperiod=60)
    ma90 = talib.MA(close_prices, timeperiod=90)
    ma120 = talib.MA(close_prices, timeperiod=120)
    ma180 = talib.MA(close_prices, timeperiod=180)
    ma360 = talib.MA(close_prices, timeperiod=360)
    ma720 = talib.MA(close_prices, timeperiod=720)
    data['m1'] = (ma5 - close_prices) / close_prices
    data['m2'] = (ma10 - close_prices) / close_prices
    data['m3'] = (ma20 - close_prices) / close_prices
    data['m4'] = (ma30 - close_prices) / close_prices
    data['m5'] = (ma60 - close_prices) / close_prices
    data['m6'] = (ma90 - close_prices) / close_prices
    data['m7'] = (ma120 - close_prices) / close_prices
    data['m8'] = (ma180 - close_prices) / close_prices
    data['m9'] = (ma360 - close_prices) / close_prices
    data['m10'] = (ma720 - close_prices) / close_prices
    data['m11'] = (ma5 - volumes) / (volumes + 1)
    data['m12'] = (ma10 - volumes) / (volumes + 1)
    data['m13'] = (ma20 - volumes) / (volumes + 1)
    data['m14'] = (ma30 - volumes) / (volumes + 1)
    data['m15'] = (ma60 - volumes) / (volumes + 1)
    data['m16'] = (ma90 - volumes) / (volumes + 1)
    data['m17'] = (ma120 - volumes) / (volumes + 1)
    data['m18'] = (ma180 - volumes) / (volumes + 1)
    data['m19'] = (ma360 - volumes) / (volumes + 1)
    data['m20'] = (ma720 - volumes) / (volumes + 1)
    # rocp = talib.ROCP(close_prices, timeperiod=1)
    # norm_volumes = (volumes - np.mean(volumes)) / math.sqrt(np.var(volumes))
    # vrocp = talib.ROCP(norm_volumes + np.max(norm_volumes) - np.min(norm_volumes), timeperiod=1)
    # data['pv'] = rocp * vrocp * 100.
    return data


# 交易量特征
def volume_trend(data):
    trade_volume = np.sqrt(data["trade_volume"].values)
    trends = []
    cross = []
    for index in range(0, len(data)):
        start = max(0, index - 20)
        end = max(1, index+1)
        volume = trade_volume[index]
        median_volume = np.median(trade_volume[start:end])
        if volume > median_volume:
            volume_trend = volume * 1.0 / median_volume
        else:
            volume_trend = median_volume * -1.0 / volume
        trends.append(volume_trend)
        cross.append(volume_trend * data["o_c"].values[index])
    return trends, cross
