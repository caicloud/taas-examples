#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
import talib
import xgboost as xgb
from sklearn import metrics
import math
import time


# 交叉特征
def cross_feature(feature_data, N=10):
    open_price, close_price, high_price, low_price, trade_volume = feature_data['open_price'].values, feature_data['close_price'].values, feature_data['high_price'].values, feature_data['low_price'].values, feature_data['trade_volume'].values
    feature_data['o_c'] = open_price - close_price  # 阴线实体
    feature_data['h_c'] = high_price - close_price  # 阳线上影线长度
    feature_data['h_o'] = high_price - open_price  # 阴线上影
    feature_data['h_l'] = high_price - low_price
    feature_data['l_c'] = close_price - low_price  # 阴线下影
    feature_data['l_o'] = open_price - low_price  # 阳线下影
    feature_data['frac_change'] = (feature_data['close_price'] - feature_data['open_price']) / feature_data['open_price']
    feature_data['frac_high']   = (feature_data['high_price'] - feature_data['open_price'])  / feature_data['open_price']
    feature_data['frac_low']    = (feature_data['open_price'] - feature_data['low_price'])   / feature_data['open_price']

    ret = lambda x,y: np.log(y/x) #Log return 
    zscore = lambda x:(x -x.mean())/x.std() # zscore

    feature_data['c_2_o'] = zscore(ret(feature_data['open_price'], feature_data['close_price']))
    feature_data['h_2_o'] = zscore(ret(feature_data['open_price'], feature_data['high_price']))
    feature_data['l_2_o'] = zscore(ret(feature_data['open_price'], feature_data['low_price']))
    feature_data['c_2_h'] = zscore(ret(feature_data['high_price'], feature_data['close_price']))
    feature_data['h_2_l'] = zscore(ret(feature_data['high_price'], feature_data['low_price']))
    feature_data['vol']   = zscore(feature_data['trade_volume'])

    feature_data["hl_perc"] = (feature_data["high_price"] - feature_data['low_price']) / feature_data['low_price'] * 100
    feature_data["co_perc"] = (feature_data["close_price"] - feature_data["open_price"]) / feature_data["open_price"] * 100
    #feature_data["price_next_30"] = feature_data["adj_close"].shift(-30)
    
    feature_data['last_close']  = feature_data['close_price'].shift(1 * N)
    #feature_data['pred_profit'] = ((feature_data['close'] - feature_data['last_close']) / feature_data['last_close'] * 100).shift(-1 * N)

    # open_high = np.zeros(len(feature_data))
    # open_low = np.zeros(len(feature_data))
    # close_high = np.zeros(len(feature_data))
    # close_low = np.zeros(len(feature_data))
    # for i in range(0, len(feature_data)):
    #     start = max(0, i-N)
    #     end = max(1, i)
    #     highest_price = high_price[start:end].max()
    #     lowest_price = low_price[start:end].min()
    #     first_open_price = open_price[start]
    #     last_close_price = close_price[end]
    #     open_high[i] = first_open_price - highest_price
    #     open_low[i] = first_open_price - lowest_price
    #     close_high[i] = last_close_price - highest_price
    #     close_low[i] = last_close_price - lowest_price
    # feature_data['open_high'] = open_high
    # feature_data['open_low'] = open_low
    # feature_data['close_high'] = close_high
    # feature_data['close_low'] = close_low
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
    return data



def calculate_macd(df):
    new_data = df.sort_index()
    start_index = 0
    new_data.ix[start_index, 'ema_12'] = new_data.ix[start_index, 'close_price']
    new_data.ix[start_index, 'ema_26'] = new_data.ix[start_index, 'close_price']
    new_data.ix[start_index, 'ema_9'] = 0
    for i in range(1, len(new_data.index)):
    #print(i)
        new_data.ix[i, 'ema_12'] = new_data.ix[i-1, 'ema_12'] * 11 / 13 + new_data.ix[i, "close_price"] * 2 / 13
        new_data.ix[i, 'ema_26'] = new_data.ix[i-1, 'ema_26'] * 25 / 27 + new_data.ix[i, 'close_price'] * 2 / 27
        new_data.ix[i, 'diff'] = new_data.ix[i, 'ema_12'] - new_data.ix[i, 'ema_26']
        new_data.ix[i, 'ema_9'] = new_data.ix[i-1, 'ema_9'] * 8 / 10 + new_data.ix[i, "diff"] * 2 / 10
        new_data.ix[i, 'bar'] = 2 * (new_data.ix[i, 'diff'] - new_data.ix[i, 'ema_9'])
    return new_data


def calculate_change_aver(df):
    new_data = df.sort_index()
    new_data.ix[0, 'ma5'] = new_data.ix[0, "close_price"]
    new_data.ix[0, 'ma10'] = new_data.ix[0, "close_price"]
    new_data.ix[0, 'ma20'] = new_data.ix[0, "close_price"]
    new_data.ix[0, 'v_ma5'] = new_data.ix[0, "trade_volume"]
    new_data.ix[0, 'v_ma10'] = new_data.ix[0, "trade_volume"]
    new_data.ix[0, 'v_ma20'] = new_data.ix[0, "trade_volume"]
    for i in range(1, len(new_data.index)):
        new_data.ix[i, "price_change"] = new_data.ix[i, "close_price"] - new_data.ix[i-1, "close_price"]
        new_data.ix[i, "p_change"] = new_data.ix[i, "price_change"] / new_data.ix[i-1, "close_price"]
        previous = 0
        v_previous = 0
        for j in range(min(i+1, 5)):
            previous += new_data.ix[i - j, "close_price"]
            v_previous += new_data.ix[i - j, "trade_volume"]
        new_data.ix[i, "ma5"] = previous / min(i+1, 5)
        new_data.ix[i, "v_ma5"] = v_previous / min(i+1, 5)
        for j in range(min(i+1, 5), min(i+1, 10)):
            previous += new_data.ix[i - j, "close_price"]
            v_previous += new_data.ix[i - j, "trade_volume"]
        new_data.ix[i, "ma10"] = previous / min(i+1, 10)
        new_data.ix[i, "v_ma10"] = v_previous / min(i+1, 10)
        for j in range(min(i+1, 10), min(i+1, 20)):
            previous += new_data.ix[i - j, "close_price"]
            v_previous += new_data.ix[i - j, "trade_volume"]
        new_data.ix[i, "ma20"] = previous / min(i+1, 20)
        new_data.ix[i, "v_ma20"] = v_previous / min(i+1, 20)
    return new_data



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



