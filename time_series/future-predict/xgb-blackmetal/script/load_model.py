#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/5/17 11:11
# @Author  : hyy
# @Site    : /Users/hyy/Desktop/gold/tradingsimulation/
# @File    : load_model.py

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


LOAD_MODEL_LONG = sys.argv[1]
LOAD_MODEL_SHORT = sys.argv[2]

N = 2000  # 预测数据的行数


def data_to_database(pd_data):  # 存储预测结果
    parts = pd_data
    conn = MySQLdb.connect(host='42.159.115.103', port=3306, user='root', passwd='caicloud', db='yuhuangshan', charset='utf8',)
    try:
        conn.ping()
    except Exception,e:
        print "read_mysql, Msql出了问题"
        print str(e)
        while True: 
            try: 
                conn = MySQLdb.connect(host='42.159.115.103', port=3306, user='root', passwd='caicloud', db='yuhuangshan', charset='utf8')
                print "Msql连接"
                break
            except Exception,e:
                print "尝试重连接失败" 
                time.sleep(1)
                continue
    values = ','.join(map(str, parts.iloc[0,2:].values))
    sql = "insert into yuhuangshan.blackmetal_result(date_dt, time, label_long, label_short) values (\"%s\", \"%s\", %s)" % \
            (parts.iloc[0,0], parts.iloc[0,1], values)
        # print parts.iloc[0,0], parts.iloc[0,1]
    select_stmt = "SELECT * FROM yuhuangshan.blackmetal_result WHERE date_dt = \"%s\" AND time = \"%s\"" % (parts.iloc[0,0], parts.iloc[0,1])
    try: 
        cur = conn.cursor()   
        data = cur.execute(select_stmt)
        # print data
        if data == 0:
            cur.execute(sql)
            conn.commit()
            print "预测结果插入blackmetal_result成功"
        else:
            cur.execute(
                "update yuhuangshan.blackmetal_result set label_long = \"%s\", label_short = \"%s\" where date_dt = \"%s\" AND time = \"%s\" " % (parts.iloc[0,2], parts.iloc[0,3],parts.iloc[0,0], parts.iloc[0,1])
            )
            conn.commit()
            print "预测结果更新blackmetal_result成功"
    except MySQLdb.Error, e: 
        print "预测结果存入blackmetal_result失败"
        print "MySQL Error %d: %s" % (e.args[0], e.args[1])
    cur.close()
    conn.close()



def data_ready():
    data=[]
    sql = 'select * from yuhuangshan.blackmetal_15min order by date_dt desc, time desc limit %d;' %N    # 查询语句
    # 数据库连接
    conn = MySQLdb.connect(host='42.159.115.103', port=3306, user='root', passwd='caicloud', db='yuhuangshan', charset='utf8',)
    try:
        conn.ping()
    except Exception,e:
        print "read_mysql, Msql出了问题"
        print str(e)
        while True: 
            try: 
                conn = MySQLdb.connect(host='42.159.115.103', port=3306, user='root', passwd='caicloud', db='yuhuangshan', charset='utf8')
                print "Msql连接"
                break
            except Exception,e:
                print "尝试重连接失败" 
                time.sleep(2)
                continue
    try:
        cursor = conn.cursor()
        cursor.execute(sql)  #
        cds = cursor.fetchall()
        data = pd.DataFrame(list(cds))
        cursor.close()
        conn.close()
        print "blackmetal_15min预测数据读取成功"
        print "待预测数据\n",data.iloc[-1:,:-1]
        return data.iloc[:,:-1]
    except MySQLdb.Error, e: 
        print "blackmetal_15min预测数据读取失败"
        print "MySQL Error %d: %s" % (e.args[0], e.args[1])
        cursor.close()
        conn.close()
    

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




def main():
    while(1):
        data = data_ready()
        data = pd.DataFrame(data)
        data.columns = ["id", "date", "time", "open_price", "close_price", "high_price", "low_price", "trade_volume", 
                     "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10",
                     "p11", "p12", "p13", "p14", "p15", "p16", "p17", "p18", "p19", "p20", 
                     "p21", "p22", "p23", "p24", "p25", "p26", "p27", "p28", "p29", "p30",
                     "p31", "p32", "p33", "p34", "p35", "p36", "p37"]
        data = data.sort_values(["date", "time"], ascending=[1, 1])
        data = data.iloc[:,1:].drop_duplicates()
        # print data.tail(5)
        
        # 加载模型
        clf_short = xgb.XGBClassifier(learning_rate =0.1, base_score=0.5, colsample_bylevel=1,\
        colsample_bytree=0.8, max_delta_step=1, max_depth=4, min_child_weight=5, missing=None,\
        n_estimators=100, nthread=-1, objective='binary:logistic', reg_alpha=10e-3,\
        reg_lambda=10e-5, scale_pos_weight=1, seed=0, silent=False, subsample=0.8)

        clf_long = xgb.XGBClassifier(learning_rate =0.1, base_score=0.5, colsample_bylevel=1,\
        colsample_bytree=0.8, max_delta_step=1, max_depth=4, min_child_weight=5, missing=None,\
        n_estimators=100, nthread=-1, objective='binary:logistic', reg_alpha=10e-3,\
        reg_lambda=10e-5, scale_pos_weight=1, seed=0, silent=False, subsample=0.8)

        short_model = xgb.Booster()
        long_model  = xgb.Booster()
        short_model.load_model(LOAD_MODEL_SHORT)
        long_model.load_model(LOAD_MODEL_LONG)
        clf_short._Booster = short_model
        clf_long._Booster = long_model
        clf_short._le = preprocessing.LabelEncoder().fit([0,1])
        clf_long._le = preprocessing.LabelEncoder().fit([0,1])

        # 特征工程
        data = cross_feature(data)
        # data = data_trend(data)
        trends, cross = volume_trend(data)
        D = data
        D["trends"] = np.array(trends)
        D["cross"] = np.array(cross)
        D = data_trend(D)
        P_D = D.fillna(0)       
        pre_feature = P_D.iloc[-10:, 2:]
        pre_short = clf_short.predict(pre_feature)
        pre_long = clf_long.predict(pre_feature)

        result_data = data.iloc[-10:, 0:2]
        result_data["label_long"] = pre_long
        result_data["label_short"] = pre_short

        print result_data
        # date_to_database(result_data)
       #  print 'sucess to database-black'
    
    

if __name__ == '__main__':
    main()


