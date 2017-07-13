#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/13 11:11
# @Author  : hyy
# @Site    : /Users/hyy/Desktop/yuhuangshan/
# @File    : garch.py

import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

try:
   import seaborn
except ImportError:
    pass

import datetime as dt
from arch import arch_model
# 输入数据
INPUT_DATA = "blackmetal/data/shuju1.csv"
raw_data = pd.read_csv(INPUT_DATA)
raw_data.columns = ["date", "time", "open", "close", "high", "low", "volume", 
                     "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10",
                     "p11", "p12", "p13", "p14", "p15", "p16", "p17", "p18", "p19", "p20", 
                     "p21", "p22", "p23", "p24", "p25", "p26", "p27", "p28", "p29", "p30",
                     "p31", "p32", "p33", "p34", "p35", "p36", "p37"]

labels = raw_data[["date", "time","high"]]
labels_tag = labels["high"]
# 画变化率的图
returns = 100 * labels_tag.pct_change().dropna()
get_ipython().magic(u'matplotlib inline')
figure = returns.plot(figsize=(20,6))

## GARCH(1,1)
am = arch_model(returns)  # 使用默认选项，可以生成一个均值恒定，误差符合正态分布，同时符合GARCH(1,1)条件方差的模型
res = am.fit(update_freq=5)  # 通过拟合获得模型的参数
print(res.summary())

fig = res.plot(annualize='D')  # plot() 函数可以快速展示 标D的标准偏差和条件波动率。


## GJR-GARCH 模型
am = arch_model(returns, p=1, o=1, q=1)  # 设置o为1， 即包含了非对称冲击的一阶滞后项，从而将原GARCH模型转换为一个GJR-GARCH模型。新的模型具有动态方差
res = am.fit(update_freq=5, disp='off')
print(res.summary())

## TARCH/ZARCH
# TARCH模型 (又称为 ZARCH模型) 是对_波动率_的绝对值进行建模. 使用该模型时，需要在arch_model建构函数中，设置power=1.0。因为默认的阶数为2，对应的是用平方项表示的方差变化过程
am = arch_model(returns, p=1, o=1, q=1, power=1.0)
res = am.fit(update_freq=5)
print(res.summary())


## 学生T分布误差
# 金融资产回报率的分布往往体现出肥尾现象，学生T分布是一种简单的方法，可以用来捕捉这种特性。在调用arch_model 构建函数时，可以将概率分布从正态分布转换为学生T分布。
am = arch_model(returns, p=1, o=1, q=1, power=1.0, dist='StudentsT')  # 标准化的新息展示出，分布函数具有一个将近10个估计自由度的肥尾。
res = am.fit(update_freq=5)
print(res.summary())

















