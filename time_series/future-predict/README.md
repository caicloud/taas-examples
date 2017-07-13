# 期货预测进化路线

# 背景

客户提供黄金期货每15min数据，要求我们预测黄金期货阶段价格的涨跌情况

- 开始阶段：预测第二天是否会涨
- 发展：预测n个bar之内，是否会涨p元钱，或者跌m元
- 进行：预测n个bar之内，是否会涨u%,或者跌d%

# 进化路线

ARIMA、GARCH → Hot-winters → xgboost → MLP → RNN → LSTM → LSTM Variants(还未深入研究)

#### 各个模型效果见：[Google-excel](https://docs.google.com/spreadsheets/d/1Vv8A0SaFR2KcdsujZD5m_nnK-HNMXIMDC-rdUqUt8yU/edit#gid=0)

# 2 各进化体

## 2.1 [ARIMA](http://www.cnblogs.com/bradleon/p/6827109.html)、[GARCH](https://zhuanlan.zhihu.com/p/21962996)

- 优点
  - 模型简单，只需要内生变量而不需要借助其它外生变量
- 不足
  - 要求时序数据是稳定的（stationary），或者通过差分化（differencing）后是稳定的，否则无法捕捉到规律（如股票数据）
  - 仅基于历史数据进行预测
  - 本质上只能捕捉线性关系而不能捕捉非线性关系
- 效果：
  - arima 预测结果前面3个之内预测教可观，约往后，数据差距越大，且p,d,q不好控制
  - garch 可以用来考虑择时，但是阿法，贝塔，伽马参数不好拟合，效果略差，代码见：
- 参考文献
  - [GARCH 模型对上海股市的一个实证研究](http://bs.ustc.edu.cn/UserFiles/File/2010101841496725.pdf)

### 2.2 Hot-winters

- 优点
  - 能很好地捕捉变化波动规律
- 不足
  - 难以进行长期预测
  - 通过指数平滑建立函数关系，只使用内生变量，与外生变量无关
- 效果
  - 序列实时困难，预测长度一长，数据趋于线性，适用性不高，有待探索
  - 预测结果较为发散，范围太大，不好精确化
- 代码
  - R语言，可参考[Holt-Winters seasonal method](https://www.otexts.org/fpp/7/5)
  - python可以考虑[import](https://gist.github.com/andrequeiroz/5888967)，也可参考[Holt-Winters Forecasting for Dummies](https://grisha.org/blog/2016/01/29/triple-exponential-smoothing-forecasting/)

### 2.3 xgboost

- 优点
  - 准确率高于其他机器学习模型
  - 训练速度快
- 缺点
  - 模型参数不好控制
  - 模型结果与feature所在的列位置相关性较高，优化起来较麻烦
- 效果
  - 在特征不足下，效果为58%左右，增加特征工程，效果提升至65%，但是和客户软件吻合度比较低，综合效果只有50%
- 代码
  - Python，见：文件目录下

### 2.4 MLP

- 优点
  - 非线性映射能力强(通过sigmoid、tanh等激活函数)
- 不足
  - 难以体现数据的时序性
  - 无法自适应调整网络结构

## 2.5 RNN

- 优点
  - 能够将上一个输出的结果作为下一次的输入，刻画复杂的历史依赖
- 不足
  - 长期依赖问题
  - 梯度弥散（Gradient Vanishing）问题（效果不一致）
    - 有效初始化+ReLU激活函数能够得到较好效果
    - 算法上的优化，例如截断的BPTT算法
    - 模型上的改进，例如LSTM、GRU单元都可以有效解决长期依赖问题
    - 在BPTT算法中加入skip connection，此时误差可以间歇的向前传播
    - 加入一些Leaky Units，思路类似于skip connection
  - 梯度爆炸（Gradient Explosion）问题（效果较好）
    - 采用截断的方式有效避免，比如gradient clipping（如果梯度的范数大于某个给定值，将梯度同比收缩）

## 2.4 LSTM

- 提出
  - [Hochreiter & Schmidhuber (1997)](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
- 简介
  - 解决梯度弥散问题
- 优化
  - 激活函数优化，sigmod,tanh,ReLU,PReLU,LeakyReLU
  - optimizer优化
  - loss优化


- 遇到的问题
  - 如何做多步预测：未来步非价格特征不存在的问题
    - 解决方法：y采用输出序列的方式
  - 如果分类，需要做到unbalanced data的matrix评价函数编写与调优
- 尚待研究
  - 如何做多步预测，并提高准确率
- 代码
  - Python，见：文件目录下

## 2.5 LSTM Variants

### 2.5.1 peephole connections

- 提出
  - [Gers & Schmidhuber (2000)](ftp://ftp.idsia.ch/pub/juergen/TimeCount-IJCNN2000.pdf)
- 简介
  - 加入了peephole connections，这意味着门限层也将单元状态作为输入

### 2.5.2 Gated Recurrent Unit(GRU)

- 提出
  - [Cho, et al. (2014)](http://arxiv.org/pdf/1406.1078v3.pdf)
- 简介
  - 将遗忘和输入门限结合输入到单个“更新门限”中。同样还将单元状态和隐藏状态合并，并做出一些其他变化。所得模型比标准LSTM模型要简单，这种做法越来越流行。

### 2.5.3 others

#### 2.5.3.1 models

- Depth Gated RNNs：[Yao, et al. (2015)](http://arxiv.org/pdf/1508.03790v2.pdf)
- Clockwork RNNs：[Koutnik, et al. (2014)](http://arxiv.org/pdf/1402.3511v1.pdf)

#### 2.5.3.2 comparison

- [Greff, et al. (2015)](http://arxiv.org/pdf/1402.3511v1.pdf)
- [Jozefowicz, et al. (2015)](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)