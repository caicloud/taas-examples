# 使用RNN预测股票价格

## 数据集
该样例使用的数据为从Yahoo Finance上下载的标普500从1993年2月1号到2017年4月26号每天的股票交易信息。下面展示了数据的格式：

```
Date,Open,High,Low,Close,Volume,Adj Close
1993-02-01,43.9687,44.25,43.9687,44.25,480500,28.077839
1993-02-02,44.2187,44.375,44.125,44.3437,201300,28.137295
1993-02-03,44.4062,44.8437,44.375,44.8125,529400,28.434761
1993-02-04,44.9687,45.0937,44.4687,45,531500,28.553735
1993-02-05,44.9687,45.0625,44.7187,44.9687,492100,28.533875
```

每一行包含了标普500一天的开盘、收盘、最高、最低、交易量和[调整后收盘价](http://www.investopedia.com/terms/a/adjusted_closing_price.asp)。


## 任务训练
通过以下脚本可以在本地训练：
```
./train_model.sh
```

运行该脚本可以得到类似下面的结果：
```
Training begins @ 2017-06-09 06:53:32.803296
Training loss at round 42 is: 0.872574619906, Testing loss is 1.60780370235, Learning rate is 0.0010000000475
Training loss at round 85 is: 0.186542152308, Testing loss is 0.589138686657, Learning rate is 0.000999000039883
Training loss at round 128 is: 0.0848133466721, Testing loss is 0.483780115843, Learning rate is 0.000998001080006
Training loss at round 171 is: 0.0477640048308, Testing loss is 0.314548194408, Learning rate is 0.000997003051452
Training loss at round 214 is: 0.0326054235826, Testing loss is 0.228872984648, Learning rate is 0.000996006070636
Training loss at round 257 is: 0.0257232856482, Testing loss is 0.178493395448, Learning rate is 0.000995010137558
Training loss at round 300 is: 0.0213768395005, Testing loss is 0.150800779462, Learning rate is 0.000994015135802
```

在训练结束后，模型在训练数据和测试数据上的拟合效果会被保存在文件```/tmp/caicloud-dist-tf/training_performance.jpg```和```/tmp/caicloud-dist-tf/testing_performance.jpg```中。其中```/tmp/caicloud-dist-tf```为默认的日志文件路径。


