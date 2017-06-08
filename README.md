# TaaS 平台样例模型训练样例任务代码

本目录下提供了一些在 TaaS 平台（[公有云平台](https://taas.caicloud.io)）上训练深度学习模型的样例代码。这些代码也可以在 Caicloud 提供的本地开发环境上运行。运行以下命令可以安装本地开发环境：

```
sudo pip install caicloud.tensorflow
```

[这里]（https://docs.caicloud.io/clever/develop/env/local-dev-pkg.html）更加详细的介绍了如何配置本地环境，但运行本代码库中的代码只需要运行以上命令即可。


## TaaS 基本操作样例代码

|    样例名称                   |              样例内容                    |
| ----------------------------- | ----------------------------------------:|
| [half\_plus\_two](https://github.com/caicloud/taas-examples/tree/master/half_plus_two)        | 导出一个线性回归推理模型，以及在导出模型时如何添加附加的文件   | 
| [two\_inputs\_three\_outputs](https://github.com/caicloud/taas-examples/tree/master/two_inputs_three_outputs)        |   包含两个输入和三个输出的模型的导出以及如何 Serving 中过滤模型输出   | 


## TaaS 实现深度学习模型

|    样例名称                   |              样例内容                    |
| ----------------------------- | ----------------------------------------:|
| [MNIST](https://github.com/caicloud/taas-examples/tree/master/mnist)      | 分布式训练手写体识别模型、自定义模型初始化、导出模型等   | 
| [Boston House](https://github.com/caicloud/taas-examples/tree/master/boston_house)      | 使用tf.contrib.learn实现房价预测问题  | 
| [PTB(Penn Tree Bank)](https://github.com/caicloud/taas-examples/tree/master/ptb)      | 使用RNN训练自然语言模型  | 
| [Stock Price Prediction](https://github.com/caicloud/taas-examples/tree/master/time_series/stock_price) | 使用RNN预测SPY价格 |
| [recommandation](https://github.com/caicloud/taas-examples/tree/master/recommandation) | 使用embedding的思想实现用户对电影评分的预测  |






