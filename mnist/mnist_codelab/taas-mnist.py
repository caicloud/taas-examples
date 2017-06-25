# -*- coding: utf-8 -*-

import tensorflow as tf
from caicloud.clever.tensorflow import dist_base
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784     # 输入节点
OUTPUT_NODE = 10     # 输出节点
LAYER1_NODE = 500    # 隐藏层数                        
BATCH_SIZE = 100     # 每次batch打包的样本个数        

# 模型相关的参数
LEARNING_RATE = 0.05      

x = None
y_ = None
train_op = None
accuracy = None
local_step = 0
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

def inference(input_tensor):
    # 生成隐藏层的参数。
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数。
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
    return tf.matmul(layer1, weights2) + biases2

def model_fn(sync, num_replicas):
    global x, y_, train_op, accuracy
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    
    # 计算不含滑动平均类的前向传播结果
    y = inference(x)
    
    # 定义训练轮数
    global_step = tf.Variable(0, name='global_step', trainable=False)    
    
    # 计算交叉熵及其平均值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    # 优化损失函数
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    if sync:
        num_workers = num_replicas
        optimizer = tf.train.SyncReplicasOptimizer(
            optimizer,
            replicas_to_aggregate=num_workers,
            total_num_replicas=num_workers,
            name="mnist_sync_replicas")
    train_op = optimizer.minimize(cross_entropy_mean, global_step=global_step)
    
    # 计算正确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    return dist_base.ModelFnHandler(
        global_step=global_step,
        optimizer=optimizer,
        summary_op=None)

def train_fn(sess, num_global_step):
    global x, y_, train_op, accuracy, local_step, mnist

    if local_step % 1000 == 0:
        validate_acc = sess.run(
            accuracy, 
            feed_dict={x: mnist.validation.images, 
                       y_: mnist.validation.labels})
        print("After %d training step(s), validation accuracy using average model is %g " % (
            num_global_step, validate_acc))

    local_step += 1
    xs, ys = mnist.train.next_batch(BATCH_SIZE)
    sess.run(train_op,feed_dict={x:xs,y_:ys})
    
    return False

def after_train_hook(sess):
    global x, y_, train_op, accuracy, mnist

    test_feed = {x: mnist.test.images, y_: mnist.test.labels}
    test_acc = sess.run(accuracy,feed_dict=test_feed)
    print(("Test accuracy using average model is %g" %(test_acc)))

if __name__=='__main__':
    distTfRunner = dist_base.DistTensorflowRunner(
        model_fn=model_fn,
        after_train_hook=after_train_hook)
    distTfRunner.run(train_fn)
