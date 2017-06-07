# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn  
import matplotlib.pyplot as plt

from caicloud.clever.tensorflow import dist_base

tf.app.flags.DEFINE_integer("rnn_hidden_nodes", 10, "Number of nodes in the LSTM structure.")
tf.app.flags.DEFINE_integer("rnn_num_steps", 20, "The length of the RNN structure.")
tf.app.flags.DEFINE_integer("batch_size", 128, "training batch size.")
tf.app.flags.DEFINE_float("learning_rate_base", 0.001, "base learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay", 0.99, "learning rate decay.")

FLAGS = tf.app.flags.FLAGS

NUM_FEATURES = 6
LABEL_INDEX = 0

_current_loss = 0.0
_start_index = 0
_end_index = 0 
_n_rounds = 0

f = open('all_data.csv') 
df = pd.read_csv(f)
dataset = df.iloc[:, 1:].values

_mean = np.mean(dataset, axis=0)
_std = np.std(dataset, axis=0)
dataset = (dataset - _mean) / _std

train_size = int(len(dataset) * 0.9)
test_size = len(dataset) - train_size
train, test = dataset[:train_size, :], dataset[train_size:len(dataset), :]
print(len(train), len(test))

def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset) - FLAGS.rnn_num_steps - 1):
        a = dataset[i:(i + FLAGS.rnn_num_steps), :]
        dataX.append(a.tolist())
        dataY.append([dataset[i + FLAGS.rnn_num_steps, LABEL_INDEX]])
    return dataX, dataY
_train_x, _train_y = create_dataset(train)
_test_x, _test_y = create_dataset(test)


def lstm(X):
    batch_size = tf.shape(X)[0]

    w_in = tf.Variable(tf.random_normal([NUM_FEATURES, FLAGS.rnn_hidden_nodes]))
    b_in = tf.Variable(tf.constant(0.1, shape=[FLAGS.rnn_hidden_nodes]))
    
    input = tf.reshape(X, [-1, NUM_FEATURES])
   
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, FLAGS.rnn_num_steps, FLAGS.rnn_hidden_nodes])
    cell = rnn.BasicLSTMCell(FLAGS.rnn_hidden_nodes, state_is_tuple=True)

    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = output_rnn[:, -1, :]
    
    w_out = tf.Variable(tf.random_normal([FLAGS.rnn_hidden_nodes, 1]))
    b_out = tf.Variable(tf.constant(0.1, shape=[1]))
    pred = tf.matmul(output, w_out) + b_out
    return pred

def model_fn(sync, num_replicas):
    global _train_op, _loss, _train_x, _test_x, _test_y, _pred, _learning_rate, _X, _Y, _n_rounds
    
    _X = tf.placeholder(tf.float32, shape=[None, FLAGS.rnn_num_steps, NUM_FEATURES])
    _Y = tf.placeholder(tf.float32, shape=[None, 1])
    
    global_step = tf.contrib.framework.get_or_create_global_step()
    _pred = lstm(_X)
    _loss = tf.reduce_mean(tf.square(_pred - _Y))

    _n_rounds = (len(_train_x) - 1) / FLAGS.batch_size + 1 
    _learning_rate = tf.train.exponential_decay(
        FLAGS.learning_rate_base,
        global_step,
        _n_rounds,
        FLAGS.learning_rate_decay,
        staircase=True)
    
    optimizer = tf.train.AdamOptimizer(_learning_rate) 
    _train_op = optimizer.minimize(_loss, global_step=global_step)
    
    def mse_evalute_fn(session):
        return session.run(_loss, feed_dict={_X:_test_x, _Y:_test_y})

    # 定义模型评测（准确率）的计算方法
    model_metric_ops = {
        "adjusted_mse": mse_evalute_fn
    }
    
    return dist_base.ModelFnHandler(
        global_step=global_step,
        optimizer=optimizer,
        model_metric_ops=model_metric_ops,
        summary_op=None)

def train_fn(session, num_global_step):
    global _train_op, _loss, _current_loss, _start_index, _end_index, _train_x, _train_y, _learning_rate
    global _n_rounds, _X, _Y, _test_x, _test_y
    
    _end_index = _start_index + FLAGS.batch_size
    if _end_index > len(_train_x): _end_index = len(_train_x)
    
    _, lr, loss = session.run([_train_op, _learning_rate, _loss],
                          feed_dict={_X:_train_x[_start_index:_end_index], _Y:_train_y[_start_index:_end_index]})
    _current_loss += loss
            
    if _end_index == len(_train_x):
        loss = session.run(_loss, feed_dict={_X:_test_x, _Y:_test_y})
        print("Training loss at round {} is: {}, Testing loss is {}, Learning rate is {}".format(
            num_global_step, _current_loss / _n_rounds, loss, lr))
        _current_loss = 0.0
    
    _start_index = _end_index
    if _start_index == len(_train_x): _start_index = 0
       
    return False

def after_train_hook(session):
    global _train_x, _train_y, _X, _Y, _test_x, _test_y, _pred, _std, _mean
    print("Training done.")
    print(dist_base.cfg.logdir)
    
    predicted = session.run(_pred, feed_dict={_X:_train_x})
    predicted = np.asarray(predicted) * _std[LABEL_INDEX] + _mean[LABEL_INDEX]
    true_y = np.asarray(_train_y) * _std[LABEL_INDEX] + _mean[LABEL_INDEX]
    plt.figure()
    plt.plot(list(range(len(predicted))), predicted, color='b', label='Predicted')
    plt.plot(list(range(len(true_y))), true_y, color='r', label='True Data')
    plt.legend(loc='upper center')
    plt.savefig(os.path.join(dist_base.cfg.logdir, "training_performance.jpg"))
    
    predicted = session.run(_pred, feed_dict={_X:_test_x}) 
    predicted = np.asarray(predicted) * std[LABEL_INDEX] + mean[LABEL_INDEX]
    true_y = np.asarray(_test_y) * std[LABEL_INDEX] + mean[LABEL_INDEX]
    plt.figure()
    plt.plot(list(range(len(predicted))), predicted, color='b', label='Predicted')
    plt.plot(list(range(len(true_y))), true_y, color='r', label='True Data')
    plt.legend(loc='upper center')
    plt.savefig(os.path.join(dist_base.cfg.logdir, "testing_performance.jpg"))
    
if __name__ == '__main__':
    distTfRunner = dist_base.DistTensorflowRunner(
        model_fn=model_fn, 
        after_train_hook=after_train_hook, 
        gen_init_fn=None)
    
    distTfRunner.run(train_fn)
