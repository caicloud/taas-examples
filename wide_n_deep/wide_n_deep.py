# coding=utf-8
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from six.moves import urllib

import pandas as pd
import tensorflow as tf

# 引入 caicloud taas 基础库
from caicloud.clever.tensorflow import dist_base

# 定义特征属性
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]

# 定义常用配置
FLAGS = tf.app.flags.FLAGS
FLAGS.model_dir = dist_base.cfg.logdir
FLAGS.train_steps = dist_base.cfg.max_steps
FLAGS.model_type = "wide_n_deep"
FLAGS.train_data = "./train_data"
FLAGS.test_data = "./test_data"

# 构建 estimator
def build_estimator(model_dir, model_type):
  """Build an estimator."""
  # Sparse base columns.
  gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender",
                                                     keys=["female", "male"])
  education = tf.contrib.layers.sparse_column_with_hash_bucket(
      "education", hash_bucket_size=1000)
  relationship = tf.contrib.layers.sparse_column_with_hash_bucket(
      "relationship", hash_bucket_size=100)
  workclass = tf.contrib.layers.sparse_column_with_hash_bucket(
      "workclass", hash_bucket_size=100)
  occupation = tf.contrib.layers.sparse_column_with_hash_bucket(
      "occupation", hash_bucket_size=1000)
  native_country = tf.contrib.layers.sparse_column_with_hash_bucket(
      "native_country", hash_bucket_size=1000)

  # Continuous base columns.
  age = tf.contrib.layers.real_valued_column("age")
  education_num = tf.contrib.layers.real_valued_column("education_num")
  capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
  capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
  hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

  # Transformations.
  age_buckets = tf.contrib.layers.bucketized_column(age,
                                                    boundaries=[
                                                        18, 25, 30, 35, 40, 45,
                                                        50, 55, 60, 65
                                                    ])

  # Wide columns and deep columns.
  wide_columns = [gender, native_country, education, occupation, workclass,
                  relationship, age_buckets,
                  tf.contrib.layers.crossed_column([education, occupation],
                                                   hash_bucket_size=int(1e4)),
                  tf.contrib.layers.crossed_column(
                      [age_buckets, education, occupation],
                      hash_bucket_size=int(1e6)),
                  tf.contrib.layers.crossed_column([native_country, occupation],
                                                   hash_bucket_size=int(1e4))]
  deep_columns = [
      tf.contrib.layers.embedding_column(workclass, dimension=8),
      tf.contrib.layers.embedding_column(education, dimension=8),
      tf.contrib.layers.embedding_column(gender, dimension=8),
      tf.contrib.layers.embedding_column(relationship, dimension=8),
      tf.contrib.layers.embedding_column(native_country,
                                         dimension=8),
      tf.contrib.layers.embedding_column(occupation, dimension=8),
      age,
      education_num,
      capital_gain,
      capital_loss,
      hours_per_week,
  ]

  if model_type == "wide":
    m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                          feature_columns=wide_columns)
  elif model_type == "deep":
    m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                       feature_columns=deep_columns,
                                       hidden_units=[100, 50])
  else:
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50],
        fix_global_step_increment_bug=True)
  return m

# 定义输入函数
def input_fn(df):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {
      k: tf.SparseTensor(
          indices=[[i, 0] for i in range(df[k].size)],
          values=df[k].values,
          dense_shape=[df[k].size, 1])
      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label

# 加载训练数据集
df_train = pd.read_csv(
  tf.gfile.Open(FLAGS.train_data),
  names=COLUMNS,
  skipinitialspace=True,
  engine="python")

# 加载测试数据集
df_test = pd.read_csv(
  tf.gfile.Open(FLAGS.test_data),
  names=COLUMNS,
  skipinitialspace=True,
  skiprows=1,
  engine="python")

# 数据处理
# remove NaN elements
df_train = df_train.dropna(how='any', axis=0)
df_test = df_test.dropna(how='any', axis=0)

df_train[LABEL_COLUMN] = (
  df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
df_test[LABEL_COLUMN] = (
  df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

print("model directory = %s" % FLAGS.model_dir)

# 构建 estimator
m = build_estimator(FLAGS.model_dir, FLAGS.model_type)

# 构建 Experiment
exp = dist_base.Experiment(
    estimator = m,
    train_input_fn = lambda: input_fn(df_train),
    eval_input_fn = lambda: input_fn(df_test),
    eval_steps = 1)

# 调用 dist.Experiment.run() 方法来分布式执行 tf.contrib.learn 模型训练任务。
exp.run()
