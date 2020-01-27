#! /usr/bin/env python
# -*-coding:utf-8-*-  # 编码兼容格式为utf-8，有中文文本时需要这个
# Author: Ming Chen
# create date: 2020-01-26 16:27:48
# description: 两层简单神经网络（全连接）(喂入多组数据)

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 停用警告

# 定义输入和参数
# 用placeholder定义输入（sess.run喂入多组数据）
# 由于不知道喂入多少组数据，所以第一个参数写为None，第二个参数写为2因为知道参数只有两个维度（体积、重量），这样在计算y时可以一次喂入多组特征
x = tf.placeholder(tf.float32, shape=(None, 2))
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))  # 随机初始化第一层的参数
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))  # 同理随机初始化第二层的参数

# 定义前向传播过程
a = tf.matmul(x, w1)  # 矩阵乘法，结果仍为一个矩阵
y = tf.matmul(a, w2)

# 定义会话计算结果
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()  # 对所有参数进行初始化
    sess.run(init_op)  # 执行初始化
    # feed_dict 表示同时喂入多组数据（注意形式是字典）
    print('the result in tf3_5.py is \n', sess.run(y, feed_dict={x: [[0.7, 0.5], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]}))
    # 可以查看随机初始化的参数w1和w2
    print('w1\n:', sess.run(w1))
    print('w2\n', sess.run(w2))

'''
the result in tf3_5.py is 
 [[3.0904665]
 [1.2236414]
 [1.7270732]
 [2.2305048]]
w1
: [[-0.8113182   1.4845988   0.06532937]
 [-2.4427042   0.0992484   0.5912243 ]]
w2
 [[-0.8113182 ]
 [ 1.4845988 ]
 [ 0.06532937]]
 '''