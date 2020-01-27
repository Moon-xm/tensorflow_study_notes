#! /usr/bin/env python
# -*-coding:utf-8-*-  # 有中文文本时需要这个
# Author: Ming Chen
# create date: 2020-01-26 15:53:22
# description: 两层简单神经网络（全连接）（喂入一组数据）


import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 将提示等级降为2，表示不显示使用加速的警告
# 定义输入和参数
x = tf.constant([[0.7, 0.5]])  # 输入参数为：矩阵为1行两列的二阶张量
# 定义第一层的权重值为一个符合正态分布的变量，其中维度为两行三列，方差为1，均值为0，随机数种子为1(输入参数有两个，第一层隐藏层有3个神经元，故维度为（2,3）)
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, mean=0, seed=1))
# 以同样的方法定义变量w2为第二层的权重值,维度为三行一列（最后一个隐藏层有3个神经元，输出有1个参数，故维度为（3,1））
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 定义前向传播过程
a = tf.matmul(x, w1)  # 矩阵乘法，结果仍为一个矩阵
y = tf.matmul(a, w2)
# 到这神经网络的架构构建完毕

# 用会话计算结果
with tf.Session() as sess:  # 将会话函数简写为sess,别掉了（）
    init_op = tf.global_variables_initializer()  # 对所有变量进行初始化
    sess.run(init_op)  # 变量初始化
    print('y in tf3_4.py is {}'.format(sess.run(y)))

'''
y in tf3_4.py is [[3.0904665]]
'''