#! /usr/bin/env python
# -*-coding:utf-8-*-  # 编码兼容格式为utf-8，有中文文本时需要这个
# Author: Ming Chen
# create date: 2020-01-27 10:21:37
# description: 设损失函数loss=（w+1）^2，令w初始值为常数5.反向传播就是求最优w，即求最小loss对应的w值

import tensorflow as tf
import os
# 定义待优化参数w初始值为5
w = tf.Variable(5, dtype=tf.float32)
# 定义损失函数loss
loss = tf.square(w + 1)
# 定义反向传播算法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
# 生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40
    for i in range(STEPS+1):
        sess.run(train_step)  # 先进行梯度下降法 更新参数
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print('after {:d} step(s), w is {:f}, loss is {:f}'.format(i, w_val, loss_val))

'''       
after 0 step(s), w is 2.600000, loss is 12.959999
after 1 step(s), w is 1.160000, loss is 4.665599
after 2 step(s), w is 0.296000, loss is 1.679616
after 3 step(s), w is -0.222400, loss is 0.604662
after 4 step(s), w is -0.533440, loss is 0.217678
after 5 step(s), w is -0.720064, loss is 0.078364
after 6 step(s), w is -0.832038, loss is 0.028211
after 7 step(s), w is -0.899223, loss is 0.010156
after 8 step(s), w is -0.939534, loss is 0.003656
after 9 step(s), w is -0.963720, loss is 0.001316
after 10 step(s), w is -0.978232, loss is 0.000474
after 11 step(s), w is -0.986939, loss is 0.000171
after 12 step(s), w is -0.992164, loss is 0.000061
after 13 step(s), w is -0.995298, loss is 0.000022
after 14 step(s), w is -0.997179, loss is 0.000008
after 15 step(s), w is -0.998307, loss is 0.000003
after 16 step(s), w is -0.998984, loss is 0.000001
after 17 step(s), w is -0.999391, loss is 0.000000
after 18 step(s), w is -0.999634, loss is 0.000000
after 19 step(s), w is -0.999781, loss is 0.000000
after 20 step(s), w is -0.999868, loss is 0.000000
after 21 step(s), w is -0.999921, loss is 0.000000
after 22 step(s), w is -0.999953, loss is 0.000000
after 23 step(s), w is -0.999972, loss is 0.000000
after 24 step(s), w is -0.999983, loss is 0.000000
after 25 step(s), w is -0.999990, loss is 0.000000
after 26 step(s), w is -0.999994, loss is 0.000000
after 27 step(s), w is -0.999996, loss is 0.000000
after 28 step(s), w is -0.999998, loss is 0.000000
after 29 step(s), w is -0.999999, loss is 0.000000
after 30 step(s), w is -0.999999, loss is 0.000000
after 31 step(s), w is -1.000000, loss is 0.000000
after 32 step(s), w is -1.000000, loss is 0.000000
after 33 step(s), w is -1.000000, loss is 0.000000
after 34 step(s), w is -1.000000, loss is 0.000000
after 35 step(s), w is -1.000000, loss is 0.000000
after 36 step(s), w is -1.000000, loss is 0.000000
after 37 step(s), w is -1.000000, loss is 0.000000
after 38 step(s), w is -1.000000, loss is 0.000000
after 39 step(s), w is -1.000000, loss is 0.000000
after 40 step(s), w is -1.000000, loss is 0.000000
'''