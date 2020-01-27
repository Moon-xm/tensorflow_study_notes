#! /usr/bin/env python
# -*-coding:utf-8-*-  # 编码兼容格式为utf-8，有中文文本时需要这个
# Author: Ming Chen
# create date: 2020-01-27 10:50:12
# description: 设损失函数loss=（w+1）^2，令w初始值为常数10.反向传播就是求最优w，即求最小loss对应的w值
# 使用指数衰减的学习率 在迭代初期得到较高的下降速度 可以在较小的训练轮数下取得更有效的收敛度

import tensorflow as tf

# 定义超参数
LEARNING_RATE_BASE = 0.1  # 初始学习率
LEARNING_RATE_DECAY = 0.99  # 学习率衰减率
LEARNING_RATE_STEP = 1  # 训练代数 表示进行第几次遍历数据集 喂入多少轮BATCH_SIZE后 更新一次学习率 一般设置为：总样本数/BATCH_SIZE

# 运行了几轮的BATCH_SIZE计数器，初始值为0，设置为不可训练
global_step = tf.Variable(0, trainable=False)
# 定义指数衰减学习率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, LEARNING_RATE_STEP,
                                           LEARNING_RATE_DECAY, staircase=False)
# 定义待优化的参数w
w = tf.Variable(tf.constant(5, dtype=tf.float32))
# 定义损失函数
loss = tf.square(w + 1)
# 定义反向传播方法
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
# 生成会话，训练40轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40
    for i in range(STEPS+1):
        sess.run(train_step)  # 先进行梯度下降法 更新参数
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        learning_rate_val = sess.run(learning_rate)
        global_step_val = sess.run(global_step)
        print('after {:d} steps: global steps is {:d}, w is {:f}, learning rate is {:f}'
              ', loss is {:f}'.format(i, global_step_val, w_val, learning_rate_val, loss_val))

'''
after 0 steps: global steps is 1, w is 3.800000, learning rate is 0.099000, loss is 23.040001
after 1 steps: global steps is 2, w is 2.849600, learning rate is 0.098010, loss is 14.819419
after 2 steps: global steps is 3, w is 2.095001, learning rate is 0.097030, loss is 9.579033
after 3 steps: global steps is 4, w is 1.494386, learning rate is 0.096060, loss is 6.221961
after 4 steps: global steps is 5, w is 1.015167, learning rate is 0.095099, loss is 4.060896
after 5 steps: global steps is 6, w is 0.631886, learning rate is 0.094148, loss is 2.663051
after 6 steps: global steps is 7, w is 0.324608, learning rate is 0.093207, loss is 1.754587
after 7 steps: global steps is 8, w is 0.077684, learning rate is 0.092274, loss is 1.161403
after 8 steps: global steps is 9, w is -0.121202, learning rate is 0.091352, loss is 0.772287
after 9 steps: global steps is 10, w is -0.281761, learning rate is 0.090438, loss is 0.515867
after 10 steps: global steps is 11, w is -0.411674, learning rate is 0.089534, loss is 0.346128
after 11 steps: global steps is 12, w is -0.517024, learning rate is 0.088638, loss is 0.233266
after 12 steps: global steps is 13, w is -0.602644, learning rate is 0.087752, loss is 0.157891
after 13 steps: global steps is 14, w is -0.672382, learning rate is 0.086875, loss is 0.107334
after 14 steps: global steps is 15, w is -0.729305, learning rate is 0.086006, loss is 0.073276
after 15 steps: global steps is 16, w is -0.775868, learning rate is 0.085146, loss is 0.050235
after 16 steps: global steps is 17, w is -0.814036, learning rate is 0.084294, loss is 0.034583
after 17 steps: global steps is 18, w is -0.845387, learning rate is 0.083451, loss is 0.023905
after 18 steps: global steps is 19, w is -0.871193, learning rate is 0.082617, loss is 0.016591
after 19 steps: global steps is 20, w is -0.892476, learning rate is 0.081791, loss is 0.011561
after 20 steps: global steps is 21, w is -0.910065, learning rate is 0.080973, loss is 0.008088
after 21 steps: global steps is 22, w is -0.924629, learning rate is 0.080163, loss is 0.005681
after 22 steps: global steps is 23, w is -0.936713, learning rate is 0.079361, loss is 0.004005
after 23 steps: global steps is 24, w is -0.946758, learning rate is 0.078568, loss is 0.002835
after 24 steps: global steps is 25, w is -0.955125, learning rate is 0.077782, loss is 0.002014
after 25 steps: global steps is 26, w is -0.962106, learning rate is 0.077004, loss is 0.001436
after 26 steps: global steps is 27, w is -0.967942, learning rate is 0.076234, loss is 0.001028
after 27 steps: global steps is 28, w is -0.972830, learning rate is 0.075472, loss is 0.000738
after 28 steps: global steps is 29, w is -0.976931, learning rate is 0.074717, loss is 0.000532
after 29 steps: global steps is 30, w is -0.980378, learning rate is 0.073970, loss is 0.000385
after 30 steps: global steps is 31, w is -0.983281, learning rate is 0.073230, loss is 0.000280
after 31 steps: global steps is 32, w is -0.985730, learning rate is 0.072498, loss is 0.000204
after 32 steps: global steps is 33, w is -0.987799, learning rate is 0.071773, loss is 0.000149
after 33 steps: global steps is 34, w is -0.989550, learning rate is 0.071055, loss is 0.000109
after 34 steps: global steps is 35, w is -0.991035, learning rate is 0.070345, loss is 0.000080
after 35 steps: global steps is 36, w is -0.992297, learning rate is 0.069641, loss is 0.000059
after 36 steps: global steps is 37, w is -0.993369, learning rate is 0.068945, loss is 0.000044
after 37 steps: global steps is 38, w is -0.994284, learning rate is 0.068255, loss is 0.000033
after 38 steps: global steps is 39, w is -0.995064, learning rate is 0.067573, loss is 0.000024
after 39 steps: global steps is 40, w is -0.995731, learning rate is 0.066897, loss is 0.000018
after 40 steps: global steps is 41, w is -0.996302, learning rate is 0.066228, loss is 0.000014
'''