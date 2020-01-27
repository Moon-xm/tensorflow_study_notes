#! /usr/bin/env python
# -*-coding:utf-8-*-  # 编码兼容格式为utf-8，有中文文本时需要这个
# Author: Ming Chen
# create date: 2020-01-26 21:23:14
# description: 均方误差损失函数实例,预测酸奶销量，x1和x2为影响因素，最后的销量满足y = 3*x1 + 2*x2 + x3，这里只有输入和输出层，无隐藏层

# 1导入模块，生成模拟数据集
import tensorflow as tf
import numpy as np
import os

BATCH_SIZE = 5
SEED = 56
rdm = np.random.RandomState(SEED)
X = rdm.rand(40, 3)
Y = [[3 * x1 + 2 * x2 + x3 + ((rdm.rand() / 10 - 0.5) * 0.02)] for (x1, x2, x3) in X]  # 加入了-0.01~0.01的噪声

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 1前向传播，定义输入 参数 输出 前向传播过程
x = tf.placeholder(tf.float32, shape=(None, 3))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([3, 1], stddev=2, mean=2, seed=5))
y = tf.matmul(x, w1)

# 2反向传播 定义损失函数 优化算法
loss = tf.reduce_mean(tf.square(y - y_))
# train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)  # 梯度下降法
# train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 3生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print('before training the w1 is:\n', sess.run(w1))
    STEPS = 50000
    for i in range(STEPS+1):
        start = (i*BATCH_SIZE) % 40
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print('After {:d} step(s), the total loss is {:f}'.format(i, total_loss))
    print('After training, the w1 is :', sess.run(w1))

'''
before training the w1 is:
 [[-0.35616565]
 [ 0.5801172 ]
 [-1.496316  ]]
After 0 step(s), the total loss is 14.549959
After 500 step(s), the total loss is 9.619417
After 1000 step(s), the total loss is 6.061406
After 1500 step(s), the total loss is 3.596547
After 2000 step(s), the total loss is 1.985290
After 2500 step(s), the total loss is 1.014870
After 3000 step(s), the total loss is 0.493396
After 3500 step(s), the total loss is 0.251238
After 4000 step(s), the total loss is 0.153413
After 4500 step(s), the total loss is 0.112212
After 5000 step(s), the total loss is 0.086477
After 5500 step(s), the total loss is 0.064233
After 6000 step(s), the total loss is 0.044407
After 6500 step(s), the total loss is 0.027983
After 7000 step(s), the total loss is 0.015695
After 7500 step(s), the total loss is 0.007590
After 8000 step(s), the total loss is 0.003033
After 8500 step(s), the total loss is 0.000947
After 9000 step(s), the total loss is 0.000218
After 9500 step(s), the total loss is 0.000038
After 10000 step(s), the total loss is 0.000010
After 10500 step(s), the total loss is 0.000008
After 11000 step(s), the total loss is 0.000008
After 11500 step(s), the total loss is 0.000007
After 12000 step(s), the total loss is 0.000007
After 12500 step(s), the total loss is 0.000007
After 13000 step(s), the total loss is 0.000007
After 13500 step(s), the total loss is 0.000007
After 14000 step(s), the total loss is 0.000007
After 14500 step(s), the total loss is 0.000007
After 15000 step(s), the total loss is 0.000007
After 15500 step(s), the total loss is 0.000007
After 16000 step(s), the total loss is 0.000008
After 16500 step(s), the total loss is 0.000007
After 17000 step(s), the total loss is 0.000008
After 17500 step(s), the total loss is 0.000007
After 18000 step(s), the total loss is 0.000008
After 18500 step(s), the total loss is 0.000007
After 19000 step(s), the total loss is 0.000008
After 19500 step(s), the total loss is 0.000007
After 20000 step(s), the total loss is 0.000008
After 20500 step(s), the total loss is 0.000007
After 21000 step(s), the total loss is 0.000008
After 21500 step(s), the total loss is 0.000007
After 22000 step(s), the total loss is 0.000008
After 22500 step(s), the total loss is 0.000007
After 23000 step(s), the total loss is 0.000008
After 23500 step(s), the total loss is 0.000007
After 24000 step(s), the total loss is 0.000008
After 24500 step(s), the total loss is 0.000007
After 25000 step(s), the total loss is 0.000008
After 25500 step(s), the total loss is 0.000007
After 26000 step(s), the total loss is 0.000008
After 26500 step(s), the total loss is 0.000007
After 27000 step(s), the total loss is 0.000008
After 27500 step(s), the total loss is 0.000007
After 28000 step(s), the total loss is 0.000008
After 28500 step(s), the total loss is 0.000007
After 29000 step(s), the total loss is 0.000008
After 29500 step(s), the total loss is 0.000007
After 30000 step(s), the total loss is 0.000008
After 30500 step(s), the total loss is 0.000007
After 31000 step(s), the total loss is 0.000008
After 31500 step(s), the total loss is 0.000007
After 32000 step(s), the total loss is 0.000008
After 32500 step(s), the total loss is 0.000007
After 33000 step(s), the total loss is 0.000008
After 33500 step(s), the total loss is 0.000007
After 34000 step(s), the total loss is 0.000008
After 34500 step(s), the total loss is 0.000007
After 35000 step(s), the total loss is 0.000008
After 35500 step(s), the total loss is 0.000007
After 36000 step(s), the total loss is 0.000008
After 36500 step(s), the total loss is 0.000007
After 37000 step(s), the total loss is 0.000008
After 37500 step(s), the total loss is 0.000007
After 38000 step(s), the total loss is 0.000008
After 38500 step(s), the total loss is 0.000007
After 39000 step(s), the total loss is 0.000008
After 39500 step(s), the total loss is 0.000007
After 40000 step(s), the total loss is 0.000008
After 40500 step(s), the total loss is 0.000007
After 41000 step(s), the total loss is 0.000008
After 41500 step(s), the total loss is 0.000007
After 42000 step(s), the total loss is 0.000008
After 42500 step(s), the total loss is 0.000007
After 43000 step(s), the total loss is 0.000008
After 43500 step(s), the total loss is 0.000007
After 44000 step(s), the total loss is 0.000008
After 44500 step(s), the total loss is 0.000007
After 45000 step(s), the total loss is 0.000008
After 45500 step(s), the total loss is 0.000007
After 46000 step(s), the total loss is 0.000008
After 46500 step(s), the total loss is 0.000007
After 47000 step(s), the total loss is 0.000008
After 47500 step(s), the total loss is 0.000007
After 48000 step(s), the total loss is 0.000008
After 48500 step(s), the total loss is 0.000007
After 49000 step(s), the total loss is 0.000008
After 49500 step(s), the total loss is 0.000007
After 50000 step(s), the total loss is 0.000008
After training, the w1 is : [[2.99379   ]
 [1.9943942 ]
 [0.99568707]]
'''