#! /usr/bin/env python
# -*-coding:utf-8-*-
# Author: Ming Chen
# create date: 2020-01-28 12:06:30
# description: 描述网络参数优化方法的反向传播过程

# 0.导入模块 生成模拟数据集
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import opt4_8.opt4_8_generateds as generateds
import opt4_8.opt4_8_forward as forward

# 定义超参数
STEPS = 40000  # 训练轮数
BATCH_SIZE = 30  # 每次迭代的样本个数 太小导致收敛速度慢 太大导致训练时间长
LEARNING_RATE_BASE = 0.01  # 初始学习率 一般取一个较小的值0.01或0.001
LEARNING_RATE_DECAY = 0.999  # 学习率的衰减率  一般取0.99或0.999
REGULARIZER = 0.01  # 正则化权重：太小导致过拟合且损失函数会变小 太大导致欠拟合且损失函数会变大


def backward():
    """
    函数说明： 反向传播过程

    Parameter：
    ----------
        None
    Return:
    -------
        None
    Author:
    -------
        Ming Chen
    Modify:
    -------
        2020-01-28 20:30:38
    """
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))  # 正确的标签

    X, Y, Y_c = generateds.generateds()

    y = forward.forward(x, REGULARIZER)  # 训练得到的标签

    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.exponential_decay(  # 指数衰减学习率
        LEARNING_RATE_BASE, global_step,
        300/BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)

    # 定义损失函数
    loss_mse = tf.reduce_mean(tf.square(y - y_))  # 均方误差定义损失函数
    loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))  # 引入正则化的损失函数

    # 定义反向传播方法：包含正则化
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)

    # 3.生成会话 训练STEPS轮
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(STEPS):
            start = (i * BATCH_SIZE) % 300
            end = start + BATCH_SIZE
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
            if i % 2000 == 0:
                loss_val = sess.run(loss_total, feed_dict={x: X, y_: Y})
                print('After {:d} steps, loss is {:f}'.format(i, loss_val))

        xx, yy = np.mgrid[-3: 3: 0.01, -3: 3: 0.01]  # 生成网格坐标集
        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = sess.run(y, feed_dict={x: grid})
        probs = probs.reshape(xx.shape)

    plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
    plt.contour(xx, yy, probs, levels=[0.5])
    plt.show()


if __name__ == '__main__':
    backward()

'''
After 0 steps, loss is 6.552093
After 2000 steps, loss is 0.204564
After 4000 steps, loss is 0.150027
After 6000 steps, loss is 0.114320
After 8000 steps, loss is 0.099608
After 10000 steps, loss is 0.094467
After 12000 steps, loss is 0.092948
After 14000 steps, loss is 0.092306
After 16000 steps, loss is 0.091898
After 18000 steps, loss is 0.091764
After 20000 steps, loss is 0.091678
After 22000 steps, loss is 0.091189
After 24000 steps, loss is 0.091016
After 26000 steps, loss is 0.090942
After 28000 steps, loss is 0.090897
After 30000 steps, loss is 0.090869
After 32000 steps, loss is 0.090850
After 34000 steps, loss is 0.090838
After 36000 steps, loss is 0.090828
After 38000 steps, loss is 0.090821
'''