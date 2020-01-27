#! /usr/bin/env python
# -*-coding:utf-8-*-  # 编码兼容格式为utf-8，有中文文本时需要这个
# Author: Ming Chen
# create date: 2020-01-27 16:12:43
# description: 正则化使用实例

# 0.导入模块 生成模拟数据集
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
BATCH_SIZE = 30
seed = 2
# 基于seed产生随机数
rdm = np.random.RandomState(seed)
# 随机数返回300行2列的矩阵 表示300组坐标点（x0, x1）作为输入数据集
X = rdm.randn(300, 2)  # 标准正态分布
# 从X这个300行2列的矩阵中取出一行 判断如果两个坐标x0, x1的平方和小于2 给Y赋值1 其余赋值0
# 作为输入数据集的标签（正确答案）
Y = [int(x0**x0 + x1**x1 < 2) for (x0, x1) in X]
# 遍历Y中的每个元素 1赋值'red'其余赋值'blue' 这样可视化显示时人可直观区分
Y_c = [['red' if y else 'blue'] for y in Y]
# 对数据集X和标签Y进行shape整理 第一个元素为-1表示随第二个参数计算得到 第二个元素表示多少列 把X整理为n行1列
X = np.vstack(X).reshape(-1, 2)
Y = np.vstack(Y).reshape(-1, 1)
print(X)
print(Y)
print(Y_c)
# 用plt.scatter画出数据集X各行中第0列元素和第一列元素的点即各行的(x0, x1) 用各行Y_c对应的值表示颜色（c是color的缩写）
plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
plt.show()


# 1.前向传播 定义神经网络的输入 参数 输出 定义前向传播过程
def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):  # 偏置b的shape其实就是该层神经元的个数
    b = tf.Variable(tf.constant(0.001, shape=shape))
    return b


x = tf.placeholder(dtype=tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = get_weight([2, 11], 0.01)
b1 = get_bias([11])
y1 = tf.nn.relu(tf.matmul(x, w1) + b1)  # 这里使用了relu激活函数

w2 = get_weight([11, 1], 0.01)
b2 = get_bias([1])
y = tf.matmul(y1, w2) + b2  # 输出层不过激活函数

# 2.反向传播 定义损失函数反向传播方法
# 定义损失函数
loss_mse = tf.reduce_mean(tf.square(y - y_))  # 均方误差损失函数
loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))
# 定义反向传播方法：不含正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)

# 3.生成会话 训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 300
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 2000 == 0:
            loss_mse_value = sess.run(loss_mse, feed_dict={x: X, y_: Y})
            print('After {:d} steps, loss is {:f}'.format(i, loss_mse_value))
    # xx在-3到3之间以步长为0.01 yy在-3到3之间以步长0.01 生成二维网格坐标点
    xx, yy = np.mgrid[-3:3:0.01, -3:3:0.01]
    # 将xx, yy拉直 并合并成一个2列的矩阵 得到一个网格坐标的集合
    grid = np.c_[xx.ravel(), yy.ravel()]
    # 将网格坐标点喂入神经网络 probs为输出
    probs = sess.run(y, feed_dict={x: grid})
    # prons的shape调整成xx的样子
    probs = probs.reshape(xx.shape)
    print('w1:\n', sess.run(w1))
    print('b1:\n', sess.run(b1))
    print('w2:\n', sess.run(w2))


plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
plt.contour(xx, yy, probs, levels=[0.5])
plt.show()


# 2.定义反向传播方法：包含正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)

# 3.生成会话 执行STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 300
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 2000 == 0:
            loss_total_val = sess.run(loss_total, feed_dict={x: X, y_: Y})
            print('After {:d} steps, loss is {:f}'.format(i, loss_total_val))

    xx, yy = np.mgrid[-3: 3: 0.01, -3: 3: 0.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = sess.run(y, feed_dict={x: grid})
    probs = probs.reshape(xx.shape)
    print('w1:\n', sess.run(w1))
    print('b1:\n', sess.run(b1))
    print('w2:\n', sess.run(w2))
    print('b2:\n', sess.run(b2))

plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
plt.contour(xx, yy, probs, levels=[.5])
plt.show()

'''
After 0 steps, loss is 12.144817
After 2000 steps, loss is 1.318070
After 4000 steps, loss is 0.293011
After 6000 steps, loss is 0.179373
After 8000 steps, loss is 0.132758
After 10000 steps, loss is 0.100489
After 12000 steps, loss is 0.077594
After 14000 steps, loss is 0.063770
After 16000 steps, loss is 0.056385
After 18000 steps, loss is 0.051990
After 20000 steps, loss is 0.049174
After 22000 steps, loss is 0.047067
After 24000 steps, loss is 0.045617
After 26000 steps, loss is 0.044766
After 28000 steps, loss is 0.043993
After 30000 steps, loss is 0.043286
After 32000 steps, loss is 0.042589
After 34000 steps, loss is 0.041995
After 36000 steps, loss is 0.041484
After 38000 steps, loss is 0.041065
w1:
 [[-0.7763609   0.3580811  -1.8237903  -1.2223221   0.5039403   1.7480683
   0.2739459  -0.00225758 -0.9721688  -0.11422906  0.8248707 ]
 [-0.9796288   0.66945165 -0.56836694 -1.5964259  -1.549233   -1.3533149
  -2.2495358  -0.0981008  -1.0928599   1.170701   -0.7077322 ]]
b1:
 [-0.12650141 -0.46506506  1.0086229   0.32397887  0.09646127  0.558807
 -0.27821523 -0.22740355  0.03187147  0.3588651   0.08363461]
w2:
 [[-0.778064  ]
 [-1.2323654 ]
 [-0.41984075]
 [-0.28998655]
 [-1.286832  ]
 [ 0.526014  ]
 [ 0.6570799 ]
 [-1.3866171 ]
 [ 1.6382204 ]
 [ 0.33649948]
 [-0.64686716]]
After 0 steps, loss is 19.680140
After 2000 steps, loss is 4.388752
After 4000 steps, loss is 1.702897
After 6000 steps, loss is 0.864388
After 8000 steps, loss is 0.409186
After 10000 steps, loss is 0.228755
After 12000 steps, loss is 0.186324
After 14000 steps, loss is 0.160637
After 16000 steps, loss is 0.141150
After 18000 steps, loss is 0.126124
After 20000 steps, loss is 0.114170
After 22000 steps, loss is 0.104022
After 24000 steps, loss is 0.096004
After 26000 steps, loss is 0.089058
After 28000 steps, loss is 0.084166
After 30000 steps, loss is 0.081148
After 32000 steps, loss is 0.079356
After 34000 steps, loss is 0.078305
After 36000 steps, loss is 0.077470
After 38000 steps, loss is 0.076929
w1:
 [[-1.2306420e-01  4.7802916e-01  2.0313358e-01 -5.9009982e-33
  -3.0259922e-01 -1.7292979e-01 -1.3724239e-01 -2.5713146e-01
  -6.3508320e-01 -1.5277162e-01  4.7707486e-01]
 [-3.2204068e-01 -8.7038077e-02 -7.3175180e-01 -4.9977670e-33
  -2.0452252e-01 -3.7532780e-01 -6.1409599e-01  1.1316938e-01
   2.8423481e-02 -9.2673264e-02  5.7133746e-01]]
b1:
 [ 0.2588098   0.34183538  0.28887513 -0.08699141 -0.07223538 -0.04522402
 -0.14686899  0.40995833  0.3818976  -0.03805553 -0.5609189 ]
w2:
 [[-3.4689659e-01]
 [ 4.2034057e-01]
 [-6.5323627e-01]
 [ 3.9244772e-33]
 [ 3.6204633e-01]
 [ 4.0625188e-01]
 [ 5.5934995e-01]
 [ 5.6417674e-01]
 [-6.4619040e-01]
 [ 1.7795861e-01]
 [-6.5410721e-01]]
b2:
 [0.31356204]
'''