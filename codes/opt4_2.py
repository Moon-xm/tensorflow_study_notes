#! /usr/bin/env python
# -*-coding:utf-8-*-  # 编码兼容格式为utf-8，有中文文本时需要这个
# Author: Ming Chen
# create date: 2020-01-26 21:23:14
# description: 自定义损失函数实例,预测酸奶销量，x1和x2为影响因素，最后正确的酸奶销量满足y = 3*x1 + 2*x2，这里只有输入和输出层，无隐藏层

# 0导入模块，生成模拟数据集
import tensorflow as tf
import numpy as np
import os
COST = 1  # 酸奶成本为1元
PROFIT = 9  # 利润为9元，预测少了损失大，故不要预测少，生成的模型会多预测一些,即w1两个参数最后会比3和4大些
BATCH_SIZE = 5
SEED = 23455

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
rng = np.random.RandomState(SEED)
X = rng.uniform(0, 10, size=(32, 2))
Y = [[3 * x1 + 4 * x2 + (rng.rand() * 0.3 - 0.01)] for (x1, x2) in X]  # 这里的rng.rand()*0.3 - 0.01)代表加入了0.02~0.04的噪声
print('X:\n', X)
print('Y:\n', Y)

# 1前向传播，定义输入、参数、输出，前向传播过程
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))  # y_为真实的酸奶销量
w1 = tf.Variable(tf.random_normal([2, 1], stddev=2, mean=2, seed=1))
y = tf.matmul(x, w1)  # y为预测的酸奶销量

# 2反向传播，定义损失函数及优化算法
# loss_mse = tf.reduce_mean(tf.square(y - y_))
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_)*COST, (y_ - y)*PROFIT))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
# train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
# train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)

# 3生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 查看随机初始化得到的参数
    print('训练前w1:\n', sess.run(w1))
    # 开始训练
    STEPS = 10000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print('After {:d} step(s), total loss is {:f}'.format(i, total_loss))
    print('训练后w1:\n', sess.run(w1))

'''
X:
 [[8.34943185 1.14829514]
 [6.6899751  4.65949873]
 [6.01816656 5.88384085]
 [3.18366559 2.05020717]
 [8.70439444 0.2679395 ]
 [4.15398112 4.39383693]
 [6.8635684  2.48334041]
 [9.73152283 6.85418494]
 [0.30816171 8.94799125]
 [2.46657146 2.85848615]
 [3.13756669 4.77183486]
 [5.66892543 7.70791484]
 [7.32160404 3.58289631]
 [1.57248425 9.42945836]
 [3.4933722  8.46344829]
 [5.0304053  8.12996193]
 [2.38698859 9.89560401]
 [4.63650096 3.25310938]
 [3.65104865 9.73655218]
 [7.33502381 8.38330132]
 [6.18101581 1.2580353 ]
 [5.92748167 1.87798282]
 [8.71502995 3.46795007]
 [2.58832194 5.00029325]
 [7.5690948  8.34298243]
 [2.93166488 0.5646578 ]
 [1.04091338 8.82351658]
 [0.67277847 5.77847609]
 [3.84927053 4.8384792 ]
 [6.92344279 1.9687348 ]
 [4.27834924 7.34169855]
 [0.96960695 0.4883936 ]]
Y:
 [[29.84155160289988], [38.93345953184356], [41.74207518565044], [17.93701015408264], [27.22306932139818], [30.11474225088459], [30.644262181692852], [56.83939952565798], [36.790606599668926], [18.91501132075534], [28.62049475268354], [47.862908871793366], [36.440992461285795], [42.5388519575809], [44.40724052576013], [47.73240098043207], [46.9147644714289], [27.181961296647962], [50.08822033451744], [55.69877769936475], [23.740764947102985], [25.290632594629493], [40.23698604607095], [27.993847059913236], [56.11883751730495], [11.311506105974667], [38.68083561756229], [25.16807606121678], [31.013846123242583], [28.760841500409896], [42.29274817740124], [5.137238026491332]]
训练前w1:
 [[0.37736356]
 [4.9691973 ]]
After 0 step(s), total loss is 2468.052246
After 500 step(s), total loss is 1551.232178
After 1000 step(s), total loss is 1018.082520
After 1500 step(s), total loss is 761.550537
After 2000 step(s), total loss is 557.596436
After 2500 step(s), total loss is 430.080139
After 3000 step(s), total loss is 336.383759
After 3500 step(s), total loss is 280.497101
After 4000 step(s), total loss is 227.643600
After 4500 step(s), total loss is 171.924896
After 5000 step(s), total loss is 113.699669
After 5500 step(s), total loss is 51.896706
After 6000 step(s), total loss is 9.076775
After 6500 step(s), total loss is 9.083950
After 7000 step(s), total loss is 9.113403
After 7500 step(s), total loss is 9.117704
After 8000 step(s), total loss is 9.107155
After 8500 step(s), total loss is 9.121527
After 9000 step(s), total loss is 9.113274
After 9500 step(s), total loss is 9.125711
训练后w1:
 [[3.0367317]
 [4.025194 ]]
'''