#! /usr/bin/env python
# -*-coding:utf-8-*-
# Author: Ming Chen
# create date: 2020-01-28 11:05:18
# description: 描述网络结构的前向传播过程

import tensorflow as tf


# 定义神经网络的输入 参数 输出 定义前向传播过程
def get_weight(shape, regularizer):
    """
    函数说明： 对权重参数w按标准正态分布进行随机初始化

    Parameter：
    ----------
        shape - 参数w的形状
        regularizer - 正则化权重
    Return:
    -------
        w - 初始化完成的参数w
    Author:
    -------
        Ming Chen
    Modify:
    -------
        2020-01-28 11:18:04
    """
    # 按正态分布对w进行随机初始化
    w = tf.Variable(tf.random_normal(shape), tf.float32)
    # if regularizer != None:
    # 将l2正则化损失加入到总损失losses中 防止过拟合
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    """
    函数说明： 对偏置参数b进行初始化 对b赋初值0.001

    Parameter：
    ----------
        shape - 参数b的形状 一般为当前层神经元的个数，如[1],[10]...
    Return:
    -------
        b - 初始化完成的b
    Author:
    -------
        Ming Chen
    Modify:
    -------
        2020-01-28 11:25:12
    """
    b = tf.Variable(tf.constant(0.001, shape=shape))
    return b


def forward(x, regularizer):
    """
    函数说明： 对前向传播网络结构进行设计 从输入到输出搭建完整的网络结构

    Parameter：
    ----------
        x - 输入数据
    Return:
    -------
        y - 输出数据 预测或分类的结果
    Author:
    -------
        Ming Chen
    Modify:
    -------
        2020-01-28 11:31:15
    """
    # 第一层
    w1 = get_weight([2, 11], regularizer)  # 这里加入了正则化参数regularizer
    b1 = get_bias([11])
    # 这里加入了relu激活函数
    a1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    # 输出层
    w2 = get_weight([11, 1], regularizer)
    b2 = get_bias([1])
    y = tf.matmul(a1, w2) + b2  # 输出层不经过激活

    return y
