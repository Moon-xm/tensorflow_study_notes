#! /usr/bin/env python
# -*-coding:utf-8-*-
# Author: Ming Chen
# create date: 2020-01-28 11:05:04
# description: 生成数据集

import numpy as np
seed = 2


def generateds():
    """
    函数说明： 生成数据集

    Parameter：
    ----------
        None
    Return:
    -------
        X - 输入数据集
        Y - 正确的输出标签
        Y_c - 正确的输出颜色标签
    Author:
    -------
        Ming Chen
    Modify:
    -------
        2020-01-28 20:33:42
    """
    # 基于seed产生随机数
    rdm = np.random.RandomState(seed)
    # 随机数返回300行2列的矩阵 表示300组坐标点(x0, x1) 作为输入数据集
    X = rdm.randn(300, 2)
    # 从X这个300行2列的矩阵中取出一行 判断如果两坐标的平方和小于2 给Y赋值1 其余赋值0
    # 作为输入数据集的标签（正确答案）
    Y = [int(x0**2 + x1**2 < 2) for (x0, x1) in X]
    # 遍历Y中的每一个元素 1赋值'red' 0赋值为'blue' 这样可视化显示时人可以直观区分
    Y_c = [['red' if y else 'blue'] for y in Y]
    # 对数据集X和标签Y进行形状整理 第一个元素为-1表示跟随第二列计算 第二个元素表示多少列 可见X为两列 Y为1列
    X = np.vstack(X).reshape(-1, 2)  # np.vstack()表示按垂直方向（行顺序）堆叠数组构成一个新的数组
    Y = np.vstack(Y).reshape(-1, 1)

    return X, Y, Y_c

