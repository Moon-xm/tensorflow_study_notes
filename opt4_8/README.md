#### 神经网络训练八股之模块化实现
神经网络八股包括前向传播过程、反向传播过程中用到的正则化、
指数衰减学习率、滑动平均方法的设置以及测试模块
  
**前向传播过程（forward.py）**  
前向传播过程完成神经网络的搭建，结构如下：  
```python
def forward(x, regularizer):
    w =
    b =
    y =
    return y
def get_weight(shape, regularizer):
def get_bias(shape):
```
前向传播过程中，需要定义神经网络中的参数w和偏置b，定义由输入到输出的网络结构。通过定义输入到输出的网络结构。
通过定义函数`get_weight()`实现对参数w的设置，包括w的形状和是否正则化的标志。同样，通过定义`get_bias`
实现对偏置b的设置。  

**反向传播过程（backward.py)**  
反向传播过程完成网络参数的训练，结构如下：
```python
def backward(mnist):
    x = tf.placeholder(dtype, shape)
    y_ = tf.placeholder(dtype, shape)
    # 定义前向传播过程
    y = forward( )
    global_step = tf.Variable(0, trainable=False)
    loss = 
    train_step = tf.GradientDescentOptimizer(learninig_rate).minimize(loss, global_step=global_step)
    # train_step = tf.MomentumOptimizer(learning_rate).minimize(loss)
    # train_step = tf.AdamOptimizer(learning_rate).minimize(loss)
    # 实例化saver对象
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 初始化所有模型参数
        init_op = tf.global_valuables_initialzer()
        sess.run(init_op)
        # 训练模型
        for i in range(STEPS):
            sess.run(train_step, feed_dict={x: , y_: })
            if i % 轮数 == 0：
                print( )
                saver.save()  # 每间隔一定轮数保存一次模型
```
