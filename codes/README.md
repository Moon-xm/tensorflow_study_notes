用于存放课程代码



神经网络训练八股之非模块化实现

##### 0. 导入模块，生成模拟数据集X、Y
import...  
常量定义  
生成数据集
##### 1. 前向传播：定义参数w，b等，定义输入、输出、前向传播方法
x = 	y_ =  
w1 = 	w2 =  
a = 	y =    
##### 2. 反向传播：定义损失函数，反向传播方法（梯度下降法）
loss =   
train_step = 
##### 3. 生成会话，训练STEPS轮
```python
with tf.Session() as sess:
	init_op = tf.global_variable_initializer()
	sess.run(init_op)
	STPES = 3000
	for i in range(STEPS):
		start = 
		end = 
		sess.run(train_step, feed_dict={x: , y_: }):
		if i % 200 == 0:
			loss_val = sess.run(loss, feed_dict={})
			print("After {:d} steps, loss is {:f}".format(i, loss_val))
```
