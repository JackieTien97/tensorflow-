import tensorflow as tf
from numpy.random import RandomState

# 定义训练数据batch的大小
batch_size = 8

# 两个输入节点
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
# 回归问题一般只有一个输出节点
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

# 定义了一个单层的神经网络前向传播的过程，这里就是简单加权和
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 定义预测多了和预测少了的成本
loss_more = 1
loss_less = 10
loss = tf.reduce_sum(tf.select(tf.greater(y, y_), loss_more * (y - y_), loss_less * (y_ - y)))


# 学习率
learning_rate = 0.001
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataSet_size = 128
X = rdm.rand(dataSet_size, 2)
Y = [[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in X]

with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.initialize_all_variables())
    # 设定训练轮数
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataSet_size
        end = min(start + batch_size, dataSet_size)
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

    print(sess.run(w1))

