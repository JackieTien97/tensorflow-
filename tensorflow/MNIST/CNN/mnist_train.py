import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import MNIST.CNN.mnist_inference as mnist_inference


# 配置神经网络中的参数
# 一个训练batch中的训练数据个数。数字越小，训练过程越接近随机梯度下降，数字越大，训练越接近梯度下降
BATCH_SIZE = 100
# 基础的学习率
LEARNING_RATE_BASE = 0.8
# 学习率的衰减率
LEARNING_RATE_DECAY = 0.99
# 描述模型复杂度的正则化项在损失函数中的系数
REGULARIZATION_RATE = 0.0001
# 训练轮数
TRAINING_STEPS = 30000
# 滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99

# 模型保存的路径和文件名
MODEL_SAVE_PATH = 'model/'
MODEL_NAME = 'model.ckpt'


# 训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32, [BATCH_SIZE,                     # 第一维表示一个batch中样例的个数
                                    mnist_inference.IMAGE_SIZE,     # 第二维和第三维表示图片的尺寸大小
                                    mnist_inference.IMAGE_SIZE,
                                    mnist_inference.NUM_CHANNELS],  # 第四维表示图片的深度
                       name="x-input")
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, mnist_inference.NUM_LABELS], name="y-input")

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, True, regularizer)

    # 定义存储训练轮数的变量。这个变量不需要计算滑动平均值，所以这里指定这个变量为不可训练变量。
    # 在使用Tensorflow训练神经网络时，一般会将代表训练轮数的变量指定为不可训练的参数
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均，其他辅助变量就不需要了(如：global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算交叉熵z作为刻画预测值和真实值之间差距的损失函数。
    # 当分类问题中只有一个正确答案时，可以使用这个函数来加速交叉熵的计算
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.arg_max(y_, 1), logits=y)
    # 计算当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,  # 基础的学习率，随着迭代的进行，更新变量时使用的学习率在这个基础上递减
                                               global_step,  # 当前迭代的轮数
                                               mnist.train.num_examples / BATCH_SIZE,  # 过完所有的训练数据需要的迭代次数
                                               LEARNING_RATE_DECAY)  # 学习率衰减速度

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 在训练神经网络模型时，每过一遍数据既需要通过反向传播更新神经网络中的参数，又需要更新每个参数的滑动平均值。
    # 为了一次完成多个操作，Tensorflow提供了tf.control_dependencies和tf.group两种机制，下面两行程序和
    # train_op = tf.group(train_step, variables_averages_op)是等价的
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name="train")

    # 初始化TensorFlow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 在训练过程中不再测试模型在验证数据上的表现，验证和测试的过程将会有一个独立的程序来实现
        for i in range(TRAINING_STEPS):
            print(i)
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, [BATCH_SIZE,
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.NUM_CHANNELS])
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

            # 每1000轮保存一次模型
            if i % 1000 == 0:
                # 输出当前的训练情况。这里只输出了模型在当前训练batch上的损失函数大小
                print("After %d training steps, loss on training batch is %g" % (step, loss_value))
                # 保存当前的模型。注意这里给出了global_step参数，这样可以让每个被保存模型的文件名末尾加上训练的轮数，
                # 比如"model.ckpt-1000"表示训练1000轮之后得到的額模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()



