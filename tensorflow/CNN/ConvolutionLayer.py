import tensorflow as tf


# 通过get_variable方式创建过滤器的权重变量和偏置项变量。
# 卷积层的参数个数只和过滤器的尺寸、深度以及当前层节点矩阵的深度有关，所以这里声明的参数变量是一个四维矩阵。
# 前面两个维度代表了过滤器的尺寸，第三个维度表示当前层的深度，第四个维度表示过滤器的深度
filter_weight = tf.get_variable("weights", [5, 5, 3, 16], initializer=tf.truncated_normal_initializer(stddev=0.1))

# 和卷积层的权重类似，当前层矩阵上不同位置的偏置项也是共享的，所以总共有下一层深度个不同的偏置项
biases = tf.get_variable("biases", [16], initializer=tf.constant_initializer(0.1))

# tf.nn.conv2d提供了一个非常方便的函数来实现卷积层前向传播的算法
# 这个函数的第一个参数为当前层的节点矩阵。注意这个矩阵是一个四维矩阵，后面三个维度对应一个节点矩阵，一个维度对应一个输入batch
# 第二个参数提供了卷积层的权重，第三个参数为不同维度上的步长
# 虽然第三个参数提供的是一个长度为4的数组，但是第一维和最后一维的数字一定要求是1。这是因为卷积层的步长只对矩阵的长和宽有效
# 最后一个参数是填充的方法，TensorFlow中提供SAME或者VALID两种选择。其中，SAME表示添加全0填充，VALID表示不添加
conv = tf.nn.conv2d(input, filter_weight, strides=[1, 1, 1, 1], padding="SAME")

# tf.nn.bias_add提供了一个方便的函数给每一个节点项加上偏置项。
bias = tf.nn.bias_add(conv, biases)

# 将计算结果通过ReLU激活函数去线性化
actived_conv = tf.nn.relu(bias)