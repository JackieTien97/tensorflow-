import tensorflow as tf


# 输入层的节点数。对于MNIST数据集，这个就等于图片的像素
INPUT_NODE = 784
# 输出层的节点数。这个等于类别的数目。因为在MNIST数据集中需要区分的是0～9这10个数字，所以这里输出层的节点数为10
OUTPUT_NODE = 10

# 配置神经网络的参数
# 隐藏层节点数。这里使用只有一个隐藏层的网络结构作为样例。这个隐藏层有500个节点
LAYER1_NODE = 500


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


# 定义神经网络前向传播的过程
def inference(input_tensor, regularizer):
    # 声明第一层神经网络的变量并完成前向传播的过程
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable('biases', [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    # 声明第一层神经网络的变量并完成前向传播的过程
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    # 返回前向传播的结果
    return layer2