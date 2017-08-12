import time
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import MNIST.CNN.mnist_inference as mnist_inference
import MNIST.CNN.mnist_train as mnist_train

# 每10秒加载一次最新的模型，并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        # 定义输入输出的格式
        x = tf.placeholder(tf.float32, [None,
                                        mnist_inference.IMAGE_SIZE,  # 第二维和第三维表示图片的尺寸大小
                                        mnist_inference.IMAGE_SIZE,
                                        mnist_inference.NUM_CHANNELS], name="x-input")
        y_ = tf.placeholder(tf.float32, [None,
                                         mnist_inference.IMAGE_SIZE,  # 第二维和第三维表示图片的尺寸大小
                                         mnist_inference.IMAGE_SIZE,
                                         mnist_inference.NUM_CHANNELS], name="y-input")

        # 准备验证数据。一般在神经网络的训练过程中会通过验证数据来大致判断停止的条件和评判训练的效果
        validate_feed = {x: np.reshape(mnist.validation.images, [5000,
                                                                 mnist_inference.IMAGE_SIZE,
                                                                 mnist_inference.IMAGE_SIZE,
                                                                 mnist_inference.NUM_CHANNELS]),
                         y_: mnist.validation.labels}

        # 测试时不关注正则化损失的值，所以这里被用作计算正则化损失的函数被设为None
        y = mnist_inference.inference(x, False, None)

        # 使用前向传播的结果计算正确率。如果需要对未知的样例进行分类，那么使用tf.arg_max(y, 1)就可以得到输入样例的预测类别了
        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求滑动平均的函数来获取平均值了
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                # tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名得到模型保存时迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training steps, validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
