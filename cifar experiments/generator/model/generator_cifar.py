from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def leaky_relu(input_tensor, negative_slope, name='leaky_relu'):
    return tf.maximum(input_tensor, negative_slope * input_tensor, name=name)


def generator(features, image_size=64, is_training=False, scope='generator'):
    """
    Creates a variant of the CifarNet model.
    """
    end_points = {}
    with slim.arg_scope([slim.conv2d_transpose], padding="SAME", activation_fn=None):
        with tf.variable_scope(scope, 'generator', [features, image_size, is_training]):
            net = slim.fully_connected(features, 8192, scope="fc7")
            net = tf.reshape(net, [-1, 4, 4, 512])
            net = slim.conv2d_transpose(net, num_outputs=256, kernel_size=[4, 4],
                                      stride=2, scope="deconv_1")
            net = slim.batch_norm(net, is_training=is_training)
            net = leaky_relu(net, 0.2, "deconv_relu_1")
            net = slim.conv2d_transpose(net, num_outputs=128, kernel_size=[4, 4],
                                      stride=2, scope="deconv_2")
            net = slim.batch_norm(net, is_training=is_training)
            net = leaky_relu(net, 0.2, "deconv_relu_2")
            net = slim.conv2d_transpose(net, num_outputs=64, kernel_size=[4, 4],
                                      stride=2, scope="deconv_3")
            net = slim.batch_norm(net, is_training=is_training)
            net = leaky_relu(net, 0.2, "deconv_relu_3")
            net = slim.conv2d_transpose(net, num_outputs=3, kernel_size=[4, 4],
                                      stride=2, scope="deconv_4")
            net = tf.nn.tanh(net)
    return net   









