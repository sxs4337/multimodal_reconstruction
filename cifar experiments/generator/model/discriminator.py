from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.contrib import slim

# TODO Add MSRA initialization and bias initialization


def discriminator(image, features, is_training, dropout_prob, scope='discriminator'):
    """
    Definition of discriminator
    :param image: input image [batch_size, 227, 227, 3]
    :param features: features [batch_size, 1024]
    :param is_training: flag for training and testing
    :param dropout_prob: probability of dropout
    :param scope: scope of discriminator [default: discriminator]
    :return: logits associated with discriminator
    """
    with tf.variable_scope(scope, [features]):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            activation_fn=tf.nn.relu):
            net = slim.conv2d(image, 32, [7, 7], 4, scope='Dconv_1')
            net = slim.conv2d(net, 64, [5, 5], 1, scope='Dconv_2')
            net = slim.conv2d(net, 128, [3, 3], 2, scope='Dconv_3')
            net = slim.conv2d(net, 256, [3, 3], 1, scope='Dconv_4')
            net = slim.conv2d(net, 256, [3, 3], 2, scope='Dconv_5')
            net = slim.avg_pool2d(net, [11, 11], 11, scope='Dpool_5')
            net = tf.squeeze(net, [1, 2], name='Dpool_reshape5')
            sec_net = slim.fully_connected(features, 1024, scope='FC_1')
            sec_net = slim.fully_connected(sec_net, 512, scope='FC_2')
            net = tf.concat([net, sec_net], name="concat_5", axis=1)
            net = slim.dropout(net, keep_prob=dropout_prob,
                               is_training=is_training, scope='Dropout_5')
            net = slim.fully_connected(net, 512, scope='Dfc6')
            net = slim.dropout(net, keep_prob=dropout_prob,
                               is_training=is_training, scope='Dropout_6')
            net = slim.fully_connected(net, 1, scope='Dfc7', activation_fn=None)
    return net







