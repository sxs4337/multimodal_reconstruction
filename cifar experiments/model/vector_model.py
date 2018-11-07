from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import slim
from model.cifar_net import CifarNet, cifarnet_arg_scope


def image_branch(image, num_classes, embedding_size, is_train, 
                scope="image_branch"):
    with tf.variable_scope(scope):
        with slim.arg_scope(cifarnet_arg_scope()):
            _, end_points = CifarNet(image, num_classes, 
                                    is_training=is_train)
        feature = end_points['fc5']
        return tf.contrib.slim.fully_connected(feature, embedding_size,
                                               scope="image_embedding",
                                               activation_fn=tf.nn.sigmoid)


def word_branch(word_vec, embedding_size, scope="word_branch"):
    with tf.variable_scope(scope):
        return tf.contrib.slim.fully_connected(word_vec, embedding_size,
                                               scope="word_embedding",
                                               activation_fn=tf.nn.sigmoid)


