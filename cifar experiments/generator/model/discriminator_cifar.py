from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(stddev=stddev)


def leaky_relu(input_tensor, negative_slope=0.2, name='leaky_relu'):
    return tf.maximum(input_tensor, negative_slope * input_tensor, name=name)


def discriminator(images, is_training=False, scope='discriminator'):
    end_points = {}
    with tf.variable_scope(scope, 'discriminator', [images]):
        net = slim.conv2d(images, 64, [4, 4], stride=2, padding='SAME', scope='conv1')
        end_points['conv1'] = net
        net = slim.batch_norm(net,is_training=is_training)
        net = slim.conv2d(net, 128, [4, 4], stride=2, padding='SAME', scope='conv2')
        end_points['conv2'] = net
        net = slim.batch_norm(net,is_training=is_training)
        net = slim.conv2d(net, 256, [4, 4], stride=2, padding='SAME', scope='conv3')
        end_points['conv3'] = net
        net = slim.batch_norm(net,is_training=is_training)
        net = slim.conv2d(net, 512, [4, 4], stride=2, padding='SAME', scope='conv4')
        net = slim.batch_norm(net,is_training=is_training)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 100, scope="fc5")
        net = slim.fully_connected(net, 1, activation_fn=None, scope="logits")
    return net

def discriminator_scope(weight_decay=0.004):
    with slim.arg_scope([slim.conv2d], 
         weights_initializer=tf.truncated_normal_initializer(stddev=5e-2),
         activation_fn=leaky_relu):
        with slim.arg_scope([slim.fully_connected],
             biases_initializer=tf.constant_initializer(0.1),
             weights_initializer=trunc_normal(0.04),
             weights_regularizer=slim.l2_regularizer(weight_decay),
             activation_fn=tf.nn.relu) as sc:
             return sc

