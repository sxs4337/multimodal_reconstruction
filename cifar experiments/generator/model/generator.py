from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.contrib import slim


def leaky_relu(input, negative_slope, name='leaky_relu'):
    """
    leaky relu operation
    :param input: Input of the operation
    :param negative_slope: alpha value
    :param name: name of the layer [default: leaky_relu]
    :return: output of relu
    """
    return tf.maximum(input, negative_slope * input, name=name)

def _old_alexnet_preprocessing(net, scope='alexnet_preprocessing'):
    #net =  tf.div(tf.subtract(net, tf.reduce_min(net)),
    #                        tf.subtract(tf.reduce_max(net), tf.reduce_min(net)))
    with tf.variable_scope(scope):
        net = tf.add(net, 1.0)
        net = tf.multiply(net, 0.5)
        net = tf.multiply(net, 255)    
        channels = tf.split(axis=3, num_or_size_splits=3, value=net)
        means = [123.151, 115.902, 103.062]
        for i in range(3):
            channels[i] -= means[i]
        image  = tf.concat(values=channels, axis=3)
    return image


def generator(features, image_size=227, is_training=False, scope='generator'):
    """
    Definition of Generator
    :param features: Features from encoder 
    :param image_size: Size of the image generated
    :param is_training: Flag to determine training/testing
    :param scope: scope of generator [default: generator]
    :return: Image [image_size, image_size, 3]
    """
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        padding='SAME',
                        activation_fn=None,
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        with tf.variable_scope(scope, [features]):
            net = slim.fully_connected(features, 4096, scope='defc7')
            net = leaky_relu(net, 0.3, name='relu_defc7')
            net = slim.fully_connected(net, 4096, scope='defc6')
            net = leaky_relu(net, 0.3, name='relu_defc6')
            net = slim.fully_connected(net, 4096,  scope='defc5')
            net = leaky_relu(net, 0.3, name='relu_defc5')
            net = tf.reshape(net, [-1, 4, 4, 256], name='reshape')
            # Deconv 5
            net = slim.conv2d_transpose(net, num_outputs=256, kernel_size=[4, 4],
                                         stride=2, scope='deconv5')
       
            net = leaky_relu(net, 0.3, name='relu_deconv5')
            net = slim.conv2d(net, 512, [3, 3], 1, scope='conv5')
            net = leaky_relu(net, 0.3, name='relu_conv5')
            # Deconv 4
            net = slim.conv2d_transpose(net, 256, kernel_size=[4, 4],
                                        stride=2, scope='deconv4')
            net = leaky_relu(net, 0.3, name='relu_deconv4')
            net = slim.conv2d(net, 256, [3, 3], 1, scope='conv4')
            net = leaky_relu(net, 0.3, name='relu_conv4')
            # Deconv 3
            net = slim.conv2d_transpose(net, 128, kernel_size=[4, 4],
                                        stride=2, scope='deconv3')
            net = leaky_relu(net, 0.3, name='relu_deconv3')
            net = slim.conv2d(net, 128, [3, 3], 1, scope='conv3')
            net = leaky_relu(net, 0.3, name='relu_conv3')
            # Deconv 2
            net = slim.conv2d_transpose(net, 64, kernel_size=[4, 4],
                                        stride=2, scope='deconv2')
            net = leaky_relu(net, 0.3, name='relu_deconv2')
            # Deconv 1
            net = slim.conv2d_transpose(net, 32, kernel_size=[4, 4],
                                        stride=2, scope='deconv1')
            net = leaky_relu(net, 0.3, name='relu_deconv1')
            net = slim.conv2d_transpose(net, 3, kernel_size=[4, 4],
                                        stride=2, scope='deconv0',
                                        activation_fn=tf.nn.tanh)
            net = tf.image.resize_images(net, [image_size, image_size]) 
            net = _old_alexnet_preprocessing(net)
    return net



