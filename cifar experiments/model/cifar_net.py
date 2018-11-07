"""Contains a variant of the CIFAR-10 model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(stddev=stddev)


def CifarNet(images, num_classes=10, is_training=False, prediction_fn=slim.softmax,
             scope='CifarNet'):
    """
    Creates a variant of the CifarNet model.
    Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, `num_classes`]
    end_points: a dictionary from components of the network to the corresponding
      activation.
    """
    end_points = {}

    with tf.variable_scope(scope, 'CifarNet', [images, num_classes]):   
        net = slim.conv2d(images, 64, [4, 4], stride=2, padding='SAME', scope='conv1')
        end_points['conv1'] = net
        # net = slim.batch_norm(net,is_training=is_training)
        net = slim.conv2d(net, 128, [4, 4], stride=2, padding='SAME', scope='conv2')
        end_points['conv2'] = net
        # net = slim.batch_norm(net,is_training=is_training)
        net = slim.conv2d(net, 256, [4, 4], stride=2, padding='SAME', scope='conv3')
        end_points['conv3'] = net
        # net = slim.batch_norm(net,is_training=is_training)
        net = slim.conv2d(net, 512, [4, 4], stride=2, padding='SAME', scope='conv4')
        end_points['conv4'] = net
        # net = slim.batch_norm(net,is_training=is_training)

        net = slim.flatten(net)
        end_points['Flatten'] = net
        net = slim.fully_connected(net, 100, scope='fc5', activation_fn=None)
        end_points['fc5'] = net
        logits = slim.fully_connected(net, num_classes,
                                      weights_regularizer=None,
                                      activation_fn=None,
                                      scope='logits')
    end_points['Logits'] = logits
    if prediction_fn is not None:
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

    return logits, end_points

CifarNet.default_image_size = 64


def cifarnet_arg_scope(weight_decay=0.004, use_batch_norm=True, 
                        batch_norm_decay=0.9,
                        batch_norm_epsilon=0.001):
  """
  Defines the default cifarnet argument scope.
  Args:
    weight_decay: The weight decay to use for regularizing the model.

  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      # collection containing update_ops.
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }
  with slim.arg_scope(
      [slim.conv2d],
      weights_initializer=tf.truncated_normal_initializer(stddev=5e-2),
      activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params):
    with slim.arg_scope(
        [slim.fully_connected],
        biases_initializer=tf.constant_initializer(0.1),
        weights_initializer=trunc_normal(0.04),
        weights_regularizer=slim.l2_regularizer(weight_decay),
        activation_fn=tf.nn.relu) as sc:
      return sc
