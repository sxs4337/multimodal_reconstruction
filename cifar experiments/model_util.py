from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf


def get_vars(variables, substring, exclude_var=False):
    matched = []
    unmatched = []
    for var in variables:
        if substring in var.op.name:
            matched.append(var)
        else:
            unmatched.append(var)
    if exclude_var:
        return unmatched
    else:
        return matched


def var_mapping(original_vars):
    var_dict = {}
    for original_var in original_vars:
        var_dict[original_var.op.name.split("/", 2)[2]] = original_var
    return var_dict


def create_batches(loader, batch_size, image_size, word2vec_embedding_size,
                   is_train=True, is_valid=False):
    image, _, vector, label = loader.make_dataset(is_train=is_train, 
                                                  is_valid=is_valid)
    image = tf.image.resize_images(image, [image_size, image_size])
    vector = tf.reshape(vector, [word2vec_embedding_size])
    image, vector, label = \
        tf.train.batch([image, vector, label], batch_size)
    return image, vector, label


def create_placeholders(batch_size, image_size, word2vec_embedding_size, num_classes):
    image_placeholder = tf.placeholder(shape=[batch_size, image_size, image_size, 3],
                                       dtype=tf.float32)
    wordvec_placeholder = tf.placeholder(shape=[batch_size, word2vec_embedding_size],
                                         dtype=tf.float32)
    groundtruth_placeholder = tf.placeholder(shape=[batch_size, 1], dtype=tf.float32)
    matrix_placeholder = tf.placeholder(shape=[num_classes, word2vec_embedding_size],
                                        dtype=tf.float32)
    return image_placeholder, wordvec_placeholder, groundtruth_placeholder, \
           matrix_placeholder
