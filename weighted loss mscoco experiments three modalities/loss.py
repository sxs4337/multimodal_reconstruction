from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
import numpy as np


def _pairwise_distance_computation(input_tensor, margin):
    input_tensor = tf.expand_dims(input_tensor, 1)
    d_sq = tf.reduce_sum(tf.square(input_tensor - tf.transpose(input_tensor, (1, 0, 2))), \
                      2, keep_dims=False)
    d = tf.sqrt(d_sq + 1e-8)
    return tf.exp(margin-d), d

def smoothed_metric_loss(input_tensor, name='smoothed_triplet_loss', margin=1, 
                         positive_weight=1.0, negative_weight=0.01):
    '''
    From https://gist.github.com/chrischoy/c233b0d25c5fbe008d9477bc1b2d234c
    input_tensor: require a tensor with predefined dimensions (No None dimension)
                  Every two consecutive vectors must be a positive pair. There
                  should not be more than one pair from each class.
    '''
    with tf.variable_scope(name):
        # Song et al., Deep Metric Learning via Lifted Structured Feature Embedding
        # Define feature X \in \mathbb{R}^{N \times C}

        # Compute the pairwise distance
        exp_d, d = _pairwise_distance_computation(input_tensor, margin)
        # Compute the loss
        # Assume that the input data is aligned in a way that two consecutive data form a pair
        batch_size = input_tensor.get_shape()[0].value 
        J_all = []
        pos_dist_batch_all = []
        for current_index in range(0, batch_size, 2):
            image_index = current_index 
            caption_index = current_index + 1
            ind_rest = np.hstack([np.arange(0, current_index),
                                  np.arange(current_index+2, batch_size)])

            pos_dist = tf.gather_nd(d, [[image_index, caption_index]])
            image_inds = [[image_index, k] for k in ind_rest]
            caption_inds  = ([[caption_index, l] for l in ind_rest])
            J_ij = negative_weight * tf.log(tf.reduce_sum(tf.gather_nd(exp_d, image_inds)) + \
                    tf.reduce_sum(tf.gather_nd(exp_d, caption_inds)) + 1e-10) + \
                    positive_weight * pos_dist
            J_all.append(J_ij)
            pos_dist_batch_all.append(pos_dist)
        J_all = tf.convert_to_tensor(J_all)
        pos_dist_batch_all = tf.convert_to_tensor(pos_dist_batch_all)
        loss = tf.divide(tf.reduce_mean(tf.square(tf.maximum(J_all, 0))), 2.0, name='metric_loss')
        positive_distance_batch = tf.reduce_mean(pos_dist_batch_all)
        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
    return loss, positive_distance_batch

def metric_loss_image_caption_pair(input_tensor, name='lifted_triplet_loss', \
                                   margin=1.0, num_positive=2):
    with tf.variable_scope(name):
        exp_d, d = _pairwise_distance_computation(input_tensor, margin) 
        batch_size = input_tensor.get_shape()[0].value
        j_all = []
        # Index of all image and caption
        all_image_index = np.arange(0, batch_size, num_positive)
        all_caption_index = np.arange(1, batch_size, num_positive)
        for current_index in range(0, batch_size, num_positive):
            image_index = current_index
            caption_index = current_index+1#range(current_index+1, current_index+num_positive)
            image_indices_rest = np.delete(all_image_index, \
                                    np.where(all_image_index==image_index))
            caption_indices_rest = np.delete(all_caption_index, \
                                    np.where(all_caption_index==caption_index))
            negative_index = [[image_index, image_index_rest] \
                                   for image_index_rest in image_indices_rest]
            negative_caption_index = [[caption_index, caption_index_rest] \
                                   for caption_index_rest in caption_indices_rest]
            negative_index.extend(negative_caption_index)
            j_all.append(tf.log(tf.reduce_sum(tf.gather_nd(exp_d, \
                         negative_index))) + tf.gather_nd(d, \
                            [[image_index, caption_index]]))
        j_all = tf.convert_to_tensor(j_all)
        loss = tf.divide(tf.reduce_mean(tf.square(tf.maximum(j_all, 0))),\
                     2.0, name='metric_loss_sep_image_caption')
        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
    return loss

def metric_loss_all_captions(input_tensor, name='lifted_triplet_loss',\
                             margin=1.0, num_positive=6):
    with tf.variable_scope(name):
        exp_d, d = _pairwise_distance_computation(input_tensor, margin)
        batch_size = input_tensor.get_shape()[0].value
        j_all = []
        all_image_index = np.arange(0, batch_size, num_positive)
        all_caption_index = np.arange(batch_size)
        all_caption_index = all_caption_index[np.in1d(all_caption_index,
                                              all_image_index, invert=True)]
        for current_index in range(0, batch_size, num_positive):
            image_index = current_index
            current_caption_index = np.arange(image_index+1, image_index+num_positive-1)
            image_indices_rest = np.delete(all_image_index, \
                                       np.where(all_image_index==image_index))
            caption_indices_rest = all_caption_index[np.in1d(all_caption_index,
                                                     current_caption_index, invert=True)]

            negative_index = [[image_index, image_index_rest] \
                                for image_index_rest in image_indices_rest]
            negative_caption_index = [[caption_index, caption_index_rest] \
                                   for caption_index_rest in caption_indices_rest]
   
            negative_index.extend(negative_caption_index)
            j_ij.append(tf.log(tf.reduce_sum(tf.gather_nd(exp_d, \
                        negative_index))) + tf.gather_nd(d, \
                        [[image_index, caption_index]]))

            j_all.append(tf.square(tf.maximum(tf.convert_to_tensor(j_ij), 0)))

    loss = tf.divide(tf.reduce_mean(tf.convert_to_tensor(j_all),\
                     2.0, name='metric_loss_sep_image_caption'))
    tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
    return loss 
