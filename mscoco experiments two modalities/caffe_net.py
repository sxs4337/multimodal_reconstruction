
from numpy import *
import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import Image
import pdb

import tensorflow as tf

from caffe_classes import class_names



def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
	'''From https://github.com/ethereon/caffe-tensorflow
	'''
	c_i = input.get_shape()[-1]
	assert c_i%group==0
	assert c_o%group==0
	convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)


	if group==1:
		conv = convolve(input, kernel)
	else:
		# pdb.set_trace()
		input_groups = tf.split(input, group, 3)#group means we split the input  into 'group' groups along the third demention
		kernel_groups = tf.split(kernel, group, 3)
		output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
		conv = tf.concat(output_groups,3)
	# pdb.set_trace()
	return  tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())

def preprocess(image):
	
	img = tf.reverse(image, [-1])#from rgb to bgr

	##########################################################
	new_shape = np.array([256, 256])#first it be scale to 256 256 
	img = tf.image.resize_images(img, [new_shape[0], new_shape[1]])
	##########################################################

	# 
	# Use the slice workaround until crop_to_bounding_box supports deferred tensor shapes
	# See: https://github.com/tensorflow/tensorflow/issues/521
	##########################################################Center crop
	offset = (new_shape - 227) / 2##croppppppppppp
	img = tf.slice(img, [offset[0], offset[1], 0], [227, 227, -1])

	##########################################################
	mean=np.array([104., 117., 124.])#substact the mean
	img=tf.to_float(img) - mean
	return img

def net(image):
				
	net_data = load("caffenet/weights.npy").item()
	#conv1
	#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
	k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
	conv1W = tf.Variable(net_data["conv1"]['weights'],trainable=False)
	conv1b = tf.Variable(net_data["conv1"]['biases'],trainable=False)
	conv1_in = conv(image, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1)
	conv1 = tf.nn.relu(conv1_in)
	# pdb.set_trace()
	#maxpool1
	#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
	k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
	maxpool1 = tf.nn.max_pool(conv1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


	#lrn1
	#lrn(2, 2e-05, 0.75, name='norm1')
	radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
	lrn1 = tf.nn.local_response_normalization(maxpool1,
													  depth_radius=radius,
													  alpha=alpha,
													  beta=beta,
													  bias=bias)

	#conv2
	#conv(5, 5, 256, 1, 1, group=2, name='conv2')
	k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group =2
	conv2W = tf.Variable(net_data["conv2"]['weights'],trainable=False)
	conv2b = tf.Variable(net_data["conv2"]['biases'],trainable=False)
	conv2_in = conv(lrn1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
	conv2 = tf.nn.relu(conv2_in)
	#maxpool2
	#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
	k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
	maxpool2 = tf.nn.max_pool(conv2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


	#lrn2
	#lrn(2, 2e-05, 0.75, name='norm2')
	radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
	lrn2 = tf.nn.local_response_normalization(maxpool2 ,
													  depth_radius=radius,
													  alpha=alpha,
													  beta=beta,
													  bias=bias)


	#conv3
	#conv(3, 3, 384, 1, 1, name='conv3')
	k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
	conv3W = tf.Variable(net_data["conv3"]['weights'],trainable=False)
	conv3b = tf.Variable(net_data["conv3"]['biases'],trainable=False)
	conv3_in = conv(lrn2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
	conv3 = tf.nn.relu(conv3_in)

	#conv4
	#conv(3, 3, 384, 1, 1, group=2, name='conv4')
	k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
	conv4W = tf.Variable(net_data["conv4"]['weights'],trainable=False)
	conv4b = tf.Variable(net_data["conv4"]['biases'],trainable=False)
	conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
	conv4 = tf.nn.relu(conv4_in)


	#conv5
	#conv(3, 3, 256, 1, 1, group=2, name='conv5')
	k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
	conv5W = tf.Variable(net_data["conv5"]['weights'],trainable=False)
	conv5b = tf.Variable(net_data["conv5"]['biases'],trainable=False)
	conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
	conv5 = tf.nn.relu(conv5_in)

	#maxpool5
	#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
	k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
	maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

	#fc6
	#fc(4096, name='fc6')
	fc6W = tf.Variable(net_data["fc6"]['weights'],trainable=False)
	fc6b = tf.Variable(net_data["fc6"]['biases'],trainable=False)
	fc6 = tf.nn.relu_layer(tf.contrib.layers.flatten(maxpool5) , fc6W, fc6b)
	
	#fc7
	#fc(4096, name='fc7')
	fc7W = tf.Variable(net_data["fc7"]['weights'],trainable=False)
	fc7b = tf.Variable(net_data["fc7"]['biases'],trainable=False)
	fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

	#fc8
	#fc(1000, relu=False, name='fc8')
	fc8W = tf.Variable(net_data["fc8"]['weights'],trainable=False)
	fc8b = tf.Variable(net_data["fc8"]['biases'],trainable=False)
	fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

	#prob
	#softmax(name='prob'))
	prob = tf.nn.softmax(fc8)

	return fc6 #conv1W,conv1b

