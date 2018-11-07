import tensorflow as tf
import pdb

import numpy as np
import os, sys
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import pickle as pkl
import scipy.misc


pretrained_caffe_weights = './generator/noiseless/generator.caffemodel'
pretrained_caffe_graph = './generator/noiseless/generator.prototxt'
input_Alexnet_vector_length = 4096

####Save image steal from ppgn
def normalize(img, out_range=(0.,1.), in_range=None):
    if not in_range:
        min_val = np.min(img)
        max_val = np.max(img)
    else:
        min_val = in_range[0]
        max_val = in_range[1]

    result = np.copy(img)
    result[result > max_val] = max_val
    result[result < min_val] = min_val
    result = (result - min_val) / (max_val - min_val) * (out_range[1] - out_range[0]) + out_range[0]
    return result

def deprocess(images, out_range=(0.,1.), in_range=None):
    num = images.shape[0]
    c = images.shape[1]
    ih = images.shape[2]
    iw = images.shape[3]

    result = np.zeros((ih, iw, 3))

    # Normalize before saving
    result[:] = images[0].copy().transpose((1,2,0))
    result = normalize(result, out_range, in_range)
    return result

def save_image(img, name, in_range):
    '''
    Normalize and save the image.
    '''
    img = img[:,::-1, :, :] # Convert from BGR to RGB
    output_img = deprocess(img, in_range)                
    scipy.misc.imsave(name, output_img)


def lrelu(l, neg_slope=0.3):
    return tf.maximum(l, neg_slope * l)

def make_fully_connected_layer(input, name_layer="", shape=[4096, 4096], activ_func = None, neg_slope=None, stddev=0.01, dtype = tf.float32, assign=False, weights_container=None, weight_or_bias=0 ):
    l = input
    with tf.variable_scope(name_layer, default_name='fc'):
        W = init_variable_or_assign('weights', shape,tf.truncated_normal_initializer(stddev=0.01), assign=assign, weights_container = weights_container, name_layer=name_layer, weight_or_bias=0, transpose=[1,0]) 
        b = init_variable_or_assign('bias', [shape[1]],tf.constant_initializer(0.1), assign=assign, weights_container = weights_container, name_layer=name_layer, weight_or_bias=1)
        l = tf.matmul(l, W) + b
        if activ_func:
            ##Should be changed later if memory usage is concerned
            l = lrelu(l, neg_slope)
    return l


def conv2d_transpose(l, W, b, strides = [1,2,2,1], padding='SAME', output_shape=None, name='deconv'): 
    l = tf.nn.conv2d_transpose(l, W, output_shape, strides = strides, padding=padding, data_format='NCHW', name=name)
    return tf.nn.bias_add(l, b, data_format='NCHW')

def init_variable_or_assign(name, shape, initializer, dtype=tf.float32, assign=False, weights_container=None, name_layer=None, weight_or_bias=0,transpose=None):
    var = tf.get_variable(name,shape, initializer=initializer, dtype=dtype)
    if assign:
		## weight_or_bias == 0: weight, 1: bias
        if transpose is not None:
            var = tf.assign(var, np.transpose(weights_container[name_layer][weight_or_bias], transpose)) 
        else:
            var = tf.assign(var, weights_container[name_layer][weight_or_bias]) 
    return var



def make_deconv(input, weight_shape=[4,4,256,256], bias_shape=[256], output_shape=[8,8,256,256], strides=[1, 2, 2, 1],padding='SAME', name_layer='deconv', assign=False, weights_container=None, neg_slope=None):
    with tf.variable_scope(name_layer):
        W_deconv5 = init_variable_or_assign('weights', weight_shape,tf.truncated_normal_initializer(stddev=0.01), assign=assign, weights_container = weights_container, name_layer=name_layer, weight_or_bias=0, transpose=[2,3,1,0])        
        b_deconv5 = init_variable_or_assign('bias', bias_shape,tf.constant_initializer(0.1), assign=assign, weights_container = weights_container, name_layer=name_layer, weight_or_bias=1)

        output_shape_deconv = tf.stack(output_shape)        
        l = conv2d_transpose(input, W_deconv5, b_deconv5, strides=strides, padding=padding, output_shape=output_shape_deconv)
        if neg_slope:
            l = lrelu(l, neg_slope=neg_slope)
    return l


class GeneratorNetwork(object):
    def __init__(self,get_pretrained_weights=None):
        self.debug = False
        self.l = None
        self.assign_weights = False
        self.params = None
        ##Get the weights from caffe
        if get_pretrained_weights:
            weight_file = '/shared/kgcoe-research/mil/caffe2tensorflow/generator_weights.pkl'
            if os.path.exists(weight_file):
                with open(weight_file, 'r') as f:
                    self.params = pkl.load(f)
            self.assign_weights = True
        
    def create_network(self, input):
        self.l = input
        self.l = make_fully_connected_layer(self.l, name_layer='defc7',shape=[4096,4096], activ_func="relu", 
                                            assign=self.assign_weights, weights_container=self.params, neg_slope=0.3)
        self.l = make_fully_connected_layer(self.l, name_layer='defc6',shape=[4096,4096], activ_func="relu", 
                                            assign=self.assign_weights, weights_container=self.params, neg_slope=0.3)
        self.l = make_fully_connected_layer(self.l, name_layer='defc5',shape=[4096,4096], activ_func="relu", 
                                            assign=self.assign_weights, weights_container=self.params, neg_slope=0.3)

        self.l = tf.reshape(self.l, [1, 256,4, 4])
        #Deconv5 stack
        self.l = make_deconv(self.l, weight_shape=[4,4,256,256], bias_shape=[256], output_shape=[1, 256, 8, 8], 
                                            strides=[1,1,2,2], name_layer='deconv5', assign=self.assign_weights, weights_container=self.params, 
                                            neg_slope=0.3)
        output_temp = self.l
        
        self.l = make_deconv(self.l, weight_shape=[3,3,512,256], bias_shape=[512], output_shape=[1, 512, 8, 8], strides=[1,1,1,1], 
                                                name_layer='conv5_1', assign=self.assign_weights, weights_container=self.params, neg_slope=0.3)

        #Deconv4 stack
        self.l = make_deconv(self.l, weight_shape=[4,4,256,512], bias_shape=[256], output_shape=[1, 256, 16, 16], strides=[1,1,2,2], name_layer='deconv4', assign=self.assign_weights, weights_container=self.params, neg_slope=0.3)

        self.l = make_deconv(self.l, weight_shape=[3,3,256,256], bias_shape=[256], output_shape=[1, 256, 16, 16], strides=[1,1,1,1], name_layer='conv4_1', assign=self.assign_weights, weights_container=self.params, neg_slope=0.3)

        #Deconv3 stack
        self.l = make_deconv(self.l, weight_shape=[4,4,128,256], bias_shape=[128], output_shape=[1, 128, 32, 32], strides=[1,1,2,2], name_layer='deconv3', assign=self.assign_weights, weights_container=self.params, neg_slope=0.3)

        self.l = make_deconv(self.l, weight_shape=[3,3,128,128], bias_shape=[128], output_shape=[1, 128, 32, 32], strides=[1,1,1,1], name_layer='conv3_1', assign=self.assign_weights, weights_container=self.params, neg_slope=0.3)

        #Deconv2 stack
        self.l = make_deconv(self.l, weight_shape=[4,4,64,128], bias_shape=[64], output_shape=[1, 64, 64, 64], strides=[1,1,2,2], name_layer='deconv2', assign=self.assign_weights, weights_container=self.params, neg_slope=0.3)
        
        #Deconv1 stack
        self.l = make_deconv(self.l, weight_shape=[4,4,32,64], bias_shape=[32], output_shape=[1, 32, 128, 128], strides=[1,1,2,2], name_layer='deconv1', assign=self.assign_weights, weights_container=self.params, neg_slope=0.3)

        #Deconv0 stack
        self.l = make_deconv(self.l, weight_shape=[4,4,3,32], bias_shape=[3], output_shape=[1, 3, 256, 256], strides=[1,1,2,2], name_layer='deconv0', assign=self.assign_weights, weights_container=self.params)
        return input, self.l
 
    def run(self, num_images_save=5, input_conditioned_vector=None):
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            session.run()
			# input_from_alexnet = tf.placeholder(tf.float32, [1, input_Alexnet_vector_length])
            output_temp , out_final= self.create_network(input_from_alexnet)
            pdb.set_trace()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess,'ppgn_generator.ckpt')
            if input_conditioned_vector is not None:
                pdb.set_trace()
                output_example, temp = sess.run([self.l, output_temp], feed_dict={input_from_alexnet:input_conditioned_vector})
                save_image(output_example,'test_ppgn.jpg')
                # saver.save(sess, 'ppgn_generator.ckpt')
 
