
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import argparse
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import pickle as pkl
from new_coco_loader import CocoLoader
from tensorflow.python import pywrap_tensorflow
import sent2vec
from sent2vec import seq2seq_model
from sent2vec import data_utils
from generator_scratch import normalize, deprocess, save_image, GeneratorNetwork
from caffeNet_TF.caffe_net import preprocess, net
import pdb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
from skimage import io
from ConfigParser import SafeConfigParser


gConfig = {}

def get_config(config_file='sent2vec/seq2seq.ini'):
	parser = SafeConfigParser()
	parser.read(config_file)
	# get the ints, floats and strings
	_conf_ints = [ (key, int(value)) for key,value in parser.items('ints') ]
	_conf_floats = [ (key, float(value)) for key,value in parser.items('floats') ]
	_conf_strings = [ (key, str(value)) for key,value in parser.items('strings') ]
	# _conf_booleans = [ (key, bool(value)) for key,value in parser.items('booleans') ]
	_conf_booleans = [ (name, parser.getboolean('booleans', name))
						for name in parser.options('booleans') ]
	return dict(_conf_ints + _conf_floats + _conf_strings + _conf_booleans)

_buckets = [(5, 5), (10, 10), (20, 20)]
  
def create_model(forward_only,encoder_inputs, decoder_inputs,target_weights,targets):
  gConfig = get_config()
  """Create model and initialize or load parameters"""
  model = seq2seq_model.Seq2SeqModel( gConfig['enc_vocab_size'],
                                      gConfig['dec_vocab_size'],
                                      _buckets, gConfig['layer_size'],
                                      gConfig['num_layers'],
                                      gConfig['max_gradient_norm'],
                                      gConfig['batch_size'],
                                      gConfig['learning_rate'],
                                      gConfig['learning_rate_decay_factor'],encoder_inputs, decoder_inputs,target_weights,targets,
                                      forward_only=forward_only,
                                      use_pretrained_embedding=gConfig['pretrained_embedding'],
                                      pretrained_embedding_path=gConfig['pretrained_embedding_path'])

  return model

def embedding_transpose(embedding, name):
    with tf.variable_scope("embedding_transpose"):
        embedding = tf.log(embedding+1e-10) - tf.log(1-embedding+1e-10)
        embedding_weight = tf.contrib.framework.get_variables_by_name(name+"/weights")
        embedding_bias = tf.contrib.framework.get_variables_by_name(name+"/biases")
        embedding_inverse_tensor = tf.matmul(tf.subtract(embedding, embedding_bias), 
                                tf.transpose(tf.squeeze(embedding_weight)))
    return embedding_inverse_tensor
	
def embedding_inversion(gru_embedding, name, batch_size):
    with tf.variable_scope("embedding_inversion"):
        gru_embedding = tf.log(gru_embedding+1e-10) - tf.log(1-gru_embedding+1e-10)
        image_embedding_weight = tf.contrib.framework.get_variables_by_name(name+"/weights")
        image_embedding_bias = tf.contrib.framework.get_variables_by_name(name+"/biases")        
        embedding_inverse_tensor = tf.matrix_solve_ls(tf.transpose(tf.tile(image_embedding_weight,
                                                    [batch_size, 1, 1]), [0, 2, 1]), 
                                                    tf.expand_dims(tf.subtract(gru_embedding,image_embedding_bias), 2), 
                                                    fast=True, l2_regularizer=1e-5)    
    return tf.expand_dims(tf.squeeze(embedding_inverse_tensor),0)

def caption_validation(gru_embedding,name, batch_size):
	with tf.variable_scope("embedding_inversion"):
		embedding = tf.log(gru_embedding) - tf.log(1-gru_embedding)
		image_embedding_weight = tf.contrib.framework.get_variables_by_name(name+"/weights")
		image_embedding_bias = tf.contrib.framework.get_variables_by_name(name+"/biases")
		
		embedding_inverse_tensor = tf.matrix_solve_ls(tf.transpose(tf.tile(image_embedding_weight,
													[batch_size, 1, 1]), [0, 2, 1]), 
													tf.expand_dims(tf.subtract(embedding,image_embedding_bias), 2), fast=True)
	return tf.squeeze(embedding_inverse_tensor), image_embedding_weight
	


    
def scale_fc6(logits, max=104.952, min=0):
    return tf.divide(tf.subtract(logits, min), tf.subtract(max, min))
    
def rescale_fc6(logits, max=104.952, min=0):
    return max*logits + min
	
def generate_images(sess, image_id_name, embedding, fc6_net, input_image, generated_image, input_from_alexnet):

	# Image to Image Embedding generated images
	recon_image = sess.run(generated_image, feed_dict={input_from_alexnet:embedding})
	max_recon = np.max(recon_image)
	min_recon = np.min(recon_image)
	save_image(recon_image, cmd_opt.reconPath+'/'+str(image_id_name)+'.png',in_range=(min_recon,max_recon))
	
	# Fc6 generated images
	recon_fc6 = sess.run(generated_image, feed_dict={input_from_alexnet:fc6_net})
	save_image(recon_fc6, cmd_opt.fc6Path+'/'+str(image_id_name)+'.png',in_range=(-120,120))
	
	#Save the original image
	io.imsave(cmd_opt.origPath+'/'+str(image_id_name)+'.png', input_image)
	
	

def main(cmd_opt):
	gConfig = get_config(config_file=cmd_opt.configFile)
	batch_size=1
	# image = tf.gfile.GFile(cmd_opt.imagePath,'r').read()
	
	coco = CocoLoader() 
	image, image_id, caption_vec, caption, caption_ids  = coco._tfrecord_decoder(<tfrecordpath>)
	# caption_vec = tf.reshape(caption_vec,[1,900])
	# decoded_image = tf.image.decode_jpeg(image,channels=3)

	# Preprocess images as in caffe net
	preprocess_image = preprocess(image)
	fc6 = net(tf.expand_dims(preprocess_image,0))
	scaled_fc6 = scale_fc6(fc6)
		
	#Pad all captions to length 20 to form a batch
	caption_length = tf.shape(caption_ids)[0]
	fill_zeros = tf.fill(tf.constant(100,shape=[1])-caption_length,0)
	caption_pad = tf.concat([tf.cast(caption_ids,tf.int32),fill_zeros],axis=0)
	input_seq = tf.slice(caption_pad, [0], tf.constant(20,shape=[1]))
	input_seq_reverse = tf.reverse(input_seq,[0])
	
	caption_split = tf.split(input_seq_reverse, num_or_size_splits=20,axis=0)
		
	capt_dict={}
	with tf.variable_scope('sent2vec'):
		# TODO Verify
		model = create_model(True, caption_split, caption_split,caption_split,caption_split)
		model.encoder_inputs = caption_split
		sen_variables = tf.global_variables()
		for v in sen_variables:
			if v.name.startswith("sent2vec") and v.name.endswith(":0"):
				capt_dict[(v.name[9:-2])]=v
				
	# Final 900d encoder state 
	gru_fin_encoder = tf.concat(model.outputs[2],axis=1)

	gru_embedding = tf.contrib.layers.fully_connected(inputs=gru_fin_encoder,
												num_outputs=cmd_opt.embeddingSize,
												activation_fn=tf.nn.sigmoid,
												weights_initializer=tf.contrib.layers.xavier_initializer(),
												biases_initializer=tf.zeros_initializer,
												scope="gru_embedding")

	# Caffe net embeddings
	image_embeddings = tf.contrib.layers.fully_connected(inputs=scaled_fc6,
												num_outputs=cmd_opt.embeddingSize,
												activation_fn=tf.nn.sigmoid,
												weights_initializer=tf.contrib.layers.xavier_initializer(),
												biases_initializer=tf.zeros_initializer,
												scope="image_embedding")
	
	

	inv_caption, image_embedding_weight = caption_validation(gru_embedding, 'image_embedding', batch_size)	# 4096 dimensional
	inv_image ,rand= caption_validation(image_embeddings, 'gru_embedding',batch_size)    # 300 dimensional
	inv_image = tf.expand_dims(inv_image,0)
	
	# I-I validation via inversion
	image_emb_i = embedding_inversion(image_embeddings, "image_embedding", 1)
	
	# I-I validation via transpose
	image_emb_t = embedding_transpose(image_embeddings, "image_embedding")
	
	# C-C validation via transpose
	caption_emb_t = embedding_transpose(gru_embedding, "gru_embedding")
	
	# C-C validation via inversion
	caption_emb_i = embedding_inversion(gru_embedding, "gru_embedding", 1)
	
	# I-C validation via inversion
	i2c_emb_i = embedding_inversion(image_embeddings, "gru_embedding", 1)
	
	# I-C validation via transpose
	i2c_emb_t = embedding_transpose(image_embeddings, "gru_embedding")

	supervisor_saver = tf.train.Saver()

	# Define Generator after the sent2vec graph as it is not in the checkpoint variables
	with tf.variable_scope('generator'):
		generator = GeneratorNetwork(get_pretrained_weights=True)
		input_from_alexnet = tf.placeholder(tf.float32, [1, 4096])
		h_code, image_gen = generator.create_network(input_from_alexnet)
	
	supervisor = tf.train.Supervisor(saver=supervisor_saver, summary_op=None)
	config_proto = tf.ConfigProto()
	config_proto.gpu_options.allow_growth = True
	
	save_sentence_vectors=[]
	
	with supervisor.managed_session(config=config_proto) as sess:       
		supervisor_saver.restore(sess,cmd_opt.modelRoot)

		for count in range(20):
			sent_vector, transpose_sen_vector, inv_sen_vector, image_id_name, im_cap_tran, im_cap_inv, orig_caption  = sess.run([gru_fin_encoder, caption_emb_t,caption_emb_i, image_id, i2c_emb_t, 																							i2c_emb_i, caption])
			save_sentence_vectors.append((image_id_name, orig_caption, sent_vector))

			if cmd_opt.generate:
				emb_input = np.expand_dims(recon_image_embedding,0)
				fc6_input = orig_fc6
				generate_images(sess,image_id_name, caption_inv_emb, fc6_input, orig_img, image_gen, input_from_alexnet)
			
			print (count, 'Evaluated : ' , image_id_name)

		np.save('orig.npy',save_sentence_vectors)
		print('Saved all the sentence vectors')
			

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--configFile", required=True, help="Configuration for sequence to sequence")
	parser.add_argument('--imagePath', required=True, help="Test generation")
	parser.add_argument('--modelRoot', required=True, help="Pretrained model for seq to seq")
	parser.add_argument('--forwardOnly', default=True, dest='forwardOnly')
	parser.add_argument('--generate',  default=False, type=bool)
	parser.add_argument('--embeddingSize',default=4096, action='store_false')
	parser.add_argument('--origPath', required=True)
	parser.add_argument('--fc6Path',  required=True)
	parser.add_argument('--reconPath', required=True)
	# parser.add_argument('--transpose', required=True)

	parser.set_defaults(forwardBackward=True)
	cmd_opt = parser.parse_args()

	if cmd_opt.generate: 
		if not os.path.isdir(cmd_opt.origPath):
			os.makedirs(cmd_opt.origPath)
		if not os.path.isdir(cmd_opt.fc6Path):
			os.makedirs(cmd_opt.fc6Path)
		if not os.path.isdir(cmd_opt.reconPath):
			os.makedirs(cmd_opt.reconPath)

	main(cmd_opt)
