
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
from coco_loader import CocoLoader
from tensorflow.python import pywrap_tensorflow
import sent2vec_thang
from sent2vec_thang import seq2seq_model_eval
from sent2vec_thang import data_utils
from generator_scratch import normalize, deprocess, save_image, GeneratorNetwork
from caffeNet_TF.caffe_net import preprocess, net
import pdb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
import pandas as pd
try:
    from ConfigParser import SafeConfigParser
except:
    from configparser import SafeConfigParser # In Python 3, ConfigParser has been renamed to configparser for PEP 8 compliance.

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

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

def create_model(forward_only):
  gConfig = get_config()
  """Create model and initialize or load parameters"""
  model = seq2seq_model_eval.Seq2SeqModel( gConfig['enc_vocab_size'],
											gConfig['dec_vocab_size'],
											_buckets, gConfig['layer_size'],
											gConfig['num_layers'],
											gConfig['max_gradient_norm'],
											gConfig['batch_size'],
											gConfig['learning_rate'],
											gConfig['learning_rate_decay_factor'],
											forward_only=forward_only,
											use_pretrained_embedding=gConfig['pretrained_embedding'],
											pretrained_embedding_path=gConfig['pretrained_embedding_path'])
  return model
 
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
    return tf.multiply(logits, tf.subtract(max, min)) + max

def main(cmd_opt):
	gConfig = get_config(config_file=cmd_opt.configFile)
	batch_size=1
	
	# Define the sent2vec model
	with tf.variable_scope('sent2vec'):
		model = create_model(cmd_opt.forwardBackward)
		vec2sent = (tf.placeholder(tf.float32, shape=[batch_size,300]), tf.placeholder(tf.float32, shape=[batch_size,300]), tf.placeholder(tf.float32, shape=[batch_size,300]))
		model.decode(vec2sent)
		model.batch_size=batch_size		
	
	enc_vocab_path = os.path.join(gConfig['working_directory'],"vocab%d.enc" % gConfig['enc_vocab_size'])
	enc_vocab, _ = data_utils.initialize_vocabulary(enc_vocab_path)
	index_to_word = {}
	for k,v in enc_vocab.iteritems():
		index_to_word[v] = k
	index_to_word = pd.Series(index_to_word)

	with open(cmd_opt.infile,'r') as f:
		sample_vector = np.load(f)
	i=1
	with open(cmd_opt.outfile+'.txt','w') as file:
		for sample in sample_vector:
		
			token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(' '.join(sample[1])), enc_vocab)
			bucket_id = min([b for b in xrange(len(_buckets))
						   if _buckets[b][0] > len(token_ids)])
			encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)
			sent2vec , input_feed= model.step(encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
			
			noisy_sample = sample[2] + np.random.normal(0, 0.01, [1,900])

			encoder_state=np.split(noisy_sample,3,axis=1)
			input_feed[vec2sent] = tuple(encoder_state)

			var_dict={}
			for var in tf.global_variables():
				new_var_name = '/'.join(var.op.name.split('/')[1:])
				var_dict[new_var_name] = var
			
			saver = tf.train.Saver(var_list=var_dict)        
			with tf.Session() as sess: 
				sess.run(tf.global_variables_initializer())
				saver.restore(sess,cmd_opt.modelRoot)
				# pdb.set_trace()
				decoded, test_sent2vec, project = sess.run([sent2vec, model.outputs, model.project], input_feed)
				output_index = np.argmax(np.squeeze(np.array(test_sent2vec)), axis=1)
				
				# Remove end of sentence tokens
				output_index = np.ndarray.tolist(output_index)
				if 2 in output_index: 
					output_index = output_index[:output_index.index(2)]  # find index of EOS
					output_words_list = index_to_word[output_index].values
				else : 
					output_words_list = index_to_word[output_index].values
				
				file.write(' '.join(sample[1])+ '    '+ ' '.join(output_words_list)+'\n')
				
				print('Written : ', i)
				i+=1
				if i==20: break  # Just for viewing 20 sentences
				
	file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--configFile", required=True, help="Configuration for sequence to sequence")
    parser.add_argument('--modelRoot', required=True, help="Pretrained model for seq to seq")
    parser.add_argument('--infile', required=True, help="Input vectors filename")
    parser.add_argument('--outfile', required=True, help="output file text")
    parser.add_argument('--forwardBackward', dest='forwardBackward', action='store_false')
    
    parser.set_defaults(forwardBackward=True)
    cmd_opt = parser.parse_args()
    main(cmd_opt)