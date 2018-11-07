from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import pdb
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import pickle as pkl
import argparse

from new_coco_loader import CocoLoader
from sent2vec import data_utils
from sent2vec import seq2seq_model
from caffeNet_TF.caffe_net import preprocess, net
from model_utils import scale_fc6
from tensorflow.python import debug as tf_debug

try:
    from ConfigParser import SafeConfigParser
except:
    from configparser import SafeConfigParser # In Python 3, ConfigParser has been renamed to configparser for PEP 8 compliance.

gConfig = {}

# TODO dynamic batching

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

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
# _buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
# _buckets = [(5, 5), (10, 10), (20, 20), (40, 40), (60, 60)]
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

def _euc_distance(logit_one, logit_two):
    return tf.sqrt(tf.reduce_sum(tf.square(
            tf.subtract(logit_one, logit_two)), 1, keep_dims=True))

def contrastive_loss(word_logits, image_logits, groundtruth_label,
                     margin=50., scope="contrastive_loss"):
    with tf.variable_scope(scope):
        distance = _euc_distance(word_logits, image_logits)
        loss = tf.reduce_mean(groundtruth_label * tf.square(distance) +
                              (1-groundtruth_label)*tf.square(tf.maximum(0.,
                                margin-distance)))*0.5
        return loss
        
def embedding_inversion(gru_embedding, name, batch_size):
    with tf.variable_scope("embedding_inversion"):
        gru_embedding = tf.log(gru_embedding+1e-10) - tf.log(1-gru_embedding+1e-10)
        image_embedding_weight = tf.contrib.framework.get_variables_by_name(name+"/weights")
        image_embedding_bias = tf.contrib.framework.get_variables_by_name(name+"/biases")        
        embedding_inverse_tensor = tf.matrix_solve_ls(tf.transpose(tf.tile(image_embedding_weight,
                                                    [batch_size, 1, 1]), [0, 2, 1]), 
                                                    tf.expand_dims(tf.subtract(gru_embedding,image_embedding_bias), 2), 
                                                    fast=True, l2_regularizer=1e-5)    
    return tf.squeeze(embedding_inverse_tensor), image_embedding_weight
    
def embedding_transpose(embedding, name, batch_size):
    with tf.variable_scope("embedding_transpose"):
        embedding = tf.log(embedding+1e-10) - tf.log(1-embedding+1e-10)
        embedding_weight = tf.contrib.framework.get_variables_by_name(name+"/weights")
        embedding_bias = tf.contrib.framework.get_variables_by_name(name+"/biases")
        embedding_inverse_tensor = tf.matmul(tf.subtract(embedding, embedding_bias), 
                                tf.transpose(tf.squeeze(embedding_weight)))
    return embedding_inverse_tensor
    
    
def reconstruction_loss(logits, embedding_logits, logit_scope, batch_size, is_transpose=False):
	if is_transpose:
		inverted_embedding = embedding_transpose(embedding_logits, logit_scope, batch_size)
	else:
		inverted_embedding, _ = embedding_inversion(embedding_logits, logit_scope, batch_size)
	embedding_distance = _euc_distance(logits, inverted_embedding)    
	return tf.reduce_mean(embedding_distance)
	
def permutate(positive_train_image, positive_train_vector):
	negative_train_image = np.copy(positive_train_image)
	negative_train_vector = np.random.permutation(positive_train_vector)
	batch_train_image = np.vstack([positive_train_image, negative_train_image])
	batch_train_vector = np.vstack([positive_train_vector, negative_train_vector])
	groundtruth_label = np.vstack([np.ones([positive_train_image.shape[0], 1]), np.zeros([negative_train_image.shape[0], 1])])
	shuffled_index = np.random.permutation(np.arange(0,batch_train_image.shape[0]))
	batch_train_image[shuffled_index] = batch_train_image
	batch_train_vector[shuffled_index] = batch_train_vector
	groundtruth_label[shuffled_index] = groundtruth_label
	
	return batch_train_image, batch_train_vector, groundtruth_label

def main(cmd_opt):
	filenames=[]
	for file in os.listdir(cmd_opt.dataDir):
		if file.startswith('train'):
			filenames.append(os.path.join(cmd_opt.dataDir, file))

	coco = CocoLoader()
	train_image_tensor, image_id, _, caption, caption_ids   = coco._tfrecord_decoder(filenames)
	
	# supervisor = tf.train.Supervisor()
	# config_proto = tf.ConfigProto()
	# config_proto.gpu_options.allow_growth = True

	# with supervisor.managed_session(config=config_proto) as sess:
		# image, image_id, caption_vec, caption, caption_ids=sess.run([train_image_tensor, image_id, _, caption, caption_ids])
		# pdb.set_trace()
		# print('Hello')
	
	# Preprocess images as in caffe net
	preprocess_image = preprocess(train_image_tensor)

	#Pad all captions to length 20 to form a batch
	caption_length = tf.shape(caption_ids)[0]
	fill_zeros = tf.fill(tf.constant(100,shape=[1])-caption_length,0)
	caption_pad = tf.concat([tf.cast(caption_ids,tf.int32),fill_zeros],axis=0)
	input_seq = tf.slice(caption_pad, [0], tf.constant(20,shape=[1]))
	input_seq_reverse = tf.reverse(input_seq,[0])

	# Maintain step value
	global_step_tensor = tf.Variable(0, trainable=False, name="global_step")

	#Generate a batch of tensors
	image_batch_t, reverse_captions_t = tf.train.batch([preprocess_image, \
												input_seq_reverse], cmd_opt.batchSize, \
												dynamic_pad=True)

	image_placeholder = tf.placeholder(dtype=tf.float32, shape=[2*cmd_opt.batchSize, cmd_opt.imageSize, \
											cmd_opt.imageSize, 3],name="image_placeholder")
	caption_id_placeholder = tf.placeholder(dtype=tf.int32, shape=[2*cmd_opt.batchSize, 20],name="caption_placeholder")
	label_placeholder = tf.placeholder(dtype=tf.int64, shape=[2*cmd_opt.batchSize,1],name="label_placeholder")


	# Get fc6 caffe net features
	fc6 = net(image_placeholder)
	scaled_fc6 =  scale_fc6(fc6,max=65.0)

	caption_split = tf.split(caption_id_placeholder, num_or_size_splits=20,axis=1)

	# Sent2vec model
	capt_dict={}
	with tf.variable_scope('sent2vec'):
		# TODO Verify
		model = create_model(True, caption_split, caption_split,caption_split,caption_split)
		model.batch_size = cmd_opt.batchSize
		model.encoder_inputs = caption_split
		sen_variables = tf.global_variables()
		for v in sen_variables:
			if v.name.startswith("sent2vec") and v.name.endswith(":0"):
				capt_dict[(v.name[9:-2])]=v

	# Final encoder state 
	gru_fin_encoder = tf.concat(model.outputs[2],axis=1)

	# Sent2vec embedding
	gru_embedding = tf.contrib.layers.fully_connected(inputs=gru_fin_encoder,
													num_outputs=cmd_opt.embeddingSize,
													activation_fn=tf.nn.sigmoid,
													weights_initializer=tf.contrib.layers.xavier_initializer(),
													biases_initializer=tf.zeros_initializer,
													scope="gru_embedding")

	# Caffe net embeddings
	image_embedding = tf.contrib.layers.fully_connected(inputs=scaled_fc6,
													num_outputs=cmd_opt.embeddingSize,
													activation_fn=tf.nn.sigmoid,
													weights_initializer=tf.contrib.layers.xavier_initializer(),
													biases_initializer=tf.zeros_initializer,
													scope="image_embedding")
	
	all_vars = tf.trainable_variables()
	# TODO verify if this is correct
	var_list=[]
	for var in all_vars:
		if ("image_embedding" in var.op.name) or ("gru_embedding" in var.op.name):
			var_list.append(var)

	# Add reconstruction loss
	recon_loss_im_t = reconstruction_loss(scaled_fc6, image_embedding, 'image_embedding', 2*cmd_opt.batchSize, 
											is_transpose=cmd_opt.transpose)
	recon_loss_caption_t = reconstruction_loss(gru_fin_encoder, gru_embedding,'gru_embedding', 2*cmd_opt.batchSize,
											is_transpose=cmd_opt.transpose)
	contrastive_loss_t = contrastive_loss(image_embedding, gru_embedding,tf.cast(label_placeholder, tf.float32), 
											cmd_opt.margin)
	image_embedding_inversion_t, _ = embedding_inversion(image_embedding, 'image_embedding', 2*cmd_opt.batchSize)
	word_embedding_inversion_t, _ = embedding_inversion(gru_embedding, 'gru_embedding', 2*cmd_opt.batchSize)
		
	# Define Contrastive Loss
	loss = cmd_opt.contrastiveScale * contrastive_loss_t + cmd_opt.imageReconScale * recon_loss_im_t + \
				cmd_opt.captionReconScale * recon_loss_caption_t
	tf.summary.scalar('Total Loss',loss)
	tf.summary.scalar('Constrative Loss', contrastive_loss_t)
	tf.summary.scalar('Recon loss Caption', recon_loss_caption_t)
	tf.summary.scalar('Recon loss Image', recon_loss_im_t)

	# Define Optimizer
	optimizer = tf.train.AdamOptimizer(0.001)
	train_op = optimizer.minimize(loss, global_step=global_step_tensor,var_list=var_list)

	saver = tf.train.Saver(capt_dict)
	def init_fn(sess):
		saver.restore(sess, cmd_opt.pretrainedModelRoot)
	
	summary_op = tf.summary.merge_all()
	filewriter = tf.summary.FileWriter(os.path.join(cmd_opt.expDir, 'train'))
	supervisor_saver = tf.train.Saver()    
	supervisor = tf.train.Supervisor(logdir=cmd_opt.expDir, global_step=global_step_tensor, saver=supervisor_saver, 
									init_fn=init_fn, summary_op=None)
	
	config_proto = tf.ConfigProto()
	config_proto.gpu_options.allow_growth = True
	with supervisor.managed_session(config=config_proto) as sess:        
		global_step = sess.run(global_step_tensor)
		for iter in range(global_step, 50000):
			start_time = time.time() 
			image_batch, caption_batch = sess.run([image_batch_t, reverse_captions_t]) 
			images, shuffled_captions, labels = permutate(image_batch, caption_batch)					   
			_, loss_step, r_loss_im, r_loss_cap, c_loss, summary = sess.run([train_op, loss, recon_loss_im_t, recon_loss_caption_t, \
																contrastive_loss_t, summary_op], 
																feed_dict={image_placeholder: images,
																label_placeholder: labels, 
																caption_id_placeholder: shuffled_captions})
			
			if (iter+1) % cmd_opt.displayIter == 0:
				end_time = time.time()
				print ("Time per iteration: ", str((end_time-start_time)/cmd_opt.displayIter))
				print ("Total Loss at " , iter+1, ": ", str(loss_step))
				print ("Contrastive Loss at " , iter+1, ": ", str(c_loss))
				print ("Reconstruction Image Loss at " , iter+1, ": ", str(r_loss_im))
				print ("Reconstruction Caption at " , iter+1, ": ", str(r_loss_cap))
				filewriter.add_summary(summary, global_step)
				
    


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--configFile", required=True, help="Configuration for sequence to sequence")
	parser.add_argument("--dataDir", required=True, help="Data directory for training")
	parser.add_argument("--batchSize", default=32, type=int, help="Batch size of the dataset")
	parser.add_argument("--embeddingSize", default=4096, type=int, help="Embedding size")
	parser.add_argument("--contrastiveScale", default=1, type=float, help="Scaling factor for contastive loss")
	parser.add_argument("--imageReconScale", default=1.0, type=float, help="Scaling factor for image reconstruction loss")
	parser.add_argument("--captionReconScale", default=1.0, type=float, help="Scaling factor for caption reconsturction loss")
	parser.add_argument("--margin", default=200, type=int, help="Margin for contrastive loss")
	parser.add_argument("--pretrainedModelRoot", required=True, help="old seq to seq model")
	parser.add_argument("--imageSize", default=227, help="Image size used for image generation")
	parser.add_argument("--displayIter", default=100, help="display iteration")
	parser.add_argument("--expDir", required=True, help="Experiment directory")
	parser.add_argument("--transpose",default=False,type=bool, help="Experiment directory")
	cmd_opt = parser.parse_args()
	print (cmd_opt)
	main(cmd_opt)