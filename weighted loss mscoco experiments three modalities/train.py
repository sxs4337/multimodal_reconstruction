from __future__ import print_function
from __future__ import absolute_import 
from __future__ import division

import numpy as np
import tensorflow as tf
import os
from tensorflow.contrib import slim
from util import create_dataset, create_batch, activation_fn, scale_data, inversion, distance, rescale_data_tf
from util import pairwise_distance
from model import model

def train(cmd_arg, data, gloveVectors):
    image, caption, keys, word = create_dataset(data)
    indices = np.arange(image.shape[0])
    batch_size = cmd_arg.batchSize
    embedding_dimension = cmd_arg.embedSize #max(image.shape[2], caption.shape[2])
    indices_copy = np.copy(indices)
    noise_amp = cmd_arg.embedNoise
    with tf.Graph().as_default():
        image_placeholder = tf.placeholder(dtype=tf.float32, \
                                           shape=[batch_size, image.shape[2]])
        caption_placeholder = tf.placeholder(dtype=tf.float32, \
                                             shape=[batch_size, caption.shape[2]])
        word_placeholder = tf.placeholder(dtype=tf.float32, \
                                             shape=[batch_size, 300])
        label_matrix_placeholder = tf.placeholder(dtype=tf.float32, \
                                            shape=[batch_size, batch_size])
                                            
        scaled_image_placeholder = scale_data(image_placeholder, min=0.0, max=65.0)
        # scaled_image_placeholder = tf.pow(image_placeholder, 1.0) # 1.0 for no scaling
        
        # noisy_image_placeholder = tf.add(scaled_image_placeholder, \
                                    # tf.random_uniform(scaled_image_placeholder.get_shape(), minval=-0.1, maxval=0.1))
        # noisy_caption_placeholder = tf.add(caption_placeholder, \
                                    # tf.random_uniform(caption_placeholder.get_shape(), minval=-0.1, maxval=0.1))
        # noisy_word_placeholder = tf.add(word_placeholder, \
                                    # tf.random_uniform(word_placeholder.get_shape(), minval=0.0, maxval=0.0))
        
        image_embedding_tensor_in, caption_embedding_tensor_in, word_embedding_tensor_in = model(scaled_image_placeholder, \
                                                   caption_placeholder, \
                                                   word_placeholder, \
                                                   embedding_dimension,
                                                   cmd_arg.numberLayers, \
                                                   activation_fn(cmd_arg.activation), \
                                                   'image_embedding', 'caption_embedding', 'word_embedding')
        
        image_embedding_tensor = image_embedding_tensor_in + tf.random_uniform(image_embedding_tensor_in.get_shape(), minval=-1*noise_amp, maxval=noise_amp)
        caption_embedding_tensor = caption_embedding_tensor_in + tf.random_uniform(caption_embedding_tensor_in.get_shape(), minval=-1*noise_amp, maxval=noise_amp)
        word_embedding_tensor = word_embedding_tensor_in + tf.random_uniform(word_embedding_tensor_in.get_shape(), minval=-1*noise_amp, maxval=noise_amp)
        
        ic_positive_distance, ic_negative_distance, distance, labels = pairwise_distance(image_embedding_tensor, caption_embedding_tensor, label_matrix_placeholder, cmd_arg.margin)
        image_positive_distance, image_negative_distance, _, _ = pairwise_distance(image_embedding_tensor, image_embedding_tensor, label_matrix_placeholder, cmd_arg.margin)
        cw_positive_distance, cw_negative_distance, _, _ = pairwise_distance(caption_embedding_tensor, word_embedding_tensor, label_matrix_placeholder, cmd_arg.margin)
        caption_positive_distance, caption_negative_distance, _, _ = pairwise_distance(caption_embedding_tensor, caption_embedding_tensor, label_matrix_placeholder, cmd_arg.margin)
        wi_positive_distance, wi_negative_distance, _, _ = pairwise_distance(word_embedding_tensor, image_embedding_tensor, label_matrix_placeholder, cmd_arg.margin)
        word_positive_distance, word_negative_distance, _, _ = pairwise_distance(word_embedding_tensor, word_embedding_tensor, label_matrix_placeholder, cmd_arg.margin)

        total_loss = image_positive_distance + caption_positive_distance + word_positive_distance \
                    + image_negative_distance + caption_negative_distance + word_negative_distance \
                    + 1.0 * ic_positive_distance + cw_positive_distance + 1.0 * wi_positive_distance \
                    + ic_negative_distance + cw_negative_distance + wi_negative_distance
                
        tf.summary.scalar('loss', total_loss)
        tf.summary.scalar('image_positive_distance', image_positive_distance)
        tf.summary.scalar('caption_positive_distance', caption_positive_distance)
        tf.summary.scalar('word_positive_distance', word_positive_distance)
        tf.summary.scalar('image_negative_distance', image_negative_distance)
        tf.summary.scalar('caption_negative_distance', caption_negative_distance)
        tf.summary.scalar('word_negative_distance', word_negative_distance)
        tf.summary.scalar('ic_positive_distance', ic_positive_distance)
        tf.summary.scalar('cw_positive_distance', cw_positive_distance)
        tf.summary.scalar('wi_positive_distance', wi_positive_distance)
        tf.summary.scalar('ic_negative_distance', ic_negative_distance)
        tf.summary.scalar('cw_negative_distance', cw_negative_distance)
        tf.summary.scalar('wi_negative_distance', wi_negative_distance)

        global_step_tensor = tf.Variable(0, trainable=False)
        # train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(total_loss,\
                                                                       # global_step_tensor)
        optimizer  = tf.train.AdamOptimizer(learning_rate=0.001)
        gvs = optimizer .compute_gradients(total_loss)
        capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs, global_step_tensor)
        
        saver = tf.train.Saver()
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        summary_tensor = tf.summary.merge_all()
        epoch_per_iteration = image.shape[0]//cmd_arg.batchSize
        with tf.Session(config=session_config) as session:
            summary_writer = tf.summary.FileWriter(os.path.join(cmd_arg.experimentDirectory,
                                                   'train'), graph=tf.get_default_graph())
            session.run([tf.global_variables_initializer()])
            for i in range(cmd_arg.totalIteration):
                if indices_copy.shape[0] == 0:
                    indices_copy = np.copy(indices)
                train_image, train_caption, train_word, train_keys, indices_copy, label_matrix = \
                                               create_batch(image, caption, word, keys, indices_copy, \
                                               batch_size, gloveVectors)
                _, loss, im_pos, cap_pos, wor_pos, im_neg, cap_neg, wor_neg, \
                    ic_pos, cw_pos, wi_pos, ic_neg, cw_neg, wi_neg, dist, labs, summary, \
                            global_step = session.run([train_op, total_loss, \
                                                    image_positive_distance, caption_positive_distance, word_positive_distance, \
                                                    image_negative_distance, caption_negative_distance, word_negative_distance, \
                                                    ic_positive_distance, cw_positive_distance, wi_positive_distance, \
                                                    ic_negative_distance, cw_negative_distance, wi_negative_distance, distance, labels, \
                                                    summary_tensor, global_step_tensor],
                                                    feed_dict={
                                                    image_placeholder: train_image,
                                                    caption_placeholder: train_caption,
                                                    word_placeholder: train_word,
                                                    label_matrix_placeholder: label_matrix
                                                    })
                epoch = i//epoch_per_iteration + 1
                iteration = i%epoch_per_iteration + 1
                print ("Epoch: ", epoch, ", Iteration: ", iteration, " Total loss: ", loss)
                print ("Epoch: ", epoch, ", Iteration: ", iteration, " : Img pos : ", im_pos)
                print ("Epoch: ", epoch, ", Iteration: ", iteration, " : Cap pos : ", cap_pos)
                print ("Epoch: ", epoch, ", Iteration: ", iteration, " : Wor pos : ", wor_pos)
                print ("Epoch: ", epoch, ", Iteration: ", iteration, " : Img neg : ", im_neg)
                print ("Epoch: ", epoch, ", Iteration: ", iteration, " : Cap neg : ", cap_neg)
                print ("Epoch: ", epoch, ", Iteration: ", iteration, " : Wor neg : ", wor_neg)
                print ("Epoch: ", epoch, ", Iteration: ", iteration, " : IC pos : ", ic_pos)
                print ("Epoch: ", epoch, ", Iteration: ", iteration, " : CW pos : ", cw_pos)
                print ("Epoch: ", epoch, ", Iteration: ", iteration, " : WI pos : ", wi_pos)
                print ("Epoch: ", epoch, ", Iteration: ", iteration, " : IC neg : ", ic_neg)
                print ("Epoch: ", epoch, ", Iteration: ", iteration, " : CW neg : ", cw_neg)
                print ("Epoch: ", epoch, ", Iteration: ", iteration, " : WI neg : ", wi_neg)
                # import pdb
                # pdb.set_trace()
                summary_writer.add_summary(summary, global_step)
                if (i+1)%cmd_arg.saveIteration == 0:
                    saver.save(session, os.path.join(cmd_arg.experimentDirectory, \
                                                     'model.ckpt'+str(global_step)))
