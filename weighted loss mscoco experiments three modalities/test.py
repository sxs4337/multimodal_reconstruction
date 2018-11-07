from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import os
import pickle as pkl
from util import create_dataset, create_batch, activation_fn, scale_data, rescale_data
from model import model

def activation_inversion(input_tensor, activation_id):
    if activation_id == 0: # no activation
        return input_tensor
    elif activation_id == 1:
        input_mask = np.ones(input_tensor.shape)
        input_mask[input_tensor < 0] = 1/0.2
        return np.multiply(input_tensor, input_mask)
    elif activation_id == 2: # sigmoid
        return np.log(input_tensor+1e-10) - np.log(1-input_tensor+1e-10)
    elif activation_id == 3: # tanh
        return np.multiply(0.5, (np.log(1.0+input_tensor+1e-10) - np.log(1.0-input_tensor+1e-10)))
    else:
        raise NotImplementedError

def inversion(embedding, weights, biases, activation_id, isTranspose = True):
    for i in range(len(weights)-1, -1, -1):
        embedding = activation_inversion(embedding, activation_id)
        embedding = np.subtract(embedding, biases[i])
        if not isTranspose:
            # embedding = np.dot(embedding, np.linalg.pinv(weights[i]))
            embedding = np.dot(embedding, weights[i])
        else:
            embedding = np.dot(embedding, weights[i].T)
    return embedding

def test(cmd_arg, data, gloveVectors):
    image, caption, keys, word = create_dataset(data)
    batch_size = cmd_arg.batchSize
    if image.shape[0] % batch_size != 0:
        idx = range((keys.shape[0] // batch_size) * batch_size)
        keys = keys[idx]
        image = image[idx,:,:]
        caption = caption[idx,:,:]
        word = [word[x] for x in idx]
    indices = np.arange(image.shape[0])
    embedding_dimension = cmd_arg.embedSize # max(image.shape[2], caption.shape[2])
    indices_copy = np.copy(indices)
    with tf.Graph().as_default():
        image_placeholder = tf.placeholder(dtype=tf.float32, \
                                           shape=[cmd_arg.batchSize, \
                                                  image.shape[2]])
        caption_placeholder = tf.placeholder(dtype=tf.float32, \
                                             shape=[cmd_arg.batchSize, \
                                                    caption.shape[2]])
        word_placeholder = tf.placeholder(dtype=tf.float32, \
                                             shape=[cmd_arg.batchSize, \
                                                    300])

        scaled_image_placeholder = scale_data(image_placeholder, min=0.0, max=65.0)
        # scaled_image_placeholder = tf.pow(image_placeholder, 1.0) # 1.0 for no scaling
        
        image_embedding_tensor, caption_embedding_tensor, word_embedding_tensor = model(scaled_image_placeholder, \
                                                   caption_placeholder, \
                                                   word_placeholder, \
                                                   embedding_dimension,
                                                   cmd_arg.numberLayers, \
                                                   activation_fn(cmd_arg.activation), \
                                                   'image_embedding', 'caption_embedding', 'word_embedding')
        
        saver = tf.train.Saver()
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True     
        with tf.Session(config=session_config) as session:
            saver.restore(session, tf.train.latest_checkpoint(cmd_arg.experimentDirectory))
            print ('Restored Checkpoint: ', tf.train.latest_checkpoint(cmd_arg.experimentDirectory))
            image_weights = [session.run(tf.contrib.framework.get_variables_by_name(\
                             'image_embedding/image_embedding_'+ str(i+1)+'/weights')[0]) \
                             for i in range(cmd_arg.numberLayers)]
            image_biases = [session.run(tf.contrib.framework.get_variables_by_name(\
                            'image_embedding/image_embedding_'+ str(i+1)+'/biases')[0]) \
                             for i in range(cmd_arg.numberLayers)]
            caption_weights = [session.run(tf.contrib.framework.get_variables_by_name(\
                             'caption_embedding/caption_embedding_'+ str(i+1)+'/weights')[0]) \
                              for i in range(cmd_arg.numberLayers)]
            caption_biases = [session.run(tf.contrib.framework.get_variables_by_name(\
                             'caption_embedding/caption_embedding_'+ str(i+1)+'/biases')[0]) \
                             for i in range(cmd_arg.numberLayers)]
            word_weights = [session.run(tf.contrib.framework.get_variables_by_name(\
                             'word_embedding/word_embedding_'+ str(i+1)+'/weights')[0]) \
                              for i in range(cmd_arg.numberLayers)]
            word_biases = [session.run(tf.contrib.framework.get_variables_by_name(\
                             'word_embedding/word_embedding_'+ str(i+1)+'/biases')[0]) \
                             for i in range(cmd_arg.numberLayers)]            

            image_weights_inv = [np.linalg.pinv(x) for x in image_weights]
            caption_weights_inv = [np.linalg.pinv(x) for x in caption_weights]
            word_weights_inv = [np.linalg.pinv(x) for x in word_weights]
            
            all_cc_embedding = []
            all_ic_embedding = []
            all_wc_embedding = []
            all_ii_embedding = []
            all_ci_embedding = []
            all_wi_embedding = []
            all_iw_embedding = []
            all_cw_embedding = []
            all_ww_embedding = []
            
            all_image_embedding = []
            all_caption_embedding = []
            all_word_embedding = []
            
            all_original_caption = []
            all_original_image = []
            all_valid_key = []
            for i in range(0, image.shape[0], batch_size):
            # for i in range(1):
                valid_image, valid_caption, valid_word, valid_keys, indices_copy, _ =\
                                               create_batch(image, caption, word, keys, indices_copy, \
                                               batch_size, gloveVectors)
                image_embedding, caption_embedding, word_embedding = session.run([image_embedding_tensor, \
                                                                caption_embedding_tensor, word_embedding_tensor],
                                                                feed_dict={
                                                                image_placeholder: valid_image,
                                                                caption_placeholder: valid_caption,
                                                                word_placeholder: valid_word
                                                                })
                
                all_image_embedding.extend(image_embedding)
                all_caption_embedding.extend(caption_embedding)
                all_word_embedding.extend(word_embedding)
                # from test_three_branch_pkl import save_image
                
                ii_vector = inversion(image_embedding, image_weights_inv, image_biases, cmd_arg.activation, isTranspose = False)
                ic_vector = inversion(image_embedding, caption_weights_inv, caption_biases, cmd_arg.activation, isTranspose = False)
                iw_vector = inversion(image_embedding, word_weights_inv, word_biases, cmd_arg.activation, isTranspose = False)
                ci_vector = inversion(caption_embedding, image_weights_inv, image_biases, cmd_arg.activation, isTranspose = False)
                cc_vector = inversion(caption_embedding, caption_weights_inv, caption_biases, cmd_arg.activation, isTranspose = False)
                cw_vector = inversion(caption_embedding, word_weights_inv, word_biases, cmd_arg.activation, isTranspose = False)
                wi_vector = inversion(word_embedding, image_weights_inv, image_biases, cmd_arg.activation, isTranspose = False)
                wc_vector = inversion(word_embedding, caption_weights_inv, caption_biases, cmd_arg.activation, isTranspose = False)
                ww_vector = inversion(word_embedding, word_weights_inv, word_biases, cmd_arg.activation, isTranspose = False)
                
                all_ii_embedding.extend(ii_vector)
                all_ic_embedding.extend(ic_vector)
                all_iw_embedding.extend(iw_vector)
                all_ci_embedding.extend(ci_vector)
                all_cc_embedding.extend(cc_vector)
                all_cw_embedding.extend(cw_vector)
                all_wi_embedding.extend(wi_vector)
                all_wc_embedding.extend(wc_vector)
                all_ww_embedding.extend(ww_vector)
                
                all_original_caption.extend(valid_caption)
                all_original_image.extend(valid_image)
                all_valid_key.extend(valid_keys)
                print ("Completed:", (i+1)//batch_size, "/", image.shape[0]//batch_size)             
            with open(os.path.join(cmd_arg.experimentDirectory,'validation_inversion_embedding.pkl'), 'wb') as f:
                pkl.dump({"image_keys": all_valid_key,
                    "cc_embedding": all_cc_embedding,
                    "ic_embedding": all_ic_embedding, 
                    "wc_embedding": all_wc_embedding,
                    "ci_embedding": all_ci_embedding,
                    "wi_embedding": all_wi_embedding,
                    "ii_embedding": all_ii_embedding,
                    "cw_embedding": all_cw_embedding,
                    "iw_embedding": all_iw_embedding,
                    "ww_embedding": all_ww_embedding,
                    "image_embedding": all_image_embedding,
                    "caption_embedding": all_caption_embedding,
                    "word_embedding": all_word_embedding,
                    "orig_image_embedding": all_original_image,
                    "orig_caption_embedding": all_original_caption }, f)
                             
                 
