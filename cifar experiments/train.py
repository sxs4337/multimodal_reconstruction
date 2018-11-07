from __future__ import print_function
from __future__ import  division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
from model.vector_model import image_branch, word_branch
from model_util import create_batches, get_vars, \
    create_placeholders
from eval import eval_model, compute_word_accuracy

import time, random
import pdb
from tensorflow.python import debug as tf_debug

word_name = "universal_embedding/word_branch/word_embedding"
image_name = "universal_embedding/image_branch/image_embedding"



def _euc_distance(logit_one, logit_two):
    return tf.sqrt(tf.reduce_sum(tf.square(
            tf.subtract(logit_one, logit_two)), 1, keep_dims=True))

            
def contrastive_loss(word_logits, image_logits, groundtruth_label,
                     margin=50., scope="contrastive_loss"):
    with tf.variable_scope(scope):
        distance = _euc_distance(word_logits, image_logits)
        loss = tf.reduce_mean(groundtruth_label * tf.square(distance) +
                              (1-groundtruth_label)*tf.square(tf.maximum(0.,
                                float(margin)-distance)))*0.5
        return loss

        
def embedding_inversion(embedding, name, batch_size):
    with tf.variable_scope("embedding_inversion"):
        embedding = tf.log(embedding+1e-7) - tf.log(1-embedding+1e-7)
        embedding_weight = tf.contrib.framework.get_variables_by_name(name+"/weights")
        embedding_bias = tf.contrib.framework.get_variables_by_name(name+"/biases")        
        embedding_inverse_tensor = tf.matrix_solve_ls(tf.transpose(tf.tile(embedding_weight,
                                                    [batch_size, 1, 1]), [0, 2, 1]), 
                                                    tf.expand_dims(tf.subtract(embedding, embedding_bias), 2), 
                                                    fast=True, l2_regularizer=1e-5)    
    return tf.squeeze(embedding_inverse_tensor)


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

    
def add_optimizer(image_embedding, word_embedding, image_logits, \
                    word_logits, groundtruth, learning_rate, \
                    global_step, loss_margin, batch_size):
    contrastive_loss_t = contrastive_loss(image_embedding, word_embedding,
                            groundtruth, loss_margin)
    image_reconstruction_loss = reconstruction_loss(image_logits, image_embedding, image_name, 
                                                    batch_size, is_transpose=True)
    word_reconstruction_loss = reconstruction_loss(word_logits, word_embedding, word_name,
                                                    batch_size, is_transpose=True)
    loss = contrastive_loss_t # + image_reconstruction_loss + word_reconstruction_loss
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        freeze_var = get_vars(tf.trainable_variables(), "CifarNet", exclude_var=True) #None
        train_op = optimizer.minimize(loss, global_step=global_step,
                                      var_list=freeze_var)
    return loss, train_op


def train_model(image,  word_vec, groundtruth, embedding_size, learning_rate,  
                global_step, loss_margin=150, num_classes=10, batch_size=64,
                scope="universal_embedding"):
    with tf.variable_scope(scope):
        end_point, image_embedding = image_branch(image, is_train=False,
                                       embedding_size=embedding_size)
        word_embedding = word_branch(word_vec, embedding_size)
    loss, train_op = add_optimizer(image_embedding, word_embedding,\
                                    end_point, word_vec,\
                                    groundtruth, learning_rate,\
                                    global_step, loss_margin, batch_size)
    return image_embedding, word_embedding, loss, train_op


def train_iter(session, train_op, loss_tensor, global_step_tensor,
          train_image_tensor, train_vector_tensor, train_label_tensor, word2vec_matrix, 
          distance_matrix, img_placeholder,
          vec_placeholder, gt_placeholder, permutation=False):
    positive_train_image, positive_train_vector, positive_train_label = \
        session.run([train_image_tensor, train_vector_tensor, train_label_tensor])
    negative_train_image = np.copy(positive_train_image)
    if permutation:
        negative_train_vector = \
            np.random.permutation(positive_train_vector)
    else:
        negative_train_vector = np.zeros_like(positive_train_vector)
        for i in xrange(positive_train_vector.shape[0]):
            negative_train_vector[i,:] = word2vec_matrix[random.choice(np.where(distance_matrix[i]>0.85))[0]]
    batch_train_image = np.vstack([positive_train_image,
                                   negative_train_image])
    batch_train_vector = np.vstack([positive_train_vector,
                                    negative_train_vector])
    groundtruth_label = \
        np.vstack([np.ones([positive_train_image.shape[0], 1]),
                   np.zeros([positive_train_image.shape[0], 1])])
    shuffled_index = np.random.permutation(np.arange(0,
                            batch_train_image.shape[0]))
    batch_train_image[shuffled_index] = batch_train_image
    batch_train_vector[shuffled_index] = batch_train_vector
    groundtruth_label[shuffled_index] = groundtruth_label
    _, loss, global_step =  \
        session.run([train_op, loss_tensor, global_step_tensor],
                    feed_dict={img_placeholder: batch_train_image,
                               vec_placeholder: batch_train_vector,
                               gt_placeholder: groundtruth_label})
    return loss, global_step
    

def train(cmd_opt, word2vec_matrix, dataset_loader, num_classes, image_size,
            word2vec_embedding_size, image_branch, validation_examples):
    global_step_tensor = tf.Variable(0, trainable=False, name="global_step")
    train_image_tensor, train_vector_tensor, train_label_tensor = \
        create_batches(dataset_loader, cmd_opt.batchSize // 2, image_size,
                       word2vec_embedding_size,)
    valid_image_tensor, valid_vector_tensor, valid_label_tensor = \
        create_batches(dataset_loader, cmd_opt.batchSize, image_size,
                        word2vec_embedding_size, is_train=False, 
                        is_valid=True)
    
    image_placeholder, wordvec_placeholder, groundtruth_placeholder, \
        matrix_placeholder = create_placeholders(cmd_opt.batchSize, image_size,
                                                 word2vec_embedding_size, num_classes)
    im_em, word_em, loss_tensor, train_op = train_model(image = image_placeholder, 
                                        word_vec = wordvec_placeholder,
                                        groundtruth = groundtruth_placeholder,
                                        embedding_size = cmd_opt.embeddingSize,
                                        learning_rate = cmd_opt.learningRate, 
                                        global_step = global_step_tensor,
                                        loss_margin=cmd_opt.margin, num_classes=num_classes,
                                        batch_size=cmd_opt.batchSize)
    label_tensor, _, embedding_inversion = eval_model(image_embedding = im_em, 
                              wordvec_embedding = word_em,
                              matrix = matrix_placeholder,                  
                              batch_size = cmd_opt.batchSize,
                              num_classes = num_classes, 
                              word_validation = cmd_opt.validation==0,
                              image_validation = cmd_opt.validation==1)
    supervisor_saver = tf.train.Saver()    
    # supervisor = tf.train.Supervisor(logdir=None,#cmd_opt.expDir, 
                                     # summary_op=None,
                                     # global_step=global_step_tensor,
                                     # saver=supervisor_saver)
    
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    # config_proto.gpu_options.per_process_gpu_memory_fraction = 0.6

    session = tf.Session(config=config_proto)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=session)
    session = tf_debug.LocalCLIDebugWrapperSession(session)

    # with supervisor.managed_session(config=config_proto) as session:
    for i in range(1): 
        session.run(tf.global_variables_initializer())
        global_step = session.run(global_step_tensor)
        start_time = time.time()
        for iter in range(global_step, cmd_opt.numIters):
            loss, global_step = train_iter(session, train_op, loss_tensor,
                                              global_step_tensor, train_image_tensor,
                                              train_vector_tensor, train_label_tensor, dataset_loader.word2vec_matrix, \
                                              dataset_loader.distance_matrix, image_placeholder,
                                              wordvec_placeholder, groundtruth_placeholder, permutation=True)
            if (iter+1) % cmd_opt.displayIters == 0:
                end_time = time.time()
                print ("Time per iteration: ", str((end_time-start_time)/cmd_opt.displayIters))
                print ("Training Loss at " , iter+1, ": ", str(loss))
                start_time = time.time()
            # if (iter+1) % cmd_opt.validIters == 0:
                # if cmd_opt.validation == 0:
                   
            print("Validation accuracy: ",
                  compute_word_accuracy(session, valid_image_tensor,
                                              valid_label_tensor, label_tensor,
                                              image_placeholder,
                                              cmd_opt.batchSize, word2vec_matrix,
                                              matrix_placeholder, validation_examples))
                # else:
                    # raise NotImplementedError
