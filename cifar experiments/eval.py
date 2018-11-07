from __future__ import print_function
from __future__ import  division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import imageio
import os
from model.vector_model import image_branch, word_branch
from model_util import create_batches, create_placeholders, get_vars
from model.generator_cifar import generator

word_weight_name = "universal_embedding/word_branch/word_embedding/weights"
word_bias_name = "universal_embedding/word_branch/word_embedding/biases"

image_weight_name = "universal_embedding/image_branch/image_embedding/weights"
image_bias_name = "universal_embedding/image_branch/image_embedding/biases"


categories = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat",
          4: "deer", 5: "dog", 6: "frog", 7: "horse", 
          8: "ship", 9:  "truck"}

def _validation(embedding, batch_size, weight_name, bias_name):
    with tf.variable_scope("embedding_inversion"):
        embedding = tf.log(embedding) - tf.log(1-embedding)
        embedding_bias = tf.contrib.framework.\
            get_variables_by_name(bias_name)
        embedding_weight = tf.contrib.framework.\
            get_variables_by_name(weight_name)
        embedding_inverse_tensor = tf.matrix_solve_ls(
            tf.transpose(tf.tile(embedding_weight,
                                 [batch_size, 1, 1]), [0, 2, 1]),
            tf.expand_dims(tf.subtract(embedding,
                                       embedding_bias), 2))
    return tf.squeeze(embedding_inverse_tensor)


def label_num(embedding_inverse, embedding_matrix, batch_size, num_classes):
    with tf.variable_scope("distance_computation"):
        tiled_embedding = tf.tile(tf.expand_dims(embedding_inverse, 1),
                                         [1, num_classes, 1])
        tiled_matrix = tf.tile(tf.expand_dims(embedding_matrix, 0),
                                    [batch_size, 1, 1])
        distance = tf.sqrt(tf.reduce_sum(tf.square(
            tf.subtract(tiled_embedding, tiled_matrix)), 2))
    return tf.argmin(distance, 1)



def eval_model(image_embedding, wordvec_embedding,
               matrix, batch_size, num_classes=10, word_validation=True,
               image_validation=False):
    label = None
    generated_image = None
    if word_validation:
        with tf.variable_scope("image_inversion"):
            embedding_inverse = _validation(image_embedding, batch_size, \
                                        word_weight_name, word_bias_name)
            label = label_num(embedding_inverse, matrix,  batch_size, 
                                    num_classes)
    elif image_validation:
        with tf.variable_scope("word_inversion"):
            embedding_inverse = _validation(wordvec_embedding, batch_size, \
                                           image_weight_name, image_bias_name)
            generated_image = generator(embedding_inverse)
            generated_image = tf.cast(tf.multiply(tf.div(tf.add(generated_image, 
                                      1.0), 2), 255), tf.uint8)
    return label, generated_image


def compute_word_accuracy(session, image_tensor, gt_label_tensor, label_tensor,
                          image_placeholder, batch_size, word_matrix, 
                          matrix_placeholder, num_examples):
    print("Starting Evaluation Run")
    running_accuracy = 0
    num_iter = num_examples // batch_size + 1
    for i in range(num_iter):
        valid_images, valid_labels = \
            session.run([image_tensor, gt_label_tensor])
        predicted_labels = session.run(label_tensor,
                                       feed_dict={image_placeholder: valid_images,
                                                  matrix_placeholder: word_matrix})

        
        running_accuracy += np.sum(np.equal(predicted_labels, valid_labels))
    print (predicted_labels)
    return running_accuracy / (num_iter * batch_size)


def generate_image(session, word_tensor, generated_image_tensor, 
                   word2vec_placeholder, batch_size, num_examples,
                   image_dir):
    print("Generating Images")
    num_iters = num_examples // batch_size + 1
    for curr_iter in range(num_iters):
        word_vec = session.run(word_tensor)
        generated_image = session.run(generated_image_tensor,
                                      feed_dict={word2vec_placeholder: word_vec})
        for batch in range(batch_size):
            imageio.imwrite(os.path.join(image_dir, 
                            str(curr_iter*batch_size + batch + 1) + '.png'), 
                            generated_image[batch])

    
def eval(cmd_opt, word2vec_matrix, cifar_loader, num_classes, image_size,
         word2vec_embedding_size, num_examples, image_dir,
         scope="universal_embedding"):
    test_image_tensor, test_vector_tensor, test_label_tensor = \
        create_batches(cifar_loader, cmd_opt.batchSize, image_size,
                           word2vec_embedding_size, is_train=False, 
                           is_valid=False)
    image_placeholder, wordvec_placeholder, __, matrix_placeholder = \
        create_placeholders(cmd_opt.batchSize, image_size,
                                        word2vec_embedding_size, num_classes)
    with tf.variable_scope(scope):
        image_embedding = image_branch(image_placeholder, num_classes, 
                                   cmd_opt.embeddingSize, is_train=False)
        word_embedding = word_branch(wordvec_placeholder, cmd_opt.embeddingSize)
    label_tensor, generated_image_tensor = eval_model(image_embedding, word_embedding,
                                                  matrix_placeholder, cmd_opt.batchSize, 
                                                  num_classes, cmd_opt.validation==0,
                                                  cmd_opt.validation==1)
    model_restore_vars = get_vars(tf.trainable_variables(), 'logits', exclude_var=True)
    model_restore_vars = get_vars(model_restore_vars, scope)
    model_saver = tf.train.Saver(var_list=model_restore_vars)
    if cmd_opt.validation == 1:
        generator_restore_vars = get_vars(tf.trainable_variables(), 'generator')
        generator_dict = {}
        for generator_var in generator_restore_vars:
            generator_dict['deepSim/' + generator_var.op.name.split('/', 1)[1]] = generator_var
            generator_saver = tf.train.Saver(var_list=generator_dict)
    def _init_fn(session):            
        model_path = tf.train.latest_checkpoint(cmd_opt.expDir)
        model_saver.restore(session, model_path)
        if cmd_opt.generatorDir is not None:
            generator_path = tf.train.latest_checkpoint(cmd_opt.generatorDir)
            generator_saver.restore(session, generator_path)
    supervisor =  tf.train.Supervisor(logdir=None, init_fn=_init_fn)
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    all_ie = []
    all_il = []
    all_we = []
    with supervisor.managed_session(config=config_proto) as session:
        # num_iter = num_examples // cmd_opt.batchSize + 1
        # for i in range(num_iter):
            # test_image, test_vector = session.run([test_image_tensor, test_vector_tensor])
            # ie, il, we = session.run([image_embedding, test_label_tensor, \
                                      # word_embedding], feed_dict={
                                        # image_placeholder: test_image,                                    
                                        # wordvec_placeholder: test_vector
                                    # })
            # all_ie.extend(ie)
            # all_il.extend(il)
            # all_we.extend(we)
            
        # image_file = open('image.txt', 'w')
        # word_file = open('word.txt', 'w')
        # for i, label in enumerate(all_il):
            # word_label = categories[label]
            # img_e = " ".join(map(str, all_ie[i].tolist()))
            # word_e = " ".join(map(str, all_we[i].tolist()))
            # image_file.write(word_label + " " + img_e + "\n")
            # word_file.write(word_label + " " + word_e + "\n")
        # image_file.close()
        # word_file.close()
    
        if cmd_opt.validation == 0:
            print("Validation accuracy: ", 
                    compute_word_accuracy(session, test_image_tensor,
                                          test_label_tensor, label_tensor,
                                          image_placeholder,
                                          cmd_opt.batchSize, word2vec_matrix,
                                          matrix_placeholder, num_examples))
        elif cmd_opt.validation == 1:
             generate_image(session, test_vector_tensor, generated_image_tensor,
                            wordvec_placeholder, cmd_opt.batchSize, num_examples, 
                            image_dir) 
        else:
            raise NotImplementedError
