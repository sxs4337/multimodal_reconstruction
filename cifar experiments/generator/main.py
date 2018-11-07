from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import time
import os
import tensorflow.contrib.slim as  slim
from imagenet_loader import ImagenetLoader
from mscoco_loader import MSCOCOLoader
from cifar_loader import CifarLoader
from model.discriminator_cifar import discriminator, discriminator_scope
from model.generator_cifar import generator
from model.encoder_cifar import encoder, encoder_scope
import pdb
import imageio
import numpy as np
from tensorflow.python import debug as tf_debug



def _adversarial_loss(discriminator_logit, groundtruth_label,
                      summary_label, loss_scope, adversarial_weight):
    """
    Helper routine to compute adversarial loss
    :param discriminator_logit: Output of discriminator
    :param groundtruth_label: groundtruth values
    :param summary_label: name of the loss used for tensorboard
    :param loss_scope: scope of the loss on the graph
    :param adversarial_weight: weight associated with the loss
    :return: tensor representing adversarial loss
    """
    with tf.variable_scope(loss_scope):
        adversarial_loss = tf.reduce_mean(tf.multiply(
                             tf.nn.sigmoid_cross_entropy_with_logits(
                               logits=discriminator_logit, labels=groundtruth_label),
                                adversarial_weight))
        tf.summary.scalar(summary_label, adversarial_loss)
    return adversarial_loss


    
    
def _pixelwise_euclidean_loss(original_image, generated_image, 
                              summary_label, loss_scope, pixelwise_weight):
    """
    Helper routine to compute image loss
    :param original_image: ground truth
    :param generated_image: image from generator
    :param summary_label: name of the loss used for tensorboard
    :param loss_scope: scope of the loss on the graph
    :param pixelwise_weight: weight associated with the loss
    :return: tensor representing pixelwise euclidean loss
    """
    with tf.variable_scope(loss_scope):
        original_image = slim.flatten(original_image)
        generated_image = slim.flatten(generated_image)
        pixelwise_loss = tf.reduce_mean(tf.multiply(tf.sqrt(tf.reduce_sum(
                             tf.square(tf.subtract(original_image,
                                       generated_image)), axis=1, keep_dims=True)),
                                       pixelwise_weight))
        tf.summary.scalar(summary_label, pixelwise_loss)
    return pixelwise_loss


def _reconstruction_loss(groundtruth_feature, reconstructed_feature,
                         summary_label, loss_scope, reconstruction_weight):
    """
    Helper routine to compute image loss
    :param groundtruth_feature: ground truth
    :param reconstructed_feature: image from generator
    :param summary_label: name of the loss used for tensorboard
    :param loss_scope: scope of the loss on the graph
    :param reconstruction_weight: weight associated with the loss
    :return: tensor representing pixelwise euclidean loss
    """
    with tf.variable_scope(loss_scope, [groundtruth_feature, reconstructed_feature]):
        groundtruth_feature = slim.flatten(groundtruth_feature)
        reconstructed_feature = slim.flatten(reconstructed_feature)
        reconstruction_loss = tf.reduce_mean(tf.multiply(tf.sqrt(
                                tf.reduce_sum(tf.square(tf.subtract(groundtruth_feature, 
				    	reconstructed_feature)), axis=1, keep_dims=True)),
                                        reconstruction_weight))
        tf.summary.scalar(summary_label, reconstruction_loss)
    return reconstruction_loss


def _add_grad_vars(grad_vars_1, grad_vars_2):
    """
    Helper routine to pairwise add gradient from different sources
    :param grad_vars_1, grad_vars_2: list of pairs containing op name and gradient values
    :return: list of pairs containing the sum of gradient
    """
    grad_var_sum = []
    for grad_var_1, grad_var_2 in zip(grad_vars_1, grad_vars_2):
        if grad_var_1[0] is None:
            if grad_var_2[0] is None:
                grad_var_sum.append((None, grad_var_2[1]))
            else:
                grad_var_sum.append(grad_var_2)
        elif grad_var_2[0] is None:
            if grad_var_1[0] is None:
                grad_var_sum.append((None, grad_var_2[1]))
            else:
                grad_var_sum.append(grad_var_1)
        else:
            grad_var_sum.append((tf.add(grad_var_1[0], grad_var_2[0]),  grad_var_2[1]))
    return grad_var_sum


def _get_grad_ops(grad_var_list, substring):
    """
    Pruning operations based on the substring
    :params grad_var_list: list of pairs (op_name and value of gradient)
    :params substring: used for pruning
    :return: pruned list of pairs
    """
    ops_list = []
    for grad_var in grad_var_list:
        if substring in grad_var[1].op.name:
            ops_list.append(grad_var)
    return ops_list

def _upscale_image(image):
        return tf.multiply(tf.multiply(tf.add(image, 1.0), 0.5), 255)     

def create_model(image, adversarial_weight, pixelwise_weight, reconstruction_weight,
                 learning_rate, encoder_is_training, generator_is_training, 
                 discriminator_is_training, image_size):
    """
    Define model definition
    :param image: Input image
    :param adversarial_weight: weight associated with adversarial loss
    :param pixelwise_weight: weight associated with pixel loss
    :param reconstruction_weight: weight associated with encoder loss
    :param learning_rate: learning rate associated with model
    :param encoder_is_training: flag to train encoder
    :param generator_is_training: flag to train generator
    :param discriminator_is_training: flag to train generator
    :return: tuple of 9 tensor elements (5 loss values, 2 training ops and 2 global steps)
    """        
    # Two global steps
    generator_global_step = tf.Variable(0, trainable=False, name="generator_global_step")
    discriminator_global_step = tf.Variable(0, trainable=False, name='discriminator_global_step')
    tf.summary.scalar('generator_global_step', generator_global_step)
    tf.summary.scalar('discriminator_global_step', discriminator_global_step)
   
    with tf.variable_scope('deepSim', [image]) as scope:        
        # Get orginal features
        with slim.arg_scope(encoder_scope()):
            _, end_points = encoder(image, is_training=encoder_is_training)
        real_conv_features = end_points["conv4"]
        fc_features = end_points["fc5"]
        # Generate images using fc6 features
        generated_image = generator(fc_features, image_size, generator_is_training)
        
        # Run discriminator on real data
        with slim.arg_scope(discriminator_scope()):
            discriminator_real = discriminator(image, discriminator_is_training)
        real_labels = tf.ones_like(discriminator_real) * 0.7
        discriminator_real_loss = _adversarial_loss(discriminator_real, real_labels,
                                                   'discriminator_real_loss',
                                                   'discriminator_real_loss',
                                                   adversarial_weight)
        scope.reuse_variables()                                        
        
        with slim.arg_scope(encoder_scope()):
            generated_image = _upscale_image(generated_image)
            _, end_points = encoder(generated_image, is_training=encoder_is_training)
        generated_conv_features = end_points["conv4"]
        generated_fc_features = end_points["fc5"]
        with slim.arg_scope(discriminator_scope()):
            discriminator_fake = discriminator(generated_image, discriminator_is_training)
        fake_labels = tf.zeros_like(discriminator_fake) #+ 0.3
        discriminator_fake_loss = _adversarial_loss(discriminator_fake, fake_labels,
                                                    'discriminator_fake_loss',
                                                    'discriminator_fake_loss',
                                                    adversarial_weight)
        real_labels = tf.ones_like(discriminator_fake)
        generator_loss = _adversarial_loss(discriminator_fake, real_labels,
                                           'generator_loss',
                                           'generator_loss', 
                                           adversarial_weight)
        # Pixelwise loss and reconstruction loss
        pixelswise_loss = _pixelwise_euclidean_loss(image, generated_image,
                                                               "pixelwise_loss",
                                                               "pixelwise_loss", 
                                                               pixelwise_weight)
        reconstruction_loss = _reconstruction_loss(real_conv_features,
                                                   generated_conv_features,
                                                   'reconstruction_loss', 
                                                   'reconstruction_loss',
                                                   reconstruction_weight)
                                                                                                      
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        model_optimizer = tf.train.AdamOptimizer(learning_rate)
        discriminator_real_gradients = model_optimizer.compute_gradients(discriminator_real_loss)
        discriminator_fake_gradients = model_optimizer.compute_gradients(discriminator_fake_loss)
        discriminator_total_gradient = _add_grad_vars(discriminator_fake_gradients, 
                                                      discriminator_real_gradients)
        discriminator_var_list = _get_grad_ops(discriminator_total_gradient, 'discriminator')
        discriminator_train_op = model_optimizer.apply_gradients(discriminator_var_list,
                                                                 discriminator_global_step)

        generator_gradient = model_optimizer.compute_gradients(generator_loss)
        pixelwise_gradient = model_optimizer.compute_gradients(pixelswise_loss)
        generator_gradient = _add_grad_vars(generator_gradient, pixelwise_gradient)
        # Gradients for encoder
        encoder_gradient = model_optimizer.compute_gradients(reconstruction_loss)
    
        # Sum of all gradients
        generator_total_gradients = _add_grad_vars(encoder_gradient, generator_gradient)
        generator_var_list = _get_grad_ops(generator_total_gradients, 'generator')
        generator_train_op = model_optimizer.apply_gradients(generator_var_list,
                                                          generator_global_step)
    return pixelswise_loss, reconstruction_loss, discriminator_real_loss, \
            discriminator_fake_loss, generator_loss, discriminator_train_op, \
            generator_train_op, discriminator_global_step, generator_global_step,\
            generated_image


def _map_ops(original_vars, substring):
    """
    get subset of operations from the path
    :substring: search string used to create the subset
    :return: operators with substring in the name
    """
    var_dict = {}
    #pdb.set_trace()
    for original_var in original_vars:
        if substring in original_var.op.name:
            var_dict["CifarNet/" + original_var.op.name.split("/", 2)[2]] = original_var
    return var_dict   

def _writable_images(image):
    # image = image+1
    # image = image/0.5
    # image =  image*255
    return image.astype(np.uint8) 


def _create_placeholders(image_size):
    image_placeholder = tf.placeholder(shape=[None, image_size,
                                                  image_size, 3], dtype=tf.float32)
    ec_placeholder = tf.placeholder(dtype=tf.bool)
    gn_placeholder = tf.placeholder(dtype=tf.bool)
    ds_placeholder = tf.placeholder(dtype=tf.bool)
    return image_placeholder, ec_placeholder, gn_placeholder, \
           ds_placeholder    

def main(dataset_name,data_root, encoder_root, tfrecord_root, exp_dir, batch_size,
        num_epochs, display_iter, adversarial_weight, pixelwise_weight, 
        reconstruction_weight, learning_rate,  image_root, gpu_id, image_size):
    """
    Training loop
    """
    if not os.path.exists(image_root):
        os.mkdir(os.path.join(os.getcwd(), image_root))

    if dataset_name == 'imagenet':
        data_loader = ImagenetLoader(data_root, tfrecord_root)
    elif dataset_name == 'cifar':
        data_loader = CifarLoader(data_root, tfrecord_root)    
    else:
        data_loader = MSCOCOLoader(data_root, tfrecord_root)
    with tf.Graph().as_default():
        image, _ = data_loader.make_dataset()
        image = tf.image.resize_images(image, [image_size, image_size])
        batch_image_tensor = tf.train.batch([image], batch_size=batch_size)
        image_placeholder, ectrain_placeholder, gntrain_placeholder, \
		dstrain_placeholder  = _create_placeholders(image_size)
        
        pixelwise_loss_tensor, reconstruction_loss_tensor, \
         discriminator_real_loss_tensor,  discriminator_fake_loss_tensor, \
         generator_loss_tensor, discriminator_train_op, \
         generator_train_op, discriminator_global_step_tensor,\
         generator_global_step_tensor, generated_image_tensor = \
			create_model(image_placeholder, adversarial_weight,
                                     pixelwise_weight,  reconstruction_weight,
                                     learning_rate, ectrain_placeholder,
                                     gntrain_placeholder, dstrain_placeholder,
                                     image_size)
        
        encoder_restorer = tf.train.Saver(_map_ops(tf.trainable_variables(), "deepSim/encoder"))        
        def _load_model(current_session):
            encoder_restorer.restore(current_session, encoder_root)
        
        # TODO change the log directory
        supervisor = tf.train.Supervisor(logdir=exp_dir, summary_op=None, init_fn=_load_model)
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        session_config.gpu_options.visible_device_list = gpu_id
        start = time.time()

        with supervisor.managed_session(config=session_config) as session:
	    for i in range(num_epochs):
		batch_image = session.run(batch_image_tensor)     
                _, __, discriminator_global_step, generator_global_step = session.run([generator_train_op,
                                                       discriminator_train_op,
                                                       generator_global_step_tensor,
                                                       generator_global_step_tensor],
                                                       feed_dict={
                                                        image_placeholder: batch_image,
                                                        ectrain_placeholder: False,
                                                        gntrain_placeholder: True,
                                                        dstrain_placeholder: True})
                if ((i+1) % display_iter) == 0:
                    discriminator_fake_loss, discriminator_real_loss,\
                    generator_adversarial_loss, pixelwise_loss,\
                    reconstruction_loss, generated_image = session.run([discriminator_fake_loss_tensor,
                                     		       discriminator_real_loss_tensor,
                                                       generator_loss_tensor,
                                                       pixelwise_loss_tensor,
                                                       reconstruction_loss_tensor,
                                                       generated_image_tensor],
                                                       feed_dict= {
                                                        image_placeholder: batch_image,
                                                        ectrain_placeholder: False,
                                                        gntrain_placeholder: False,
                                                        dstrain_placeholder: False})
                    print ("Iteration ", str(i+1), ": Time: ", time.time()-start)
                    print ("Pixelwise Loss: ", str(pixelwise_loss))
                    print ("Feature Loss: ", str(reconstruction_loss))
                    print ("Discriminator Real Loss: ", str(discriminator_real_loss))
                    print ("Discriminator Fake Loss: ", str(discriminator_fake_loss))
                    print ("Generator Loss: ", str(generator_adversarial_loss))
                    real_image_name = "real_image_" + str(i) + ".jpg"
                    generated_image_name = "generated_image_" + str(i) + ".jpg"
                    batch_uint8_image = _writable_images(batch_image[0])
                    generated_uint8_image = _writable_images(generated_image[0])
                    imageio.imwrite(os.path.join(image_root, real_image_name),
                                                 batch_uint8_image)
                    imageio.imwrite(os.path.join(image_root, generated_image_name),
                                                 generated_uint8_image)  
                    start = time.time()
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetName', required=True,
                        help='name if the dataset (imagenet | cifar | mscoco)')
    parser.add_argument('--dataRoot', required=True,
                        help='root folder of the dataset')
    parser.add_argument('--imageRoot', required=True,
                        help='root folder for all images')
    parser.add_argument('--tfRecordRoot', required=False, default=None,
                        help='root folder for tfrecord')
    parser.add_argument('--encoderRoot', default=None,
                        help='checkpoint for pretrained encoder')
    parser.add_argument('--expDir', required=True,
                        help='directory where model checkpoints are stored')
    parser.add_argument('--batchSize', default=64, type=int,
                        help='batch size used for training')
    parser.add_argument('--numIters', default=20000, type=int,
                        help='number of epochs')
    parser.add_argument('--display', default=50, type=int,
                        help='display after this iteration')
    parser.add_argument('--adversarialWeight', default=100., type=float,
                        help='weight associated with advesarial loss')
    parser.add_argument('--reconstructionWeight', default=0.001, type=float,
                        help='weight associated with reconstruction loss')
    parser.add_argument('--pixelwiseWeight', default=0.000001, type=float,
                        help='weight associated with pixelwise loss')
    parser.add_argument('--lr', default=0.0002, type=float,
                        help='learning rate of encoder')
    parser.add_argument('--gpu_id', default="0", help='gpu id used for training')
    parser.add_argument('--imageSize', default=64, type=int,
                        help="size of the image generated")
    cmd_opt = parser.parse_args()
    print (cmd_opt)
    main(dataset_name= cmd_opt.datasetName, data_root=cmd_opt.dataRoot,
         encoder_root=cmd_opt.encoderRoot, tfrecord_root=cmd_opt.tfRecordRoot,
         exp_dir=cmd_opt.expDir, batch_size=cmd_opt.batchSize, 
         num_epochs=cmd_opt.numIters, display_iter=cmd_opt.display,
         adversarial_weight=cmd_opt.adversarialWeight,
         pixelwise_weight=cmd_opt.pixelwiseWeight,
         reconstruction_weight=cmd_opt.reconstructionWeight,
         learning_rate=cmd_opt.lr, gpu_id=cmd_opt.gpu_id,
         image_root=cmd_opt.imageRoot, image_size=cmd_opt.imageSize)
