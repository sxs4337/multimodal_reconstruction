from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from data.cifar_loader import CifarLoader
from model.cifar_net import CifarNet, cifarnet_arg_scope

categories = {"airplane": 0, "automobile": 1, "bird": 2, 
              "deer": 3, "dog": 4, "frog": 5, "horse": 6, 
              "ship": 7, "cat": 8, "truck": 9}
              
excluded_categories = {"cat": 8, "truck": 9}              
num_training_examples = 40000
num_validation_examples = 10000

def main(cifar_path, num_classes, log_dir, batch_size, num_epochs, 
         learning_rate, zero_shot):
    if zero_shot:
        num_classes = 8
    else:
        num_classes = 10
    pkl_root = os.path.join(cifar_path, "cifar10_classWord2vec.pkl")
    loader = CifarLoader(cifar_path, pkl_root, zero_shot=zero_shot)
    with tf.Graph().as_default():
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        image_placeholder = tf.placeholder(shape=[None, 64, 64, 3], dtype=tf.float32)
        groundtruth_placeholder = tf.placeholder(shape=[None, num_classes], dtype=tf.int32)
        is_training_placeholder = tf.placeholder(dtype=tf.bool)
        lr_placeholder = tf.placeholder(dtype=tf.float32)
        with slim.arg_scope(cifarnet_arg_scope()):
            logits_tensor, end_points = CifarNet(image_placeholder, 
                                      num_classes=num_classes,
                                      is_training=is_training_placeholder,
                                      prediction_fn=None)
        loss_tensor = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                     labels=groundtruth_placeholder, 
                                     logits=logits_tensor))
        #tf.summary.scalar("loss", loss_tensor)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=lr_placeholder)
            train_op = optimizer.minimize(loss_tensor, global_step=global_step)
        image_tensor, _, _, label_tensor = loader.make_dataset()
        valid_image_tensor, _, _, valid_label_tensor = loader.make_dataset(is_train=False, 
                                                                           is_valid=True)
        image_tensor = tf.image.resize_images(image_tensor, [64, 64])
        valid_image_tensor = tf.image.resize_images(valid_image_tensor, [64, 64])
        image_tensor, label_tensor = tf.train.batch([image_tensor, label_tensor],
                                                     batch_size)
        valid_image_tensor, valid_label_tensor = tf.train.batch([valid_image_tensor,
                                                   valid_label_tensor], batch_size)
        summary_op = tf.summary.merge_all()
        file_writer = tf.summary.FileWriter(logdir=os.path.join(log_dir, "train"))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        supervisor = tf.train.Supervisor(logdir=log_dir, summary_op=None,
                                             global_step=global_step)
        with supervisor.managed_session(config=config) as session:
            gb_init = session.run(global_step)
            current_epoch = (gb_init*batch_size) // num_training_examples
            for i in range(current_epoch, num_epochs):
                num_iter = num_training_examples // batch_size + 1
                running_loss = 0
                for current_iter in range(num_iter):
                    images, labels = session.run([image_tensor, label_tensor])
                    label_onehot = np.zeros([batch_size, num_classes])
                    label_onehot[np.arange(batch_size), labels] = 1
                    loss, _, summary, gb= session.run([loss_tensor, train_op, 
                                                       summary_op, global_step], 
                                                       feed_dict= {image_placeholder:images,
                                                        groundtruth_placeholder: label_onehot,
                                                        is_training_placeholder: True,
                                                        lr_placeholder: learning_rate})
                    running_loss += loss
                running_loss = running_loss / num_iter
                print ("Average Loss at ", i+1 ,": ",   running_loss)                
                num_iter = num_validation_examples // batch_size + 1
                running_accuracy = 0
                for current_iter in range(num_iter):
                     valid_images, valid_labels = session.run([valid_image_tensor, 
                                                          valid_label_tensor])
                     logits = session.run(logits_tensor,
                                      feed_dict={image_placeholder:valid_images,
                                                 is_training_placeholder:False})
                     train_logits = session.run(logits_tensor, 
                                      feed_dict={image_placeholder: images,
                                                 is_training_placeholder:False})
                     
                     running_accuracy += np.sum(np.argmax(logits, axis=1)==valid_labels)
                print ("Validation Accuracy: ", (running_accuracy/(num_iter*batch_size))) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/shared/kgcoe-research/mil/cifar',
                        required=True, help="dataset directory")
    parser.add_argument('--log_dir', required=True,  help="log directory")
    parser.add_argument('--batch_size', default=64, type=int, help="batch size")
    parser.add_argument('--num_epochs', default=4, type=int, 
                       help="number of iterations")
    parser.add_argument('--learning_rate', default=0.01, type=float, 
                       help='learning rate')
    parser.add_argument('--num_classes', default=10, type=int, 
			help='number of classes in the training set')
    parser.add_argument('--zeroshot', dest='zero_shot', action='store_true')
    parser.set_defaults(zero_shot=False)
    cmd_opt = parser.parse_args()
    print(cmd_opt)
    main(log_dir=cmd_opt.log_dir, batch_size=cmd_opt.batch_size, 
         num_epochs=cmd_opt.num_epochs, learning_rate=cmd_opt.learning_rate,
         zero_shot=cmd_opt.zero_shot, cifar_path=cmd_opt.data_dir, 
         num_classes=cmd_opt.num_classes)
