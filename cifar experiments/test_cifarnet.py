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
num_examples = 10000


def main(cifar_path, log_dir, batch_size, zero_shot):
    if zero_shot:
        num_classes = 8
    else:
        num_classes = 10
    pkl_root = os.path.join(cifar_path, "cifar10_classWord2vec.pkl")
    loader = CifarLoader(cifar_path, pkl_root, zero_shot=zero_shot)
    with tf.Graph().as_default():
        image_placeholder = tf.placeholder(shape=[None, 64, 64, 3], dtype=tf.float32)
        groundtruth_placeholder = tf.placeholder(shape=[None], dtype=tf.int32)
        is_training_placeholder = tf.placeholder(dtype=tf.bool)
        lr_placeholder = tf.placeholder(dtype=tf.float32)
        with slim.arg_scope(cifarnet_arg_scope()):
            logits_tensor, end_points = CifarNet(image_placeholder, 
                                          num_classes=num_classes,
                                          is_training=is_training_placeholder,
                                          prediction_fn=None)
        
        test_image_tensor, _, _, test_label_tensor = loader.make_dataset(is_train=False, 
                                                                           is_valid=True)
        test_image_tensor = tf.image.resize_images(test_image_tensor, [64, 64])
        test_image_tensor, test_label_tensor = tf.train.batch([test_image_tensor,
                                                   test_label_tensor], batch_size)
        logits_tensor = tf.argmax(logits_tensor, axis=1)
        accuracy_tensor, update_op = tf.contrib.metrics.streaming_accuracy(logits_tensor,
                                                             groundtruth_placeholder)
        saver = tf.train.Saver()
        def _init_fn(session):
            saver.restore(session, tf.train.latest_checkpoint(log_dir))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        supervisor = tf.train.Supervisor(logdir=None, summary_op=None, init_fn=_init_fn)
        with supervisor.managed_session(config=config) as session:
            num_iters = num_examples // batch_size + 1        
            for i in range(num_iters):
                test_images, test_labels = \
                         session.run([test_image_tensor, test_label_tensor])
                _, logits = session.run([update_op, logits_tensor],
                                        feed_dict={image_placeholder:test_images,\
                                                      groundtruth_placeholder: test_labels,\
                                                      is_training_placeholder:False})
                
            accuracy = session.run(accuracy_tensor)
            print ("Accuracy: ", accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/shared/kgcoe-research/mil/cifar',
                        help="dataset directory")
    parser.add_argument('--log_dir', required=True,  help="log directory")
    parser.add_argument('--batch_size', default=64, type=int, help="batch size")
    parser.add_argument('--zeroshot', dest='zero_shot', action='store_true')
    parser.set_defaults(zero_shot=False)
    cmd_opt = parser.parse_args()
    print(cmd_opt)
    main(log_dir=cmd_opt.log_dir, batch_size=cmd_opt.batch_size, 
         zero_shot=cmd_opt.zero_shot, cifar_path=cmd_opt.data_dir) 
