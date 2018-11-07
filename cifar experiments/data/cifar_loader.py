from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pkl
import os

import numpy as np
import tensorflow as tf

from data.data_loader import DataLoader

categories = {"airplane": 0, "automobile": 1, "bird": 2, 
              "deer": 3, "dog": 4, "frog": 5, "horse": 6,
              "ship": 7, "cat": 8,  "truck": 9}
              
heldout_categories = {"cat": 8, "truck": 9}


class CifarLoader(DataLoader):
    def __init__(self, data_root, pkl_root, tfrecord_root=None, record_name="cifar", 
                zero_shot=False):
        self.data_root = data_root
        if tfrecord_root is None:
            self.tfrecord_root = data_root
        else:
            self.tfrecord_root = tfrecord_root
        self.record_name = record_name
        if zero_shot:
            self.record_name = record_name + "_zeroshot"
        
        self.pkl_root = pkl_root
        self.zero_shot = zero_shot
        
        
    def _preprocess_image(self, image):
        # TODO add appropriate preprocessing step
        pass

    def _make_example(self, image_path, class_label, vector_embedding):
        img_str = tf.gfile.FastGFile(image_path, "rb").read()
        vector_str = np.array(map(float, vector_embedding), np.float32).tobytes()
        features_dict = {"image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_str])),
                         "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[class_label])),
                         "word2vec": tf.train.Feature(bytes_list=
                                        tf.train.BytesList(value=[vector_str])),
                         "label_num": tf.train.Feature(int64_list=
                                        tf.train.Int64List(value=[categories[class_label]]))}
        example = tf.train.Example(features=tf.train.Features(feature=features_dict))
        return example
        
    def _make_dataset(self, do_shuffle, is_train, is_valid, record_path):
        if is_train:
            label_file_path = os.path.join(self.data_root, "trainLabels.csv")
            partial_image_path = os.path.join(self.data_root, "train")
        elif is_valid:
            label_file_path = os.path.join(self.data_root, "validLabels.csv")
            partial_image_path = os.path.join(self.data_root, "valid")
        else:
            label_file_path = os.path.join(self.data_root, "testLabels.csv")
            partial_image_path = os.path.join(self.data_root, "test")
        all_labels = np.genfromtxt(label_file_path, delimiter=",", dtype=object, skip_header=1)
        if self.zero_shot and (is_train or is_valid):
            for key in heldout_categories.keys():
                all_labels = all_labels[np.where(all_labels[:, 1]!=key)]
        data_writer = tf.python_io.TFRecordWriter(record_path)
        with open(self.pkl_root, 'rb') as pickle_file:
            word2vec_dict = pkl.load(pickle_file)
        if do_shuffle:
            np.random.shuffle(all_labels)
        for i in range(all_labels.shape[0]):
            image_name, label = map(str, all_labels[i, :])
            image_path = os.path.join(partial_image_path, image_name + ".png")
            word_vec = word2vec_dict[label]
            tf_example = self._make_example(image_path, label, word_vec)
            data_writer.write(tf_example.SerializeToString())
            print ("Written:", str(i+1), "/", str(all_labels.shape[0]))
        
    def _tfrecord_decoder(self, record_path):
        filename_queue = tf.train.string_input_producer([record_path])
        data_reader = tf.TFRecordReader()
        _, serialized_example = data_reader.read(filename_queue)
        example = tf.parse_single_example(serialized_example, 
                                        features={
                                        "image": tf.FixedLenFeature([], tf.string),
                                        "label": tf.FixedLenFeature([], tf.string),
                                        "word2vec": tf.FixedLenFeature([], tf.string),
                                        "label_num": tf.FixedLenFeature([], tf.int64)    
                                        })
        image = tf.image.decode_png(example["image"], channels=3)
        label = tf.cast(example["label"], tf.string)
        vector = tf.decode_raw(example["word2vec"], tf.float32)
        label_num = tf.cast(example["label_num"], tf.int64)
        return image, label, vector, label_num
        
    def make_dataset(self, do_shuffle=True, is_train=True, is_valid=False):
        if is_train:
            record_path = os.path.join(self.tfrecord_root, self.record_name \
                                + "_train.tfrecord")            
        elif is_valid:
            record_path = os.path.join(self.tfrecord_root, self.record_name \
                                + "_valid.tfrecord")
        else:
            record_path = os.path.join(self.tfrecord_root, self.record_name \
                                + "_test.tfrecord")
        if not os.path.exists(record_path):
            self._make_dataset(do_shuffle, is_train, is_valid, record_path)
        return self._tfrecord_decoder(record_path)
        
            
            
