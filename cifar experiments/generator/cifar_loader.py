from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
import cPickle as pkl
from data_loader import DataLoader

categories = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3,
              "deer": 4, "dog": 5, "frog": 6, "horse": 7,
              "ship": 8, "truck": 9}


class CifarLoader(DataLoader):
    def __init__(self, data_root, tfrecord_root=None, record_name="cifar_gen"):
        self.data_root = data_root
        if tfrecord_root is None:
            self.tfrecord_root = data_root
        else:
            self.tfrecord_root = tfrecord_root
        self.record_name = record_name
        
    def _make_example(self, image_path, class_label):
        img_str = tf.gfile.FastGFile(image_path, "rb").read()
        features_dict = {"image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_str])),
                         "label_num": tf.train.Feature(int64_list=
                                        tf.train.Int64List(value=[categories[class_label]]))}
        example = tf.train.Example(features=tf.train.Features(feature=features_dict))
        return example
        
    def _make_dataset(self, do_shuffle, is_train, record_path):
        if is_train:
            label_file_path = os.path.join(self.data_root, "trainLabels.csv")
            partial_image_path = os.path.join(self.data_root, "train")
        else:
            label_file_path = os.path.join(self.data_root, "validLabels.csv")
            partial_image_path = os.path.join(self.data_root, "valid")
        all_labels = np.genfromtxt(label_file_path, delimiter=",", dtype=object,
                                    skip_header=1)
        data_writer = tf.python_io.TFRecordWriter(record_path)
        if do_shuffle:
            np.random.shuffle(all_labels)
        for i in range(all_labels.shape[0]):
            image_name, label = map(str, all_labels[i, :])
            image_path = os.path.join(partial_image_path, image_name + ".png")     
            tf_example = self._make_example(image_path, label)
            data_writer.write(tf_example.SerializeToString())
            print ("Written:", str(i+1), "/", str(all_labels.shape[0]))
        
    def _tfrecord_decoder(self, record_path):
        filename_queue = tf.train.string_input_producer([record_path])
        data_reader = tf.TFRecordReader()
        _, serialized_example = data_reader.read(filename_queue)
        example = tf.parse_single_example(serialized_example, 
                                        features={
                                        "image": tf.FixedLenFeature([], tf.string),
                                        "label_num": tf.FixedLenFeature([], tf.int64)    
                                        })
        image = tf.image.decode_png(example["image"], channels=3)
        #image = self._preprocess_image(image)        
        label_num = tf.cast(example["label_num"], tf.int64)
        return image, label_num
        
    def make_dataset(self, do_shuffle=True, is_train=True):
        if is_train:
            record_path = os.path.join(self.tfrecord_root, self.record_name \
                                + "_train.tfrecord")            
        else:
            record_path = os.path.join(self.tfrecord_root, self.record_name \
                                + "_valid.tfrecord")
        if not os.path.exists(record_path):
            self._make_dataset(do_shuffle, is_train, record_path)
        return self._tfrecord_decoder(record_path)
        
            
            
