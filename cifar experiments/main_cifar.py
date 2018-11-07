from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pkl
import os
import numpy as np
import tensorflow as tf
from data.cifar_loader import CifarLoader
from config import config

from train import train
from eval import eval


image_size = 64
word2vec_embedding_size = 300
categories = {"airplane": 0, "automobile": 1, "bird": 2, 
              "deer": 3, "dog": 4, "frog": 5, "horse": 6,
              "ship": 7, "cat": 8,  "truck": 9}
              
heldout_categories = {"cat": 8, "truck": 9}

              
def main(cmd_opt):
    if cmd_opt.zero_shot and cmd_opt.train:
        num_classes = 8        
    else:
        num_classes = cmd_opt.numClasses
    if cmd_opt.imageDir is not None and not os.path.exists(cmd_opt.imageDir):
        os.mkdir(cmd_opt.imageDir)
    cifar_loader = CifarLoader(cmd_opt.dataRoot, cmd_opt.pklRoot, 
                                    cmd_opt.tfRecordRoot, 
                                    zero_shot=cmd_opt.zero_shot) 
                                    
    with open(cmd_opt.pklRoot, 'rb') as pickle_file:
            word2vec_dict = pkl.load(pickle_file)
    word2vec_matrix = np.zeros([num_classes, word2vec_embedding_size])
    for key, value in word2vec_dict.iteritems():
        if cmd_opt.zero_shot and cmd_opt.train:
            if key not in heldout_categories.keys():
               word2vec_matrix[categories[key], :] = value	    
        else:   
            word2vec_matrix[categories[key], :] = value
    with tf.Graph().as_default(): 
        if cmd_opt.train:
            train(cmd_opt, word2vec_matrix, cifar_loader, num_classes, 
                  image_size, word2vec_embedding_size, "CifarNet", 
                  cmd_opt.validationExamples)
        else:
            eval(cmd_opt, word2vec_matrix, cifar_loader, num_classes, 
                image_size, word2vec_embedding_size, 
                cmd_opt.testExamples, cmd_opt.imageDir)
        

if __name__ == "__main__":
    cmd_opt = config()
    print (cmd_opt)
    main(cmd_opt)
