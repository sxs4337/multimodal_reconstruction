from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import pickle as pkl
from tensorflow.contrib import slim
import argparse
from train import train
from test import test

def main(cmd_arg):
    np.random.seed(2000)
    with open(cmd_arg.pickleRoot, 'rb') as f:
        data = pkl.load(f)
    with open(cmd_arg.wordGloveVectors, 'rb') as f:
        gloveVectors = pkl.load(f) 
    if cmd_arg.train:
        train(cmd_arg, data, gloveVectors)
    else:
        test(cmd_arg, data, gloveVectors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickleRoot', required=True,\
                         help='dataset pickle file')
    parser.add_argument('--experimentDirectory', required=True,\
                        help='experiment directory')
    parser.add_argument('--batchSize', type=int, default=200,\
                        help='Batch Size')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='train', action='store_false')
    parser.add_argument('--numberLayers', type=int, default=1, help='Number of layers')
    parser.add_argument('--relu', dest='activation', const=1, action='store_const',\
                       help='relu activation')
    parser.add_argument('--sigmoid', dest='activation', const=2, action='store_const',\
                       help='sigmoid activation')
    parser.add_argument('--tanh', dest='activation', const=3, action='store_const', \
                       help='tanh activation')
    parser.add_argument('--margin', type=float, default=1, help='margin of the loss')
    parser.add_argument('--embedSize', type=int, default=4096, help='Embedding Size')
    parser.add_argument('--embedNoise', type=float, default=0.0, help='Embedding noise for training')
    parser.add_argument('--totalIteration', type=int, default=1000, \
                        help='total number of iteration')
    parser.add_argument('--saveIteration', type=int, default=100, help='saver iteration')
    parser.add_argument('--negativeWeight', type=float, default=1.0, help='weight \
                         associated with negative aspect of loss')
    parser.add_argument('--positiveWeight', type=float, default=0.001, help='weight \
                         associated with positive aspect of loss')
    parser.add_argument('--wordGloveVectors', required=True,\
                        help='Glove Vectot file path')                         
    parser.set_defaults(activation=0)
    parser.set_defaults(train=True)
    main(parser.parse_args())
  
