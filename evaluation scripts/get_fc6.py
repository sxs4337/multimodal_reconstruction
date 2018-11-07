import pickle as pkl
import pdb
import json, os, sys
import numpy as np
from PIL import Image   
from random import shuffle
import cv2
from scipy.misc import imsave
from random import randint

import caffe

caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()
#caffe.set_mode_cpu()
model_def = 'caffenet.prototxt'
model_weights = 'bvlc_reference_caffenet.caffemodel'
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
mu = np.load('mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
data_blob_shape = net.blobs['data'].data.shape
data_blob_shape = list(data_blob_shape)
net.blobs['data'].reshape(1, data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])

batch_feat = []
files = open(sys.argv[1],'r').readlines()
for file in files:
    image_path = file.split(' ')[0].strip()
    image_orig = caffe.io.load_image(image_path)
    image = cv2.resize(image_orig, (256, 256))
    width, height,channels = image_orig.shape   # Get dimensions
    left = (width - 227)/2
    top = (height - 227)/2
    right = (width + 227)/2
    bottom = (height + 227)/2
    cImg = image_orig[left:right, top:bottom, :]
    transformed_image = transformer.preprocess('data', cImg)
    batch_image = np.transpose(transformed_image[...,np.newaxis],(3,0,1,2))
    net.blobs['data'].data[...] = batch_image
    output = net.forward()
    feat = net.blobs['fc6'].data
    batch_feat.append(feat.copy())

np.save(sys.argv[2],np.squeeze(np.array(batch_feat,copy=True)))
