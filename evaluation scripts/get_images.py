import os, sys
import pdb
import numpy as np
import caffe
import scipy.misc
import cv2

def normalize(img, out_range=(0.,1.), in_range=None):
    if not in_range:
        min_val = np.min(img)
        max_val = np.max(img)
    else:
        min_val = in_range[0]
        max_val = in_range[1]

    result = np.copy(img)
    result[result > max_val] = max_val
    result[result < min_val] = min_val
    result = (result - min_val) / (max_val - min_val) * (out_range[1] - out_range[0]) + out_range[0]
    return result

def deprocess(images, out_range=(0.,1.), in_range=[-120.0,120.0]):
    num = images.shape[0]
    c = images.shape[1]
    ih = images.shape[2]
    iw = images.shape[3]

    result = np.zeros((ih, iw, 3))

    # Normalize before saving
    result[:] = images[0].copy().transpose((1,2,0))
    result = normalize(result, out_range, in_range)
    return result

caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()
model_def = 'generator.prototxt'
model_weights = 'generator.caffemodel'
image_generator = caffe.Net(model_def, model_weights, caffe.TEST)
blob = 'deconv0'

if not os.path.exists(sys.argv[2]):
    os.makedirs(sys.argv[2])

# load input data
input = np.load(sys.argv[1])
no_inputs = input.shape[0]

for i in range(no_inputs):
    image_generator.blobs["feat"].data[:] = input[i,:]
    image_generator.forward()
    # image = np.transpose(image_generator.blobs[blob].data.squeeze(),(1,2,0))
    # scipy.misc.imsave(os.path.join(sys.argv[2],str(i+1)+'.jpg'), image[:,:,[2,1,0]])
    generated = image_generator.blobs[blob].data
    img = generated[:,::-1,:,:] # Convert from BGR to RGB
    output_img = deprocess(img, in_range = (np.min(img), np.max(img)))
    scipy.misc.imsave(os.path.join(sys.argv[2],str(i+1)+'.png'), output_img)
