import tensorflow as tf
from tensorflow.contrib import slim

def _embedding_model(input_tensor, \
                     num_layers, \
                     embedding_dimension, \
                     scope):
    return slim.repeat(input_tensor, num_layers, slim.fully_connected, \
                       embedding_dimension, scope=scope)


def model(image, caption, word, embedding_size, number_layer, activation_fn, \
          image_scope, caption_scope, word_scope):
     with slim.arg_scope([slim.fully_connected], \
                         activation_fn=activation_fn, \
                         weights_regularizer=slim.l2_regularizer(0.001)):
         image_embedding = _embedding_model(image, number_layer, \
                                            embedding_size, image_scope)
         caption_embedding = _embedding_model(caption, number_layer, \
                                            embedding_size, caption_scope)
         word_embedding = _embedding_model(word, number_layer, \
                                            embedding_size, word_scope)
     return image_embedding, caption_embedding, word_embedding
