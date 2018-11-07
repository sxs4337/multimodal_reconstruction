import numpy as np
import tensorflow as tf

def pairwise_distance(input_tensor1, input_tensor2, labels, margin):
    input_tensor1 = tf.expand_dims(input_tensor1, 1)
    input_tensor2 = tf.expand_dims(input_tensor2, 1)
    d_sq = tf.reduce_sum(tf.square(input_tensor1 - tf.transpose(input_tensor2, (1, 0, 2))), \
                      2, keep_dims=False)
    distances = d_sq # tf.sqrt(d_sq + 1e-8)
    # expanded_a = tf.expand_dims(a, 1)
    # expanded_b = tf.expand_dims(a, 0)
    # distances = tf.reduce_sum(tf.squared_difference(expanded_a, expanded_b), 2)
    positive_distance = tf.reduce_mean(tf.multiply(distances, labels))
    negative_distance = tf.log(tf.reduce_sum(tf.exp(tf.multiply((margin-distances),(1.0-labels)))) + 1e-10)
    return positive_distance, negative_distance, distances, labels

def activation_inversion(input_tensor, activation_id):
    if activation_id == 0: # no activation
        return input_tensor
    elif activation_id == 1: # leaky relu
        return lrelu(input_tensor, 5.0)
    elif activation_id == 2: # sigmoid
        return tf.log(input_tensor+1e-10) - tf.log(1-input_tensor+1e-10)
    elif activation_id == 3: # tanh
        return tf.multiply(0.5, (tf.log(1.0+input_tensor+1e-10) - tf.log(1.0-input_tensor+1e-10)))
    else:
        raise NotImplementedError

def inversion(embedding, weights, biases, activation_id):
    for i in range(len(weights)-1, -1, -1):
        embedding = activation_inversion(embedding, activation_id)
        embedding = tf.subtract(embedding, biases[i])
        embedding = tf.matmul(embedding, weights[i], transpose_b=True)
    return embedding

def distance(logit_one, logit_two, metric = 'euc'):
    if metric == 'euc':
        return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square( \
            tf.subtract(logit_one, logit_two)), 1, keep_dims=True)))
    elif metric == 'cos':
        return tf.losses.cosine_distance(tf.nn.l2_normalize(logit_one, 1), \
            tf.nn.l2_normalize(logit_two, 1), dim = 1)
    
def create_dataset(sample_data):
    image_batches = []
    caption_batches = []
    word_batches = []
    key_batches = []
    for i in range(0, len(sample_data.values())):
        current_key = sample_data.keys()[i]
        current_data = sample_data.values()[i]
        image_batch = np.array(current_data[3], copy=True)
        image_batch = np.expand_dims(image_batch, axis=0)
        word_batch = current_data[1]
        caption_batch = np.array(current_data[4][:5], copy=True)
        caption_batch = np.expand_dims(caption_batch, axis=0)
        image_batches.extend(image_batch)
        caption_batches.extend(caption_batch)
        key_batches.extend([current_key])
        word_batches.extend([word_batch])
    return np.squeeze(image_batches), np.array(caption_batches), \
           np.squeeze(key_batches), word_batches

def create_label_matrix(input_labels):
    label_matrix = np.zeros([len(input_labels), len(input_labels)])
    for idx, label in enumerate(input_labels):
        label_matrix[idx, [i for i, x in enumerate(input_labels) if x == label]] = 1
    # label_matrix = label_matrix - np.identity(len(input_labels))
    return label_matrix
    
def create_batch(images, captions, words, keys, indices, batch_size, gloveVectors):
    batch_indices = np.random.choice(indices.shape[0], batch_size, replace=False)
    batch_image = np.squeeze(images[indices[batch_indices], np.random.randint(0, 2), :])
    # batch_image = images[indices[batch_indices], :]
    batch_caption = np.squeeze(captions[indices[batch_indices], np.random.randint(0, 5), :])
    batch_word = np.array([gloveVectors[words[x]] for x in indices[batch_indices]])
    label_matrix = create_label_matrix([words[x] for x in indices[batch_indices]])
    batch_keys = keys[indices[batch_indices]]
    new_indices = np.delete(indices, batch_indices)
    return batch_image, batch_caption, batch_word, batch_keys, new_indices, label_matrix

def lrelu(x, alpha = 0.2):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def activation_fn(activation_id):
    if activation_id == 0:
        return None
    elif activation_id == 1:
        return lrelu
    elif activation_id == 2:
        return tf.nn.sigmoid
    elif activation_id == 3:
        return tf.nn.tanh
    else:
       return NotImplementedError

def scale_data(logits, max=1.0, min=-1.0):
    logits = tf.divide(tf.subtract(logits, min), tf.subtract(max, min))
    return logits #tf.subtract(tf.multiply(logits, 2.0) , 1.0)
    
def rescale_data(logits, max=1.0, min=-1.0):
    # logits = np.divide(np.add(logits, 1.0),2.0)
    return np.multiply(logits, np.subtract(max, min)) + min

def rescale_data_tf(logits, max=1.0, min=-1.0):
    # logits = np.divide(np.add(logits, 1.0),2.0)
    return tf.multiply(logits, tf.subtract(max, min)) + min