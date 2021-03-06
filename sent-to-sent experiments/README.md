## Sentence-to-sentence Model

This work is based on easy_seq2seq. Original code can be found [here](https://github.com/suriyadeepan/easy_seq2seq).

Create directory
```
mkdir log_dir
mkdir model

Note that some characters in the original data is not in the right codec. We are using our own dataset.

All sentences pairs were extracted from MSCOCO + Flickr30k + MSR-VTT + MSVD.
Additional files that are required to start training are the glove embeddings for all the words in the dictionary

### Training

Edit *seq2seq.ini* file to set *mode = train*. To use pre-trained embedding, set *pretrained_embedding = true*
```
python execute.py
```
Note: Set *trainable=True* in *embedding = vs.get_variable(...)* (line 762) in *embedding/rnn_cell.py* to enable training on pre-trained embedding.
Note: To assgin a GPU device, use
```
export CUDA_VISIBLE_DEVICES="0"
```

### Inference

Edit *seq2seq.ini* file to set *mode = test*.
Edit *seq2seq.ini* file to set *mode = generate* to generate paraphrasing sentences, given a file containing multiple input sentences.
```
python execute.py
```

### TensorBoard

Run a TensorBoard server in a separate process for real-time monitoring of training progress and evaluation metrics.
```
tensorboard --logdir=log_dir/ --port=6364
```
