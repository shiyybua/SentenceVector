# -*- coding: utf-8 -*-
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

tf.app.flags.DEFINE_string("src_file", 'resource/source_mini.txt', "Training data.")
tf.app.flags.DEFINE_string("src_before_file", 'resource/source_before_mini.txt', "Training data.")
tf.app.flags.DEFINE_string("src_after_file", 'resource/source_after_mini.txt', "Training data.")
tf.app.flags.DEFINE_string("src_predict_file", 'resource/predict.txt', "Training data.")
tf.app.flags.DEFINE_string("model_path", 'resource/model/', "Training data.")
tf.app.flags.DEFINE_string("word_embedding_file", None, "Training data.")

tf.app.flags.DEFINE_string("src_vocab_file", 'resource/source_vocab.txt', "source vocabulary.")
tf.app.flags.DEFINE_string("attention_type", 'bahdanau', "attention.")

tf.app.flags.DEFINE_string("unit_type", 'lstm', "the type of rnn cell, lstm|gru")
tf.app.flags.DEFINE_integer("embeddings_size", 300, "Size of word embedding.")
# tf.app.flags.DEFINE_integer("num_units", 300, "the number of cell unit")
tf.app.flags.DEFINE_integer("max_sequence", 100, "the number of cell unit")
tf.app.flags.DEFINE_integer("batch_size", 2, "the number of cell unit")
tf.app.flags.DEFINE_integer("num_layer", 2, "the number of cell unit")
tf.app.flags.DEFINE_float("dropout", 0.6, "drop out")

tf.app.flags.DEFINE_integer("epoch", 2000000, "epoch")

FLAGS = tf.app.flags.FLAGS

