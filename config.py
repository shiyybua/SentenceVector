# -*- coding: utf-8 -*-
import tensorflow as tf

tf.app.flags.DEFINE_string("unit_type", 'lstm', "the type of rnn cell, lstm|gru")
tf.app.flags.DEFINE_integer("embeddings_size", 300, "Size of word embedding.")
tf.app.flags.DEFINE_integer("num_units", 1024, "the number of cell unit")
tf.app.flags.DEFINE_float("dropout", 0.6, "drop out")

FLAGS = tf.app.flags.FLAGS

