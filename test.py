# -*- coding: utf-8 -*-

from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.seq2seq import tile_batch
from tensorflow.contrib.seq2seq import AttentionWrapper

from tensorflow.contrib.seq2seq import BahdanauAttention