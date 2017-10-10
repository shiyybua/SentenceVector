# -*- coding: utf-8 -*

import tensorflow as tf
import utils
import config
import numpy as np

FLAG = config.FLAGS


def _single_cell(unit_type, num_units, dropout,
                mode, forget_bias=1.0, residual_connection=False, device_str=None):
    """Create an instance of a single RNN cell."""
    # dropout (= 1 - keep_prob) is set to 0 during eval and infer
    dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

    # Cell Type
    if unit_type == "lstm":
        utils.print_out("  LSTM, forget_bias=%g" % forget_bias, new_line=False)
        single_cell = tf.contrib.rnn.BasicLSTMCell(
            num_units,
            forget_bias=forget_bias)
    elif unit_type == "gru":
        utils.print_out("  GRU", new_line=False)
        single_cell = tf.contrib.rnn.GRUCell(num_units)
    elif unit_type == "layer_norm_lstm":
        utils.print_out("  Layer Normalized LSTM, forget_bias=%g" % forget_bias,
                        new_line=False)
        single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
            num_units,
            forget_bias=forget_bias,
            layer_norm=True)
    else:
        raise ValueError("Unknown unit type %s!" % unit_type)

    # Dropout (= 1 - keep_prob)
    if dropout > 0.0:
        single_cell = tf.contrib.rnn.DropoutWrapper(
            cell=single_cell, input_keep_prob=(1.0 - dropout))
        utils.print_out("  %s, dropout=%g " % (type(single_cell).__name__, dropout),
                        new_line=False)

    # Residual
    if residual_connection:
        single_cell = tf.contrib.rnn.ResidualWrapper(single_cell)
        utils.print_out("  %s" % type(single_cell).__name__, new_line=False)

    # Device Wrapper
    if device_str:
        single_cell = tf.contrib.rnn.DeviceWrapper(single_cell, device_str)
        utils.print_out("  %s, device=%s" %
                        (type(single_cell).__name__, device_str), new_line=False)

    return single_cell


def cell_list(mode, num_layer, num_units):
    unit_type, dropout = FLAG.unit_type, FLAG.dropout
    cell = _single_cell(unit_type, num_units, dropout, mode)
    cells = [cell for _ in range(num_layer)]
    return cells


def create_cells(mode, num_layer, num_units):
    assert num_layer > 0
    unit_type, dropout = FLAG.unit_type, FLAG.dropout
    if num_layer == 1:
        return _single_cell(unit_type, num_units, dropout, mode)
    else:  # Multi layers
        return tf.contrib.rnn.MultiRNNCell(cell_list(mode, num_layer, num_units))


def create_attention_mechanism(attention_option, num_units, memory,
                               source_sequence_length):
  """Create attention mechanism based on the attention_option."""

  # Mechanism
  if attention_option == "luong":
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units, memory, memory_sequence_length=source_sequence_length)
  elif attention_option == "scaled_luong":
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units,
        memory,
        memory_sequence_length=source_sequence_length,
        scale=True)
  elif attention_option == "bahdanau":
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        num_units, memory, memory_sequence_length=source_sequence_length)
  elif attention_option == "normed_bahdanau":
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        num_units,
        memory,
        memory_sequence_length=source_sequence_length,
        normalize=True)
  else:
    raise ValueError("Unknown attention option %s" % attention_option)

  return attention_mechanism


def embedding_initializer(vocab_size):
    '''
    :param vocab_size: 不重复的单词数，不包括known，padding
    :return: 
    '''
    embedding_size = FLAG.embeddings_size
    embeddings = np.random.uniform(-1, 1, (vocab_size + 2, embedding_size))
    embeddings = np.asarray(embeddings, dtype=np.float32)
    return embeddings
