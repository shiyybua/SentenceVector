# -*- coding: utf-8 -*

import utils
from model_helper import *
from tensorflow.python.layers import core as layers_core
from tensorflow.python.util import nest


# TODO: time_major
class Model():
    def __init__(self, mode=tf.contrib.learn.ModeKeys.TRAIN):
        self.iterator = utils.get_iterator(utils.src_vocab_table,
                                           utils.vocab_size, utils.batch_size)
        self.mode = mode
        self.source = self.iterator.source
        self.source_length = self.iterator.source_length
        self.source_before = self.iterator.source_before
        self.source_before_length = self.iterator.source_before_length
        self.embedding_weights = \
            tf.get_variable("embeddings", dtype=tf.float32,
                            initializer=embedding_initializer(utils.vocab_size))

    def _build_encoder(self):
        '''
        
        :return: encoder_outputs : (?,?,2*unit_num)
        '''
        num_layer = int(utils.num_layer/1)
        cell_forward = create_cells(self.mode, num_layer, utils.embeddings_size)
        cell_backward = create_cells(self.mode, num_layer, utils.embeddings_size)
        source_embedding = tf.nn.embedding_lookup(self.embedding_weights, self.source)
        encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
            cell_forward, cell_backward, source_embedding, dtype=tf.float32,
            sequence_length=self.source_length)
        encoder_outputs = tf.concat(encoder_outputs, axis=-1)
        # model.py: 537

        encoder_state = []
        for layer_id in range(num_layer):
            encoder_state.append(bi_encoder_state[0][layer_id])  # forward
            encoder_state.append(bi_encoder_state[1][layer_id])  # backward
        encoder_state = tuple(encoder_state)


        return encoder_outputs, encoder_state

    def _build_decoder(self, encoder_outputs, encoder_state):
        '''
        :param encoder_outputs: 默认(time_step, batch_size, 2 * num_units)
        :param encoder_state: 
        :return: 
        '''
        alignment_history = self.mode == tf.contrib.learn.ModeKeys.INFER
        # TODO: check the shape
        encoder_outputs = tf.transpose(encoder_outputs, [1, 0, 2])
        print encoder_outputs
        # encoder_outputs should be batch-major
        attention_mechanism = create_attention_mechanism(
            FLAG.attention_type, FLAG.embeddings_size,
            encoder_outputs, self.source_length)
        cell = cell_list(self.mode, utils.num_layer, FLAG.embeddings_size)
        attention_cell = cell.pop(0)
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(
            attention_cell,
            attention_mechanism,
            attention_layer_size=None,  # don't use attenton layer.
            output_attention=False,
            alignment_history=alignment_history)

        cell = GNMTAttentionMultiCell(
            attention_cell, cell)

        # cell_backward = create_cells(self.model, utils.num_layer)
        # cell_backward = tf.contrib.seq2seq.AttentionWrapper(
        #     cell_backward,
        #     attention_mechanism,
        #     attention_layer_size=FLAG.embeddings_size * 2,
        #     alignment_history=alignment_history)

        decoder_initial_state = tuple(
            zs.clone(cell_state=es)
            if isinstance(zs, tf.contrib.seq2seq.AttentionWrapperState) else es
            for zs, es in zip(
                cell.zero_state(FLAG.batch_size, tf.float32), encoder_state))
        source_before_embedding = tf.nn.embedding_lookup(
            self.embedding_weights, self.source_before)
        helper = tf.contrib.seq2seq.TrainingHelper(
            source_before_embedding, self.source_before_length)
        my_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell,
            helper,
            decoder_initial_state, )
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder, swap_memory=True)

        sample_id = outputs.sample_id
        # 所有的地方的vocab_size都没有算padding, known
        self.output_layer = layers_core.Dense(
            utils.vocab_size+2, use_bias=False, name="output_projection")
        logits = self.output_layer(outputs.rnn_output)

        return logits, sample_id, final_context_state

    def train(self):
        out, state = self._build_encoder()
        self._build_decoder(out, state)
        # print state
        # print len(state)


class GNMTAttentionMultiCell(tf.nn.rnn_cell.MultiRNNCell):
  """A MultiCell with GNMT attention style."""

  def __init__(self, attention_cell, cells, use_new_attention=False):
    """Creates a GNMTAttentionMultiCell.

    Args:
      attention_cell: An instance of AttentionWrapper.
      cells: A list of RNNCell wrapped with AttentionInputWrapper.
      use_new_attention: Whether to use the attention generated from current
        step bottom layer's output. Default is False.
    """
    cells = [attention_cell] + cells
    self.use_new_attention = use_new_attention
    super(GNMTAttentionMultiCell, self).__init__(cells, state_is_tuple=True)

  def __call__(self, inputs, state, scope=None):
    """Run the cell with bottom layer's attention copied to all upper layers."""
    if not nest.is_sequence(state):
      raise ValueError(
          "Expected state to be a tuple of length %d, but received: %s"
          % (len(self.state_size), state))

    with tf.variable_scope(scope or "multi_rnn_cell"):
      new_states = []

      with tf.variable_scope("cell_0_attention"):
        attention_cell = self._cells[0]
        attention_state = state[0]
        cur_inp, new_attention_state = attention_cell(inputs, attention_state)
        new_states.append(new_attention_state)

      for i in range(1, len(self._cells)):
        with tf.variable_scope("cell_%d" % i):

          cell = self._cells[i]
          cur_state = state[i]

          if not isinstance(cur_state, tf.contrib.rnn.LSTMStateTuple):
            raise TypeError("`state[{}]` must be a LSTMStateTuple".format(i))

          if self.use_new_attention:
            cur_state = cur_state._replace(h=tf.concat(
                [cur_state.h, new_attention_state.attention], 1))
          else:
            cur_state = cur_state._replace(h=tf.concat(
                [cur_state.h, attention_state.attention], 1))

          cur_inp, new_state = cell(cur_inp, cur_state)
          new_states.append(new_state)

    return cur_inp, tuple(new_states)

if __name__ == '__main__':
    model = Model()
    model.train()

