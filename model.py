# -*- coding: utf-8 -*

import utils
from model_helper import *
from tensorflow.python.layers import core as layers_core


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
        cell_forward = create_cells(self.model, utils.num_layer)
        cell_backward = create_cells(self.model, utils.num_layer)
        source_embedding = tf.nn.embedding_lookup(self.embedding_weights, self.source)
        encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
            cell_forward, cell_backward, source_embedding, dtype=tf.float32,
            sequence_length=self.source_length)
        encoder_outputs = tf.concat(encoder_outputs, axis=-1)
        # model.py: 537
        return encoder_outputs, bi_encoder_state

    def _build_decoder(self, encoder_outputs, encoder_state):
        '''
        :param encoder_outputs: 默认(time_step, batch_size, 2 * num_units)
        :param encoder_state: 
        :return: 
        '''
        alignment_history = self.mode == tf.contrib.learn.ModeKeys.INFER
        # TODO: check the shape
        encoder_outputs = tf.transpose(encoder_outputs, [1, 0, 2])
        # encoder_outputs should be batch-major
        attention_mechanism = create_attention_mechanism(
            FLAG.attention_type, FLAG.embeddings_size*2,
            encoder_outputs, self.source_length)
        cell = create_cells(self.model, utils.num_layer)
        cell = tf.contrib.seq2seq.AttentionWrapper(
            cell,
            attention_mechanism,
            attention_layer_size=FLAG.embeddings_size*2,
            alignment_history=alignment_history)
        # cell_backward = create_cells(self.model, utils.num_layer)
        # cell_backward = tf.contrib.seq2seq.AttentionWrapper(
        #     cell_backward,
        #     attention_mechanism,
        #     attention_layer_size=FLAG.embeddings_size * 2,
        #     alignment_history=alignment_history)

        # TODO: encoder_state没有做任何处理
        decoder_initial_state = cell.zero_state(FLAG.batch_size, tf.flot32).clone(
            cell_state=encoder_state)
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
        print state
        print len(state)


if __name__ == '__main__':
    model = Model()
    model.train()

