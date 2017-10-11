# -*- coding: utf-8 -*
import tensorflow as tf
from tensorflow.python.ops import lookup_ops
import numpy as np
import collections
import config
import os

src_file = config.FLAGS.src_file
src_before_file = config.FLAGS.src_before_file
src_after_file = config.FLAGS.src_after_file
# 只有在预测结果时使用。

src_vocab_file = config.FLAGS.src_vocab_file
word_embedding_file = config.FLAGS.word_embedding_file
model_path = config.FLAGS.model_path
embeddings_size = config.FLAGS.embeddings_size
max_sequence = config.FLAGS.max_sequence
batch_size = config.FLAGS.batch_size
num_layer = config.FLAGS.num_layer
model_path = config.FLAGS.model_path
epoch = config.FLAGS.epoch
pred_file = config.FLAGS.src_predict_file


def print_out(line, new_line):
    print line


class BatchedInput(collections.namedtuple("BatchedInput",
                                          ("initializer",
                                           "source",
                                           "source_before",
                                           "source_after",
                                           "source_length",
                                           "source_before_length",
                                           "source_after_length"))):
  pass


def build_word_index(line_num_limit=100):
    if not os.path.exists(src_vocab_file):
        dict_word = {}
        with open(src_file, 'r') as source:
            try:
                index = 0
                while True:
                    line = source.readline()
                    line = line.strip()
                    index += 1
                    if index > line_num_limit: break
                    if line == '': continue
                    words = line.split()
                    for w in words:
                        dict_word[w] = dict_word.get(w, 0) + 1

            except EOFError:
                pass
        top_words = sorted(dict_word.items(), key=lambda s: s[1], reverse=True)
        with open(src_vocab_file, 'w') as s_vocab:
            for word, frequence in top_words:
                s_vocab.write(word + '\n')

    else:
        print 'target vocabulary file has already existed, continue to next stage.'

    if not os.path.exists(model_path):
        os.makedirs(model_path)


def get_src_vocab_size():
    '''
    :return: 训练数据中共有多少不重复的词。
    '''
    size = 0
    with open(src_vocab_file, 'r') as vocab_file:
        for content in vocab_file.readlines():
            content = content.strip()
            if content != '':
                size += 1
    return size


def create_vocab_tables(src_vocab_file, src_unknown_id):
    src_vocab_table = lookup_ops.index_table_from_file(
        src_vocab_file, default_value=src_unknown_id)
    return src_vocab_table


def get_iterator(src_vocab_table, vocab_size, batch_size, buffer_size=None, random_seed=None,
                 num_threads=8, src_max_len=max_sequence, num_buckets=5):
    if buffer_size is None:
        # 如果buffer_size比总数据大很多，则会报End of sequence warning。
        # https://github.com/tensorflow/tensorflow/issues/12414
        buffer_size = batch_size * 10

    src_dataset = tf.contrib.data.TextLineDataset(src_file)
    src_before_dataset = tf.contrib.data.TextLineDataset(src_before_file)
    src_after_dataset = tf.contrib.data.TextLineDataset(src_after_file)
    src_tgt_dataset = tf.contrib.data.Dataset.zip((src_dataset, src_before_dataset, src_after_dataset))

    src_tgt_dataset = src_tgt_dataset.shuffle(
        buffer_size, random_seed)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, src_before, src_after: (
            tf.string_split([src]).values, tf.string_split([src_before]).values,
            tf.string_split([src_after]).values),
        num_threads=num_threads,
        output_buffer_size=buffer_size)

    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, src_before, src_after: (src[:src_max_len], src_before[:src_max_len],
                                                src_after[:src_max_len]),
            num_threads=num_threads,
            output_buffer_size=buffer_size)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, src_before, src_after: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                                            tf.cast(src_vocab_table.lookup(src_before), tf.int32),
                                            tf.cast(src_vocab_table.lookup(src_after), tf.int32)),
        num_threads=num_threads, output_buffer_size=buffer_size)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, src_before, src_after: (
            src, src_before, src_after, tf.size(src), tf.size(src_before), tf.size(src_after)),
        num_threads=num_threads,
        output_buffer_size=buffer_size)

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The first three entries are the source and target line rows;
            # these have unknown-length vectors.  The last two entries are
            # the source and target row sizes; these are scalars.
            padded_shapes=(tf.TensorShape([None]),  # src
                           tf.TensorShape([None]),  #
                           tf.TensorShape([None]),  #
                           tf.TensorShape([]),  # len
                           tf.TensorShape([]),  #
                           tf.TensorShape([])),  #
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(vocab_size+1,  # src
                            vocab_size + 1,  #
                            vocab_size + 1,  #
                            0,  # src_len -- unused
                            0,
                            0))

    def key_func(unused_1, unused_2, unused_3, src_len, src_before_len, src_after_len):
        if src_max_len:
            bucket_width = (src_max_len + num_buckets - 1) // num_buckets
        else:
            bucket_width = 10

        temp_id = tf.maximum(src_len // bucket_width, src_before_len // bucket_width)
        bucket_id = tf.maximum(temp_id, src_after_len // bucket_width)
        return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    def reduce_func(unused_key, windowed_data):
        return batching_func(windowed_data)

    batched_dataset = src_tgt_dataset.group_by_window(
        key_func=key_func, reduce_func=reduce_func, window_size=batch_size)

    batched_iter = batched_dataset.make_initializable_iterator()
    (src_ids, src_before_ids, src_after_ids, src_len, src_before_len, src_after_len) = (
        batched_iter.get_next())

    return BatchedInput(
        initializer=batched_iter.initializer,
        source=src_ids,
        source_before=src_before_ids,
        source_after=src_after_ids,
        source_length=src_len,
        source_before_length=src_before_len,
        source_after_length=src_after_len)


def get_predict_iterator(src_vocab_table, vocab_size, batch_size, max_len=max_sequence):
    pred_dataset = tf.contrib.data.TextLineDataset(pred_file)
    pred_dataset = pred_dataset.map(
        lambda src: tf.string_split([src]).values)
    if max_len:
        pred_dataset = pred_dataset.map(lambda src: src[:max_sequence])

    pred_dataset = pred_dataset.map(
        lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))

    pred_dataset = pred_dataset.map(lambda src: (src, tf.size(src)))

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=(tf.TensorShape([None]),  # src
                           tf.TensorShape([])),  # src_len
            padding_values=(vocab_size+1,  # src
                            0))  # src_len -- unused

    batched_dataset = batching_func(pred_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    (src_ids, src_seq_len) = batched_iter.get_next()

    return BatchedInput(
        initializer=batched_iter.initializer,
        source=src_ids,
        source_length=src_seq_len)


def load_word2vec_embedding(vocab_size):
    '''
        加载外接的词向量。
        :return:
    '''
    print 'loading word embedding, it will take few minutes...'
    embeddings = np.random.uniform(-1,1,(vocab_size + 2, embeddings_size))
    # 保证每次随机出来的数一样。
    rng = np.random.RandomState(23455)
    unknown = np.asarray(rng.normal(size=(embeddings_size)))
    padding = np.asarray(rng.normal(size=(embeddings_size)))
    f = open(word_embedding_file)
    for index, line in enumerate(f):
        values = line.split()
        try:
            coefs = np.asarray(values[1:], dtype='float32')  # 取向量
        except ValueError:
            # 如果真的这个词出现在了训练数据里，这么做就会有潜在的bug。那coefs的值就是上一轮的值。
            print values[0], values[1:]

        embeddings[index] = coefs   # 将词和对应的向量存到字典里
    f.close()
    # 顺序不能错，这个和unkown_id和padding id需要一一对应。
    embeddings[-2] = unknown
    embeddings[-1] = padding

    return tf.get_variable("embeddings", dtype=tf.float32,
                           shape=[vocab_size + 2, embeddings_size],
                           initializer=tf.constant_initializer(embeddings), trainable=False)


def id2word():
    return lookup_ops.index_to_string_table_from_file(
        src_vocab_file, default_value='<unknown>')


build_word_index()
vocab_size = get_src_vocab_size()
src_vocab_table = create_vocab_tables(src_vocab_file, vocab_size + 1)

if __name__ == "__main__":
    vocab_size = get_src_vocab_size()
    src_vocab_table = create_vocab_tables(src_vocab_file, vocab_size + 1)
    iterator = get_iterator(src_vocab_table, vocab_size, 8)
    id2word = id2word()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        tf.tables_initializer().run()
        for i in range(10):
            try:
                source = sess.run(iterator.source)
                print source.shape
                # print sess.run(id2word.lookup(tf.constant(10, dtype=tf.int64)))
            except tf.errors.OutOfRangeError:
                sess.run(iterator.initializer)
