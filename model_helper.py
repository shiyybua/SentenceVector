import tensorflow as tf
import utils
import config

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


def create_cells(mode, num_layer):
    assert num_layer > 0
    unit_type, num_units, dropout = FLAG.unit_type, FLAG.num_units, FLAG.dropout
    cell = _single_cell(unit_type, num_units, dropout, mode)
    if num_layer == 1:
        return cell
    else:  # Multi layers
        return tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layer)])
