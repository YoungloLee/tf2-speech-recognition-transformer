import tensorflow as tf


def greedy_decode(y_pred, seq_len):
    y_pred = tf.transpose(y_pred, perm=[1, 0, 2])
    seq_len = tf.squeeze(seq_len, axis=-1)
    decoded = tf.nn.ctc_greedy_decoder(inputs=y_pred, sequence_length=seq_len)[0][0]
    return decoded.values.numpy()


def beam_search_decode(y_pred, seq_len):
    y_pred = tf.transpose(y_pred, perm=[1, 0, 2])
    seq_len = tf.squeeze(seq_len, axis=-1)
    decoded = tf.nn.ctc_beam_search_decoder(inputs=y_pred, sequence_length=seq_len, beam_widt=10, top_paths=1)[0][0]
    return decoded.values.numpy()
