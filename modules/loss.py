import tensorflow as tf


def loss_function(y_true, y_pred):
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    mask = tf.logical_not(tf.math.equal(y_true, 0))
    mask = tf.cast(mask, dtype=loss.dtype)
    loss = mask * loss
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def label_error_rate(y_true, y_pred, label_length):
    y_pred = tf.argmax(y_pred, axis=-1)
    sparse_y_pred = tf.keras.backend.ctc_label_dense_to_sparse(y_pred, label_length)
    sparse_y_true = tf.keras.backend.ctc_label_dense_to_sparse(y_true, label_length)
    sparse_y_pred = tf.cast(sparse_y_pred, tf.int64)
    sparse_y_true = tf.cast(sparse_y_true, tf.int64)
    ed_tensor = tf.edit_distance(sparse_y_pred, sparse_y_true, normalize=True)
    return tf.reduce_mean(ed_tensor)
