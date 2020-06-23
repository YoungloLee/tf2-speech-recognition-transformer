from modules.loss import loss_function, label_error_rate
from modules.models.transformer import create_masks
import tensorflow as tf


def valid_step_teacher_forcing(inp, tar, model, loss_tb, ler_tb, acc_tb):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    predictions, attention_weights = model.model(inp, tar_inp, False, enc_padding_mask, combined_mask, dec_padding_mask)

    loss = loss_function(tar_real, predictions)

    tar_weight = tf.cast(tf.logical_not(tf.math.equal(tar_real, 0)), tf.int32)
    tar_len = tf.reduce_sum(tar_weight, axis=-1)
    ler = label_error_rate(tar[:, 1:], predictions, tar_len)

    loss_tb(loss)
    ler_tb(ler)
    acc_tb(tar_real, predictions, sample_weight=tar_weight)

    return predictions.numpy()[0], attention_weights['decoder_layer4_block2'].numpy()[0, :, :tar_len[0], :]
