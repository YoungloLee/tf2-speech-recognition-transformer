from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Layer, Embedding, Masking,\
    Conv2D, BatchNormalization, ReLU, Bidirectional, GRU
from tensorflow.keras.models import Model

import numpy as np
import tensorflow as tf


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_encoder_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_encoder_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


def create_encoder_padding_mask(seq):
    seq = tf.reduce_prod(tf.cast(tf.math.equal(seq, 0), tf.float32), axis=-1)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])


class Encoder(Model):
    def __init__(self, d_model, rate):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.rate = rate
        self.conv_encoder, self.rnn_encoder = [], []

        # Convolutional encoder
        self.conv_encoder.append(Conv2D(filters=32, kernel_size=(11, 41), strides=(2, 2), padding='same'))
        self.conv_encoder.append(BatchNormalization())
        self.conv_encoder.append(ReLU(max_value=20))
        self.conv_encoder.append(Conv2D(filters=32, kernel_size=(11, 21), strides=(1, 2), padding="same"))
        self.conv_encoder.append(BatchNormalization())
        self.conv_encoder.append(ReLU(max_value=20))

        # Mask layer
        self.masking = Masking()

        # Recurrent encoder
        self.rnn_encoder.append(Bidirectional(GRU(units=self.d_model // 2,
                                                  dropout=self.rate,
                                                  return_sequences=True,
                                                  return_state=False,
                                                  recurrent_initializer='glorot_uniform')))

    def call(self, inputs, input_lengths, hidden, training=None):
        x = tf.expand_dims(inputs, axis=-1)
        # Convolutional encoder
        for conv_enc in self.conv_encoder:
            x = conv_enc(x, training=training)
        x = tf.concat(tf.unstack(x, axis=2), axis=-1)
        mask = tf.sequence_mask(input_lengths, tf.shape(x)[1], dtype=tf.float32)
        x *= tf.expand_dims(mask, axis=-1)
        x = self.masking(x)
        # print(x._keras_mask)
        # Recurrent encoder
        for rnn_enc in self.rnn_encoder:
            output = rnn_enc(x, initial_state=hidden, training=training)
        return output

    def get_config(self):
        return super(Encoder, self).get_config()

    def initialize_hidden_state(self, batch_size):
        return [tf.zeros((batch_size, self.d_model // 2)), tf.zeros((batch_size, self.d_model // 2))]

    def get_seq_lens(self, input_lengths):
        seq_len = input_lengths
        for m in self.conv_encoder:
            if 'conv2d' in m.name:
                seq_len = tf.math.ceil(seq_len / m.strides[0])
        return seq_len


class DecoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Decoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(d_model, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

        self.final_layer = Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        inp_len = tf.reduce_sum(tf.cast(tf.logical_not(tf.math.equal(inp, 0)), tf.int32), axis=-2)[:, 0]
        inp_len = self.encoder.get_seq_lens(inp_len)
        init_state = self.encoder.initialize_hidden_state(tf.shape(inp)[0])
        enc_output = self.encoder(inp, inp_len, init_state, training)

        dec_padding_mask = 1 - tf.sequence_mask(inp_len, tf.reduce_max(inp_len), dtype=tf.float32)
        dec_padding_mask = dec_padding_mask[:, tf.newaxis, tf.newaxis, :]
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


class SpeechNetwork:
    def __init__(self, save_dir, args):
        self.args = args
        self.save_dir = save_dir
        self.encoder, self.decoder, self.final_layer, self.model = None, None, None, None
        self.generate_model()

    def generate_model(self):
        num_classes = self.args.num_classes
        num_layers = self.args.num_layers
        d_model = self.args.d_model
        dff = self.args.dff
        num_heads = self.args.num_heads
        dropout_rate = self.args.dropout_rate

        model = Transformer(num_layers, d_model, num_heads, dff,
                            target_vocab_size=num_classes,
                            pe_target=10000,
                            rate=dropout_rate)

        temp_input = tf.random.normal((64, 38, self.args.num_mels * self.args.lfr_m))
        temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=70)

        fn_out, _ = model(temp_input, temp_target,
                          training=False,
                          enc_padding_mask=None,
                          look_ahead_mask=None,
                          dec_padding_mask=None)

        print("Transformer encoder input shape: (batch_size, enc_length, num_mels) {}".format(temp_input.shape))
        print("Transformer decoder input shape: (batch_size, dec_length) {}".format(temp_target.shape))
        print("Transformer result shape: (batch_size, dec_length, target_vocab_size) {}".format(fn_out.shape))

        self.encoder = model.encoder
        self.decoder = model.decoder
        self.final_layer = model.final_layer
        self.model = model

