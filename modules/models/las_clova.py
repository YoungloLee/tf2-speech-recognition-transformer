from tensorflow.keras.layers import Dense, Input, Conv2D, Bidirectional, Layer, Embedding, GRU, BatchNormalization,\
    ReLU, MaxPool2D, Masking, Conv1D
from tensorflow.keras.models import Model
import tensorflow as tf

from glob import glob
import os


def find_files(directory, pattern='**/*.h5'):
    return sorted(glob(os.path.join(directory, pattern), recursive=True))


class Encoder(Model):
    def __init__(self, hparams):
        super(Encoder, self).__init__()
        self.hidden_dim = hparams.hidden_dim
        self.drop_rate_rnn = hparams.drop_rate_rnn
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
        self.rnn_encoder.append(Bidirectional(GRU(units=self.hidden_dim // 2,
                                                  dropout=self.drop_rate_rnn,
                                                  return_sequences=True,
                                                  return_state=True,
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
            output, fw_state, bw_state = rnn_enc(x, initial_state=hidden, training=training)
        return output, fw_state, bw_state

    def get_config(self):
        return super(Encoder, self).get_config()

    def initialize_hidden_state(self, batch_size):
        return [tf.zeros((batch_size, self.hidden_dim // 2)), tf.zeros((batch_size, self.hidden_dim // 2))]

    def get_seq_lens(self, input_lengths):
        seq_len = input_lengths
        for m in self.conv_encoder:
            if 'conv2d' in m.name:
                seq_len = tf.math.ceil(seq_len / m.strides[0])
        return seq_len


class LocationSensitiveAttention(Layer):
    def __init__(self, hparams):
        super(LocationSensitiveAttention, self).__init__()
        self.location_convolution = Conv1D(filters=32, kernel_size=(31, ), padding='same', use_bias=True)
        self.location_layer = Dense(hparams.attention_dim)
        self.W_Q = Dense(hparams.attention_dim)
        self.W_V = Dense(hparams.attention_dim)
        self.attention_bias = self.add_weight(name='attention_bias',
                                              shape=[hparams.attention_dim],
                                              initializer=tf.zeros_initializer,
                                              trainable=True)
        self.attention_projection = Dense(1, use_bias=False)

    def call(self, query, values, previous_alignment):
        # query hidden state shape == (batch_size, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # previous_align == (batch_size, max_len)
        f = self.location_convolution(tf.expand_dims(previous_alignment, axis=-1))
        processed_location_features = self.location_layer(f)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.location_sensitive_score(self.W_Q(query), processed_location_features, self.W_V(values))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, tf.squeeze(attention_weights, axis=-1)

    def location_sensitive_score(self, W_query, W_location, W_values):
        return self.attention_projection(tf.tanh(W_values + W_query + W_location + self.attention_bias))

    def initialize_alignment(self, batch_size, input_lengths):
        return tf.zeros((batch_size, input_lengths))


class Decoder(Model):
    def __init__(self, num_classes, hparams):
        super().__init__()
        self.hidden_dim = hparams.hidden_dim
        self.decoder_embedding = Embedding(input_dim=num_classes, output_dim=hparams.embedding_dim)
        self.decoder_rnn = GRU(hparams.hidden_dim,
                               return_sequences=True,
                               return_state=True,
                               recurrent_initializer='glorot_uniform')
        self.dense_layer = Dense(num_classes)
        self.attention = LocationSensitiveAttention(hparams)

    def call(self, inputs, state, enc_output, context, prev_align, training=None):
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.decoder_embedding(inputs)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        dec_output, state = self.decoder_rnn(x, initial_state=state)

        # enc_output shape == (batch_size, max_length, hidden_size)
        context, attention_weights = self.attention(dec_output, enc_output, prev_align)

        # output shape == (batch_size * 1, hidden_size)
        outputs = tf.reshape(dec_output, (-1, dec_output.shape[2]))

        outputs = self.dense_layer(tf.concat([outputs, context], axis=-1))

        return outputs, state, attention_weights, context

    def initialize_context(self, batch_size):
        return tf.zeros((batch_size, self.hidden_dim))

    def initialize_alignment(self, batch_size, input_lengths):
        return tf.zeros((batch_size, input_lengths))


class SpeechNetwork:
    def __init__(self, save_dir, args):
        self.args = args
        self.save_dir = save_dir
        self.encoder, self.decoder = None, None
        self.generate_model()

    def generate_model(self):
        num_classes = self.args.num_classes
        batch_size = self.args.batch_size
        num_mels = self.args.num_mels

        # Listen; Lower time resolution
        self.encoder = Encoder(self.args)
        sample_hidden = self.encoder.initialize_hidden_state(batch_size)
        example_input_batch = tf.random.uniform((batch_size, 47, num_mels))
        seq_len = self.encoder.get_seq_lens(tf.constant(batch_size * [33]))
        sample_output, fw_sample_hidden, bw_sample_hidden = self.encoder(example_input_batch, seq_len, sample_hidden)
        print('Encoder input shape: (batch size, sequence length, num_mels) {}'.format(example_input_batch.shape))
        print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
        print('Encoder Hidden state shape: (batch size, units) {}'.format(fw_sample_hidden.shape))

        # Attend and Spell
        attention_layer = LocationSensitiveAttention(self.args)
        sample_hidden = tf.concat([fw_sample_hidden, bw_sample_hidden], axis=-1)
        previous_alignment = attention_layer.initialize_alignment(sample_output.shape[0], sample_output.shape[1])
        attention_result, attention_weights = attention_layer(tf.expand_dims(sample_hidden, axis=1), sample_output, previous_alignment)
        print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
        print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

        self.decoder = Decoder(num_classes, self.args)
        previous_context = self.decoder.initialize_context(sample_output.shape[0])
        sample_decoder_output, _, _, _ = self.decoder(tf.random.uniform((batch_size, 1)), sample_hidden, sample_output,
                                                   previous_context, previous_alignment)
        print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
