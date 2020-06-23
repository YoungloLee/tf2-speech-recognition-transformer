from tensorflow.keras.layers import Dense, Input, LSTM, Bidirectional, TimeDistributed, Masking, LayerNormalization, Softmax
from tensorflow.keras.models import Model
import tensorflow as tf

from glob import glob
import os


def find_files(directory, pattern='**/*.h5'):
    return sorted(glob(os.path.join(directory, pattern), recursive=True))


class SpeechNetwork:
    def __init__(self, save_dir, args):
        self.args = args
        self.save_dir = save_dir
        self.model, self.latest_file = self.generate_model()

    def generate_model(self):
        # Input
        input = Input(shape=[None, self.args.num_mels], name='input_speech')
        x = Masking(mask_value=-self.args.max_abs_value, input_shape=(None, self.args.num_mels))(input)

        # Bidirectional LSTM
        for b in range(self.args.num_lstm_layers):
            x = Bidirectional(LSTM(units=self.args.num_units_per_lstm//2, return_sequences=True))(x)
            x = LayerNormalization()(x)

        # output layer with softmax
        x = TimeDistributed(Dense(self.args.num_classes, name='speech_encoder'))(x)
        logit = Softmax()(x)

        # Generate model
        model = Model(inputs=input, outputs=logit)

        # Check if pre-trained model exists
        h5_list = find_files(self.save_dir)
        if h5_list:
            latest_file = max(h5_list, key=os.path.getctime)
            model.load_weights(latest_file)
            return model, latest_file
        else:
            return model, None
