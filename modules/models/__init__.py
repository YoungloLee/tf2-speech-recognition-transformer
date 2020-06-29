def create_model(name, checkpoint_state, hparams):
    if name == 'Transformer':
        from .transformer import SpeechNetwork
        return SpeechNetwork(checkpoint_state, hparams)
    else:
        raise Exception('Unknown model: ' + name)
