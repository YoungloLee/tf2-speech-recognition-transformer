# coding: utf-8
import pickle

# loading
with open('./text_syllable/syllable_tokenizer_all.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
# Mappings from symbol to numeric ID and vice versa:

tokens = [x[0] for x in sorted(tokenizer.word_counts.items(), key=lambda item: item[1], reverse=True)]


PAD = '<p>'
SOS = '<s>'
EOS = '</s>'

tokens = [PAD] + tokens
# start of utterance (for LAS)
tokens.append(SOS)
tokens.append(EOS)

index_token = {idx: ch for idx, ch in enumerate(tokens)}
token_index = {ch: idx for idx, ch in enumerate(tokens)}
