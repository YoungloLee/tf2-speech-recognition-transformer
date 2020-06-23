import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np


class ValueWindow():
    def __init__(self, window_size=100):
        self._window_size = window_size
        self._values = []

    def append(self, x):
        self._values = self._values[-(self._window_size - 1):] + [x]

    @property
    def sum(self):
        return sum(self._values)

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        return self.sum / max(1, self.count)

    def reset(self):
        self._values = []


def split_title_line(title_text, max_words=5):
    """
    A function that splits any string based on specific character
    (returning it with the string), with maximum number of words on it
    """
    seq = title_text.split()
    return '\n'.join([' '.join(seq[i:i + max_words]) for i in range(0, len(seq), max_words)])


def plot_alignment(alignment, path, title=None, split_title=False, max_len=None):
    if max_len is not None:
        alignment = alignment[:, :max_len]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    im = ax.imshow(alignment, aspect='auto', origin='lower', interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'

    if split_title:
        title = split_title_line(title)

    plt.xlabel(xlabel)
    plt.title(title)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.savefig(path, format='png')
    plt.close()

