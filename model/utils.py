from collections import defaultdict
import tensorflow as tf


class Vocabulary:
    def __init__(self):
        _allowed_characters = 'abcdefghijklmnopqrstuvwxyz1234567890!.,#$%@()=+*/'
        self.vocabulary_size = len(_allowed_characters)
        self._char2idx = dict([(_allowed_characters[i], i) for i in range(len(_allowed_characters))])
        self._idx2char = dict([(value, key) for key , value in self._char2idx.items()])
        self._char2idx = defaultdict(lambda: self.vocabulary_size, self._char2idx)
        self._idx2char = defaultdict(lambda: self.vocabulary_size, self._idx2char)

    def text2idx(self, text):
        encoded = tf.py_func(lambda x: self._char2idx[x.lower()], [text], tf.int64, stateful=False)
        return encoded