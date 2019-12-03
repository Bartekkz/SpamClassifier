"""
Functions to extract word_key_maps and perform tokenization
"""
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, key_word_map=None, pad_sequences=True, maxlen=50, padding='pre'):
        self.key_word_map = key_word_map
        self.pad_sequences = pad_sequences
        self.maxlen = maxlen
        self.padding = padding

    def __getitem__(self, word):
        for key in self.key_word_map.keys():
            if key == word:
                return self.key_word_map.get(word)
    
    def tokenize(self, text, strategy="unknown"):
        # TODO: add other strategies, change function
        # TODO: optimize functions (atm sphagetti code :) )
        tokenized = []
        # TODO: change type of initing key_word_map
        if self.key_word_map is None:
            self.key_word_map = self.create_key_word_dict(text)
        if isinstance(text, str):
            for word in text.split():
                if word in self.key_word_map.keys():
                    token = self.key_word_map.get(word)
                    tokenized.append(token)
                else:
                    if strategy == "unknown":
                        token = self.key_word_map.get("<unk>")
                        tokenized.append(token)
                    elif strategy == 'zeros':
                        tokenized.append(0)
            return tokenized, self.key_word_map
        for sent in text:
            tokenized_text = []
            for word in sent.split():
                if word in self.key_word_map.keys():
                    token = self.key_word_map.get(word)
                    tokenized_text.append(token)
                else:
                    if strategy == "unknown":
                        token = self.key_word_map.get("<unk>")
                        tokenized_text.append(token)
                    elif strategy == 'zeros':
                        tokenized_text.append(0)
            # TODO: change appending
            tokenized.append(np.asarray(tokenized_text))
        return tokenized, self.key_word_map

    @staticmethod
    def create_key_word_dict(text: [str]):
        '''
        Create dictionary of word, token pairs
        @params:
        - text: str or list(str)
        - clean_text: bool -> lowercase text and remove punctuation
        '''
        key_word_map = {}
        if isinstance(text, str):
            text = [text]
        i = 1 
        for sentence in text:
            for word in sentence.split(): 
                if word in key_word_map.keys():
                    continue 
                key_word_map[word] = i
                i += 1
        # TODO: think about unknows
        key_word_map["<unk>"] = i
        return key_word_map

    @staticmethod
    def pad_seq(tokenized_text, maxlen, padding='pre'):
        if isinstance(tokenized_text, list):
            tokenized_text = np.asarray(tokenized_text)
        padded_seq = np.zeros((len(tokenized_text), maxlen), dtype='int32')
        for i, text in enumerate(tokenized_text):
            if text.shape[0] < maxlen:
                if padding == 'pre':
                    padded_seq[i] = np.pad(text, (0, maxlen - len(text)), 'constant')
                elif padding == 'post':
                    padded_seq[i] = np.pad(text, (maxlen - len(text), 0), 'constant')
            elif text.shape[0] > maxlen:
                padded_seq[i] = text[:maxlen]
        return padded_seq

    def transform(self, X, y=None):
        print('Tokenizing...')
        tokenized, key_word_map = self.tokenize(X)
        # TODO: check if this code has any sense or need to change it
        print('Tokenized!')
        if self.pad_sequences:
            padded = Tokenizer.pad_seq(tokenized, maxlen=self.maxlen, padding=self.padding)
            print('Finished')
            return padded, key_word_map
        print('Finished!')
        return tokenized, key_word_map

    def fit(self, X, y=None):
        return self

