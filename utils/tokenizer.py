"""
Functions to extract word_key_maps and perform tokenization
"""
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import json
import pickle


class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, key_word_map=None, pad_sequences=True, maxlen=50, padding='pre', save_path=None):
        self.key_word_map = key_word_map
        self.pad_sequences = pad_sequences
        self.maxlen = maxlen
        self.padding = padding
        self.save_path = save_path

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
            if self.save_path:
                self.key_word_map = self.create_key_word_dict(text, self.save_path)
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
            return [tokenized], self.key_word_map
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
            # TODO: change function
            tokenized_text = np.asarray(tokenized_text, dtype=np.float32)
            tokenized.append(tokenized_text)
        if self.save_path is not None:
            with open(self.save_path, 'w') as json_file:
                key_word = json.dumps(self.key_word_map)
                json_file.write(key_word)
        return np.asarray(tokenized), self.key_word_map

    @staticmethod
    def create_key_word_dict(text: [str], save_path=None):
        """
        Create dictionary of word, token pairs
        @params:
        - text: str or list(str)
        - clean_text: bool -> lowercase text and remove punctuation
        """
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
        if save_path:
            assert (save_path[-3:] == 'pkl'),"At the moment You can only save files with pickle(.pkl) extension" 
            
            with open(save_path, 'wb') as f:
                pickle.dump(key_word_map, f) 
        return key_word_map

    @staticmethod
    def pad_seq(tokenized_text, maxlen, padding='pre'):
        if isinstance(tokenized_text, list):
            tokenized_text = np.asarray(tokenized_text)
        padded_seq = np.zeros((len(tokenized_text), maxlen), dtype='float32')
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

