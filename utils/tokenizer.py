"""
Functions to extract word_key_maps and perform tokenization
"""
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, key_word_map=None): 
        self.key_word_map = key_word_map
    
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
            tokenized.append(tokenized_text)
        return np.asarray(tokenized), self.key_word_map


    def create_key_word_dict(self, text: [str]):
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
        key_word_map["<unk>"] = i
        return key_word_map

    
    def transform(self, X, y=None):
        print('Tokenizing...')
        tokenized = self.tokenize(X)
        print('Finished!')
        return tokenized
    

    def fit(self, X, y=None):
        return self

