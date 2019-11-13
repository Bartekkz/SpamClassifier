import os
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin       
import string 


class SmsPreprocessor(BaseEstimator, TransformerMixin):
    #TODO: 
    ''' 
    UPDATE docs
    helper class to clean sms messages, tokenizer them
    '''
    def __init__(self, load):
        self.load = load

    
    def preprocess(self, text):
        '''
        @params:
        - text: str
        @return:
        - text: str -> lowercased and without punctuation 
        '''
        text = text.lower()
        text = ''.join(v for v in text if v not in string.punctuation)
        return text


    def create_key_word_dict(self, text, preprocess=True):
        '''
        Create dictionary of word, token pairs
        @params:
        - text: str
        '''
        key_word_map = {}
        if preprocess:
            text = self.preprocess(text)
        i = 1 
        for word in text.split():
            if word in key_word_map.keys():
                continue 
            key_word_map[word] = i 
            i += 1
        return key_word_map


    def tokenize(self, text, key_word_map={}, strategy="unknow"):
        tokenized = []
        text = self.preprocess(text)
        if key_word_map is {}:
            key_word_map = self.create_key_word_dict(text)
         
        for word in text.split():
            if word in key_word_map.keys():
                token = key_word_map.get(word)
                tokenized.append(token)
            else:
                tokenized.append("<unk>")

        return tokenized




            



