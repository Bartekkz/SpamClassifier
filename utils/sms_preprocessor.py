import os
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin       
import string 
import nltk
from nltk.corpus import stopwords


class SmsPreprocessor(BaseEstimator, TransformerMixin):
    #TODO: 
    ''' 
    UPDATE docs
    helper class to clean sms messages, tokenizer them
    '''
    def __init__(self, load):
        self.load = load

    
    def preprocess(self, text, only_punctuation):
        '''
        @params:
        - text: str
        @return:
        - text: str -> lowercased and without punctuation 
        '''
        cleaned_text = self.clean_text(text, only_punctuation)
        
        return cleaned_text


    def clean_text(self, text: [str], only_punctuation=False) -> [str]:
        '''
        lowercase text and remove punctuation

        @params:
            - text: str or list(str)
        
        @return:
           - list of lowered strings(sentences) without punctuation
        '''
        if isinstance(text, str):
            if only_punctuation:            
                text = ''.join(v for v in text if v not in string.punctuation)
                return text
            text = text.lower()
            text = ''.join(v for v in text if v not in string.punctuation)
            return text
        final_text = []
        for sent in text:
            cleaned_text = []
            for word in sent.split(): 
                if only_punctuation:
                    text = ''.join(v for v in word if v not in string.punctuation)
                    cleaned_text.append(text)
                else:
                    word = word.lower()
                    text = ''.join(v for v in word if v not in string.punctuation)
                    cleaned_text.append(text)                    
            cleaned_sent = " ".join([word for word in cleaned_text])
            final_text.append(cleaned_sent)
        return final_text 


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
        return key_word_map


    def remove_stopwords(self, text):
        cleaned_text = []
        stop_words = set(stopwords.words('english'))
        if isinstance(text, str):
            for word in text.split():
                if word not in stop_words:
                    cleaned_text.append(word)
            if len(cleaned_text) > 1:
                final = " ".join(word for word in cleaned_text)
                return final
            else:
                return cleaned_text[0]
        final_text = []
        for sentence in text:
            cleaned_text = []
            for word in sentence.split():
                if word not in stop_words:
                    cleaned_text.append(word)
            sent = " ".join([word for word in cleaned_text])
            final_text.append(sent) 
        return final_text 


    def tokenize(self, text, key_word_map=None, strategy="unknown"):
        #TODO: add other strategies, change function
        tokenized = []
        if key_word_map is None:
            key_word_map = self.create_key_word_dict(text)
        print(key_word_map)         
        if isinstance(text, str):
            for word in text.split():
                if word in key_word_map.keys():
                    token = key_word_map.get(word)
                    tokenized.append(token)
                else:
                    if strategy == "unknown":
                        tokenized.append("<unk>")
                    elif strategy == 'zeros':
                        tokenized.append(0)

        return tokenized
