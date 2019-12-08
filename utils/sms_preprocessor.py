import os
from sklearn.base import BaseEstimator, TransformerMixin
import string 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle


class SmsPreprocessor(BaseEstimator, TransformerMixin):
    #TODO: Update docs
    @staticmethod
    def preprocess(text):
        '''
        @params:
        - text: str
        @return:
        - text: str -> lowercased and without punctuation 
        '''
        text = SmsPreprocessor.clean_text(text)
        text = SmsPreprocessor.remove_stopwords(text)
        text = SmsPreprocessor.stem_text(text)
        return text

    @staticmethod
    def clean_text(text: [str]) -> [str]:
        '''
        lowercase text and remove punctuation
        @params:
            - text: str or list(str)
        
        @return:
           - list of lowered strings(sentences) without punctuation
        '''
        if isinstance(text, str):
            text = text.lower()
            text = ''.join(v for v in text if v not in string.punctuation)
            return text
        final_text = []
        for sent in text:
            cleaned_text = []
            for word in sent.split(): 
                    word = word.lower()
                    text = ''.join(v for v in word if v not in string.punctuation)
                    cleaned_text.append(text)                    
            cleaned_sent = " ".join([word for word in cleaned_text])
            final_text.append(cleaned_sent)
        return final_text

    @staticmethod
    def remove_stopwords(text):
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

    @staticmethod
    def stem_text(text):
        ps = PorterStemmer()
        if isinstance(text, str):
            text = ' '.join([ps.stem(word) for word in text.split()])
            return text
        stemmed = []
        for sent in text:
            stemmed_text = " ".join([ps.stem(word) for word in sent.split()])
            stemmed.append(stemmed_text)
        return stemmed
 
    def transform(self, X, y=None):
        print('Processing...')
        processed_sms = self.preprocess(X)
        print('Done!')
        return processed_sms

    def fit(self, X, y=None):
        return self
