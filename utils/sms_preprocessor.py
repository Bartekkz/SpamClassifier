import os
import random
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin       
import string 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle


class SmsPreprocessor(BaseEstimator, TransformerMixin):
    #TODO: Update docs
    def __init__(self, load):
        self.load = load
    
    def preprocess(self, text):
        '''
        @params:
        - text: str
        @return:
        - text: str -> lowercased and without punctuation 
        '''
        text = self.clean_text(text)
        text = self.remove_stopwords(text) 
        text = self.stem_text(text)
        return text 

    def clean_text(self, text: [str]) -> [str]:
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
    
    def stem_text(self, text):
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
        path = os.path.join(os.getcwd(), 'data/pickled/processed_sms.pickle')
        if self.load:
            if os.path.exists(path):
                print('Loading...')
                print("PATH EXISTS")
                with open(path, 'rb') as f:
                    processed_sms = pickle.load(f)
                    print('Done')
            else:
                print('Processing...') 
                processed_sms = self.preprocess(X)
                with open(path, 'wb') as f:
                    pickle.dump(processed_sms, f)
                    print('Done!')
        else:
            print('Processing...')
            processed_sms = self.preprocess(X)
            print('Done!')

        return processed_sms


    def fit(self, X, y=None):
        return self
