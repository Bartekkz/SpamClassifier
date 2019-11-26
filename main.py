#!/usr/bin/env python3
from utils.data_loader import load_sms_data
from utils.sms_preprocessor import SmsPreprocessor
from utils.tokenizer import Tokenizer
from sklearn.pipeline import Pipeline



text = ["Hello man what do you want? what is going on", 'what do going You want']


def 
if __name__ == '__main__':
    pipeline = Pipeline([
        ("preprocessor", SmsPreprocessor(False)),
        ("tokenizer", Tokenizer())
    ])
    sms_text, labels = load_sms_data('data/sms_data')
    print(len(sms_text))
    #tokenized, key_word_map = pipeline.fit_transform(text)
    #token, key_word_map = pipeline.fit_transform("Barek")
    #print(token)
    #print(key_word_map)




