#!/usr/bin/env python3
from utils.data_loader import DataLoader 
from utils.sms_preprocessor import SmsPreprocessor
from utils.tokenizer import Tokenizer
from sklearn.pipeline import Pipeline


text = ["Hello man what do you want? what is going on", 'what do going You want']

if __name__ == '__main__':
    pipeline = Pipeline([
        ("preprocessor", SmsPreprocessor(False)),
        ("tokenizer", Tokenizer())
    ])
    loader = DataLoader('./data/sms_data')
    tokenized, key_word_map = pipeline.fit_transform(loader.sms_data)
    print(tokenized[100:120])
  




