#!/usr/bin/env python3

from utils.sms_preprocessor import SmsPreprocessor
from utils.tokenizer import Tokenizer
from sklearn.pipeline import Pipeline



text = ["Hello man what do you want? what is going on", 'what do going You want']



if __name__ == '__main__':
    pipeline = Pipeline([
        ("preprocessor", SmsPreprocessor(False)),
        ("tokenizer", Tokenizer())
    ])
    tokenized, key_word_map = pipeline.fit_transform(text)
    token, _ = pipeline.fit_transform("Barek")
    print(token)
    print(_)




