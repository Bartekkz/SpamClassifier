#!/usr/bin/env python3
from utils.data_loader import DataLoader 
from utils.sms_preprocessor import SmsPreprocessor
from utils.tokenizer import Tokenizer
from sklearn.pipeline import Pipeline
from models.create_cnn import build_convolutional_model

text = ["Hello man what do you want? what is going on", 'what do going You want']

if __name__ == '__main__':
    pipeline = Pipeline([
        ("preprocessor", SmsPreprocessor(True)),
        ("tokenizer", Tokenizer())
    ])
    loader = DataLoader('./data/sms_data', True)
    print(loader.labels[:5])
    #tokenized, key_word_map = pipeline.fit_transform(loader.sms_data)
    #model = build_convolutional_model(filters=32, kernel_size=3, padding="valid", strides=1, data_format=None,
    #                                  classes=2, layers=3, fc1=True, fc_dropout=0.5, pooling='max', pool_size=2)

    #print(model.summary())

