#!/usr/bin/env python3
from utils.data_loader import DataLoader 
from utils.sms_preprocessor import SmsPreprocessor
from utils.tokenizer import Tokenizer
from sklearn.pipeline import Pipeline
from models.create_cnn import build_convolutional_model
from models.train import train_model

import numpy as np


text = ["Hello man what do you want? what is going on", 'what do going You want']
class_weight = {
    0: 1.,
    1: 2.0 
}
if __name__ == '__main__':
    pipeline = Pipeline([
        ("preprocessor", SmsPreprocessor(True)),
        ("tokenizer", Tokenizer())
    ])
    loader = DataLoader('./data/sms_data', convert_to_int=True)
    tokenized, key_word_map = pipeline.fit_transform(loader.sms_data)
    print(tokenized[1002])
    print(print(len(loader.labels)))
    print(sum(loader.labels))
    
    model = build_convolutional_model(filters=32, kernel_size=3, padding="valid", strides=1, data_format=None,
                                      classes=2, layers=3, fc=True, fc_dropout=0.5, pooling='max', pool_size=2)

    model = train_model(model=model, 
                        X=tokenized, 
                        y=loader.labels, 
                        save_model=True, 
                        model_path='model_2.json',
                        weights_path='model_weights_2.h5',
                        epochs=4,
                        batch_size=16,
                        class_weight=class_weight,
                        conv_dropout=True)
