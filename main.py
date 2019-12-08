#!/usr/bin/env python3
from utils.data_loader import DataLoader 
from utils.sms_preprocessor import SmsPreprocessor
from utils.tokenizer import Tokenizer
from sklearn.pipeline import Pipeline
from models.create_cnn import build_convolutional_model
from models.train import train_model

import numpy as np
import argparse
from sklearn.utils import class_weight

parser = argparse.ArgumentParser()
feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--save', dest='save', action='store_true')
feature_parser.add_argument('--no-save', dest='save', action='store_false')
parser.set_defaults(save=True)

args = parser.parse_args()

if __name__ == '__main__':
    pipeline = Pipeline([
        ("preprocessor", SmsPreprocessor(True)),
        ("tokenizer", Tokenizer())
    ])
    loader = DataLoader('./data/sms_data', convert_to_int=True)
    tokenized, key_word_map = pipeline.fit_transform(loader.sms_data)
    labels = loader.labels


    class_weight = class_weight.compute_class_weight('balanced',
                                                     np.unique(labels),
                                                     labels)

    model = build_convolutional_model(filters=32, kernel_size=3, padding="valid", strides=1, data_format=None,
                                      classes=2, layers=3, fc=True, fc_dropout=0.5, pooling='max', pool_size=2)
    # TODO: add parameters to the function instead of hard coded 
    model = train_model(model=model, 
                        X=tokenized,
                        y=labels, 
                        save_model=args.save, 
                        model_path='model_2.json',
                        weights_path='model_weights_2.h5',
                        epochs=4,
                        batch_size=16,
                        class_weight=class_weight,
                        conv_dropout=True,
                        shuffle_data=True)
