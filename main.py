#!/usr/bin/env python3
from utils.data_loader import DataLoader 
from utils.sms_preprocessor import SmsPreprocessor
from utils.tokenizer import Tokenizer
from models.utils import load_model, predict
from utils.helper_functions import create_pipeline
from models.create_cnn import build_convolutional_model
from models.train import train_model

import numpy as np
from sklearn.pipeline import Pipeline
import argparse
from sklearn.utils import class_weight
import json
import pickle
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--save', dest='save', action='store_true')
feature_parser.add_argument('--no-save', dest='save', action='store_false')
parser.set_defaults(save=True)

args = parser.parse_args()

if __name__ == '__main__':
    loader = DataLoader('data/sms_data')
    pipeline = create_pipeline(key_word_path="data/pickled/key_word_map_new_data_1.pkl")

    # this class wieghts have better accuracy than the one computed with sklearn
    class_weight = [1., 1.5]

    tokenized, key_word_map = pipeline.fit_transform(loader.sms_data)

    model = build_convolutional_model(filters=32, kernel_size=3, padding="valid", strides=1,
                                      data_format=None, classes=2, layers=3, fc=True, fc_dropout=0.5,
                                      pooling='max', pool_size=2, conv_dropout=False)

    model = train_model(model=model,
                        X=tokenized,
                        y=loader.labels,
                        save_model=args.save,
                        model_path='model_conv_drop_false_15_new_data_1.json',
                        weights_path='model_weights_conv_drop_false_15_new_data_1.h5',
                        epochs=4,
                        batch_size=16,
                        class_weight=class_weight,
                        shuffle_data=True)


