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
    pipeline = create_pipeline(key_word_path="data/key_word_map.pkl")

    text = ['Press this button to win 500 dollats', 'I will be late today. Do not wait for me honey',
            'REMINDER FROM O2: To get 2.50 pounds free call credit and details of great offers pls' 
            'reply 2 this text with your valid name, house no and postcode',
            'To earn 1000 dollars send your valid name']
    #Emodel = load_model('data/models_data/model_2.json', 'data/models_data/model_weights_2.h5')
    #predict(text, model, pipeline)

    class_weight = class_weight.compute_class_weight('balanced',
                                                     np.unique(loader.labels),
                                                     loader.labels)
    # this class wieghts have better accuracy than the one computed with sklearn
    class_weight = [1., 1.5]
    tokenized, key_word_map = pipeline.fit_transform(loader.sms_data)

    model = build_convolutional_model(filters=32, kernel_size=3, padding="valid", strides=1,
                                      data_format=None, classes=2, layers=3, fc=True, fc_dropout=0.5,
                                      pooling='max', pool_size=2, conv_dropout=True)

    #loss: 0.0124 - acc: 0.9971 - binary_accuracy: 0.9971 - val_loss: 0.1277 - val_acc: 0.9525 - val_binary_accuracy: 0.9525class_weight=class_weight
    #loss: 0.0182 - acc: 0.9946 - binary_accuracy: 0.9946 - val_loss: 0.1411 - val_acc: 0.9632 - val_binary_accuracy: 0.9632
    #loss: 0.0095 - acc: 0.9973 - binary_accuracy: 0.9973 - val_loss: 0.1279 - val_acc: 0.9686 - val_binary_accuracy: 0.9686

    model = train_model(model=model,
                        X=tokenized,
                        y=loader.labels,
                        save_model=args.save,
                        model_path='model_2.json',
                        weights_path='model_weights_2.h5',
                        epochs=4,
                        batch_size=16,
                        class_weight=class_weight,
                        shuffle_data=True)
    

    # TODO: change pipeline, sae key_wor_map to the file
