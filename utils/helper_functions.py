from sklearn.pipeline import Pipeline
from utils.sms_preprocessor import SmsPreprocessor
from utils.tokenizer import Tokenizer

import pickle
import logging
import gzip
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os

def create_pipeline(key_word_path=None, save_path=None):
    if key_word_path is not None:
        try:
            with open(key_word_path, 'rb') as f:
                key_word_map = pickle.load(f)
        except IOError as e:
            logger.error("No such file!")
            logger.error("Creating pipeline without pre defined key_word_map!")
            key_word_map = None
    else:
        key_word_map = None
    pipeline = Pipeline([
        ("preprocessor", SmsPreprocessor()),
        ("tokenizer", Tokenizer(key_word_map=key_word_map, save_path=save_path))
    ])
    return pipeline
