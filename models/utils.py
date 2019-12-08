#!/usr/bin/env python3
from tensorflow.keras.models import model_from_json
from tensorflow.keras import metrics
from sklearn.pipeline import Pipeline 
import json
import sys
sys.path.append('..')

from utils.sms_preprocessor import SmsPreprocessor
from utils.tokenizer import Tokenizer


def load_model(arch_path, weights_path):
    #model = model_from_json(json.dumps(arch_path))
    with open(arch_path, 'r') as json_file:
        architecture = json_file.read() 
        model = model_from_json(architecture)
    model.load_weights(weights_path)
    print(model.summary())
    print('Model loaded succesfully!')
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc', metrics.binary_accuracy])
    print('Model Compiled!')
    return model



def predict(sms, model, key_word_map=None):
    pr = SmsPreprocessor(False)
    tk = Tokenizer(key_word_map=key_word_map)
    clean_sms = pr.preprocess(sms)
    tokenized = tk.tokenize(clean_sms)
    #prediction = model.evaluate(sms)
    print(tokenized)


