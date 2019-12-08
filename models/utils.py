#!/usr/bin/env python3
from tensorflow.keras.models import model_from_json
from tensorflow.keras import metrics
from sklearn.pipeline import Pipeline 
import numpy as np
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
    print('Model loaded succesfully!')
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc', metrics.binary_accuracy])
    print('Model Compiled!')
    return model


def predict(sms, model, pipeline):
    tokenized, _ = pipeline.transform(sms)
    print('\n' * 2)
    i = 0 
    for msg in tokenized:
        msg = np.expand_dims(msg, axis=0)
        prediction = model.predict(msg)
        print(prediction[0][0])
        if prediction[0][0] >= 0.5:
            prettify_print('SPAM!', sms[i])
        else:
            prettify_print('HAM!', sms[i])
        i += 1 
    print('\n' * 2)

def prettify_print(text, original):
    print('   -------    ')
    if len(text) == 5:
        print(f'   |{text}| -> {original}')
    else:
        print(f'   | {text}| -> {original}')
    print('   -------    ')
