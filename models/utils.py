#!/usr/bin/env python3
from tensorflow.keras.models import model_from_json
from tensorflow.keras import metrics
import numpy as np
import sys
sys.path.append('..')


def load_model(arch_path, weights_path):
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
    preds = []
    for msg in tokenized:
        msg = np.expand_dims(msg, axis=0)
        prediction = model.predict(msg)
        preds.append(prediction)
    return preds


def prettify_print(text, original):
    print('   -------    ')
    if len(text) == 5:
        print(f'   |{text}| -> {original}')
    else:
        print(f'   | {text}| -> {original}')
    print('   -------    ')
