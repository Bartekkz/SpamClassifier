#!/usr/bin/env python3
from utils.data_loader import DataLoader 
from utils.sms_preprocessor import SmsPreprocessor
from utils.tokenizer import Tokenizer
from sklearn.pipeline import Pipeline
from models.create_cnn import build_convolutional_model
from models.train_model import train_model

text = ["Hello man what do you want? what is going on", 'what do going You want']
class_weight = {
    0: 1.,
    1: 1.5
}
if __name__ == '__main__':
    pipeline = Pipeline([
        ("preprocessor", SmsPreprocessor(True)),
        ("tokenizer", Tokenizer())
    ])
    loader = DataLoader('./data/sms_data', convert_to_int=True)
    tokenized, key_word_map = pipeline.fit_transform(loader.sms_data)

    model = build_convolutional_model(filters=32, kernel_size=3, padding="valid", strides=1, data_format=None,
                                      classes=2, layers=3, fc1=True, fc_dropout=0.5, pooling='max', pool_size=2)

    model = train_model(model=model, X=tokenized, y=loader.labels, save_model=True,
                        model_path='data/models/model_1.json', weights_path='data/weights/model_weights1.h5', epochs=4,
                        batch_size=16, class_weight=class_weight)
