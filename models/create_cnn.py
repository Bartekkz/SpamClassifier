#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPool1D, AveragePooling1D, BatchNormalization, Dense, Flatten, \
                                    Activation, Dropout, Embedding, GlobalMaxPooling1D
from tensorflow.keras.regularizers import l2                                
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras import metrics

from typing import Union
import json

i = 0


def convolutional_layer_with_pooling(model,
                                     filters, 
                                     kernel_size, 
                                     strides, 
                                     padding,                                     
                                     conv_dropout,
                                     data_format=None,
                                     pooling='max', 
                                     activation='relu',
                                     pool_size=2
                                     ):
    """
    creates Keras based convolutional layer
    :param model: Model
    :param filters: int -> nubmer of output filters in convolution
    :param kernel_size -?  int or tuple/list of 2 int's -> the height and width of the 2D convolution window
    :param strides: int or tuple/list of 2 int's -> the height and width of the convolution
    :param padding: string -> type of padding
    :param conv_dropout: bool -> add BatchNormalization layer
    :param data_format: order of the dimensions in the inputs
    :param pooling: str -> type of pooling
    :param activation: str -> activation function for convolution layer
    :param pool_size: int/tuple -> width and hegith of pool

    :returns: keras.layers.Conv2d
    : #TODO: add error raising
    """

    if pooling == 'max':
        pool = MaxPool1D(pool_size)
    else:
        pool = AveragePooling1D(pool_size)

    global i
    with tf.name_scope(f'conv2d_block{i}'):
        model.add(Conv1D(filters=filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding=padding,
                         activation=activation,
                         batch_input_shape=(None, 50, 1),
                         data_format=data_format))
        model.add(pool)
    if conv_dropout:
        model.add(BatchNormalization())
    return model


def build_convolutional_model(filters: int, kernel_size: Union[int, tuple], padding: str, 
                              strides: Union[int, tuple], data_format: Union[str, None], 
                              classes: int, **kwargs: object) -> Sequential:
    layers = kwargs.get('layers', 1)
    fc_dropout = kwargs.get('fc_dropout', 0)
    conv_dropout = kwargs.get('conv_dropout', False)
    conv_activation = kwargs.get('conv_activation', 'relu')
    fc = kwargs.get('fc', True)
    loss_l2 = kwargs.get('loss_l2', 0)
    lr = kwargs.get('lr', 0.001)
    clipnorm = kwargs.get('clipnorm', 0)
    pooling = kwargs.get('pooling', 'max')
    pool_size = kwargs.get('pool_size', 2)
    maxlen = kwargs.get('maxlen', 50)

    print('Creating model...')
    model = Sequential()
    model.add(Embedding(10000, 100, input_length=maxlen))
    for layer in range(layers): 
        model = convolutional_layer_with_pooling(model=model,
                                                 filters=filters,
                                                 kernel_size=kernel_size,
                                                 strides=strides,
                                                 padding=padding,
                                                 data_format=data_format,
                                                 pooling=pooling,
                                                 conv_dropout=conv_dropout,
                                                 activation=conv_activation,
                                                 pool_size=pool_size
                                                 )
    if fc:
        model.add(Flatten())
        model.add(Dense(128, activation='relu', activity_regularizer=l2(loss_l2)))
        if fc_dropout > 0:
            model.add(Dropout(fc_dropout))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(optimizer=Adam(lr=lr, clipnorm=clipnorm),
                  loss="binary_crossentropy",
                  metrics=['acc', metrics.binary_accuracy])
    print('Model Compiled Properly!')
    return model


def load_model(arch_path, weights_path):
    #model = model_from_json(json.dumps(arch_path))
    with open(arch_path, 'r') as json_file:
        architecture = json_file.read() 
        model = model_from_json(architecture)
    model.load_weights(weights_path)
    print(model.summary())
    print('model loaded succesfully!')
    return model
