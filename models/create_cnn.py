#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Input, \
                                    BatchNormalization, Dense, Flatten, Activation, Dropout
from tensorflow.keras.regularizers import l2                                
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from typing import Union

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
    '''
    creates Keras based convolutional layer

    :param filters: int -> nubmer of output filters in convolution
    :param kernel_size: int or tuple/list of 2 int's -> the height and width of the 2D convolution window
    :param strides: int or tuple/list of 2 int's -> the height and width of the convolution
    :param padding: string: type of padding
    :param conv_dropout: bool -> add BatchNormalization layer
    :param data_format: order of the dimensions in the inputs
    :param pooling: type of pooling

    :returns: keras.layers.Conv2d
    : #TODO: add error raising
    '''

    if pooling == 'max':
        pool = MaxPool2D(pool_size)
    else:
        pool = AveragePooling2D(pool_size)

    global i
    with tf.name_scope(f'conv2d_block{i}'):
        model.add(Conv2D(filters=filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding=padding,
                         activation=activation,
                         data_format=data_format))
        model.add(pool)
    if conv_dropout:
        model.add(BatchNormalization())
    return model


def build_convolutional_model(filters: int, kernel_size: Union[int, tuple], padding: str, strides: Union[int, tuple],
                              data_format: Union[str, None], classes: int, **kwargs: object) -> Model:
    layers = kwargs.get('layers', 1)
    fc_dropout = kwargs.get('fc_dropout', 0)
    conv_dropout = kwargs.get('conv_dropout', False)
    conv_activation = kwargs.get('conv_activation', 'relu')
    fc1 = kwargs.get('fc1', False)
    loss_l2 = kwargs.get('loss_l2', 0)
    lr = kwargs.get('lr', 0.001)
    clipnorm = kwargs.get('clipnorm', 0)
    pooling = kwargs.get('pooling', 'max')
    pool_size = kwargs.get('pool_size', 2)

    # TODO: change function for Sequentail model
    print('Creating model...')
    for layer in range(layers - 1):
        if layer == 0:
            inpts=inputs
        else:
            inpts=conv_block

        conv_block = convolutional_layer_with_pooling(inputs=inpts,
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
    if fc1:
        flatten = Flatten()(conv_block)
        fc1 = Dense(classes, activation='relu', activity_regularizer=l2(loss_l2))(flatten)
        if fc_dropout > 0:
            fc_dropout = Dropout(fc_dropout)(fc1)
            outputs = Activation('sigmoid')(fc_dropout)
            model = Model(inputs, outputs)
        else:
            outputs = Activation('sigmoid')(fc1)
            model = Model(inputs, outputs)
    else:
        model = Model(inputs, conv_block)

    model.compile(optimizer=Adam(clipnorm=clipnorm, lr=lr),
                  loss="binary_crossentropy")
    print('Model Compiled Properly!')
    return model
