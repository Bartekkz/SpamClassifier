#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Input, BatchNormalization
from tensorflow.keras.models import Model

i = 0

def convolutional_layer_with_pooling(inputs, 
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
    Pool = MaxPool2D(pool_size)
  else:
    Pool = AveragePooling2D(pool_size)

  global i
  with tf.name_scope(f'conv2d_block{i}'):
    conv_layer = Conv2D(filters=filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding=padding,
                        activation=activation,
                        data_format=data_format)(inputs)
    pooling = Pool(conv_layer)  
    if conv_dropout:
      batch_norm = BatchNormalization()(pooling)
      return batch_norm 
  return pooling


def build_convolutional_model(filters, kernel_size, padding, data_format, classes, **kwargs):
  layers = kwargs.get('layers', 1)
  fc_dropout = kwargs.get('fc_dropout', 0)
  conv_dropout = kwargs.get('conv_dropout', False)
  fc1 = kwargs.get('fc1', False)
  loss_l2 = kwargs.get('loss_l2', 0)
  lr = kwargs.get('lr', 0.001)
  conv_block = None

  print('Creating model...')
  inputs = Input(shape=(32, 32, 3))
  for layer in range(layers):
    if layer == 0:
      inpts=inputs
    else:
      inpts=conv_block
    
    conv_block = convolutional_layer_with_pooling(inputs=inpts,
                                                  filters=32,
                                                  kernel_size=(3,3),
                                                  strides=(1,1),
                                                  padding='valid',
                                                  data_format=None,
                                                  pooling='max',
                                                  conv_dropout=conv_dropout,
                                                  activation='relu',
                                                  pool_size=(2,2)
                                                  )

  model = Model(inputs, conv_block) 
  return model
  
  
if __name__ == '__main__':
  model = build_convolutional_model(32, (3,3), 'valid', None, 3, layers=3, conv_dropout=True)
  print(model.summary())