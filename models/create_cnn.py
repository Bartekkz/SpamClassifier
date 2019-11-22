#!/usr/bin/env python3
from keras.layers import Conv2D
from keras.models import Sequential


def convolutional_layer_with_pooling(filters, kernel_size, strides, padding, data_format=None, pooling='max'):
  '''
  creates Keras based convolutional layer
  
  :param filters: int -> nubmer of output filters in convolution
  :param kernel_size: int or tuple/list of 2 int's -> the height and width of the 2D convolution window
  :param strides: int or tuple/list of 2 int's -> the height and width of the convolution
  :param padding: string: type of padding
  :param data_format: order of the dimensions in the inputs
  :param pooling: type of pooling
  
  :returns: keras.layers.Conv2d
  : #TODO: add error raising
  '''
  conv_layer = Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding=padding,
                      data_format=dataformat)


def build_convolutional_model(filters, kernel_size, padding, data_format, classes, **kwargs):
  layers = kwargs.get('layers', 1)
  dropout = kwargs.get('dropout', 0)
  fc1 = kwargs.get('fc1', False)
  loss_l2 = kwargs.get('loss_l2', 0)
  lr = kwargs.get('lr', 0.001)

  print('Creating model...')
  model = Sequential()
