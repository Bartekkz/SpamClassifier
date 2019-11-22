

def convolutional_layer(filters, kernel_size, strides, padding, data_format):
  '''
  creates Keras based convolutional layer
  
  :param filters: int -> nubmer of output filters in convolution
  :param kernel_size: int or tuple/list of 2 int's -> the height and width of the 2D convolution window
  :param strides: int or tuple/list of 2 int's -> the height and width of the convolution
  :param padding: string: type of padding
  :param data_format: order of the dimensions in the inputs
  
  :returns: keras.layers.Conv2d
  : #TODO: add error raising
  '''