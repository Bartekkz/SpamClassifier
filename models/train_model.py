import os


def train_model(model, X, y, save_model, path='', **kwargs):
    """
    Trains the model with given params and saves the model architecture
    to json file, and weights to hdf5 format
    :param model: Model -> keras or tensorflow model
    :param X: array or list -> data
    :param y: array or list -> labels for given data
    :param save_model: bool -> save model architecture and weights for later reproduction
    :param path: str -> directory for saving model (ends with .json), if You want to save model and weights to
    different directories You can omit this parameter and use "model_path" and "weights_path"
    :param kwargs:
    :return: Model
    """
    # TODO: add some callbacks
    epochs = kwargs.get('epochs', 4)
    batch_size = kwargs.get('batch_size', 16)
    class_weight = kwargs.get('class_weight', None)
    validation_split = kwargs.get('validation_split', 0.2)
    model_path = kwargs.get('model_path', os.path.join(path, 'model_json.json'))
    weights_path = kwargs.get('weights_path', os.path.join(path, 'model_weights.h5'))

    assert model_path[-5:] == '.json'
    assert weights_path[-3:] == '.h5'

    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, class_weight=class_weight)
    model_json = model.to_json()

    if os.path.exists(model_path):
        raise FileExistsError
    else:
        with open(model_path, 'w') as json_file:
            json_file.write(model_json)

    if os.path.exists(weights_path):
        raise FileExistsError
    else:
        model.save_weights(weights_path)

    return model

