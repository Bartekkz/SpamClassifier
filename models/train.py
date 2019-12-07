import os


def train_model(model, X, y, save_model, **kwargs):
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
    model_path = kwargs.get('model_path', 'model_json.json')
    weights_path = kwargs.get('weights_path', 'model_weights.h5')
    shuffle = kwargs.get('shuffle', True)
    save = kwargs.get('save_model', True)

    data_path = os.path.abspath('data')

    # TODO: fix paths
    model_path = os.path.join(data_path, model_path)
    weights_path = os.path.join(data_path, weights_path)

    print(model_path)

    assert model_path[-5:] == '.json'
    assert weights_path[-3:] == '.h5'


    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, class_weight=class_weight, shuffle=shuffle)
    model_json = model.to_json()

    if save:
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

