from keras.models import load_model
from keras import backend as K


class Autoencoder(object):
    def __init__(self, model_file_name):
        self.model = load_model(model_file_name)

        num_layers = len(self.model.layers)
        middle_layer = int(num_layers/2)
        self.encoder = K.function(
            [self.model.layers[0].input],
            [self.model.layers[middle_layer].output]
        )

    def get_encoding(self, vectors):
        # activate the network until the smallest hidden layer
        return self.encoder([vectors])[0]
