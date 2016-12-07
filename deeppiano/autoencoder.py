from keras.models import load_model
from keras import backend as K


class Autoencoder(object):
    def __init__(self, model_file_name):
        self.model = load_model(model_file_name)

        # hardcodes the idea of a shallow autoencoder
        self.encoder = K.function(
            [self.model.layers[0].input],
            [self.model.layers[1].output]
        )
        self.decoder = K.function(
            [self.model.layers[1].output],
            [self.model.layers[2].output]
        )

    def get_encoding(self, vectors):
        # activate the network until the smallest hidden layer
        return self.encoder([vectors])[0]

    def get_decoding(self, encoding):
        # activate all layers after the smallest hidden layer
        return self.decoder([encoding])[0]
