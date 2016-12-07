import numpy as np
from context import deeppiano as dp


class Imagination(object):
    def __init__(self, autoencoder, frame_width):
        self.encoder = autoencoder.get_encoding
        self.frame_width = frame_width

    # given .wav data, returns its musical HLR
    def infer_hlr(self, song_wav):
        # initial song = self.sample_song()
        # while get_distance(initial song, target song) is large
        # transition initial song
        pass

    # given two .wav datas, returns the MSE of their encodings
    def get_distance(self, wav_a, wav_b):
        # get their spectrograms
        vectors_a = dp.get_vectors_from_data(
            wav_a, self.frame_width
        )
        vectors_b = dp.get_vectors_from_data(
            wav_b, self.frame_width
        )

        # encode wav a and b with the autoencoder
        encoding_a = self.encoder(vectors_a)
        encoding_b = self.encoder(vectors_b)

        # return their squared difference sum
        mse = np.sum(np.square(encoding_b - encoding_a))
        return mse

    # returns a randomly sampled HLR
    def sample_song(self):
        # sample via a similar process used to generate songs
        pass

    # given a HLR, modifies it slightly according to a prior
    def transition(self, hlr):
        # pick a dimension (a melody note or chord) to change
        # change it according to some distribution
        pass
