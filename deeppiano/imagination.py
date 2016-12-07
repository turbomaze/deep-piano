class Imagination(object):
    def __init__(self, autoencoder, frame_width):
        self.encoder = autoencoder.encoder
        self.frame_width

    # given a .wav file location, returns its musical HLR
    def infer_hlr(self, file_name):
        # target song = open file_name
        # initial song = self.sample_song()
        # while get_distance(initial song, target song) is large
        # transition initial song
        pass

    # given two .wav files, returns the MSE of their encodings
    def get_distance(self, wav_a, wav_b):
        # encode wav a and b with the autoencoder
        # return their squared difference sum
        pass

    # returns a randomly sampled HLR
    def sample_song(self):
        # sample via a similar process used to generate songs
        pass

    # given a HLR, modifies it slightly according to a prior
    def transition(self, hlr):
        # pick a dimension (a melody note or chord) to change
        # change it according to some distribution
        pass
