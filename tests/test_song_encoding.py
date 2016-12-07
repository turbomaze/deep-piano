from context import deeppiano as dp
from context import Autoencoder
import numpy as np

model_file = '../models/8M.1025-128-1025.60ep-100ba.h5'
song_file = '../data/songs/gen-song-I-0.wav'
frame_width = 2048

autoencoder = Autoencoder(model_file)

mangogram = dp.get_vectorized_wav(song_file, frame_width)

encoding = autoencoder.get_encoding(mangogram)
decoding = autoencoder.get_decoding(encoding)

# compute difference between mangogram and decoding
mse = np.sum(
    np.square(np.array(decoding) - np.array(mangogram))
) / np.prod(mangogram.shape)
print 'Average MSE b/w input and its reconstruction: %f' % mse
dp.plot_mangogram(np.transpose(mangogram))
dp.plot_mangogram(np.transpose(decoding))
