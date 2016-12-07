# code initially based on:
# https://blog.keras.io/building-autoencoders-in-keras.html

from keras.layers import Input, Dense
from keras.models import Model
from context import deeppiano as dp
import numpy as np
import cPickle

# location of the spectral data
input_file_name = '../data/spectrums/spectralized-songs.p'

# this is the size of our inputs
input_dim = 1025

# this is the size of our encoded representations
encoding_dim = 128

# number of epochs to train for
num_epochs = 10

# number of training vectors per epoch
epoch_batch_size = 100

# location of the model file to output
model_name = np.base_repr(int(9**4 * np.random.rand()), 36)
output_file_name = '../models/%s.%d-%d-%d.%dep-%dba.h5' % (
    model_name,
    input_dim, encoding_dim, input_dim,
    num_epochs, epoch_batch_size
)


def load_spectrogram_data():
    # load the data
    raw_data = cPickle.load(open(input_file_name, 'rb'))
    raw_data = np.array(raw_data)

    # reshape the data
    data_shape = raw_data.shape
    num_frames = data_shape[0] * data_shape[1]
    raw_data = np.array(raw_data).reshape(
        (num_frames, data_shape[2])
    )

    # normalize the data
    largest_magnitude = np.amax(raw_data)
    raw_data = raw_data / largest_magnitude

    # split into train and test data
    cutoff = int(num_frames * 0.8)
    train = raw_data[:cutoff]
    test = raw_data[cutoff:]
    return ((train, False), (test, False))

# this is our input placeholder
input_frame = Input(shape=(input_dim,))

# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_frame)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(input_dim, activation='linear')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input=input_frame, output=decoded)

# this model maps an input to its encoded representation
encoder = Model(input=input_frame, output=encoded)

# create a placeholder for an encoded input
encoded_input = Input(shape=(encoding_dim,))

# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]

# create the decoder model
decoder = Model(
    input=encoded_input,
    output=decoder_layer(encoded_input)
)

autoencoder.compile(
    # optimizer='adadelta', loss='binary_crossentropy'
    optimizer='adadelta', loss='mean_squared_error'
)

# train the autoencoder on the spectrogram data
(x_train, _), (x_test, _) = load_spectrogram_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

autoencoder.fit(
    x_train,
    x_train,
    nb_epoch=num_epochs,
    batch_size=epoch_batch_size,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# visualize a particular spectrogram
spectrogram = x_test[:91]
encoded_frames = encoder.predict(spectrogram)
decoded_frames = decoder.predict(encoded_frames)

# show spectrogram of some wavs
dp.plot_mangogram(np.transpose(spectrogram))
dp.plot_mangogram(np.transpose(decoded_frames))
