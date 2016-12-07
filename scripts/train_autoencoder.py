# code initially based on:
# https://blog.keras.io/building-autoencoders-in-keras.html

from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import cPickle
import sys

if len(sys.argv) < 2:
    print 'You must specify the location of the spectrum.'
if len(sys.argv) < 3:
    print 'You need to specify the hidden layer size.'
if len(sys.argv) < 4:
    print 'You need to specify the number of epochs.'
if len(sys.argv) < 5:
    print 'You need to specify the batch size.'
    sys.exit(0)


def load_spectrogram_data(raw):
    # reshape the data
    data_shape = raw.shape
    num_frames = data_shape[0] * data_shape[1]
    raw = np.array(raw).reshape(
        (num_frames, data_shape[2])
    )

    # normalize the data
    largest_magnitude = np.amax(raw_data)
    raw = raw / largest_magnitude

    # split into train and test data
    cutoff = int(num_frames * 0.8)
    train = raw[:cutoff]
    test = raw[cutoff:]
    return ((train, False), (test, False))

# location of the spectral data
input_file_name = sys.argv[1]

# unpickle the file
raw_data = cPickle.load(open(input_file_name, 'rb'))
raw_data = np.array(raw_data)

# load the spectrogram data
(x_train, _), (x_test, _) = load_spectrogram_data(raw_data)

# cast them as floats
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# controls the size of the autoencoder layers
input_dim = raw_data.shape[2]
encoding_dim = int(sys.argv[2])

# parameters for the epochs
num_epochs = int(sys.argv[3])
epoch_batch_size = int(sys.argv[4])

# generate a file name for the model file to output
model_name = np.base_repr(int(9**4 * np.random.rand()), 36)
output_file_name = '../models/%s.%d-%d-%d.%dep-%dba.h5' % (
    model_name,
    input_dim, encoding_dim, input_dim,
    num_epochs, epoch_batch_size
)

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
    optimizer='adadelta', loss='mean_squared_error'
)

# train the autoencoder
autoencoder.fit(
    x_train,
    x_train,
    nb_epoch=num_epochs,
    batch_size=epoch_batch_size,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# save the model to a file
print 'Saving to %s...' % output_file_name
autoencoder.save(output_file_name)
