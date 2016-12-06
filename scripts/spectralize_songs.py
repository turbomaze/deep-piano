from os import walk
import cPickle
import numpy as np
from timeit import default_timer as timer
from context import deeppiano as dp

frame_size = 2048
songs_dir = '../data/songs'
output_file = '../data/spectrums/spectralized-songs.p'

start = timer()

# get all the song file names in the songs directory
song_file_names = [
    songs_dir + '/' + f for f in walk(songs_dir).next()[2]
]

# compute their raw spectrograms
spectralized_songs = map(
    lambda x: dp.get_vectorized_wav(x, frame_size),
    song_file_names
)
print 'Dimensions of the spectralized songs'
print np.array(spectralized_songs).shape

# normalize the spectrograms by dividing by their max

# dump the resulting structure to a file
cPickle.dump(spectralized_songs, open(output_file, 'wb'))

end = timer()
print 'Time elapsed: %fs' % (end - start)
