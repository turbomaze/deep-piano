from context import Imagination
from context import Autoencoder
from scipy.io import wavfile

model_name = '../models/8M.1025-128-1025.60ep-100ba.h5'
song_prefix = '../data/songs/gen-song-I-'
song_suffix = '.wav'
frame_width = 2048
num_to_compare = 15

autoencoder = Autoencoder(model_name)
imagination = Imagination(autoencoder, frame_width)

wavs = [
    wavfile.read(song_prefix + str(i) + song_suffix)[1]
    for i in range(100)
]

smallest_dist = float('inf')
biggest_dist = float('-inf')
smallest_dist_pair = [-1, -1]
biggest_dist_pair = [-1, -1]

count = 0
for i in range(num_to_compare):
    for j in range(i+1, num_to_compare):
        count += 1
        if count % 10 == 0:
            print 'Total compared: %d' % count

        dist = imagination.get_distance(wavs[i], wavs[j])
        if dist < smallest_dist:
            smallest_dist = dist
            smallest_dist_pair = [i, j]
        if dist > biggest_dist:
            biggest_dist = dist
            biggest_dist_pair = [i, j]

print 'smallest'
print smallest_dist_pair
print 'biggest'
print biggest_dist_pair
