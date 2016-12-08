from timeit import default_timer as timer
from context import Imagination
from context import Autoencoder
from context import deeppiano as dp
import numpy as np

model_name = '../models/8M.1025-128-1025.60ep-100ba.h5'
frame_width = 2048
num_rounds = 1000
target_song_prefix = '../data/test-songs/target-song-'
guess_song_prefix = '../data/test-songs/guess-song-'
suffix = '.wav'
session_id = np.base_repr(int(9**4 * np.random.rand()), 36)

start = timer()

# load the autoencoder and set up the imagination
print 'Loading autoencoder from "%s"' % model_name
autoencoder = Autoencoder(model_name)
imagination = Imagination(autoencoder, frame_width)

# generate a target song to learn and save it
print 'Generating target song for session %s' % session_id
target_hlr = dp.generate_song(4, 1, 0.25)
target_timeline = dp.get_timeline_from_hlr(target_hlr)
target_wav = dp.get_wav_from_timeline(target_timeline)
dp.save_timeline_to_wav(
    target_timeline,
    target_song_prefix + session_id + suffix
)

# run inference for a bit
print 'Beginning inference on target song %s' % session_id
guess_hlr = imagination.infer_hlr(target_wav, num_rounds)

# save the inferred HLR as a .wav
guess_timeline = dp.get_timeline_from_hlr(guess_hlr)
dp.save_timeline_to_wav(
    guess_timeline,
    guess_song_prefix + session_id + suffix
)

# logs
end = timer()
print 'Inference completed for %s in %fs' % (
    session_id, end - start
)
