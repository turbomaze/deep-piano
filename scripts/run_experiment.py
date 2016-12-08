from timeit import default_timer as timer
from context import Autoencoder
from context import Imagination
from context import deeppiano as dp
import os
import sys
import numpy as np
import cPickle

if len(sys.argv) < 2:
    print 'You must specify the root experiments directory.'
if len(sys.argv) < 3:
    print 'You must specify the # notes per chord.'
if len(sys.argv) < 4:
    print 'You must specify the # progression repeats.'
if len(sys.argv) < 5:
    print 'You must specify the # inference rounds per song.'
if len(sys.argv) < 6:
    print 'You must specify the # songs to infer.'
if len(sys.argv) < 7:
    print 'You must specify the autoencoder model.'
    sys.exit(0)

root_dir = sys.argv[1]
notes_per_chord = int(sys.argv[2])
num_repeats = int(sys.argv[3])
inference_rounds = int(sys.argv[4])
num_songs = int(sys.argv[5])
model_name = sys.argv[6]
frame_width = 2048
experiment_id = np.base_repr(int(9**4 * np.random.rand()), 36)

start = timer()

# create a directory to save all the data
dir_name = '%s/experiment-%s' % (root_dir, experiment_id)
target_dir_name = '%s/target-songs' % dir_name
guess_dir_name = '%s/guess-songs' % dir_name
os.makedirs(dir_name)
os.makedirs(target_dir_name)
os.makedirs(guess_dir_name)

# load the autoencoder and set up the imagination
print 'Loading autoencoder from "%s"' % model_name
autoencoder = Autoencoder(model_name)
imagination = Imagination(autoencoder, frame_width)

# run inference against each song
results = []
for i in range(num_songs):
    start_song = timer()
    song_name = 'song-%s-%d.wav' % (experiment_id, i)

    # generate the song hlr
    target_hlr = dp.generate_song(notes_per_chord, num_repeats)

    # synthesize the target hlr and save it to the target dir
    target_name = '%s/target-%s' % (target_dir_name, song_name)
    target_timeline = dp.get_timeline_from_hlr(target_hlr)
    target_wav = dp.get_wav_from_timeline(target_timeline)
    dp.save_timeline_to_wav(target_timeline, target_name)

    # perform inference on the wav
    guess_hlr, obs_loss, real_loss = imagination.infer_hlr(
        inference_rounds,
        target_wav,
        target_hlr
    )

    # synthesize the guess hlr and save it to the guess dir
    guess_name = '%s/guess-%s' % (guess_dir_name, song_name)
    guess_timeline = dp.get_timeline_from_hlr(guess_hlr)
    dp.save_timeline_to_wav(guess_timeline, guess_name)

    # reporting
    end_song = timer()
    duration = end_song - start_song
    print 'Song %d inferred in %fs with:' % (i, duration)
    print 'obs loss: %f, real loss: %f' % (obs_loss, real_loss)
    print

    # add the guess hlr/target hlr/loss to the results array
    results.append({
        'experiment_id': experiment_id,
        'autoencoder_model': model_name,
        'song': song_name,
        'song_hlr': target_hlr,
        'inferred_hlr': guess_hlr,
        'observed_loss': obs_loss,
        'real_loss': real_loss,
        'inference_duration': duration,
        'inference_rounds': inference_rounds
    })

# pickle the results array to the experiments dir
print 'Saving results array to the experiments folder...'
results_name = '%s/results-%s.p' % (dir_name, experiment_id)
cPickle.dump(results, open(results_name, 'wb'))
print

# print the results to the console for convenience
summaries = map(lambda song_result: {
    'song': song_result['song'],
    'obs_loss': round(song_result['observed_loss'], 2),
    'real_loss': round(song_result['real_loss'], 2),
    'dur': round(song_result['inference_duration'], 2),
}, results)
for summary in summaries:
    print summary

# timing
end = timer()
print 'Experiment "%s" completed in %fs' % (
    experiment_id, end - start
)
