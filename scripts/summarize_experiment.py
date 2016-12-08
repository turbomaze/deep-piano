import sys
import cPickle

if len(sys.argv) < 2:
    print 'You must specify the results file.'

results_name = sys.argv[1]
results = cPickle.load(open(results_name, 'rb'))

summaries = map(lambda song_result: {
    'song': song_result['song'],
    'obs_loss': round(song_result['observed_loss'], 2),
    'real_loss': round(song_result['real_loss'], 2),
    'dur': round(song_result['inference_duration'], 2),
}, results)
for summary in summaries:
    print summary
