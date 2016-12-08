import sys
from timeit import default_timer as timer
import random as rand
from context import deeppiano as dp
from multiprocessing import Process

if len(sys.argv) < 2:
    print 'You must supply a "number of songs" argument.'
    sys.exit(0)

if len(sys.argv) < 3:
    print 'You must supply a "time per note" argument.'
    sys.exit(0)

root = '../data/songs'
num_threads = 2
total_songs = int(sys.argv[1])
time_per_note = float(sys.argv[2])
letter = unichr(65 + rand.randrange(26))
song_count = 0


def generate_and_save_song():
    global song_count
    global total_songs
    global time_per_note
    global letter
    global root

    notes_per_chord = 4
    num_repeats = 1

    while song_count < total_songs:
        song_count += 1
        next_song_idx = song_count - 1
        song = dp.generate_song(
            notes_per_chord, num_repeats, time_per_note
        )
        song_timeline = dp.get_timeline_from_hlr(song)
        file_name = '%s/gen-song-%s-%d.wav' % (
            root, letter, next_song_idx
        )
        dp.save_timeline_to_wav(song_timeline, file_name)
        print '\nSaved %s' % file_name

start_time = timer()

processes = []
for _ in range(num_threads):
    p = Process(target=generate_and_save_song)
    processes.append(p)
    p.start()

for process in processes:
    process.join()

end_time = timer()
print 'Time elapsed: %fs' % (end_time - start_time)
