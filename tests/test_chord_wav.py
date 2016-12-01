from musical.theory import Note, Chord
from context import deeppiano as dp

song = dp.get_timeline_from_hlr([
    (0, Chord.minor(Note('a3'))),
    (1, Chord.major(Note('f3'))),
    (2, Chord.major(Note('c3'))),
    (3, Chord.major(Note('g3')))
])
file_name = '../data/song.wav'
dp.save_timeline_to_wav(song, file_name)
dp.get_vectorized_wav(file_name, 2048)
