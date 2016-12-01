import sys
from timeline import Hit, Timeline
from musical.theory import Note, Chord
from musical.audio import save


def get_timeline_from_hlr(hlr):
    timeline = Timeline()
    for t, chord in hlr:
        for note in chord.notes:
            timeline.add(t, Hit(note, 1.0))
    return timeline

song = get_timeline_from_hlr([
    (0, Chord.minor(Note('a3'))),
    (1, Chord.major(Note('f3'))),
    (2, Chord.major(Note('c3'))),
    (3, Chord.major(Note('g3')))
])
data = 0.25 * song.render()
save.save_wave(data, 'data/song.wav')
