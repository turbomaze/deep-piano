from timeline import Hit, Timeline
from musical.theory import Note, Chord
from musical.audio import save


def get_timeline_from_hlr(hlr):
    timeline = Timeline()
    for t, chord in hlr:
        for note in chord.notes:
            timeline.add(t, Hit(note, 1.0))
    return timeline

def save_timeline_to_wav(timeline, out):
    data = 0.25 * timeline.render()
    save.save_wave(data, out)
