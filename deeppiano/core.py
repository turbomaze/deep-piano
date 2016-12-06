"""
Deep Piano
@author Anthony Liu <igliu@mit.edu>
@version 0.1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import specgram
from scipy.io import wavfile
from timeline import Hit, Timeline
from musical.audio import save
from musical.theory import Chord, Note, Scale
import random as rand


# given a high level representation of a song, returns the Timeline
def get_timeline_from_hlr(hlr):
    timeline = Timeline()
    for t, chord, duration in hlr:
        for note in chord.notes:
            timeline.add(t, Hit(note, duration))
    return timeline


# given the Timeline of a song, saves that song to a .wav file
def save_timeline_to_wav(timeline, out):
    data = 0.25 * timeline.render()
    save.save_wave(data, out)


# randomly generates a song and returns its timeline
def generate_song(notes_per_chord, num_repeats, note_time=1.0):
    # generate a random major key
    root = Note(rand.choice(Note.NOTES))
    scale_notes = Scale(root, 'major')

    # generate a I-V-vi-IV progression
    octave = 3
    intervals = (7, 2, -4, 5)
    progression = Chord.progression(Scale(root, intervals), octave)
    progression[2] = major_to_minor(progression[2])

    # generates a melody for the progression
    low_octave = 4
    prev_note = rand.choice(list(scale_notes)).at_octave(low_octave)
    melody = []
    for _ in range(notes_per_chord * len(progression) * num_repeats):
        note_dist = int(round(rand.gauss(0, 2)))
        prev_note = scale_notes.transpose(prev_note, note_dist)
        melody.append(prev_note)

    # build up the HLR from the melody progression
    song = []
    t = 0
    for i in range(num_repeats):
        for chord in progression:
            song.append(
                (t*note_time, chord, note_time*notes_per_chord)
            )
            t += notes_per_chord

    for i in range(len(melody)):
        note = melody[i]
        song.append((i*note_time, Chord([note]), note_time))

    return get_timeline_from_hlr(song)


# given a .wav file name, returns its audio spectrogram
def get_wav_spectrogram(file_name, frame_size):
    fs, data = wavfile.read(file_name)
    expectopatronagram = specgram(data.T, NFFT=frame_size)
    return expectopatronagram[0]


# given a spectrogram, plots the frequency magnitudes
def plot_spectrogram(spectrogram):
    mangogram = [[abs(x) for x in r] for r in spectrogram]
    plt.pcolormesh(mangogram)
    plt.show()


# given a .wav file name, returns vectors for the autoencoder
def get_vectorized_wav(file_name, frame_size):
    spectrogram = get_wav_spectrogram(file_name, frame_size)
    mangogram = [[abs(x) for x in r] for r in spectrogram]

    # autoencoder vectors are lists of frequency magnitudes
    return np.transpose(mangogram)


def major_to_minor(chord):
    root = chord.notes[0]
    third = chord.notes[1]
    minor_third = third.transpose(-1)
    fifth = chord.notes[2]
    return Chord([root, minor_third, fifth])
