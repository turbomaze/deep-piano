"""
Deep Piano
@author Anthony Liu <igliu@mit.edu>
@version 1.0.0
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import specgram
from scipy.io import wavfile
from timeline import Hit, Timeline
from musical.audio import save, encode
from musical.theory import Chord, Note, Scale
import random as rand


# given a high level representation of a song, returns the Timeline
def get_timeline_from_hlr(hlr):
    timeline = Timeline()
    for t, chord, duration in hlr:
        for note in chord.notes:
            timeline.add(t, Hit(note, duration))
    return timeline


# given the Timeline of a song, returns that song's .wav data
def get_wav_from_timeline(timeline):
    data = 0.25 * timeline.render()
    data = encode.as_int16(data)
    return np.array(data)


# given the Timeline of a song, saves that song to a .wav file
def save_timeline_to_wav(timeline, out):
    data = 0.25 * timeline.render()
    save.save_wave(data, out)


# randomly generates a song and returns its timeline
def generate_song(
    notes_per_chord,
    num_repeats,
    note_time=0.25,
    prog_intervals=(7, 2, -4j, 5)
):
    # generate a random major key
    root = Note(rand.choice(Note.NOTES))
    scale_notes = Scale(root, 'major')

    octave = 3
    progression = Chord.progression(
        Scale(
            root,
            [int(p.real + p.imag) for p in prog_intervals]
        ),
        octave
    )
    for i, z in enumerate(prog_intervals):
        if z.imag != 0:
            # TODO: cannot have a repeated chord be minor
            progression[i] = major_to_minor(progression[i])

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

    return song


# given a .wav file name, returns its audio spectrogram
def get_wav_spectrogram(file_name, frame_size):
    fs, data = wavfile.read(file_name)
    return get_spectrogram_from_data(data, frame_size)


# given .wav data, returns its audio spectrogram
def get_spectrogram_from_data(data, frame_size):
    expectopatronagram = specgram(data.T, NFFT=frame_size)
    return expectopatronagram[0]


# given a spectrogram, plots the frequency magnitudes
def plot_spectrogram(spectrogram):
    mangogram = [[abs(x) for x in r] for r in spectrogram]
    plot_mangogram(mangogram)


# plot given a magnitude spectrogram
def plot_mangogram(mangogram, title='Magnitude spectrogram'):
    plt.pcolormesh(mangogram)
    plt.title(title)
    plt.xlim([0, 91])
    plt.xlabel('Frame index')
    plt.ylim([0, 1025])
    plt.ylabel('Frequency bin')
    plt.show()


# given a .wav file name, returns vectors for the autoencoder
def get_vectorized_wav(file_name, frame_size):
    fs, data = wavfile.read(file_name)
    return get_vectors_from_data(data, frame_size)


# given a .wav file name, returns vectors for the autoencoder
def get_vectors_from_data(data, frame_size):
    spectrogram = get_spectrogram_from_data(
        data, frame_size
    )
    mangogram = [
        [np.log(abs(x)) for x in r] for r in spectrogram
    ]

    # autoencoder vectors are lists of frequency magnitudes
    return np.transpose(mangogram)


def major_to_minor(chord):
    root = chord.notes[0]
    third = chord.notes[1]
    minor_third = third.transpose(-1)
    fifth = chord.notes[2]
    return Chord([root, minor_third, fifth])
