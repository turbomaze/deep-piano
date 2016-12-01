import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import specgram
from scipy.io import wavfile
from timeline import Hit, Timeline
from musical.audio import save


# given a high level representation of a song, returns the Timeline
def get_timeline_from_hlr(hlr):
    timeline = Timeline()
    for t, chord in hlr:
        for note in chord.notes:
            timeline.add(t, Hit(note, 1.0))
    return timeline


# given the Timeline of a song, saves that song to a .wav file
def save_timeline_to_wav(timeline, out):
    data = 0.25 * timeline.render()
    save.save_wave(data, out)


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
