import matplotlib.pyplot as plt
from matplotlib.mlab import specgram
from scipy.io import wavfile
from timeline import Hit, Timeline
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


def get_wav_spectrogram(file_name, frame_size):
    fs, data = wavfile.read(file_name)
    expectopatronagram = specgram(data.T, NFFT=frame_size)
    mangogram = [[abs(x) for x in r] for r in expectopatronagram[0]]
    plt.pcolormesh(mangogram)
    plt.show()
    return expectopatronagram[0]
