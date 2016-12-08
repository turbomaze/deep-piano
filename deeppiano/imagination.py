import numpy as np
from musical.theory import Note, Chord
from context import deeppiano as dp
import random as rand


class Imagination(object):
    # set up the autoencoder and some parameters for the prior
    def __init__(self, autoencoder, frame_width):
        self.encoder = autoencoder.get_encoding
        self.frame_width = frame_width

        # prior assumptions
        # 4-chord chord progression under a melody
        self.num_chords = 4
        self.notes_per_chord = 4

        # the chord progression repeats
        self.num_repeats = 1

        # melody notes all share the same duration and offset
        self.time_per_note = 0.25

    # given .wav data, returns its musical HLR
    def infer_hlr(self, target_wav, rounds):
        z = self.sample_latent()
        song = self.get_hlr_from_latent(z)
        song_timeline = dp.get_timeline_from_hlr(song)
        wav = dp.get_wav_from_timeline(song_timeline)
        dist = self.get_distance(wav, target_wav)

        for i in range(rounds):
            print 'Round %d loss: %f' % (i, dist)

            # transition initial song
            z_ = self.transition(z)
            song_ = self.get_hlr_from_latent(z_)
            song_timeline_ = dp.get_timeline_from_hlr(song_)
            wav_ = dp.get_wav_from_timeline(song_timeline_)
            dist_ = self.get_distance(wav_, target_wav)

            if dist > dist_:
                dist, z, song, wav = dist_, z_, song_, wav_

        return song

    # given two .wav datas, returns the MSE of their encodings
    def get_distance(self, wav_a, wav_b):
        # get their spectrograms
        vectors_a = dp.get_vectors_from_data(
            wav_a, self.frame_width
        )
        vectors_b = dp.get_vectors_from_data(
            wav_b, self.frame_width
        )

        # encode wav a and b with the autoencoder
        encoding_a = self.encoder(vectors_a)
        encoding_b = self.encoder(vectors_b)

        # return their squared difference sum
        mse = np.sum(np.square(encoding_b - encoding_a))
        return mse

    # returns a random assignment of the latent variables
    def sample_latent(self):
        z = []

        # prior: chords centered normally around third octave
        chord_center = 3 * len(Note.NOTES)
        for i in range(self.num_chords):
            # indices corresponding to notes on a keyboard
            z.append(self.sample_note(chord_center, 5))

            # major or minor
            z.append(rand.random() > 0.5)

        # prior: consistent number of melody notes per chord
        melody_center = 4 * len(Note.NOTES)
        num_notes = self.num_chords * self.notes_per_chord
        for i in range(self.num_repeats):
            for j in range(num_notes):
                z.append(self.sample_note(melody_center, 5))

        return z

    # sample a note from a normal distribution; bounded [0, 100]
    def sample_note(self, mean, stddev):
        # cap from 0 to 100; approximately octaves 0 to 8
        note = np.random.normal(mean, stddev)
        return min(max(int(round(note)), 0), 100)

    # given latent variables, modifies it according to a prior
    def transition(self, latent):
        # TODO: metropolis style updates
        return self.sample_latent()

    # given a list of latent variables, returns the song HLR
    def get_hlr_from_latent(self, latent):
        song = []
        t = 0
        for i in range(self.num_repeats):
            for j in range(self.num_chords):
                chord_root = latent[2 * j]
                is_major = latent[2 * j + 1]
                chord = self.get_chord(chord_root, is_major)
                song.append((
                    t * self.time_per_note,
                    chord,
                    self.time_per_note * self.notes_per_chord
                ))
                t += self.notes_per_chord

        num_melody_notes = self.num_repeats * self.num_chords
        num_melody_notes *= self.notes_per_chord
        offset = 2 * self.num_chords
        for i in range(num_melody_notes):
            song.append((
                i * self.time_per_note,
                Chord([self.get_note(latent[offset + i])]),
                self.time_per_note
            ))

        return song

    # returns a Chord given its root and major/minor status
    def get_chord(self, root, is_major):
        third = root + 4 if is_major else root + 3
        fifth = root + 7
        return Chord([
            self.get_note(root),
            self.get_note(third),
            self.get_note(fifth)
        ])

    # given an abstract note id, return a Note
    def get_note(self, root):
        note = Note.NOTES[root % len(Note.NOTES)]
        octave = root / len(Note.NOTES)
        return Note('%s%d' % (note, octave))
