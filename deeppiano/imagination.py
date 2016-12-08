import numpy as np
from musical.theory import Note, Chord
from context import deeppiano as dp
import random as rand


class Imagination(object):
    # set up the autoencoder and some parameters for the prior
    def __init__(
        self,
        autoencoder,
        frame_width,
        num_chords=4,
        time_per_note=0.25
    ):
        self.encoder = autoencoder.get_encoding
        self.frame_width = frame_width

        # could easily be inferred in the future
        self.num_chords = num_chords

        # prior assumptions about the melody
        self.notes_per_chord = 4

        # the chord progression repeats
        self.num_repeats = 1

        # melody notes all share the same duration and offset
        self.time_per_note = time_per_note

        # highest note to play
        self.max_note = 100

    # given .wav data, returns its musical HLR
    def infer_hlr(self, rounds, target_wav, target_hlr=None):
        z = self.sample_latent()
        song = self.get_hlr_from_latent(z)
        song_timeline = dp.get_timeline_from_hlr(song)
        wav = dp.get_wav_from_timeline(song_timeline)
        obs_dist = self.get_distance(wav, target_wav)
        real_dist = self.get_real_distance(song, target_hlr)

        for i in range(rounds):
            print 'Round %d' % i
            print 'observed loss: %f, real loss: %f' % (
                obs_dist,
                real_dist
            )
            print

            # transition initial song
            z_ = self.transition(z, i)
            song_ = self.get_hlr_from_latent(z_)
            song_timeline_ = dp.get_timeline_from_hlr(song_)
            wav_ = dp.get_wav_from_timeline(song_timeline_)
            obs_dist_ = self.get_distance(wav_, target_wav)
            real_dist_ = self.get_real_distance(
                song_,
                target_hlr
            )

            if obs_dist > obs_dist_:
                obs_dist, real_dist = obs_dist_, real_dist_
                z, song, wav = z_, song_, wav_

        return song, obs_dist, real_dist

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

    # given two Timelines, return their 'Levenstein' distance
    @classmethod
    def get_real_distance(cls, a, b):
        if a is None or b is None:
            return float('inf')

        # construct a dictionary keyed by hit time
        songs_by_time = {}
        for hit in a:
            if hit[0] not in songs_by_time:
                songs_by_time[hit[0]] = ([], [])
            songs_by_time[hit[0]][0].append(tuple(hit[1:]))
        for hit in b:
            if hit[0] not in songs_by_time:
                songs_by_time[hit[0]] = ([], [])
            songs_by_time[hit[0]][1].append(tuple(hit[1:]))

        # sum the differences of each corresponding slot
        real_dist = 0
        for slot in songs_by_time:
            U, V = songs_by_time[slot]
            real_dist += cls.get_real_universe_dist(U, V)

        # compute the normed real dist; roughly, error per note
        normed_real_dist = (real_dist/max(len(a), len(b)))**0.5
        return normed_real_dist

    # given 2 lists of mutually concurrent notes,
    # computes the distance between all pairs between the lists
    @classmethod
    def get_real_universe_dist(cls, U, V):
        zero_hit = (Chord([Note('c0')]), 0)
        if len(U) == 0:
            U = [zero_hit]
        if len(V) == 0:
            V = [zero_hit]

        # sum(dist(u,v) for all u,v)
        total_sum = 0
        for u in U:
            for v in V:
                total_sum += cls.get_real_pairwise_dist(u, v)

        normed_sum = total_sum / (len(U) * len(V))
        return normed_sum

    # computes the L2-distance b/w two notes by pitch/duration
    @classmethod
    def get_real_pairwise_dist(cls, u, v):
        c_pitch = 4.0
        c_time = 1.0

        chord_u, chord_v = u[0], v[0]
        duration_u, duration_v = u[1], v[1]
        pitch_dist = cls.get_real_chord_dist(chord_u, chord_v)
        time_dist = (duration_u - duration_v)**2
        return c_pitch * pitch_dist + c_time * time_dist

    # computes the distance between two chords
    @classmethod
    def get_real_chord_dist(cls, a, b):
        chord_dist = 0
        for note_a in a.notes:
            for note_b in b.notes:
                chord_dist += (note_a.index - note_b.index)**2
        return chord_dist

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
        return min(max(int(round(note)), 0), self.max_note)

    # given latent variables, modifies it according to a prior
    def transition(self, latent, i):
        # choose a random latent variable to change
        idx = i % len(latent)
        latent_ = latent[:]

        # check to see if we're changing the chord type
        if idx < 2 * self.num_chords and idx % 2 == 1:
            latent_[idx] = not latent[idx]
            return latent_

        # otherwise latent variable is a melody or chord root
        latent_[idx] = self.sample_note(latent[idx], 3)

        # enforce change
        if latent[idx] == latent_[idx]:
            latent_[idx] += 1 if rand.random() < 0.5 else -1
            latent_[idx] = min(
                max(latent_[idx], 0), self.max_note
            )

        return latent_

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
