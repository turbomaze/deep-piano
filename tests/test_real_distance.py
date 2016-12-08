from context import Imagination
from context import Autoencoder
from musical.theory import Chord, Note

model_name = '../models/8M.1025-128-1025.60ep-100ba.h5'
frame_width = 2048

# load the autoencoder and set up the imagination
autoencoder = Autoencoder(model_name)
imagination = Imagination(autoencoder, frame_width)

# compare two Timelines
print 'Chord dist: %f' % Imagination.get_real_chord_dist(
    Chord([Note('c4')]),
    Chord([Note('c#4')])
)
print 'Pairwise dist: %f' % Imagination.get_real_pairwise_dist(
    (Chord([Note('c4')]), 1),
    (Chord([Note('c#4')]), 2)
)
print 'Universe dist: %f' % Imagination.get_real_universe_dist(
    [
        (Chord([Note('c#4')]), 1),
        (Chord([Note('c4')]), 1)
    ],
    [
        (Chord([Note('c4')]), 1)
    ]
)
print 'Real dist: %f' % Imagination.get_real_distance([
    (0, Chord([Note('a4')]), 1),
    (0, Chord([Note('g4'), Note('c4')]), 1),
    (1, Chord([Note('b4')]), 1),
], [
    (0, Chord([Note('a4')]), 1),
    (1, Chord([Note('b4')]), 1),
    (2, Chord([Note('c4')]), 1),
])
