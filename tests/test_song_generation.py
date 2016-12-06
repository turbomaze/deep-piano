from context import deeppiano as dp

song = dp.generate_song(4, 1, 0.5)
file_name = '../data/test-songs/song_test.wav'
dp.save_timeline_to_wav(song, file_name)
