from context import deeppiano as dp

song = dp.generate_song(4, 1, 0.5)
song_timeline = dp.get_timeline_from_hlr(song)
file_name = '../data/test-songs/song_test.wav'
dp.save_timeline_to_wav(song_timeline, file_name)
