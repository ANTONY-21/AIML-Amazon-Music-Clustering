import unittest
import pandas as pd
from music_similarity import find_matches, nearest_in_cluster


class MusicSimilarityTest(unittest.TestCase):
    def setUp(self):
        self.songs = pd.DataFrame({'name_song': ['First', 'Close', 'Far', 'Other cluster'],
                                   'name_artists': ['An Artist', 'B', 'C', 'D'],
                                   'popularity_songs': [20, 1, 90, 100],
                                   'cluster': [0, 0, 0, 1], 'a': [0., .1, 8., .01], 'b': [0., .1, 8., .01]})

    def test_nearest_is_audio_distance_not_popularity_and_stays_in_cluster(self):
        out = nearest_in_cluster(self.songs, ['a', 'b'], 0)
        self.assertEqual(out['name_song'].tolist(), ['Close', 'Far'])
        self.assertLess(out.iloc[0]['audio_distance'], out.iloc[1]['audio_distance'])

    def test_search_is_case_insensitive_literal_and_includes_artist(self):
        self.assertEqual(find_matches(self.songs, 'ARTIST')['name_song'].tolist(), ['First'])
        self.assertTrue(find_matches(self.songs, '.*').empty)
        self.assertTrue(find_matches(self.songs, ' ').empty)


if __name__ == '__main__': unittest.main()
