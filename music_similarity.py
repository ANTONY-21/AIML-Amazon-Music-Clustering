"""Find tracks close to a selected song in standardized audio-feature space."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def find_matches(songs, query, limit=100):
    query = query.strip()
    if not query:
        return songs.iloc[0:0]
    title = songs['name_song'].fillna('').astype(str)
    artist = songs['name_artists'].fillna('').astype(str)
    matches = songs[title.str.contains(query, case=False, regex=False) |
                    artist.str.contains(query, case=False, regex=False)]
    if 'popularity_songs' in matches:
        matches = matches.sort_values('popularity_songs', ascending=False, kind='stable')
    return matches.head(limit)


def nearest_in_cluster(songs, features, selected_index, limit=20):
    """Scale using the full dataset, then rank other songs in the same cluster."""
    values = songs[features].apply(pd.to_numeric, errors='coerce')
    values = values.fillna(values.median()).fillna(0)
    scaled = StandardScaler().fit_transform(values)
    position = songs.index.get_loc(selected_index)
    if not isinstance(position, (int, np.integer)):
        raise ValueError('Song index must be unique.')
    candidates = np.flatnonzero(songs['cluster'].to_numpy() == songs.iloc[position]['cluster'])
    candidates = candidates[candidates != position]
    distances = np.linalg.norm(scaled[candidates] - scaled[position], axis=1)
    order = np.argsort(distances, kind='stable')[:limit]
    result = songs.iloc[candidates[order]].copy()
    result['audio_distance'] = distances[order]
    return result
