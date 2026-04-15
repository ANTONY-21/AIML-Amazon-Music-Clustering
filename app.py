"""
Amazon Music Clustering - Streamlit App
=======================================
Interactive dashboard to explore the clusters produced by clustering.py.

Run from terminal:
    streamlit run app.py

The app reads the files written by clustering.py:
    outputs/songs_with_clusters.csv
    outputs/cluster_profile.csv
    outputs/model_scores.csv

Features:
    - Sidebar filter to pick a cluster
    - KPI cards (number of clusters, songs, best silhouette)
    - Cluster profile table
    - Mean-feature bar chart per cluster (radar-style)
    - PCA scatter plot coloured by cluster
    - Top songs in the selected cluster
    - "Find similar songs" - pick a song and see its cluster mates
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
OUTPUT_DIR = "outputs"
SONGS_FILE = os.path.join(OUTPUT_DIR, "songs_with_clusters.csv")
PROFILE_FILE = os.path.join(OUTPUT_DIR, "cluster_profile.csv")
SCORES_FILE = os.path.join(OUTPUT_DIR, "model_scores.csv")

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
]

st.set_page_config(page_title="Amazon Music Clustering",
                   page_icon="🎵", layout="wide")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
@st.cache_data
def load_outputs():
    if not os.path.exists(SONGS_FILE):
        return None, None, None
    songs = pd.read_csv(SONGS_FILE)
    profile = pd.read_csv(PROFILE_FILE)
    scores = pd.read_csv(SCORES_FILE) if os.path.exists(SCORES_FILE) else None
    return songs, profile, scores


songs, profile, scores = load_outputs()

st.title("🎵 Amazon Music Clustering")
st.caption("Unsupervised grouping of songs by their audio characteristics.")

if songs is None:
    st.error(
        "No outputs found. Please run **`python clustering.py`** first to "
        "generate the clustered dataset."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")
cluster_options = sorted(songs["cluster"].unique())
selected_cluster = st.sidebar.selectbox(
    "Select a cluster",
    options=["All"] + [int(c) for c in cluster_options],
)

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total songs", f"{len(songs):,}")
c2.metric("Clusters found", len(cluster_options))
if scores is not None and not scores.empty:
    best = scores.loc[scores["silhouette"].idxmax()]
    c3.metric("Best algorithm", best["algorithm"])
    c4.metric("Silhouette score", f"{best['silhouette']:.3f}")

st.divider()

# ---------------------------------------------------------------------------
# Cluster profile table
# ---------------------------------------------------------------------------
st.subheader("Cluster profile (mean feature value per cluster)")
st.dataframe(profile, use_container_width=True)

# ---------------------------------------------------------------------------
# Mean feature bar chart for selected cluster
# ---------------------------------------------------------------------------
st.subheader("Audio fingerprint")

if selected_cluster == "All":
    mean_df = songs[AUDIO_FEATURES].mean()
    title = "Average across ALL songs"
else:
    sub = songs[songs["cluster"] == selected_cluster]
    mean_df = sub[AUDIO_FEATURES].mean()
    title = f"Cluster {selected_cluster} - {len(sub):,} songs"

fig, ax = plt.subplots(figsize=(10, 4))
# Normalize to 0..1 for visual comparison (loudness/tempo/duration are off-scale)
norm = (mean_df - songs[AUDIO_FEATURES].min()) / \
       (songs[AUDIO_FEATURES].max() - songs[AUDIO_FEATURES].min())
sns.barplot(x=norm.index, y=norm.values, ax=ax, palette="viridis")
ax.set_ylim(0, 1)
ax.set_ylabel("normalised value (0..1)")
ax.set_title(title)
plt.xticks(rotation=30, ha="right")
st.pyplot(fig)

# ---------------------------------------------------------------------------
# PCA scatter
# ---------------------------------------------------------------------------
st.subheader("PCA 2D projection of clusters")

@st.cache_data
def compute_pca(df):
    X = StandardScaler().fit_transform(df[AUDIO_FEATURES])
    pcs = PCA(n_components=2, random_state=42).fit_transform(X)
    return pcs

pcs = compute_pca(songs)
fig, ax = plt.subplots(figsize=(9, 6))
sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=songs["cluster"],
                palette="tab10", s=12, alpha=0.6, ax=ax, legend="full")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.set_title("Each point is a song; colour is its cluster")
st.pyplot(fig)

# ---------------------------------------------------------------------------
# Top songs in selected cluster
# ---------------------------------------------------------------------------
st.subheader("Songs in the selected cluster")
if selected_cluster == "All":
    st.info("Pick a cluster from the sidebar to see its top songs.")
else:
    sub = songs[songs["cluster"] == selected_cluster]
    cols = [c for c in
            ["name_song", "name_artists", "popularity_songs", "mood_label"]
            if c in sub.columns]
    if "popularity_songs" in sub.columns:
        sub = sub.sort_values("popularity_songs", ascending=False)
    st.dataframe(sub[cols].head(50), use_container_width=True)

# ---------------------------------------------------------------------------
# Recommend similar songs (same cluster)
# ---------------------------------------------------------------------------
st.subheader("🎧 Find similar songs")
if "name_song" in songs.columns:
    track = st.selectbox("Pick a song",
                         options=sorted(songs["name_song"].dropna().unique())[:2000])
    if track:
        row = songs[songs["name_song"] == track].iloc[0]
        st.write(f"**{track}** belongs to cluster **{row['cluster']}** "
                 f"({row.get('mood_label', '')})")
        sims = songs[(songs["cluster"] == row["cluster"]) &
                     (songs["name_song"] != track)]
        if "popularity_songs" in sims.columns:
            sims = sims.sort_values("popularity_songs", ascending=False)
        st.dataframe(sims[[c for c in
                           ["name_song", "name_artists", "popularity_songs"]
                           if c in sims.columns]].head(20),
                     use_container_width=True)

st.divider()
st.caption("Built with KMeans / DBSCAN / Hierarchical clustering on audio features.")
