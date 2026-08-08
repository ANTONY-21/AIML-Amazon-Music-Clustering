"""
Amazon Music Clustering - Capstone Project
==========================================
Goal: Group similar songs into clusters using ONLY their audio characteristics
      (no genre labels), so streaming platforms can use the clusters for
      playlist generation, recommendation, and market segmentation.

Pipeline (each step is a function so the code is modular and reusable):
    1. load_data            -> read CSV
    2. explore_data         -> shape / dtypes / nulls / duplicates / describe
    3. clean_data           -> drop ID/text columns, drop nulls/duplicates
    4. visualize_features   -> distribution + correlation heatmap (EDA plots)
    5. scale_features       -> StandardScaler (clustering is distance-based)
    6. find_optimal_k       -> Elbow method + Silhouette score
    7. run_kmeans           -> final KMeans with chosen k on the FULL dataset
    8. run_dbscan           -> density-based clustering (finds noise/outliers)
    9. run_hierarchical     -> agglomerative clustering (+ dendrogram on sample)
   10. evaluate_clustering  -> Silhouette, Davies-Bouldin, Calinski-Harabasz
   11. reduce_pca           -> PCA(2) for 2D visualization
   12. profile_clusters     -> mean feature value per cluster -> human label
   13. export_results       -> save labeled CSV + cluster profile CSV

The final KMeans model is fit on ALL songs so every row in the exported CSV
gets a cluster label. DBSCAN and Hierarchical are O(n^2)-ish, so they are
compared on random subsamples (documented in the report).

Run from terminal:
    pip install -r requirements.txt
    python clustering.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Resolve paths relative to this file so the script works from any launch dir.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "single_genre_artists.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Audio features used for clustering. These describe HOW a song sounds
# (rhythm, mood, instrumentation, energy) and have nothing to do with the
# artist or title - perfect for unsupervised grouping.
AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]

# KMeans scales to the full ~96k rows easily, so the final model uses ALL
# songs. DBSCAN / Hierarchical need pairwise-distance style work, so they are
# benchmarked on a random subsample of this size instead.
COMPARE_SAMPLE = 20000
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Read the CSV into a DataFrame."""
    print(f"[1] Loading {path} ...")
    df = pd.read_csv(path)
    print(f"    shape = {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 2. Explore
# ---------------------------------------------------------------------------
def explore_data(df: pd.DataFrame) -> None:
    """Print the answers to the EDA-guide questions."""
    print("\n[2] Exploratory Data Analysis")
    print("    columns :", list(df.columns))
    print("    dtypes  :")
    print(df.dtypes.to_string())
    print(f"    missing : {df.isna().sum().sum()} total nulls")
    print(f"    duplicates: {df.duplicated().sum()}")
    print("    describe (audio features):")
    print(df[AUDIO_FEATURES].describe().round(3).to_string())


# ---------------------------------------------------------------------------
# 3. Clean
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unused columns, missing values, duplicates."""
    print("\n[3] Cleaning data ...")
    # Keep a copy of the reference columns so we can attach them back later.
    keep_for_reference = [c for c in
                          ["id_songs", "name_song", "name_artists", "genres",
                           "popularity_songs", "release_date"]
                          if c in df.columns]

    df = df.dropna(subset=AUDIO_FEATURES).drop_duplicates(subset=AUDIO_FEATURES)
    df = df.reset_index(drop=True)
    print(f"    after cleaning shape = {df.shape}")
    return df[keep_for_reference + AUDIO_FEATURES]


# ---------------------------------------------------------------------------
# 4. Visualize (EDA plots)
# ---------------------------------------------------------------------------
def visualize_features(df: pd.DataFrame) -> None:
    """Histograms for every audio feature + correlation heatmap."""
    print("\n[4] Saving EDA plots ...")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Distribution plots
    df[AUDIO_FEATURES].hist(figsize=(14, 10), bins=40, color="#3b82f6")
    plt.suptitle("Distribution of Audio Features", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "01_feature_distributions.png"), dpi=120)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[AUDIO_FEATURES].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0)
    plt.title("Correlation between Audio Features")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "02_correlation_heatmap.png"), dpi=120)
    plt.close()
    print(f"    plots saved to {PLOTS_DIR}/")


# ---------------------------------------------------------------------------
# 5. Scale
# ---------------------------------------------------------------------------
def scale_features(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """
    StandardScaler -> mean 0, std 1.

    WHY scaling is mandatory for clustering:
    KMeans, DBSCAN and Hierarchical clustering all use Euclidean distance.
    Without scaling, `tempo` (0..250) and `duration_ms` (~200000) would
    dominate distances and tiny features like `danceability` (0..1) would
    be ignored. Scaling puts every feature on equal footing.
    """
    print("\n[5] Scaling features with StandardScaler ...")
    scaler = StandardScaler()
    X = scaler.fit_transform(df[AUDIO_FEATURES])
    return X, scaler


# ---------------------------------------------------------------------------
# 6. Find optimal k (Elbow + Silhouette)
# ---------------------------------------------------------------------------
def find_optimal_k(X: np.ndarray, k_range=range(2, 11)) -> int:
    """
    Two complementary methods:
      Elbow      : plot inertia (within-cluster sum of squares) vs k.
                   The 'elbow' is where adding more clusters stops helping much.
      Silhouette : measures how well each point fits its own cluster vs the
                   nearest other cluster. Range -1..+1, higher = better.
    """
    print("\n[6] Searching for the best k ...")
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels, sample_size=5000,
                                            random_state=RANDOM_STATE))
        print(f"    k={k}  inertia={km.inertia_:.0f}  silhouette={silhouettes[-1]:.4f}")

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(list(k_range), inertias, "o-", color="#3b82f6")
    ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia (SSE)")
    ax[0].set_title("Elbow Method")
    ax[1].plot(list(k_range), silhouettes, "o-", color="#10b981")
    ax[1].set_xlabel("k"); ax[1].set_ylabel("Silhouette score")
    ax[1].set_title("Silhouette Score")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "03_elbow_silhouette.png"), dpi=120)
    plt.close()

    best_k = list(k_range)[int(np.argmax(silhouettes))]
    print(f"    -> best k by silhouette = {best_k}")
    return best_k


# ---------------------------------------------------------------------------
# 7-9. Clustering algorithms
# ---------------------------------------------------------------------------
def run_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    print(f"\n[7] Running KMeans with k={k} ...")
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    return km.fit_predict(X)


def run_dbscan(X: np.ndarray, eps: float = 1.0, min_samples: int = 10) -> np.ndarray:
    """
    DBSCAN groups points in dense regions and labels sparse points as -1 (noise).
    Good for finding outlier / niche tracks. eps and min_samples need tuning.
    """
    print(f"\n[8] Running DBSCAN (eps={eps}, min_samples={min_samples}) ...")
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    print(f"    clusters found = {n_clusters},  noise points = {n_noise}")
    return labels


def run_hierarchical(X: np.ndarray, k: int, sample: int = 3000):
    """
    Agglomerative clustering merges nearest points step-by-step.
    On large data it is O(n^2) memory, so we run it on a random subsample.
    Returns (labels, Xs) so the caller can evaluate metrics on the same rows.
    """
    print(f"\n[9] Running Hierarchical clustering on {sample} sample rows ...")
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    Xs = X[idx]
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = agg.fit_predict(Xs)

    # Dendrogram (very small sample so it's readable)
    Z = linkage(Xs[:200], method="ward")
    plt.figure(figsize=(12, 5))
    dendrogram(Z, truncate_mode="level", p=5)
    plt.title("Hierarchical Clustering Dendrogram (200-row sample)")
    plt.xlabel("Sample index"); plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "04_dendrogram.png"), dpi=120)
    plt.close()
    return labels, Xs


# ---------------------------------------------------------------------------
# 10. Evaluation metrics
# ---------------------------------------------------------------------------
def evaluate_clustering(X: np.ndarray, labels: np.ndarray, name: str) -> dict:
    """
    Silhouette          (-1..+1)  higher = better
    Davies-Bouldin      ( 0..inf) lower  = better
    Calinski-Harabasz   ( 0..inf) higher = better
    """
    mask = labels != -1                          # ignore DBSCAN noise
    if len(set(labels[mask])) < 2:
        print(f"    {name}: not enough clusters to evaluate.")
        return {}
    sil = silhouette_score(X[mask], labels[mask], sample_size=5000,
                           random_state=RANDOM_STATE)
    db = davies_bouldin_score(X[mask], labels[mask])
    ch = calinski_harabasz_score(X[mask], labels[mask])
    print(f"    {name:<14}  silhouette={sil:.4f}  davies_bouldin={db:.4f}  "
          f"calinski_harabasz={ch:.0f}")
    return {"algorithm": name, "silhouette": sil,
            "davies_bouldin": db, "calinski_harabasz": ch}


# ---------------------------------------------------------------------------
# 11. PCA visualization
# ---------------------------------------------------------------------------
def reduce_pca(X: np.ndarray, labels: np.ndarray, name: str) -> None:
    """PCA to 2 components -> scatter plot coloured by cluster."""
    print(f"\n[11] PCA scatter for {name} ...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X2 = pca.fit_transform(X)
    plt.figure(figsize=(9, 7))
    palette = sns.color_palette("tab10", n_colors=max(len(set(labels)), 3))
    # Smaller markers when plotting the full ~96k songs so the plot stays readable
    size, alpha = (6, 0.35) if len(X2) > 30000 else (12, 0.6)
    sns.scatterplot(x=X2[:, 0], y=X2[:, 1], hue=labels, palette=palette,
                    s=size, alpha=alpha, legend="full")
    plt.title(f"{name} clusters (PCA 2D projection)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"05_pca_{name}.png"), dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# 12. Profile clusters (mean feature value -> human readable label)
# ---------------------------------------------------------------------------
def profile_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    For every cluster compute the mean of every audio feature.
    Then we attach a human-friendly label by looking at which features
    are highest. This is what you SHOW in the live evaluation -
    it answers "what does this cluster mean?".
    """
    print("\n[12] Profiling clusters ...")
    df = df.copy()
    df["cluster"] = labels
    profile = df.groupby("cluster")[AUDIO_FEATURES].mean().round(3)
    profile["n_songs"] = df.groupby("cluster").size()

    def label_row(row):
        if row["danceability"] > 0.65 and row["energy"] > 0.65:
            return "High-energy danceable / Party"
        if row["acousticness"] > 0.6 and row["energy"] < 0.4:
            return "Acoustic / Chill"
        if row["instrumentalness"] > 0.5:
            return "Instrumental / Background"
        if row["valence"] > 0.6 and row["energy"] > 0.5:
            return "Happy / Upbeat"
        if row["energy"] < 0.4 and row["valence"] < 0.4:
            return "Sad / Mellow"
        if row["speechiness"] > 0.3:
            return "Speech / Spoken-word / Hip-hop"
        return "Mixed / Mainstream"

    profile["mood_label"] = profile.apply(label_row, axis=1)
    print(profile.to_string())
    return profile


# ---------------------------------------------------------------------------
# 13. Export
# ---------------------------------------------------------------------------
def export_results(df: pd.DataFrame, labels: np.ndarray,
                   profile: pd.DataFrame, scores: list[dict]) -> None:
    """Save final labeled songs + cluster profile + metric comparison."""
    print("\n[13] Exporting results ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = df.copy()
    df["cluster"] = labels
    df["mood_label"] = df["cluster"].map(profile["mood_label"])

    df.to_csv(os.path.join(OUTPUT_DIR, "songs_with_clusters.csv"), index=False)
    profile.to_csv(os.path.join(OUTPUT_DIR, "cluster_profile.csv"))
    pd.DataFrame(scores).to_csv(os.path.join(OUTPUT_DIR, "model_scores.csv"),
                                index=False)
    print(f"    saved {len(df)} rows -> {OUTPUT_DIR}/songs_with_clusters.csv")
    print(f"    saved cluster profile -> {OUTPUT_DIR}/cluster_profile.csv")
    print(f"    saved model comparison -> {OUTPUT_DIR}/model_scores.csv")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    df = load_data(DATA_PATH)
    explore_data(df)
    df = clean_data(df)

    visualize_features(df)
    X, _ = scale_features(df)

    best_k = find_optimal_k(X)

    # ---- KMeans (final model, fit on ALL songs) ----
    km_labels = run_kmeans(X, best_k)
    scores = []
    scores.append(evaluate_clustering(X, km_labels, "KMeans"))
    reduce_pca(X, km_labels, "KMeans")

    # ---- DBSCAN (compared on a random subsample - O(n^2)-ish on full data) ----
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(COMPARE_SAMPLE, len(X)), replace=False)
    Xs = X[idx]
    db_labels = run_dbscan(Xs)
    scores.append(evaluate_clustering(Xs, db_labels, "DBSCAN"))
    reduce_pca(Xs, db_labels, "DBSCAN")

    # ---- Hierarchical (on subsample) ----
    hc_labels, hc_X = run_hierarchical(X, best_k)
    # hc_labels was fit on hc_X (the same random subsample) - evaluate on those.
    scores.append(evaluate_clustering(hc_X, hc_labels, "Hierarchical"))

    # ---- Profile + export using KMeans (best for interpretation) ----
    profile = profile_clusters(df, km_labels)
    export_results(df, km_labels, profile, scores)

    print("\nDONE - check the 'outputs/' folder.")


if __name__ == "__main__":
    main()
