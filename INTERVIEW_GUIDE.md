# 🎤 Live Evaluation Study Guide — Amazon Music Clustering

This is your script for the **30-min live evaluation** (3-min self intro + 12-min project walk + 10-min mock Qs + 5-min feedback). Read it 2-3 times and you'll be able to defend every line of code.

---

## 0. Self introduction (3 mins) — sample template

> "Hi, I'm **[name]**. I'm pursuing the AIML track at Guvi. I'm comfortable with Python, Pandas, scikit-learn, and basic visualisation libraries (matplotlib, seaborn). I've worked on EDA-heavy projects and I'm now exploring unsupervised learning. My capstone is the **Amazon Music Clustering** project, where I group songs into mood-based clusters using only their audio features."

Keep it under 60 seconds, then dive into the project.

---

## 1. Project walkthrough script (12 mins)

Read this in order — it covers every section the **Capstone Explanation Guideline** asks about (domain → problem → cleaning → EDA → feature engineering → models → evaluation → conclusion → business value).

### 1.1 Domain (3 lines)
> "The domain is **music analytics on streaming platforms**. Services like Amazon Music, Spotify and YouTube Music host millions of songs and need to organise them automatically so users can discover music. Each song has measurable audio properties — tempo, energy, danceability — and these properties make grouping possible without any human labels."

### 1.2 Project introduction (2 lines)
> "This project takes ~96,000 songs from Amazon Music and groups them into clusters using **unsupervised machine learning**. The model never sees the genre — it only sees the sound."

### 1.3 Objective (2 lines)
> "Objective: build a clustering pipeline that produces meaningful, interpretable groups of similar songs, and use those groups to enable playlist generation, recommendation, and listener segmentation."

### 1.4 Problem statement (1 line)
> "Manual genre tagging doesn't scale to millions of songs, so we need a model that organises tracks automatically based on how they sound."

### 1.5 Data cleaning & preprocessing
- Loaded the CSV with `pandas.read_csv` → 95,838 rows, 23 columns
- Dropped **identifier / text columns** (`id_songs`, `name_song`, `name_artists`, `genres`, `release_date`) — they don't help distance-based clustering
- Removed **null values** in audio features (very few)
- Removed **duplicate rows** based on the audio features
- Used a **20,000 random sample** for development speed (clustering on 96k is slow; sampling preserves the distribution)

### 1.6 EDA findings (1–2 lines per finding)
- **Distributions** — `acousticness` and `instrumentalness` are heavily skewed toward 0; `loudness` is in dB (negative); `duration_ms` has a long tail
- **Correlation heatmap** — `energy` ↔ `loudness` is strongly positive (~0.75); `energy` ↔ `acousticness` is strongly negative (~–0.7). This is expected: louder songs are perceived as more energetic, acoustic songs are quieter
- **Outliers** — A few songs with extreme `tempo` (>200 BPM) and very long `duration_ms` (>10 min). We keep them because they may form their own niche cluster
- No critical missing values, no class imbalance issue (it's unsupervised)

### 1.7 Feature engineering / selection
- **Feature selection** — kept the 10 numeric audio features that describe sound: `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`
- **No new features engineered** — the audio features are already on a meaningful scale and adding ratios would add noise
- **Scaling** with `StandardScaler` (mean 0, std 1) — **mandatory** because clustering uses Euclidean distance and `tempo` (0–250) would otherwise dominate `danceability` (0–1)

### 1.8 Statistical technique
- **Elbow method** — plotted inertia vs k for k=2..10 and looked for the bend
- **Silhouette score** — computed for each k; this is a more reliable indicator than the elbow
- (Both are in `outputs/plots/03_elbow_silhouette.png`)

### 1.9 Models tried & why
| Model | Why I tried it | Result |
|---|---|---|
| **KMeans** | Fast, simple, works well on globular clusters; standard baseline for unsupervised tasks | Best interpretability — used as final model |
| **DBSCAN** | Density-based; finds arbitrarily-shaped clusters and labels outliers as noise. Good for niche tracks | Found a few clusters + many noise points; harder to use for playlists |
| **Hierarchical (Agglomerative + Ward linkage)** | Doesn't need k upfront; produces a dendrogram for visual inspection | Confirmed the same cluster count as KMeans, but O(n²) memory so I ran it on a subsample |

**Why KMeans is the final model:** highest silhouette, even cluster sizes, most interpretable profiles → best for the business goal (recommendation/playlists).

### 1.10 Evaluation metrics
| Metric | What it measures | Direction |
|---|---|---|
| **Silhouette score** | How tight & well-separated each cluster is. Range −1..+1 | Higher is better |
| **Davies-Bouldin index** | Average similarity between each cluster and its most similar one | Lower is better |
| **Calinski-Harabasz index** | Ratio of between-cluster vs within-cluster variance | Higher is better |
| **Inertia (KMeans only)** | Sum of squared distances of points to their cluster centre | Used for elbow plot |

### 1.11 Final model
> "I picked KMeans because it gave the best silhouette score, balanced cluster sizes, and the most human-readable cluster profiles — essential for the business goal of generating playlists."

### 1.12 Conclusion (cluster profiles)
After taking the **mean of every audio feature per cluster**, I labelled them:
- **Cluster A — Party / High-energy** : high danceability + high energy
- **Cluster B — Chill / Acoustic** : high acousticness + low energy
- **Cluster C — Instrumental / Background** : high instrumentalness
- **Cluster D — Speech / Spoken-word** : high speechiness
- **Cluster E — Mainstream** : moderate values everywhere

(See `outputs/cluster_profile.csv`.)

### 1.13 Business suggestion / solution
- Build **mood-based auto-playlists** straight from the cluster column
- Power a **"songs like this"** recommender by serving cluster mates
- Help **artists** see who their sonic competitors are inside their cluster
- Let the **marketing team** target cluster X listeners with cluster-X promotions

---

## 2. Mock interview Q&A (the 10-mark section)

The evaluator will pick a few of these. Read every answer aloud once.

### Why clustering and not classification?
> "Classification needs labels — we don't have any. The dataset has a `genres` column for some artists but it's unreliable and the goal is to discover groupings without supervision. Clustering is unsupervised, which fits."

### Why did you drop `name_song`, `id_songs`, `name_artists`, `genres`?
> "They are identifiers / categorical text. Clustering uses Euclidean distance on numeric features — strings have no meaningful distance. They are kept only as reference columns to attach back at the end so we can see which song landed in which cluster."

### Why scale the data? What happens if you don't?
> "KMeans uses Euclidean distance. Without scaling, `tempo` ranges 0–250 and `duration_ms` is in hundreds of thousands, while `danceability` is 0–1. The big-range features would dominate the distance and the small ones would be ignored. `StandardScaler` puts them all on mean 0, std 1 so each feature contributes equally."

### Difference between StandardScaler and MinMaxScaler?
> "`StandardScaler` → (x − mean) / std → centred at 0, std 1, sensitive to outliers but great for distance-based ML. `MinMaxScaler` → (x − min) / (max − min) → squeezed into 0..1, preserves shape but is very sensitive to outliers. I picked StandardScaler because it handles the long tails (duration, tempo) better."

### What is the elbow method?
> "I run KMeans for k = 2..10 and plot the inertia (sum of squared distances of each point to its cluster centre). At small k the line drops fast; at some point adding more clusters barely reduces inertia — that's the elbow, and the k at the bend is a good choice."

### What is the silhouette score? Range?
> "It measures how similar a point is to its own cluster compared to the nearest other cluster. For each point: `(b − a) / max(a, b)` where `a` is mean intra-cluster distance and `b` is mean nearest-cluster distance. Range: −1 to +1. Above 0.5 is strong, around 0.25 is weak, below 0 means the point is in the wrong cluster."

### Why KMeans? What are its assumptions?
> "Fast, simple, scales to large data. Assumptions: clusters are roughly spherical, similar size, and you must specify k upfront. It minimises within-cluster variance using Lloyd's algorithm."

### How does KMeans actually work? (algorithm)
> "1) Pick k random initial centroids (`k-means++` does it smartly). 2) Assign every point to the nearest centroid. 3) Move each centroid to the mean of its assigned points. 4) Repeat until centroids stop moving or `max_iter` is reached. The objective is to minimise inertia."

### What is `n_init=10` in KMeans?
> "It runs the whole algorithm 10 times with different random seeds and keeps the run with the lowest inertia. KMeans is sensitive to initialisation, so multiple runs give a stable result."

### How is DBSCAN different from KMeans?
> "DBSCAN is **density-based**. It groups points that have at least `min_samples` neighbours within distance `eps`. Strengths: it doesn't need k, finds arbitrary shapes, labels sparse points as **noise** (label = −1). Weaknesses: very sensitive to `eps`, struggles with varying density."

### Why does DBSCAN give noise points and KMeans doesn't?
> "KMeans must assign every point to one of the k clusters. DBSCAN can refuse — if a point has too few neighbours it's marked −1. That's actually useful: those are outlier or niche tracks."

### What is hierarchical / agglomerative clustering?
> "Bottom-up: start with each point as its own cluster, then repeatedly merge the two closest clusters until you have k. The merging rule is the **linkage** — I used **Ward**, which merges the pair that increases total within-cluster variance the least. The result can be visualised as a dendrogram."

### Why did you only run hierarchical on a subsample?
> "It's O(n²) memory because it builds a full pairwise distance matrix. Running it on 96k rows would need ~70 GB. A 3,000-row subsample preserves structure and runs in seconds."

### What is PCA and why did you use it?
> "**Principal Component Analysis** projects high-dimensional data onto orthogonal axes that capture the maximum variance. I used `n_components=2` *only for visualisation* — clustering itself was done in the full 10-D space. The 2D scatter shows whether the clusters separate visually."

### Did you use PCA before clustering?
> "No. PCA is **only for visualisation** here. Reducing to 2 components before clustering would lose ~50% of the variance and hurt cluster quality. The clustering uses the full 10 scaled features."

### Davies-Bouldin index — what is it?
> "Average similarity between each cluster and its most similar one, where similarity = (intra-cluster scatter sum) / (centroid distance). **Lower is better** — a low value means clusters are tight and far apart."

### Calinski-Harabasz index?
> "Ratio of between-cluster variance to within-cluster variance, scaled by `(n − k)/(k − 1)`. Higher = better-separated clusters."

### How did you label the clusters?
> "I computed the mean of every audio feature per cluster and looked at which features stood out. For example: high `danceability` + high `energy` → 'Party'; high `acousticness` + low `energy` → 'Chill / Acoustic'; high `instrumentalness` → 'Background instrumental'."

### How would you recommend a song to a user with this model?
> "1) Get the user's recently played songs. 2) Look up which cluster each one belongs to. 3) Pick the user's most frequent cluster. 4) Recommend top tracks from that cluster they haven't heard yet. The Streamlit app demos this in the 'Find similar songs' section."

### What if your model performs poorly (low silhouette)?
> "Options: try a different k, try a different algorithm (DBSCAN, Gaussian Mixture), engineer better features, remove redundant features (e.g. drop `loudness` which is highly correlated with `energy`), or normalise differently."

### How would you deploy this in production?
> "Save the fitted `StandardScaler` and `KMeans` model with `joblib`, expose a small Flask/FastAPI endpoint that takes a song's audio features and returns its cluster, and the Streamlit app or front-end queries that endpoint."

### How do you handle new songs that come in tomorrow?
> "Use the **fitted** scaler and KMeans (don't refit). Call `scaler.transform(new_song)` then `kmeans.predict(...)`. Periodically (weekly/monthly) refit on the new combined data so the clusters stay current."

### Why didn't you use a supervised model?
> "Because there's no reliable target column. The `genres` column is sparse, multi-valued, and inconsistent. Clustering lets the data speak for itself."

### What's the difference between supervised and unsupervised learning?
> "**Supervised** = you have labelled X → y pairs and the model learns the mapping (classification, regression). **Unsupervised** = you only have X and the model finds structure (clustering, dimensionality reduction)."

### What is inertia in KMeans?
> "Sum of squared Euclidean distances from each point to its assigned cluster centroid. Lower = tighter clusters. It always decreases as k grows, which is why you can't just minimise it — you need the elbow / silhouette to pick k."

### What is `random_state=42` for?
> "It seeds the random number generator so results are reproducible. Same seed → same centroid initialisation → same final clusters."

### Why did you sample 20,000 rows instead of using all 96,000?
> "Speed during development. KMeans scales linearly but silhouette is O(n²). 20k rows is large enough to be statistically representative and runs in seconds. For the final model you'd run on the full dataset."

### Did you test the model accuracy?
> "Accuracy doesn't apply here — it's unsupervised, there's no ground truth. I evaluated cluster **quality** with silhouette, Davies-Bouldin and Calinski-Harabasz, and **interpretability** by profiling the cluster means."

### What is modular code? Show me where you used it.
> "Each step of the pipeline is its own function: `load_data`, `clean_data`, `scale_features`, `find_optimal_k`, `run_kmeans`, `run_dbscan`, `evaluate_clustering`, `profile_clusters`, `export_results`. Each function does one job, can be tested or reused independently, and `main()` just orchestrates them."

### What are the limitations of your project?
> "1) The 'mood' labels are heuristic, not learned. 2) KMeans assumes spherical clusters — real music taste is fuzzier. 3) I sampled 20k rows — production should use the full dataset. 4) DBSCAN was barely tuned (only one eps value tried). 5) No content-based features like lyrics or artist embeddings."

### How would you improve it?
> "1) Try Gaussian Mixture Models (soft cluster assignment). 2) Use **t-SNE** or **UMAP** instead of PCA for non-linear visualisation. 3) Add lyric embeddings using a sentence transformer. 4) A/B test the recommendations against a baseline."

---

## 3. Last-minute checklist before live eval

- [ ] Run `python clustering.py` once and confirm the `outputs/` folder contains all CSVs and plots
- [ ] Run `streamlit run app.py` and click through every section
- [ ] Have the README, the plot images, and this guide open in tabs
- [ ] Be ready to share your screen and **walk through `clustering.py` top-to-bottom**
- [ ] If asked "what would you do differently?" → mention t-SNE/UMAP and Gaussian Mixture Models
- [ ] Speak slowly, point at the plots, explain WHY before WHAT
