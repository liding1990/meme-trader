# TS2Vec Clustering Pipeline Design

## Goal

Use TS2Vec contrastive learning to encode Memecoin 4D DNA fingerprints into fixed-length embeddings, then project into 3D via UMAP and discover behavioral clusters via HDBSCAN.

## Data Source

302 Solana Memecoin tokens from the `radar_token` table on `intel.threekeys.fund`. Token addresses exported to `token_list.csv`.

## Pipeline Overview

```
batch_fetch.py → preprocess.py → train_ts2vec.py → cluster.py
```

---

## Step 1: Batch Fetch (`batch_fetch.py`)

Iterate through `token_list.csv`, call `gmgn_api.fetch_token_data()` for each token.

- **Rate limit**: 2 second interval between requests
- **Caching**: Skip tokens that already have data in `data/{address}/`
- **Fault tolerance**: Log failures and continue; print progress (e.g. `[42/302] Fetching ZORA...`)
- **Output**: Raw JSON files in `data/{address}/` for each token

Estimated time: ~10 minutes for 302 tokens.

---

## Step 2: Preprocess (`preprocess.py`)

### 2a. Extract DNA for each token

For each token with cached data, run DNA extraction (mcap, holder_count, top10_pct, mcap_per_holder) at 1-hour granularity.

### 2b. Main Rally Extraction

- **Start**: First hour where mcap >= 100,000
- **End**: Hour where mcap reaches its all-time high (ATH)
- **Discard**: Tokens that never reach 100K mcap; tokens where start == end (instant ATH)

### 2c. Resample to Fixed Length

Linearly interpolate the variable-length main rally segment to exactly **200 time steps** across all 4 dimensions. Output shape per token: `(200, 4)`.

### 2d. Z-Score Standardization

Compute global mean and std across ALL tokens for each of the 4 dimensions. Apply z-score: `(x - mean) / std`. This preserves absolute magnitude differences — a token that reaches 50M mcap will have different values than one reaching 500K.

### 2e. Output

- `data/dataset.npy` — shape `(N, 200, 4)`, the training matrix
- `data/metadata.csv` — per-token info: address, symbol, name, ath_mcap, rally_duration_hours, start_mcap, holder_count_at_ath, top10_pct_at_ath

---

## Step 3: TS2Vec Training & Encoding (`train_ts2vec.py`)

### Model

Use the official TS2Vec implementation (`yuezhihan/ts2vec`), PyTorch-based.

### Training Config

- **Input**: `(N, 200, 4)` multivariate time series
- **Epochs**: 200 (TS2Vec default)
- **Batch size**: 16
- **Hidden dims**: 64
- **Output dims**: 320
- **Depth**: 10 (dilated conv layers)
- **Device**: MPS (Apple M4) with CPU fallback

### Encoding

After training, encode all samples to get instance-level embeddings via `model.encode(data)` with `encoding_window='full_series'`.

### Output

- `models/ts2vec.pkl` — trained encoder checkpoint
- `data/embeddings.npy` — shape `(N, 320)`, one embedding per token

---

## Step 4: Clustering & Visualization (`cluster.py`)

### UMAP 3D Projection

- **Input**: `(N, 320)` embedding matrix
- **Output dims**: 3
- **n_neighbors**: 15
- **min_dist**: 0.1
- **metric**: euclidean

### HDBSCAN Clustering

- **Input**: `(N, 3)` UMAP coordinates
- **min_cluster_size**: 5
- **Noise**: Points with label=-1 are outliers

### Visualization

**Interactive 3D scatter** (plotly):
- Each point = one token
- Color = cluster label
- Hover = symbol, name, ATH, rally duration
- Save as `plots/cluster_3d.html`

**Static PNG**: Save a default-angle snapshot to `plots/cluster_3d.png`

### Cluster Analysis Report

For each cluster, compute and print:
- Number of tokens
- Mean/median ATH mcap
- Mean rally duration (hours)
- Mean holder growth rate
- Mean top10 concentration change
- Representative tokens (closest to cluster centroid)

Output to `data/clusters.csv` (address, symbol, cluster_label, umap_x, umap_y, umap_z).

---

## File Structure

```
meme-trader/
├── CLAUDE.md
├── requirements.txt
├── token_list.csv
├── .gitignore
├── gmgn_api.py                 # (existing)
├── dna_extractor.py            # (existing)
├── batch_fetch.py              # Step 1
├── preprocess.py               # Step 2
├── train_ts2vec.py             # Step 3
├── cluster.py                  # Step 4
├── models/
├── plots/
├── docs/plans/
└── data/
    ├── {address}/
    ├── dataset.npy
    ├── metadata.csv
    └── embeddings.npy
```

## Dependencies

```
numpy
pandas
matplotlib
torch
umap-learn
hdbscan
plotly
kaleido
ts2vec (from github)
```
