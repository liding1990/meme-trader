# TS2Vec Clustering Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a pipeline that batch-fetches 302 Memecoin DNA fingerprints, extracts main rally segments, trains TS2Vec to encode them into embeddings, and clusters them in 3D space.

**Architecture:** Four sequential scripts — batch_fetch.py, preprocess.py, train_ts2vec.py, cluster.py — each reading the previous step's output. All intermediate data stored under `data/`, models under `models/`, plots under `plots/`.

**Tech Stack:** Python 3.11, PyTorch (MPS), ts2vec, umap-learn, hdbscan, plotly, numpy, pandas

---

### Task 1: Install dependencies and update requirements.txt

**Files:**
- Modify: `requirements.txt`

**Step 1: Install all dependencies**

Run:
```bash
source .venv/bin/activate
pip install torch numpy pandas matplotlib ts2vec umap-learn hdbscan plotly kaleido
```

**Step 2: Update requirements.txt**

```
numpy
pandas
matplotlib
torch
ts2vec
umap-learn
hdbscan
plotly
kaleido
```

**Step 3: Verify imports work**

Run:
```bash
source .venv/bin/activate
python -c "import torch; import ts2vec; import umap; import hdbscan; import plotly; print('All imports OK'); print(f'MPS available: {torch.backends.mps.is_available()}')"
```

Expected: `All imports OK` and `MPS available: True`

**Step 4: Commit**

```bash
git add requirements.txt
git commit -m "Add TS2Vec pipeline dependencies"
```

---

### Task 2: Batch Fetch (`batch_fetch.py`)

**Files:**
- Create: `batch_fetch.py`
- Read: `token_list.csv` (302 rows, format: `address,symbol,name`)
- Read: `gmgn_api.py` (existing `fetch_token_data` function)

**Step 1: Write batch_fetch.py**

```python
"""Batch fetch GMGN data for all tokens in token_list.csv."""

import csv
import os
import sys
import time

from gmgn_api import fetch_token_data

TOKEN_LIST = "token_list.csv"
DATA_DIR = "data"
REQUEST_INTERVAL = 2  # seconds between requests


def load_token_list():
    tokens = []
    with open(TOKEN_LIST, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                tokens.append({"address": row[0], "symbol": row[1], "name": row[2] if len(row) > 2 else row[1]})
    return tokens


def has_cached_data(address):
    """Check if token already has both required data files."""
    token_dir = os.path.join(DATA_DIR, address)
    if not os.path.isdir(token_dir):
        return False
    files = os.listdir(token_dir)
    has_trends = any(f.startswith("token_trends_") for f in files)
    has_candles = any(f.startswith("token_mcap_candles_") for f in files)
    return has_trends and has_candles


def main():
    tokens = load_token_list()
    total = len(tokens)
    print(f"Loaded {total} tokens from {TOKEN_LIST}")

    skipped = 0
    fetched = 0
    failed = 0

    for i, token in enumerate(tokens, 1):
        address = token["address"]
        symbol = token["symbol"]

        if has_cached_data(address):
            skipped += 1
            print(f"[{i}/{total}] {symbol} — cached, skipping")
            continue

        print(f"[{i}/{total}] Fetching {symbol} ({address[:8]}...)...")

        try:
            fetch_token_data("sol", address)
            fetched += 1
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            failed += 1

        if i < total:
            time.sleep(REQUEST_INTERVAL)

    print(f"\nDone: {fetched} fetched, {skipped} cached, {failed} failed")


if __name__ == "__main__":
    main()
```

**Step 2: Test with a dry run (first 3 tokens)**

Run:
```bash
source .venv/bin/activate
python -c "
from batch_fetch import load_token_list, has_cached_data
tokens = load_token_list()
print(f'Loaded {len(tokens)} tokens')
for t in tokens[:3]:
    print(f\"  {t['symbol']}: cached={has_cached_data(t['address'])}\")
"
```

Expected: Shows 302 tokens loaded, cache status for first 3.

**Step 3: Run full batch fetch**

Run:
```bash
source .venv/bin/activate
python batch_fetch.py
```

Expected: ~10 min runtime, 302 tokens processed with progress output. Already-fetched token (FtSRg...) should show as cached.

**Step 4: Commit**

```bash
git add batch_fetch.py
git commit -m "Add batch fetch script for 302 tokens"
```

---

### Task 3: Preprocess (`preprocess.py`)

**Files:**
- Create: `preprocess.py`
- Read: `dna_extractor.py` (existing `extract_candle_series`, `extract_trend_series`, `build_dna_dataframe`)
- Read: `token_list.csv`

**Step 1: Write preprocess.py**

```python
"""Preprocess token DNA data: extract main rally, resample, z-score standardize.

Reads raw JSON from data/{address}/, outputs:
  - data/dataset.npy  — shape (N, 200, 4)
  - data/metadata.csv — per-token info
"""

import csv
import json
import os
import sys
import glob

import numpy as np
import pandas as pd

from dna_extractor import extract_candle_series, extract_trend_series, build_dna_dataframe

TOKEN_LIST = "token_list.csv"
DATA_DIR = "data"
RESAMPLE_POINTS = 200
MCAP_THRESHOLD = 100_000  # $100K minimum mcap
DNA_COLUMNS = ["mcap", "holder_count", "top10_pct", "mcap_per_holder"]


def load_token_list():
    tokens = []
    with open(TOKEN_LIST, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                tokens.append({"address": row[0], "symbol": row[1], "name": row[2] if len(row) > 2 else row[1]})
    return tokens


def find_latest_json(token_dir, prefix):
    """Find the most recent JSON file with given prefix in token_dir."""
    pattern = os.path.join(token_dir, f"{prefix}_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return files[-1]


def load_token_data(address):
    """Load raw JSON data for a token. Returns (candles_data, trends_data) or (None, None)."""
    token_dir = os.path.join(DATA_DIR, address)
    if not os.path.isdir(token_dir):
        return None, None

    candles_file = find_latest_json(token_dir, "token_mcap_candles")
    trends_file = find_latest_json(token_dir, "token_trends")

    if not candles_file or not trends_file:
        return None, None

    with open(candles_file, "r") as f:
        candles_data = json.load(f)
    with open(trends_file, "r") as f:
        trends_data = json.load(f)

    return candles_data, trends_data


def extract_main_rally(dna_df):
    """Extract the main rally segment: first mcap >= 100K to ATH.

    Returns the sliced DataFrame, or None if criteria not met.
    """
    mcap = dna_df["mcap"].values

    # Find start: first hour where mcap >= 100K
    above_threshold = np.where(mcap >= MCAP_THRESHOLD)[0]
    if len(above_threshold) == 0:
        return None
    start_idx = above_threshold[0]

    # Find end: ATH (global max) after start
    ath_idx = start_idx + np.argmax(mcap[start_idx:])

    # Discard if start == end (instant ATH, no rally)
    if ath_idx <= start_idx:
        return None

    return dna_df.iloc[start_idx:ath_idx + 1].reset_index(drop=True)


def resample_series(df, n_points):
    """Resample a DataFrame to exactly n_points using linear interpolation."""
    n_orig = len(df)
    if n_orig == n_points:
        return df[DNA_COLUMNS].values

    orig_indices = np.linspace(0, 1, n_orig)
    new_indices = np.linspace(0, 1, n_points)

    result = np.zeros((n_points, len(DNA_COLUMNS)))
    for i, col in enumerate(DNA_COLUMNS):
        result[:, i] = np.interp(new_indices, orig_indices, df[col].values)

    return result


def main():
    tokens = load_token_list()
    print(f"Loaded {len(tokens)} tokens")

    samples = []    # list of (200, 4) arrays
    metadata = []   # list of metadata dicts

    for i, token in enumerate(tokens):
        address = token["address"]
        symbol = token["symbol"]

        candles_data, trends_data = load_token_data(address)
        if candles_data is None:
            continue

        candle_df = extract_candle_series(candles_data)
        trend_df = extract_trend_series(trends_data)
        if candle_df is None or trend_df is None:
            continue

        dna_df = build_dna_dataframe(candle_df, trend_df)
        if dna_df is None or len(dna_df) < 3:
            continue

        rally_df = extract_main_rally(dna_df)
        if rally_df is None:
            continue

        # Need at least 2 points to resample
        if len(rally_df) < 2:
            continue

        resampled = resample_series(rally_df, RESAMPLE_POINTS)
        samples.append(resampled)

        ath_row = rally_df.iloc[-1]
        start_row = rally_df.iloc[0]
        metadata.append({
            "address": address,
            "symbol": symbol,
            "name": token["name"],
            "ath_mcap": ath_row["mcap"],
            "start_mcap": start_row["mcap"],
            "rally_duration_hours": len(rally_df),
            "holder_count_at_ath": ath_row["holder_count"],
            "top10_pct_at_ath": ath_row["top10_pct"],
        })

    if not samples:
        print("ERROR: No valid samples after preprocessing", file=sys.stderr)
        sys.exit(1)

    # Stack into (N, 200, 4)
    dataset = np.stack(samples, axis=0)
    print(f"Dataset shape before z-score: {dataset.shape}")

    # Z-score standardization (global, per dimension)
    mean = dataset.mean(axis=(0, 1))  # shape (4,)
    std = dataset.std(axis=(0, 1))    # shape (4,)
    std[std == 0] = 1  # avoid division by zero
    dataset = (dataset - mean) / std

    # Save normalization params for later use
    np.save(os.path.join(DATA_DIR, "zscore_params.npy"), np.stack([mean, std]))

    # Save dataset
    np.save(os.path.join(DATA_DIR, "dataset.npy"), dataset)
    print(f"Saved dataset: {dataset.shape} → data/dataset.npy")

    # Save metadata
    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(os.path.join(DATA_DIR, "metadata.csv"), index=False)
    print(f"Saved metadata: {len(metadata)} tokens → data/metadata.csv")

    # Summary
    print(f"\n{'='*50}")
    print(f"  Preprocessing Summary")
    print(f"{'='*50}")
    print(f"  Total tokens:     {len(tokens)}")
    print(f"  Valid samples:    {len(samples)}")
    print(f"  Discarded:        {len(tokens) - len(samples)}")
    print(f"  Dataset shape:    {dataset.shape}")
    print(f"  Z-score mean:     {mean}")
    print(f"  Z-score std:      {std}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
```

**Step 2: Test with existing data**

Run:
```bash
source .venv/bin/activate
python preprocess.py
```

Expected: Should process at least the 1 token we already fetched (FtSRg...), print dataset shape `(1, 200, 4)` if only 1 token has data. After batch_fetch completes, rerun for full dataset.

**Step 3: Commit**

```bash
git add preprocess.py
git commit -m "Add preprocessing: main rally extraction, resample, z-score"
```

---

### Task 4: TS2Vec Training & Encoding (`train_ts2vec.py`)

**Files:**
- Create: `train_ts2vec.py`

**Step 1: Write train_ts2vec.py**

Note: TS2Vec does NOT support MPS device. Use CPU (fast enough for ~200 samples of length 200).

```python
"""Train TS2Vec encoder on preprocessed DNA dataset and output embeddings.

Reads:  data/dataset.npy — shape (N, 200, 4)
Writes: models/ts2vec.pkl — trained model checkpoint
        data/embeddings.npy — shape (N, 320)
"""

import os
import sys
import time

import numpy as np

DATA_DIR = "data"
MODEL_DIR = "models"


def main():
    from ts2vec import TS2Vec

    # Load dataset
    dataset_path = os.path.join(DATA_DIR, "dataset.npy")
    if not os.path.isfile(dataset_path):
        print("ERROR: data/dataset.npy not found. Run preprocess.py first.", file=sys.stderr)
        sys.exit(1)

    data = np.load(dataset_path)
    n_samples, seq_len, n_features = data.shape
    print(f"Loaded dataset: {data.shape} ({n_samples} tokens, {seq_len} steps, {n_features} dims)")

    # Initialize model
    model = TS2Vec(
        input_dims=n_features,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device="cpu",
        lr=0.001,
        batch_size=min(16, n_samples),
    )

    # Train
    print("Training TS2Vec...")
    t0 = time.time()
    loss_log = model.fit(data, n_epochs=200, verbose=True)
    elapsed = time.time() - t0
    print(f"Training complete in {elapsed:.1f}s, final loss: {loss_log[-1]:.6f}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "ts2vec.pkl")
    model.save(model_path)
    print(f"Model saved: {model_path}")

    # Encode
    print("Encoding embeddings...")
    embeddings = model.encode(data, encoding_window="full_series")
    print(f"Embeddings shape: {embeddings.shape}")

    # Save embeddings
    emb_path = os.path.join(DATA_DIR, "embeddings.npy")
    np.save(emb_path, embeddings)
    print(f"Embeddings saved: {emb_path}")


if __name__ == "__main__":
    main()
```

**Step 2: Test with available data**

Run:
```bash
source .venv/bin/activate
python train_ts2vec.py
```

Expected: Trains for 200 epochs, prints loss per epoch, saves model and embeddings. On M4 CPU with ~200 samples, should take ~1-2 minutes.

**Step 3: Commit**

```bash
git add train_ts2vec.py
git commit -m "Add TS2Vec training and embedding generation"
```

---

### Task 5: Clustering & Visualization (`cluster.py`)

**Files:**
- Create: `cluster.py`

**Step 1: Write cluster.py**

```python
"""UMAP 3D projection + HDBSCAN clustering + interactive plotly visualization.

Reads:  data/embeddings.npy — shape (N, 320)
        data/metadata.csv — token metadata
Writes: data/clusters.csv — cluster assignments with UMAP coords
        plots/cluster_3d.html — interactive 3D scatter
        plots/cluster_3d.png — static snapshot
"""

import os
import sys

import numpy as np
import pandas as pd
import umap
import hdbscan
import plotly.express as px

DATA_DIR = "data"
PLOTS_DIR = "plots"


def main():
    # Load data
    emb_path = os.path.join(DATA_DIR, "embeddings.npy")
    meta_path = os.path.join(DATA_DIR, "metadata.csv")

    if not os.path.isfile(emb_path):
        print("ERROR: data/embeddings.npy not found. Run train_ts2vec.py first.", file=sys.stderr)
        sys.exit(1)

    embeddings = np.load(emb_path)
    metadata = pd.read_csv(meta_path)
    print(f"Loaded {len(embeddings)} embeddings, {len(metadata)} metadata rows")

    # UMAP 3D projection
    print("Running UMAP 3D projection...")
    reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
    coords_3d = reducer.fit_transform(embeddings)
    print(f"UMAP output shape: {coords_3d.shape}")

    # HDBSCAN clustering
    print("Running HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    labels = clusterer.fit_predict(coords_3d)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"Found {n_clusters} clusters, {n_noise} noise points")

    # Build results DataFrame
    results = metadata.copy()
    results["cluster"] = labels
    results["umap_x"] = coords_3d[:, 0]
    results["umap_y"] = coords_3d[:, 1]
    results["umap_z"] = coords_3d[:, 2]

    # Save cluster assignments
    results.to_csv(os.path.join(DATA_DIR, "clusters.csv"), index=False)
    print(f"Saved cluster assignments: data/clusters.csv")

    # Cluster summary
    print(f"\n{'='*60}")
    print(f"  Cluster Analysis")
    print(f"{'='*60}")
    for label in sorted(set(labels)):
        cluster_data = results[results["cluster"] == label]
        name = f"Cluster {label}" if label >= 0 else "Noise"
        print(f"\n  {name} ({len(cluster_data)} tokens):")
        print(f"    ATH mcap:      median ${cluster_data['ath_mcap'].median():,.0f}")
        print(f"    Rally duration: median {cluster_data['rally_duration_hours'].median():.0f}h")
        print(f"    Holders at ATH: median {cluster_data['holder_count_at_ath'].median():,.0f}")
        print(f"    Top10% at ATH:  median {cluster_data['top10_pct_at_ath'].median():.1f}%")
        print(f"    Tokens: {', '.join(cluster_data['symbol'].head(5).tolist())}")
    print(f"{'='*60}")

    # 3D interactive plot
    os.makedirs(PLOTS_DIR, exist_ok=True)

    results["cluster_label"] = results["cluster"].apply(lambda x: f"Cluster {x}" if x >= 0 else "Noise")
    results["hover"] = results.apply(
        lambda r: f"{r['symbol']}<br>ATH: ${r['ath_mcap']:,.0f}<br>Rally: {r['rally_duration_hours']}h<br>Holders: {r['holder_count_at_ath']:,.0f}",
        axis=1,
    )

    fig = px.scatter_3d(
        results,
        x="umap_x", y="umap_y", z="umap_z",
        color="cluster_label",
        hover_name="symbol",
        hover_data={"umap_x": False, "umap_y": False, "umap_z": False,
                     "ath_mcap": ":,.0f", "rally_duration_hours": True,
                     "holder_count_at_ath": ":,.0f", "cluster_label": False},
        title="Memecoin DNA Clusters (TS2Vec + UMAP + HDBSCAN)",
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(scene=dict(
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        zaxis_title="UMAP-3",
    ))

    html_path = os.path.join(PLOTS_DIR, "cluster_3d.html")
    fig.write_html(html_path)
    print(f"\nInteractive plot saved: {html_path}")

    png_path = os.path.join(PLOTS_DIR, "cluster_3d.png")
    fig.write_image(png_path, width=1200, height=800, scale=2)
    print(f"Static plot saved: {png_path}")


if __name__ == "__main__":
    main()
```

**Step 2: Test with available data**

Run:
```bash
source .venv/bin/activate
python cluster.py
```

Expected: Prints cluster summary, saves `data/clusters.csv`, `plots/cluster_3d.html`, `plots/cluster_3d.png`.

**Step 3: Commit**

```bash
git add cluster.py
git commit -m "Add UMAP + HDBSCAN clustering with 3D plotly visualization"
```

---

### Task 6: Update .gitignore and final commit

**Files:**
- Modify: `.gitignore`

**Step 1: Update .gitignore**

Add `models/` and `plots/` to gitignore (generated artifacts):

```
.venv/
__pycache__/
*.pyc
.DS_Store
data/
.claude/
models/
plots/
```

**Step 2: Commit everything**

```bash
git add .gitignore
git commit -m "Update gitignore for models and plots dirs"
```

---

### Task 7: End-to-end run

**Step 1: Run the full pipeline**

```bash
source .venv/bin/activate
python batch_fetch.py       # ~10 min
python preprocess.py        # seconds
python train_ts2vec.py      # ~1-2 min
python cluster.py           # seconds
```

**Step 2: Open the interactive 3D plot**

```bash
open plots/cluster_3d.html
```

**Step 3: Review cluster results**

```bash
cat data/clusters.csv | head -20
```

Verify clusters make sense by checking if tokens within the same cluster share similar ATH ranges, rally durations, or holder patterns.
