# Path Signature Trajectory Matching — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Given a new Memecoin token with K hours of data, find the historical tokens whose early trajectory is most similar, then show their continuations to predict the new token's future.

**Architecture:** Compute path signatures (depth 3) on log-transformed, robust-normalized 4D DNA + time augmentation. Build a signature index from ~200 historical tokens' full trajectories. At query time, compute signature of new token's prefix, find nearest neighbors, display their post-K continuations as prediction reference.

**Tech Stack:** Python 3.11, iisignature, scipy, numpy, pandas, plotly, dtaidistance (optional DTW refinement)

---

### Task 1: Install new dependencies, update requirements.txt

**Files:**
- Modify: `requirements.txt`

**Step 1: Install**

```bash
source .venv/bin/activate
pip install iisignature scipy dtaidistance
```

Note: iisignature requires `--no-build-isolation` if numpy isn't in build env.

**Step 2: Update requirements.txt**

```
numpy<2
pandas
matplotlib
torch
ts2vec
umap-learn
hdbscan
plotly
kaleido
scikit-learn
iisignature
scipy
dtaidistance
```

**Step 3: Verify**

```bash
python -c "import iisignature, scipy, dtaidistance; print('OK')"
```

**Step 4: Commit**

```bash
git add requirements.txt
git commit -m "Add path signature dependencies: iisignature, scipy, dtaidistance"
```

---

### Task 2: Preprocess v2 — full trajectories from 100K onward (`preprocess.py`)

**Files:**
- Modify: `preprocess.py`

The current preprocess.py extracts 100K→ATH and resamples to 200 points. We need to change it:
- Keep everything from first 100K onward (including post-ATH decline)
- Do NOT resample to fixed length — keep hourly granularity, variable length
- Apply log transform + robust normalization
- Save each token's full trajectory separately (not stacked into one array)

**Step 1: Rewrite preprocess.py**

```python
"""Preprocess token DNA data for path signature matching.

For each token:
  1. Extract hourly DNA (mcap, holder_count, top10_pct, mcap_per_holder)
  2. Trim to 100K mcap onward (full trajectory, NOT just to ATH)
  3. Log transform + robust z-normalize per token
  4. Save individual trajectories as .npy files

Outputs:
  - data/trajectories/{address}.npy — per-token normalized trajectory
  - data/metadata.csv — token metadata (address, symbol, ath_mcap, total_hours, etc.)
"""

import csv
import json
import os
import sys
import glob

import numpy as np
import pandas as pd
from scipy.stats import iqr

from dna_extractor import extract_candle_series, extract_trend_series, build_dna_dataframe

TOKEN_LIST = "token_list.csv"
DATA_DIR = "data"
TRAJ_DIR = os.path.join(DATA_DIR, "trajectories")
MCAP_THRESHOLD = 100_000
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
    pattern = os.path.join(token_dir, f"{prefix}_*.json")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def load_token_data(address):
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


def normalize_trajectory(values):
    """Log transform + robust z-normalize (median/IQR) per dimension."""
    # Log transform to compress fat tails (handle zeros)
    log_vals = np.log1p(np.abs(values)) * np.sign(values)

    # Robust z-normalize per dimension using median and IQR
    for d in range(log_vals.shape[1]):
        col = log_vals[:, d]
        med = np.median(col)
        q_iqr = iqr(col)
        if q_iqr > 0:
            log_vals[:, d] = (col - med) / q_iqr
        else:
            log_vals[:, d] = col - med

    return log_vals


def main():
    tokens = load_token_list()
    print(f"Loaded {len(tokens)} tokens")

    os.makedirs(TRAJ_DIR, exist_ok=True)
    metadata = []

    for token in tokens:
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

        # Trim: start from first hour where mcap >= 100K
        mcap = dna_df["mcap"].values
        above = np.where(mcap >= MCAP_THRESHOLD)[0]
        if len(above) == 0:
            continue
        start_idx = above[0]
        trimmed = dna_df.iloc[start_idx:].reset_index(drop=True)

        if len(trimmed) < 3:
            continue

        # Extract raw values
        raw = trimmed[DNA_COLUMNS].values  # (T, 4)

        # Normalize
        normalized = normalize_trajectory(raw)

        # Save trajectory
        traj_path = os.path.join(TRAJ_DIR, f"{address}.npy")
        np.save(traj_path, normalized)

        # Metadata
        ath_idx = np.argmax(raw[:, 0])  # mcap column
        metadata.append({
            "address": address,
            "symbol": symbol,
            "name": token["name"],
            "total_hours": len(trimmed),
            "ath_mcap": raw[ath_idx, 0],
            "ath_hour": int(ath_idx),
            "start_mcap": raw[0, 0],
            "holders_at_ath": raw[ath_idx, 1],
            "top10_pct_at_ath": raw[ath_idx, 2],
        })

    if not metadata:
        print("ERROR: No valid tokens", file=sys.stderr)
        sys.exit(1)

    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(os.path.join(DATA_DIR, "metadata.csv"), index=False)

    print(f"\n{'='*50}")
    print(f"  Preprocessing Summary")
    print(f"{'='*50}")
    print(f"  Total tokens:    {len(tokens)}")
    print(f"  Valid tokens:    {len(metadata)}")
    print(f"  Trajectories:    {TRAJ_DIR}/")
    print(f"  Hours range:     {meta_df['total_hours'].min()} — {meta_df['total_hours'].max()}")
    print(f"  Median hours:    {meta_df['total_hours'].median():.0f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
```

**Step 2: Run**

```bash
source .venv/bin/activate
python preprocess.py
```

Expected: Creates `data/trajectories/{address}.npy` for each valid token, prints summary.

**Step 3: Commit**

```bash
git add preprocess.py
git commit -m "Rewrite preprocess: full trajectories from 100K, log+robust normalization"
```

---

### Task 3: Signature index builder (`signature_index.py`)

**Files:**
- Create: `signature_index.py`

Builds a signature index from all historical trajectories. For each token, computes path signatures at multiple prefix lengths (every hour) so we can match against any K.

**Step 1: Write signature_index.py**

```python
"""Build path signature index from historical token trajectories.

For each historical token, compute the cumulative path signature at every
hour — so for a token with T hours of data, we store T signatures
(prefix of 3 hours, 4 hours, ..., T hours).

This allows matching a new token at ANY point in its lifecycle.

Reads:  data/trajectories/{address}.npy — per-token normalized trajectory
        data/metadata.csv
Writes: data/signature_index.pkl — {address: {k: signature_vector}} for all tokens and prefix lengths
"""

import os
import sys
import pickle

import numpy as np
import iisignature
import pandas as pd

DATA_DIR = "data"
TRAJ_DIR = os.path.join(DATA_DIR, "trajectories")
SIG_DEPTH = 3
MIN_PREFIX_LEN = 3  # minimum 3 hours to compute meaningful signature


def augment_path(trajectory):
    """Add normalized time as 5th dimension.

    Input:  (T, 4) — normalized DNA
    Output: (T, 5) — with time column prepended
    """
    T = len(trajectory)
    time_col = np.linspace(0, 1, T).reshape(-1, 1)
    return np.hstack([time_col, trajectory])


def compute_prefix_signatures(trajectory, depth=SIG_DEPTH, min_len=MIN_PREFIX_LEN):
    """Compute path signature for every prefix length >= min_len.

    Returns dict: {prefix_length: signature_vector}
    """
    augmented = augment_path(trajectory)
    T = len(augmented)
    signatures = {}

    for k in range(min_len, T + 1):
        prefix = augmented[:k]
        sig = iisignature.sig(prefix, depth)
        signatures[k] = sig

    return signatures


def build_index():
    """Build signature index for all historical tokens."""
    meta_path = os.path.join(DATA_DIR, "metadata.csv")
    if not os.path.isfile(meta_path):
        print("ERROR: data/metadata.csv not found. Run preprocess.py first.", file=sys.stderr)
        sys.exit(1)

    metadata = pd.read_csv(meta_path)
    print(f"Building signature index for {len(metadata)} tokens (depth={SIG_DEPTH})...")

    index = {}
    total_sigs = 0

    for i, row in metadata.iterrows():
        address = row["address"]
        symbol = row["symbol"]
        traj_path = os.path.join(TRAJ_DIR, f"{address}.npy")

        if not os.path.isfile(traj_path):
            continue

        trajectory = np.load(traj_path)
        sigs = compute_prefix_signatures(trajectory)
        index[address] = sigs
        total_sigs += len(sigs)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(metadata)}] processed...")

    # Save index
    index_path = os.path.join(DATA_DIR, "signature_index.pkl")
    with open(index_path, "wb") as f:
        pickle.dump(index, f)

    print(f"\nSignature index built:")
    print(f"  Tokens: {len(index)}")
    print(f"  Total signatures: {total_sigs}")
    print(f"  Saved: {index_path}")

    return index


if __name__ == "__main__":
    build_index()
```

**Step 2: Run**

```bash
source .venv/bin/activate
python signature_index.py
```

Expected: Builds index, prints token count and total signatures.

**Step 3: Commit**

```bash
git add signature_index.py
git commit -m "Add path signature index builder (depth-3, cumulative prefixes)"
```

---

### Task 4: Query engine — find similar tokens (`query.py`)

**Files:**
- Create: `query.py`

The core matching engine. Given a new token's address, fetches its current data, computes its signature, finds the top-K most similar historical prefixes, and displays their continuations.

**Step 1: Write query.py**

```python
"""Query engine: find historical tokens most similar to a new token's trajectory.

Given a token address:
1. Fetch its current DNA data from GMGN API
2. Compute path signature of its trajectory from 100K onward
3. Find top-N historical tokens with most similar prefix signatures
4. Display their continuations (what happened after hour K)

Usage:
    python query.py <contract_address> [--top 5] [--chain sol]
"""

import argparse
import os
import sys
import pickle

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import iisignature
from gmgn_api import fetch_token_data
from dna_extractor import extract_candle_series, extract_trend_series, build_dna_dataframe
from preprocess import normalize_trajectory, MCAP_THRESHOLD, DNA_COLUMNS
from signature_index import augment_path, SIG_DEPTH, MIN_PREFIX_LEN, TRAJ_DIR

DATA_DIR = "data"


def load_index():
    index_path = os.path.join(DATA_DIR, "signature_index.pkl")
    if not os.path.isfile(index_path):
        print("ERROR: signature index not found. Run signature_index.py first.", file=sys.stderr)
        sys.exit(1)
    with open(index_path, "rb") as f:
        return pickle.load(f)


def fetch_new_token_trajectory(chain, address):
    """Fetch and preprocess a new token's trajectory."""
    _, loaded_data = fetch_token_data(chain, address)

    candle_df = extract_candle_series(loaded_data.get("token_mcap_candles", {}))
    trend_df = extract_trend_series(loaded_data.get("token_trends", {}))
    if candle_df is None or trend_df is None:
        return None, None

    dna_df = build_dna_dataframe(candle_df, trend_df)
    if dna_df is None or len(dna_df) < 3:
        return None, None

    # Trim from 100K
    mcap = dna_df["mcap"].values
    above = np.where(mcap >= MCAP_THRESHOLD)[0]
    if len(above) == 0:
        return None, None

    start_idx = above[0]
    trimmed = dna_df.iloc[start_idx:].reset_index(drop=True)

    raw = trimmed[DNA_COLUMNS].values
    normalized = normalize_trajectory(raw)

    return normalized, raw


def find_similar(new_trajectory, sig_index, metadata, top_n=5):
    """Find top-N historical tokens with most similar prefix signature."""
    K = len(new_trajectory)
    if K < MIN_PREFIX_LEN:
        print(f"ERROR: Token only has {K} hours of data (need >= {MIN_PREFIX_LEN})", file=sys.stderr)
        return []

    # Compute new token's signature
    augmented = augment_path(new_trajectory)
    new_sig = iisignature.sig(augmented, SIG_DEPTH)

    # Compare against all historical tokens at prefix length K
    candidates = []
    for address, sigs in sig_index.items():
        if K in sigs:
            hist_sig = sigs[K]
            dist = np.linalg.norm(new_sig - hist_sig)
            candidates.append((address, dist))

    # Sort by distance
    candidates.sort(key=lambda x: x[1])

    # Enrich with metadata
    meta_dict = {row["address"]: row for _, row in metadata.iterrows()}
    results = []
    for address, dist in candidates[:top_n]:
        meta = meta_dict.get(address, {})
        results.append({
            "address": address,
            "symbol": meta.get("symbol", "?"),
            "name": meta.get("name", "?"),
            "distance": dist,
            "ath_mcap": meta.get("ath_mcap", 0),
            "ath_hour": meta.get("ath_hour", 0),
            "total_hours": meta.get("total_hours", 0),
        })

    return results


def print_results(results, K):
    """Print match results."""
    print(f"\n{'='*70}")
    print(f"  Top {len(results)} Similar Tokens (matched at hour {K})")
    print(f"{'='*70}")
    for i, r in enumerate(results, 1):
        hours_left = r["total_hours"] - K
        ath_in = r["ath_hour"] - K if r["ath_hour"] > K else "already passed"
        print(f"\n  #{i} {r['symbol']} ({r['name']}) — distance: {r['distance']:.4f}")
        print(f"     ATH: ${r['ath_mcap']:,.0f} (at hour {r['ath_hour']})")
        print(f"     Total lifespan: {r['total_hours']}h")
        print(f"     Hours after current point: {hours_left}h")
        print(f"     ATH relative to now: {ath_in}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Find similar historical tokens")
    parser.add_argument("address", help="Contract address of the new token")
    parser.add_argument("--top", type=int, default=5, help="Number of matches to return")
    parser.add_argument("--chain", default="sol", help="Blockchain (default: sol)")
    args = parser.parse_args()

    # Load index and metadata
    sig_index = load_index()
    metadata = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))

    # Fetch new token data
    print(f"Fetching data for {args.address[:8]}...")
    normalized, raw = fetch_new_token_trajectory(args.chain, args.address)

    if normalized is None:
        print("ERROR: Could not fetch or process token data", file=sys.stderr)
        sys.exit(1)

    K = len(normalized)
    print(f"Token has {K} hours of data from 100K onward")
    print(f"Current mcap: ${raw[-1, 0]:,.0f}")

    # Find similar
    results = find_similar(normalized, sig_index, metadata, top_n=args.top)

    if not results:
        print("No matching historical tokens found at this prefix length")
        sys.exit(0)

    print_results(results, K)

    return results, K, raw


if __name__ == "__main__":
    main()
```

**Step 2: Test with the known token**

```bash
source .venv/bin/activate
python query.py FtSRgyCEhKTc1PPgEAXvuHN3NyiP6LS9uyB28KCN3CAP --top 5
```

Expected: Fetches data, computes signature, finds top-5 similar historical tokens, prints their ATH and continuation info.

**Step 3: Commit**

```bash
git add query.py
git commit -m "Add query engine: path signature prefix matching with top-N retrieval"
```

---

### Task 5: Prediction visualization (`visualize.py`)

**Files:**
- Create: `visualize.py`

Show the new token's trajectory so far, overlaid with the continuations of matching historical tokens. This is the key output: "here's where you are, here's what happened to similar tokens."

**Step 1: Write visualize.py**

```python
"""Visualize prediction: new token trajectory + similar historical continuations.

Creates a chart showing:
- The new token's trajectory so far (black, bold)
- Top-N matching historical tokens' full trajectories (colored, with the matched
  prefix portion solid and the continuation/prediction portion highlighted)

Each of the 4 DNA dimensions gets its own subplot.

Usage: called from query.py or standalone
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_DIR = "data"
TRAJ_DIR = os.path.join(DATA_DIR, "trajectories")
PLOTS_DIR = "plots"

DNA_LABELS = ["Market Cap", "Holders", "Top10 Holder %", "MCap / Holder"]
MATCH_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def load_raw_trajectory(address):
    """Load raw (un-normalized) trajectory from cached JSON data."""
    import json
    import glob
    from dna_extractor import extract_candle_series, extract_trend_series, build_dna_dataframe
    from preprocess import DNA_COLUMNS, MCAP_THRESHOLD

    token_dir = os.path.join(DATA_DIR, address)
    candles_file = sorted(glob.glob(os.path.join(token_dir, "token_mcap_candles_*.json")))
    trends_file = sorted(glob.glob(os.path.join(token_dir, "token_trends_*.json")))
    if not candles_file or not trends_file:
        return None

    with open(candles_file[-1]) as f:
        candles_data = json.load(f)
    with open(trends_file[-1]) as f:
        trends_data = json.load(f)

    candle_df = extract_candle_series(candles_data)
    trend_df = extract_trend_series(trends_data)
    if candle_df is None or trend_df is None:
        return None

    dna_df = build_dna_dataframe(candle_df, trend_df)
    if dna_df is None:
        return None

    mcap = dna_df["mcap"].values
    above = np.where(mcap >= MCAP_THRESHOLD)[0]
    if len(above) == 0:
        return None

    trimmed = dna_df.iloc[above[0]:].reset_index(drop=True)
    return trimmed[DNA_COLUMNS].values


def plot_prediction(new_token_raw, new_symbol, matches, K):
    """Create prediction visualization.

    Args:
        new_token_raw: (K, 4) raw DNA values of the new token
        new_symbol: symbol of the new token
        matches: list of dicts with 'address', 'symbol', 'distance'
        K: current prefix length (hours)
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=DNA_LABELS,
        vertical_spacing=0.06,
    )

    # Plot new token (bold black)
    hours = list(range(K))
    for dim in range(4):
        fig.add_trace(go.Scatter(
            x=hours, y=new_token_raw[:, dim],
            mode="lines",
            line=dict(color="black", width=3),
            name=f"{new_symbol} (current)" if dim == 0 else None,
            legendgroup="new",
            showlegend=(dim == 0),
        ), row=dim + 1, col=1)

    # Plot each matching historical token
    for i, match in enumerate(matches):
        raw = load_raw_trajectory(match["address"])
        if raw is None:
            continue

        color = MATCH_COLORS[i % len(MATCH_COLORS)]
        symbol = match["symbol"]
        T = len(raw)
        hours_full = list(range(T))

        for dim in range(4):
            # Matched prefix portion (solid)
            prefix_len = min(K, T)
            fig.add_trace(go.Scatter(
                x=hours_full[:prefix_len],
                y=raw[:prefix_len, dim],
                mode="lines",
                line=dict(color=color, width=1.5, dash="dot"),
                name=f"{symbol} (d={match['distance']:.3f})" if dim == 0 else None,
                legendgroup=f"match_{i}",
                showlegend=(dim == 0),
                opacity=0.6,
            ), row=dim + 1, col=1)

            # Continuation/prediction portion (bold, highlighted)
            if T > K:
                fig.add_trace(go.Scatter(
                    x=hours_full[K - 1:],  # overlap by 1 for continuity
                    y=raw[K - 1:, dim],
                    mode="lines",
                    line=dict(color=color, width=2.5),
                    legendgroup=f"match_{i}",
                    showlegend=False,
                    opacity=0.8,
                ), row=dim + 1, col=1)

    # Add vertical line at current hour K
    for dim in range(4):
        fig.add_vline(
            x=K, line_dash="dash", line_color="red", line_width=1,
            annotation_text="NOW" if dim == 0 else None,
            row=dim + 1, col=1,
        )

    fig.update_layout(
        title=f"Trajectory Prediction for {new_symbol} — Top {len(matches)} Matches at Hour {K}",
        height=1000,
        width=1200,
        legend=dict(x=1.02, y=1),
    )
    fig.update_xaxes(title_text="Hours from 100K", row=4, col=1)

    html_path = os.path.join(PLOTS_DIR, f"prediction_{new_symbol}.html")
    fig.write_html(html_path)
    print(f"Prediction plot saved: {html_path}")

    png_path = os.path.join(PLOTS_DIR, f"prediction_{new_symbol}.png")
    fig.write_image(png_path, width=1200, height=1000, scale=2)
    print(f"Static plot saved: {png_path}")

    return html_path
```

**Step 2: Integrate with query.py**

Add to the end of query.py's main():

```python
    # Visualize
    from visualize import plot_prediction
    html_path = plot_prediction(raw, args.address[:8], results, K)
```

**Step 3: Test**

```bash
source .venv/bin/activate
python query.py FtSRgyCEhKTc1PPgEAXvuHN3NyiP6LS9uyB28KCN3CAP --top 5
open plots/prediction_FtSRgyCE.html
```

Expected: Opens interactive chart with new token's trajectory (black) and 5 similar historical tokens' full trajectories. Red vertical line at "NOW". Continuation portions are bold colored.

**Step 4: Commit**

```bash
git add visualize.py query.py
git commit -m "Add prediction visualization: new token vs similar historical continuations"
```

---

### Task 6: End-to-end test run

**Step 1: Run full pipeline**

```bash
source .venv/bin/activate

# Step 1: Preprocess (recompute with new normalization)
python preprocess.py

# Step 2: Build signature index
python signature_index.py

# Step 3: Query a token
python query.py FtSRgyCEhKTc1PPgEAXvuHN3NyiP6LS9uyB28KCN3CAP --top 5
```

**Step 2: Open visualization**

```bash
open plots/prediction_*.html
```

**Step 3: Verify results make sense**

Check:
- Matches have reasonable distance scores
- Historical continuations show diverse outcomes (not all identical)
- The 4 dimensions are plotted correctly
- The "NOW" line is at the right position

**Step 4: Commit and clean up**

```bash
git add -A
git commit -m "Path signature trajectory matching pipeline complete"
```
