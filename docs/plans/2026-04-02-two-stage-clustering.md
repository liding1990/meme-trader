# Two-Stage Clustering Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace single-pass DTW clustering with a two-stage system: first split tokens into outcome tiers (Runner/Mid/Weak) by result metrics, then cluster trajectories within each tier to discover behavior patterns — ensuring high intra-cluster result consistency.

**Architecture:** Stage 1 extracts 4 outcome scalars (ATH, price speed, volume speed, holder speed) and uses KMeans(k=3) to create tiers. Stage 2 resamples full trajectories into 4D (log_mcap, log_holders, log_volume, top10_concentration), computes DTW distance matrices within each tier, and runs HDBSCAN directly on the precomputed distance (no UMAP intermediary). Purity scores validate cluster quality.

**Tech Stack:** pandas, numpy, scikit-learn (KMeans), dtaidistance (DTW), hdbscan, plotly, streamlit

---

### Task 1: Extract outcome features

**Files:**
- Modify: `app_3d.py` — add `compute_outcome_features()` after `build_trajectory()` (line ~170)

**Step 1: Write `compute_outcome_features`**

Add this function after `build_trajectory` in `app_3d.py`:

```python
def compute_outcome_features(df: pd.DataFrame) -> dict | None:
    """Extract 4 outcome scalars from a token trajectory.

    Returns dict with: ath, price_speed, volume_speed, holder_speed
    All speeds are per-hour rates measured from start to ATH.
    """
    if df is None or len(df) < 2:
        return None

    mcap = df["mcap"].values
    ath_idx = np.argmax(mcap)
    ath = mcap[ath_idx]
    hours_to_ath = max(df["hours"].iloc[ath_idx], 1)  # avoid div by 0

    # Price speed: mcap gain per hour
    price_speed = (ath - MCAP_THRESHOLD) / hours_to_ath

    # Volume speed: cumulative volume to ATH / hours
    vol_to_ath = df["volume"].iloc[:ath_idx + 1].sum()
    volume_speed = vol_to_ath / hours_to_ath

    # Holder speed: holder gain per hour
    holders_start = df["holders"].iloc[0]
    holders_at_ath = df["holders"].iloc[ath_idx]
    holder_speed = (holders_at_ath - holders_start) / hours_to_ath

    return {
        "ath": ath,
        "price_speed": price_speed,
        "volume_speed": volume_speed,
        "holder_speed": holder_speed,
    }
```

**Step 2: Verify it works**

Run in Python REPL:
```bash
cd /Users/liding/code/meme-trader && source .venv/bin/activate && python -c "
from app_3d import load_from_cache, build_trajectory, compute_outcome_features, load_radar_metadata
meta = load_radar_metadata()
addr = list(meta.keys())[0]
mcap, trend = load_from_cache(addr)
df = build_trajectory(mcap, trend, address=addr)
if df is not None:
    feat = compute_outcome_features(df)
    print(f'Token: {meta[addr][\"symbol\"]}')
    print(f'ATH: \${feat[\"ath\"]:,.0f}')
    print(f'Price speed: \${feat[\"price_speed\"]:,.0f}/h')
    print(f'Volume speed: \${feat[\"volume_speed\"]:,.0f}/h')
    print(f'Holder speed: {feat[\"holder_speed\"]:,.1f}/h')
else:
    print('No trajectory for first token, try another')
"
```

Expected: prints 4 scalar values without errors.

**Step 3: Commit**

```bash
git add app_3d.py
git commit -m "feat: add compute_outcome_features for outcome tier extraction"
```

---

### Task 2: Stage 1 — KMeans outcome tiering

**Files:**
- Modify: `app_3d.py` — add `assign_outcome_tiers()` function, replace `cluster_tokens_dtw`

**Step 1: Write `assign_outcome_tiers`**

Add after `compute_outcome_features`:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def assign_outcome_tiers(trajectories: dict) -> tuple[dict, np.ndarray, list]:
    """Stage 1: Split tokens into 3 outcome tiers via KMeans on result metrics.

    Returns:
        tier_map: {address: tier_label} where tier_label is 1 (Runner), 2 (Mid), 3 (Weak)
        feature_matrix: (N, 4) log-scaled outcome features
        addrs: list of addresses in same order as feature_matrix
    """
    addrs = []
    features = []
    for addr, df in trajectories.items():
        feat = compute_outcome_features(df)
        if feat is None:
            continue
        addrs.append(addr)
        features.append([
            np.log1p(max(feat["ath"], 0)),
            np.log1p(max(feat["price_speed"], 0)),
            np.log1p(max(feat["volume_speed"], 0)),
            np.log1p(max(feat["holder_speed"], 0)),
        ])

    X = np.array(features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    # Rank tiers by median ATH: highest ATH cluster = Tier 1 (Runner)
    tier_aths = {}
    for i, addr in enumerate(addrs):
        lbl = labels[i]
        tier_aths.setdefault(lbl, []).append(X[i, 0])  # log ATH

    tier_rank = sorted(tier_aths.keys(), key=lambda k: np.median(tier_aths[k]), reverse=True)
    label_to_tier = {lbl: rank + 1 for rank, lbl in enumerate(tier_rank)}

    tier_map = {addr: label_to_tier[labels[i]] for i, addr in enumerate(addrs)}
    return tier_map, X, addrs
```

**Step 2: Verify tiers make sense**

```bash
cd /Users/liding/code/meme-trader && source .venv/bin/activate && python -c "
from app_3d import load_all_trajectories, assign_outcome_tiers
import numpy as np
trajectories, meta, skipped = load_all_trajectories.__wrapped__()
tier_map, X, addrs = assign_outcome_tiers(trajectories)
for t in [1, 2, 3]:
    members = [a for a, v in tier_map.items() if v == t]
    aths = [trajectories[a]['mcap'].max() for a in members]
    print(f'Tier {t}: {len(members)} tokens, median ATH: \${np.median(aths):,.0f}, range: \${min(aths):,.0f} - \${max(aths):,.0f}')
"
```

Expected: 3 tiers with clearly separated ATH ranges. Tier 1 should have the highest ATH tokens.

**Step 3: Commit**

```bash
git add app_3d.py
git commit -m "feat: add KMeans outcome tiering (Stage 1)"
```

---

### Task 3: Load top10 concentration data

**Files:**
- Modify: `app_3d.py` — update `build_trajectory` to include `top10_pct` column

**Step 1: Add top10 concentration loading**

In `build_trajectory()`, after the holder data merge (around line 152), add top10 concentration loading and merge:

```python
# Inside build_trajectory(), after merged is built and sorted:

# Load top10 holder concentration from trends
top10_series = None
if trend_data:
    trend_section = trend_data.get("data")
    if trend_section:
        trends = trend_section.get("trends", {})
        t10 = trends.get("top10_holder_percent", [])
        if t10:
            top10_df = pd.DataFrame(t10)
            top10_df["datetime"] = pd.to_datetime(top10_df["timestamp"].astype(int), unit="s")
            top10_df["top10_pct"] = top10_df["value"].astype(float)
            top10_df = top10_df[["datetime", "top10_pct"]].sort_values("datetime")
            top10_df["datetime"] = top10_df["datetime"].dt.floor("h")
            top10_df = top10_df.groupby("datetime", as_index=False).last()
            merged = pd.merge(merged, top10_df, on="datetime", how="left")
            merged["top10_pct"] = merged["top10_pct"].ffill().bfill().fillna(0.5)

if "top10_pct" not in merged.columns:
    merged["top10_pct"] = 0.5  # default 50% if no data
```

**Step 2: Verify top10_pct column exists**

```bash
cd /Users/liding/code/meme-trader && source .venv/bin/activate && python -c "
from app_3d import load_from_cache, build_trajectory, load_radar_metadata
meta = load_radar_metadata()
count = 0
has_real = 0
for addr in list(meta.keys())[:20]:
    mcap, trend = load_from_cache(addr)
    df = build_trajectory(mcap, trend, address=addr)
    if df is not None:
        count += 1
        if df['top10_pct'].nunique() > 1:
            has_real += 1
        print(f'{meta[addr][\"symbol\"]}: top10_pct range {df[\"top10_pct\"].min():.2f}-{df[\"top10_pct\"].max():.2f} ({len(df)} rows)')
print(f'{has_real}/{count} tokens have real top10 data')
"
```

Expected: most tokens show varying top10_pct values (not all 0.5).

**Step 3: Commit**

```bash
git add app_3d.py
git commit -m "feat: add top10 holder concentration to trajectory data"
```

---

### Task 4: Stage 2 — Within-tier DTW clustering

**Files:**
- Modify: `app_3d.py` — rewrite `cluster_tokens_dtw` → `cluster_two_stage`

**Step 1: Update `resample_trajectory` to 4D**

Replace existing `resample_trajectory`:

```python
def resample_trajectory(df: pd.DataFrame, n_points: int = RESAMPLE_LEN) -> np.ndarray:
    """Resample trajectory to fixed points via arc-length interpolation in 4D log space."""
    log_mcap = np.log1p(df["mcap"].values)
    log_holders = np.log1p(df["holders"].values)
    log_volume = np.log1p(df["volume"].values) if "volume" in df.columns else np.zeros_like(log_mcap)
    top10 = df["top10_pct"].values if "top10_pct" in df.columns else np.full_like(log_mcap, 0.5)

    points = np.column_stack([log_mcap, log_holders, log_volume, top10])
    diffs = np.diff(points, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    arc = np.concatenate([[0], np.cumsum(seg_lens)])

    total_len = arc[-1]
    if total_len < 1e-9:
        return np.tile(points[0], (n_points, 1))

    target_arc = np.linspace(0, total_len, n_points)
    resampled = np.column_stack([
        np.interp(target_arc, arc, col) for col in points.T
    ])
    return resampled
```

**Step 2: Write `cluster_two_stage`**

Replace `cluster_tokens_dtw` with:

```python
@st.cache_data
def cluster_two_stage(_trajectories: dict):
    """Two-stage clustering: outcome tiers → within-tier trajectory clustering.

    Stage 1: KMeans on outcome features → 3 tiers (Runner/Mid/Weak)
    Stage 2: Full-trajectory DTW + HDBSCAN within each tier

    Returns:
        tier_map: {addr: 1|2|3}
        cluster_map: {addr: (tier, sub_cluster_id)}
    """
    # Stage 1: outcome tiers
    tier_map, _, _ = assign_outcome_tiers(_trajectories)

    # Stage 2: within-tier trajectory clustering
    cluster_map = {}

    for tier in [1, 2, 3]:
        tier_addrs = [a for a, t in tier_map.items() if t == tier]
        if len(tier_addrs) < 2:
            for a in tier_addrs:
                cluster_map[a] = (tier, 0)
            continue

        # Resample full trajectories (not trimmed to ATH)
        series_list = [resample_trajectory(_trajectories[a]) for a in tier_addrs]

        # Z-normalize per trajectory
        series_znorm = [_z_normalize(s) for s in series_list]

        # Shape DTW distance
        dist_shape = np.array(dtw_ndim.distance_matrix_fast(series_znorm))
        dist_shape[np.isinf(dist_shape)] = 0
        dist_shape = dist_shape + dist_shape.T

        # Derivative DTW distance
        series_deriv = [_z_normalize(np.diff(s, axis=0)) for s in series_list]
        dist_deriv = np.array(dtw_ndim.distance_matrix_fast(series_deriv))
        dist_deriv[np.isinf(dist_deriv)] = 0
        dist_deriv = dist_deriv + dist_deriv.T

        # Normalize and blend 50/50
        d_shape_max = dist_shape.max()
        d_deriv_max = dist_deriv.max()
        if d_shape_max > 0:
            dist_shape /= d_shape_max
        if d_deriv_max > 0:
            dist_deriv /= d_deriv_max
        dist_blend = 0.5 * dist_shape + 0.5 * dist_deriv

        # HDBSCAN directly on precomputed distance (no UMAP)
        min_cs = max(3, len(tier_addrs) // 10)  # adaptive min_cluster_size
        clusterer = HDBSCAN(
            min_cluster_size=min_cs,
            min_samples=2,
            metric="precomputed",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(dist_blend)

        for i, addr in enumerate(tier_addrs):
            cluster_map[addr] = (tier, int(labels[i]))

    return tier_map, cluster_map
```

**Step 3: Verify clustering output**

```bash
cd /Users/liding/code/meme-trader && source .venv/bin/activate && python -c "
from app_3d import load_all_trajectories, cluster_two_stage
import numpy as np
trajectories, meta, skipped = load_all_trajectories.__wrapped__()
tier_map, cluster_map = cluster_two_stage.__wrapped__(trajectories)
for tier in [1, 2, 3]:
    members = [(a, c) for a, (t, c) in cluster_map.items() if t == tier]
    sub_ids = set(c for _, c in members)
    aths = [trajectories[a]['mcap'].max() for a, _ in members]
    print(f'Tier {tier}: {len(members)} tokens, {len(sub_ids)} sub-clusters, ATH median: \${np.median(aths):,.0f}')
    for sid in sorted(sub_ids):
        sub_members = [a for a, c in members if c == sid]
        sub_aths = [trajectories[a]['mcap'].max() for a in sub_members]
        label = 'noise' if sid == -1 else f'sub-{sid}'
        print(f'  {label}: {len(sub_members)} tokens, ATH \${np.median(sub_aths):,.0f} (CV={np.std(sub_aths)/max(np.mean(sub_aths),1):.2f})')
"
```

Expected: Within each tier, sub-clusters show low ATH CV (< 0.5 ideally). Tiers have clearly separated ATH ranges.

**Step 4: Commit**

```bash
git add app_3d.py
git commit -m "feat: two-stage clustering — outcome tiers + within-tier DTW HDBSCAN"
```

---

### Task 5: Purity score computation

**Files:**
- Modify: `app_3d.py` — add `compute_purity` function

**Step 1: Write purity scoring**

Add after `cluster_two_stage`:

```python
def compute_purity(trajectories: dict, cluster_map: dict) -> dict:
    """Compute purity score for each (tier, sub_cluster).

    Purity = 1 - mean(CV) across 4 outcome metrics.
    CV = std/mean (coefficient of variation). Lower CV = more consistent results.
    Returns {(tier, sub_id): {"purity": float, "n": int, "stats": dict}}
    """
    # Group by (tier, sub_cluster)
    groups = {}
    for addr, (tier, sub_id) in cluster_map.items():
        groups.setdefault((tier, sub_id), []).append(addr)

    result = {}
    for key, addrs in groups.items():
        feats = [compute_outcome_features(trajectories[a]) for a in addrs]
        feats = [f for f in feats if f is not None]
        if len(feats) < 2:
            result[key] = {"purity": 0.0, "n": len(addrs), "stats": {}}
            continue

        metrics = {}
        for metric in ["ath", "price_speed", "volume_speed", "holder_speed"]:
            vals = np.array([f[metric] for f in feats])
            vals = vals[vals > 0]  # exclude zeros
            if len(vals) >= 2:
                cv = np.std(vals) / max(np.mean(vals), 1e-9)
            else:
                cv = 1.0
            metrics[metric] = {"median": float(np.median(vals)) if len(vals) else 0, "cv": cv}

        avg_cv = np.mean([m["cv"] for m in metrics.values()])
        purity = max(0, 1 - avg_cv)

        result[key] = {"purity": purity, "n": len(addrs), "stats": metrics}

    return result
```

**Step 2: Verify purity scores**

```bash
cd /Users/liding/code/meme-trader && source .venv/bin/activate && python -c "
from app_3d import load_all_trajectories, cluster_two_stage, compute_purity
trajectories, meta, skipped = load_all_trajectories.__wrapped__()
tier_map, cluster_map = cluster_two_stage.__wrapped__(trajectories)
purity = compute_purity(trajectories, cluster_map)
for key in sorted(purity.keys()):
    p = purity[key]
    if p['n'] >= 3:
        print(f'Tier {key[0]} Sub {key[1]:>2d}: {p[\"n\"]:>3d} tokens, purity={p[\"purity\"]:.2f}')
"
```

Expected: purity values between 0-1, with most clusters > 0.3.

**Step 3: Commit**

```bash
git add app_3d.py
git commit -m "feat: add cluster purity score computation"
```

---

### Task 6: Update UI — sidebar and color scheme

**Files:**
- Modify: `app_3d.py` — rewrite sidebar and `build_figure` for two-stage display

**Step 1: Define tier color scheme**

Replace existing `GRADE_COLORS`/`GRADE_LABELS` with:

```python
TIER_NAMES = {1: "Runner", 2: "Mid", 3: "Weak"}
TIER_BASE_COLORS = {
    1: [(255, 215, 0), (255, 180, 0), (218, 165, 32), (184, 134, 11)],     # gold shades
    2: [(59, 130, 246), (37, 99, 235), (29, 78, 216), (30, 64, 175)],       # blue shades
    3: [(156, 163, 175), (107, 114, 128), (75, 85, 99), (55, 65, 81)],      # gray shades
}
NOISE_COLOR = "#e41a1c"


def get_cluster_color(tier: int, sub_id: int) -> str:
    """Get color for a (tier, sub_cluster) combination."""
    if sub_id == -1:
        return NOISE_COLOR
    shades = TIER_BASE_COLORS.get(tier, TIER_BASE_COLORS[3])
    r, g, b = shades[sub_id % len(shades)]
    return f"rgb({r},{g},{b})"
```

**Step 2: Rewrite sidebar**

Replace the existing sidebar section (starting at line ~604) with:

```python
# Sidebar: tier and cluster toggles
with st.sidebar:
    st.header("Token Tiers & Clusters")
    st.caption(f"{len(trajectories)} tokens · {skipped} skipped")

    visible = set()  # set of (tier, sub_id) tuples

    for tier in [1, 2, 3]:
        tier_name = TIER_NAMES[tier]
        tier_members = [a for a, (t, _) in cluster_map.items() if t == tier]
        st.subheader(f"Tier {tier}: {tier_name} ({len(tier_members)})")

        # Get sub-clusters in this tier
        sub_ids = sorted(set(s for a, (t, s) in cluster_map.items() if t == tier))

        for sub_id in sub_ids:
            key = (tier, sub_id)
            p = purity_scores.get(key, {})
            n = p.get("n", 0)
            purity_val = p.get("purity", 0)
            stats = p.get("stats", {})
            med_ath = stats.get("ath", {}).get("median", 0)

            if sub_id == -1:
                label = f"  Noise ({n})"
                default = False
            else:
                label = f"  C{sub_id} ({n}) — ${med_ath:,.0f} ATH, purity {purity_val:.0%}"
                default = tier <= 2  # show Runner + Mid by default

            if st.checkbox(label, value=default, key=f"t{tier}_c{sub_id}"):
                visible.add(key)
```

**Step 3: Rewrite `build_figure`**

Update `build_figure` signature and body to use `(tier, sub_id)` cluster keys:

```python
def build_figure(trajectories: dict, token_meta: dict, cluster_map: dict,
                 visible: set, purity_scores: dict, query_token: tuple = None):
    fig = go.Figure()

    for addr, df in trajectories.items():
        key = cluster_map.get(addr)
        if key is None or key not in visible:
            continue

        tier, sub_id = key
        color = get_cluster_color(tier, sub_id)
        width = 3 if tier == 1 else 2.5 if tier == 2 else 1.5

        meta = token_meta.get(addr, {})
        name = meta.get("name", "Unknown")
        symbol = meta.get("symbol", addr[:8])
        tier_name = TIER_NAMES[tier]
        p = purity_scores.get(key, {})
        purity_val = p.get("purity", 0)

        fig.add_trace(go.Scatter3d(
            x=df["hours"],
            y=df["mcap"],
            z=df["holders"],
            mode="lines",
            line=dict(color=color, width=width),
            text=[
                f"<b>{symbol}</b> ({name})<br>"
                f"Tier {tier} {tier_name} / C{sub_id}<br>"
                f"Purity: {purity_val:.0%}<br>"
                f"T+{row['hours']:.0f}h<br>"
                f"MCap: ${row['mcap']:,.0f}<br>"
                f"Holders: {row['holders']:,.0f}"
                for _, row in df.iterrows()
            ],
            hoverinfo="text",
            name=symbol,
            showlegend=False,
        ))

    # Query token overlay (same as before, adapt key lookup)
    if query_token is not None:
        q_df, q_symbol, q_tier, q_sub = query_token
        color = get_cluster_color(q_tier, q_sub)
        fig.add_trace(go.Scatter3d(
            x=q_df["hours"], y=q_df["mcap"], z=q_df["holders"],
            mode="lines+markers",
            line=dict(color=QUERY_TOKEN_COLOR, width=6),
            marker=dict(size=3, color=QUERY_TOKEN_COLOR),
            text=[
                f"<b>★ {q_symbol}</b> (QUERY)<br>"
                f"→ Tier {q_tier} {TIER_NAMES[q_tier]} / C{q_sub}<br>"
                f"T+{row['hours']:.0f}h<br>"
                f"MCap: ${row['mcap']:,.0f}<br>"
                f"Holders: {row['holders']:,.0f}"
                for _, row in q_df.iterrows()
            ],
            hoverinfo="text",
            name=f"★ {q_symbol} (query)",
            showlegend=True,
        ))
        fig.add_trace(go.Scatter3d(
            x=[q_df["hours"].iloc[0]], y=[q_df["mcap"].iloc[0]], z=[q_df["holders"].iloc[0]],
            mode="markers", marker=dict(size=8, color=color, symbol="diamond"),
            showlegend=False, hovertext=f"{q_symbol} START", hoverinfo="text",
        ))
        fig.add_trace(go.Scatter3d(
            x=[q_df["hours"].iloc[-1]], y=[q_df["mcap"].iloc[-1]], z=[q_df["holders"].iloc[-1]],
            mode="markers", marker=dict(size=10, color=color, symbol="diamond"),
            showlegend=False, hovertext=f"{q_symbol} NOW", hoverinfo="text",
        ))

    n_visible = sum(1 for a in trajectories if cluster_map.get(a) in visible)
    fig.update_layout(
        title=dict(
            text=f"Token Trajectories ({n_visible} visible) — Two-Stage Clustering",
            font=dict(size=16, color="black"),
        ),
        scene=dict(
            xaxis_title="Time (hours since MCap > $50K)",
            yaxis_title="Market Cap ($)",
            zaxis_title="Holders",
            xaxis=dict(backgroundcolor="white", gridcolor="rgb(200,200,200)"),
            yaxis=dict(backgroundcolor="white", gridcolor="rgb(200,200,200)", type="log"),
            zaxis=dict(backgroundcolor="white", gridcolor="rgb(200,200,200)"),
            bgcolor="white",
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        height=800,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig
```

**Step 4: Commit**

```bash
git add app_3d.py
git commit -m "feat: update UI for two-stage tier/cluster display with purity scores"
```

---

### Task 7: Update classify_token for two-stage

**Files:**
- Modify: `app_3d.py` — rewrite `classify_token` for two-stage lookup

**Step 1: Rewrite `classify_token`**

```python
def classify_token(new_df: pd.DataFrame, trajectories: dict, tier_map: dict,
                   cluster_map: dict) -> tuple:
    """Classify a new token: determine tier, then find nearest sub-cluster.

    Returns (tier, sub_cluster_id, tier_distances, cluster_distances).
    """
    # Stage 1: determine tier by outcome features
    feat = compute_outcome_features(new_df)
    if feat is None:
        return 1, 0, {}, {}

    # Compare to tier centroids (median of each tier's features)
    tier_dists = {}
    for tier in [1, 2, 3]:
        tier_addrs = [a for a, t in tier_map.items() if t == tier]
        if not tier_addrs:
            continue
        tier_feats = [compute_outcome_features(trajectories[a]) for a in tier_addrs]
        tier_feats = [f for f in tier_feats if f is not None]
        if not tier_feats:
            continue
        # Euclidean distance in log-space
        centroid = np.array([
            np.median([np.log1p(max(f["ath"], 0)) for f in tier_feats]),
            np.median([np.log1p(max(f["price_speed"], 0)) for f in tier_feats]),
            np.median([np.log1p(max(f["volume_speed"], 0)) for f in tier_feats]),
            np.median([np.log1p(max(f["holder_speed"], 0)) for f in tier_feats]),
        ])
        point = np.array([
            np.log1p(max(feat["ath"], 0)),
            np.log1p(max(feat["price_speed"], 0)),
            np.log1p(max(feat["volume_speed"], 0)),
            np.log1p(max(feat["holder_speed"], 0)),
        ])
        tier_dists[tier] = np.linalg.norm(point - centroid)

    best_tier = min(tier_dists, key=tier_dists.get) if tier_dists else 1

    # Stage 2: DTW distance to sub-clusters within the tier
    new_resampled = resample_trajectory(new_df)
    new_znorm = _z_normalize(new_resampled)
    new_deriv = _z_normalize(np.diff(new_resampled, axis=0))

    sub_clusters = {}
    for addr, (t, s) in cluster_map.items():
        if t == best_tier and s >= 0:
            sub_clusters.setdefault(s, []).append(addr)

    cluster_dists = {}
    for sub_id, addrs in sub_clusters.items():
        shape_dists = []
        deriv_dists = []
        for addr in addrs:
            member = resample_trajectory(trajectories[addr])
            member_znorm = _z_normalize(member)
            member_deriv = _z_normalize(np.diff(member, axis=0))
            shape_dists.append(dtw_ndim.distance(new_znorm, member_znorm))
            deriv_dists.append(dtw_ndim.distance(new_deriv, member_deriv))
        cluster_dists[sub_id] = 0.5 * np.mean(shape_dists) + 0.5 * np.mean(deriv_dists)

    best_sub = min(cluster_dists, key=cluster_dists.get) if cluster_dists else 0
    return best_tier, best_sub, tier_dists, cluster_dists
```

**Step 2: Update query section in main UI**

Update the query section (around line 657) to use the new classify_token signature and pass `(q_df, q_symbol, best_tier, best_sub)` as `query_token_result`.

**Step 3: Commit**

```bash
git add app_3d.py
git commit -m "feat: update classify_token for two-stage tier + sub-cluster lookup"
```

---

### Task 8: Wire up main Streamlit flow

**Files:**
- Modify: `app_3d.py` — replace the main execution flow (lines ~586-763)

**Step 1: Rewrite the main flow**

Replace everything from `st.set_page_config` onward:

```python
st.set_page_config(page_title="3D Token Trajectories", layout="wide")

with st.spinner("Loading radar token data..."):
    trajectories, token_meta, skipped = load_all_trajectories()

with st.spinner("Stage 1: Computing outcome tiers..."):
    tier_map, cluster_map = cluster_two_stage(trajectories)

with st.spinner("Computing purity scores..."):
    purity_scores = compute_purity(trajectories, cluster_map)

# Sidebar
with st.sidebar:
    st.header("Token Tiers & Clusters")
    st.caption(f"{len(trajectories)} tokens · {skipped} skipped")

    visible = set()

    for tier in [1, 2, 3]:
        tier_name = TIER_NAMES[tier]
        tier_members = [a for a, (t, _) in cluster_map.items() if t == tier]
        st.subheader(f"Tier {tier}: {tier_name} ({len(tier_members)})")

        sub_ids = sorted(set(s for a, (t, s) in cluster_map.items() if t == tier))

        for sub_id in sub_ids:
            key = (tier, sub_id)
            p = purity_scores.get(key, {})
            n = p.get("n", 0)
            purity_val = p.get("purity", 0)
            stats = p.get("stats", {})
            med_ath = stats.get("ath", {}).get("median", 0)

            if sub_id == -1:
                label = f"Noise ({n})"
                default = False
            else:
                label = f"C{sub_id} ({n}) — ${med_ath:,.0f} ATH, purity {purity_val:.0%}"
                default = tier <= 2

            if st.checkbox(label, value=default, key=f"t{tier}_c{sub_id}"):
                visible.add(key)

# Token query
st.sidebar.divider()
st.sidebar.header("Classify New Token")
query_address = st.sidebar.text_input("Contract address", placeholder="Enter Solana token address...", key="query_addr")
query_chain = st.sidebar.selectbox("Chain", ["sol", "base"], index=0, key="query_chain")

query_token_result = None

if query_address:
    with st.spinner(f"Fetching data for {query_address[:12]}..."):
        try:
            q_df = fetch_and_build_trajectory(query_address, chain=query_chain)
        except Exception as e:
            q_df = None
            st.sidebar.error(f"Fetch error: {e}")

    if q_df is not None and len(q_df) >= 2:
        with st.spinner("Classifying token..."):
            best_tier, best_sub, tier_dists, cluster_dists = classify_token(
                q_df, trajectories, tier_map, cluster_map
            )

        q_symbol = query_address[:8]
        q_meta_path = os.path.join(DATA_DIR, query_address)
        price_files = sorted(glob.glob(os.path.join(q_meta_path, "token_mcap_candles_*.json")))
        if price_files:
            try:
                with open(price_files[-1]) as f:
                    pdata = json.load(f)
                s = pdata.get("data", {}).get("symbol")
                if s:
                    q_symbol = s
            except Exception:
                pass

        query_token_result = (q_df, q_symbol, best_tier, best_sub)

        st.sidebar.divider()
        tier_name = TIER_NAMES[best_tier]
        color = get_cluster_color(best_tier, best_sub)
        st.sidebar.markdown(
            f'### Result: <span style="color: {color};">Tier {best_tier} {tier_name} / C{best_sub}</span>',
            unsafe_allow_html=True,
        )
        st.sidebar.caption(f"Data: {len(q_df)} hours")
        st.sidebar.caption(f"Current MCap: ${q_df['mcap'].iloc[-1]:,.0f}")
        st.sidebar.caption(f"Peak MCap: ${q_df['mcap'].max():,.0f}")
        st.sidebar.caption(f"Holders: {q_df['holders'].iloc[-1]:,.0f}")
    elif query_address:
        st.sidebar.warning("Could not build trajectory (no data or mcap < $50K)")

# Main chart
fig = build_figure(trajectories, token_meta, cluster_map, visible,
                   purity_scores, query_token=query_token_result)
st.plotly_chart(fig, use_container_width=True)

# Cluster description panel
st.divider()
st.subheader("Cluster Analysis")

for tier in [1, 2, 3]:
    tier_visible = [(t, s) for t, s in visible if t == tier and s >= 0]
    if not tier_visible:
        continue

    tier_name = TIER_NAMES[tier]
    st.markdown(f"### Tier {tier}: {tier_name}")

    cols = st.columns(min(len(tier_visible), 3))
    for i, key in enumerate(sorted(tier_visible)):
        tier_id, sub_id = key
        addrs = [a for a, k in cluster_map.items() if k == key]
        p = purity_scores.get(key, {})
        purity_val = p.get("purity", 0)
        stats = p.get("stats", {})
        color = get_cluster_color(tier_id, sub_id)

        with cols[i % len(cols)]:
            st.markdown(
                f'<div style="border-left: 4px solid {color}; padding-left: 12px;">'
                f'<h4 style="margin: 0;">C{sub_id} ({len(addrs)} tokens) — Purity {purity_val:.0%}</h4>'
                f'</div>',
                unsafe_allow_html=True,
            )
            for metric, label in [("ath", "ATH"), ("price_speed", "Price Speed"),
                                   ("volume_speed", "Vol Speed"), ("holder_speed", "Holder Speed")]:
                m = stats.get(metric, {})
                med = m.get("median", 0)
                cv = m.get("cv", 0)
                st.caption(f"{label}: ${med:,.0f} (CV={cv:.2f})" if "speed" in metric or metric == "ath"
                           else f"{label}: {med:,.1f}/h (CV={cv:.2f})")

            with st.expander("Tokens"):
                for a in addrs:
                    m = token_meta.get(a, {})
                    ath = trajectories[a]["mcap"].max()
                    st.text(f"{m.get('symbol', '?'):>10s}  ATH ${ath:>12,.0f}")
```

**Step 2: Run the app and verify**

```bash
cd /Users/liding/code/meme-trader && source .venv/bin/activate && streamlit run app_3d.py --server.port 8502
```

Expected: app loads with 3 tier sections in sidebar, each with sub-clusters showing purity scores. 3D chart shows gold/blue/gray colored trajectories.

**Step 3: Commit**

```bash
git add app_3d.py
git commit -m "feat: wire up two-stage clustering UI with tier/cluster display"
```

---

### Task 9: Remove dead code

**Files:**
- Modify: `app_3d.py` — remove old functions no longer used

**Step 1: Remove these functions/constants:**
- `INSTANT_PEAK_LABEL`
- `cluster_tokens_dtw` (replaced by `cluster_two_stage`)
- `GRADE_COLORS`, `GRADE_LABELS`
- `score_cluster`
- `rank_clusters`
- `describe_cluster`
- Old `CLUSTER_COLORS` and `OUTLIER_COLOR`

**Step 2: Verify app still runs**

```bash
cd /Users/liding/code/meme-trader && source .venv/bin/activate && python -c "import app_3d; print('OK')"
```

**Step 3: Commit**

```bash
git add app_3d.py
git commit -m "chore: remove old single-pass clustering code"
```
