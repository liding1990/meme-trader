"""3D Token Trajectory Viewer — Radar tokens, blended shape+derivative DTW clustering."""

import csv
import json
import os
import glob

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dtaidistance import dtw_ndim
from hdbscan import HDBSCAN
import umap

from gmgn_api import fetch_token_data


DATA_DIR = "data"
RADAR_CSV = os.path.join(DATA_DIR, "radar_tokens.csv")
MCAP_THRESHOLD = 50_000
MAX_HOURS = 90 * 24  # 3 months max
RESAMPLE_LEN = 50
QUERY_TOKEN_COLOR = "#000000"  # black for queried token

CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]
OUTLIER_COLOR = "#e41a1c"  # red for outliers


def load_radar_metadata() -> dict:
    """Load token name/symbol from radar CSV. Returns {address: {name, symbol}}."""
    meta = {}
    if not os.path.isfile(RADAR_CSV):
        return meta
    with open(RADAR_CSV) as f:
        for row in csv.reader(f):
            if len(row) >= 4:
                meta[row[0]] = {"name": row[2], "symbol": row[3]}
    return meta


def load_from_cache(address: str):
    data_dir = os.path.join(DATA_DIR, address)
    if not os.path.isdir(data_dir):
        return None, None

    mcap_files = sorted([
        f for f in glob.glob(os.path.join(data_dir, "token_mcap_candles_[0-9]*.json"))
        if "5m" not in os.path.basename(f)
    ])
    trend_files = sorted(glob.glob(os.path.join(data_dir, "token_trends_*.json")))

    if not mcap_files or not trend_files:
        return None, None

    with open(mcap_files[-1]) as f:
        mcap_data = json.load(f)
    with open(trend_files[-1]) as f:
        trend_data = json.load(f)

    return mcap_data, trend_data


def _load_moralis_holders(address: str):
    """Load Moralis hourly holder data if available. Returns DataFrame or None."""
    moralis_path = os.path.join(DATA_DIR, address, "moralis_holders_1h.json")
    if not os.path.isfile(moralis_path):
        return None
    try:
        with open(moralis_path) as f:
            data = json.load(f)
        if not data:
            return None
        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
        df["holders"] = df["totalHolders"].astype(float)
        df = df[["datetime", "holders"]].sort_values("datetime").reset_index(drop=True)
        df = df[df["holders"] > 0]
        return df if len(df) >= 2 else None
    except Exception:
        return None


def _load_gmgn_holders(trend_data: dict):
    """Load GMGN trend holder data (daily granularity fallback). Returns DataFrame or None."""
    trend_section = trend_data.get("data") if trend_data else None
    if not trend_section:
        return None
    trends = trend_section.get("trends", {})
    holder_series = trends.get("holder_count", [])
    if not holder_series:
        return None
    df = pd.DataFrame(holder_series)
    df["datetime"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
    df["holders"] = df["value"].astype(float)
    df = df[["datetime", "holders"]].sort_values("datetime")
    df = df.set_index("datetime").resample("1h").last().dropna().reset_index()
    return df if len(df) >= 2 else None


def build_trajectory(mcap_data: dict, trend_data: dict, address: str = None):
    """Build hourly trajectory starting from first hour mcap >= 50K, capped at 3 months.

    Uses Moralis hourly holders when available (true hourly granularity),
    falls back to GMGN trends (daily granularity) otherwise.
    """
    if not mcap_data:
        return None
    data_section = mcap_data.get("data")
    if not data_section:
        return None
    candles = data_section.get("list", [])
    if not candles:
        return None

    mcap_df = pd.DataFrame(candles)
    mcap_df["datetime"] = pd.to_datetime(mcap_df["time"].astype(int), unit="ms")
    mcap_df["mcap"] = mcap_df["close"].astype(float)
    mcap_df["volume"] = mcap_df["volume"].astype(float)
    mcap_df = mcap_df[["datetime", "mcap", "volume"]].sort_values("datetime").reset_index(drop=True)

    # Prefer Moralis hourly holders, fall back to GMGN trends
    holder_df = None
    if address:
        holder_df = _load_moralis_holders(address)
    if holder_df is None:
        holder_df = _load_gmgn_holders(trend_data)
    if holder_df is None:
        return None

    mcap_df["datetime"] = mcap_df["datetime"].dt.floor("h")
    mcap_df = mcap_df.groupby("datetime", as_index=False).last()
    holder_df["datetime"] = holder_df["datetime"].dt.floor("h")
    holder_df = holder_df.groupby("datetime", as_index=False).last()

    merged = pd.merge(mcap_df, holder_df, on="datetime", how="inner")
    if merged.empty:
        merged = pd.merge(mcap_df, holder_df, on="datetime", how="outer").sort_values("datetime")
        merged["mcap"] = merged["mcap"].ffill()
        merged["holders"] = merged["holders"].ffill()
        merged["volume"] = merged["volume"].ffill().fillna(0)
        merged = merged.dropna(subset=["mcap", "holders"])

    if merged.empty:
        return None

    merged = merged.sort_values("datetime").reset_index(drop=True)

    # Origin: first crossing of MCAP_THRESHOLD
    above = merged[merged["mcap"] >= MCAP_THRESHOLD]
    if above.empty:
        return None

    start_idx = above.index[0]
    merged = merged.loc[start_idx:].reset_index(drop=True)

    t0 = merged["datetime"].iloc[0]
    merged["hours"] = (merged["datetime"] - t0).dt.total_seconds() / 3600

    # Cap at 3 months
    merged = merged[merged["hours"] <= MAX_HOURS].reset_index(drop=True)
    if len(merged) < 2:
        return None

    return merged


@st.cache_data
def load_all_trajectories():
    """Load trajectories for radar tokens only."""
    meta = load_radar_metadata()
    radar_addrs = set(meta.keys())

    trajectories = {}
    token_meta = {}
    skipped = 0

    for addr in radar_addrs:
        mcap_data, trend_data = load_from_cache(addr)
        if mcap_data is None:
            skipped += 1
            continue
        df = build_trajectory(mcap_data, trend_data, address=addr)
        if df is not None and len(df) >= 2:
            trajectories[addr] = df
            token_meta[addr] = meta[addr]
        else:
            skipped += 1

    return trajectories, token_meta, skipped


def resample_trajectory(df: pd.DataFrame, n_points: int = RESAMPLE_LEN) -> np.ndarray:
    """Resample trajectory to fixed points via arc-length interpolation in 3D log space."""
    log_mcap = np.log1p(df["mcap"].values)
    log_holders = np.log1p(df["holders"].values)
    log_volume = np.log1p(df["volume"].values) if "volume" in df.columns else np.zeros_like(log_mcap)

    points = np.column_stack([log_mcap, log_holders, log_volume])
    diffs = np.diff(points, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    arc = np.concatenate([[0], np.cumsum(seg_lens)])

    total_len = arc[-1]
    if total_len < 1e-9:
        return np.tile(points[0], (n_points, 1))

    target_arc = np.linspace(0, total_len, n_points)
    resampled = np.column_stack([
        np.interp(target_arc, arc, log_mcap),
        np.interp(target_arc, arc, log_holders),
        np.interp(target_arc, arc, log_volume),
    ])
    return resampled


def _z_normalize(s):
    """Per-trajectory z-normalization: removes amplitude, preserves shape."""
    mean = s.mean(axis=0)
    std = s.std(axis=0)
    std[std < 1e-9] = 1
    return (s - mean) / std


def _trim_to_ath(df):
    """Trim trajectory to 50K → ATH only (discard post-ATH decline)."""
    mcap = df["mcap"].values
    ath_idx = np.argmax(mcap)
    return df.iloc[:ath_idx + 1].copy()


# Label for tokens whose ATH is in first 2 hours (too short for shape comparison)
INSTANT_PEAK_LABEL = -2


@st.cache_data
def cluster_tokens_dtw(_trajectories: dict):
    """Cluster tokens by their launch-to-ATH trajectory shape.

    Only the growth phase (50K → ATH) is used — the post-ATH decline is discarded
    as it is noise for identifying runner patterns.

    Pipeline:
    1. Trim each trajectory to ATH
    2. Tokens with ATH in first 2 hours → "Instant Peak" category (too short for shape)
    3. Resample remaining to fixed-length 2D, z-normalize (shape-only)
    4. Blended DTW: 50% shape + 50% derivative (rate-of-change)
    5. UMAP 3D → HDBSCAN (eom) clustering
    """
    addrs = list(_trajectories.keys())

    # Split: valid trajectories (>=3 points to ATH) vs instant-peak
    valid_addrs = []
    instant_addrs = []
    trimmed = {}

    for addr in addrs:
        df = _trim_to_ath(_trajectories[addr])
        trimmed[addr] = df
        if len(df) >= 3:
            valid_addrs.append(addr)
        else:
            instant_addrs.append(addr)

    # Cluster valid trajectories
    series_list = [resample_trajectory(trimmed[a]) for a in valid_addrs]

    series_znorm = [_z_normalize(s) for s in series_list]
    dist_shape = dtw_ndim.distance_matrix_fast(series_znorm)
    dist_shape = np.array(dist_shape)
    dist_shape[np.isinf(dist_shape)] = 0
    dist_shape = dist_shape + dist_shape.T

    series_deriv = [_z_normalize(np.diff(s, axis=0)) for s in series_list]
    dist_deriv = dtw_ndim.distance_matrix_fast(series_deriv)
    dist_deriv = np.array(dist_deriv)
    dist_deriv[np.isinf(dist_deriv)] = 0
    dist_deriv = dist_deriv + dist_deriv.T

    d_shape_max = dist_shape.max()
    d_deriv_max = dist_deriv.max()
    if d_shape_max > 0:
        dist_shape = dist_shape / d_shape_max
    if d_deriv_max > 0:
        dist_deriv = dist_deriv / d_deriv_max
    dist_blend = 0.5 * dist_shape + 0.5 * dist_deriv

    reducer = umap.UMAP(
        n_components=3, n_neighbors=5, min_dist=0.0,
        metric="precomputed", random_state=42,
    )
    embedding = reducer.fit_transform(dist_blend)

    clusterer = HDBSCAN(
        min_cluster_size=5,
        min_samples=2,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(embedding)

    # Build final cluster map
    cluster_map = {}
    for i, addr in enumerate(valid_addrs):
        cluster_map[addr] = int(labels[i])
    for addr in instant_addrs:
        cluster_map[addr] = INSTANT_PEAK_LABEL

    return cluster_map


GRADE_COLORS = {
    "S": "#FFD700",  # gold
    "A": "#22c55e",  # green
    "B": "#3b82f6",  # blue
    "C": "#a855f7",  # purple
    "D": "#f97316",  # orange
}
GRADE_LABELS = {
    "S": "Runner",
    "A": "Strong",
    "B": "Decent",
    "C": "Average",
    "D": "Weak",
}


def score_cluster(trajectories: dict, addrs: list) -> dict:
    """Compute cluster quality metrics and composite score."""
    aths = [trajectories[a]["mcap"].max() for a in addrs]
    hours = [trajectories[a]["hours"].iloc[-1] for a in addrs]
    holders = [trajectories[a]["holders"].max() for a in addrs]
    volumes = [trajectories[a]["volume"].max() for a in addrs if "volume" in trajectories[a].columns]

    med_ath = np.median(aths)
    med_hours = max(np.median(hours), 1)
    med_holders = np.median(holders)
    med_volume = np.median(volumes) if volumes else 1

    # Efficiency: how much ATH per hour of growth
    efficiencies = [a / max(h, 1) for a, h in zip(aths, hours)]
    med_eff = np.median(efficiencies)

    # Composite score: high ATH (x2) + fast + organic holders + volume
    score = (np.log10(max(med_ath, 1)) * 2
             + np.log10(max(med_eff, 1))
             + np.log10(max(med_holders, 1))
             + np.log10(max(med_volume, 1))
             - np.log10(max(med_hours, 1)))

    return {
        "med_ath": med_ath,
        "med_hours": med_hours,
        "med_holders": med_holders,
        "med_volume": med_volume,
        "med_efficiency": med_eff,
        "score": score,
    }


def rank_clusters(trajectories: dict, cluster_addrs: dict) -> dict:
    """Score and rank all clusters. Each cluster gets a unique rank number.

    Returns {cluster_id: {"grade": "S", "rank": 1, "stats": {...}}}
    Grade assignment: #1 = S, #2 = A, #3-4 = B, #5-7 = C, rest = D
    """
    scored = []
    for cid, addrs in cluster_addrs.items():
        if cid < 0:
            continue
        stats = score_cluster(trajectories, addrs)
        scored.append((cid, stats))

    scored.sort(key=lambda x: x[1]["score"], reverse=True)

    result = {}
    for rank, (cid, stats) in enumerate(scored):
        # Each rank gets a unique grade — top tiers are scarce
        if rank == 0:
            grade = "S"
        elif rank == 1:
            grade = "A"
        elif rank <= 3:
            grade = "B"
        elif rank <= 6:
            grade = "C"
        else:
            grade = "D"

        result[cid] = {
            "grade": grade,
            "rank": rank + 1,
            "stats": stats,
        }

    return result


def describe_cluster(trajectories: dict, addrs: list, grade_info: dict = None) -> str:
    """Generate cluster description with grade and key metrics."""
    stats = score_cluster(trajectories, addrs)

    if grade_info:
        grade = grade_info["grade"]
        grade_label = GRADE_LABELS[grade]
        header = f'**Grade {grade} — {grade_label}**'
    else:
        header = ""

    lines = [
        header,
        "",
        f"ATH: ${stats['med_ath']:,.0f} (median)",
        f"Time to ATH: {stats['med_hours']:.0f}h",
        f"Efficiency: ${stats['med_efficiency']:,.0f}/h",
        f"Holders: {stats['med_holders']:,.0f}",
        f"Peak Volume: {stats['med_volume']:,.0f}",
    ]
    return "\n".join(lines)


def fetch_and_build_trajectory(address: str, chain: str = "sol"):
    """Fetch a new token's data from GMGN API and build its trajectory."""
    _, loaded_data = fetch_token_data(chain, address)

    mcap_data = loaded_data.get("token_mcap_candles", {})
    trend_data = loaded_data.get("token_trends", {})

    return build_trajectory(mcap_data, trend_data, address=address)


def classify_token(new_df: pd.DataFrame, trajectories: dict, cluster_addrs: dict):
    """Classify a new token using blended shape+derivative DTW distance.

    Returns (best_cluster_id, distances_dict).
    """
    new_resampled = resample_trajectory(new_df)
    new_znorm = _z_normalize(new_resampled)
    new_deriv = _z_normalize(np.diff(new_resampled, axis=0))

    cluster_distances = {}
    for cid, addrs in cluster_addrs.items():
        if cid < 0:  # skip noise and instant-peak
            continue
        shape_dists = []
        deriv_dists = []
        for addr in addrs:
            member = resample_trajectory(trajectories[addr])
            member_znorm = _z_normalize(member)
            member_deriv = _z_normalize(np.diff(member, axis=0))
            shape_dists.append(dtw_ndim.distance(new_znorm, member_znorm))
            deriv_dists.append(dtw_ndim.distance(new_deriv, member_deriv))
        cluster_distances[cid] = 0.5 * np.mean(shape_dists) + 0.5 * np.mean(deriv_dists)

    best_cid = min(cluster_distances, key=cluster_distances.get)
    return best_cid, cluster_distances


def _compute_roc(df: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """Compute rolling rate-of-change for mcap and holders.

    Returns DataFrame with hours, mcap_roc (% change per window), holders_roc (absolute change per window).
    """
    roc = df[["hours"]].copy()
    # MCap: percentage change over rolling window
    roc["mcap_roc"] = df["mcap"].pct_change(periods=window).fillna(0) * 100
    # Holders: absolute change over rolling window
    roc["holders_roc"] = df["holders"].diff(periods=window).fillna(0)
    # Clip extreme values for better visualization
    roc["mcap_roc"] = roc["mcap_roc"].clip(-100, 500)
    roc["holders_roc"] = roc["holders_roc"].clip(-1000, 5000)
    # Keep raw values for hover
    roc["mcap"] = df["mcap"]
    roc["holders"] = df["holders"]
    return roc


def build_figure(trajectories: dict, token_meta: dict, cluster_map: dict, visible_clusters: set,
                 cluster_grades: dict = None, query_token: tuple = None, view_mode: str = "absolute"):
    fig = go.Figure()
    is_roc = view_mode == "rate_of_change"

    for addr, df in trajectories.items():
        cid = cluster_map.get(addr, -1)
        if cid not in visible_clusters:
            continue

        if cid == INSTANT_PEAK_LABEL:
            color = "#999999"
            width = 1.5
        elif cid == -1:
            color = OUTLIER_COLOR
            width = 2
        elif cluster_grades and cid in cluster_grades:
            grade = cluster_grades[cid]["grade"]
            color = GRADE_COLORS.get(grade, CLUSTER_COLORS[cid % len(CLUSTER_COLORS)])
            width = 3 if grade == "S" else 2.5 if grade == "A" else 2
        else:
            color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
            width = 2

        meta = token_meta.get(addr, {})
        name = meta.get("name", "Unknown")
        symbol = meta.get("symbol", addr[:8])

        if is_roc:
            roc = _compute_roc(df)
            x_vals, y_vals, z_vals = roc["hours"], roc["mcap_roc"], roc["holders_roc"]
            hover_texts = [
                f"<b>{symbol}</b> ({name})<br>"
                f"Grade {cluster_grades[cid]['grade']} {GRADE_LABELS[cluster_grades[cid]['grade']]}<br>"
                f"T+{row['hours']:.0f}h<br>"
                f"MCap RoC: {row['mcap_roc']:+.1f}%/6h<br>"
                f"Holder RoC: {row['holders_roc']:+.0f}/6h<br>"
                f"MCap: ${row['mcap']:,.0f} | Holders: {row['holders']:,.0f}"
                if cluster_grades and cid in cluster_grades else
                f"<b>{symbol}</b> ({name})<br>"
                f"T+{row['hours']:.0f}h<br>"
                f"MCap RoC: {row['mcap_roc']:+.1f}%/6h<br>"
                f"Holder RoC: {row['holders_roc']:+.0f}/6h"
                for _, row in roc.iterrows()
            ]
        else:
            x_vals, y_vals, z_vals = df["hours"], df["mcap"], df["holders"]
            hover_texts = [
                f"<b>{symbol}</b> ({name})<br>"
                f"Grade {cluster_grades[cid]['grade']} {GRADE_LABELS[cluster_grades[cid]['grade']]}<br>"
                f"T+{row['hours']:.0f}h<br>"
                f"MCap: ${row['mcap']:,.0f}<br>"
                f"Holders: {row['holders']:,.0f}"
                if cluster_grades and cid in cluster_grades else
                f"<b>{symbol}</b> ({name})<br>"
                f"T+{row['hours']:.0f}h<br>"
                f"MCap: ${row['mcap']:,.0f}<br>"
                f"Holders: {row['holders']:,.0f}"
                for _, row in df.iterrows()
            ]

        fig.add_trace(go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,
            mode="lines",
            line=dict(color=color, width=width),
            text=hover_texts,
            hoverinfo="text",
            name=symbol,
            showlegend=False,
        ))

    # Overlay the queried token if present
    if query_token is not None:
        q_df, q_symbol, q_cid = query_token
        cluster_color = CLUSTER_COLORS[q_cid % len(CLUSTER_COLORS)] if q_cid >= 0 else OUTLIER_COLOR

        if is_roc:
            q_roc = _compute_roc(q_df)
            qx, qy, qz = q_roc["hours"], q_roc["mcap_roc"], q_roc["holders_roc"]
            q_hover = [
                f"<b>★ {q_symbol}</b> (QUERY)<br>"
                f"→ Cluster {q_cid}<br>"
                f"T+{row['hours']:.0f}h<br>"
                f"MCap RoC: {row['mcap_roc']:+.1f}%/6h<br>"
                f"Holder RoC: {row['holders_roc']:+.0f}/6h"
                for _, row in q_roc.iterrows()
            ]
        else:
            qx, qy, qz = q_df["hours"], q_df["mcap"], q_df["holders"]
            q_hover = [
                f"<b>★ {q_symbol}</b> (QUERY)<br>"
                f"→ Cluster {q_cid}<br>"
                f"T+{row['hours']:.0f}h<br>"
                f"MCap: ${row['mcap']:,.0f}<br>"
                f"Holders: {row['holders']:,.0f}"
                for _, row in q_df.iterrows()
            ]

        fig.add_trace(go.Scatter3d(
            x=qx, y=qy, z=qz,
            mode="lines+markers",
            line=dict(color=QUERY_TOKEN_COLOR, width=6),
            marker=dict(size=3, color=QUERY_TOKEN_COLOR),
            text=q_hover,
            hoverinfo="text",
            name=f"★ {q_symbol} (query)",
            showlegend=True,
        ))
        fig.add_trace(go.Scatter3d(
            x=[qx.iloc[0]], y=[qy.iloc[0]], z=[qz.iloc[0]],
            mode="markers", marker=dict(size=8, color=cluster_color, symbol="diamond"),
            showlegend=False, hovertext=f"{q_symbol} START", hoverinfo="text",
        ))
        fig.add_trace(go.Scatter3d(
            x=[qx.iloc[-1]], y=[qy.iloc[-1]], z=[qz.iloc[-1]],
            mode="markers", marker=dict(size=10, color=cluster_color, symbol="diamond"),
            showlegend=False, hovertext=f"{q_symbol} NOW", hoverinfo="text",
        ))

    n_visible = sum(1 for a in trajectories if cluster_map.get(a, -1) in visible_clusters)

    if is_roc:
        title_text = f"Rate of Change View ({n_visible} visible) — 6h rolling window"
        scene = dict(
            xaxis_title="Time (hours since MCap > $50K)",
            yaxis_title="MCap Change (%/6h)",
            zaxis_title="Holder Change (/6h)",
            xaxis=dict(backgroundcolor="white", gridcolor="rgb(200,200,200)"),
            yaxis=dict(backgroundcolor="white", gridcolor="rgb(200,200,200)"),
            zaxis=dict(backgroundcolor="white", gridcolor="rgb(200,200,200)"),
            bgcolor="white",
        )
    else:
        title_text = f"Radar Token Trajectories ({n_visible} visible) — Origin: first MCap > $50K"
        scene = dict(
            xaxis_title="Time (hours since MCap > $50K)",
            yaxis_title="Market Cap ($)",
            zaxis_title="Holders",
            xaxis=dict(backgroundcolor="white", gridcolor="rgb(200,200,200)"),
            yaxis=dict(backgroundcolor="white", gridcolor="rgb(200,200,200)", type="log"),
            zaxis=dict(backgroundcolor="white", gridcolor="rgb(200,200,200)"),
            bgcolor="white",
        )

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=16, color="black")),
        scene=scene,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        height=800,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="3D Token Trajectories", layout="wide")

with st.spinner("Loading radar token data..."):
    trajectories, token_meta, skipped = load_all_trajectories()

with st.spinner("Computing blended DTW distances & clustering..."):
    cluster_map = cluster_tokens_dtw(trajectories)

# Build cluster info
cluster_ids = sorted(set(cluster_map.values()))
cluster_counts = {cid: sum(1 for v in cluster_map.values() if v == cid) for cid in cluster_ids}
cluster_addrs = {cid: [a for a, c in cluster_map.items() if c == cid] for cid in cluster_ids}

# Rank clusters by quality score
cluster_grades = rank_clusters(trajectories, cluster_addrs)

# Sidebar: cluster toggles (sorted by grade, best first)
with st.sidebar:
    st.header("Token Grades")
    n_real = len([c for c in cluster_ids if c >= 0])
    n_noise = cluster_counts.get(-1, 0)
    n_instant = cluster_counts.get(INSTANT_PEAK_LABEL, 0)
    st.caption(f"{len(trajectories)} tokens · {n_real} grades · {n_noise} outliers · {n_instant} instant-peak")

    # Sort cluster IDs by rank (best grade first)
    sorted_cids = sorted(
        [c for c in cluster_ids if c >= 0],
        key=lambda c: cluster_grades.get(c, {}).get("rank", 999),
    )
    # Append special categories at the end
    if -1 in cluster_ids:
        sorted_cids.append(-1)
    if INSTANT_PEAK_LABEL in cluster_ids:
        sorted_cids.append(INSTANT_PEAK_LABEL)

    visible = set()
    for cid in sorted_cids:
        if cid == INSTANT_PEAK_LABEL:
            label = f"Instant Peak ({cluster_counts[cid]})"
            color_hex = "#999999"
            default = False
        elif cid == -1:
            label = f"Outliers ({cluster_counts[cid]})"
            color_hex = OUTLIER_COLOR
            default = False
        else:
            info = cluster_grades[cid]
            grade = info["grade"]
            grade_label = GRADE_LABELS[grade]
            ath = info["stats"]["med_ath"]
            hrs = info["stats"]["med_hours"]
            color_hex = GRADE_COLORS[grade]
            label = f"{grade} {grade_label} ({cluster_counts[cid]}) — ${ath:,.0f} / {hrs:.0f}h"
            default = True

        if st.checkbox(label, value=default, key=f"c_{cid}"):
            visible.add(cid)

# View mode toggle
st.sidebar.divider()
view_mode = st.sidebar.radio(
    "View Mode",
    ["Absolute (MCap × Holders)", "Rate of Change (RoC)"],
    index=0,
    key="view_mode",
)
view_mode_key = "rate_of_change" if "Rate" in view_mode else "absolute"

# ── Token Query Section ──────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.header("Classify New Token")
query_address = st.sidebar.text_input(
    "Contract address",
    placeholder="Enter Solana token address...",
    key="query_addr",
)
query_chain = st.sidebar.selectbox("Chain", ["sol", "base"], index=0, key="query_chain")

query_token_result = None  # (df, symbol, cluster_id) for overlay on chart

if query_address:
    with st.spinner(f"Fetching data for {query_address[:12]}..."):
        try:
            q_df = fetch_and_build_trajectory(query_address, chain=query_chain)
        except Exception as e:
            q_df = None
            st.sidebar.error(f"Fetch error: {e}")

    if q_df is not None and len(q_df) >= 2:
        with st.spinner("Classifying token..."):
            best_cid, cluster_dists = classify_token(q_df, trajectories, cluster_addrs)

        # Get token symbol from fetched data (or use short address)
        q_meta_path = os.path.join(DATA_DIR, query_address)
        q_symbol = query_address[:8]
        # Try to extract symbol from GMGN price info if cached
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

        query_token_result = (q_df, q_symbol, best_cid)

        # Show classification result with grade
        st.sidebar.divider()
        if best_cid in cluster_grades:
            info = cluster_grades[best_cid]
            grade = info["grade"]
            grade_label = GRADE_LABELS[grade]
            grade_color = GRADE_COLORS[grade]
            st.sidebar.markdown(
                f'### Result: <span style="color: {grade_color};">Grade {grade} — {grade_label}</span>',
                unsafe_allow_html=True,
            )
        else:
            st.sidebar.markdown(f'### Result: Cluster {best_cid}')

        # Show distances to all clusters (sorted by grade)
        st.sidebar.caption("Distance to each grade:")
        ranked_cids = sorted(cluster_dists.keys(), key=lambda c: cluster_grades.get(c, {}).get("rank", 999))
        for cid in ranked_cids:
            dist = cluster_dists[cid]
            marker = " **← match**" if cid == best_cid else ""
            if cid in cluster_grades:
                g = cluster_grades[cid]["grade"]
                gc = GRADE_COLORS[g]
                gl = GRADE_LABELS[g]
                st.sidebar.markdown(
                    f'<span style="color: {gc};">{g} {gl}</span>: {dist:.2f}{marker}',
                    unsafe_allow_html=True,
                )
            else:
                st.sidebar.markdown(f'C{cid}: {dist:.2f}{marker}')

        # Token summary stats
        st.sidebar.caption(f"Data: {len(q_df)} hours")
        st.sidebar.caption(f"Current MCap: ${q_df['mcap'].iloc[-1]:,.0f}")
        st.sidebar.caption(f"Peak MCap: ${q_df['mcap'].max():,.0f}")
        st.sidebar.caption(f"Holders: {q_df['holders'].iloc[-1]:,.0f}")
    elif query_address:
        st.sidebar.warning("Could not build trajectory (no data or mcap < $50K)")

# Main area: 3D chart + cluster descriptions
fig = build_figure(trajectories, token_meta, cluster_map, visible,
                   cluster_grades=cluster_grades, query_token=query_token_result,
                   view_mode=view_mode_key)
st.plotly_chart(fig, width="stretch")

# Cluster description panel
st.divider()
st.subheader("Token Grade Analysis")

# Sort visible clusters by grade rank (best first)
visible_ranked = sorted(
    [c for c in visible if c >= 0],
    key=lambda c: cluster_grades.get(c, {}).get("rank", 999),
)
desc_cols = st.columns(min(len(visible_ranked), 3) or 1)
col_idx = 0
for cid in visible_ranked:
    addrs = cluster_addrs[cid]
    grade_info = cluster_grades.get(cid)
    desc = describe_cluster(trajectories, addrs, grade_info)

    grade = grade_info["grade"] if grade_info else "?"
    grade_color = GRADE_COLORS.get(grade, "#888888")
    grade_label = GRADE_LABELS.get(grade, "")

    with desc_cols[col_idx % len(desc_cols)]:
        st.markdown(
            f'<div style="border-left: 4px solid {grade_color}; padding-left: 12px; margin-bottom: 16px;">'
            f'<h4 style="color: {grade_color}; margin: 0;">{grade} {grade_label} ({len(addrs)} tokens)</h4>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(desc)
        with st.expander("Tokens in this grade"):
            for a in addrs:
                m = token_meta.get(a, {})
                ath = trajectories[a]["mcap"].max()
                st.text(f"{m.get('symbol', '?'):>10s}  ATH ${ath:>12,.0f}")
    col_idx += 1
