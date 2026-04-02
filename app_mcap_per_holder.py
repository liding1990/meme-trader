"""2D MCap/Holder Trajectory — First week from $50K, clustered."""

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


DATA_DIR = "data"
RADAR_CSV = os.path.join(DATA_DIR, "radar_tokens.csv")
MCAP_THRESHOLD = 50_000
MAX_HOURS = 7 * 24  # 1 week
RESAMPLE_LEN = 50

CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]
OUTLIER_COLOR = "#cccccc"
QUERY_TOKEN_COLOR = "#000000"

GRADE_COLORS = {
    "S": "#FFD700",
    "A": "#22c55e",
    "B": "#3b82f6",
    "C": "#a855f7",
    "D": "#f97316",
}
GRADE_LABELS = {
    "S": "Runner",
    "A": "Strong",
    "B": "Decent",
    "C": "Average",
    "D": "Weak",
}


def load_radar_metadata() -> dict:
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
    """Build hourly trajectory: first week from MCap >= $50K.

    Computes mcap_per_holder = mcap / holders for each hour.
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

    # Filter: holders must be > 0 for division
    merged = merged[merged["holders"] > 0].reset_index(drop=True)

    above = merged[merged["mcap"] >= MCAP_THRESHOLD]
    if above.empty:
        return None

    start_idx = above.index[0]
    merged = merged.loc[start_idx:].reset_index(drop=True)

    t0 = merged["datetime"].iloc[0]
    merged["hours"] = (merged["datetime"] - t0).dt.total_seconds() / 3600

    # Cap at 1 week
    merged = merged[merged["hours"] <= MAX_HOURS].reset_index(drop=True)
    if len(merged) < 2:
        return None

    # Core metric
    merged["mcap_per_holder"] = merged["mcap"] / merged["holders"]

    return merged


@st.cache_data
def load_all_trajectories():
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


# ── Clustering on mcap_per_holder curves ────────────────────────────────────


def resample_curve(df: pd.DataFrame, n_points: int = RESAMPLE_LEN) -> np.ndarray:
    """Resample mcap_per_holder to fixed-length via arc-length interpolation in log space."""
    log_mph = np.log1p(df["mcap_per_holder"].values)
    hours = df["hours"].values

    points = np.column_stack([hours, log_mph])
    diffs = np.diff(points, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    arc = np.concatenate([[0], np.cumsum(seg_lens)])

    total_len = arc[-1]
    if total_len < 1e-9:
        return np.tile(log_mph[0], (n_points, 1))

    target_arc = np.linspace(0, total_len, n_points)
    resampled = np.interp(target_arc, arc, log_mph).reshape(-1, 1)
    return resampled


def _z_normalize(s):
    mean = s.mean(axis=0)
    std = s.std(axis=0)
    std[std < 1e-9] = 1
    return (s - mean) / std


INSTANT_PEAK_LABEL = -2


@st.cache_data
def cluster_curves(_trajectories: dict):
    """Cluster tokens by their first-week mcap_per_holder curve shape.

    Pipeline:
    1. Resample each curve to fixed length (log space)
    2. Z-normalize (shape only)
    3. Blended DTW: 50% shape + 50% derivative
    4. UMAP 2D → HDBSCAN
    """
    addrs = list(_trajectories.keys())

    # Filter: need at least 3 data points for meaningful shape
    valid_addrs = []
    short_addrs = []
    for addr in addrs:
        if len(_trajectories[addr]) >= 3:
            valid_addrs.append(addr)
        else:
            short_addrs.append(addr)

    series_list = [resample_curve(_trajectories[a]) for a in valid_addrs]

    # Shape distance
    series_znorm = [_z_normalize(s) for s in series_list]
    dist_shape = np.array(dtw_ndim.distance_matrix_fast(series_znorm))
    dist_shape[np.isinf(dist_shape)] = 0
    dist_shape = dist_shape + dist_shape.T

    # Derivative distance
    series_deriv = [_z_normalize(np.diff(s, axis=0)) for s in series_list]
    dist_deriv = np.array(dtw_ndim.distance_matrix_fast(series_deriv))
    dist_deriv[np.isinf(dist_deriv)] = 0
    dist_deriv = dist_deriv + dist_deriv.T

    # Normalize and blend
    d_shape_max = dist_shape.max()
    d_deriv_max = dist_deriv.max()
    if d_shape_max > 0:
        dist_shape /= d_shape_max
    if d_deriv_max > 0:
        dist_deriv /= d_deriv_max
    dist_blend = 0.5 * dist_shape + 0.5 * dist_deriv

    # UMAP → HDBSCAN
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

    cluster_map = {}
    for i, addr in enumerate(valid_addrs):
        cluster_map[addr] = int(labels[i])
    for addr in short_addrs:
        cluster_map[addr] = INSTANT_PEAK_LABEL

    return cluster_map


# ── Scoring & Grading ───────────────────────────────────────────────────────


def score_cluster(trajectories: dict, addrs: list) -> dict:
    """Score cluster by mcap_per_holder peak and trajectory quality."""
    mph_peaks = [trajectories[a]["mcap_per_holder"].max() for a in addrs]
    aths = [trajectories[a]["mcap"].max() for a in addrs]
    hours = [trajectories[a]["hours"].iloc[-1] for a in addrs]
    holders = [trajectories[a]["holders"].max() for a in addrs]

    med_mph = np.median(mph_peaks)
    med_ath = np.median(aths)
    med_hours = max(np.median(hours), 1)
    med_holders = np.median(holders)

    score = (np.log10(max(med_mph, 1)) * 2
             + np.log10(max(med_ath, 1))
             + np.log10(max(med_holders, 1))
             - np.log10(max(med_hours, 1)))

    return {
        "med_mph": med_mph,
        "med_ath": med_ath,
        "med_hours": med_hours,
        "med_holders": med_holders,
        "score": score,
    }


def rank_clusters(trajectories: dict, cluster_addrs: dict) -> dict:
    scored = []
    for cid, addrs in cluster_addrs.items():
        if cid < 0:
            continue
        stats = score_cluster(trajectories, addrs)
        scored.append((cid, stats))

    scored.sort(key=lambda x: x[1]["score"], reverse=True)

    result = {}
    for rank, (cid, stats) in enumerate(scored):
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
        result[cid] = {"grade": grade, "rank": rank + 1, "stats": stats}

    return result


def describe_cluster(trajectories: dict, addrs: list, grade_info: dict = None) -> str:
    stats = score_cluster(trajectories, addrs)
    if grade_info:
        grade = grade_info["grade"]
        header = f'**Grade {grade} — {GRADE_LABELS[grade]}**'
    else:
        header = ""
    lines = [
        header, "",
        f"MCap/Holder peak: ${stats['med_mph']:,.0f}",
        f"ATH: ${stats['med_ath']:,.0f}",
        f"Data span: {stats['med_hours']:.0f}h",
        f"Max Holders: {stats['med_holders']:,.0f}",
    ]
    return "\n".join(lines)


# ── Figure ──────────────────────────────────────────────────────────────────


def build_figure(trajectories, token_meta, cluster_map, visible_clusters,
                 cluster_grades=None, query_token=None):
    fig = go.Figure()

    for addr, df in trajectories.items():
        cid = cluster_map.get(addr, -1)
        if cid not in visible_clusters:
            continue

        if cid == INSTANT_PEAK_LABEL:
            color = "#999999"
            width = 1
        elif cid == -1:
            color = OUTLIER_COLOR
            width = 1
        elif cluster_grades and cid in cluster_grades:
            grade = cluster_grades[cid]["grade"]
            color = GRADE_COLORS.get(grade, CLUSTER_COLORS[cid % len(CLUSTER_COLORS)])
            width = 3 if grade == "S" else 2.5 if grade == "A" else 2
        else:
            color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
            width = 2

        meta = token_meta.get(addr, {})
        symbol = meta.get("symbol", addr[:8])
        name = meta.get("name", "")

        fig.add_trace(go.Scatter(
            x=df["hours"],
            y=df["mcap_per_holder"],
            mode="lines",
            line=dict(color=color, width=width),
            text=[
                f"<b>{symbol}</b> ({name})<br>"
                + (f"Grade {cluster_grades[cid]['grade']} {GRADE_LABELS[cluster_grades[cid]['grade']]}<br>"
                   if cluster_grades and cid in cluster_grades else "")
                + f"T+{row['hours']:.0f}h<br>"
                  f"MCap/Holder: ${row['mcap_per_holder']:,.0f}<br>"
                  f"MCap: ${row['mcap']:,.0f} | Holders: {row['holders']:,.0f}"
                for _, row in df.iterrows()
            ],
            hoverinfo="text",
            name=symbol,
            showlegend=False,
        ))

    if query_token is not None:
        q_df, q_symbol, q_cid = query_token
        fig.add_trace(go.Scatter(
            x=q_df["hours"],
            y=q_df["mcap_per_holder"],
            mode="lines+markers",
            line=dict(color=QUERY_TOKEN_COLOR, width=4),
            marker=dict(size=4, color=QUERY_TOKEN_COLOR),
            text=[
                f"<b>★ {q_symbol}</b> (QUERY)<br>"
                f"→ Cluster {q_cid}<br>"
                f"T+{row['hours']:.0f}h<br>"
                f"MCap/Holder: ${row['mcap_per_holder']:,.0f}<br>"
                f"MCap: ${row['mcap']:,.0f} | Holders: {row['holders']:,.0f}"
                for _, row in q_df.iterrows()
            ],
            hoverinfo="text",
            name=f"★ {q_symbol} (query)",
            showlegend=True,
        ))

    n_visible = sum(1 for a in trajectories if cluster_map.get(a, -1) in visible_clusters)
    fig.update_layout(
        title=dict(
            text=f"MCap / Holder — First Week ({n_visible} tokens visible)",
            font=dict(size=16, color="black"),
        ),
        xaxis_title="Hours since MCap > $50K",
        yaxis_title="MCap per Holder ($)",
        yaxis_type="log",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        height=700,
        margin=dict(l=60, r=20, t=50, b=50),
        hovermode="closest",
    )

    return fig


# ── Streamlit UI ────────────────────────────────────────────────────────────

st.set_page_config(page_title="MCap/Holder Curves", layout="wide")

with st.spinner("Loading radar token data..."):
    trajectories, token_meta, skipped = load_all_trajectories()

with st.spinner("Clustering mcap/holder curves..."):
    cluster_map = cluster_curves(trajectories)

cluster_ids = sorted(set(cluster_map.values()))
cluster_counts = {cid: sum(1 for v in cluster_map.values() if v == cid) for cid in cluster_ids}
cluster_addrs = {cid: [a for a, c in cluster_map.items() if c == cid] for cid in cluster_ids}

cluster_grades = rank_clusters(trajectories, cluster_addrs)

with st.sidebar:
    st.header("Token Grades")
    n_real = len([c for c in cluster_ids if c >= 0])
    n_noise = cluster_counts.get(-1, 0)
    n_instant = cluster_counts.get(INSTANT_PEAK_LABEL, 0)
    st.caption(f"{len(trajectories)} tokens · {n_real} grades · {n_noise} outliers · {n_instant} short")

    sorted_cids = sorted(
        [c for c in cluster_ids if c >= 0],
        key=lambda c: cluster_grades.get(c, {}).get("rank", 999),
    )
    if -1 in cluster_ids:
        sorted_cids.append(-1)
    if INSTANT_PEAK_LABEL in cluster_ids:
        sorted_cids.append(INSTANT_PEAK_LABEL)

    visible = set()
    for cid in sorted_cids:
        if cid == INSTANT_PEAK_LABEL:
            label = f"Short ({cluster_counts[cid]})"
            default = False
        elif cid == -1:
            label = f"Outliers ({cluster_counts[cid]})"
            default = False
        else:
            info = cluster_grades[cid]
            grade = info["grade"]
            mph = info["stats"]["med_mph"]
            ath = info["stats"]["med_ath"]
            label = f"{grade} {GRADE_LABELS[grade]} ({cluster_counts[cid]}) — ${mph:,.0f}/holder, ${ath:,.0f} ATH"
            default = True

        if st.checkbox(label, value=default, key=f"c_{cid}"):
            visible.add(cid)

# Query
st.sidebar.divider()
st.sidebar.header("Classify New Token")
query_address = st.sidebar.text_input("Contract address", placeholder="Enter Solana token address...", key="query_addr")
query_chain = st.sidebar.selectbox("Chain", ["sol", "base"], index=0, key="query_chain")

query_token_result = None
if query_address:
    from gmgn_api import fetch_token_data
    with st.spinner(f"Fetching {query_address[:12]}..."):
        try:
            _, loaded_data = fetch_token_data(query_chain, query_address)
            mcap_data = loaded_data.get("token_mcap_candles", {})
            trend_data = loaded_data.get("token_trends", {})
            q_df = build_trajectory(mcap_data, trend_data, address=query_address)
        except Exception as e:
            q_df = None
            st.sidebar.error(f"Fetch error: {e}")

    if q_df is not None and len(q_df) >= 2:
        # Simple nearest-cluster by DTW
        new_series = resample_curve(q_df)
        new_znorm = _z_normalize(new_series)
        best_cid, best_dist = -1, float("inf")
        for cid, addrs in cluster_addrs.items():
            if cid < 0:
                continue
            dists = []
            for addr in addrs:
                member = resample_curve(trajectories[addr])
                member_znorm = _z_normalize(member)
                dists.append(dtw_ndim.distance(new_znorm, member_znorm))
            avg_dist = np.mean(dists)
            if avg_dist < best_dist:
                best_dist = avg_dist
                best_cid = cid

        q_symbol = query_address[:8]
        query_token_result = (q_df, q_symbol, best_cid)

        st.sidebar.divider()
        if best_cid in cluster_grades:
            info = cluster_grades[best_cid]
            grade = info["grade"]
            gc = GRADE_COLORS[grade]
            st.sidebar.markdown(
                f'### Result: <span style="color: {gc};">Grade {grade} — {GRADE_LABELS[grade]}</span>',
                unsafe_allow_html=True,
            )
        st.sidebar.caption(f"Data: {len(q_df)} hours")
        st.sidebar.caption(f"Peak MCap/Holder: ${q_df['mcap_per_holder'].max():,.0f}")
    elif query_address:
        st.sidebar.warning("Could not build trajectory")

# Chart
fig = build_figure(trajectories, token_meta, cluster_map, visible,
                   cluster_grades=cluster_grades, query_token=query_token_result)
st.plotly_chart(fig, use_container_width=True)

# Cluster descriptions
st.divider()
st.subheader("Grade Analysis")

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
    grade_color = GRADE_COLORS.get(grade, "#888")

    with desc_cols[col_idx % len(desc_cols)]:
        st.markdown(
            f'<div style="border-left: 4px solid {grade_color}; padding-left: 12px; margin-bottom: 16px;">'
            f'<h4 style="color: {grade_color}; margin: 0;">{grade} {GRADE_LABELS.get(grade, "")} ({len(addrs)} tokens)</h4>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(desc)
        with st.expander("Tokens"):
            for a in sorted(addrs, key=lambda x: trajectories[x]["mcap_per_holder"].max(), reverse=True):
                m = token_meta.get(a, {})
                mph_peak = trajectories[a]["mcap_per_holder"].max()
                ath = trajectories[a]["mcap"].max()
                st.text(f"{m.get('symbol', '?'):>10s}  $/holder ${mph_peak:>10,.0f}  ATH ${ath:>12,.0f}")
    col_idx += 1
