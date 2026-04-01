"""t-SNE Point Cloud — Every hourly data point across all radar tokens."""

import csv
import json
import os
import glob

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from hdbscan import HDBSCAN


DATA_DIR = "data"
RADAR_CSV = os.path.join(DATA_DIR, "radar_tokens.csv")
MCAP_THRESHOLD = 50_000
MAX_HOURS = 90 * 24
FUTURE_HOURS = 4

CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


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


def build_trajectory(mcap_data: dict, trend_data: dict):
    if not mcap_data or not trend_data:
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
    mcap_df = mcap_df[["datetime", "mcap"]].sort_values("datetime").reset_index(drop=True)

    trend_section = trend_data.get("data")
    if not trend_section:
        return None
    trends = trend_section.get("trends", {})
    holder_series = trends.get("holder_count", [])
    if not holder_series:
        return None

    holder_df = pd.DataFrame(holder_series)
    holder_df["datetime"] = pd.to_datetime(holder_df["timestamp"].astype(int), unit="s")
    holder_df["holders"] = holder_df["value"].astype(float)
    holder_df = holder_df[["datetime", "holders"]].sort_values("datetime")
    holder_df = holder_df.set_index("datetime").resample("1h").last().dropna().reset_index()

    mcap_df["datetime"] = mcap_df["datetime"].dt.floor("h")
    mcap_df = mcap_df.groupby("datetime", as_index=False).last()
    holder_df["datetime"] = holder_df["datetime"].dt.floor("h")
    holder_df = holder_df.groupby("datetime", as_index=False).last()

    merged = pd.merge(mcap_df, holder_df, on="datetime", how="inner")
    if merged.empty:
        merged = pd.merge(mcap_df, holder_df, on="datetime", how="outer").sort_values("datetime")
        merged["mcap"] = merged["mcap"].ffill()
        merged["holders"] = merged["holders"].ffill()
        merged = merged.dropna()

    if merged.empty:
        return None

    merged = merged.sort_values("datetime").reset_index(drop=True)

    above = merged[merged["mcap"] >= MCAP_THRESHOLD]
    if above.empty:
        return None

    start_idx = above.index[0]
    merged = merged.loc[start_idx:].reset_index(drop=True)

    t0 = merged["datetime"].iloc[0]
    merged["hours"] = (merged["datetime"] - t0).dt.total_seconds() / 3600
    merged = merged[merged["hours"] <= MAX_HOURS].reset_index(drop=True)

    if len(merged) < FUTURE_HOURS + 1:
        return None

    return merged


@st.cache_data
def build_point_cloud():
    """Build the 4D feature matrix: every hourly point across all radar tokens."""
    meta = load_radar_metadata()
    radar_addrs = set(meta.keys())

    rows = []
    for addr in radar_addrs:
        mcap_data, trend_data = load_from_cache(addr)
        if mcap_data is None:
            continue
        df = build_trajectory(mcap_data, trend_data)
        if df is None:
            continue

        mcap_arr = df["mcap"].values
        hours_arr = df["hours"].values
        holders_arr = df["holders"].values
        n = len(df)

        sym = meta.get(addr, {}).get("symbol", addr[:8])
        name = meta.get(addr, {}).get("name", "")

        for i in range(n - FUTURE_HOURS):
            current_mcap = mcap_arr[i]
            future_mcap = mcap_arr[i + FUTURE_HOURS]
            if current_mcap <= 0:
                continue
            future_return = (future_mcap - current_mcap) / current_mcap * 100

            # Clip extreme returns to avoid dominating the embedding
            future_return_clipped = np.clip(future_return, -100, 500)

            rows.append({
                "address": addr,
                "symbol": sym,
                "name": name,
                "log_mcap": np.log1p(current_mcap),
                "hours": hours_arr[i],
                "log_holders": np.log1p(holders_arr[i]),
                "future_return_4h": future_return_clipped,
                # Raw values for tooltip
                "mcap": current_mcap,
                "holders": holders_arr[i],
                "future_return_4h_raw": future_return,
            })

    return pd.DataFrame(rows)


@st.cache_data
def compute_tsne(points_df: pd.DataFrame, perplexity: int = 30):
    """PCA → t-SNE 3D on the 4 feature dimensions."""
    feature_cols = ["log_mcap", "hours", "log_holders", "future_return_4h"]
    X = points_df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-9] = 1
    X = (X - mean) / std

    # PCA to 4D (keep all variance, just decorrelate)
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X)

    # t-SNE to 3D
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42,
                max_iter=1000, init="pca", learning_rate="auto")
    X_3d = tsne.fit_transform(X_pca)

    return X_3d, pca.explained_variance_ratio_


@st.cache_data
def cluster_points(X_3d: np.ndarray):
    """HDBSCAN on the t-SNE 3D coordinates."""
    clusterer = HDBSCAN(min_cluster_size=150, min_samples=5, cluster_selection_method="leaf")
    labels = clusterer.fit_predict(X_3d)
    return labels


def describe_point_cluster(df_cluster: pd.DataFrame) -> str:
    """Generate description for a cluster of data points."""
    med_mcap = np.median(df_cluster["mcap"])
    med_holders = np.median(df_cluster["holders"])
    med_hours = np.median(df_cluster["hours"])
    med_return = np.median(df_cluster["future_return_4h"])
    mean_return = df_cluster["future_return_4h"].mean()
    pct_positive = (df_cluster["future_return_4h"] > 0).mean() * 100
    n_tokens = df_cluster["address"].nunique()

    if med_return > 10:
        momentum = "Strong upward momentum"
    elif med_return > 2:
        momentum = "Mild upward momentum"
    elif med_return > -2:
        momentum = "Sideways / neutral"
    elif med_return > -10:
        momentum = "Mild downward pressure"
    else:
        momentum = "Strong sell-off"

    lines = [
        f"**{momentum}**",
        "",
        f"4h Return: {med_return:+.1f}% median, {mean_return:+.1f}% mean",
        f"Win Rate (4h): {pct_positive:.0f}%",
        f"MCap: ${med_mcap:,.0f} (median)",
        f"Holders: {med_holders:,.0f}",
        f"Age: T+{med_hours:.0f}h",
        f"Tokens: {n_tokens} unique",
        f"Points: {len(df_cluster):,}",
    ]
    return "\n".join(lines)


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="t-SNE Point Cloud", layout="wide")

with st.spinner("Building point cloud from all radar tokens..."):
    points_df = build_point_cloud()

st.sidebar.header("t-SNE Settings")
perplexity = st.sidebar.slider("Perplexity", 5, 100, 30)

with st.spinner(f"Computing PCA → t-SNE 3D ({len(points_df):,} points)..."):
    X_3d, pca_var = compute_tsne(points_df, perplexity=perplexity)

with st.spinner("Clustering..."):
    labels = cluster_points(X_3d)

points_df["tsne_x"] = X_3d[:, 0]
points_df["tsne_y"] = X_3d[:, 1]
points_df["tsne_z"] = X_3d[:, 2]
points_df["cluster"] = labels

cluster_ids = sorted(set(labels))
cluster_counts = {cid: (labels == cid).sum() for cid in cluster_ids}

# Sidebar: cluster toggles
st.sidebar.divider()
st.sidebar.header("Clusters")
n_real = len([c for c in cluster_ids if c != -1])
st.sidebar.caption(f"{len(points_df):,} points · {n_real} clusters")

visible = set()
for cid in cluster_ids:
    if cid == -1:
        label = f"Noise ({cluster_counts[cid]:,})"
    else:
        label = f"Cluster {cid} ({cluster_counts[cid]:,})"
    default = cid != -1
    if st.sidebar.checkbox(label, value=default, key=f"pc_{cid}"):
        visible.add(cid)

# Color by: cluster or future return
color_mode = st.sidebar.radio("Color by", ["Cluster", "4h Return %"], index=0)

# Build figure
fig = go.Figure()

if color_mode == "Cluster":
    for cid in cluster_ids:
        if cid not in visible:
            continue
        mask = points_df["cluster"] == cid
        sub = points_df[mask]
        if cid == -1:
            color = "rgba(180,180,180,0.15)"
            size = 1.5
        else:
            color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
            size = 2.5

        fig.add_trace(go.Scatter3d(
            x=sub["tsne_x"], y=sub["tsne_y"], z=sub["tsne_z"],
            mode="markers",
            marker=dict(size=size, color=color),
            text=[
                f"<b>{row['symbol']}</b> ({row['name']})<br>"
                f"Cluster {row['cluster']}<br>"
                f"MCap: ${row['mcap']:,.0f}<br>"
                f"Holders: {row['holders']:,.0f}<br>"
                f"Age: T+{row['hours']:.0f}h<br>"
                f"4h Return: {row.get('future_return_4h_raw', row['future_return_4h']):+.1f}%"
                for _, row in sub.iterrows()
            ],
            hoverinfo="text",
            name=f"C{cid}" if cid != -1 else "Noise",
            showlegend=False,
        ))
else:
    mask = points_df["cluster"].isin(visible)
    sub = points_df[mask]
    # Clip returns for color scale
    ret_clipped = sub["future_return_4h"].clip(-50, 50)
    fig.add_trace(go.Scatter3d(
        x=sub["tsne_x"], y=sub["tsne_y"], z=sub["tsne_z"],
        mode="markers",
        marker=dict(
            size=2.5,
            color=ret_clipped,
            colorscale="RdYlGn",
            cmin=-50, cmid=0, cmax=50,
            colorbar=dict(title="4h Return %", thickness=15),
        ),
        text=[
            f"<b>{row['symbol']}</b> ({row['name']})<br>"
            f"Cluster {row['cluster']}<br>"
            f"MCap: ${row['mcap']:,.0f}<br>"
            f"Holders: {row['holders']:,.0f}<br>"
            f"Age: T+{row['hours']:.0f}h<br>"
            f"4h Return: {row.get('future_return_4h_raw', row['future_return_4h']):+.1f}%"
            for _, row in sub.iterrows()
        ],
        hoverinfo="text",
        showlegend=False,
    ))

n_visible = points_df["cluster"].isin(visible).sum()
fig.update_layout(
    title=dict(
        text=f"t-SNE Point Cloud ({n_visible:,} points visible)",
        font=dict(size=16, color="black"),
    ),
    scene=dict(
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        zaxis_title="t-SNE 3",
        xaxis=dict(backgroundcolor="white", gridcolor="rgb(220,220,220)"),
        yaxis=dict(backgroundcolor="white", gridcolor="rgb(220,220,220)"),
        zaxis=dict(backgroundcolor="white", gridcolor="rgb(220,220,220)"),
        bgcolor="white",
    ),
    paper_bgcolor="white",
    font=dict(color="black"),
    height=800,
    margin=dict(l=0, r=0, t=40, b=0),
)

st.plotly_chart(fig, width="stretch")

# PCA info
st.caption(f"PCA variance explained: {pca_var[0]:.1%}, {pca_var[1]:.1%}, {pca_var[2]:.1%}, {pca_var[3]:.1%}")

# Cluster descriptions
st.divider()
st.subheader("Cluster Descriptions")

desc_cols = st.columns(min(len([c for c in visible if c != -1]), 3) or 1)
col_idx = 0
for cid in sorted(visible):
    if cid == -1:
        continue
    color_hex = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
    sub = points_df[points_df["cluster"] == cid]
    desc = describe_point_cluster(sub)

    with desc_cols[col_idx % len(desc_cols)]:
        st.markdown(
            f'<div style="border-left: 4px solid {color_hex}; padding-left: 12px; margin-bottom: 16px;">'
            f'<h4 style="color: {color_hex}; margin: 0;">Cluster {cid} ({len(sub):,} points)</h4>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(desc)
    col_idx += 1
