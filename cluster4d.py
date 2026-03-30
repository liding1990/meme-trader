"""Cluster tokens in the 4D state space and visualize per-cluster patterns.

Clustering is done on TOKEN-level (not point-level): each token's full
trajectory in 4D space is represented by its path signature, then clustered.

After clustering, for each cluster we plot 3 charts (MCap, Holders, Top10%)
with all member tokens overlaid, so you can see the shared pattern.

Usage:
    python cluster4d.py [--min-cluster 5]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import hdbscan
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from space4d import extract_trajectory, load_token_list, MCAP_THRESHOLD, _log_transform

DATA_DIR = "data"
PLOTS_DIR = "plots"

CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#aec7e8",
    "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
]
NOISE_COLOR = "rgba(180, 180, 180, 0.3)"


def build_token_features():
    """Build a feature vector for each token from its 4D trajectory.

    Uses path signature (log-transformed 4D + time augmentation, depth 3)
    to capture the full trajectory shape in a fixed-length vector.
    """
    import iisignature

    tokens = load_token_list()
    features = []
    token_info = []
    raw_trajs = {}

    for token in tokens:
        address = token["address"]
        traj = extract_trajectory(address)
        if traj is None or len(traj) < 3:
            continue

        raw_trajs[address] = traj

        # Build 4D path: [time, log(mcap), log(holders), top10%]
        T = len(traj)
        time_col = np.arange(T, dtype=float).reshape(-1, 1)
        log_mcap = np.log1p(traj[:, 0]).reshape(-1, 1)
        log_holders = np.log1p(traj[:, 1]).reshape(-1, 1)
        top10 = traj[:, 2].reshape(-1, 1)

        path = np.hstack([time_col, log_mcap, log_holders, top10])

        # Compute path signature
        sig = iisignature.sig(path, 3)
        features.append(sig)

        ath_idx = np.argmax(traj[:, 0])
        token_info.append({
            "address": address,
            "symbol": token["symbol"],
            "name": token["name"],
            "total_hours": T,
            "ath_mcap": traj[ath_idx, 0],
            "ath_hour": ath_idx,
            "holders_at_ath": traj[ath_idx, 1],
            "top10_at_ath": traj[ath_idx, 2],
        })

    return np.array(features), token_info, raw_trajs


def cluster_tokens(features, min_cluster_size=5):
    """UMAP dimensionality reduction → HDBSCAN clustering.

    84-dim signature space is too sparse for HDBSCAN density estimation.
    UMAP to 10D first concentrates the density structure.
    """
    import umap
    from sklearn.preprocessing import StandardScaler

    scaled = StandardScaler().fit_transform(features)

    # UMAP to 10D for better density estimation
    reducer = umap.UMAP(
        n_components=10,
        n_neighbors=15,
        min_dist=0.0,
        metric="euclidean",
        random_state=42,
    )
    embedding = reducer.fit_transform(scaled)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=2,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embedding)
    return labels


def plot_cluster_selector(token_info, labels):
    """Plot overview: one scatter showing all tokens colored by cluster, with summary stats."""
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    df = pd.DataFrame(token_info)
    df["cluster"] = labels

    fig = go.Figure()

    for label in sorted(set(labels)):
        mask = df["cluster"] == label
        subset = df[mask]

        if label == -1:
            color = NOISE_COLOR
            name = f"Noise ({len(subset)})"
        else:
            color = CLUSTER_COLORS[label % len(CLUSTER_COLORS)]
            name = f"C{label} ({len(subset)}) — med ATH ${subset['ath_mcap'].median():,.0f}"

        fig.add_trace(go.Scatter(
            x=subset["ath_hour"],
            y=subset["ath_mcap"],
            mode="markers",
            marker=dict(color=color, size=10, line=dict(width=1, color="white")),
            text=subset["symbol"],
            hovertext=[
                f"<b>{r['symbol']}</b><br>"
                f"ATH: ${r['ath_mcap']:,.0f} @ hour {r['ath_hour']}<br>"
                f"Life: {r['total_hours']}h<br>"
                f"Holders@ATH: {r['holders_at_ath']:,.0f}<br>"
                f"Top10@ATH: {r['top10_at_ath']:.1f}%"
                for _, r in subset.iterrows()
            ],
            hoverinfo="text",
            name=name,
        ))

    fig.update_layout(
        title="Token Clusters Overview (ATH vs Time to ATH)",
        xaxis_title="Hours to ATH (from 50K)",
        yaxis_title="ATH Market Cap ($)",
        yaxis_type="log",
        width=1200, height=700,
        hovermode="closest",
    )

    path = os.path.join(PLOTS_DIR, "cluster_overview.html")
    fig.write_html(path)
    print(f"Cluster overview: {path}")


def plot_cluster_detail(cluster_id, token_info, labels, raw_trajs):
    """Plot 3 charts (MCap, Holders, Top10%) for a single cluster with all members overlaid."""
    df = pd.DataFrame(token_info)
    df["cluster"] = labels
    cluster_df = df[df["cluster"] == cluster_id]

    if len(cluster_df) == 0:
        return

    color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
    dim_labels = ["Market Cap ($)", "Holders", "Top10 Holder %"]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=dim_labels,
        vertical_spacing=0.08,
    )

    for _, row in cluster_df.iterrows():
        traj = raw_trajs.get(row["address"])
        if traj is None:
            continue

        hours = list(range(len(traj)))
        symbol = row["symbol"]

        for dim in range(3):
            fig.add_trace(go.Scatter(
                x=hours, y=traj[:, dim],
                mode="lines",
                line=dict(color=color, width=1.8),
                opacity=0.6,
                name=symbol if dim == 0 else None,
                legendgroup=symbol,
                showlegend=(dim == 0),
                hovertext=[f"{symbol} h{h}" for h in hours],
                hoverinfo="text+y",
            ), row=dim + 1, col=1)

            # Mark ATH
            ath_h = int(row["ath_hour"])
            if ath_h < len(traj):
                fig.add_trace(go.Scatter(
                    x=[ath_h], y=[traj[ath_h, dim]],
                    mode="markers",
                    marker=dict(color=color, size=6, symbol="star"),
                    showlegend=False,
                    hovertext=f"{symbol} ATH",
                    hoverinfo="text",
                ), row=dim + 1, col=1)

    # Summary stats in title
    med_ath = cluster_df["ath_mcap"].median()
    med_hours = cluster_df["total_hours"].median()
    med_ath_h = cluster_df["ath_hour"].median()

    fig.update_layout(
        title=(f"Cluster {cluster_id} — {len(cluster_df)} tokens | "
               f"Median ATH: ${med_ath:,.0f} @ hour {med_ath_h:.0f} | "
               f"Median life: {med_hours:.0f}h"),
        height=900, width=1200,
    )
    fig.update_xaxes(title_text="Hours from 50K", row=3, col=1)

    path = os.path.join(PLOTS_DIR, f"cluster4d_{cluster_id}.html")
    fig.write_html(path)
    print(f"  Cluster {cluster_id} detail: {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-cluster", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Build features
    print("Building token features (path signatures in 4D log-space)...")
    features, token_info, raw_trajs = build_token_features()
    print(f"  {len(features)} tokens, {features.shape[1]}-dim signature vectors")

    # Cluster
    labels = cluster_tokens(features, min_cluster_size=args.min_cluster)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  Found {n_clusters} clusters, {n_noise} noise points")

    # Summary
    df = pd.DataFrame(token_info)
    df["cluster"] = labels

    print(f"\n{'='*70}")
    for label in sorted(set(labels)):
        cluster_data = df[df["cluster"] == label]
        name = f"Cluster {label}" if label >= 0 else "Noise"
        print(f"  {name:12s} ({len(cluster_data):3d}) | "
              f"ATH med ${cluster_data['ath_mcap'].median():>12,.0f} | "
              f"ATH@{cluster_data['ath_hour'].median():>4.0f}h | "
              f"Life {cluster_data['total_hours'].median():>4.0f}h | "
              f"Holders {cluster_data['holders_at_ath'].median():>6,.0f} | "
              f"Top10 {cluster_data['top10_at_ath'].median():>5.1f}%")
    print(f"{'='*70}")

    # Save
    df.to_csv(os.path.join(DATA_DIR, "clusters_4d.csv"), index=False)

    # Plot overview
    plot_cluster_selector(token_info, labels)

    # Plot each cluster detail
    for cluster_id in range(n_clusters):
        plot_cluster_detail(cluster_id, token_info, labels, raw_trajs)

    # Open overview
    import subprocess
    subprocess.run(["open", os.path.join(PLOTS_DIR, "cluster_overview.html")])


if __name__ == "__main__":
    main()
