"""Signature-based clustering: understand what patterns exist in the data.

1. Compute full-trajectory path signature for each token (155-dim vector)
2. UMAP 2D projection for visualization
3. HDBSCAN clustering to discover natural groups
4. Interactive scatter plot (hover = token info)
5. Per-cluster DNA pattern plots (overlay all members' raw 4D curves)

Reads:  data/trajectories/{address}.npy
        data/metadata.csv
Writes: plots/sig_clusters.html — interactive 2D scatter
        plots/cluster_patterns.html — per-cluster DNA overlay
"""

import os
import sys

import numpy as np
import pandas as pd
import iisignature
import umap
import hdbscan
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from signature_index import augment_path, SIG_DEPTH
from visualize import load_raw_trajectory

DATA_DIR = "data"
TRAJ_DIR = os.path.join(DATA_DIR, "trajectories")
PLOTS_DIR = "plots"

DNA_LABELS = ["Market Cap ($)", "Holders", "Top10 Holder %", "MCap / Holder ($)"]
CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#aec7e8",
]
NOISE_COLOR = "#cccccc"


def compute_full_signatures(metadata):
    """Compute full-trajectory signature for each token."""
    signatures = []
    valid_indices = []

    for i, row in metadata.iterrows():
        traj_path = os.path.join(TRAJ_DIR, f"{row['address']}.npy")
        if not os.path.isfile(traj_path):
            continue

        traj = np.load(traj_path)
        if len(traj) < 3:
            continue

        augmented = augment_path(traj)
        sig = iisignature.sig(augmented, SIG_DEPTH)
        signatures.append(sig)
        valid_indices.append(i)

    return np.array(signatures), valid_indices


def cluster_and_project(signatures):
    """UMAP 2D + HDBSCAN clustering."""
    print("Running UMAP 2D projection...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )
    coords = reducer.fit_transform(signatures)

    print("Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
    labels = clusterer.fit_predict(coords)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"Found {n_clusters} clusters, {n_noise} noise points")

    return coords, labels


def plot_scatter(coords, labels, meta_df):
    """Interactive 2D scatter plot with hover info."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig = go.Figure()

    unique_labels = sorted(set(labels))

    for label in unique_labels:
        mask = labels == label
        subset = meta_df[mask]
        x, y = coords[mask, 0], coords[mask, 1]

        if label == -1:
            color = NOISE_COLOR
            name = f"Noise ({mask.sum()})"
        else:
            color = CLUSTER_COLORS[label % len(CLUSTER_COLORS)]
            name = f"Cluster {label} ({mask.sum()})"

        hover_text = [
            f"<b>{row['symbol']}</b> ({row['name']})<br>"
            f"ATH: ${row['ath_mcap']:,.0f}<br>"
            f"ATH hour: {row['ath_hour']}<br>"
            f"Total hours: {row['total_hours']}<br>"
            f"Holders@ATH: {row['holders_at_ath']:,.0f}<br>"
            f"Top10%@ATH: {row['top10_pct_at_ath']:.1f}%"
            for _, row in subset.iterrows()
        ]

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="markers+text",
            marker=dict(color=color, size=10, line=dict(width=1, color="white")),
            text=subset["symbol"].values,
            textposition="top center",
            textfont=dict(size=8),
            hovertext=hover_text,
            hoverinfo="text",
            name=name,
        ))

    fig.update_layout(
        title="Memecoin Trajectory Clusters (Path Signature + UMAP + HDBSCAN)",
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        width=1200,
        height=800,
        hovermode="closest",
    )

    path = os.path.join(PLOTS_DIR, "sig_clusters.html")
    fig.write_html(path)
    print(f"Cluster scatter saved: {path}")
    return path


def plot_cluster_patterns(labels, meta_df):
    """For each cluster, overlay all members' raw 4D DNA curves."""
    unique_clusters = sorted(c for c in set(labels) if c >= 0)
    if not unique_clusters:
        print("No clusters found, skipping pattern plots")
        return

    # Create one figure with subplots: rows = clusters, cols = 4 dims
    n_clusters = len(unique_clusters)
    max_v_spacing = 1.0 / max(n_clusters - 1, 1) - 0.01
    fig = make_subplots(
        rows=n_clusters, cols=4,
        subplot_titles=[f"C{c} — {dim}" for c in unique_clusters for dim in DNA_LABELS],
        vertical_spacing=min(0.08, max_v_spacing),
        horizontal_spacing=0.05,
    )

    for row_idx, cluster_id in enumerate(unique_clusters):
        mask = labels == cluster_id
        cluster_meta = meta_df[mask]
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]

        # Collect all raw trajectories for this cluster
        raw_trajs = []
        for _, token_row in cluster_meta.iterrows():
            raw = load_raw_trajectory(token_row["address"])
            if raw is not None:
                raw_trajs.append((token_row["symbol"], raw))

        for traj_idx, (symbol, raw) in enumerate(raw_trajs):
            hours = list(range(len(raw)))
            for dim in range(4):
                fig.add_trace(go.Scatter(
                    x=hours, y=raw[:, dim],
                    mode="lines",
                    line=dict(color=color, width=1.5),
                    opacity=0.5,
                    name=symbol if dim == 0 and traj_idx == 0 else None,
                    legendgroup=f"c{cluster_id}",
                    showlegend=(dim == 0 and traj_idx == 0),
                    hovertext=symbol,
                    hoverinfo="text+y",
                ), row=row_idx + 1, col=dim + 1)

    total_height = max(400, 300 * n_clusters)
    fig.update_layout(
        title="Cluster Patterns — Raw DNA Curves per Cluster",
        height=total_height,
        width=1600,
        showlegend=True,
    )

    # Label bottom row x-axes
    for col in range(1, 5):
        fig.update_xaxes(title_text="Hours", row=n_clusters, col=col)

    path = os.path.join(PLOTS_DIR, "cluster_patterns.html")
    fig.write_html(path)
    print(f"Cluster patterns saved: {path}")
    return path


def print_cluster_summary(labels, meta_df):
    """Print cluster statistics."""
    print(f"\n{'='*70}")
    print(f"  Cluster Summary")
    print(f"{'='*70}")

    for label in sorted(set(labels)):
        cluster_data = meta_df[labels == label]
        name = f"Cluster {label}" if label >= 0 else "Noise"

        print(f"\n  {name} ({len(cluster_data)} tokens):")
        print(f"    ATH mcap:       median ${cluster_data['ath_mcap'].median():,.0f}  "
              f"(range ${cluster_data['ath_mcap'].min():,.0f} — ${cluster_data['ath_mcap'].max():,.0f})")
        print(f"    ATH hour:       median {cluster_data['ath_hour'].median():.0f}h")
        print(f"    Total hours:    median {cluster_data['total_hours'].median():.0f}h")
        print(f"    Holders@ATH:    median {cluster_data['holders_at_ath'].median():,.0f}")
        print(f"    Top10%@ATH:     median {cluster_data['top10_pct_at_ath'].median():.1f}%")
        print(f"    Tokens: {', '.join(cluster_data['symbol'].head(8).tolist())}")

    print(f"{'='*70}")


def main():
    metadata = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
    print(f"Loaded {len(metadata)} tokens")

    # Compute signatures
    print("Computing full-trajectory signatures...")
    signatures, valid_indices = compute_full_signatures(metadata)
    meta_valid = metadata.iloc[valid_indices].reset_index(drop=True)
    print(f"Computed {len(signatures)} signatures ({signatures.shape[1]}-dim)")

    # Cluster
    coords, labels = cluster_and_project(signatures)

    # Summary
    print_cluster_summary(labels, meta_valid)

    # Save cluster assignments
    meta_valid["cluster"] = labels
    meta_valid["umap_x"] = coords[:, 0]
    meta_valid["umap_y"] = coords[:, 1]
    meta_valid.to_csv(os.path.join(DATA_DIR, "sig_clusters.csv"), index=False)

    # Plots
    scatter_path = plot_scatter(coords, labels, meta_valid)
    pattern_path = plot_cluster_patterns(labels, meta_valid)

    print(f"\nDone! Open in browser:")
    print(f"  {scatter_path}")
    print(f"  {pattern_path}")


if __name__ == "__main__":
    main()
