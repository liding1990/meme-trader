"""3D trajectory visualization + trajectory clustering.

Reads:  data/trajectory_embeddings.npy — shape (N, 200, 320) timestamp-level
        data/embeddings.npy — shape (N, 320) instance-level
        data/metadata.csv — token metadata
Writes: data/clusters.csv — cluster assignments
        plots/trajectories_3d.html — interactive 3D trajectory lines
        plots/trajectories_3d.png — static snapshot
"""

import os
import sys

import numpy as np
import pandas as pd
import umap
import hdbscan
import plotly.graph_objects as go

DATA_DIR = "data"
PLOTS_DIR = "plots"

# Distinct colors for clusters
CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]
NOISE_COLOR = "rgba(180, 180, 180, 0.3)"


def main():
    # Load data
    traj_path = os.path.join(DATA_DIR, "trajectory_embeddings.npy")
    emb_path = os.path.join(DATA_DIR, "embeddings.npy")
    meta_path = os.path.join(DATA_DIR, "metadata.csv")

    if not os.path.isfile(traj_path):
        print("ERROR: data/trajectory_embeddings.npy not found. Run train_ts2vec.py first.", file=sys.stderr)
        sys.exit(1)

    traj_emb = np.load(traj_path)       # (N, 200, 320)
    inst_emb = np.load(emb_path)         # (N, 320)
    metadata = pd.read_csv(meta_path)

    n_tokens, n_steps, emb_dim = traj_emb.shape
    print(f"Loaded {n_tokens} tokens, {n_steps} steps, {emb_dim}D embeddings")

    # --- UMAP: flatten all timesteps, project to 3D ---
    print("Running UMAP on all timesteps...")
    all_points = traj_emb.reshape(-1, emb_dim)  # (N*200, 320)
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=30,
        min_dist=0.05,
        metric="euclidean",
        random_state=42,
    )
    coords_3d = reducer.fit_transform(all_points)  # (N*200, 3)
    coords_3d = coords_3d.reshape(n_tokens, n_steps, 3)  # (N, 200, 3)
    print(f"UMAP output shape: {coords_3d.shape}")

    # --- Trajectory clustering using instance-level embeddings ---
    print("Running HDBSCAN on instance-level embeddings...")
    if n_tokens < 5:
        labels = np.full(n_tokens, -1)
    else:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
        labels = clusterer.fit_predict(inst_emb)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"Found {n_clusters} clusters, {n_noise} noise points")

    # --- Save cluster assignments ---
    results = metadata.copy()
    results["cluster"] = labels
    results.to_csv(os.path.join(DATA_DIR, "clusters.csv"), index=False)

    # --- Cluster summary ---
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

    # --- 3D trajectory plot ---
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig = go.Figure()

    # Draw noise trajectories first (background, faded)
    noise_idx = np.where(labels == -1)[0]
    for idx in noise_idx:
        symbol = metadata.iloc[idx]["symbol"]
        x, y, z = coords_3d[idx, :, 0], coords_3d[idx, :, 1], coords_3d[idx, :, 2]
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            line=dict(color=NOISE_COLOR, width=2),
            name=f"{symbol} (noise)",
            legendgroup="Noise",
            showlegend=(idx == noise_idx[0]),
            legendgrouptitle_text="Noise" if idx == noise_idx[0] else None,
            hovertext=[f"{symbol}<br>Step {s}/{n_steps}" for s in range(1, n_steps + 1)],
            hoverinfo="text",
        ))

    # Draw cluster trajectories
    for cluster_id in range(n_clusters):
        cluster_idx = np.where(labels == cluster_id)[0]
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]

        for j, idx in enumerate(cluster_idx):
            symbol = metadata.iloc[idx]["symbol"]
            ath = metadata.iloc[idx]["ath_mcap"]
            duration = metadata.iloc[idx]["rally_duration_hours"]
            x, y, z = coords_3d[idx, :, 0], coords_3d[idx, :, 1], coords_3d[idx, :, 2]

            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="lines",
                line=dict(color=color, width=3),
                name=f"{symbol}",
                legendgroup=f"Cluster {cluster_id}",
                showlegend=(j == 0),
                legendgrouptitle_text=f"Cluster {cluster_id} ({len(cluster_idx)})" if j == 0 else None,
                hovertext=[
                    f"{symbol}<br>ATH: ${ath:,.0f}<br>Rally: {duration}h<br>Step {s}/{n_steps}"
                    for s in range(1, n_steps + 1)
                ],
                hoverinfo="text",
            ))

            # Start marker
            fig.add_trace(go.Scatter3d(
                x=[x[0]], y=[y[0]], z=[z[0]],
                mode="markers",
                marker=dict(color=color, size=4, symbol="circle"),
                showlegend=False,
                hovertext=[f"{symbol} START"],
                hoverinfo="text",
            ))
            # End marker (ATH)
            fig.add_trace(go.Scatter3d(
                x=[x[-1]], y=[y[-1]], z=[z[-1]],
                mode="markers",
                marker=dict(color=color, size=6, symbol="diamond"),
                showlegend=False,
                hovertext=[f"{symbol} ATH ${ath:,.0f}"],
                hoverinfo="text",
            ))

    fig.update_layout(
        title="Memecoin DNA Trajectories (100K → ATH)",
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            zaxis_title="UMAP-3",
        ),
        width=1400,
        height=900,
        legend=dict(groupclick="togglegroup"),
    )

    html_path = os.path.join(PLOTS_DIR, "trajectories_3d.html")
    fig.write_html(html_path)
    print(f"\nInteractive plot saved: {html_path}")

    png_path = os.path.join(PLOTS_DIR, "trajectories_3d.png")
    fig.write_image(png_path, width=1400, height=900, scale=2)
    print(f"Static plot saved: {png_path}")


if __name__ == "__main__":
    main()
