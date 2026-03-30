"""3D trajectory visualization (PCA on raw DNA) + trajectory clustering (TS2Vec + HDBSCAN).

Visualization: 4D DNA → PCA 3D (preserves trajectory continuity)
Clustering:    TS2Vec instance embeddings → HDBSCAN (groups similar trajectories)
Coloring:      Cluster labels applied to PCA trajectories

Reads:  data/dataset.npy — shape (N, 200, 4) z-scored DNA
        data/embeddings.npy — shape (N, 320) instance-level TS2Vec embeddings
        data/metadata.csv — token metadata
Writes: data/clusters.csv — cluster assignments
        plots/trajectories_3d.html — interactive 3D trajectory lines
        plots/trajectories_3d.png — static snapshot
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import hdbscan
import plotly.graph_objects as go

DATA_DIR = "data"
PLOTS_DIR = "plots"

CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
NOISE_COLOR = "rgba(180, 180, 180, 0.25)"


def main():
    # --- Load data ---
    dataset = np.load(os.path.join(DATA_DIR, "dataset.npy"))      # (N, 200, 4)
    inst_emb = np.load(os.path.join(DATA_DIR, "embeddings.npy"))   # (N, 320)
    metadata = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))

    n_tokens, n_steps, n_dims = dataset.shape
    print(f"Loaded {n_tokens} tokens, {n_steps} steps, {n_dims}D DNA")

    # --- Per-token min-max normalization for visualization ---
    # Clustering uses z-score (preserves magnitude); visualization uses per-token [0,1] (compares shape)
    print("Normalizing per-token to [0,1] for visualization...")
    vis_data = dataset.copy()
    for i in range(n_tokens):
        for d in range(n_dims):
            col = vis_data[i, :, d]
            col_min, col_max = col.min(), col.max()
            if col_max > col_min:
                vis_data[i, :, d] = (col - col_min) / (col_max - col_min)
            else:
                vis_data[i, :, d] = 0.0

    # --- PCA: 4D DNA → 3D (all timesteps together, preserves continuity) ---
    print("Running PCA 4D → 3D...")
    all_points = vis_data.reshape(-1, n_dims)  # (N*200, 4)
    pca = PCA(n_components=3)
    coords_3d = pca.fit_transform(all_points)  # (N*200, 3)
    coords_3d = coords_3d.reshape(n_tokens, n_steps, 3)  # (N, 200, 3)

    explained = pca.explained_variance_ratio_
    print(f"PCA explained variance: {explained[0]:.1%}, {explained[1]:.1%}, {explained[2]:.1%} (total {sum(explained):.1%})")
    print(f"PCA components (loadings):")
    for i, comp in enumerate(pca.components_):
        labels = ["mcap", "holders", "top10%", "mcap/holder"]
        loadings = ", ".join(f"{l}: {v:+.3f}" for l, v in zip(labels, comp))
        print(f"  PC{i+1}: {loadings}")

    # --- Clustering: TS2Vec instance embeddings → HDBSCAN ---
    print("\nRunning HDBSCAN on TS2Vec instance embeddings...")
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

    # Noise trajectories (faded background)
    noise_idx = np.where(labels == -1)[0]
    for i, idx in enumerate(noise_idx):
        symbol = metadata.iloc[idx]["symbol"]
        x, y, z = coords_3d[idx, :, 0], coords_3d[idx, :, 1], coords_3d[idx, :, 2]
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            line=dict(color=NOISE_COLOR, width=2),
            name=symbol,
            legendgroup="Noise",
            showlegend=bool(i == 0),
            legendgrouptitle_text="Noise" if i == 0 else None,
            hovertext=[f"{symbol}<br>Step {s}/{n_steps}" for s in range(1, n_steps + 1)],
            hoverinfo="text",
        ))

    # Cluster trajectories
    for cluster_id in range(n_clusters):
        cluster_idx = np.where(labels == cluster_id)[0]
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]

        for j, idx in enumerate(cluster_idx):
            symbol = metadata.iloc[idx]["symbol"]
            ath = metadata.iloc[idx]["ath_mcap"]
            duration = metadata.iloc[idx]["rally_duration_hours"]
            x, y, z = coords_3d[idx, :, 0], coords_3d[idx, :, 1], coords_3d[idx, :, 2]

            # Trajectory line
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="lines",
                line=dict(color=color, width=3),
                name=symbol,
                legendgroup=f"Cluster {cluster_id}",
                showlegend=bool(j == 0),
                legendgrouptitle_text=f"Cluster {cluster_id} ({len(cluster_idx)})" if j == 0 else None,
                hovertext=[
                    f"{symbol}<br>ATH: ${ath:,.0f}<br>Rally: {duration}h<br>Step {s}/{n_steps}"
                    for s in range(1, n_steps + 1)
                ],
                hoverinfo="text",
            ))

            # Start marker (100K)
            fig.add_trace(go.Scatter3d(
                x=[x[0]], y=[y[0]], z=[z[0]],
                mode="markers",
                marker=dict(color=color, size=3, symbol="circle"),
                showlegend=False,
                hovertext=[f"{symbol} START (100K)"],
                hoverinfo="text",
            ))
            # End marker (ATH)
            fig.add_trace(go.Scatter3d(
                x=[x[-1]], y=[y[-1]], z=[z[-1]],
                mode="markers",
                marker=dict(color=color, size=5, symbol="diamond"),
                showlegend=False,
                hovertext=[f"{symbol} ATH ${ath:,.0f}"],
                hoverinfo="text",
            ))

    fig.update_layout(
        title="Memecoin DNA Trajectories (100K → ATH) — PCA 3D + TS2Vec Clusters",
        scene=dict(
            xaxis_title=f"PC1 ({explained[0]:.0%})",
            yaxis_title=f"PC2 ({explained[1]:.0%})",
            zaxis_title=f"PC3 ({explained[2]:.0%})",
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
