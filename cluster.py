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
    if len(coords_3d) < 5:
        print(f"  Only {len(coords_3d)} samples — too few for HDBSCAN (min_cluster_size=5). Labelling all as noise.")
        labels = np.full(len(coords_3d), -1)
    else:
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
