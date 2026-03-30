"""Deep analysis of a specific cluster: understand the typical trajectory pattern.

For a given cluster:
1. Overlay all members' raw 4D DNA curves (individual lines)
2. Compute and plot the MEDIAN trajectory + IQR envelope (the "typical" path)
3. Normalize all trajectories to percentage of ATH for comparable overlay
4. Show key phase analysis: how fast to ATH, what happens after ATH

Usage:
    python analyze_cluster.py [cluster_id]  (default: 1)
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visualize import load_raw_trajectory

DATA_DIR = "data"
PLOTS_DIR = "plots"
DNA_LABELS = ["Market Cap ($)", "Holders", "Top10 Holder %", "MCap / Holder ($)"]
LINE_COLOR = "rgba(31, 119, 180, 0.3)"
MEDIAN_COLOR = "#d62728"
ENVELOPE_COLOR = "rgba(214, 39, 40, 0.15)"


def load_cluster_data(cluster_id):
    """Load metadata and raw trajectories for a cluster."""
    df = pd.read_csv(os.path.join(DATA_DIR, "sig_clusters.csv"))
    cluster = df[df["cluster"] == cluster_id]
    print(f"Cluster {cluster_id}: {len(cluster)} tokens")

    trajs = []
    metas = []
    for _, row in cluster.iterrows():
        raw = load_raw_trajectory(row["address"])
        if raw is not None:
            trajs.append(raw)
            metas.append(row)

    return trajs, metas


def normalize_to_ath_pct(trajs, metas):
    """Normalize each trajectory's mcap to % of its ATH.
    Also normalize time to % of ATH hour.

    Returns list of (time_pct, mcap_pct, holders_pct, top10, mcap_per_holder_pct) arrays.
    """
    normalized = []
    for traj, meta in zip(trajs, metas):
        ath_hour = int(meta["ath_hour"])
        ath_mcap = meta["ath_mcap"]
        max_holders = traj[:, 1].max()
        max_mph = traj[:, 3].max()

        if ath_mcap <= 0 or max_holders <= 0 or max_mph <= 0:
            continue

        T = len(traj)
        time_pct = np.arange(T) / max(ath_hour, 1) * 100  # 100% = ATH hour

        mcap_pct = traj[:, 0] / ath_mcap * 100
        holders_pct = traj[:, 1] / max_holders * 100
        top10 = traj[:, 2]
        mph_pct = traj[:, 3] / max_mph * 100

        normalized.append({
            "time_pct": time_pct,
            "mcap_pct": mcap_pct,
            "holders_pct": holders_pct,
            "top10": top10,
            "mph_pct": mph_pct,
            "symbol": meta["symbol"],
            "ath_mcap": ath_mcap,
            "ath_hour": ath_hour,
            "total_hours": meta["total_hours"],
        })

    return normalized


def compute_envelope(trajs, max_hours):
    """Compute median + IQR envelope from raw trajectories, aligned by hour."""
    # Pad shorter trajectories with NaN
    n_dims = 4
    padded = np.full((len(trajs), max_hours, n_dims), np.nan)
    for i, traj in enumerate(trajs):
        T = min(len(traj), max_hours)
        padded[i, :T, :] = traj[:T, :]

    median = np.nanmedian(padded, axis=0)
    q25 = np.nanpercentile(padded, 25, axis=0)
    q75 = np.nanpercentile(padded, 75, axis=0)
    count = np.sum(~np.isnan(padded[:, :, 0]), axis=0)  # how many tokens alive at each hour

    return median, q25, q75, count


def plot_raw_overlay(trajs, metas, cluster_id):
    """Plot 1: Raw DNA curves overlay with median envelope."""
    max_hours = max(len(t) for t in trajs)
    median, q25, q75, count = compute_envelope(trajs, max_hours)
    hours = list(range(max_hours))

    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        subplot_titles=DNA_LABELS + ["Surviving Tokens"],
        vertical_spacing=0.05,
    )

    # Individual trajectories
    for i, (traj, meta) in enumerate(zip(trajs, metas)):
        h = list(range(len(traj)))
        for dim in range(4):
            fig.add_trace(go.Scatter(
                x=h, y=traj[:, dim],
                mode="lines", line=dict(color=LINE_COLOR, width=1),
                showlegend=False, hovertext=meta["symbol"], hoverinfo="text+y",
            ), row=dim + 1, col=1)

    # Median + envelope
    for dim in range(4):
        # Envelope
        fig.add_trace(go.Scatter(
            x=hours + hours[::-1],
            y=list(q75[:, dim]) + list(q25[::-1, dim]),
            fill="toself", fillcolor=ENVELOPE_COLOR,
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=dim + 1, col=1)
        # Median
        fig.add_trace(go.Scatter(
            x=hours, y=median[:, dim],
            mode="lines", line=dict(color=MEDIAN_COLOR, width=3),
            name="Median" if dim == 0 else None, showlegend=(dim == 0),
        ), row=dim + 1, col=1)

    # Survival curve
    fig.add_trace(go.Scatter(
        x=hours, y=count,
        mode="lines", line=dict(color="#2ca02c", width=2),
        fill="tozeroy", fillcolor="rgba(44, 160, 44, 0.2)",
        name="Tokens alive", showlegend=True,
    ), row=5, col=1)

    fig.update_layout(
        title=f"Cluster {cluster_id} Pattern Analysis — {len(trajs)} tokens (Median + IQR envelope)",
        height=1200, width=1200,
    )
    fig.update_xaxes(title_text="Hours from 100K", row=5, col=1)

    path = os.path.join(PLOTS_DIR, f"cluster_{cluster_id}_raw.html")
    fig.write_html(path)
    print(f"Raw overlay plot: {path}")
    return path


def plot_normalized_overlay(normalized, cluster_id):
    """Plot 2: All trajectories normalized — time as % of ATH hour, mcap as % of ATH."""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "Market Cap (% of ATH)",
            "Holders (% of peak)",
            "Top10 Holder %",
            "MCap/Holder (% of peak)",
        ],
        vertical_spacing=0.06,
    )

    dims = ["mcap_pct", "holders_pct", "top10", "mph_pct"]
    for item in normalized:
        for dim_idx, dim_key in enumerate(dims):
            fig.add_trace(go.Scatter(
                x=item["time_pct"], y=item[dim_key],
                mode="lines", line=dict(width=1.5),
                opacity=0.5,
                name=item["symbol"] if dim_idx == 0 else None,
                legendgroup=item["symbol"],
                showlegend=(dim_idx == 0),
                hovertext=f"{item['symbol']} ATH=${item['ath_mcap']:,.0f}",
                hoverinfo="text+y",
            ), row=dim_idx + 1, col=1)

    # Add vertical line at 100% (ATH moment)
    for dim_idx in range(4):
        fig.add_vline(
            x=100, line_dash="dash", line_color="red", line_width=1,
            annotation_text="ATH" if dim_idx == 0 else None,
            row=dim_idx + 1, col=1,
        )

    fig.update_layout(
        title=f"Cluster {cluster_id} — Normalized Trajectories (time as % of ATH hour)",
        height=1000, width=1200,
    )
    fig.update_xaxes(title_text="Time (% of ATH hour, 100% = ATH)", row=4, col=1)

    path = os.path.join(PLOTS_DIR, f"cluster_{cluster_id}_normalized.html")
    fig.write_html(path)
    print(f"Normalized overlay plot: {path}")
    return path


def print_phase_analysis(trajs, metas):
    """Analyze the lifecycle phases of the cluster."""
    print(f"\n{'='*70}")
    print(f"  Lifecycle Phase Analysis")
    print(f"{'='*70}")

    ath_hours = [m["ath_hour"] for m in metas]
    total_hours = [m["total_hours"] for m in metas]
    ath_mcaps = [m["ath_mcap"] for m in metas]

    # Phase 1: pre-ATH
    print(f"\n  Pre-ATH (growth phase):")
    print(f"    Hours to ATH:    median {np.median(ath_hours):.0f}h  (range {min(ath_hours)}—{max(ath_hours)}h)")

    # Holder growth rate in pre-ATH phase
    holder_growth_rates = []
    for traj, meta in zip(trajs, metas):
        ath_h = int(meta["ath_hour"])
        if ath_h > 0:
            start_holders = traj[0, 1]
            ath_holders = traj[min(ath_h, len(traj) - 1), 1]
            if start_holders > 0:
                rate = (ath_holders / start_holders - 1) * 100
                holder_growth_rates.append(rate)
    if holder_growth_rates:
        print(f"    Holder growth:   median {np.median(holder_growth_rates):+.0f}%  to ATH")

    # Phase 2: post-ATH
    print(f"\n  Post-ATH (decline/sustain phase):")
    post_ath_hours = [t - a for t, a in zip(total_hours, ath_hours)]
    print(f"    Hours after ATH: median {np.median(post_ath_hours):.0f}h  (range {min(post_ath_hours)}—{max(post_ath_hours)}h)")

    # How much mcap retained after ATH
    retention_rates = []
    for traj, meta in zip(trajs, metas):
        ath_h = int(meta["ath_hour"])
        ath_mcap = meta["ath_mcap"]
        if ath_mcap > 0 and len(traj) > ath_h:
            final_mcap = traj[-1, 0]
            retention = final_mcap / ath_mcap * 100
            retention_rates.append(retention)
    if retention_rates:
        print(f"    MCap retention:  median {np.median(retention_rates):.1f}%  of ATH at end")
        print(f"                     (range {min(retention_rates):.1f}% — {max(retention_rates):.1f}%)")

    # Top10 change
    top10_changes = []
    for traj, meta in zip(trajs, metas):
        if len(traj) >= 3:
            start_t10 = traj[0, 2]
            end_t10 = traj[-1, 2]
            top10_changes.append(end_t10 - start_t10)
    if top10_changes:
        print(f"\n  Concentration shift:")
        print(f"    Top10% change:   median {np.median(top10_changes):+.1f}pp  (start→end)")

    print(f"{'='*70}")


def main():
    cluster_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    os.makedirs(PLOTS_DIR, exist_ok=True)

    trajs, metas = load_cluster_data(cluster_id)
    if not trajs:
        print(f"No valid trajectories for cluster {cluster_id}")
        sys.exit(1)

    # Phase analysis
    print_phase_analysis(trajs, metas)

    # Plot 1: Raw overlay with median envelope
    raw_path = plot_raw_overlay(trajs, metas, cluster_id)

    # Plot 2: Normalized overlay
    normalized = normalize_to_ath_pct(trajs, metas)
    if normalized:
        norm_path = plot_normalized_overlay(normalized, cluster_id)

    print(f"\nOpen in browser to explore patterns.")


if __name__ == "__main__":
    main()
