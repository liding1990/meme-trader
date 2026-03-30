"""Visualize prediction: new token trajectory + similar historical continuations.

Creates a chart showing:
- The new token's trajectory so far (black, bold)
- Top-N matching historical tokens' full trajectories (colored, with the matched
  prefix portion solid and the continuation/prediction portion highlighted)

Each of the 4 DNA dimensions gets its own subplot.

Usage: called from query.py or standalone
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_DIR = "data"
TRAJ_DIR = os.path.join(DATA_DIR, "trajectories")
PLOTS_DIR = "plots"

DNA_LABELS = ["Market Cap", "Holders", "Top10 Holder %", "MCap / Holder"]
MATCH_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def load_raw_trajectory(address):
    """Load raw (un-normalized) trajectory from cached JSON data."""
    import json
    import glob
    from dna_extractor import extract_candle_series, extract_trend_series, build_dna_dataframe
    from preprocess import DNA_COLUMNS, MCAP_THRESHOLD

    token_dir = os.path.join(DATA_DIR, address)
    candles_file = sorted(glob.glob(os.path.join(token_dir, "token_mcap_candles_*.json")))
    trends_file = sorted(glob.glob(os.path.join(token_dir, "token_trends_*.json")))
    if not candles_file or not trends_file:
        return None

    with open(candles_file[-1]) as f:
        candles_data = json.load(f)
    with open(trends_file[-1]) as f:
        trends_data = json.load(f)

    candle_df = extract_candle_series(candles_data)
    trend_df = extract_trend_series(trends_data)
    if candle_df is None or trend_df is None:
        return None

    dna_df = build_dna_dataframe(candle_df, trend_df)
    if dna_df is None:
        return None

    mcap = dna_df["mcap"].values
    above = np.where(mcap >= MCAP_THRESHOLD)[0]
    if len(above) == 0:
        return None

    trimmed = dna_df.iloc[above[0]:].reset_index(drop=True)
    return trimmed[DNA_COLUMNS].values


def plot_prediction(new_token_raw, new_symbol, matches, K):
    """Create prediction visualization.

    Args:
        new_token_raw: (K, 4) raw DNA values of the new token
        new_symbol: symbol of the new token
        matches: list of dicts with 'address', 'symbol', 'distance'
        K: current prefix length (hours)
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=DNA_LABELS,
        vertical_spacing=0.06,
    )

    # Plot new token (bold black)
    hours = list(range(K))
    for dim in range(4):
        fig.add_trace(go.Scatter(
            x=hours, y=new_token_raw[:, dim],
            mode="lines",
            line=dict(color="black", width=3),
            name=f"{new_symbol} (current)" if dim == 0 else None,
            legendgroup="new",
            showlegend=(dim == 0),
        ), row=dim + 1, col=1)

    # Plot each matching historical token
    for i, match in enumerate(matches):
        raw = load_raw_trajectory(match["address"])
        if raw is None:
            continue

        color = MATCH_COLORS[i % len(MATCH_COLORS)]
        symbol = match["symbol"]
        T = len(raw)
        hours_full = list(range(T))

        for dim in range(4):
            # Matched prefix portion (dotted, faded)
            prefix_len = min(K, T)
            fig.add_trace(go.Scatter(
                x=hours_full[:prefix_len],
                y=raw[:prefix_len, dim],
                mode="lines",
                line=dict(color=color, width=1.5, dash="dot"),
                name=f"{symbol} (d={match['distance']:.3f})" if dim == 0 else None,
                legendgroup=f"match_{i}",
                showlegend=(dim == 0),
                opacity=0.6,
            ), row=dim + 1, col=1)

            # Continuation/prediction portion (bold, highlighted)
            if T > K:
                fig.add_trace(go.Scatter(
                    x=hours_full[K - 1:],
                    y=raw[K - 1:, dim],
                    mode="lines",
                    line=dict(color=color, width=2.5),
                    legendgroup=f"match_{i}",
                    showlegend=False,
                    opacity=0.8,
                ), row=dim + 1, col=1)

    # Add vertical line at current hour K
    for dim in range(4):
        fig.add_vline(
            x=K, line_dash="dash", line_color="red", line_width=1,
            annotation_text="NOW" if dim == 0 else None,
            row=dim + 1, col=1,
        )

    fig.update_layout(
        title=f"Trajectory Prediction for {new_symbol} — Top {len(matches)} Matches at Hour {K}",
        height=1000,
        width=1200,
        legend=dict(x=1.02, y=1),
    )
    fig.update_xaxes(title_text="Hours from 100K", row=4, col=1)

    html_path = os.path.join(PLOTS_DIR, f"prediction_{new_symbol}.html")
    fig.write_html(html_path)
    print(f"Prediction plot saved: {html_path}")

    png_path = os.path.join(PLOTS_DIR, f"prediction_{new_symbol}.png")
    fig.write_image(png_path, width=1200, height=1000, scale=2)
    print(f"Static plot saved: {png_path}")

    return html_path
