"""4D state space: time × mcap × holders × top10%.

Every token's every hour is a point in this 4D space.
No normalization, no encoding — raw values.

Two capabilities:
1. Cluster: find common "states" that many tokens pass through
2. Predict: given a new token's current point, find historical tokens
   that passed through the same neighborhood, show where they went next

Usage:
    # Build the state space from all historical tokens
    python space4d.py build

    # Query a new token
    python space4d.py query <contract_address> [--top 5] [--radius auto]
"""

import argparse
import json
import glob
import os
import sys
import pickle

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DATA_DIR = "data"
PLOTS_DIR = "plots"
MCAP_THRESHOLD = 50_000
DNA_COLUMNS = ["mcap", "holder_count", "top10_pct"]


def load_token_list():
    tokens = []
    with open("token_list.csv", "r") as f:
        import csv
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                tokens.append({"address": row[0], "symbol": row[1], "name": row[2] if len(row) > 2 else row[1]})
    return tokens


def extract_trajectory(address):
    """Load raw trajectory from cached JSON. Returns (T, 3) array [mcap, holders, top10%] or None."""
    from dna_extractor import extract_candle_series, extract_trend_series, build_dna_dataframe

    token_dir = os.path.join(DATA_DIR, address)
    candles_files = sorted(glob.glob(os.path.join(token_dir, "token_mcap_candles_*.json")))
    trends_files = sorted(glob.glob(os.path.join(token_dir, "token_trends_*.json")))
    if not candles_files or not trends_files:
        return None

    with open(candles_files[-1]) as f:
        candles_data = json.load(f)
    with open(trends_files[-1]) as f:
        trends_data = json.load(f)

    candle_df = extract_candle_series(candles_data)
    trend_df = extract_trend_series(trends_data)
    if candle_df is None or trend_df is None:
        return None

    dna_df = build_dna_dataframe(candle_df, trend_df)
    if dna_df is None or len(dna_df) < 2:
        return None

    mcap = dna_df["mcap"].values
    above = np.where(mcap >= MCAP_THRESHOLD)[0]
    if len(above) == 0:
        return None

    trimmed = dna_df.iloc[above[0]:].reset_index(drop=True)
    return trimmed[DNA_COLUMNS].values  # (T, 3)


def build_state_space():
    """Build the 4D state space from all historical tokens.

    Each point: [time_hour, mcap, holders, top10_pct]
    Also stores which token and which hour each point belongs to.
    """
    tokens = load_token_list()
    print(f"Building 4D state space from {len(tokens)} tokens...")

    all_points = []    # (N, 4) — [time, mcap, holders, top10%]
    all_meta = []      # per-point: {address, symbol, hour, total_hours}
    token_trajs = {}   # address → full raw trajectory

    for token in tokens:
        address = token["address"]
        traj = extract_trajectory(address)
        if traj is None:
            continue

        T = len(traj)
        token_trajs[address] = traj

        for h in range(T):
            point = [float(h), traj[h, 0], traj[h, 1], traj[h, 2]]
            all_points.append(point)
            all_meta.append({
                "address": address,
                "symbol": token["symbol"],
                "name": token["name"],
                "hour": h,
                "total_hours": T,
            })

    all_points = np.array(all_points)
    print(f"State space: {len(all_points)} points from {len(token_trajs)} tokens")
    print(f"  Time range:    0 — {all_points[:, 0].max():.0f} hours")
    print(f"  MCap range:    ${all_points[:, 1].min():,.0f} — ${all_points[:, 1].max():,.0f}")
    print(f"  Holders range: {all_points[:, 2].min():.0f} — {all_points[:, 2].max():.0f}")
    print(f"  Top10% range:  {all_points[:, 3].min():.1f}% — {all_points[:, 3].max():.1f}%")

    # Build KD-tree for fast nearest neighbor lookup
    # Normalize dimensions for distance computation (different scales)
    scales = all_points.max(axis=0) - all_points.min(axis=0)
    scales[scales == 0] = 1
    normalized = all_points / scales

    tree = cKDTree(normalized)

    # Save everything
    space = {
        "points": all_points,
        "meta": all_meta,
        "scales": scales,
        "token_trajs": token_trajs,
    }

    space_path = os.path.join(DATA_DIR, "state_space_4d.pkl")
    with open(space_path, "wb") as f:
        pickle.dump(space, f)
    print(f"Saved: {space_path}")

    return space


def load_state_space():
    space_path = os.path.join(DATA_DIR, "state_space_4d.pkl")
    if not os.path.isfile(space_path):
        print("ERROR: state space not found. Run 'python space4d.py build' first.", file=sys.stderr)
        sys.exit(1)
    with open(space_path, "rb") as f:
        return pickle.load(f)


def query_token(address, space, top_n=5, chain="sol"):
    """Find historical tokens that passed through the same region as the new token's current state."""
    from gmgn_api import fetch_token_data
    from dna_extractor import extract_candle_series, extract_trend_series, build_dna_dataframe

    # Fetch new token
    print(f"Fetching {address[:8]}...")
    _, loaded_data = fetch_token_data(chain, address)

    candle_df = extract_candle_series(loaded_data.get("token_mcap_candles", {}))
    trend_df = extract_trend_series(loaded_data.get("token_trends", {}))
    if candle_df is None or trend_df is None:
        print("ERROR: Could not fetch data", file=sys.stderr)
        return

    dna_df = build_dna_dataframe(candle_df, trend_df)
    if dna_df is None:
        print("ERROR: Could not build DNA", file=sys.stderr)
        return

    mcap = dna_df["mcap"].values
    above = np.where(mcap >= MCAP_THRESHOLD)[0]
    if len(above) == 0:
        print(f"ERROR: Token never reached ${MCAP_THRESHOLD:,} mcap", file=sys.stderr)
        return

    trimmed = dna_df.iloc[above[0]:].reset_index(drop=True)
    new_traj = trimmed[DNA_COLUMNS].values
    K = len(new_traj)

    # Current state = last point
    current_hour = K - 1
    current_state = np.array([float(current_hour), new_traj[-1, 0], new_traj[-1, 1], new_traj[-1, 2]])

    print(f"\nNew token state at hour {current_hour}:")
    print(f"  MCap:    ${current_state[1]:,.0f}")
    print(f"  Holders: {current_state[2]:,.0f}")
    print(f"  Top10%:  {current_state[3]:.1f}%")

    # Find nearest neighbors in normalized space
    points = space["points"]
    scales = space["scales"]
    meta = space["meta"]
    token_trajs = space["token_trajs"]

    normalized_points = points / scales
    normalized_query = current_state / scales

    tree = cKDTree(normalized_points)

    # Find nearest points (more than needed, then deduplicate by token)
    distances, indices = tree.query(normalized_query, k=min(200, len(points)))

    # Deduplicate: keep only the closest point per unique token
    seen_tokens = set()
    # Exclude the queried token itself if it's in the index
    matches = []
    for dist, idx in zip(distances, indices):
        m = meta[idx]
        addr = m["address"]
        if addr == address or addr in seen_tokens:
            continue
        seen_tokens.add(addr)

        # Get the full trajectory of this matching token
        if addr not in token_trajs:
            continue

        full_traj = token_trajs[addr]
        match_hour = m["hour"]
        hours_remaining = m["total_hours"] - match_hour - 1

        # What happened after this point?
        future_traj = full_traj[match_hour:] if match_hour < len(full_traj) else None

        matches.append({
            "address": addr,
            "symbol": m["symbol"],
            "name": m["name"],
            "distance": dist,
            "matched_hour": match_hour,
            "matched_state": points[idx],
            "total_hours": m["total_hours"],
            "hours_remaining": hours_remaining,
            "future_traj": future_traj,
        })

        if len(matches) >= top_n:
            break

    # Print results
    print(f"\n{'='*70}")
    print(f"  Top {len(matches)} Historical Matches (nearest in 4D state space)")
    print(f"{'='*70}")

    for i, m in enumerate(matches, 1):
        ms = m["matched_state"]
        print(f"\n  #{i} {m['symbol']} ({m['name']}) — distance: {m['distance']:.4f}")
        print(f"     Matched at hour {m['matched_hour']}: MCap=${ms[1]:,.0f}, Holders={ms[2]:,.0f}, Top10={ms[3]:.1f}%")
        print(f"     That token lived {m['hours_remaining']} more hours after this point")
        if m["future_traj"] is not None and len(m["future_traj"]) > 1:
            fut = m["future_traj"]
            mcap_start = fut[0, 0]
            mcap_end = fut[-1, 0]
            mcap_max = fut[:, 0].max()
            print(f"     Future MCap: ${mcap_start:,.0f} → peak ${mcap_max:,.0f} → final ${mcap_end:,.0f}")
            if mcap_start > 0:
                print(f"     Peak upside: {(mcap_max/mcap_start - 1)*100:+.0f}%  |  Final: {(mcap_end/mcap_start - 1)*100:+.0f}%")

    print(f"{'='*70}")

    # Visualize
    _plot_query_result(new_traj, address[:8], current_hour, matches)

    return matches


def _plot_query_result(new_traj, new_symbol, current_hour, matches):
    """Plot the new token's trajectory and matching tokens' future trajectories."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    os.makedirs(PLOTS_DIR, exist_ok=True)

    dim_labels = ["Market Cap ($)", "Holders", "Top10 Holder %"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=dim_labels,
        vertical_spacing=0.08,
    )

    # New token trajectory (black, bold)
    hours_new = list(range(len(new_traj)))
    for dim in range(3):
        fig.add_trace(go.Scatter(
            x=hours_new, y=new_traj[:, dim],
            mode="lines", line=dict(color="black", width=3),
            name=f"{new_symbol} (current)" if dim == 0 else None,
            legendgroup="new", showlegend=(dim == 0),
        ), row=dim + 1, col=1)

    # Matching tokens
    for i, m in enumerate(matches):
        color = colors[i % len(colors)]
        symbol = m["symbol"]
        match_hour = m["matched_hour"]
        fut = m["future_traj"]
        if fut is None:
            continue

        # Offset future trajectory so it starts at current_hour on the x-axis
        future_hours = [current_hour + h for h in range(len(fut))]

        for dim in range(3):
            fig.add_trace(go.Scatter(
                x=future_hours, y=fut[:, dim],
                mode="lines", line=dict(color=color, width=2),
                name=f"{symbol} (d={m['distance']:.3f})" if dim == 0 else None,
                legendgroup=f"m{i}", showlegend=(dim == 0),
                opacity=0.8,
            ), row=dim + 1, col=1)

    # NOW line
    for dim in range(3):
        fig.add_vline(
            x=current_hour, line_dash="dash", line_color="red", line_width=1,
            annotation_text="NOW" if dim == 0 else None,
            row=dim + 1, col=1,
        )

    fig.update_layout(
        title=f"4D State Space Query: {new_symbol} at Hour {current_hour}",
        height=900, width=1200,
    )
    fig.update_xaxes(title_text="Hours from 50K", row=3, col=1)

    path = os.path.join(PLOTS_DIR, f"space4d_{new_symbol}.html")
    fig.write_html(path)
    print(f"\nPlot saved: {path}")

    png_path = os.path.join(PLOTS_DIR, f"space4d_{new_symbol}.png")
    fig.write_image(png_path, width=1200, height=900, scale=2)
    print(f"Static: {png_path}")

    import subprocess
    subprocess.run(["open", path])


def main():
    parser = argparse.ArgumentParser(description="4D State Space: time × mcap × holders × top10%")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("build", help="Build state space from all historical tokens")

    q = sub.add_parser("query", help="Query a new token")
    q.add_argument("address", help="Contract address")
    q.add_argument("--top", type=int, default=5)
    q.add_argument("--chain", default="sol")

    args = parser.parse_args()

    if args.command == "build":
        build_state_space()
    elif args.command == "query":
        space = load_state_space()
        query_token(args.address, space, top_n=args.top, chain=args.chain)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
