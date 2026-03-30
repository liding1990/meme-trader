"""Preprocess token DNA data for path signature matching.

For each token:
  1. Extract hourly DNA (mcap, holder_count, top10_pct, mcap_per_holder)
  2. Trim to 100K mcap onward (full trajectory, NOT just to ATH)
  3. Log transform + robust z-normalize per token
  4. Save individual trajectories as .npy files

Outputs:
  - data/trajectories/{address}.npy — per-token normalized trajectory
  - data/metadata.csv — token metadata (address, symbol, ath_mcap, total_hours, etc.)
"""

import csv
import json
import os
import sys
import glob

import numpy as np
import pandas as pd
from scipy.stats import iqr

from dna_extractor import extract_candle_series, extract_trend_series, build_dna_dataframe

TOKEN_LIST = "token_list.csv"
DATA_DIR = "data"
TRAJ_DIR = os.path.join(DATA_DIR, "trajectories")
MCAP_THRESHOLD = 100_000
DNA_COLUMNS = ["mcap", "holder_count", "top10_pct", "mcap_per_holder"]


def load_token_list():
    tokens = []
    with open(TOKEN_LIST, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                tokens.append({"address": row[0], "symbol": row[1], "name": row[2] if len(row) > 2 else row[1]})
    return tokens


def find_latest_json(token_dir, prefix):
    pattern = os.path.join(token_dir, f"{prefix}_*.json")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def load_token_data(address):
    token_dir = os.path.join(DATA_DIR, address)
    if not os.path.isdir(token_dir):
        return None, None
    candles_file = find_latest_json(token_dir, "token_mcap_candles")
    trends_file = find_latest_json(token_dir, "token_trends")
    if not candles_file or not trends_file:
        return None, None
    with open(candles_file, "r") as f:
        candles_data = json.load(f)
    with open(trends_file, "r") as f:
        trends_data = json.load(f)
    return candles_data, trends_data


def normalize_trajectory(values):
    """Log transform + robust z-normalize (median/IQR) per dimension."""
    log_vals = np.log1p(np.abs(values)) * np.sign(values)

    for d in range(log_vals.shape[1]):
        col = log_vals[:, d]
        med = np.median(col)
        q_iqr = iqr(col)
        if q_iqr > 0:
            log_vals[:, d] = (col - med) / q_iqr
        else:
            log_vals[:, d] = col - med

    return log_vals


def main():
    tokens = load_token_list()
    print(f"Loaded {len(tokens)} tokens")

    os.makedirs(TRAJ_DIR, exist_ok=True)
    metadata = []

    for token in tokens:
        address = token["address"]
        symbol = token["symbol"]

        candles_data, trends_data = load_token_data(address)
        if candles_data is None:
            continue

        candle_df = extract_candle_series(candles_data)
        trend_df = extract_trend_series(trends_data)
        if candle_df is None or trend_df is None:
            continue

        dna_df = build_dna_dataframe(candle_df, trend_df)
        if dna_df is None or len(dna_df) < 3:
            continue

        mcap = dna_df["mcap"].values
        above = np.where(mcap >= MCAP_THRESHOLD)[0]
        if len(above) == 0:
            continue
        start_idx = above[0]
        trimmed = dna_df.iloc[start_idx:].reset_index(drop=True)

        if len(trimmed) < 3:
            continue

        raw = trimmed[DNA_COLUMNS].values
        normalized = normalize_trajectory(raw)

        traj_path = os.path.join(TRAJ_DIR, f"{address}.npy")
        np.save(traj_path, normalized)

        ath_idx = np.argmax(raw[:, 0])
        metadata.append({
            "address": address,
            "symbol": symbol,
            "name": token["name"],
            "total_hours": len(trimmed),
            "ath_mcap": raw[ath_idx, 0],
            "ath_hour": int(ath_idx),
            "start_mcap": raw[0, 0],
            "holders_at_ath": raw[ath_idx, 1],
            "top10_pct_at_ath": raw[ath_idx, 2],
        })

    if not metadata:
        print("ERROR: No valid tokens", file=sys.stderr)
        sys.exit(1)

    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(os.path.join(DATA_DIR, "metadata.csv"), index=False)

    print(f"\n{'='*50}")
    print(f"  Preprocessing Summary")
    print(f"{'='*50}")
    print(f"  Total tokens:    {len(tokens)}")
    print(f"  Valid tokens:    {len(metadata)}")
    print(f"  Trajectories:    {TRAJ_DIR}/")
    print(f"  Hours range:     {meta_df['total_hours'].min()} — {meta_df['total_hours'].max()}")
    print(f"  Median hours:    {meta_df['total_hours'].median():.0f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
