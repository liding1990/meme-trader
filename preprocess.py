"""Preprocess token DNA data: extract main rally, resample, z-score standardize.

Reads raw JSON from data/{address}/, outputs:
  - data/dataset.npy  — shape (N, 200, 4)
  - data/metadata.csv — per-token info
"""

import csv
import json
import os
import sys
import glob

import numpy as np
import pandas as pd

from dna_extractor import extract_candle_series, extract_trend_series, build_dna_dataframe

TOKEN_LIST = "token_list.csv"
DATA_DIR = "data"
RESAMPLE_POINTS = 200
MCAP_THRESHOLD = 100_000  # $100K minimum mcap
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
    """Find the most recent JSON file with given prefix in token_dir."""
    pattern = os.path.join(token_dir, f"{prefix}_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return files[-1]


def load_token_data(address):
    """Load raw JSON data for a token. Returns (candles_data, trends_data) or (None, None)."""
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


def extract_main_rally(dna_df):
    """Extract the main rally segment: first mcap >= 100K to ATH.

    Returns the sliced DataFrame, or None if criteria not met.
    """
    mcap = dna_df["mcap"].values

    # Find start: first hour where mcap >= 100K
    above_threshold = np.where(mcap >= MCAP_THRESHOLD)[0]
    if len(above_threshold) == 0:
        return None
    start_idx = above_threshold[0]

    # Find end: ATH (global max) after start
    ath_idx = start_idx + np.argmax(mcap[start_idx:])

    # Discard if start == end (instant ATH, no rally)
    if ath_idx <= start_idx:
        return None

    return dna_df.iloc[start_idx:ath_idx + 1].reset_index(drop=True)


def resample_series(df, n_points):
    """Resample a DataFrame to exactly n_points using linear interpolation."""
    n_orig = len(df)
    if n_orig == n_points:
        return df[DNA_COLUMNS].values

    orig_indices = np.linspace(0, 1, n_orig)
    new_indices = np.linspace(0, 1, n_points)

    result = np.zeros((n_points, len(DNA_COLUMNS)))
    for i, col in enumerate(DNA_COLUMNS):
        result[:, i] = np.interp(new_indices, orig_indices, df[col].values)

    return result


def main():
    tokens = load_token_list()
    print(f"Loaded {len(tokens)} tokens")

    samples = []    # list of (200, 4) arrays
    metadata = []   # list of metadata dicts

    for i, token in enumerate(tokens):
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

        rally_df = extract_main_rally(dna_df)
        if rally_df is None:
            continue

        # Need at least 2 points to resample
        if len(rally_df) < 2:
            continue

        resampled = resample_series(rally_df, RESAMPLE_POINTS)
        samples.append(resampled)

        ath_row = rally_df.iloc[-1]
        start_row = rally_df.iloc[0]
        metadata.append({
            "address": address,
            "symbol": symbol,
            "name": token["name"],
            "ath_mcap": ath_row["mcap"],
            "start_mcap": start_row["mcap"],
            "rally_duration_hours": len(rally_df),
            "holder_count_at_ath": ath_row["holder_count"],
            "top10_pct_at_ath": ath_row["top10_pct"],
        })

    if not samples:
        print("ERROR: No valid samples after preprocessing", file=sys.stderr)
        sys.exit(1)

    # Stack into (N, 200, 4)
    dataset = np.stack(samples, axis=0)
    print(f"Dataset shape before z-score: {dataset.shape}")

    # Z-score standardization (global, per dimension)
    mean = dataset.mean(axis=(0, 1))  # shape (4,)
    std = dataset.std(axis=(0, 1))    # shape (4,)
    std[std == 0] = 1  # avoid division by zero
    dataset = (dataset - mean) / std

    # Save normalization params for later use
    np.save(os.path.join(DATA_DIR, "zscore_params.npy"), np.stack([mean, std]))

    # Save dataset
    np.save(os.path.join(DATA_DIR, "dataset.npy"), dataset)
    print(f"Saved dataset: {dataset.shape} → data/dataset.npy")

    # Save metadata
    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(os.path.join(DATA_DIR, "metadata.csv"), index=False)
    print(f"Saved metadata: {len(metadata)} tokens → data/metadata.csv")

    # Summary
    print(f"\n{'='*50}")
    print(f"  Preprocessing Summary")
    print(f"{'='*50}")
    print(f"  Total tokens:     {len(tokens)}")
    print(f"  Valid samples:    {len(samples)}")
    print(f"  Discarded:        {len(tokens) - len(samples)}")
    print(f"  Dataset shape:    {dataset.shape}")
    print(f"  Z-score mean:     {mean}")
    print(f"  Z-score std:      {std}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
