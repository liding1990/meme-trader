"""Query engine: find historical tokens most similar to a new token's trajectory.

Given a token address:
1. Fetch its current DNA data from GMGN API
2. Compute path signature of its trajectory from 100K onward
3. Find top-N historical tokens with most similar prefix signatures
4. Display their continuations (what happened after hour K)

Usage:
    python query.py <contract_address> [--top 5] [--chain sol]
"""

import argparse
import os
import sys
import pickle

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import iisignature
from gmgn_api import fetch_token_data
from dna_extractor import extract_candle_series, extract_trend_series, build_dna_dataframe
from preprocess import normalize_trajectory, MCAP_THRESHOLD, DNA_COLUMNS
from signature_index import augment_path, SIG_DEPTH, MIN_PREFIX_LEN, TRAJ_DIR

DATA_DIR = "data"


def load_index():
    index_path = os.path.join(DATA_DIR, "signature_index.pkl")
    if not os.path.isfile(index_path):
        print("ERROR: signature index not found. Run signature_index.py first.", file=sys.stderr)
        sys.exit(1)
    with open(index_path, "rb") as f:
        return pickle.load(f)


def fetch_new_token_trajectory(chain, address):
    """Fetch and preprocess a new token's trajectory."""
    _, loaded_data = fetch_token_data(chain, address)

    candle_df = extract_candle_series(loaded_data.get("token_mcap_candles", {}))
    trend_df = extract_trend_series(loaded_data.get("token_trends", {}))
    if candle_df is None or trend_df is None:
        return None, None

    dna_df = build_dna_dataframe(candle_df, trend_df)
    if dna_df is None or len(dna_df) < 3:
        return None, None

    # Trim from 100K
    mcap = dna_df["mcap"].values
    above = np.where(mcap >= MCAP_THRESHOLD)[0]
    if len(above) == 0:
        return None, None

    start_idx = above[0]
    trimmed = dna_df.iloc[start_idx:].reset_index(drop=True)

    raw = trimmed[DNA_COLUMNS].values
    normalized = normalize_trajectory(raw)

    return normalized, raw


def find_similar(new_trajectory, sig_index, metadata, top_n=5):
    """Find top-N historical tokens with most similar prefix signature."""
    K = len(new_trajectory)
    if K < MIN_PREFIX_LEN:
        print(f"ERROR: Token only has {K} hours of data (need >= {MIN_PREFIX_LEN})", file=sys.stderr)
        return []

    # Compute new token's signature
    augmented = augment_path(new_trajectory)
    new_sig = iisignature.sig(augmented, SIG_DEPTH)

    # Compare against all historical tokens at prefix length K
    candidates = []
    for address, sigs in sig_index.items():
        if K in sigs:
            hist_sig = sigs[K]
            dist = np.linalg.norm(new_sig - hist_sig)
            candidates.append((address, dist))

    # Sort by distance
    candidates.sort(key=lambda x: x[1])

    # Enrich with metadata
    meta_dict = {row["address"]: row for _, row in metadata.iterrows()}
    results = []
    for address, dist in candidates[:top_n]:
        meta = meta_dict.get(address, {})
        results.append({
            "address": address,
            "symbol": meta.get("symbol", "?"),
            "name": meta.get("name", "?"),
            "distance": dist,
            "ath_mcap": meta.get("ath_mcap", 0),
            "ath_hour": meta.get("ath_hour", 0),
            "total_hours": meta.get("total_hours", 0),
        })

    return results


def print_results(results, K):
    """Print match results."""
    print(f"\n{'='*70}")
    print(f"  Top {len(results)} Similar Tokens (matched at hour {K})")
    print(f"{'='*70}")
    for i, r in enumerate(results, 1):
        hours_left = r["total_hours"] - K
        ath_in = r["ath_hour"] - K if r["ath_hour"] > K else "already passed"
        print(f"\n  #{i} {r['symbol']} ({r['name']}) — distance: {r['distance']:.4f}")
        print(f"     ATH: ${r['ath_mcap']:,.0f} (at hour {r['ath_hour']})")
        print(f"     Total lifespan: {r['total_hours']}h")
        print(f"     Hours after current point: {hours_left}h")
        print(f"     ATH relative to now: {ath_in}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Find similar historical tokens")
    parser.add_argument("address", help="Contract address of the new token")
    parser.add_argument("--top", type=int, default=5, help="Number of matches to return")
    parser.add_argument("--chain", default="sol", help="Blockchain (default: sol)")
    args = parser.parse_args()

    # Load index and metadata
    sig_index = load_index()
    metadata = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))

    # Fetch new token data
    print(f"Fetching data for {args.address[:8]}...")
    normalized, raw = fetch_new_token_trajectory(args.chain, args.address)

    if normalized is None:
        print("ERROR: Could not fetch or process token data", file=sys.stderr)
        sys.exit(1)

    K = len(normalized)
    print(f"Token has {K} hours of data from 100K onward")
    print(f"Current mcap: ${raw[-1, 0]:,.0f}")

    # Find similar
    results = find_similar(normalized, sig_index, metadata, top_n=args.top)

    if not results:
        print("No matching historical tokens found at this prefix length")
        sys.exit(0)

    print_results(results, K)

    return results, K, raw


if __name__ == "__main__":
    main()
