"""Build path signature index from historical token trajectories.

For each historical token, compute the cumulative path signature at every
hour — so for a token with T hours of data, we store T signatures
(prefix of 3 hours, 4 hours, ..., T hours).

This allows matching a new token at ANY point in its lifecycle.

Reads:  data/trajectories/{address}.npy — per-token normalized trajectory
        data/metadata.csv
Writes: data/signature_index.pkl — {address: {k: signature_vector}} for all tokens and prefix lengths
"""

import os
import sys
import pickle

import numpy as np
import iisignature
import pandas as pd

DATA_DIR = "data"
TRAJ_DIR = os.path.join(DATA_DIR, "trajectories")
SIG_DEPTH = 3
MIN_PREFIX_LEN = 3  # minimum 3 hours to compute meaningful signature


def augment_path(trajectory):
    """Add normalized time as 5th dimension.

    Input:  (T, 4) — normalized DNA
    Output: (T, 5) — with time column prepended
    """
    T = len(trajectory)
    time_col = np.linspace(0, 1, T).reshape(-1, 1)
    return np.hstack([time_col, trajectory])


def compute_prefix_signatures(trajectory, depth=SIG_DEPTH, min_len=MIN_PREFIX_LEN):
    """Compute path signature for every prefix length >= min_len.

    Returns dict: {prefix_length: signature_vector}
    """
    augmented = augment_path(trajectory)
    T = len(augmented)
    signatures = {}

    for k in range(min_len, T + 1):
        prefix = augmented[:k]
        sig = iisignature.sig(prefix, depth)
        signatures[k] = sig

    return signatures


def build_index():
    """Build signature index for all historical tokens."""
    meta_path = os.path.join(DATA_DIR, "metadata.csv")
    if not os.path.isfile(meta_path):
        print("ERROR: data/metadata.csv not found. Run preprocess.py first.", file=sys.stderr)
        sys.exit(1)

    metadata = pd.read_csv(meta_path)
    print(f"Building signature index for {len(metadata)} tokens (depth={SIG_DEPTH})...")

    index = {}
    total_sigs = 0

    for i, row in metadata.iterrows():
        address = row["address"]
        symbol = row["symbol"]
        traj_path = os.path.join(TRAJ_DIR, f"{address}.npy")

        if not os.path.isfile(traj_path):
            continue

        trajectory = np.load(traj_path)
        sigs = compute_prefix_signatures(trajectory)
        index[address] = sigs
        total_sigs += len(sigs)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(metadata)}] processed...")

    # Save index
    index_path = os.path.join(DATA_DIR, "signature_index.pkl")
    with open(index_path, "wb") as f:
        pickle.dump(index, f)

    print(f"\nSignature index built:")
    print(f"  Tokens: {len(index)}")
    print(f"  Total signatures: {total_sigs}")
    print(f"  Saved: {index_path}")

    return index


if __name__ == "__main__":
    build_index()
