"""Expand token dataset by discovering tokens from Codex and fetching GMGN data.

Stratified sampling across mcap tiers and weekly windows to build a diverse dataset
covering various market caps and momentum patterns.

Usage:
    python expand_dataset.py discover          # Step 1: Discover tokens from Codex
    python expand_dataset.py fetch             # Step 2: Fetch GMGN 1h data
    python expand_dataset.py rebuild           # Step 3: Rebuild signatures & clusters
    python expand_dataset.py status            # Show dataset status
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from codex_api import _query, SOLANA_NETWORK_ID, REQUEST_INTERVAL

DATA_DIR = "data"
TOKEN_LIST = "token_list.csv"
DISCOVERY_LOG = os.path.join(DATA_DIR, "discovery_log.json")

# Stratified sampling tiers: (min_mcap, max_mcap, min_holders, target_per_week)
TIERS = [
    {"label": "small",  "min_mcap": 50_000,    "max_mcap": 500_000,   "min_holders": 300,  "target": 40},
    {"label": "mid",    "min_mcap": 500_000,   "max_mcap": 2_000_000, "min_holders": 500,  "target": 35},
    {"label": "large",  "min_mcap": 2_000_000, "max_mcap": 10_000_000,"min_holders": 500,  "target": 25},
    {"label": "mega",   "min_mcap": 10_000_000,"max_mcap": None,      "min_holders": 500,  "target": 20},
]

N_WEEKS = 8  # 2 months


def load_existing_addresses():
    existing = set()
    if os.path.isfile(TOKEN_LIST):
        with open(TOKEN_LIST, "r") as f:
            for row in csv.reader(f):
                if row:
                    existing.add(row[0])
    return existing


def _fetch_tier(start_ts, end_ts, min_mcap, max_mcap, min_holders, limit=200):
    """Fetch tokens from Codex for a single tier and time window."""
    filter_parts = [
        f"network: [{SOLANA_NETWORK_ID}]",
        f"marketCap: {{gte: {min_mcap}}}",
        f"holders: {{gte: {min_holders}}}",
        f"createdAt: {{gte: {start_ts}, lt: {end_ts}}}",
    ]
    filters_str = ", ".join(filter_parts)

    all_tokens = []
    offset = 0

    while True:
        query = f"""
        {{
          filterTokens(
            filters: {{{filters_str}}}
            limit: {limit}
            offset: {offset}
            rankings: {{attribute: createdAt, direction: DESC}}
          ) {{
            results {{
              token {{ address name symbol createdAt }}
              marketCap
              holders
              liquidity
            }}
            count
          }}
        }}
        """
        result = _query(query)
        ft = result.get("filterTokens", {})
        tokens = ft.get("results", [])
        all_tokens.extend(tokens)

        if len(tokens) < limit:
            break
        offset += limit
        time.sleep(REQUEST_INTERVAL * 2)

    # Filter by max_mcap if specified
    if max_mcap is not None:
        all_tokens = [
            t for t in all_tokens
            if t.get("marketCap") is not None and float(t["marketCap"]) < max_mcap
        ]

    return all_tokens


def discover(target_total=700):
    """Discover tokens from Codex across mcap tiers and weekly windows."""
    existing = load_existing_addresses()
    print(f"Existing tokens in {TOKEN_LIST}: {len(existing)}")

    now = datetime.now(timezone.utc)
    last_monday = now - timedelta(days=now.weekday())
    last_monday = last_monday.replace(hour=0, minute=0, second=0, microsecond=0)

    target_new = max(0, target_total - len(existing))
    print(f"Target: {target_total} total, need ~{target_new} new tokens\n")

    all_discovered = {}
    tier_counts = {t["label"]: 0 for t in TIERS}

    for week_idx in range(N_WEEKS):
        end_dt = last_monday - timedelta(weeks=week_idx)
        start_dt = end_dt - timedelta(weeks=1)
        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())
        week_label = start_dt.strftime("%m/%d")

        print(f"Week of {week_label}:")

        for tier in TIERS:
            time.sleep(REQUEST_INTERVAL * 2)

            try:
                tokens = _fetch_tier(
                    start_ts, end_ts,
                    tier["min_mcap"], tier["max_mcap"],
                    tier["min_holders"],
                )
            except Exception as e:
                print(f"  {tier['label']:>6s}: ERROR {e}")
                continue

            # Filter out existing and already discovered
            new_tokens = []
            for t in tokens:
                addr = t["token"]["address"]
                if addr not in existing and addr not in all_discovered:
                    new_tokens.append(t)

            # Sample up to target per tier per week
            if len(new_tokens) > tier["target"]:
                indices = np.linspace(0, len(new_tokens) - 1, tier["target"], dtype=int)
                new_tokens = [new_tokens[i] for i in indices]

            for t in new_tokens:
                addr = t["token"]["address"]
                all_discovered[addr] = {
                    "address": addr,
                    "symbol": t["token"]["symbol"],
                    "name": t["token"]["name"],
                    "created_at": t["token"].get("createdAt"),
                    "mcap": t.get("marketCap"),
                    "holders": t.get("holders"),
                    "tier": tier["label"],
                    "week": week_label,
                }
                tier_counts[tier["label"]] += 1

            print(f"  {tier['label']:>6s}: {len(tokens):>4d} found, {len(new_tokens):>3d} selected")

        # Check progress
        print(f"  → total new so far: {len(all_discovered)}")
        if len(all_discovered) >= target_new:
            print(f"\nReached target ({len(all_discovered)} new tokens)")
            break

    # Summary
    print(f"\n{'='*60}")
    print(f"  Discovery Summary")
    print(f"{'='*60}")
    print(f"  New tokens discovered: {len(all_discovered)}")
    for label, count in tier_counts.items():
        print(f"    {label:>6s}: {count}")
    print(f"  Existing tokens: {len(existing)}")
    print(f"  Total after merge: {len(existing) + len(all_discovered)}")
    print(f"{'='*60}")

    # Append to token_list.csv
    if all_discovered:
        with open(TOKEN_LIST, "a", newline="") as f:
            writer = csv.writer(f)
            for addr, info in all_discovered.items():
                writer.writerow([addr, info["symbol"], info["name"]])
        print(f"Appended {len(all_discovered)} tokens to {TOKEN_LIST}")

    # Save discovery log
    with open(DISCOVERY_LOG, "w") as f:
        json.dump(all_discovered, f, indent=2)
    print(f"Discovery log: {DISCOVERY_LOG}")

    return all_discovered


def fetch_gmgn_data():
    """Fetch GMGN 1h candle + trend data for all tokens in token_list.csv."""
    from gmgn_api import fetch_token_data
    import glob

    tokens = []
    with open(TOKEN_LIST, "r") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                tokens.append({"address": row[0], "symbol": row[1]})

    total = len(tokens)
    fetched = 0
    skipped = 0
    failed = 0
    failed_addrs = []

    for i, token in enumerate(tokens, 1):
        address = token["address"]
        symbol = token["symbol"]
        token_dir = os.path.join(DATA_DIR, address)

        # Skip if already has both candle and trend data
        if os.path.isdir(token_dir):
            candle_files = [
                f for f in glob.glob(os.path.join(token_dir, "token_mcap_candles_[0-9]*.json"))
                if "5m" not in os.path.basename(f)
            ]
            trend_files = glob.glob(os.path.join(token_dir, "token_trends_*.json"))
            if candle_files and trend_files:
                skipped += 1
                continue

        print(f"[{i}/{total}] {symbol:>12s}...", end=" ", flush=True)

        try:
            _, loaded = fetch_token_data("sol", address)

            candles = loaded.get("token_mcap_candles", {}).get("data", {}).get("list", [])
            trends = loaded.get("token_trends", {}).get("data", {}).get("trends", {}).get("holder_count", [])

            if not candles or not trends:
                print(f"incomplete (candles={len(candles)}, trends={len(trends)})")
                failed += 1
                failed_addrs.append(address)
            else:
                print(f"OK ({len(candles)} candles, {len(trends)} trend pts)")
                fetched += 1

        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1
            failed_addrs.append(address)

    print(f"\n{'='*60}")
    print(f"  Fetch Summary")
    print(f"{'='*60}")
    print(f"  Total tokens:     {total}")
    print(f"  Fetched:          {fetched}")
    print(f"  Skipped (cached): {skipped}")
    print(f"  Failed:           {failed}")
    print(f"{'='*60}")

    if failed_addrs:
        failed_path = os.path.join(DATA_DIR, "fetch_failed.txt")
        with open(failed_path, "w") as f:
            f.write("\n".join(failed_addrs))
        print(f"Failed addresses saved to {failed_path}")


def rebuild_pipeline():
    """Run the full rebuild: preprocess → signature_index → sig_cluster."""
    import subprocess

    steps = [
        ("Preprocessing trajectories", ["python", "preprocess.py"]),
        ("Building signature index", ["python", "signature_index.py"]),
        ("Running signature clustering", ["python", "sig_cluster.py"]),
    ]

    for desc, cmd in steps:
        print(f"\n{'='*60}")
        print(f"  {desc}...")
        print(f"{'='*60}")
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"ERROR: {desc} failed with exit code {result.returncode}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Pipeline rebuild complete!")
    print(f"{'='*60}")


def show_status():
    """Show current dataset status."""
    import glob

    n_tokens = 0
    if os.path.isfile(TOKEN_LIST):
        with open(TOKEN_LIST) as f:
            n_tokens = sum(1 for _ in f)

    n_with_both = 0
    for d in os.listdir(DATA_DIR):
        full = os.path.join(DATA_DIR, d)
        if not os.path.isdir(full) or d in ("codex", "trajectories"):
            continue
        files = os.listdir(full)
        has_candles = any(
            f.startswith("token_mcap_candles_") and "5m" not in f for f in files
        )
        has_trends = any(f.startswith("token_trends_") for f in files)
        if has_candles and has_trends:
            n_with_both += 1

    traj_dir = os.path.join(DATA_DIR, "trajectories")
    n_trajs = 0
    if os.path.isdir(traj_dir):
        n_trajs = len([f for f in os.listdir(traj_dir) if f.endswith(".npy")])

    cluster_file = os.path.join(DATA_DIR, "sig_clusters.csv")
    n_clustered = 0
    n_clusters = 0
    if os.path.isfile(cluster_file):
        df = pd.read_csv(cluster_file)
        n_clustered = len(df)
        n_clusters = df["cluster"].nunique() - (1 if -1 in df["cluster"].values else 0)

    meta_file = os.path.join(DATA_DIR, "metadata.csv")
    n_meta = 0
    if os.path.isfile(meta_file):
        n_meta = len(pd.read_csv(meta_file))

    print(f"\n{'='*60}")
    print(f"  Dataset Status")
    print(f"{'='*60}")
    print(f"  Token list:        {n_tokens}")
    print(f"  With GMGN data:    {n_with_both} (candles + trends)")
    print(f"  Trajectories:      {n_trajs} (.npy files)")
    print(f"  Metadata:          {n_meta}")
    print(f"  Clustered:         {n_clustered} ({n_clusters} clusters)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Expand token dataset for clustering")
    sub = parser.add_subparsers(dest="command")

    d = sub.add_parser("discover", help="Discover tokens from Codex")
    d.add_argument("--target", type=int, default=700, help="Target total token count")

    sub.add_parser("fetch", help="Fetch GMGN 1h data for all tokens")
    sub.add_parser("rebuild", help="Rebuild preprocess → signatures → clusters")
    sub.add_parser("status", help="Show dataset status")

    args = parser.parse_args()

    if args.command == "discover":
        discover(target_total=args.target)
    elif args.command == "fetch":
        fetch_gmgn_data()
    elif args.command == "rebuild":
        rebuild_pipeline()
    elif args.command == "status":
        show_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
