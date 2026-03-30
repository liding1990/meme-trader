"""Discover Solana memecoins by weekly window using Codex API.

For each week going back N weeks, finds tokens that:
  - Were created in that week
  - Reached market cap >= 100K at some point
  - Had holders >= 1000 at some point

Then fetches 5m candle data (with buy/sell volume) for each via Codex.

Usage:
    python discover_tokens.py scan --weeks 8     # scan last 8 weeks
    python discover_tokens.py fetch              # fetch 5m data for discovered tokens
    python discover_tokens.py status             # show dataset status
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from codex_api import filter_tokens, get_token_bars, REQUEST_INTERVAL

DATA_DIR = "data"
TOKEN_LIST = "token_list.csv"
CODEX_DATA_DIR = os.path.join(DATA_DIR, "codex")


def get_weekly_windows(n_weeks=8):
    """Generate weekly time windows going back n_weeks from now."""
    now = datetime.now(timezone.utc)
    # Start from last Monday
    last_monday = now - timedelta(days=now.weekday())
    last_monday = last_monday.replace(hour=0, minute=0, second=0, microsecond=0)

    windows = []
    for i in range(n_weeks):
        end = last_monday - timedelta(weeks=i)
        start = end - timedelta(weeks=1)
        windows.append({
            "start": int(start.timestamp()),
            "end": int(end.timestamp()),
            "label": start.strftime("%Y-%m-%d"),
        })

    return windows


def load_existing_addresses():
    """Load all known token addresses."""
    existing = set()
    if os.path.isfile(TOKEN_LIST):
        with open(TOKEN_LIST, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    existing.add(row[0])
    return existing


def scan_weekly(n_weeks=8, min_mcap=100_000, min_holders=1000):
    """Scan Codex for tokens created in each weekly window."""
    windows = get_weekly_windows(n_weeks)
    existing = load_existing_addresses()

    print(f"Scanning {n_weeks} weekly windows (mcap>=${min_mcap:,}, holders>={min_holders:,})")
    print(f"Existing tokens: {len(existing)}")

    all_new = {}
    total_found = 0

    for w in windows:
        print(f"\n  Week of {w['label']}...", end=" ", flush=True)

        try:
            tokens, count, _ = filter_tokens(
                created_after=w["start"],
                created_before=w["end"],
                min_mcap=min_mcap,
                min_holders=min_holders,
                limit=200,
            )
            total_found += count

            new_count = 0
            for t in tokens:
                addr = t["token"]["address"]
                if addr not in existing and addr not in all_new:
                    all_new[addr] = {
                        "symbol": t["token"]["symbol"],
                        "name": t["token"]["name"],
                        "created_at": t["token"]["createdAt"],
                        "mcap": t.get("marketCap", 0),
                        "holders": t.get("holders", 0),
                    }
                    new_count += 1

            print(f"{count} total, {len(tokens)} returned, {new_count} new")

        except Exception as e:
            print(f"ERROR: {e}")

        time.sleep(REQUEST_INTERVAL * 2)

    print(f"\n{'='*60}")
    print(f"  Total found across all weeks: {total_found}")
    print(f"  New tokens discovered: {len(all_new)}")
    print(f"{'='*60}")

    # Append to token_list.csv
    if all_new:
        with open(TOKEN_LIST, "a", newline="") as f:
            writer = csv.writer(f)
            for addr, info in all_new.items():
                writer.writerow([addr, info["symbol"], info["name"]])
        print(f"Appended {len(all_new)} tokens to {TOKEN_LIST}")

    # Save detailed discovery info
    os.makedirs(CODEX_DATA_DIR, exist_ok=True)
    discovery_path = os.path.join(CODEX_DATA_DIR, "discovered_tokens.json")
    with open(discovery_path, "w") as f:
        json.dump(all_new, f, indent=2)
    print(f"Discovery details: {discovery_path}")

    return all_new


def fetch_codex_data(limit_tokens=None):
    """Fetch 5m candle data (with buy/sell volume) from Codex for all tokens."""
    tokens = []
    with open(TOKEN_LIST, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                tokens.append({"address": row[0], "symbol": row[1]})

    os.makedirs(CODEX_DATA_DIR, exist_ok=True)

    total = len(tokens)
    if limit_tokens:
        tokens = tokens[:limit_tokens]
        total = len(tokens)

    fetched = 0
    skipped = 0
    failed = 0

    for i, token in enumerate(tokens, 1):
        address = token["address"]
        symbol = token["symbol"]

        # Check if already fetched from Codex
        codex_file = os.path.join(CODEX_DATA_DIR, f"{address}_5m.json")
        if os.path.isfile(codex_file):
            skipped += 1
            continue

        print(f"[{i}/{total}] {symbol}...", end=" ", flush=True)

        try:
            bars = get_token_bars(address, resolution="5", countback=1000)

            if not bars:
                print("no data")
                failed += 1
            else:
                with open(codex_file, "w") as f:
                    json.dump(bars, f)
                print(f"OK ({len(bars)} bars)")
                fetched += 1

        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

        time.sleep(REQUEST_INTERVAL)

    print(f"\nDone: {fetched} fetched, {skipped} cached, {failed} failed")


def show_status():
    """Show dataset status."""
    # Count token_list
    n_tokens = 0
    if os.path.isfile(TOKEN_LIST):
        with open(TOKEN_LIST) as f:
            n_tokens = sum(1 for _ in f)

    # Count GMGN data
    n_gmgn = 0
    n_gmgn_5m = 0
    for d in os.listdir(DATA_DIR):
        full = os.path.join(DATA_DIR, d)
        if not os.path.isdir(full) or d == "codex" or d == "trajectories":
            continue
        files = os.listdir(full)
        if any(f.startswith("token_mcap_candles_") and not f.startswith("token_mcap_candles_5m_") for f in files):
            n_gmgn += 1
        if any(f.startswith("token_mcap_candles_5m_") for f in files):
            n_gmgn_5m += 1

    # Count Codex data
    n_codex = 0
    if os.path.isdir(CODEX_DATA_DIR):
        n_codex = len([f for f in os.listdir(CODEX_DATA_DIR) if f.endswith("_5m.json")])

    print(f"\n{'='*50}")
    print(f"  Dataset Status")
    print(f"{'='*50}")
    print(f"  Token list:     {n_tokens}")
    print(f"  GMGN 1h data:   {n_gmgn}")
    print(f"  GMGN 5m data:   {n_gmgn_5m}")
    print(f"  Codex 5m data:  {n_codex} (with buy/sell volume)")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Discover & fetch Solana memecoin data via Codex")
    sub = parser.add_subparsers(dest="command")

    s = sub.add_parser("scan", help="Scan for tokens by weekly window")
    s.add_argument("--weeks", type=int, default=8, help="Number of weeks to scan back")
    s.add_argument("--min-mcap", type=int, default=100000)
    s.add_argument("--min-holders", type=int, default=1000)

    f = sub.add_parser("fetch", help="Fetch 5m candle data from Codex")
    f.add_argument("--limit", type=int, default=None, help="Max tokens to fetch")

    sub.add_parser("status", help="Show dataset status")

    args = parser.parse_args()

    if args.command == "scan":
        scan_weekly(n_weeks=args.weeks, min_mcap=args.min_mcap, min_holders=args.min_holders)
    elif args.command == "fetch":
        fetch_codex_data(limit_tokens=args.limit)
    elif args.command == "status":
        show_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
