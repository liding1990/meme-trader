"""Batch fetch GMGN data for all tokens in token_list.csv."""

import csv
import os
import sys
import time

from gmgn_api import fetch_token_data

TOKEN_LIST = "token_list.csv"
DATA_DIR = "data"
REQUEST_INTERVAL = 2  # seconds between requests


def load_token_list():
    tokens = []
    with open(TOKEN_LIST, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                tokens.append({"address": row[0], "symbol": row[1], "name": row[2] if len(row) > 2 else row[1]})
    return tokens


def has_cached_data(address):
    """Check if token already has both required data files."""
    token_dir = os.path.join(DATA_DIR, address)
    if not os.path.isdir(token_dir):
        return False
    files = os.listdir(token_dir)
    has_trends = any(f.startswith("token_trends_") for f in files)
    has_candles = any(f.startswith("token_mcap_candles_") for f in files)
    return has_trends and has_candles


def main():
    tokens = load_token_list()
    total = len(tokens)
    print(f"Loaded {total} tokens from {TOKEN_LIST}")

    skipped = 0
    fetched = 0
    failed = 0

    for i, token in enumerate(tokens, 1):
        address = token["address"]
        symbol = token["symbol"]

        if has_cached_data(address):
            skipped += 1
            print(f"[{i}/{total}] {symbol} — cached, skipping")
            continue

        print(f"[{i}/{total}] Fetching {symbol} ({address[:8]}...)...")

        try:
            fetch_token_data("sol", address)
            fetched += 1
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            failed += 1

        if i < total:
            time.sleep(REQUEST_INTERVAL)

    print(f"\nDone: {fetched} fetched, {skipped} cached, {failed} failed")


if __name__ == "__main__":
    main()
