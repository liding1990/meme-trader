"""Expand the token dataset by fetching token lists from GMGN ranking APIs.

Fetches tokens from multiple ranking endpoints (trending, new, volume leaders)
across different time windows, deduplicates, then batch-fetches their data.

Usage:
    python expand_dataset.py discover   # discover new token addresses
    python expand_dataset.py fetch      # batch fetch data for all discovered tokens
"""

import csv
import json
import os
import sys
import time

from gmgn_api import _build_url, _curl_get, _COMMON_PARAMS, _GMGN_BASE_URL, fetch_token_data
from urllib.parse import urlencode

DATA_DIR = "data"
TOKEN_LIST = "token_list.csv"
REQUEST_INTERVAL = 2


def fetch_ranking(orderby="marketcap", direction="desc", time_window="1h", limit=50):
    """Fetch token ranking from GMGN."""
    common = urlencode(_COMMON_PARAMS)
    url = (f"{_GMGN_BASE_URL}/defi/quotation/v1/rank/sol/swaps/{time_window}"
           f"?orderby={orderby}&direction={direction}&limit={limit}&{common}")
    body = _curl_get(url)
    data = json.loads(body)
    tokens = data.get("data", {}).get("rank", [])
    return tokens


def discover_tokens():
    """Discover new token addresses from multiple GMGN ranking endpoints."""
    # Load existing tokens
    existing = set()
    if os.path.isfile(TOKEN_LIST):
        with open(TOKEN_LIST, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    existing.add(row[0])

    print(f"Existing tokens: {len(existing)}")

    # Define queries to explore
    queries = [
        {"orderby": "marketcap", "direction": "desc", "time_window": "1h", "limit": 100},
        {"orderby": "marketcap", "direction": "desc", "time_window": "6h", "limit": 100},
        {"orderby": "marketcap", "direction": "desc", "time_window": "24h", "limit": 100},
        {"orderby": "volume", "direction": "desc", "time_window": "1h", "limit": 100},
        {"orderby": "volume", "direction": "desc", "time_window": "6h", "limit": 100},
        {"orderby": "volume", "direction": "desc", "time_window": "24h", "limit": 100},
        {"orderby": "swaps", "direction": "desc", "time_window": "1h", "limit": 100},
        {"orderby": "swaps", "direction": "desc", "time_window": "6h", "limit": 100},
        {"orderby": "swaps", "direction": "desc", "time_window": "24h", "limit": 100},
        {"orderby": "change", "direction": "desc", "time_window": "1h", "limit": 100},
        {"orderby": "change", "direction": "desc", "time_window": "6h", "limit": 100},
        {"orderby": "change", "direction": "desc", "time_window": "24h", "limit": 100},
    ]

    new_tokens = {}  # address → {symbol, name}

    for i, q in enumerate(queries):
        desc = f"{q['orderby']}/{q['time_window']}"
        print(f"[{i+1}/{len(queries)}] Fetching {desc}...", end=" ")

        try:
            tokens = fetch_ranking(**q)
            found = 0
            for t in tokens:
                addr = t.get("address", "")
                if addr and addr not in existing and addr not in new_tokens:
                    new_tokens[addr] = {
                        "symbol": t.get("symbol", "???"),
                        "name": t.get("name", t.get("symbol", "???")),
                    }
                    found += 1
            print(f"{len(tokens)} results, {found} new")
        except Exception as e:
            print(f"ERROR: {e}")

        time.sleep(REQUEST_INTERVAL)

    print(f"\nDiscovered {len(new_tokens)} new tokens")

    # Append to token_list.csv
    if new_tokens:
        with open(TOKEN_LIST, "a", newline="") as f:
            writer = csv.writer(f)
            for addr, info in new_tokens.items():
                writer.writerow([addr, info["symbol"], info["name"]])
        print(f"Appended to {TOKEN_LIST}")

    # Report totals
    total = len(existing) + len(new_tokens)
    print(f"Total tokens now: {total}")

    return new_tokens


def batch_fetch_new():
    """Fetch GMGN data for all tokens that don't have cached data yet."""
    tokens = []
    with open(TOKEN_LIST, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                tokens.append({"address": row[0], "symbol": row[1]})

    total = len(tokens)
    fetched = 0
    skipped = 0
    failed = 0

    for i, token in enumerate(tokens, 1):
        address = token["address"]
        symbol = token["symbol"]

        # Check if already has data
        token_dir = os.path.join(DATA_DIR, address)
        if os.path.isdir(token_dir):
            files = os.listdir(token_dir)
            has_data = (any(f.startswith("token_trends_") for f in files) and
                       any(f.startswith("token_mcap_candles_") for f in files))
            if has_data:
                skipped += 1
                continue

        print(f"[{i}/{total}] Fetching {symbol} ({address[:8]}...)...")

        try:
            fetch_token_data("sol", address)
            fetched += 1
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            failed += 1

        time.sleep(REQUEST_INTERVAL)

    print(f"\nDone: {fetched} fetched, {skipped} cached, {failed} failed (total: {total})")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["discover", "fetch"])
    args = parser.parse_args()

    if args.command == "discover":
        discover_tokens()
    elif args.command == "fetch":
        batch_fetch_new()


if __name__ == "__main__":
    main()
