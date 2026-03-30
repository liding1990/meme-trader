"""Fetch 5-minute candle data (1000 bars) + buy/sell volume snapshots.

Fetches:
  1. 5m candles with limit=1000 (~3.7 days of history)
  2. price_info with buy/sell volume breakdowns per time window

Usage:
    python fetch_5m.py              # fetch new tokens only
    python fetch_5m.py --refresh    # re-fetch all (upgrade from 400 to 1000 candles)
"""

import argparse
import csv
import os
import sys
import time
import json

from gmgn_api import _curl_get, _COMMON_PARAMS, _GMGN_BASE_URL
from urllib.parse import urlencode

DATA_DIR = "data"
TOKEN_LIST = "token_list.csv"
REQUEST_INTERVAL = 2
CANDLE_LIMIT = 1000  # ~3.7 days of 5m data


def fetch_5m_candles(chain, address):
    """Fetch 5m candles (1000 bars) for a single token."""
    common = urlencode(_COMMON_PARAMS)
    params = urlencode({"resolution": "5m", "limit": str(CANDLE_LIMIT), "pool_type": "unified"})
    url = f"{_GMGN_BASE_URL}/api/v1/token_mcap_candles/{chain}/{address}?{common}&{params}"

    body = _curl_get(url)
    if body.startswith("<!DOCTYPE") or "Cloudflare" in body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def fetch_price_info(chain, address):
    """Fetch buy/sell volume snapshot from price_info endpoint."""
    common = urlencode(_COMMON_PARAMS)
    url = f"{_GMGN_BASE_URL}/api/v1/token_price_info/{chain}/{address}?chain={chain}&{common}"

    body = _curl_get(url)
    if body.startswith("<!DOCTYPE") or "Cloudflare" in body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def should_refresh(token_dir):
    """Check if existing 5m data has fewer than CANDLE_LIMIT candles."""
    files = [f for f in os.listdir(token_dir) if f.startswith("token_mcap_candles_5m_")]
    if not files:
        return True
    # Check the latest file's candle count
    latest = sorted(files)[-1]
    try:
        with open(os.path.join(token_dir, latest)) as f:
            data = json.load(f)
        count = len(data.get("data", {}).get("list", []))
        return count < CANDLE_LIMIT * 0.9  # refresh if <90% of target
    except Exception:
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Re-fetch all tokens (upgrade to 1000 candles)")
    args = parser.parse_args()

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
        token_dir = os.path.join(DATA_DIR, address)

        if not os.path.isdir(token_dir):
            os.makedirs(token_dir, exist_ok=True)

        # Skip if already has good data (unless --refresh)
        if not args.refresh and not should_refresh(token_dir):
            skipped += 1
            continue

        print(f"[{i}/{total}] {symbol}...", end=" ", flush=True)

        try:
            # Fetch 5m candles
            candle_data = fetch_5m_candles("sol", address)
            if candle_data is None:
                print("BLOCKED")
                failed += 1
                time.sleep(REQUEST_INTERVAL)
                continue

            ts = int(time.time() * 1000)
            candle_path = os.path.join(token_dir, f"token_mcap_candles_5m_{ts}.json")
            with open(candle_path, "w") as f:
                json.dump(candle_data, f, indent=2, ensure_ascii=False)
            n_candles = len(candle_data.get("data", {}).get("list", []))

            time.sleep(1)  # brief pause between requests

            # Fetch price_info (buy/sell volumes)
            price_data = fetch_price_info("sol", address)
            if price_data:
                price_path = os.path.join(token_dir, f"token_price_info_{ts}.json")
                with open(price_path, "w") as f:
                    json.dump(price_data, f, indent=2, ensure_ascii=False)

            print(f"OK ({n_candles} candles + price_info)")
            fetched += 1

        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

        time.sleep(REQUEST_INTERVAL)

    print(f"\nDone: {fetched} fetched, {skipped} skipped, {failed} failed (total: {total})")


if __name__ == "__main__":
    main()
