"""Fetch 5-minute candle data for all tokens that don't have it yet."""

import csv
import os
import sys
import time
import json

from gmgn_api import _build_url, _curl_get, _COMMON_PARAMS, _GMGN_BASE_URL
from urllib.parse import urlencode

DATA_DIR = "data"
TOKEN_LIST = "token_list.csv"
REQUEST_INTERVAL = 2


def fetch_5m_candles(chain, address):
    """Fetch 5m candles for a single token."""
    common = urlencode(_COMMON_PARAMS)
    params = urlencode({"resolution": "5m", "limit": "400", "pool_type": "unified"})
    url = f"{_GMGN_BASE_URL}/api/v1/token_mcap_candles/{chain}/{address}?{common}&{params}"

    body = _curl_get(url)
    if body.startswith("<!DOCTYPE") or "Cloudflare" in body:
        return None

    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def main():
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
            skipped += 1
            continue

        # Check if already has 5m data
        existing = [f for f in os.listdir(token_dir) if f.startswith("token_mcap_candles_5m_")]
        if existing:
            skipped += 1
            continue

        print(f"[{i}/{total}] Fetching 5m candles for {symbol}...", end=" ")

        try:
            data = fetch_5m_candles("sol", address)
            if data is None:
                print("BLOCKED/ERROR")
                failed += 1
            else:
                ts = int(time.time() * 1000)
                filepath = os.path.join(token_dir, f"token_mcap_candles_5m_{ts}.json")
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                candles = data.get("data", {}).get("list", [])
                print(f"OK ({len(candles)} candles)")
                fetched += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

        time.sleep(REQUEST_INTERVAL)

    print(f"\nDone: {fetched} fetched, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
