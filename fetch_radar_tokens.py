"""Batch fetch GMGN data for all radar tokens missing local cache."""

import csv
import os
import sys
import time

from gmgn_api import fetch_token_data

DATA_DIR = "data"


def get_radar_tokens(csv_path="data/radar_tokens.csv"):
    tokens = []
    with open(csv_path) as f:
        for row in csv.reader(f):
            if len(row) >= 4:
                chain = row[1]
                # Map chain names to GMGN API chain param
                if chain == "solana":
                    chain = "sol"
                elif chain == "base":
                    chain = "base"
                tokens.append({"address": row[0], "chain": chain, "name": row[2], "symbol": row[3]})
    return tokens


def has_cache(address):
    d = os.path.join(DATA_DIR, address)
    if not os.path.isdir(d):
        return False
    files = os.listdir(d)
    has_mcap = any(f.startswith("token_mcap_candles_") and "5m" not in f for f in files)
    has_trends = any(f.startswith("token_trends_") for f in files)
    return has_mcap and has_trends


def main():
    tokens = get_radar_tokens()
    missing = [t for t in tokens if not has_cache(t["address"])]
    print(f"Total radar tokens: {len(tokens)}")
    print(f"Missing cache: {len(missing)}")

    for i, t in enumerate(missing):
        addr = t["address"]
        sym = t["symbol"]
        chain = t["chain"]

        print(f"[{i+1}/{len(missing)}] Fetching {sym} ({addr[:8]}...)...", end=" ", flush=True)
        try:
            _, data = fetch_token_data(chain, addr)
            mcap_ok = bool(data.get("token_mcap_candles", {}).get("data"))
            trend_ok = bool(data.get("token_trends", {}).get("data"))
            print(f"mcap={'OK' if mcap_ok else 'EMPTY'} trends={'OK' if trend_ok else 'EMPTY'}")
        except Exception as e:
            print(f"ERROR: {e}")

        time.sleep(2)  # rate limit


if __name__ == "__main__":
    main()
