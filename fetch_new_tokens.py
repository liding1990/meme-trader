"""Fetch GMGN data for newly discovered tokens and add to radar.

Usage:
    python fetch_new_tokens.py [--max N] [--dry-run]
"""

import argparse
import csv
import json
import os
import sys
import time

DATA_DIR = "data"
RADAR_CSV = os.path.join(DATA_DIR, "radar_tokens.csv")
NEW_TOKENS_FILE = os.path.join(DATA_DIR, "new_tokens_2026.json")


def load_existing_radar():
    addrs = set()
    with open(RADAR_CSV) as f:
        for row in csv.reader(f):
            if len(row) >= 4:
                addrs.add(row[0].lower())
    return addrs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=500, help="Max tokens to fetch")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(NEW_TOKENS_FILE) as f:
        new_tokens = json.load(f)

    existing = load_existing_radar()
    print(f"新发现 token: {len(new_tokens)}")
    print(f"已有 radar: {len(existing)}")

    # Filter to Solana tokens only (chain detection)
    sol_tokens = []
    for t in new_tokens:
        addr = t["token"]["address"]
        if addr.lower() in existing:
            continue
        # Solana addresses are base58, ~44 chars, no 0x prefix
        if not addr.startswith("0x") and len(addr) > 30:
            sol_tokens.append(t)

    print(f"Solana 新 token: {len(sol_tokens)}")

    if args.dry_run:
        for t in sol_tokens[:20]:
            tok = t["token"]
            print(f"  {tok.get('symbol','?'):>12s}  {tok['address'][:20]}...")
        return

    from gmgn_api import fetch_token_data

    added = 0
    errors = 0

    for i, t in enumerate(sol_tokens[:args.max]):
        addr = t["token"]["address"]
        sym = t["token"].get("symbol", "?")
        name = t["token"].get("name", "?")

        try:
            _, loaded = fetch_token_data("sol", addr)
            candles = loaded.get("token_mcap_candles", {}).get("data", {}).get("list", [])

            if candles and len(candles) >= 10:
                # Add to radar CSV
                with open(RADAR_CSV, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([addr, "solana", name, sym, ""])
                added += 1
                print(f"  [{added}] {sym:>12s} — {len(candles)} candles")
            else:
                errors += 1

        except Exception as e:
            errors += 1

        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{min(len(sol_tokens), args.max)}, 成功: {added}, 失败: {errors}")

        time.sleep(2)  # rate limit

    print(f"\n完成: 新增 {added} 个 token, {errors} 个失败")
    print(f"Radar 总数: {len(existing) + added}")


if __name__ == "__main__":
    main()
