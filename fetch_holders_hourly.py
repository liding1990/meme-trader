"""Batch fetch Moralis hourly historical holders for all radar tokens.

Fetches 1h-granularity holder data covering each token's full candle time range.
Saves to data/{address}/moralis_holders_1h.json

Usage:
    python fetch_holders_hourly.py
"""

import asyncio
import csv
import json
import os
import glob
import sys

import aiohttp
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"
RADAR_CSV = os.path.join(DATA_DIR, "radar_tokens.csv")
_API_KEY = os.getenv("MORALIS_API_KEY", "")
_BASE_URL = "https://solana-gateway.moralis.io"
CONCURRENCY = 8
MAX_PAGES = 50  # more pages for hourly (longer time ranges)


def get_candle_time_range(address: str):
    """Get the full time range from 1h candle data."""
    data_dir = os.path.join(DATA_DIR, address)
    # Use the main candle file (not 5m, not 4h)
    files = sorted([
        f for f in glob.glob(os.path.join(data_dir, "token_mcap_candles_[0-9]*.json"))
        if "5m" not in os.path.basename(f) and "4h" not in os.path.basename(f)
    ])
    if not files:
        return None, None
    try:
        with open(files[-1]) as f:
            data = json.load(f)
        candles = data.get("data", {}).get("list", [])
        if not candles:
            return None, None
        times = [int(c["time"]) for c in candles]
        return min(times) // 1000, max(times) // 1000
    except Exception:
        return None, None


def has_hourly_cache(address: str) -> bool:
    return os.path.isfile(os.path.join(DATA_DIR, address, "moralis_holders_1h.json"))


def save_holder_data(address: str, results: list):
    path = os.path.join(DATA_DIR, address, "moralis_holders_1h.json")
    with open(path, "w") as f:
        json.dump(results, f)


async def fetch_one(session: aiohttp.ClientSession, sem: asyncio.Semaphore,
                    token: dict, idx: int, total: int):
    """Fetch hourly holder history for one token with pagination."""
    addr = token["address"]
    sym = token["symbol"]
    from_date = pd.Timestamp(token["start"], unit="s").isoformat() + "Z"
    to_date = pd.Timestamp(token["end"], unit="s").isoformat() + "Z"

    url = f"{_BASE_URL}/token/mainnet/holders/{addr}/historical"
    headers = {"X-Api-Key": _API_KEY, "Accept": "application/json"}

    all_results = []
    cursor = None
    page = 0

    async with sem:
        while page < MAX_PAGES:
            params = {
                "timeFrame": "1h",
                "fromDate": from_date,
                "toDate": to_date,
                "limit": 100,
            }
            if cursor:
                params["cursor"] = cursor

            try:
                async with session.get(url, headers=headers, params=params,
                                       timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(2)
                        continue
                    if resp.status != 200:
                        print(f"[{idx}/{total}] {sym:>12s}: HTTP {resp.status}")
                        break
                    data = await resp.json()
            except Exception as e:
                print(f"[{idx}/{total}] {sym:>12s}: ERROR {e}")
                break

            results = data.get("result", [])
            all_results.extend(results)
            cursor = data.get("cursor")
            if not cursor or not results:
                break
            page += 1

        if all_results:
            save_holder_data(addr, all_results)
        print(f"[{idx}/{total}] {sym:>12s}: {len(all_results)} hourly points")


async def main_async():
    if not _API_KEY:
        print("ERROR: MORALIS_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    # Load radar tokens (Solana only — Moralis endpoint is Solana-specific)
    tokens = []
    with open(RADAR_CSV) as f:
        for row in csv.reader(f):
            if len(row) >= 4 and row[1] == "solana":
                tokens.append({"address": row[0], "symbol": row[3]})

    # Determine which need fetching
    to_fetch = []
    cached = 0
    no_candles = 0

    for t in tokens:
        if has_hourly_cache(t["address"]):
            cached += 1
            continue
        start, end = get_candle_time_range(t["address"])
        if start and end:
            to_fetch.append({**t, "start": start, "end": end})
        else:
            no_candles += 1

    print(f"Radar tokens:    {len(tokens)}")
    print(f"Already cached:  {cached}")
    print(f"No candle data:  {no_candles}")
    print(f"To fetch:        {len(to_fetch)} (concurrency={CONCURRENCY})")

    if not to_fetch:
        print("Nothing to fetch.")
        return

    sem = asyncio.Semaphore(CONCURRENCY)
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_one(session, sem, t, i + 1, len(to_fetch))
            for i, t in enumerate(to_fetch)
        ]
        await asyncio.gather(*tasks)

    print("\nDone.")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
