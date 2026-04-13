"""Batch fetch Moralis 5-min historical holders — concurrent version."""

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
CONCURRENCY = 10  # parallel requests
MAX_PAGES = 20    # max pages per token


def get_5m_candle_time_range(address: str):
    data_dir = os.path.join(DATA_DIR, address)
    files = sorted(glob.glob(os.path.join(data_dir, "token_mcap_candles_5m_*.json")))
    if not files:
        return None, None
    with open(files[-1]) as f:
        data = json.load(f)
    ds = data.get("data")
    if not ds:
        return None, None
    candles = ds.get("list", [])
    if not candles:
        return None, None
    times = [int(c["time"]) for c in candles]
    return min(times) // 1000, max(times) // 1000


def has_holder_cache(address: str) -> bool:
    return os.path.isfile(os.path.join(DATA_DIR, address, "moralis_holders_5m.json"))


def save_holder_data(address: str, results: list):
    path = os.path.join(DATA_DIR, address, "moralis_holders_5m.json")
    with open(path, "w") as f:
        json.dump(results, f)


async def fetch_one(session: aiohttp.ClientSession, sem: asyncio.Semaphore,
                    token: dict, idx: int, total: int):
    """Fetch holder history for one token with pagination."""
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
                "timeFrame": "5min",
                "fromDate": from_date,
                "toDate": to_date,
                "limit": 100,
            }
            if cursor:
                params["cursor"] = cursor

            try:
                async with session.get(url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(2)
                        continue
                    if resp.status != 200:
                        print(f"[{idx}/{total}] {sym}: HTTP {resp.status}")
                        break
                    data = await resp.json()
            except Exception as e:
                print(f"[{idx}/{total}] {sym}: ERROR {e}")
                break

            results = data.get("result", [])
            all_results.extend(results)
            cursor = data.get("cursor")
            if not cursor or not results:
                break
            page += 1

        save_holder_data(addr, all_results)
        print(f"[{idx}/{total}] {sym}: {len(all_results)} points")


async def main_async():
    tokens = []
    with open(RADAR_CSV) as f:
        for row in csv.reader(f):
            if len(row) >= 4 and row[1] == "solana":
                tokens.append({"address": row[0], "symbol": row[3]})

    to_fetch = []
    for t in tokens:
        if has_holder_cache(t["address"]):
            continue
        s, e = get_5m_candle_time_range(t["address"])
        if s and e:
            to_fetch.append({**t, "start": s, "end": e})

    print(f"Radar tokens: {len(tokens)}")
    print(f"Already cached: {len(tokens) - len(to_fetch)}")
    print(f"To fetch: {len(to_fetch)} (concurrency={CONCURRENCY})")

    sem = asyncio.Semaphore(CONCURRENCY)
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_one(session, sem, t, i + 1, len(to_fetch))
            for i, t in enumerate(to_fetch)
        ]
        await asyncio.gather(*tasks)

    print("Done.")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
