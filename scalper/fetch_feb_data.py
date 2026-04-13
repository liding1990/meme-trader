"""Fetch Codex 5m bars + Moralis holders for Feb 2026 tokens (validation set)."""

import asyncio
import json
import os
import time

import aiohttp
import pandas as pd
from dotenv import load_dotenv

from codex_api import get_token_bars

load_dotenv()

DATA_DIR = "data"
MORALIS_KEY = os.getenv("MORALIS_API_KEY", "")
MORALIS_BASE = "https://solana-gateway.moralis.io"
CONCURRENCY = 8


def fetch_codex_bars(address: str, created_at: int):
    """Fetch 5m bars from Codex for the first 24h after creation."""
    from_ts = created_at
    to_ts = created_at + 24 * 3600  # first 24 hours

    bars = get_token_bars(address, resolution="5", from_ts=from_ts, to_ts=to_ts, countback=1000)
    return bars


async def fetch_moralis_holders(session, sem, address, from_ts, to_ts):
    """Fetch 5min holders from Moralis."""
    url = f"{MORALIS_BASE}/token/mainnet/holders/{address}/historical"
    headers = {"X-Api-Key": MORALIS_KEY, "Accept": "application/json"}
    params = {
        "timeFrame": "5min",
        "fromDate": pd.Timestamp(from_ts, unit="s").isoformat() + "Z",
        "toDate": pd.Timestamp(to_ts, unit="s").isoformat() + "Z",
        "limit": 100,
    }

    all_results = []
    cursor = None
    page = 0

    async with sem:
        while page < 15:
            if cursor:
                params["cursor"] = cursor
            try:
                async with session.get(url, headers=headers, params=params,
                                       timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(2)
                        continue
                    if resp.status != 200:
                        break
                    data = await resp.json()
            except Exception:
                break

            results = data.get("result", [])
            all_results.extend(results)
            cursor = data.get("cursor")
            if not cursor or not results:
                break
            page += 1

    return all_results


async def main():
    with open(os.path.join(DATA_DIR, "feb2026_tokens.json")) as f:
        tokens = json.load(f)

    print(f"Feb tokens: {len(tokens)}")

    # Filter to tokens we don't have data for yet
    to_fetch = []
    for t in tokens:
        addr = t["token"]["address"]
        data_dir = os.path.join(DATA_DIR, addr)
        bars_path = os.path.join(data_dir, "codex_5m_bars.json")
        holders_path = os.path.join(data_dir, "moralis_holders_5m.json")
        if os.path.isfile(bars_path) and os.path.isfile(holders_path):
            continue
        to_fetch.append(t)

    print(f"Already cached: {len(tokens) - len(to_fetch)}")
    print(f"To fetch: {len(to_fetch)}")

    # Step 1: Fetch Codex bars (synchronous, rate limited)
    print("\n--- Fetching Codex 5m bars ---")
    for i, t in enumerate(to_fetch):
        addr = t["token"]["address"]
        sym = t["token"]["symbol"][:12]
        created_at = t["token"]["createdAt"]
        data_dir = os.path.join(DATA_DIR, addr)
        os.makedirs(data_dir, exist_ok=True)

        bars_path = os.path.join(data_dir, "codex_5m_bars.json")
        if not os.path.isfile(bars_path):
            try:
                bars = fetch_codex_bars(addr, created_at)
                with open(bars_path, "w") as f:
                    json.dump(bars, f)
                print(f"[{i+1}/{len(to_fetch)}] {sym}: {len(bars)} bars")
            except Exception as e:
                print(f"[{i+1}/{len(to_fetch)}] {sym}: ERROR {e}")
                with open(bars_path, "w") as f:
                    json.dump([], f)
            time.sleep(0.3)

    # Step 2: Fetch Moralis holders (concurrent)
    print("\n--- Fetching Moralis holders ---")
    sem = asyncio.Semaphore(CONCURRENCY)
    async with aiohttp.ClientSession() as session:
        tasks = []
        task_info = []
        for t in to_fetch:
            addr = t["token"]["address"]
            holders_path = os.path.join(DATA_DIR, addr, "moralis_holders_5m.json")
            if os.path.isfile(holders_path):
                continue
            created_at = t["token"]["createdAt"]
            from_ts = created_at
            to_ts = created_at + 24 * 3600
            tasks.append(fetch_moralis_holders(session, sem, addr, from_ts, to_ts))
            task_info.append((addr, t["token"]["symbol"][:12]))

        if tasks:
            results = await asyncio.gather(*tasks)
            for (addr, sym), holders in zip(task_info, results):
                holders_path = os.path.join(DATA_DIR, addr, "moralis_holders_5m.json")
                with open(holders_path, "w") as f:
                    json.dump(holders, f)
                print(f"  {sym}: {len(holders)} holder points")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
