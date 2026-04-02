"""Check 5-minute data completeness and quality for early warning system.

Reports:
1. Coverage: how many tokens have 5m data covering their early period
2. Density: gaps in 5m data (missing candles)
3. First-24h quality: specifically checks if we have good data for T+0 to T+24h from $100K
"""

import csv
import json
import os
import glob
import sys

import numpy as np
import pandas as pd


DATA_DIR = "data"
RADAR_CSV = os.path.join(DATA_DIR, "radar_tokens.csv")
MCAP_THRESHOLD = 100_000  # $100K start
TARGET_HOURS = 24


def load_radar_tokens():
    tokens = []
    with open(RADAR_CSV) as f:
        for row in csv.reader(f):
            if len(row) >= 4:
                tokens.append({"address": row[0], "chain": row[1], "symbol": row[3]})
    return tokens


def load_5m_candles(address: str) -> pd.DataFrame | None:
    """Load best available 5m candles (prefer _full file, fallback to latest snapshot)."""
    data_dir = os.path.join(DATA_DIR, address)

    # Prefer consolidated full file
    full_path = os.path.join(data_dir, "token_mcap_candles_5m_full.json")
    if os.path.isfile(full_path):
        with open(full_path) as f:
            data = json.load(f)
        candles = (data or {}).get("data", {}).get("list", []) if data else []
        if candles:
            df = pd.DataFrame(candles)
            df["time_ms"] = df["time"].astype(int)
            df["datetime"] = pd.to_datetime(df["time_ms"], unit="ms")
            df["mcap"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(float)
            return df.sort_values("datetime").reset_index(drop=True)

    # Fallback: latest 5m snapshot
    m_files = sorted(glob.glob(os.path.join(data_dir, "token_mcap_candles_5m_*.json")))
    if not m_files:
        return None
    try:
        with open(m_files[-1]) as f:
            data = json.load(f)
        candles = (data or {}).get("data", {}).get("list", []) if data else []
    except Exception:
        return None
    if not candles:
        return None
    df = pd.DataFrame(candles)
    df["time_ms"] = df["time"].astype(int)
    df["datetime"] = pd.to_datetime(df["time_ms"], unit="ms")
    df["mcap"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df.sort_values("datetime").reset_index(drop=True)


def load_1h_candles(address: str) -> pd.DataFrame | None:
    """Load hourly candles for reference (earliest timestamp)."""
    data_dir = os.path.join(DATA_DIR, address)
    h_files = sorted([
        f for f in glob.glob(os.path.join(data_dir, "token_mcap_candles_[0-9]*.json"))
        if "5m" not in os.path.basename(f)
    ])
    if not h_files:
        return None
    try:
        with open(h_files[-1]) as f:
            data = json.load(f)
        candles = (data or {}).get("data", {}).get("list", []) if data else []
    except Exception:
        return None
    if not candles:
        return None
    df = pd.DataFrame(candles)
    df["time_ms"] = df["time"].astype(int)
    df["datetime"] = pd.to_datetime(df["time_ms"], unit="ms")
    df["mcap"] = df["close"].astype(float)
    return df.sort_values("datetime").reset_index(drop=True)


def check_token(token: dict) -> dict:
    """Check data quality for a single token."""
    addr = token["address"]
    symbol = token["symbol"]

    result = {"symbol": symbol, "address": addr}

    df_5m = load_5m_candles(addr)
    df_1h = load_1h_candles(addr)

    if df_5m is None:
        result["status"] = "no_5m_data"
        return result

    if df_1h is None:
        result["status"] = "no_1h_data"
        return result

    result["total_5m_candles"] = len(df_5m)
    result["5m_start"] = df_5m["datetime"].iloc[0].isoformat()
    result["5m_end"] = df_5m["datetime"].iloc[-1].isoformat()
    result["1h_start"] = df_1h["datetime"].iloc[0].isoformat()

    # Check if 5m covers the early period
    gap_hours = (df_5m["datetime"].iloc[0] - df_1h["datetime"].iloc[0]).total_seconds() / 3600
    result["gap_hours"] = round(gap_hours, 1)

    # Find when mcap first crosses $100K in 5m data
    above_100k = df_5m[df_5m["mcap"] >= MCAP_THRESHOLD]
    if above_100k.empty:
        result["status"] = "never_reached_100k"
        return result

    t0 = above_100k["datetime"].iloc[0]
    result["100k_time"] = t0.isoformat()

    # Extract first 24h from $100K crossing
    window = df_5m[(df_5m["datetime"] >= t0) &
                   (df_5m["datetime"] <= t0 + pd.Timedelta(hours=TARGET_HOURS))]

    result["first_24h_candles"] = len(window)
    expected_candles = TARGET_HOURS * 12  # 12 per hour for 5m data
    result["first_24h_coverage"] = round(len(window) / expected_candles * 100, 1)

    # Check for gaps > 30 minutes in the first 24h
    if len(window) >= 2:
        diffs = window["datetime"].diff().iloc[1:]
        max_gap_min = diffs.max().total_seconds() / 60
        result["max_gap_minutes"] = round(max_gap_min, 1)
        n_big_gaps = (diffs > pd.Timedelta(minutes=30)).sum()
        result["gaps_over_30min"] = int(n_big_gaps)
    else:
        result["max_gap_minutes"] = None
        result["gaps_over_30min"] = None

    # MCap range in first 24h
    if len(window) >= 2:
        result["mcap_min_24h"] = round(window["mcap"].min())
        result["mcap_max_24h"] = round(window["mcap"].max())

    # Quality rating
    if len(window) >= expected_candles * 0.7 and gap_hours <= 1:
        result["status"] = "good"
    elif len(window) >= expected_candles * 0.4:
        result["status"] = "partial"
    elif len(window) >= 2:
        result["status"] = "sparse"
    else:
        result["status"] = "insufficient"

    return result


def main():
    tokens = load_radar_tokens()
    print(f"Checking {len(tokens)} radar tokens...\n")

    results = []
    for token in tokens:
        results.append(check_token(token))

    # Summary
    status_counts = {}
    for r in results:
        s = r["status"]
        status_counts[s] = status_counts.get(s, 0) + 1

    print("=== DATA QUALITY SUMMARY ===\n")
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {status:>20s}: {count}")

    good = [r for r in results if r["status"] == "good"]
    partial = [r for r in results if r["status"] == "partial"]
    usable = good + partial

    print(f"\n  Usable for training: {len(usable)} tokens (good + partial)")
    print(f"  Total radar tokens: {len(tokens)}")

    # Coverage details for usable tokens
    if usable:
        coverages = [r["first_24h_coverage"] for r in usable]
        candle_counts = [r["first_24h_candles"] for r in usable]
        print(f"\n  First-24h coverage: median={np.median(coverages):.0f}%, "
              f"min={min(coverages):.0f}%, max={max(coverages):.0f}%")
        print(f"  First-24h candles:  median={np.median(candle_counts):.0f}, "
              f"min={min(candle_counts)}, max={max(candle_counts)}")

    # Show some problem tokens
    problems = [r for r in results if r["status"] not in ("good", "partial")]
    if problems:
        print(f"\n=== PROBLEM TOKENS (sample) ===\n")
        for r in problems[:15]:
            sym = r["symbol"]
            status = r["status"]
            gap = r.get("gap_hours", "?")
            n5m = r.get("total_5m_candles", 0)
            print(f"  {sym:>12s}: {status:>20s}  gap={gap}h  5m_candles={n5m}")

    # Show good tokens sample
    if good:
        print(f"\n=== GOOD TOKENS (sample) ===\n")
        for r in sorted(good, key=lambda x: x.get("mcap_max_24h", 0), reverse=True)[:15]:
            sym = r["symbol"]
            cov = r["first_24h_coverage"]
            n = r["first_24h_candles"]
            mcap_max = r.get("mcap_max_24h", 0)
            gaps = r.get("gaps_over_30min", 0)
            print(f"  {sym:>12s}: {cov:.0f}% coverage, {n} candles, "
                  f"ATH_24h=${mcap_max:,.0f}, gaps>30m={gaps}")


if __name__ == "__main__":
    main()
