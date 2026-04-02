"""Batch-fetch early 5-minute candle data for all radar tokens.

Uses GMGN API pagination (the 'to' parameter) to fetch candles from the
token's earliest known time forward, filling the gap between current 5m
data and the token's actual launch.

Usage:
    python fetch_early_5m.py [--dry-run] [--concurrency N] [--delay SECS]
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlencode


DATA_DIR = "data"
RADAR_CSV = os.path.join(DATA_DIR, "radar_tokens.csv")
CURL_BIN = os.path.expanduser("~/bin/curl_chrome116")

COMMON_PARAMS = {
    "device_id": "41394353-62B9-41D6-98BC-1B2AFC706103",
    "client_id": "memetracker_ios_2020513",
    "from_app": "memetracker",
    "app_ver": "2020513",
    "pkg": "io.gracematrix.gmgn",
    "app_lang": "en",
    "sys_lang": "en-CN",
    "brand": "Apple",
    "model": "iPhone 15 Pro",
    "os": "ios",
    "os_api": "26.3",
    "tz_name": "Asia/Shanghai",
    "tz_offset": "-480",
}


def load_radar_tokens() -> list[dict]:
    """Load radar tokens. Returns list of {address, chain, symbol}."""
    tokens = []
    with open(RADAR_CSV) as f:
        for row in csv.reader(f):
            if len(row) >= 4:
                tokens.append({"address": row[0], "chain": row[1], "symbol": row[3]})
    return tokens


def get_earliest_1h_timestamp(address: str) -> int | None:
    """Get the earliest hourly candle timestamp for a token (milliseconds)."""
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
        candles = data.get("data", {}).get("list", [])
        if not candles:
            return None
        return min(int(c["time"]) for c in candles)
    except Exception:
        return None


def get_existing_5m_candles(address: str) -> list[dict]:
    """Load all existing 5m candles for a token, merged from all files."""
    data_dir = os.path.join(DATA_DIR, address)
    m_files = sorted(glob.glob(os.path.join(data_dir, "token_mcap_candles_5m_*.json")))
    all_candles = []
    for fpath in m_files:
        try:
            with open(fpath) as f:
                data = json.load(f)
            candles = data.get("data", {}).get("list", [])
            all_candles.extend(candles)
        except Exception:
            continue
    # Deduplicate by timestamp
    seen = set()
    unique = []
    for c in all_candles:
        t = int(c["time"])
        if t not in seen:
            seen.add(t)
            unique.append(c)
    unique.sort(key=lambda c: int(c["time"]))
    return unique


def needs_fetch(address: str) -> dict | None:
    """Check if token needs early 5m data fetched.

    Returns {target_ts, existing_earliest_5m, gap_hours} or None if not needed.
    """
    earliest_1h = get_earliest_1h_timestamp(address)
    if earliest_1h is None:
        return None

    existing_5m = get_existing_5m_candles(address)
    if not existing_5m:
        # No 5m data at all — need full fetch from earliest_1h
        return {"target_ts": earliest_1h, "existing_earliest_5m": None, "gap_hours": None}

    earliest_5m = min(int(c["time"]) for c in existing_5m)
    gap_hours = (earliest_5m - earliest_1h) / 3600000

    if gap_hours <= 1:  # already covered
        return None

    return {
        "target_ts": earliest_1h,
        "existing_earliest_5m": earliest_5m,
        "gap_hours": gap_hours,
    }


def curl_get(url: str) -> str:
    result = subprocess.run(
        [CURL_BIN, "-s", "--max-time", "15", url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed (exit {result.returncode})")
    return result.stdout


def fetch_early_5m_candles(chain: str, address: str, target_ts: int,
                           stop_ts: int | None = None, max_pages: int = 20) -> list[dict]:
    """Fetch 5m candles backwards from stop_ts until we reach target_ts.

    If stop_ts is None, fetches most recent and paginates backwards.
    """
    all_candles = []
    to_ts = stop_ts  # Start pagination from existing earliest 5m data

    for page in range(max_pages):
        params = {"resolution": "5m", "limit": "400", "pool_type": "unified"}
        if to_ts:
            params["to"] = str(to_ts)

        common_qs = urlencode(COMMON_PARAMS)
        param_qs = urlencode(params)
        url = f"https://gmgn.gracematrix.net/api/v1/token_mcap_candles/{chain}/{address}?{common_qs}&{param_qs}"

        try:
            body = curl_get(url)
            if body.startswith("<!DOCTYPE") or "Cloudflare" in body:
                break
            parsed = json.loads(body)
            candles = parsed.get("data", {}).get("list", [])
        except Exception:
            break

        if not candles:
            break

        all_candles.extend(candles)

        earliest = min(int(c["time"]) for c in candles)

        # Have we reached the target?
        if earliest <= target_ts:
            break

        # No progress?
        if to_ts is not None and earliest >= to_ts:
            break

        to_ts = earliest

        if len(candles) < 400:
            break  # Last page

        time.sleep(0.5)

    # Deduplicate and sort
    seen = set()
    unique = []
    for c in all_candles:
        t = int(c["time"])
        if t not in seen:
            seen.add(t)
            unique.append(c)
    unique.sort(key=lambda c: int(c["time"]))
    return unique


def save_5m_candles(address: str, new_candles: list[dict]):
    """Merge new candles with existing and save as single consolidated file."""
    existing = get_existing_5m_candles(address)

    # Merge
    seen = set()
    merged = []
    for c in existing + new_candles:
        t = int(c["time"])
        if t not in seen:
            seen.add(t)
            merged.append(c)
    merged.sort(key=lambda c: int(c["time"]))

    # Save as consolidated file
    data_dir = os.path.join(DATA_DIR, address)
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "token_mcap_candles_5m_full.json")

    output = {
        "code": 0,
        "reason": "success",
        "message": "ok",
        "data": {"list": merged},
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return len(merged), out_path


def process_token(token: dict, dry_run: bool = False) -> dict:
    """Process a single token: check if fetch needed, fetch, save."""
    addr = token["address"]
    chain = token["chain"]
    symbol = token["symbol"]

    info = needs_fetch(addr)
    if info is None:
        return {"symbol": symbol, "status": "skip", "reason": "already covered"}

    target_ts = info["target_ts"]
    existing_earliest = info["existing_earliest_5m"]
    gap = info["gap_hours"]

    if dry_run:
        return {
            "symbol": symbol, "status": "dry_run",
            "gap_hours": gap, "target_ts": target_ts,
        }

    # Fetch
    try:
        new_candles = fetch_early_5m_candles(
            chain, addr, target_ts,
            stop_ts=existing_earliest,
            max_pages=20,
        )
    except Exception as e:
        return {"symbol": symbol, "status": "error", "error": str(e)}

    if not new_candles:
        return {"symbol": symbol, "status": "no_data", "gap_hours": gap}

    # Save
    total, path = save_5m_candles(addr, new_candles)
    return {
        "symbol": symbol, "status": "ok",
        "new_candles": len(new_candles), "total_candles": total,
        "gap_hours": gap,
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch early 5m candle data for radar tokens")
    parser.add_argument("--dry-run", action="store_true", help="Check gaps without fetching")
    parser.add_argument("--concurrency", type=int, default=3, help="Concurrent fetch threads")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between tokens (seconds)")
    args = parser.parse_args()

    if not os.path.isfile(CURL_BIN):
        print(f"ERROR: curl-impersonate not found at {CURL_BIN}", file=sys.stderr)
        sys.exit(1)

    tokens = load_radar_tokens()
    print(f"Loaded {len(tokens)} radar tokens")

    # Pre-scan: which tokens need fetching?
    to_fetch = []
    skipped = 0
    for token in tokens:
        info = needs_fetch(token["address"])
        if info is not None:
            to_fetch.append(token)
        else:
            skipped += 1

    print(f"Need fetch: {len(to_fetch)}, already covered: {skipped}")

    if args.dry_run:
        print("\n--- DRY RUN ---")
        for token in to_fetch[:20]:
            info = needs_fetch(token["address"])
            gap = info.get("gap_hours", "N/A")
            print(f"  {token['symbol']:>12s}: gap={gap}h")
        if len(to_fetch) > 20:
            print(f"  ... and {len(to_fetch) - 20} more")
        return

    # Fetch with concurrency
    results = {"ok": 0, "error": 0, "no_data": 0, "skip": 0}
    errors = []

    def fetch_with_delay(token, idx):
        # Stagger starts
        time.sleep(idx * args.delay / args.concurrency)
        return process_token(token)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(fetch_with_delay, token, i): token
            for i, token in enumerate(to_fetch)
        }

        for future in as_completed(futures):
            token = futures[future]
            try:
                result = future.result()
                status = result["status"]
                results[status] = results.get(status, 0) + 1

                if status == "ok":
                    print(f"  OK {result['symbol']:>12s}: +{result['new_candles']} candles "
                          f"(total {result['total_candles']}, gap was {result.get('gap_hours', '?')}h)")
                elif status == "error":
                    errors.append(result)
                    print(f"  ERR {result['symbol']:>12s}: {result.get('error', '?')}")
                elif status == "no_data":
                    print(f"  EMPTY {result['symbol']:>12s}: no early data available (gap {result.get('gap_hours', '?')}h)")
            except Exception as e:
                print(f"  CRASH {token['symbol']:>12s}: {e}")
                results["error"] += 1

    print(f"\nDone: {results['ok']} ok, {results['error']} errors, "
          f"{results['no_data']} no_data, {results['skip']} skip")


if __name__ == "__main__":
    main()
