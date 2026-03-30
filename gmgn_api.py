"""GMGN API data fetcher — fetches token trend and candle data for DNA fingerprinting."""

import json
import os
import subprocess
import sys
import time
from urllib.parse import urlencode


_GMGN_BASE_URL = "https://gmgn.gracematrix.net"
_CURL_BIN = os.path.expanduser("~/bin/curl_chrome116")

_COMMON_PARAMS = {
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


def _build_endpoints(chain, address):
    """Build the two endpoints needed for DNA fingerprinting."""
    return [
        {
            "name": "token_trends",
            "path": f"/api/v1/token_trends/{chain}/{address}",
            "param_string": "trends_type=avg_holding_balance&trends_type=holder_count&trends_type=top10_holder_percent&trends_type=top100_holder_percent",
        },
        {
            "name": "token_mcap_candles",
            "path": f"/api/v1/token_mcap_candles/{chain}/{address}",
            "params": {"resolution": "1h", "limit": "400", "pool_type": "unified"},
        },
    ]


def _build_url(endpoint):
    common = urlencode(_COMMON_PARAMS)
    extra = endpoint.get("param_string", "")
    if not extra and endpoint.get("params"):
        extra = urlencode(endpoint["params"])
    qs = "&".join(filter(None, [common, extra]))
    return f"{_GMGN_BASE_URL}{endpoint['path']}?{qs}"


def _curl_get(url):
    """Fetch URL using curl-impersonate (chrome116)."""
    result = subprocess.run(
        [_CURL_BIN, "-s", "--max-time", "15", url],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed (exit {result.returncode}): {result.stderr}")
    return result.stdout


def fetch_token_data(chain, address):
    """Fetch trend and candle data for a token.

    Returns (data_dir, loaded_data) where loaded_data is a dict:
        {
            "token_trends": {...},
            "token_mcap_candles": {...},
        }
    """
    if not os.path.isfile(_CURL_BIN):
        raise FileNotFoundError(f"curl-impersonate not found at {_CURL_BIN}")

    out_dir = os.path.join("data", address)
    os.makedirs(out_dir, exist_ok=True)

    endpoints = _build_endpoints(chain, address)
    timestamp = int(time.time() * 1000)

    loaded_data = {}

    for i, endpoint in enumerate(endpoints):
        url = _build_url(endpoint)
        filename = f"{endpoint['name']}_{timestamp}.json"
        filepath = os.path.join(out_dir, filename)

        print(f"Fetching {endpoint['name']}...", file=sys.stderr)

        try:
            body = _curl_get(url)

            if body.startswith("<!DOCTYPE") or "Cloudflare" in body:
                print(f"  Blocked by Cloudflare, saving raw response", file=sys.stderr)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(body)
                loaded_data[endpoint["name"]] = {}
                continue

            try:
                parsed = json.loads(body)
                content = json.dumps(parsed, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                print(f"  Warning: not valid JSON, saving raw text", file=sys.stderr)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(body)
                loaded_data[endpoint["name"]] = {}
                continue

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  Saved: {filepath}", file=sys.stderr)
            loaded_data[endpoint["name"]] = parsed

        except Exception as e:
            print(f"  Error fetching {endpoint['name']}: {e}", file=sys.stderr)
            loaded_data[endpoint["name"]] = {}

        if i < len(endpoints) - 1:
            time.sleep(1)

    print(f"Fetch complete → {out_dir}", file=sys.stderr)
    return out_dir, loaded_data
