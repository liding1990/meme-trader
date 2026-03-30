"""Codex.io GraphQL API client for Solana token data.

Provides:
  - filterTokens: discover tokens by creation date, mcap, holders
  - getTokenBars: 5m OHLCV with buy/sell volume per candle
  - token metadata and holder info

API docs: https://docs.codex.io/reference/overview
Endpoint: https://graph.codex.io/graphql
Auth: API key in Authorization header
"""

import json
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

CODEX_ENDPOINT = "https://graph.codex.io/graphql"
CODEX_API_KEY = os.environ.get("CODEX_API_KEY", "")
SOLANA_NETWORK_ID = 1399811149

# Rate limit: 5 req/s on free tier
REQUEST_INTERVAL = 0.25


def _query(query_str, variables=None):
    """Execute a GraphQL query against Codex API."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": CODEX_API_KEY,
    }
    payload = {"query": query_str}
    if variables:
        payload["variables"] = variables

    resp = requests.post(CODEX_ENDPOINT, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {data['errors']}")

    return data.get("data", {})


def filter_tokens(created_after=None, created_before=None,
                  min_mcap=100_000, min_holders=1000,
                  limit=200, cursor=None):
    """Discover Solana tokens by creation date, market cap, and holder filters.

    Returns (tokens_list, next_cursor).
    """
    filters = {
        "network": [SOLANA_NETWORK_ID],
    }
    if min_mcap:
        filters["marketCap"] = {"gte": min_mcap}
    if min_holders:
        filters["holders"] = {"gte": min_holders}
    if created_after or created_before:
        date_filter = {}
        if created_after:
            date_filter["gte"] = int(created_after)
        if created_before:
            date_filter["lt"] = int(created_before)
        filters["createdAt"] = date_filter

    # Build query string with inline filters (Codex doesn't support variables well for filters)
    filter_parts = []
    filter_parts.append(f"network: [{SOLANA_NETWORK_ID}]")
    if min_mcap:
        filter_parts.append(f"marketCap: {{gte: {min_mcap}}}")
    if min_holders:
        filter_parts.append(f"holders: {{gte: {min_holders}}}")
    if created_after or created_before:
        date_parts = []
        if created_after:
            date_parts.append(f"gte: {int(created_after)}")
        if created_before:
            date_parts.append(f"lt: {int(created_before)}")
        filter_parts.append(f"createdAt: {{{', '.join(date_parts)}}}")

    filters_str = ", ".join(filter_parts)

    query = f"""
    {{
      filterTokens(
        filters: {{{filters_str}}}
        limit: {limit}
        rankings: {{attribute: createdAt, direction: DESC}}
      ) {{
        results {{
          token {{ address name symbol createdAt }}
          marketCap
          holders
          liquidity
        }}
        count
        page
      }}
    }}
    """

    result = _query(query)
    ft = result.get("filterTokens", {})
    tokens = ft.get("results", [])
    count = ft.get("count", 0)
    page = ft.get("page", None)

    return tokens, count, page


def get_token_bars(address, resolution="5", from_ts=0, to_ts=None,
                   countback=1000, currency="USD"):
    """Get OHLCV candles with buy/sell volume for a Solana token.

    resolution: "1", "5", "15", "30", "60", "240", "720", "1D"
    Returns list of bar dicts.
    """
    if to_ts is None:
        to_ts = int(time.time())

    symbol = f"{address}:{SOLANA_NETWORK_ID}"

    # Use inline query (Codex has issues with parameterized queries)
    query = (
        f'{{getTokenBars(symbol:"{symbol}",from:{from_ts},to:{to_ts},'
        f'resolution:"{resolution}",countback:{countback},'
        f'currencyCode:"{currency}",statsType:FILTERED,'
        f'removeLeadingNullValues:true)'
        f'{{t o h l c volume buyVolume sellVolume buyers sellers buys sells transactions liquidity}}}}'
    )

    result = _query(query)
    bars_data = result.get("getTokenBars", {})

    if not bars_data or not bars_data.get("t"):
        return []

    def _safe_float(val):
        if val is None:
            return 0.0
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    def _safe_int(val):
        if val is None:
            return 0
        try:
            return int(val)
        except (ValueError, TypeError):
            return 0

    # Convert parallel arrays to list of dicts
    n = len(bars_data["t"])
    bars = []
    for i in range(n):
        bar = {
            "timestamp": bars_data["t"][i],
            "open": _safe_float(bars_data["o"][i]),
            "high": _safe_float(bars_data["h"][i]),
            "low": _safe_float(bars_data["l"][i]),
            "close": _safe_float(bars_data["c"][i]),
            "volume": _safe_float(bars_data.get("volume", [None])[i] if bars_data.get("volume") and i < len(bars_data["volume"]) else None),
            "buy_volume": _safe_float(bars_data.get("buyVolume", [None])[i] if bars_data.get("buyVolume") and i < len(bars_data["buyVolume"]) else None),
            "sell_volume": _safe_float(bars_data.get("sellVolume", [None])[i] if bars_data.get("sellVolume") and i < len(bars_data["sellVolume"]) else None),
            "buyers": _safe_int(bars_data.get("buyers", [None])[i] if bars_data.get("buyers") and i < len(bars_data["buyers"]) else None),
            "sellers": _safe_int(bars_data.get("sellers", [None])[i] if bars_data.get("sellers") and i < len(bars_data["sellers"]) else None),
            "buys": _safe_int(bars_data.get("buys", [None])[i] if bars_data.get("buys") and i < len(bars_data["buys"]) else None),
            "sells": _safe_int(bars_data.get("sells", [None])[i] if bars_data.get("sells") and i < len(bars_data["sells"]) else None),
            "transactions": _safe_int(bars_data.get("transactions", [None])[i] if bars_data.get("transactions") and i < len(bars_data["transactions"]) else None),
            "liquidity": _safe_float(bars_data.get("liquidity", [None])[i] if bars_data.get("liquidity") and i < len(bars_data["liquidity"]) else None),
        }
        bars.append(bar)

    return bars


def get_token_info(address):
    """Get token metadata."""
    query = """
    query GetToken($address: String!, $networkId: Int!) {
      token(input: {address: $address, networkId: $networkId}) {
        name
        symbol
        decimals
        totalSupply
        createdAt
        socialLinks {
          twitter
          telegram
          website
        }
      }
    }
    """
    result = _query(query, {"address": address, "networkId": SOLANA_NETWORK_ID})
    return result.get("token", {})


def get_holders(address, limit=10):
    """Get current holder info including count and top10%."""
    query = """
    query GetHolders($tokenId: String!, $limit: Int) {
      holders(input: {tokenId: $tokenId, limit: $limit}) {
        count
        top10HoldersPercent
      }
    }
    """
    token_id = f"{address}:{SOLANA_NETWORK_ID}"
    result = _query(query, {"tokenId": token_id, "limit": limit})
    return result.get("holders", {})


if __name__ == "__main__":
    """Quick test of all API functions."""
    import sys

    if not CODEX_API_KEY:
        print("ERROR: Set CODEX_API_KEY in .env", file=sys.stderr)
        sys.exit(1)

    print("=== Testing Codex API ===\n")

    # Test filterTokens
    print("1. filterTokens (Solana, mcap>=100K, holders>=1000)...")
    tokens, count, page = filter_tokens(min_mcap=100000, min_holders=1000, limit=5)
    print(f"   Found {count} total tokens, showing {len(tokens)}")
    for t in tokens[:3]:
        tok = t["token"]
        print(f"   {tok['symbol']:12s} mcap=${t['circulatingMarketCap']:>12,.0f}  "
              f"holders={t['holders']:>6,}  vol24=${t['volume24']:>12,.0f}")

    time.sleep(REQUEST_INTERVAL)

    # Test getTokenBars
    if tokens:
        addr = tokens[0]["token"]["address"]
        sym = tokens[0]["token"]["symbol"]
        print(f"\n2. getTokenBars (5m, {sym})...")
        bars = get_token_bars(addr, resolution="5", countback=20)
        print(f"   Got {len(bars)} bars")
        if bars:
            b = bars[-1]
            print(f"   Latest: close={b['close']}, vol={b['volume']:.2f}, "
                  f"buyVol={b['buy_volume']:.2f}, sellVol={b['sell_volume']:.2f}, "
                  f"buyers={b['buyers']}, sellers={b['sellers']}")

        time.sleep(REQUEST_INTERVAL)

        # Test holders
        print(f"\n3. getHolders ({sym})...")
        holders = get_holders(addr)
        print(f"   Count: {holders.get('count', '?')}, "
              f"Top10%: {holders.get('top10HoldersPercent', '?')}")

    print("\n=== All tests passed ===")
