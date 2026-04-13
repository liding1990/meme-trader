"""Real-time price/mcap fetching via GMGN API."""

import json
import logging
import time

from gmgn_api import _build_url, _curl_get

log = logging.getLogger("scalper.price")

# Cache prices to avoid excessive API calls
_price_cache = {}  # address -> (timestamp, price_data)
CACHE_TTL = 30  # 30 seconds


def get_token_price(address: str, chain: str = "sol") -> dict | None:
    """Get real-time token price and mcap from GMGN.

    Returns dict with: price, market_cap, buy_volume_5m, sell_volume_5m, etc.
    Returns None if API fails.
    """
    now = time.time()
    if address in _price_cache:
        cached_ts, cached_data = _price_cache[address]
        if now - cached_ts < CACHE_TTL:
            return cached_data

    endpoint = {
        "name": "price_info",
        "path": f"/api/v1/token_price_info/{chain}/{address}",
        "params": {},
    }
    url = _build_url(endpoint)

    try:
        body = _curl_get(url)
        data = json.loads(body)

        if data.get("code") != 0 or not data.get("data"):
            return None

        d = data["data"]
        price = float(d.get("price", 0) or 0)

        result = {
            "price": price,
            "market_cap": float(d.get("market_cap", 0) or 0),
            "buy_volume_5m": float(d.get("buy_volume_5m", 0) or 0),
            "sell_volume_5m": float(d.get("sell_volume_5m", 0) or 0),
            "volume_5m": float(d.get("volume_5m", 0) or 0),
            "buys_5m": int(d.get("buys_5m", 0) or 0),
            "sells_5m": int(d.get("sells_5m", 0) or 0),
            "price_change_5m": float(d.get("price_5m", 0) or 0),
        }

        _price_cache[address] = (now, result)
        return result

    except Exception as e:
        log.debug(f"GMGN price error {address[:8]}: {e}")
        return None


def get_mcap(address: str) -> float | None:
    """Get current market cap. Returns None if unavailable."""
    data = get_token_price(address)
    if data and data["market_cap"] > 0:
        return data["market_cap"]
    # Fallback: if mcap not available but price is, that's all we have
    if data and data["price"] > 0:
        return data["price"]  # This is actually token price, not mcap
    return None
