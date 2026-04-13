"""Layer 2: Real-time monitoring + V4c entry signal generation.

Scans watchlist tokens for entry conditions:
  1. Token age < 6h
  2. Price at running ATH (±5%)
  3. Holder growth > 0 (last 30min)
  4. Swap-in > 5 (last 30min)
  5. ROC 30min > 2%
"""

import logging
import time

import numpy as np
import pandas as pd

from codex_api import get_token_bars
from scalper import config
from scalper.moralis_api import get_historical_holders
from scalper.price import get_token_price
from scalper.db import get_active_watchlist, update_watchlist_token, get_position

log = logging.getLogger("scalper.scanner")

# Cache holder data to avoid repeated API calls
_holder_cache = {}  # address -> (timestamp, data)
HOLDER_CACHE_TTL = 120  # 2 min


def _get_recent_holders(address: str) -> list | None:
    """Get last 30min of 5-min holder data, with caching."""
    now = time.time()
    if address in _holder_cache:
        cached_ts, cached_data = _holder_cache[address]
        if now - cached_ts < HOLDER_CACHE_TTL:
            return cached_data

    try:
        to_date = pd.Timestamp(now, unit="s").isoformat() + "Z"
        from_date = pd.Timestamp(now - 1800, unit="s").isoformat() + "Z"  # 30 min
        data = get_historical_holders(
            address=address, from_date=from_date, to_date=to_date,
            time_frame="5min", max_pages=2,
        )
        _holder_cache[address] = (now, data)
        return data
    except Exception as e:
        log.debug(f"Moralis error {address[:8]}: {e}")
        return None


def scan_token(address: str, symbol: str, created_at_ts: int) -> dict | None:
    """Scan a single token for entry signals.

    Returns signal dict if entry conditions met, None otherwise.
    """
    now_ts = int(time.time())
    token_age_h = (now_ts - created_at_ts) / 3600

    # Condition 1: Token age < 6h
    if token_age_h >= 6:
        return None

    # Already holding this token?
    if get_position(address):
        return None

    # Get real-time mcap from GMGN
    gmgn_data = get_token_price(address)
    if not gmgn_data or gmgn_data["market_cap"] <= 0:
        return None
    current_mcap = gmgn_data["market_cap"]

    # Get 5m bars from Codex for ROC calculation + buy/sell volume
    try:
        bars = get_token_bars(address, resolution="5", countback=36)
    except Exception as e:
        log.debug(f"Codex error {address[:8]}: {e}")
        return None

    if not bars or len(bars) < 7:
        return None

    # Use Codex bar prices for ROC (relative change is valid regardless of price vs mcap)
    prices = [b["close"] for b in bars if b["close"] > 0]
    if len(prices) < 7:
        return None

    # Condition 2: At running ATH — use GMGN mcap history isn't available per-bar,
    # so we check if current mcap is near the max of Codex bar prices (proportional)
    running_ath_price = max(prices)
    current_price = prices[-1]
    if current_price < running_ath_price * 0.95:
        return None

    # Condition 5: ROC 30m > 2%
    price_30m_ago = prices[-7]
    if price_30m_ago <= 0:
        return None
    roc_30m = (current_price / price_30m_ago - 1) * 100
    if roc_30m < 2.0:
        return None

    # Get holder data from Moralis
    holders_data = _get_recent_holders(address)
    if not holders_data or len(holders_data) < 2:
        return None

    # Condition 3: Holder growth > 0
    total_net_change = sum(d["netHolderChange"] for d in holders_data)
    if total_net_change <= 0:
        return None

    # Condition 4: Swap-in > 5
    total_swap_in = sum(d["newHoldersByAcquisition"].get("swap", 0) for d in holders_data)
    if total_swap_in < 5:
        return None

    current_holders = holders_data[0]["totalHolders"]
    holder_growth_rate = total_net_change / current_holders * 100 if current_holders > 0 else 0

    # Compute buy/sell volume from Codex bars
    recent_bars = bars[-6:]  # last 30 min
    buy_vol = sum(b.get("buy_volume", 0) or 0 for b in recent_bars)
    sell_vol = sum(b.get("sell_volume", 0) or 0 for b in recent_bars)

    update_watchlist_token(address, mcap=current_mcap, holders=current_holders)

    signal = {
        "address": address,
        "symbol": symbol,
        "price": current_mcap,
        "roc_30m": roc_30m,
        "holders": current_holders,
        "holder_growth": total_net_change,
        "holder_growth_rate": holder_growth_rate,
        "swap_in": total_swap_in,
        "buy_vol": buy_vol,
        "sell_vol": sell_vol,
        "token_age_h": token_age_h,
        "gmgn_buy_vol_5m": gmgn_data.get("buy_volume_5m", 0),
        "gmgn_sell_vol_5m": gmgn_data.get("sell_volume_5m", 0),
    }

    log.info(f"SIGNAL: {symbol} | ROC={roc_30m:.1f}% holders={current_holders} "
             f"swap={total_swap_in} age={token_age_h:.1f}h mcap=${current_mcap:,.0f}")

    return signal


def scan_watchlist(batch_size: int = None) -> list[dict]:
    """Scan watchlist tokens for entry signals.

    Returns list of signal dicts sorted by holder_growth_rate (highest first).
    """
    batch_size = batch_size or config.SCAN_BATCH_SIZE
    watchlist = get_active_watchlist()

    # Prioritize younger tokens
    now_ts = int(time.time())
    watchlist = sorted(watchlist, key=lambda r: r["created_at_ts"] or 0, reverse=True)

    signals = []
    scanned = 0

    for row in watchlist[:batch_size]:
        signal = scan_token(
            row["token_address"],
            row["symbol"] or "?",
            row["created_at_ts"] or 0,
        )
        scanned += 1

        if signal:
            signals.append(signal)

        time.sleep(config.SCAN_RATE_LIMIT)

    # Sort by holder growth rate (strongest momentum first)
    signals.sort(key=lambda s: s["holder_growth_rate"], reverse=True)

    return signals
