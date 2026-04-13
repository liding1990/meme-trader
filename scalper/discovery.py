"""Layer 1: Token discovery + scam filter.

Discovers new tokens via Codex, evaluates them with Moralis holder data,
filters out scam patterns, and adds qualified tokens to watchlist.
"""

import logging
import time

from codex_api import filter_tokens, get_holders
from scalper import config
from scalper.moralis_api import get_historical_holders
from scalper.db import (
    add_to_watchlist, get_active_watchlist, remove_from_watchlist,
    watchlist_count, log_discovery,
)

log = logging.getLogger("scalper.discovery")


def discover_new_tokens():
    """Find tokens created in the last 3 hours with basic thresholds."""
    now = int(time.time())
    created_after = now - 3 * 3600  # last 3 hours

    try:
        tokens, count, _ = filter_tokens(
            created_after=created_after,
            min_mcap=config.DISCOVER_MIN_MCAP,
            min_holders=config.DISCOVER_MIN_HOLDERS,
            limit=200,
        )
    except Exception as e:
        log.error(f"Discovery failed: {e}")
        return []

    log.info(f"Discovery: {count} total, {len(tokens)} returned")
    return tokens


def evaluate_scam_score(address: str, created_at_ts: int) -> dict:
    """Evaluate a token's scam likelihood using Moralis holder data.

    Returns dict with scam metrics and pass/fail boolean.
    """
    now_ts = int(time.time())
    token_age_h = (now_ts - created_at_ts) / 3600

    # Fetch last 3 hours of holder data (or since creation, whichever is shorter)
    from_ts = max(created_at_ts, now_ts - 3 * 3600)
    try:
        import pandas as pd
        from_date = pd.Timestamp(from_ts, unit="s").isoformat() + "Z"
        to_date = pd.Timestamp(now_ts, unit="s").isoformat() + "Z"
        holders_data = get_historical_holders(
            address=address,
            from_date=from_date,
            to_date=to_date,
            time_frame="5min",
            max_pages=5,
        )
    except Exception as e:
        log.warning(f"Moralis fetch failed for {address[:8]}: {e}")
        return {"passed": False, "reason": "moralis_error"}

    if not holders_data or len(holders_data) < 3:
        return {"passed": False, "reason": "insufficient_data"}

    # Compute scam metrics
    total_holders_start = holders_data[-1]["totalHolders"]  # data is reverse chronological
    total_holders_end = holders_data[0]["totalHolders"]

    if total_holders_start <= 0:
        holder_growth_pct = 0
    else:
        holder_growth_pct = (total_holders_end - total_holders_start) / total_holders_start * 100

    # Net changes
    net_changes = [d["netHolderChange"] for d in holders_data]
    positive_bars = sum(1 for n in net_changes if n > 0)
    positive_bars_pct = positive_bars / len(net_changes) * 100 if net_changes else 0

    # Acquisition methods
    total_swap = sum(d["newHoldersByAcquisition"].get("swap", 0) for d in holders_data)
    total_transfer = sum(d["newHoldersByAcquisition"].get("transfer", 0) for d in holders_data)
    total_airdrop = sum(d["newHoldersByAcquisition"].get("airdrop", 0) for d in holders_data)
    total_acquired = total_swap + total_transfer + total_airdrop

    # Big money activity
    big_money_in = sum(
        d["holdersIn"].get("whales", 0) + d["holdersIn"].get("sharks", 0)
        for d in holders_data
    )

    result = {
        "holders": total_holders_end,
        "holder_growth_pct": holder_growth_pct,
        "positive_bars_pct": positive_bars_pct,
        "total_swap": total_swap,
        "total_transfer": total_transfer,
        "total_airdrop": total_airdrop,
        "big_money_in": big_money_in,
        "data_points": len(holders_data),
    }

    # === SCAM FILTER RULES ===
    reasons = []

    if total_holders_end < 100:
        reasons.append(f"holders={total_holders_end}<100")

    if holder_growth_pct < 10 and token_age_h > 1:
        reasons.append(f"growth={holder_growth_pct:.0f}%<10%")

    if positive_bars_pct < 30:
        reasons.append(f"positive_bars={positive_bars_pct:.0f}%<30%")

    if total_swap < 5 and total_acquired > 10:
        reasons.append(f"swap={total_swap}<5 (no organic buyers)")

    if total_airdrop > total_swap and total_airdrop > 5:
        reasons.append(f"airdrop={total_airdrop}>swap={total_swap}")

    if big_money_in == 0 and total_holders_end > 50:
        reasons.append("no_big_money")

    result["passed"] = len(reasons) == 0
    result["reason"] = "; ".join(reasons) if reasons else "ok"

    return result


def run_discovery_cycle():
    """Full discovery cycle: find tokens → scam filter → add to watchlist."""
    tokens = discover_new_tokens()
    if not tokens:
        return

    # Get existing watchlist addresses
    existing = {row["token_address"] for row in get_active_watchlist()}

    new_count = 0
    passed_count = 0
    blocked_count = 0

    for t in tokens:
        tok = t["token"]
        address = tok["address"]

        if address in existing:
            continue

        new_count += 1
        symbol = tok.get("symbol", "?")[:16]
        name = tok.get("name", "?")[:32]
        created_at = tok.get("createdAt", 0)
        liquidity = float(t.get("liquidity", 0) or 0)

        # Must be graduated (has DEX liquidity)
        if liquidity <= 0:
            blocked_count += 1
            log.debug(f"NOT GRADUATED: {symbol} ({address[:8]}) | liquidity=0 (still on bonding curve)")
            continue

        # Scam evaluation
        scam_result = evaluate_scam_score(address, created_at)

        if scam_result["passed"]:
            passed_count += 1
            add_to_watchlist(
                address=address, symbol=symbol, name=name,
                created_at_ts=created_at,
                mcap=float(t.get("marketCap", 0)),
                holders=t.get("holders", 0),
                scam_passed=True,
            )
            log.info(f"WATCHLIST ADD: {symbol} ({address[:8]}) | "
                     f"holders={scam_result['holders']} swap={scam_result['total_swap']}")
        else:
            blocked_count += 1
            log.debug(f"SCAM BLOCKED: {symbol} ({address[:8]}) | {scam_result['reason']}")

    # Prune expired tokens (> 12h old)
    now = int(time.time())
    for row in get_active_watchlist():
        if row["created_at_ts"] and (now - row["created_at_ts"]) > 12 * 3600:
            remove_from_watchlist(row["token_address"], reason="expired_12h")

    # Enforce watchlist max size
    while watchlist_count() > config.WATCHLIST_MAX:
        oldest = get_active_watchlist()
        if oldest:
            remove_from_watchlist(oldest[-1]["token_address"], reason="watchlist_full")

    log_discovery(len(tokens), new_count, passed_count, blocked_count)
    log.info(f"Discovery cycle: {len(tokens)} found, {new_count} new, "
             f"{passed_count} passed, {blocked_count} blocked, "
             f"watchlist={watchlist_count()}")
