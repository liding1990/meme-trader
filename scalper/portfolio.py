"""Position management + exit logic.

Exit conditions:
  1. Trailing stop: -20% from peak
  2. Hard stop: -30% from entry
  3. Time stop: 90 min (18 bars)
  4. Holder exit: holders declining 3 consecutive checks
"""

import logging

from scalper import config
from scalper.db import get_open_positions, update_position, get_position

log = logging.getLogger("scalper.portfolio")


def check_exit(position: dict, current_price: float, holders_declining: bool = False) -> tuple[bool, str | None]:
    """Check if a position should be exited.

    Args:
        position: DB row from scalper_positions
        current_price: current mcap/price
        holders_declining: True if holders have been declining for 3+ checks

    Returns:
        (should_exit, reason)
    """
    entry_price = position["entry_price"]
    peak_mcap = position["peak_mcap"]
    bars_held = position["bars_held"] or 0

    if current_price <= 0 or entry_price <= 0:
        return True, "invalid_price"

    pnl_pct = (current_price / entry_price - 1) * 100
    drawdown_from_peak = (1 - current_price / peak_mcap) * 100 if peak_mcap > 0 else 0

    # 1. Trailing stop: -20% from peak
    if drawdown_from_peak >= config.TRAILING_STOP_PCT:
        return True, "trailing_stop"

    # 2. Hard stop: -30% from entry
    if pnl_pct <= -config.HARD_STOP_PCT:
        return True, "hard_stop"

    # 3. Time stop: 18 bars (90 min)
    if bars_held >= config.TIME_STOP_BARS:
        return True, "time_stop"

    # 4. Holder exit: sustained holder decline
    if holders_declining and bars_held > 3:
        return True, "holder_exit"

    return False, None


def update_peak(address: str, current_price: float):
    """Update peak price if new high."""
    pos = get_position(address)
    if pos and current_price > (pos["peak_mcap"] or 0):
        update_position(address, peak_mcap=current_price)


def increment_bars(address: str):
    """Increment bars held counter."""
    pos = get_position(address)
    if pos:
        update_position(address, bars_held=(pos["bars_held"] or 0) + 1)
