"""Scalper V4 engine — main trading loop.

Every 60s: scan watchlist → manage positions → open new trades
Every 10min: discover new tokens + scam filter
"""

import logging
import sys
import time
from datetime import datetime, timezone

from scalper import config
from scalper.db import (
    get_open_positions, record_equity, log_tick, open_positions_count,
)
from scalper.discovery import run_discovery_cycle
from scalper.scanner import scan_watchlist
from scalper.portfolio import check_exit, update_peak, increment_bars
from scalper.executor import PaperExecutor

log = logging.getLogger("scalper")


def setup_logging():
    import os
    os.makedirs(config.LOG_DIR, exist_ok=True)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # File
    fh = logging.FileHandler(os.path.join(config.LOG_DIR, "scalper.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    # Debug file
    dh = logging.FileHandler(os.path.join(config.LOG_DIR, "scalper.debug.log"))
    dh.setLevel(logging.DEBUG)
    dh.setFormatter(fmt)

    root = logging.getLogger("scalper")
    root.setLevel(logging.DEBUG)
    root.addHandler(ch)
    root.addHandler(fh)
    root.addHandler(dh)


def _get_current_price(address: str) -> float | None:
    """Quick price check via GMGN (faster than Codex bars)."""
    from scalper.price import get_mcap
    return get_mcap(address)


def _check_holders_declining(address: str) -> bool:
    """Check if holders have been declining."""
    from scalper.scanner import _get_recent_holders
    data = _get_recent_holders(address)
    if not data or len(data) < 3:
        return False
    # Check last 3 data points
    recent = data[:3]
    return all(d["netHolderChange"] < 0 for d in recent)


def run():
    setup_logging()
    log.info("=" * 60)
    log.info("Scalper V4 — Pump Rider Strategy")
    log.info(f"Capital: ${config.INITIAL_CAPITAL:,.0f}")
    log.info(f"Position size: ${config.POSITION_SIZE}")
    log.info(f"Max positions: {config.MAX_POSITIONS}")
    log.info("=" * 60)

    executor = PaperExecutor()
    tick = 0
    last_discovery = 0

    while True:
        tick_start = time.time()
        tick += 1
        executor.reset_daily()

        # === Phase 1: Discovery (every 10 min) ===
        if time.time() - last_discovery > config.DISCOVER_INTERVAL:
            try:
                run_discovery_cycle()
                last_discovery = time.time()
            except Exception as e:
                log.error(f"Discovery error: {e}")

        # === Phase 2: Manage open positions ===
        positions = get_open_positions()
        for pos in positions:
            addr = pos["token_address"]
            try:
                price = _get_current_price(addr)
                if price is None:
                    continue

                update_peak(addr, price)
                increment_bars(addr)

                holders_declining = _check_holders_declining(addr)
                should_exit, reason = check_exit(pos, price, holders_declining)

                if should_exit:
                    executor.close_position(
                        addr, price, reason,
                        peak_mcap=max(pos["peak_mcap"] or 0, price),
                        bars_held=(pos["bars_held"] or 0) + 1,
                    )
            except Exception as e:
                log.error(f"Position mgmt error {addr[:8]}: {e}")

            time.sleep(config.SCAN_RATE_LIMIT)

        # === Phase 3: Scan for new entries ===
        signals = []
        try:
            signals = scan_watchlist()
        except Exception as e:
            log.error(f"Scan error: {e}")

        # === Phase 4: Open new positions ===
        opened = 0
        for signal in signals:
            if executor.open_position(signal):
                opened += 1

        # === Phase 5: Log tick ===
        tick_duration = int((time.time() - tick_start) * 1000)
        n_positions = open_positions_count()

        log_tick(tick, 0, len(signals), len(signals), n_positions, executor.daily_pnl, tick_duration)

        # Record equity every 10 ticks (~10 min)
        if tick % 10 == 0:
            record_equity(
                capital=executor.capital,
                open_positions=n_positions,
                unrealized_pnl=0,  # simplified
                total_equity=executor.capital,
                daily_pnl=executor.daily_pnl,
                daily_trades=executor.daily_trades,
                daily_wins=executor.daily_wins,
            )

        log.info(f"TICK #{tick} | positions={n_positions} signals={len(signals)} "
                 f"opened={opened} daily_pnl=${executor.daily_pnl:+.2f} "
                 f"capital=${executor.capital:.2f} | {tick_duration}ms")

        # Sleep until next tick
        elapsed = time.time() - tick_start
        sleep_time = max(0, config.TICK_INTERVAL - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
