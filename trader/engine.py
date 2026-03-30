"""Main trading engine: orchestrates discovery, scanning, and position management.

Usage:
    python -m trader.engine
"""

import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime, timezone

from trader.config import (
    TICK_INTERVAL, DISCOVER_INTERVAL, LOG_DIR, LOG_LEVEL, LOG_ROTATE_DAYS,
    WATCHLIST_MAX, WATCHLIST_REMOVE_BELOW_MCAP, WATCHLIST_REMOVE_INACTIVE_HOURS,
)
from trader.db import TradingDB
from trader.scanner import discover_tokens, scan_token, fetch_token_data, compute_signals
from trader.portfolio import Position, check_exit, format_position_status
from trader.executor import PaperExecutor

log = logging.getLogger("trader")


def setup_logging():
    """Configure dual logging: INFO to console+file, DEBUG to separate file."""
    os.makedirs(LOG_DIR, exist_ok=True)

    # Root logger
    root = logging.getLogger("trader")
    root.setLevel(logging.DEBUG)

    # Console handler (INFO)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, LOG_LEVEL))
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(console)

    # File handler (INFO, daily rotation)
    info_handler = logging.handlers.TimedRotatingFileHandler(
        os.path.join(LOG_DIR, "trader.log"),
        when="midnight", backupCount=LOG_ROTATE_DAYS, utc=True,
    )
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(info_handler)

    # Debug file handler (DEBUG, daily rotation)
    debug_handler = logging.handlers.TimedRotatingFileHandler(
        os.path.join(LOG_DIR, "trader.debug.log"),
        when="midnight", backupCount=7, utc=True,
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(debug_handler)

    return root


def run():
    """Main trading loop."""
    setup_logging()
    log.info("=" * 60)
    log.info("  Meme Trader Paper Trading System Starting")
    log.info("=" * 60)

    # Initialize
    db = TradingDB()
    executor = PaperExecutor(db)

    # Load watchlist from DB
    watchlist = {t["token_address"]: t for t in db.get_active_watchlist()}
    log.info(f"Watchlist: {len(watchlist)} tokens")
    log.info(f"Open positions: {len(executor.positions)}")
    log.info(f"Capital: ${executor.capital:,.0f}")

    last_discover = 0
    last_day = datetime.now(timezone.utc).date()
    tick_count = 0

    while True:
        tick_start = time.time()
        tick_count += 1
        errors = []
        signals_generated = 0
        signals_filtered = 0
        tokens_scanned = 0

        try:
            today = datetime.now(timezone.utc).date()

            # Daily reset
            if today != last_day:
                executor.reset_daily()
                last_day = today
                log.info(f"=== New Day: {today} ===")

            # === Phase 1: Discover (hourly) ===
            if time.time() - last_discover > DISCOVER_INTERVAL:
                watchlist_addrs = set(watchlist.keys())
                new_tokens = discover_tokens(watchlist_addrs)

                for t in new_tokens:
                    if len(watchlist) < WATCHLIST_MAX:
                        watchlist[t["address"]] = t
                        db.add_to_watchlist(t["address"], t["symbol"], t.get("name"))

                last_discover = time.time()

            # === Phase 2: Scan watchlist for entry signals ===
            entry_candidates = []

            for address, token_info in list(watchlist.items()):
                symbol = token_info.get("symbol", address[:8])

                signals, entry_pass, meta_score = scan_token(address, symbol)
                tokens_scanned += 1

                if signals:
                    # Update watchlist metadata
                    db.update_watchlist_token(
                        address,
                        mcap=signals.get("price", 0),
                        volume=signals.get("volume", 0),
                    )

                if entry_pass:
                    signals_generated += 1
                    signals_filtered += 1
                    entry_candidates.append({
                        "address": address,
                        "symbol": symbol,
                        "signals": signals,
                        "meta_score": meta_score,
                    })

            # === Phase 3: Manage existing positions ===
            for address in list(executor.positions.keys()):
                pos = executor.positions[address]

                # Fetch latest price
                df = fetch_token_data(address)
                if df is None or len(df) == 0:
                    continue

                latest_price = df.iloc[-1].get("close", df.iloc[-1].get("mcap", 0))
                if latest_price <= 0:
                    continue

                # Update position
                pos.update(float(latest_price))

                # Get latest signals for exit management
                signals = compute_signals(df)

                # Check exit
                should_exit, reason = check_exit(pos, signals)

                log.info(format_position_status(pos, signals) + f" | {'EXIT:'+reason if should_exit else 'HOLD'}")

                if should_exit:
                    executor.close_position(address, reason, exit_signals=signals)

                    # Persist remaining positions
                    for addr, p in executor.positions.items():
                        db.save_position(
                            addr, p.symbol, "", p.entry_price,
                            p.position_size, p.peak_price, p.bars_held, p.trade_id,
                        )

            # === Phase 4: Open new positions ===
            for candidate in entry_candidates:
                pos = executor.open_position(
                    token_address=candidate["address"],
                    symbol=candidate["symbol"],
                    price=candidate["signals"]["price"],
                    signals=candidate["signals"],
                    meta_score=candidate["meta_score"],
                )
                if pos:
                    # Persist
                    db.save_position(
                        pos.token_address, pos.symbol,
                        datetime.now(timezone.utc).isoformat(),
                        pos.entry_price, pos.position_size,
                        pos.peak_price, pos.bars_held, pos.trade_id,
                    )

        except Exception as e:
            log.error(f"TICK #{tick_count} ERROR: {e}", exc_info=True)
            errors.append(str(e))

        # === Log tick ===
        tick_duration = int((time.time() - tick_start) * 1000)

        log.info(
            f"TICK #{tick_count} | watchlist={len(watchlist)} scanned={tokens_scanned} "
            f"signals={signals_generated} positions={len(executor.positions)} "
            f"daily_pnl=${executor.daily_pnl:+.2f} | {tick_duration/1000:.1f}s"
        )

        db.log_tick(
            tick_number=tick_count,
            watchlist_size=len(watchlist),
            tokens_scanned=tokens_scanned,
            signals_generated=signals_generated,
            signals_filtered=signals_filtered,
            open_positions=len(executor.positions),
            tick_duration_ms=tick_duration,
            daily_pnl=executor.daily_pnl,
            errors="; ".join(errors) if errors else None,
        )

        # === Wait for next tick ===
        elapsed = time.time() - tick_start
        sleep_time = max(0, TICK_INTERVAL - elapsed)

        if sleep_time > 0:
            log.debug(f"Sleeping {sleep_time:.0f}s until next tick")
            time.sleep(sleep_time)


if __name__ == "__main__":
    run()
