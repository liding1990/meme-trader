"""Structured logging for funnel bot.

Two log files:
- funnel.log: key events only (scan results, entries, exits, L3 actions)
- funnel.debug.log: everything including feature values, model scores, prices

Both rotate daily. JSON-structured for easy parsing in post-analysis.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def _json_formatter():
    """Custom formatter that outputs structured JSON per line."""
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "layer": getattr(record, "layer", ""),
                "event": record.getMessage(),
            }
            # Merge extra data if attached
            if hasattr(record, "data") and record.data:
                entry["data"] = record.data
            return json.dumps(entry, ensure_ascii=False, default=str)
    return JsonFormatter()


def setup_logger():
    """Set up and return the main logger with console + file handlers."""
    logger = logging.getLogger("funnel_bot")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger  # already set up

    # Console: INFO level, human-readable
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(console)

    # Main log: INFO level, JSON, daily rotation
    main_handler = TimedRotatingFileHandler(
        os.path.join(LOG_DIR, "funnel.log"),
        when="midnight", backupCount=30, utc=True,
    )
    main_handler.setLevel(logging.INFO)
    main_handler.setFormatter(_json_formatter())
    logger.addHandler(main_handler)

    # Debug log: DEBUG level, JSON, daily rotation
    debug_handler = TimedRotatingFileHandler(
        os.path.join(LOG_DIR, "funnel.debug.log"),
        when="midnight", backupCount=14, utc=True,
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(_json_formatter())
    logger.addHandler(debug_handler)

    return logger


class FunnelLogger:
    """Convenience wrapper for structured logging with layer context."""

    def __init__(self):
        self.logger = setup_logger()

    def _log(self, level, layer, event, data=None):
        record = self.logger.makeRecord(
            "funnel_bot", level, "", 0, event, (), None
        )
        record.layer = layer
        record.data = data
        self.logger.handle(record)

    # ── L0 ──
    def l0_scan_start(self):
        self._log(logging.INFO, "L0", "scan_start")

    def l0_scan_result(self, n_candidates, candidates=None):
        self._log(logging.INFO, "L0", f"scan_result: {n_candidates} candidates",
                  {"n_candidates": n_candidates})
        if candidates:
            for c in candidates[:10]:
                tok = c.get("token", {})
                self._log(logging.DEBUG, "L0", f"candidate: {tok.get('symbol','?')}",
                          {"address": tok.get("address"), "symbol": tok.get("symbol"),
                           "mcap": c.get("marketCap"), "holders": c.get("holders")})

    # ── L1 ──
    def l1_filter(self, address, symbol, passed, reason, mcap=0, holders=0):
        level = logging.DEBUG if passed else logging.INFO
        self._log(level, "L1", f"{'PASS' if passed else 'FILTER'} {symbol}: {reason}",
                  {"address": address, "symbol": symbol, "passed": passed,
                   "reason": reason, "mcap": mcap, "holders": holders})

    def l1_summary(self, passed, filtered):
        self._log(logging.INFO, "L1", f"summary: {passed} passed, {filtered} filtered",
                  {"passed": passed, "filtered": filtered})

    # ── L2 ──
    def l2_predict(self, address, symbol, proba, should_enter, features=None):
        level = logging.INFO if should_enter else logging.DEBUG
        self._log(level, "L2", f"{'SIGNAL' if should_enter else 'skip'} {symbol}: proba={proba:.3f}",
                  {"address": address, "symbol": symbol, "proba": proba,
                   "should_enter": should_enter})
        if features:
            self._log(logging.DEBUG, "L2", f"features: {symbol}",
                      {"address": address, "features": features})

    def l2_summary(self, signals, rejected):
        self._log(logging.INFO, "L2", f"summary: {signals} signals, {rejected} rejected",
                  {"signals": signals, "rejected": rejected})

    # ── L3 ──
    def l3_action(self, address, symbol, action, price, pnl_pct,
                   remaining_before, remaining_after, features=None):
        self._log(logging.INFO, "L3", f"{action} {symbol}: pnl={pnl_pct:+.1f}% remain={remaining_after*100:.0f}%",
                  {"address": address, "symbol": symbol, "action": action,
                   "price": price, "pnl_pct": pnl_pct,
                   "remaining_before": remaining_before, "remaining_after": remaining_after})
        if features:
            self._log(logging.DEBUG, "L3", f"features: {symbol}",
                      {"address": address, "features": features})

    def l3_hold(self, address, symbol, price, pnl_pct, remaining):
        self._log(logging.DEBUG, "L3", f"HOLD {symbol}: pnl={pnl_pct:+.1f}% remain={remaining*100:.0f}%",
                  {"address": address, "symbol": symbol, "price": price,
                   "pnl_pct": pnl_pct, "remaining": remaining})

    # ── Trade Events ──
    def entry(self, address, symbol, price, position_size, l2_proba, l1_label):
        self._log(logging.INFO, "TRADE", f"ENTRY {symbol}: price=${price:,.0f} size=${position_size} L2={l2_proba:.2f}",
                  {"address": address, "symbol": symbol, "price": price,
                   "position_size": position_size, "l2_proba": l2_proba, "l1_label": l1_label})

    def exit(self, address, symbol, return_pct, pnl_usd, reason, bars_held):
        self._log(logging.INFO, "TRADE", f"EXIT {symbol}: {return_pct:+.1f}% (${pnl_usd:+.1f}) reason={reason} bars={bars_held}",
                  {"address": address, "symbol": symbol, "return_pct": return_pct,
                   "pnl_usd": pnl_usd, "reason": reason, "bars_held": bars_held})

    # ── System ──
    def tick_start(self, tick, capital, n_positions):
        self._log(logging.INFO, "SYS", f"tick {tick}: capital=${capital:,.0f} positions={n_positions}",
                  {"tick": tick, "capital": capital, "positions": n_positions})

    def tick_end(self, tick, duration_ms):
        self._log(logging.DEBUG, "SYS", f"tick {tick} done: {duration_ms:.0f}ms",
                  {"tick": tick, "duration_ms": duration_ms})

    def error(self, msg, exc=None):
        self._log(logging.ERROR, "SYS", f"ERROR: {msg}",
                  {"error": str(exc) if exc else msg})

    def equity_snapshot(self, capital, positions, equity, daily_pnl, daily_trades, daily_wins):
        self._log(logging.INFO, "SYS", f"equity: ${equity:,.0f} pnl=${daily_pnl:+.1f} trades={daily_trades} wins={daily_wins}",
                  {"capital": capital, "open_positions": positions, "total_equity": equity,
                   "daily_pnl": daily_pnl, "daily_trades": daily_trades, "daily_wins": daily_wins})
