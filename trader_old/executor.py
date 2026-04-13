"""Paper trade executor: simulates fills, manages capital, enforces risk limits."""

import logging
from datetime import datetime, timezone, timedelta

from trader.config import (
    MAX_POSITIONS, POSITION_SIZE, INITIAL_CAPITAL,
    DAILY_LOSS_LIMIT, COOLDOWN_HOURS,
)
from trader.db import TradingDB
from trader.portfolio import Position

log = logging.getLogger("trader.executor")


class PaperExecutor:
    """Simulates trade execution for paper trading."""

    def __init__(self, db: TradingDB):
        self.db = db
        self.positions = {}  # address → Position
        self.capital = INITIAL_CAPITAL
        self.daily_pnl = 0.0
        self.paused = False  # True when daily loss limit hit

        # Restore positions from DB
        self._restore_positions()

    def _restore_positions(self):
        """Restore open positions from database after restart."""
        saved = self.db.get_positions()
        for p in saved:
            pos = Position(
                token_address=p["token_address"],
                symbol=p["symbol"],
                entry_price=p["entry_price"],
                position_size=p["position_size"],
                trade_id=p["trade_id"],
            )
            pos.peak_price = p["peak_price"]
            pos.bars_held = p["bars_held"]
            self.positions[p["token_address"]] = pos

        if self.positions:
            log.info(f"Restored {len(self.positions)} positions from database")

        # Restore daily PnL
        self.daily_pnl = self.db.get_daily_pnl()

    def can_open_position(self, token_address):
        """Check if we can open a new position."""
        # Max positions
        if len(self.positions) >= MAX_POSITIONS:
            return False, "max_positions"

        # Already holding this token
        if token_address in self.positions:
            return False, "already_holding"

        # Daily loss limit
        if self.daily_pnl < -DAILY_LOSS_LIMIT:
            if not self.paused:
                log.warning(f"DAILY LOSS LIMIT HIT: ${self.daily_pnl:.2f}. Pausing new entries.")
                self.paused = True
            return False, "daily_loss_limit"

        # Cooldown after loss on same token
        last_loss = self.db.get_recent_loss_time(token_address)
        if last_loss:
            try:
                loss_time = datetime.fromisoformat(last_loss)
                cooldown_end = loss_time + timedelta(hours=COOLDOWN_HOURS)
                if datetime.now(timezone.utc) < cooldown_end:
                    return False, "cooldown"
            except (ValueError, TypeError):
                pass

        # Sufficient capital
        if self.capital < POSITION_SIZE:
            return False, "insufficient_capital"

        return True, "ok"

    def open_position(self, token_address, symbol, price, signals, meta_score=None):
        """Open a paper position."""
        can_open, reason = self.can_open_position(token_address)
        if not can_open:
            log.debug(f"Cannot open {symbol}: {reason}")
            return None

        # Record in DB
        trade_id = self.db.open_trade(
            token_address=token_address,
            symbol=symbol,
            entry_price=price,
            position_size=POSITION_SIZE,
            entry_signals=signals,
            meta_score=meta_score,
        )

        # Create position
        pos = Position(token_address, symbol, price, POSITION_SIZE, trade_id)
        self.positions[token_address] = pos

        # Persist position
        self.db.save_position(
            token_address, symbol,
            datetime.now(timezone.utc).isoformat(),
            price, POSITION_SIZE, price, 0, trade_id,
        )

        # Deduct capital
        self.capital -= POSITION_SIZE

        log.info(
            f"OPEN: {symbol} | ${POSITION_SIZE:.0f} @ ${price:.8f} | "
            f"positions: {len(self.positions)}/{MAX_POSITIONS} | "
            f"capital: ${self.capital:.0f}"
        )

        return pos

    def close_position(self, token_address, reason, exit_signals=None):
        """Close a paper position."""
        pos = self.positions.get(token_address)
        if pos is None:
            return None

        # Calculate PnL
        return_pct = pos.pnl_pct
        pnl_usd = pos.pnl_usd

        # Record in DB
        self.db.close_trade(
            trade_id=pos.trade_id,
            exit_price=pos.current_price,
            exit_reason=reason,
            exit_signals=exit_signals,
            return_pct=return_pct,
            pnl_usd=pnl_usd,
            peak_price=pos.peak_price,
            bars_held=pos.bars_held,
        )

        # Remove position
        del self.positions[token_address]
        self.db.remove_position(token_address)

        # Update capital and daily PnL
        self.capital += POSITION_SIZE + pnl_usd
        self.daily_pnl += pnl_usd

        # Stats
        stats = self.db.get_summary_stats()
        total_trades = stats["total_trades"] or 0
        wins = stats["wins"] or 0
        total_pnl = stats["total_pnl"] or 0
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        log.info(
            f"CLOSE: {pos.symbol} | reason={reason} | bars={pos.bars_held} "
            f"pnl={return_pct:+.1f}% (${pnl_usd:+.2f}) | "
            f"cumulative: ${total_pnl:+.2f} | win_rate={win_rate:.0f}% ({wins}/{total_trades})"
        )

        return pos

    def reset_daily(self):
        """Reset daily counters (call at start of new day)."""
        self.daily_pnl = 0.0
        self.paused = False
        log.info("Daily reset: PnL counter and pause flag cleared")

    def get_status(self):
        """Get current executor status for logging."""
        return {
            "positions": len(self.positions),
            "max_positions": MAX_POSITIONS,
            "capital": self.capital,
            "daily_pnl": self.daily_pnl,
            "paused": self.paused,
        }
