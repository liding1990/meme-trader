"""Paper trade executor — opens/closes positions with Jupiter pricing."""

import logging
from datetime import datetime, timezone, timedelta

from scalper import config
from scalper.db import (
    open_trade, close_trade, get_open_positions, open_positions_count,
    get_closed_trades,
)
from scalper.jupiter_quote import check_liquidity

log = logging.getLogger("scalper.executor")


class PaperExecutor:
    def __init__(self):
        self.capital = config.INITIAL_CAPITAL
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_wins = 0
        self.paused = False
        self.last_reset_date = None

        # Restore capital from DB
        closed = get_closed_trades(limit=10000)
        total_pnl = sum(r["pnl_usd"] for r in closed if r["pnl_usd"])
        held_capital = sum(
            (r["position_size"] or 0) for r in get_open_positions()
        )
        self.capital = config.INITIAL_CAPITAL + total_pnl - held_capital
        log.info(f"Executor init: capital=${self.capital:.2f} (pnl=${total_pnl:.2f}, held=${held_capital:.2f})")

    def reset_daily(self):
        today = datetime.now(timezone.utc).date()
        if self.last_reset_date != today:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_wins = 0
            self.paused = False
            self.last_reset_date = today
            log.info(f"Daily reset: {today}")

    def can_open(self, address: str) -> tuple[bool, str]:
        """Check if we can open a new position."""
        if self.paused:
            return False, "paused"

        if open_positions_count() >= config.MAX_POSITIONS:
            return False, f"max_positions={config.MAX_POSITIONS}"

        if self.daily_trades >= config.MAX_TRADES_PER_DAY:
            return False, f"max_trades={config.MAX_TRADES_PER_DAY}"

        if self.daily_pnl <= -config.DAILY_LOSS_LIMIT:
            self.paused = True
            return False, f"daily_loss_limit=${config.DAILY_LOSS_LIMIT}"

        if self.capital < config.POSITION_SIZE:
            return False, f"insufficient_capital=${self.capital:.2f}"

        # Check win rate pause
        if self.daily_trades >= 10 and self.daily_wins / self.daily_trades < config.WIN_RATE_PAUSE_THRESHOLD:
            self.paused = True
            return False, f"win_rate_pause={self.daily_wins}/{self.daily_trades}"

        # Cooldown check
        closed = get_closed_trades(limit=50)
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=config.COOLDOWN_MINUTES)).isoformat()
        for t in closed:
            if t["token_address"] == address and t["exit_time"] and t["exit_time"] > cutoff:
                return False, "cooldown"

        return True, "ok"

    def open_position(self, signal: dict) -> bool:
        """Open a paper position based on a signal."""
        address = signal["address"]
        symbol = signal["symbol"]
        price = signal["price"]

        can, reason = self.can_open(address)
        if not can:
            log.debug(f"Cannot open {symbol}: {reason}")
            return False

        # Check liquidity via Jupiter
        liq = check_liquidity(address, config.POSITION_SIZE)
        jupiter_impact = liq.get("price_impact_pct")

        if liq["ok"]:
            # Jupiter route available — use real price impact
            effective_price = price * (1 + abs(jupiter_impact or 0) / 100)
        elif liq.get("reason") == "no_quote":
            # Jupiter can't route yet (common for very new graduated tokens)
            # Paper trade with estimated slippage; live would skip
            jupiter_impact = config.BACKTEST_SLIPPAGE_PCT
            effective_price = price * (1 + config.BACKTEST_SLIPPAGE_PCT / 100)
        else:
            # Jupiter route exists but impact too high — skip
            log.info(f"SKIP {symbol}: high price impact ({liq['reason']})")
            return False

        trade_id = open_trade(
            address=address, symbol=symbol,
            entry_price=effective_price, entry_mcap=price,
            entry_holders=signal.get("holders", 0),
            position_size=config.POSITION_SIZE,
            signals=signal,
            jupiter_impact=jupiter_impact,
        )

        self.capital -= config.POSITION_SIZE
        self.daily_trades += 1

        log.info(f"OPEN: {symbol} | ${config.POSITION_SIZE} @ mcap=${price:,.0f} | "
                 f"impact={jupiter_impact:.1f}% | trade#{trade_id}")
        return True

    def close_position(self, address: str, current_price: float, reason: str,
                       peak_mcap: float, bars_held: int):
        """Close a paper position."""
        from scalper.db import get_position
        pos = get_position(address)
        if not pos:
            return

        entry_price = pos["entry_price"]
        position_size = pos["position_size"] or config.POSITION_SIZE

        # Apply exit slippage (estimate 1% or use Jupiter if available)
        exit_price = current_price * 0.99  # 1% sell slippage

        return_pct = (exit_price / entry_price - 1) * 100
        pnl_usd = position_size * (return_pct / 100)

        close_trade(
            address=address, exit_price=exit_price, exit_mcap=current_price,
            exit_reason=reason, return_pct=return_pct, pnl_usd=pnl_usd,
            peak_mcap=peak_mcap, bars_held=bars_held,
        )

        self.capital += position_size + pnl_usd
        self.daily_pnl += pnl_usd
        if pnl_usd > 0:
            self.daily_wins += 1

        log.info(f"CLOSE: {pos['symbol']} | reason={reason} | "
                 f"bars={bars_held} ret={return_pct:+.1f}% pnl=${pnl_usd:+.2f} | "
                 f"capital=${self.capital:.2f}")
