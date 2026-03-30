"""Position management: update prices, check exit conditions, track P&L."""

import logging
import pickle

import numpy as np

from trader.config import (
    TRAILING_STOP_PCT, TIGHT_STOP_PCT, HARD_STOP_PCT,
    HAZARD_THRESHOLD, GRACE_PERIOD_BARS, SURVIVAL_MODEL_PATH,
)

log = logging.getLogger("trader.portfolio")

_survival_model = None


def load_survival_model():
    """Load Kaplan-Meier survival model (lazy, cached)."""
    global _survival_model
    if _survival_model is None:
        try:
            with open(SURVIVAL_MODEL_PATH, "rb") as f:
                d = pickle.load(f)
            _survival_model = d["kmf"]
            log.info(f"Loaded survival model from {SURVIVAL_MODEL_PATH}")
        except Exception as e:
            log.warning(f"Could not load survival model: {e}")
    return _survival_model


def compute_hazard(bars_held):
    """Compute P(pump dies in next bar | survived to bars_held)."""
    kmf = load_survival_model()
    if kmf is None:
        return 0.0

    survival = float(kmf.predict(bars_held))
    survival_next = float(kmf.predict(bars_held + 1))

    if survival > 0:
        hazard = 1 - (survival_next / survival)
    else:
        hazard = 1.0

    return float(np.clip(hazard, 0, 1))


class Position:
    """A single open position."""

    def __init__(self, token_address, symbol, entry_price, position_size, trade_id):
        self.token_address = token_address
        self.symbol = symbol
        self.entry_price = entry_price
        self.position_size = position_size
        self.trade_id = trade_id
        self.peak_price = entry_price
        self.bars_held = 0
        self.current_price = entry_price

    @property
    def pnl_pct(self):
        if self.entry_price <= 0:
            return 0.0
        return (self.current_price / self.entry_price - 1) * 100

    @property
    def pnl_usd(self):
        return self.position_size * (self.pnl_pct / 100)

    @property
    def drawdown_from_peak(self):
        if self.peak_price <= 0:
            return 0.0
        return (1 - self.current_price / self.peak_price) * 100

    def update(self, new_price):
        """Update position with new price. Returns (should_exit, reason, signals)."""
        self.current_price = new_price
        self.bars_held += 1

        if new_price > self.peak_price:
            self.peak_price = new_price

    def to_dict(self):
        return {
            "token_address": self.token_address,
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "position_size": self.position_size,
            "trade_id": self.trade_id,
            "peak_price": self.peak_price,
            "bars_held": self.bars_held,
            "current_price": self.current_price,
            "pnl_pct": self.pnl_pct,
            "pnl_usd": self.pnl_usd,
            "drawdown_from_peak": self.drawdown_from_peak,
        }


def check_exit(position, signals=None):
    """Check all exit conditions for a position.

    Args:
        position: Position object (already updated with current price)
        signals: latest indicator signals dict (for tighten_stop, etc.)

    Returns: (should_exit: bool, reason: str)
    """
    bars = position.bars_held
    pnl = position.pnl_pct
    drawdown = position.drawdown_from_peak

    # 1. Hard stop — always active (catastrophic protection)
    if pnl < -HARD_STOP_PCT:
        return True, "hard_stop"

    # 2. During grace period, only hard stop active
    if bars < GRACE_PERIOD_BARS:
        return False, None

    # 3. Survival hazard
    hazard = compute_hazard(bars)
    if hazard > HAZARD_THRESHOLD:
        return True, "hazard"

    # 4. Trailing stop (adaptive: tight when momentum weakening)
    tighten = False
    if signals:
        tighten = signals.get("tighten_stop", False)

    stop_level = TIGHT_STOP_PCT if tighten else TRAILING_STOP_PCT
    if drawdown > stop_level:
        reason = "tight_stop" if tighten else "trailing_stop"
        return True, reason

    # 5. Momentum collapse — ROC deeply negative
    if signals:
        roc = signals.get("roc_30m", 0)
        if not np.isnan(roc) and roc < -ROC_THRESHOLD_COLLAPSE:
            return True, "momentum_collapse"

    return False, None


# Collapse threshold: 2x entry threshold
ROC_THRESHOLD_COLLAPSE = 6.0  # ROC < -6% = momentum collapsed


def format_position_status(position, signals=None):
    """Format position status for logging."""
    hazard = compute_hazard(position.bars_held)
    tighten = signals.get("tighten_stop", False) if signals else False

    return (
        f"POSITION: {position.symbol:12s} | bars={position.bars_held} "
        f"pnl={position.pnl_pct:+.1f}% peak={position.drawdown_from_peak:.1f}%dd "
        f"hazard={hazard:.2f} tighten={tighten}"
    )
