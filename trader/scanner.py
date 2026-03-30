"""Token discovery and signal generation.

Phase 1: Discover new tokens via Codex filterTokens
Phase 2: Scan watchlist tokens — fetch bars, compute indicators, generate entry signals
"""

import logging
import time
import pickle

import numpy as np
import pandas as pd

from codex_api import filter_tokens, get_token_bars, SOLANA_NETWORK_ID
from indicators import compute_all_indicators, generate_trade_management_signals
from trader.config import (
    CODEX_BAR_COUNTBACK, CODEX_RESOLUTION,
    DISCOVER_MIN_MCAP, DISCOVER_MIN_HOLDERS,
    ROC_THRESHOLD, RVOL_THRESHOLD,
    ENTRY_REQUIRE_OFI, ENTRY_REQUIRE_EBSW, ENTRY_REQUIRE_ITREND,
    META_MODEL_THRESHOLD, META_MODEL_PATH,
)

log = logging.getLogger("trader.scanner")


def discover_tokens(watchlist_addresses):
    """Find new tokens meeting criteria. Returns list of {address, symbol, name}."""
    try:
        tokens, count, _ = filter_tokens(
            min_mcap=DISCOVER_MIN_MCAP,
            min_holders=DISCOVER_MIN_HOLDERS,
            limit=200,
        )
    except Exception as e:
        log.error(f"Discovery failed: {e}")
        return []

    new_tokens = []
    for t in tokens:
        addr = t["token"]["address"]
        if addr not in watchlist_addresses:
            new_tokens.append({
                "address": addr,
                "symbol": t["token"]["symbol"],
                "name": t["token"]["name"],
                "mcap": t.get("marketCap", 0),
                "holders": t.get("holders", 0),
            })

    log.info(f"Discovery: {count} total, {len(tokens)} returned, {len(new_tokens)} new")
    return new_tokens


def fetch_token_data(address):
    """Fetch 5m bars from Codex for a single token. Returns DataFrame or None."""
    try:
        bars = get_token_bars(
            address,
            resolution=CODEX_RESOLUTION,
            countback=CODEX_BAR_COUNTBACK,
        )
    except Exception as e:
        log.debug(f"Failed to fetch bars for {address[:8]}: {e}")
        return None

    if not bars or len(bars) < 20:
        return None

    df = pd.DataFrame(bars)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Codex returns price as 'close', add 'mcap' alias for indicators
    if "close" in df.columns and "mcap" not in df.columns:
        df["mcap"] = df["close"]

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    # Ensure buy/sell columns
    for col in ["buy_volume", "sell_volume", "buyers", "sellers"]:
        if col not in df.columns:
            df[col] = 0

    return df


def _sanitize(val):
    """Convert numpy types to Python native for JSON serialization."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, float) and np.isnan(val):
        return None
    return val


def compute_signals(df):
    """Compute all indicators and return the latest row's signal values."""
    df = compute_all_indicators(df)
    df = generate_trade_management_signals(df)

    if len(df) < 2:
        return None

    latest = df.iloc[-1]

    signals = {
        "roc_30m": _sanitize(latest.get("roc_30m", np.nan)),
        "roc_1h": _sanitize(latest.get("roc_1h", np.nan)),
        "roc_accel_30m": _sanitize(latest.get("roc_accel_30m", np.nan)),
        "roc_accel_1h": _sanitize(latest.get("roc_accel_1h", np.nan)),
        "rvol": _sanitize(latest.get("rvol", np.nan)),
        "ofi_30m": _sanitize(latest.get("ofi_30m", np.nan)),
        "ofi_1h": _sanitize(latest.get("ofi_1h", np.nan)),
        "fisher": _sanitize(latest.get("fisher", np.nan)),
        "fisher_signal": _sanitize(latest.get("fisher_signal", np.nan)),
        "fisher_cross": _sanitize(latest.get("fisher_cross", np.nan)),
        "ebsw": _sanitize(latest.get("ebsw", np.nan)),
        "above_itrend": _sanitize(latest.get("above_itrend", np.nan)),
        "hurst": _sanitize(latest.get("hurst", np.nan)),
        "macd_hist": _sanitize(latest.get("macd_hist", np.nan)),
        "macd_hist_slope": _sanitize(latest.get("macd_hist_slope", np.nan)),
        "bs_ratio": _sanitize(latest.get("bs_ratio", np.nan)),
        "buyer_seller_ratio": _sanitize(latest.get("buyer_seller_ratio", np.nan)),
        "momentum_quality": _sanitize(latest.get("momentum_quality", np.nan)),
        "tighten_stop": bool(latest.get("tighten_stop", False)),
        "price": _sanitize(latest.get("close", latest.get("mcap", 0))),
        "volume": _sanitize(latest.get("volume", 0)),
        "buy_volume": _sanitize(latest.get("buy_volume", 0)),
        "sell_volume": _sanitize(latest.get("sell_volume", 0)),
        "liquidity": _sanitize(latest.get("liquidity", 0)),
    }

    return signals


def check_entry_conditions(signals):
    """Check if entry conditions are met. Returns (pass, reason_if_fail)."""
    roc = signals.get("roc_30m", 0)
    rvol = signals.get("rvol", 0)
    accel = signals.get("roc_accel_30m", 0)
    ofi = signals.get("ofi_30m", np.nan)
    ebsw = signals.get("ebsw", np.nan)
    above_itrend = signals.get("above_itrend", 1)

    if pd.isna(roc) or pd.isna(rvol):
        return False, "missing_data"

    if roc <= ROC_THRESHOLD:
        return False, f"roc={roc:.1f}<{ROC_THRESHOLD}"

    if rvol <= RVOL_THRESHOLD:
        return False, f"rvol={rvol:.1f}<{RVOL_THRESHOLD}"

    if accel <= 0:
        return False, f"accel={accel:.2f}<=0"

    if ENTRY_REQUIRE_OFI and not pd.isna(ofi) and ofi < 0:
        return False, f"ofi={ofi:.2f}<0"

    if ENTRY_REQUIRE_EBSW and not pd.isna(ebsw) and ebsw < 0:
        return False, f"ebsw={ebsw:.2f}<0"

    if ENTRY_REQUIRE_ITREND and above_itrend == 0:
        return False, "below_itrend"

    return True, "pass"


_meta_model = None


def load_meta_model():
    """Load LightGBM meta-model (lazy, cached)."""
    global _meta_model
    if _meta_model is None and META_MODEL_THRESHOLD is not None:
        try:
            with open(META_MODEL_PATH, "rb") as f:
                _meta_model = pickle.load(f)
            log.info(f"Loaded meta-model from {META_MODEL_PATH}")
        except Exception as e:
            log.warning(f"Could not load meta-model: {e}")
    return _meta_model


def apply_meta_filter(signals):
    """Apply LightGBM meta-model filter. Returns (pass, score)."""
    if META_MODEL_THRESHOLD is None:
        return True, None

    model = load_meta_model()
    if model is None:
        return True, None  # no model = no filter

    from meta_model import FEATURE_COLS

    features = {col: signals.get(col, np.nan) for col in FEATURE_COLS}
    X = pd.DataFrame([features])
    X = X.replace([np.inf, -np.inf], np.nan)

    try:
        score = model.predict_proba(X)[0, 1]
    except Exception as e:
        log.debug(f"Meta-model prediction failed: {e}")
        return True, None

    return score >= META_MODEL_THRESHOLD, float(score)


def scan_token(address, symbol):
    """Full scan of a single token: fetch → compute → check entry.

    Returns (signals_dict, entry_pass, meta_score) or (None, False, None) on failure.
    """
    df = fetch_token_data(address)
    if df is None:
        return None, False, None

    signals = compute_signals(df)
    if signals is None:
        return None, False, None

    entry_pass, reason = check_entry_conditions(signals)

    if not entry_pass:
        log.debug(f"SCAN {symbol:12s} | SKIP ({reason})")
        return signals, False, None

    # Meta-model filter
    meta_pass, meta_score = apply_meta_filter(signals)

    if not meta_pass:
        log.debug(f"SCAN {symbol:12s} | ENTRY but meta={meta_score:.2f}<{META_MODEL_THRESHOLD}")
        return signals, False, meta_score

    log.info(
        f"ENTRY SIGNAL: {symbol} | ROC={signals['roc_30m']:.1f}% RVOL={signals['rvol']:.1f} "
        f"OFI={signals['ofi_30m']:.2f} EBSW={signals['ebsw']:.2f} "
        f"Fisher={signals['fisher']:.2f} Meta={meta_score or 'N/A'} | ${signals['price']:.8f}"
    )

    return signals, True, meta_score
