"""Momentum quality indicators for 5-minute bar trading.

Includes P3 (Hurst exponent) and P4 (ROC, MACD, RVOL, OFI) indicators.

P4 indicators (computationally trivial, no training needed):
  - ROC (Rate of Change) at multiple timeframes
  - ROC acceleration (2nd derivative of price — momentum of momentum)
  - MACD histogram slope (momentum direction change)
  - Relative Volume (RVOL — current volume vs rolling average)
  - Buy/Sell pressure ratio (from GMGN candle data)

All indicators are computed online (streaming-compatible).
"""

import numpy as np
import pandas as pd


def compute_hurst(series, min_window=4, max_window=None):
    """Compute the Hurst exponent using R/S analysis.

    H > 0.5: trending (persistent) — good for momentum
    H = 0.5: random walk — no edge
    H < 0.5: mean-reverting — bad for momentum

    Returns scalar Hurst exponent.
    """
    data = np.asarray(series, dtype=float)
    n = len(data)
    if n < 20:
        return 0.5  # insufficient data

    if max_window is None:
        max_window = n // 2

    windows = []
    rs_values = []

    window_size = min_window
    while window_size <= max_window:
        n_windows = n // window_size
        if n_windows < 1:
            break

        rs_list = []
        for i in range(n_windows):
            chunk = data[i * window_size:(i + 1) * window_size]
            mean = chunk.mean()
            deviations = chunk - mean
            cumdev = np.cumsum(deviations)
            R = cumdev.max() - cumdev.min()
            S = chunk.std(ddof=1)
            if S > 0:
                rs_list.append(R / S)

        if rs_list:
            windows.append(window_size)
            rs_values.append(np.mean(rs_list))

        window_size = int(window_size * 1.5)
        if window_size == int(window_size / 1.5):
            window_size += 1

    if len(windows) < 3:
        return 0.5

    log_w = np.log(windows)
    log_rs = np.log(rs_values)
    hurst, _ = np.polyfit(log_w, log_rs, 1)

    return float(np.clip(hurst, 0, 1))


def compute_rolling_hurst(series, window=48):
    """Compute rolling Hurst exponent over a window.

    window=48 = 4 hours of 5-min bars.
    """
    result = pd.Series(0.5, index=series.index)
    values = series.values

    for i in range(window, len(values)):
        chunk = values[i - window:i]
        if np.std(chunk) > 0:
            result.iloc[i] = compute_hurst(chunk)

    return result


def compute_roc(series, period):
    """Rate of Change: (price[t] / price[t-period] - 1) * 100."""
    return (series / series.shift(period) - 1) * 100


def compute_roc_acceleration(series, roc_period=6, accel_period=3):
    """2nd derivative of price: change in ROC over time.

    Positive = momentum accelerating (price going up faster)
    Negative = momentum decelerating (price still up but slowing)
    """
    roc = compute_roc(series, roc_period)
    accel = roc.diff(accel_period)
    return roc, accel


def compute_macd(series, fast=6, slow=13, signal=5):
    """MACD with shortened parameters for 5-min crypto.

    Returns (macd_line, signal_line, histogram, histogram_slope)
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    histogram_slope = histogram.diff()
    return macd_line, signal_line, histogram, histogram_slope


def compute_rvol(volume, window=48):
    """Relative Volume: current volume / rolling average.

    window=48 = 4 hours of 5-min bars.
    RVOL > 3 = unusually high activity.
    """
    avg_vol = volume.rolling(window, min_periods=1).mean()
    return volume / avg_vol.replace(0, np.nan)


def compute_buy_sell_ratio(buy_volume, sell_volume, window=6):
    """Buy/sell pressure ratio over rolling window.

    > 1 = net buying pressure
    < 1 = net selling pressure
    """
    buy_sum = buy_volume.rolling(window, min_periods=1).sum()
    sell_sum = sell_volume.rolling(window, min_periods=1).sum()
    return buy_sum / sell_sum.replace(0, np.nan)


def compute_ofi(buy_volume, sell_volume, window=6):
    """Order Flow Imbalance: net buying pressure in [-1, 1].

    OFI > 0.3 = strong buying pressure
    OFI < -0.3 = strong selling pressure
    """
    buy_sum = buy_volume.rolling(window, min_periods=1).sum()
    sell_sum = sell_volume.rolling(window, min_periods=1).sum()
    total = buy_sum + sell_sum
    return (buy_sum - sell_sum) / total.replace(0, np.nan)


def compute_all_indicators(df):
    """Compute all indicators on a DataFrame.

    Supports both GMGN data (mcap, volume) and Codex data (+ buy_volume, sell_volume, buyers, sellers).
    Returns DataFrame with all indicator columns added.
    """
    result = df.copy()

    mcap = result["mcap"]
    volume = result["volume"] if "volume" in result.columns else pd.Series(0, index=result.index)

    # ROC at multiple timeframes (6 bars = 30min, 12 = 1h, 36 = 3h)
    result["roc_30m"], result["roc_accel_30m"] = compute_roc_acceleration(mcap, roc_period=6, accel_period=3)
    result["roc_1h"], result["roc_accel_1h"] = compute_roc_acceleration(mcap, roc_period=12, accel_period=6)
    result["roc_3h"], result["roc_accel_3h"] = compute_roc_acceleration(mcap, roc_period=36, accel_period=12)

    # MACD
    result["macd"], result["macd_signal"], result["macd_hist"], result["macd_hist_slope"] = compute_macd(mcap)

    # Relative volume
    result["rvol"] = compute_rvol(volume)

    # Hurst exponent (P3 — trend strength for position sizing)
    result["hurst"] = compute_rolling_hurst(mcap, window=48)

    # Buy/sell indicators (Codex data only)
    if "buy_volume" in result.columns and result["buy_volume"].sum() > 0:
        bv = result["buy_volume"].fillna(0)
        sv = result["sell_volume"].fillna(0)

        # Order Flow Imbalance (rolling 30min = 6 bars)
        result["ofi_30m"] = compute_ofi(bv, sv, window=6)
        result["ofi_1h"] = compute_ofi(bv, sv, window=12)

        # Buy/sell ratio (rolling)
        result["bs_ratio"] = compute_buy_sell_ratio(bv, sv, window=6)

        # Buyer/seller count ratio
        if "buyers" in result.columns:
            buyers = result["buyers"].fillna(0)
            sellers = result["sellers"].fillna(0).replace(0, np.nan)
            result["buyer_seller_ratio"] = (
                buyers.rolling(6, min_periods=1).sum() /
                sellers.rolling(6, min_periods=1).sum()
            )
    else:
        result["ofi_30m"] = np.nan
        result["ofi_1h"] = np.nan
        result["bs_ratio"] = np.nan
        result["buyer_seller_ratio"] = np.nan

    return result


def generate_trade_management_signals(indicators_df):
    """Generate intra-trade management signals from P4 indicators.

    Returns DataFrame with columns:
      - momentum_quality: composite score [-1, 1]
      - tighten_stop: True if momentum weakening (should tighten trailing stop)
      - extend_hold: True if momentum accelerating (can extend hold time)
    """
    df = indicators_df.copy()

    # Momentum quality composite: average of normalized signals
    signals = []

    # ROC acceleration: positive = good
    for col in ["roc_accel_30m", "roc_accel_1h", "roc_accel_3h"]:
        if col in df.columns:
            s = df[col]
            std = s.std()
            if std > 0:
                signals.append(np.clip(s / std, -2, 2) / 2)  # normalize to ~[-1, 1]

    # MACD histogram slope: positive = good
    if "macd_hist_slope" in df.columns:
        s = df["macd_hist_slope"]
        std = s.std()
        if std > 0:
            signals.append(np.clip(s / std, -2, 2) / 2)

    # RVOL: high = good (but normalize differently)
    if "rvol" in df.columns:
        rvol_signal = np.clip((df["rvol"] - 1) / 3, -1, 1)  # RVOL=4 → signal=1
        signals.append(rvol_signal)

    # OFI signal (if available)
    if "ofi_30m" in df.columns and df["ofi_30m"].notna().any():
        signals.append(df["ofi_30m"].fillna(0))  # already in [-1, 1]

    if signals:
        df["momentum_quality"] = pd.concat(signals, axis=1).mean(axis=1)
    else:
        df["momentum_quality"] = 0.0

    # Trade management signals
    df["tighten_stop"] = (
        (df.get("roc_accel_30m", 0) < 0) &
        (df.get("macd_hist_slope", 0) < 0)
    )
    # OFI turning negative = selling pressure = tighten stop
    if "ofi_30m" in df.columns:
        df["tighten_stop"] = df["tighten_stop"] | (df["ofi_30m"].fillna(0) < -0.3)

    df["extend_hold"] = (
        (df.get("roc_accel_30m", 0) > 0) &
        (df.get("rvol", 0) > 2)
    )

    return df


def extract_5m_candles(address):
    """Load 5-minute candle data. Prefers Codex (has buy/sell volume), falls back to GMGN."""
    import os
    import json
    import glob

    # Try Codex data first (has buy/sell volume)
    codex_file = os.path.join("data", "codex", f"{address}_5m.json")
    if os.path.isfile(codex_file):
        with open(codex_file) as f:
            bars = json.load(f)
        if bars:
            df = pd.DataFrame(bars)
            df = df.sort_values("timestamp").reset_index(drop=True)
            # Codex close is price, compute mcap proxy (use close directly)
            if "close" in df.columns and "mcap" not in df.columns:
                df["mcap"] = df["close"]
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
            # Ensure buy/sell columns exist
            for col in ["buy_volume", "sell_volume", "buyers", "sellers"]:
                if col not in df.columns:
                    df[col] = 0
            return df

    # Fall back to GMGN data (no buy/sell volume)
    data_dir = os.path.join("data", address)
    files = sorted(glob.glob(os.path.join(data_dir, "token_mcap_candles_5m_*.json")))
    if not files:
        return None

    with open(files[-1]) as f:
        data = json.load(f)

    candles = data.get("data", {}).get("list", [])
    if not candles:
        return None

    rows = []
    for c in candles:
        t = int(c["time"])
        if t > 1e12:
            t = t // 1000
        rows.append({
            "timestamp": t,
            "mcap": float(c.get("close", 0)),
            "volume": float(c.get("volume", 0)),
            "high": float(c.get("high", 0)),
            "low": float(c.get("low", 0)),
            "open": float(c.get("open", 0)),
            "buy_volume": 0,
            "sell_volume": 0,
            "buyers": 0,
            "sellers": 0,
        })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


if __name__ == "__main__":
    """Quick test: compute indicators on a sample token."""
    import sys

    address = sys.argv[1] if len(sys.argv) > 1 else "FtSRgyCEhKTc1PPgEAXvuHN3NyiP6LS9uyB28KCN3CAP"

    df = extract_5m_candles(address)
    if df is None:
        print(f"No 5m data for {address[:8]}. Run fetch_5m.py first.")
        sys.exit(1)

    print(f"Loaded {len(df)} 5-min candles for {address[:8]}")

    df = compute_all_indicators(df)
    df = generate_trade_management_signals(df)

    print(f"\nLast 12 bars (1 hour):")
    cols = ["datetime", "mcap", "roc_30m", "roc_accel_30m", "macd_hist_slope", "rvol", "momentum_quality"]
    print(df[cols].tail(12).to_string(index=False))

    # Summary
    print(f"\nMomentum quality: {df['momentum_quality'].iloc[-1]:+.3f}")
    print(f"Tighten stop: {df['tighten_stop'].iloc[-1]}")
    print(f"Extend hold: {df['extend_hold'].iloc[-1]}")
