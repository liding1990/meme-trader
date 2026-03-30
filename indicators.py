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


def compute_rolling_hurst(series, window=48, step=12):
    """Compute rolling Hurst exponent over a window.

    window=48 = 4 hours of 5-min bars.
    step=12: only compute every 12 bars (1 hour) for speed, forward-fill between.
    """
    result = np.full(len(series), 0.5)
    values = series.values

    last_hurst = 0.5
    for i in range(window, len(values)):
        if i % step == 0:
            chunk = values[i - window:i]
            if np.std(chunk) > 0:
                last_hurst = compute_hurst(chunk)
        result[i] = last_hurst

    return pd.Series(result, index=series.index)


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


def ehlers_roofing_filter(close, hp_period=48, lp_period=10):
    """Ehlers Roofing Filter: high-pass (remove trend) + Super Smoother (remove noise).

    Extracts the "useful" frequency band from price data.
    """
    src = close.values.astype(float)
    n = len(src)

    angle_hp = 0.707 * 2.0 * np.pi / hp_period
    alpha_hp = (np.cos(angle_hp) + np.sin(angle_hp) - 1.0) / np.cos(angle_hp)

    a1 = np.exp(-np.sqrt(2.0) * np.pi / lp_period)
    b1 = 2.0 * a1 * np.cos(np.sqrt(2.0) * np.pi / lp_period)
    c2, c3 = b1, -a1 * a1
    c1 = 1.0 - c2 - c3

    hp = np.zeros(n)
    filt = np.zeros(n)

    for i in range(2, n):
        if np.isnan(src[i]):
            hp[i] = hp[i - 1]
            continue
        hp[i] = ((1.0 - alpha_hp / 2.0) ** 2 * (src[i] - 2.0 * src[i - 1] + src[i - 2])
                 + 2.0 * (1.0 - alpha_hp) * hp[i - 1]
                 - (1.0 - alpha_hp) ** 2 * hp[i - 2])

    for i in range(2, n):
        filt[i] = c1 * (hp[i] + hp[i - 1]) / 2.0 + c2 * filt[i - 1] + c3 * filt[i - 2]

    return pd.Series(filt, index=close.index)


def ehlers_fisher_transform(close, period=10):
    """Ehlers Fisher Transform: converts price to Gaussian, sharp turning points.

    Returns (fisher, signal) Series. Fisher crossing Signal = regime change.
    """
    src = close.values.astype(float)
    n = len(src)

    mid = src  # use close directly (or roofing filter output)
    value = np.zeros(n)
    fisher = np.zeros(n)

    for i in range(period, n):
        window = mid[i - period + 1:i + 1]
        max_h = np.nanmax(window)
        min_l = np.nanmin(window)
        rng = max_h - min_l
        if rng > 0:
            raw = 2.0 * ((mid[i] - min_l) / rng - 0.5)
        else:
            raw = 0.0

        value[i] = 0.33 * raw + 0.67 * value[i - 1]
        value[i] = np.clip(value[i], -0.999, 0.999)

        fisher[i] = 0.5 * np.log((1.0 + value[i]) / (1.0 - value[i]))
        fisher[i] = 0.5 * fisher[i] + 0.5 * fisher[i - 1]

    fisher_s = pd.Series(fisher, index=close.index)
    signal_s = fisher_s.shift(1)
    return fisher_s, signal_s


def ehlers_ebsw(close, hp_period=40, lp_period=10):
    """Ehlers Even Better Sinewave: detects trending vs cycling market.

    EBSW > 0: market is trending (momentum strategy active)
    EBSW < 0: market is cycling/choppy (sit out)
    """
    src = close.values.astype(float)
    n = len(src)

    angle_hp = 0.707 * 2.0 * np.pi / hp_period
    alpha_hp = (np.cos(angle_hp) + np.sin(angle_hp) - 1.0) / np.cos(angle_hp)

    a1 = np.exp(-np.sqrt(2.0) * np.pi / lp_period)
    b1 = 2.0 * a1 * np.cos(np.sqrt(2.0) * np.pi / lp_period)
    c2, c3 = b1, -a1 * a1
    c1 = 1.0 - c2 - c3

    hp = np.zeros(n)
    filt = np.zeros(n)
    ebsw = np.zeros(n)

    for i in range(2, n):
        if np.isnan(src[i]):
            hp[i] = hp[i - 1]
            continue
        hp[i] = ((1.0 - alpha_hp / 2.0) ** 2 * (src[i] - 2.0 * src[i - 1] + src[i - 2])
                 + 2.0 * (1.0 - alpha_hp) * hp[i - 1]
                 - (1.0 - alpha_hp) ** 2 * hp[i - 2])

    for i in range(2, n):
        filt[i] = c1 * (hp[i] + hp[i - 1]) / 2.0 + c2 * filt[i - 1] + c3 * filt[i - 2]

    for i in range(1, n):
        pwr = (filt[i] ** 2 + filt[i - 1] ** 2) / 2.0
        wave = filt[i] / np.sqrt(pwr) if pwr > 0 else 0.0
        wave = np.clip(wave, -1.0, 1.0)
        ebsw[i] = 0.67 * wave + 0.33 * ebsw[i - 1]

    return pd.Series(ebsw, index=close.index)


def ehlers_instantaneous_trendline(close, period=10):
    """Ehlers Instantaneous Trendline (Super Smoother): minimal-lag trend.

    Returns (itrend, trigger) Series. Price above itrend = uptrend.
    Trigger leads itrend for early warning.
    """
    src = close.values.astype(float)
    n = len(src)

    a1 = np.exp(-np.sqrt(2.0) * np.pi / period)
    b1 = 2.0 * a1 * np.cos(np.sqrt(2.0) * np.pi / period)
    c2, c3 = b1, -a1 * a1
    c1 = 1.0 - c2 - c3

    itrend = np.zeros(n)
    itrend[0] = src[0] if not np.isnan(src[0]) else 0.0
    if n > 1:
        itrend[1] = src[1] if not np.isnan(src[1]) else itrend[0]

    for i in range(2, n):
        if np.isnan(src[i]):
            itrend[i] = itrend[i - 1]
        else:
            itrend[i] = c1 * (src[i] + src[i - 1]) / 2.0 + c2 * itrend[i - 1] + c3 * itrend[i - 2]

    trigger = 2.0 * itrend - np.roll(itrend, 2)
    trigger[:2] = itrend[:2]

    return pd.Series(itrend, index=close.index), pd.Series(trigger, index=close.index)


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

    # MACD (kept for backward compatibility / meta-model features)
    result["macd"], result["macd_signal"], result["macd_hist"], result["macd_hist_slope"] = compute_macd(mcap)

    # Ehlers DSP indicators (replace MACD as primary signals)
    # 1. Roofing filter → Fisher Transform (sharp turning points, replaces MACD crossover)
    roofed = ehlers_roofing_filter(mcap, hp_period=48, lp_period=10)
    result["fisher"], result["fisher_signal"] = ehlers_fisher_transform(roofed, period=8)
    result["fisher_cross"] = (result["fisher"] - result["fisher_signal"]).apply(np.sign)

    # 2. Even Better Sinewave (trending vs cycling regime filter)
    result["ebsw"] = ehlers_ebsw(mcap, hp_period=40, lp_period=10)

    # 3. Instantaneous Trendline (minimal-lag trend confirmation)
    result["itrend"], result["itrend_trigger"] = ehlers_instantaneous_trendline(mcap, period=10)
    result["above_itrend"] = (mcap > result["itrend"]).astype(int)

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

    # Ehlers Fisher Transform: positive = bullish momentum
    if "fisher" in df.columns:
        fisher = df["fisher"]
        std = fisher.std()
        if std > 0:
            signals.append(np.clip(fisher / std, -2, 2) / 2)

    # Ehlers EBSW: > 0 = trending (good for momentum), < 0 = cycling (bad)
    if "ebsw" in df.columns:
        signals.append(df["ebsw"].fillna(0))  # already in [-1, 1]

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
    # Tighten stop when: ROC decelerating + Fisher turning down OR EBSW going negative
    df["tighten_stop"] = (df.get("roc_accel_30m", 0) < 0)

    # Fisher crossing signal downward = momentum fading
    if "fisher_cross" in df.columns:
        fisher_turning = (df["fisher_cross"] < 0) & (df["fisher_cross"].shift(1) >= 0)
        df["tighten_stop"] = df["tighten_stop"] | fisher_turning

    # EBSW going negative = cycling market = tighten
    if "ebsw" in df.columns:
        df["tighten_stop"] = df["tighten_stop"] | (df["ebsw"] < -0.3)

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
