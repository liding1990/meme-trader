"""Build training dataset for CatBoost position management model.

For each token, simulates multiple entry points and generates a sample
at each subsequent hourly tick with features + optimal action label.

Usage:
    python posbot_train/build_dataset.py
    python posbot_train/build_dataset.py --output posbot_train/dataset.parquet
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from hmmlearn import hmm

DATA_DIR = "data"
RADAR_CSV = os.path.join(DATA_DIR, "radar_tokens.csv")
MCAP_THRESHOLD = 100_000
ENTRY_STEP_HOURS = 12      # try an entry every 12 hours
TICK_STEP_HOURS = 1         # 1 sample per hour after entry
MAX_HOLD_HOURS = 48         # max holding period
FUTURE_LOOK_HOURS = 6       # look 6h ahead for label


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_radar_tokens():
    tokens = []
    with open(RADAR_CSV) as f:
        for row in csv.reader(f):
            if len(row) >= 4:
                tokens.append({"address": row[0], "chain": row[1], "symbol": row[3]})
    return tokens


def load_1h_candles(address):
    data_dir = os.path.join(DATA_DIR, address)
    h_files = sorted([
        f for f in glob.glob(os.path.join(data_dir, "token_mcap_candles_[0-9]*.json"))
        if "5m" not in os.path.basename(f)
    ])
    if not h_files:
        return None
    try:
        with open(h_files[-1]) as f:
            data = json.load(f)
        candles = (data or {}).get("data", {}).get("list", [])
        if not candles:
            return None
        df = pd.DataFrame(candles)
        df["time_ms"] = df["time"].astype(int)
        df["datetime"] = pd.to_datetime(df["time_ms"], unit="ms")
        df["mcap"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["open"] = df["open"].astype(float)
        return df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    except Exception:
        return None


def load_holders(address):
    # Moralis first
    path = os.path.join(DATA_DIR, address, "moralis_holders_1h.json")
    if os.path.isfile(path):
        try:
            with open(path) as f:
                data = json.load(f)
            if data:
                df = pd.DataFrame(data)
                df["datetime"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
                df["holders"] = df["totalHolders"].astype(float)
                return df[["datetime", "holders"]].sort_values("datetime").reset_index(drop=True)
        except Exception:
            pass
    # GMGN fallback
    data_dir = os.path.join(DATA_DIR, address)
    t_files = sorted(glob.glob(os.path.join(data_dir, "token_trends_*.json")))
    if not t_files:
        return None
    try:
        with open(t_files[-1]) as f:
            data = json.load(f)
        series = (data or {}).get("data", {}).get("trends", {}).get("holder_count", [])
        if not series:
            return None
        df = pd.DataFrame(series)
        df["datetime"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        df["holders"] = df["value"].astype(float)
        return df[["datetime", "holders"]].sort_values("datetime").reset_index(drop=True)
    except Exception:
        return None


def load_top10(address):
    data_dir = os.path.join(DATA_DIR, address)
    t_files = sorted(glob.glob(os.path.join(data_dir, "token_trends_*.json")))
    if not t_files:
        return None
    try:
        with open(t_files[-1]) as f:
            data = json.load(f)
        t10 = (data or {}).get("data", {}).get("trends", {}).get("top10_holder_percent", [])
        if not t10:
            return None
        df = pd.DataFrame(t10)
        df["datetime"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        df["top10_pct"] = df["value"].astype(float)
        return df[["datetime", "top10_pct"]].sort_values("datetime").reset_index(drop=True)
    except Exception:
        return None


# ── Feature Engineering ──────────────────────────────────────────────────────


def compute_vwap(mcap_arr, volume_arr, window=24):
    """Rolling VWAP over given window."""
    cum_vol = pd.Series(volume_arr).rolling(window, min_periods=1).sum()
    cum_vp = pd.Series(mcap_arr * volume_arr).rolling(window, min_periods=1).sum()
    vwap = cum_vp / cum_vol.replace(0, np.nan)
    return vwap.fillna(pd.Series(mcap_arr)).values


def fit_hmm_states(returns, n_states=3):
    """Fit HMM on returns, return state sequence and transition matrix."""
    returns_clean = returns[~np.isnan(returns)].reshape(-1, 1)
    if len(returns_clean) < 20:
        return np.zeros(len(returns), dtype=int), np.eye(n_states), np.zeros(n_states)

    try:
        model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag",
                                 n_iter=50, random_state=42)
        model.fit(returns_clean)
        states = model.predict(returns_clean)

        # Pad back to original length
        full_states = np.zeros(len(returns), dtype=int)
        valid_idx = np.where(~np.isnan(returns))[0]
        full_states[valid_idx] = states

        # Rank states by mean return: 0=down, 1=neutral, 2=up
        state_means = [returns_clean[states == s].mean() for s in range(n_states)]
        rank = np.argsort(state_means)
        remap = {rank[i]: i for i in range(n_states)}
        full_states = np.array([remap.get(s, s) for s in full_states])

        transmat = model.transmat_
        state_means_sorted = [state_means[rank[i]] for i in range(n_states)]

        return full_states, transmat, np.array(state_means_sorted)
    except Exception:
        return np.zeros(len(returns), dtype=int), np.eye(n_states), np.zeros(n_states)


def precompute_token_features(df_1h, holder_df, top10_df):
    """Precompute all time-series features for a token."""
    n = len(df_1h)
    mcap = df_1h["mcap"].values
    volume = df_1h["volume"].values
    hours = np.arange(n, dtype=float)

    # Returns
    returns = np.diff(mcap) / np.maximum(mcap[:-1], 1)
    returns = np.concatenate([[0], returns])

    # ROC at different windows
    def roc(arr, w):
        r = np.full(len(arr), 0.0)
        for i in range(w, len(arr)):
            if arr[i - w] > 0:
                r[i] = (arr[i] - arr[i - w]) / arr[i - w]
        return r

    roc_1h = roc(mcap, 1)
    roc_4h = roc(mcap, 4)
    roc_12h = roc(mcap, 12)

    # Volatility
    def rolling_std(arr, w):
        s = pd.Series(arr).rolling(w, min_periods=2).std().fillna(0).values
        return s

    vol_1h = rolling_std(returns, 6)   # ~1h of 10min-ish data
    vol_4h = rolling_std(returns, 4)

    # Volume trend
    vol_ma_short = pd.Series(volume).rolling(4, min_periods=1).mean().values
    vol_ma_long = pd.Series(volume).rolling(12, min_periods=1).mean().values
    volume_trend = np.where(vol_ma_long > 0, (vol_ma_short - vol_ma_long) / vol_ma_long, 0)

    # Volume concentration (max hourly / sum in window)
    vol_roll_sum = pd.Series(volume).rolling(12, min_periods=1).sum().values
    vol_roll_max = pd.Series(volume).rolling(12, min_periods=1).max().values
    volume_concentration = np.where(vol_roll_sum > 0, vol_roll_max / vol_roll_sum, 0)

    # VWAP
    vwap_24h = compute_vwap(mcap, volume, window=24)
    price_vs_vwap = mcap / np.maximum(vwap_24h, 1)
    vwap_slope = np.concatenate([[0], np.diff(vwap_24h) / np.maximum(vwap_24h[:-1], 1)])

    # HMM
    hmm_states, transmat, state_means = fit_hmm_states(returns)
    hmm_duration = np.zeros(n)
    for i in range(1, n):
        if hmm_states[i] == hmm_states[i - 1]:
            hmm_duration[i] = hmm_duration[i - 1] + 1
        else:
            hmm_duration[i] = 1
    # Transition prob to state 0 (down state)
    hmm_trans_to_down = np.array([transmat[s][0] if s < len(transmat) else 0.33
                                   for s in hmm_states])

    # Holders (merge onto hourly index)
    holders_arr = np.zeros(n)
    if holder_df is not None and len(holder_df) >= 2:
        h_resampled = holder_df.set_index("datetime").resample("1h").last().ffill().reset_index()
        merged = pd.merge_asof(df_1h[["datetime"]], h_resampled, on="datetime", direction="backward")
        if "holders" in merged.columns:
            holders_arr = merged["holders"].fillna(0).values

    holder_growth_1h = np.concatenate([[0], np.diff(holders_arr) / np.maximum(holders_arr[:-1], 1)])
    holder_growth_4h = roc(holders_arr, 4)
    mcap_per_holder = np.where(holders_arr > 0, mcap / holders_arr, mcap)

    # Top10
    top10_arr = np.full(n, np.nan)
    if top10_df is not None and len(top10_df) >= 1:
        t10_merged = pd.merge_asof(df_1h[["datetime"]], top10_df, on="datetime", direction="backward")
        if "top10_pct" in t10_merged.columns:
            top10_arr = t10_merged["top10_pct"].values
    top10_change = np.concatenate([[0], np.diff(np.nan_to_num(top10_arr))])

    return {
        "mcap": mcap, "volume": volume, "returns": returns,
        "roc_1h": roc_1h, "roc_4h": roc_4h, "roc_12h": roc_12h,
        "volatility_1h": vol_1h, "volatility_4h": vol_4h,
        "volume_trend": volume_trend, "volume_concentration": volume_concentration,
        "vwap_24h": vwap_24h, "price_vs_vwap": price_vs_vwap, "vwap_slope": vwap_slope,
        "hmm_state": hmm_states, "hmm_duration": hmm_duration,
        "hmm_trans_to_down": hmm_trans_to_down,
        "holders": holders_arr, "holder_growth_1h": holder_growth_1h,
        "holder_growth_4h": holder_growth_4h, "mcap_per_holder": mcap_per_holder,
        "top10_pct": top10_arr, "top10_change": top10_change,
    }


# ── Label Generation ─────────────────────────────────────────────────────────


ACTIONS = ["HOLD", "TP_25", "TP_50", "TP_100", "SL_25", "SL_50", "EXIT"]
ACTION_SELL_PCT = {
    "HOLD": 0.0, "TP_25": 0.25, "TP_50": 0.50, "TP_100": 1.0,
    "SL_25": 0.25, "SL_50": 0.50, "EXIT": 1.0,
}


def find_optimal_actions(mcap_series, entry_price):
    """Find optimal action sequence using hindsight-based rules.

    Key insight: we know the full future. Use it to decide:
    - TP at local peaks before drawdowns
    - SL when clearly no recovery
    - Scale out gradually, don't go all-or-nothing

    Returns list of (tick_offset, action, remaining_before, remaining_after).
    """
    n = len(mcap_series)
    if n < 2:
        return [(0, "HOLD", 1.0, 1.0)] * n

    prices = mcap_series
    results = []
    remaining = 1.0

    # Precompute: for each tick, find the next local max and local min
    # Local max: price higher than next 3 bars
    is_local_peak = np.zeros(n, dtype=bool)
    for t in range(n - 3):
        if prices[t] >= prices[t + 1] and prices[t] >= prices[t + 2] and prices[t] >= prices[t + 3]:
            is_local_peak[t] = True

    # Future max from each point
    future_max_from = np.zeros(n)
    future_max_from[-1] = prices[-1]
    for t in range(n - 2, -1, -1):
        future_max_from[t] = max(prices[t], future_max_from[t + 1])

    # Future min in next 6h
    future_min_6h = np.zeros(n)
    for t in range(n):
        end = min(t + 6, n)
        future_min_6h[t] = prices[t:end].min()

    for t in range(n):
        if remaining <= 0.01:
            results.append((t, "HOLD", 0.0, 0.0))
            continue

        current = prices[t]
        pnl = (current - entry_price) / max(entry_price, 1)
        future_upside = (future_max_from[t] - current) / max(current, 1)
        near_drawdown = (future_min_6h[t] - current) / max(current, 1)

        action = "HOLD"

        # === TAKE PROFIT at local peaks ===
        if is_local_peak[t] and pnl > 0:
            if pnl > 1.5 and future_upside < 0.20:
                # Huge gain, little upside left → sell big
                action = "TP_100" if remaining <= 0.5 else "TP_50"
            elif pnl > 0.5 and near_drawdown < -0.20:
                # Good gain, big drawdown coming → partial TP
                action = "TP_50" if remaining > 0.5 else "TP_25"
            elif pnl > 0.2 and near_drawdown < -0.15:
                # Moderate gain, moderate drawdown → small TP
                action = "TP_25"

        # === STOP LOSS when clearly dying ===
        if pnl < -0.30 and future_upside < 0.10:
            action = "EXIT"
        elif pnl < -0.20 and future_upside < 0.05:
            action = "SL_50" if remaining > 0.5 else "EXIT"
        elif pnl < -0.15 and future_upside < 0.03 and near_drawdown < -0.05:
            action = "SL_25"

        # === Force exit near end of window ===
        if t >= n - 2 and remaining > 0 and pnl > 0.1:
            action = "TP_100"
        elif t >= n - 2 and remaining > 0:
            action = "EXIT"

        # Apply
        sell_frac = min(ACTION_SELL_PCT[action], remaining)
        new_remaining = remaining - sell_frac
        results.append((t, action, remaining, new_remaining))
        remaining = new_remaining

    return results


# ── Sample Generation ────────────────────────────────────────────────────────


def generate_token_samples(token, early_warning_scores=None):
    """Generate all training samples for one token."""
    addr = token["address"]
    symbol = token["symbol"]

    df_1h = load_1h_candles(addr)
    if df_1h is None or len(df_1h) < MAX_HOLD_HOURS + FUTURE_LOOK_HOURS:
        return []

    holder_df = load_holders(addr)
    top10_df = load_top10(addr)

    # Precompute all features
    feat = precompute_token_features(df_1h, holder_df, top10_df)
    mcap = feat["mcap"]
    n = len(mcap)

    # Find valid entry points (mcap >= threshold)
    above = np.where(mcap >= MCAP_THRESHOLD)[0]
    if len(above) == 0:
        return []

    # Early warning scores (if available)
    ew_scores = early_warning_scores.get(addr, {}) if early_warning_scores else {}

    samples = []
    entry_indices = above[::ENTRY_STEP_HOURS]

    for entry_idx in entry_indices:
        entry_price = mcap[entry_idx]
        max_tick = min(entry_idx + MAX_HOLD_HOURS, n)

        # Find optimal action sequence for this entry using DP
        trade_prices = mcap[entry_idx:max_tick]
        if len(trade_prices) < 3:
            continue
        optimal_actions = find_optimal_actions(trade_prices, entry_price)

        # Generate samples with true position state from optimal sequence
        peak_price = entry_price
        for offset, (t_off, label, remaining_before, remaining_after) in enumerate(optimal_actions):
            tick_idx = entry_idx + t_off
            if tick_idx >= n:
                break

            current_price = mcap[tick_idx]
            holding_hours = t_off
            peak_price = max(peak_price, current_price)
            unrealized_pnl = (current_price - entry_price) / max(entry_price, 1)
            sold_pct = 1.0 - remaining_before

            i = tick_idx
            sample = {
                "address": addr,
                "symbol": symbol,
                "entry_idx": entry_idx,
                "tick_idx": tick_idx,
                "label": label,
                # A. Position state (reflects actual position changes)
                "unrealized_pnl": unrealized_pnl,
                "holding_hours": holding_hours,
                "sold_pct": sold_pct,
                "remaining_pct": remaining_before,
                "distance_from_peak": (peak_price - current_price) / max(peak_price, 1),
                # B. Price momentum
                "roc_1h": feat["roc_1h"][i],
                "roc_4h": feat["roc_4h"][i],
                "roc_12h": feat["roc_12h"][i],
                "volatility_1h": feat["volatility_1h"][i],
                "volatility_4h": feat["volatility_4h"][i],
                "volume_trend": feat["volume_trend"][i],
                "buy_sell_ratio": 0,
                "volume_concentration": feat["volume_concentration"][i],
                # C. VWAP
                "vwap_24h": feat["vwap_24h"][i],
                "price_vs_vwap": feat["price_vs_vwap"][i],
                "vwap_slope": feat["vwap_slope"][i],
                # D. HMM
                "hmm_state": int(feat["hmm_state"][i]),
                "hmm_state_duration": feat["hmm_duration"][i],
                "hmm_trans_to_down": feat["hmm_trans_to_down"][i],
                # E. On-chain
                "holder_growth_1h": feat["holder_growth_1h"][i],
                "holder_growth_4h": feat["holder_growth_4h"][i],
                "mcap_per_holder": feat["mcap_per_holder"][i],
                "top10_pct": feat["top10_pct"][i],
                "top10_change": feat["top10_change"][i],
                # F. Early warning confidence
                "ew_predict_mult": ew_scores.get(tick_idx, 1.0),
                "ew_grade": 5,
            }
            samples.append(sample)

    return samples


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="构建 CatBoost 训练数据集")
    parser.add_argument("--output", default="posbot_train/dataset.parquet")
    args = parser.parse_args()

    tokens = load_radar_tokens()
    print(f"加载了 {len(tokens)} 个雷达 token")

    all_samples = []
    n_tokens_used = 0

    for i, token in enumerate(tokens):
        samples = generate_token_samples(token)
        if samples:
            all_samples.extend(samples)
            n_tokens_used += 1

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(tokens)}, {len(all_samples):,} 个样本, {n_tokens_used} 个 token")

    df = pd.DataFrame(all_samples)
    print(f"\n数据集: {len(df):,} 个样本, {n_tokens_used} 个 token")
    print(f"标签分布:")
    print(df["label"].value_counts().to_string())

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"\n已保存到 {args.output}")


if __name__ == "__main__":
    main()
