"""Momentum Signal Model — CatBoost regression predicting 6h forward return.

Phase 1: Build training data from organic tokens
Phase 2: Train CatBoost regressor with GroupKFold
Phase 3: Backtest momentum-driven position management
Phase 4: Save model for live use

Usage:
    PYTHONPATH=. python token_discovery/momentum_model.py
"""

import csv
import glob
import json
import os
import sys
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = "data"
CLUSTER_DIR = "baseline_cluster_v2_data"
OUTPUT_DIR = "token_discovery"
FORWARD_HOURS = 6
POSITION_SIZE = 5000


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_organic_tokens():
    df = pd.read_csv(os.path.join(CLUSTER_DIR, "clusters.csv"))
    return df[df["rank"].isin([5, 6])]


def load_token_history(address):
    h_files = sorted([f for f in glob.glob(os.path.join(DATA_DIR, address, "token_mcap_candles_[0-9]*.json"))
                       if "5m" not in os.path.basename(f)])
    if not h_files:
        return None, None, None
    try:
        with open(h_files[-1]) as f:
            data = json.load(f)
        candles = (data or {}).get("data", {}).get("list", [])
        if not candles or len(candles) < 20:
            return None, None, None
        cdf = pd.DataFrame(sorted(candles, key=lambda x: int(x["time"])))
        mcap = cdf["close"].astype(float).values
        volume = cdf["volume"].astype(float).values

        # Find $100K start
        start = next((i for i in range(len(mcap)) if mcap[i] >= 100000), None)
        if start is None:
            return None, None, None
        return mcap[start:], volume[start:], address
    except Exception:
        return None, None, None


def load_holders(address, n_candles):
    """Load holder array aligned to candle length."""
    holders = np.zeros(n_candles)
    moralis = os.path.join(DATA_DIR, address, "moralis_holders_1h.json")
    if os.path.isfile(moralis):
        try:
            with open(moralis) as f:
                hdata = json.load(f)
            if hdata:
                hdf = pd.DataFrame(hdata)
                hdf["holders"] = hdf["totalHolders"].astype(float)
                h_vals = hdf["holders"].values
                # Simple alignment: take last N values
                if len(h_vals) >= n_candles:
                    holders = h_vals[-n_candles:]
                else:
                    holders[-len(h_vals):] = h_vals
        except Exception:
            pass
    return holders


# ── Feature Engineering ──────────────────────────────────────────────────────


def compute_features_at_t(mcap, volume, holders, t):
    """Compute ~25 features at time t."""
    if t < 12:
        return None

    feat = {}
    n = len(mcap)

    # Price momentum
    feat["roc_1h"] = (mcap[t] - mcap[t-1]) / max(mcap[t-1], 1) if t >= 1 else 0
    feat["roc_4h"] = (mcap[t] - mcap[t-4]) / max(mcap[t-4], 1) if t >= 4 else 0
    feat["roc_12h"] = (mcap[t] - mcap[t-12]) / max(mcap[t-12], 1) if t >= 12 else 0

    # Volatility
    returns = np.diff(mcap[max(0,t-12):t+1]) / np.maximum(mcap[max(0,t-12):t], 1)
    feat["volatility_4h"] = np.std(returns[-4:]) if len(returns) >= 4 else 0
    feat["volatility_12h"] = np.std(returns) if len(returns) > 1 else 0

    # Volume momentum
    vol_window = volume[max(0,t-24):t+1]
    if len(vol_window) >= 8:
        mid = len(vol_window) // 2
        v1 = vol_window[:mid].mean()
        v2 = vol_window[mid:].mean()
        feat["volume_trend"] = (v2 - v1) / max(v1, 1)
    else:
        feat["volume_trend"] = 0

    vol_recent = volume[max(0,t-4):t+1].sum()
    vol_avg = volume[max(0,t-24):t+1].mean() * 4 if t >= 24 else vol_recent
    feat["volume_surge"] = vol_recent / max(vol_avg, 1)

    if volume[t-4:t+1].sum() > 0:
        feat["volume_concentration"] = volume[t-4:t+1].max() / volume[t-4:t+1].sum()
    else:
        feat["volume_concentration"] = 0

    # Holder momentum
    if holders[t] > 0 and t >= 1:
        feat["holder_growth_1h"] = (holders[t] - holders[t-1]) / max(holders[t-1], 1)
    else:
        feat["holder_growth_1h"] = 0

    if holders[t] > 0 and t >= 4:
        feat["holder_growth_4h"] = (holders[t] - holders[t-4]) / max(holders[t-4], 1)
    else:
        feat["holder_growth_4h"] = 0

    # Holder acceleration (change in growth rate)
    if t >= 8 and holders[t-4] > 0 and holders[t-8] > 0:
        g1 = (holders[t-4] - holders[t-8]) / max(holders[t-8], 1)
        g2 = (holders[t] - holders[t-4]) / max(holders[t-4], 1)
        feat["holder_acceleration"] = g2 - g1
    else:
        feat["holder_acceleration"] = 0

    # VWAP
    vwap_window = min(24, t+1)
    cum_vol = volume[t-vwap_window+1:t+1].sum()
    cum_vp = (mcap[t-vwap_window+1:t+1] * volume[t-vwap_window+1:t+1]).sum()
    vwap = cum_vp / max(cum_vol, 1)
    feat["price_vs_vwap"] = mcap[t] / max(vwap, 1)

    # Drawdown from running max
    running_max = np.maximum.accumulate(mcap[:t+1])
    feat["drawdown_from_peak"] = (mcap[t] - running_max[t]) / max(running_max[t], 1)

    # Distance from ATH (how far from all-time high)
    feat["pct_of_ath"] = mcap[t] / max(mcap[:t+1].max(), 1)

    # Market structure
    feat["mcap_per_holder"] = mcap[t] / max(holders[t], 1)
    feat["mcap_level"] = np.log1p(mcap[t])
    feat["holders_level"] = np.log1p(holders[t])

    # Price position in recent range
    recent = mcap[max(0,t-24):t+1]
    feat["price_position"] = (mcap[t] - recent.min()) / max(recent.max() - recent.min(), 1)

    # Trend strength (linear regression slope of last 12h)
    if t >= 12:
        x = np.arange(12)
        y = mcap[t-11:t+1]
        slope = np.polyfit(x, y, 1)[0]
        feat["trend_slope"] = slope / max(mcap[t], 1)
    else:
        feat["trend_slope"] = 0

    # Mean reversion signal
    ma_12 = mcap[max(0,t-11):t+1].mean()
    feat["ma_deviation"] = (mcap[t] - ma_12) / max(ma_12, 1)

    return feat


FEATURE_NAMES = [
    "roc_1h", "roc_4h", "roc_12h", "volatility_4h", "volatility_12h",
    "volume_trend", "volume_surge", "volume_concentration",
    "holder_growth_1h", "holder_growth_4h", "holder_acceleration",
    "price_vs_vwap", "drawdown_from_peak", "pct_of_ath",
    "mcap_per_holder", "mcap_level", "holders_level",
    "price_position", "trend_slope", "ma_deviation",
]


# ── Phase 1: Build Training Data ─────────────────────────────────────────────


def build_dataset():
    print("Phase 1: Building training data...")
    organic = load_organic_tokens()
    print(f"  Organic tokens: {len(organic)}")

    samples = []
    for i, (_, row) in enumerate(organic.iterrows()):
        mcap, volume, addr = load_token_history(row["address"])
        if mcap is None or len(mcap) < FORWARD_HOURS + 20:
            continue

        holders = load_holders(row["address"], len(mcap))

        for t in range(12, len(mcap) - FORWARD_HOURS):
            feat = compute_features_at_t(mcap, volume, holders, t)
            if feat is None:
                continue

            # Label: 6h forward return
            future_return = (mcap[t + FORWARD_HOURS] - mcap[t]) / max(mcap[t], 1)
            # Clip extreme values
            future_return = np.clip(future_return, -0.95, 10.0)

            samples.append({
                "address": row["address"],
                "symbol": row["symbol"],
                "t": t,
                "label": future_return,
                **feat,
            })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(organic)}, {len(samples):,} samples")

    df = pd.DataFrame(samples)
    print(f"  Total: {len(df):,} samples from {df['address'].nunique()} tokens")
    return df


# ── Phase 2: Train Model ─────────────────────────────────────────────────────


def train_model(df):
    print("\nPhase 2: Training CatBoost regressor...")

    X = df[FEATURE_NAMES].values.astype(float)
    X = np.nan_to_num(X, nan=0)
    y = df["label"].values
    groups = df["address"].values

    gkf = GroupKFold(n_splits=5)
    y_pred_oos = np.full(len(y), np.nan)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        model = CatBoostRegressor(
            iterations=500, depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bylevel=0.8,
            reg_lambda=1.0, random_seed=42, verbose=0,
        )
        model.fit(X[train_idx], y[train_idx])
        y_pred_oos[val_idx] = model.predict(X[val_idx])
        print(f"  Fold {fold}: train={len(train_idx):,}, val={len(val_idx):,}")

    # Evaluation
    from scipy import stats
    corr = stats.spearmanr(y, y_pred_oos)[0]
    mae = np.mean(np.abs(y - y_pred_oos))

    # Directional accuracy: does the model get the sign right?
    dir_acc = ((y > 0) == (y_pred_oos > 0)).mean()

    # Signal quality: when model predicts >5%, what actually happens?
    strong_bull = y_pred_oos > 0.05
    strong_bear = y_pred_oos < -0.05

    print(f"\n  OOS Results:")
    print(f"    Spearman corr:     {corr:.3f}")
    print(f"    MAE:               {mae:.3f}")
    print(f"    Direction accuracy: {dir_acc:.1%}")
    if strong_bull.sum() > 0:
        print(f"    Pred >+5%: actual avg = {y[strong_bull].mean()*100:+.2f}% ({strong_bull.sum():,} samples)")
    if strong_bear.sum() > 0:
        print(f"    Pred <-5%: actual avg = {y[strong_bear].mean()*100:+.2f}% ({strong_bear.sum():,} samples)")

    # Train final model on all data
    model_final = CatBoostRegressor(
        iterations=500, depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bylevel=0.8,
        reg_lambda=1.0, random_seed=42, verbose=0,
    )
    model_final.fit(X, y)

    # Feature importance
    importances = sorted(zip(FEATURE_NAMES, model_final.feature_importances_),
                          key=lambda x: x[1], reverse=True)
    print(f"\n  Top features:")
    for name, imp in importances[:10]:
        print(f"    {name:>25s}: {imp:.1f}")

    return model_final, y_pred_oos, y


# ── Phase 3: Backtest with Momentum Signal ────────────────────────────────────


def backtest_momentum(df, model):
    print("\nPhase 3: Backtesting momentum-driven position management...")

    organic = load_organic_tokens()
    all_trades_momentum = []
    all_trades_fixed = []
    all_trades_hold = []

    # Fixed TP/SL for comparison
    FIXED_TP = [(0.30, 0.15), (0.80, 0.20), (2.00, 0.20), (5.00, 0.20), (10.00, 0.15)]
    FIXED_SL = [(-0.15, 0.30), (-0.30, 0.30), (-0.50, 1.00)]

    for _, row in organic.iterrows():
        mcap, volume, addr = load_token_history(row["address"])
        if mcap is None or len(mcap) < 60:
            continue
        holders = load_holders(row["address"], len(mcap))

        entry_price = mcap[0]
        n = len(mcap)

        # ── Momentum strategy ──
        remaining = 1.0
        realized = 0.0
        prev_signal = 0.0
        peak_signal = 0.0

        for t in range(12, n):
            if remaining <= 0.01:
                break

            pnl = (mcap[t] - entry_price) / max(entry_price, 1)
            feat = compute_features_at_t(mcap, volume, holders, t)
            if feat is None:
                continue

            vec = np.array([[feat.get(f, 0) for f in FEATURE_NAMES]])
            vec = np.nan_to_num(vec, nan=0)
            signal = float(model.predict(vec)[0])  # predicted 6h return

            peak_signal = max(peak_signal, signal)

            # Momentum-driven decisions
            if signal < -0.10 and remaining > 0:
                # Strong bearish: exit everything
                sell = remaining
                realized += sell * pnl * POSITION_SIZE * (1 - 0.03)
                remaining = 0

            elif signal < -0.03 and pnl < -0.10:
                # Moderate bearish + losing position: cut half
                sell = min(0.50, remaining)
                realized += sell * pnl * POSITION_SIZE * (1 - 0.03)
                remaining -= sell

            elif pnl > 0.30 and signal < 0.02 and prev_signal > 0.05:
                # Profit + momentum fading: take partial profit
                sell = min(0.25, remaining)
                realized += sell * pnl * POSITION_SIZE * (1 - 0.03)
                remaining -= sell

            elif pnl > 1.0 and signal < prev_signal * 0.5 and signal < 0.05:
                # Big profit + momentum collapsing: take more
                sell = min(0.30, remaining)
                realized += sell * pnl * POSITION_SIZE * (1 - 0.03)
                remaining -= sell

            prev_signal = signal

        # Force close remaining
        final_pnl = (mcap[-1] - entry_price) / max(entry_price, 1)
        realized += remaining * final_pnl * POSITION_SIZE * (1 - 0.03)
        momentum_total = realized

        # ── Fixed TP/SL strategy ──
        remaining_f = 1.0
        realized_f = 0.0
        tp_triggered_f = [False] * len(FIXED_TP)
        sl_triggered_f = [False] * len(FIXED_SL)

        for t in range(n):
            if remaining_f <= 0.01:
                break
            pnl = (mcap[t] - entry_price) / max(entry_price, 1)

            for j, (trigger, sell_pct) in enumerate(FIXED_TP):
                if not tp_triggered_f[j] and pnl >= trigger:
                    s = min(sell_pct, remaining_f)
                    realized_f += s * pnl * POSITION_SIZE * (1 - 0.03)
                    remaining_f -= s
                    tp_triggered_f[j] = True

            if pnl < 0:
                for j, (trigger, sell_pct) in enumerate(FIXED_SL):
                    if not sl_triggered_f[j] and pnl <= trigger:
                        s = min(sell_pct, remaining_f)
                        realized_f += s * pnl * POSITION_SIZE * (1 - 0.03)
                        remaining_f -= s
                        sl_triggered_f[j] = True

        realized_f += remaining_f * final_pnl * POSITION_SIZE * (1 - 0.03)
        fixed_total = realized_f

        # ── Buy & Hold ──
        hold_total = POSITION_SIZE * (1 + final_pnl) * (1 - 0.03)

        all_trades_momentum.append(momentum_total / POSITION_SIZE - 1)
        all_trades_fixed.append(fixed_total / POSITION_SIZE - 1)
        all_trades_hold.append(hold_total / POSITION_SIZE - 1)

    # Results
    def print_stats(name, returns):
        r = np.array(returns)
        wins = r > 0
        print(f"\n  {name} ({len(r)} trades):")
        print(f"    Mean return:    {r.mean()*100:+.1f}%")
        print(f"    Median return:  {np.median(r)*100:+.1f}%")
        print(f"    Win rate:       {wins.mean()*100:.0f}%")
        print(f"    Avg win:        {r[wins].mean()*100:+.1f}%" if wins.sum() > 0 else "")
        print(f"    Avg loss:       {r[~wins].mean()*100:+.1f}%" if (~wins).sum() > 0 else "")
        gp = r[wins].sum() if wins.sum() > 0 else 0
        gl = abs(r[~wins].sum()) if (~wins).sum() > 0 else 1e-9
        print(f"    Profit Factor:  {gp/gl:.2f}")
        sharpe = r.mean() / max(r.std(), 1e-9) * np.sqrt(252)
        print(f"    Sharpe:         {sharpe:.2f}")

    print_stats("Momentum Model", all_trades_momentum)
    print_stats("Fixed TP/SL", all_trades_fixed)
    print_stats("Buy & Hold", all_trades_hold)

    return all_trades_momentum, all_trades_fixed, all_trades_hold


# ── Phase 4: Save ─────────────────────────────────────────────────────────────


def save_model(model):
    path = os.path.join(OUTPUT_DIR, "momentum_model.cbm")
    model.save_model(path)

    with open(os.path.join(OUTPUT_DIR, "momentum_features.json"), "w") as f:
        json.dump(FEATURE_NAMES, f)

    print(f"\nPhase 4: Model saved to {path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("Momentum Signal Model — Train + Backtest")
    print("=" * 60)

    df = build_dataset()
    model, y_pred, y_true = train_model(df)
    momentum_rets, fixed_rets, hold_rets = backtest_momentum(df, model)
    save_model(model)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
