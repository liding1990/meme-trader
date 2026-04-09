"""Lifecycle Strategy — Holder-based position management.

Core principle: don't use price for stop loss. Use "life signals":
  - Holder still growing → HOLD (no matter how much price drops)
  - Holder stagnating → start taking profit
  - Holder declining → accelerate selling
  - Holder collapsing + volume dying → EXIT

Backtest covers FULL lifecycle (entry → end of data), not just to ATH.

Usage:
    PYTHONPATH=. python token_discovery/lifecycle_strategy.py
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = "data"
CLUSTER_DIR = "baseline_cluster_v2_data"
POSITION_SIZE = 5000
SLIPPAGE = 0.03


def load_organic_tokens():
    df = pd.read_csv(os.path.join(CLUSTER_DIR, "clusters.csv"))
    return df[df["rank"].isin([5, 6])]


def load_token_data(address):
    """Load mcap + volume + holders aligned arrays."""
    h_files = sorted([f for f in glob.glob(os.path.join(DATA_DIR, address, "token_mcap_candles_[0-9]*.json"))
                       if "5m" not in os.path.basename(f)])
    if not h_files:
        return None

    try:
        with open(h_files[-1]) as f:
            data = json.load(f)
        candles = (data or {}).get("data", {}).get("list", [])
        if not candles or len(candles) < 20:
            return None
        cdf = pd.DataFrame(sorted(candles, key=lambda x: int(x["time"])))
        mcap = cdf["close"].astype(float).values
        volume = cdf["volume"].astype(float).values

        start = next((i for i in range(len(mcap)) if mcap[i] >= 100000), None)
        if start is None:
            return None
        mcap = mcap[start:]
        volume = volume[start:]
    except Exception:
        return None

    n = len(mcap)
    holders = np.zeros(n)

    moralis = os.path.join(DATA_DIR, address, "moralis_holders_1h.json")
    if os.path.isfile(moralis):
        try:
            with open(moralis) as f:
                hdata = json.load(f)
            if hdata:
                hdf = pd.DataFrame(hdata)
                hdf["datetime"] = pd.to_datetime(hdf["timestamp"], utc=True).dt.tz_localize(None)
                hdf["holders"] = hdf["totalHolders"].astype(float)
                hdf = hdf.sort_values("datetime")
                h_vals = hdf["holders"].values
                if len(h_vals) >= n:
                    holders = h_vals[-n:]
                elif len(h_vals) > 0:
                    holders[-len(h_vals):] = h_vals
        except Exception:
            pass

    if holders.max() == 0:
        # Try GMGN trends
        t_files = sorted(glob.glob(os.path.join(DATA_DIR, address, "token_trends_*.json")))
        if t_files:
            try:
                with open(t_files[-1]) as f:
                    tdata = json.load(f)
                series = (tdata or {}).get("data", {}).get("trends", {}).get("holder_count", [])
                if series:
                    h_vals = np.array([float(s["value"]) for s in sorted(series, key=lambda x: int(x["timestamp"]))])
                    if len(h_vals) >= n:
                        holders = h_vals[-n:]
                    elif len(h_vals) > 0:
                        holders[-len(h_vals):] = h_vals
            except Exception:
                pass

    return {"mcap": mcap, "volume": volume, "holders": holders, "n": n, "address": address}


# ── Life Signal Computation ──────────────────────────────────────────────────


def compute_life_signals(mcap, volume, holders, t, window=12):
    """Compute life signals at time t.

    Returns dict with:
      holder_trend: avg hourly holder change over window (positive = growing)
      holder_accel: change in holder_trend (accelerating/decelerating)
      volume_trend: recent vs historical volume ratio
      holder_growing: bool, is holder count increasing?
      holder_declining: number of consecutive hours of holder decline
    """
    if t < window:
        return None

    # Holder trend: average hourly change over window
    h_window = holders[t-window:t+1]
    if h_window[0] > 0:
        holder_changes = np.diff(h_window) / np.maximum(h_window[:-1], 1)
        holder_trend = np.mean(holder_changes)
    else:
        holder_trend = 0

    # Holder acceleration
    if t >= window * 2:
        h_prev = holders[t-window*2:t-window+1]
        if h_prev[0] > 0:
            prev_changes = np.diff(h_prev) / np.maximum(h_prev[:-1], 1)
            prev_trend = np.mean(prev_changes)
        else:
            prev_trend = 0
        holder_accel = holder_trend - prev_trend
    else:
        holder_accel = 0

    # Volume trend
    vol_recent = volume[max(0,t-6):t+1].mean()
    vol_hist = volume[max(0,t-24):t+1].mean()
    volume_trend = vol_recent / max(vol_hist, 1)

    # Consecutive declining hours
    declining_hours = 0
    for i in range(t, max(t-24, 0), -1):
        if i > 0 and holders[i] < holders[i-1]:
            declining_hours += 1
        else:
            break

    return {
        "holder_trend": holder_trend,
        "holder_accel": holder_accel,
        "volume_trend": volume_trend,
        "holder_growing": holder_trend > 0.001,
        "declining_hours": declining_hours,
    }


# ── Strategy ─────────────────────────────────────────────────────────────────


def run_lifecycle_strategy(mcap, volume, holders):
    """Run lifecycle strategy on a single token.

    Rules:
    1. Holder growing → HOLD (even if price drops 50%)
    2. Holder stagnating (trend ≈ 0) + profit > 50% → TP 20%
    3. Holder declining for 6+ consecutive hours → TP 25%
    4. Holder declining for 12+ hours + volume shrinking → TP 40%
    5. Holder declining for 24+ hours → EXIT remaining
    6. Hard stop: -70% from peak AND holder declining → EXIT (rug protection)
    """
    n = len(mcap)
    entry_price = mcap[0]
    remaining = 1.0
    realized = 0.0
    peak_price = entry_price
    actions = []

    for t in range(n):
        if remaining <= 0.001:
            break

        current = mcap[t]
        pnl = (current - entry_price) / max(entry_price, 1)
        peak_price = max(peak_price, current)
        drawdown_from_peak = (current - peak_price) / max(peak_price, 1)

        signals = compute_life_signals(mcap, volume, holders, t)
        if signals is None:
            continue

        action = None

        # Adaptive thresholds: during first 36h, require stronger signals
        # (holder data is naturally noisy during launch)
        early_phase = t < 24
        decline_6h = 8 if early_phase else 6     # require 8h decline in early, 6h later
        decline_12h = 15 if early_phase else 12  # require 15h in early, 12h later
        decline_exit = 28 if early_phase else 24 # require 28h in early, 24h later
        stagnant_profit = 0.70 if early_phase else 0.50

        # Rule 6: Rug protection — deep drawdown + holder declining
        if drawdown_from_peak < -0.70 and not signals["holder_growing"]:
            sell = remaining
            realized += sell * current * (1 - SLIPPAGE) / entry_price * POSITION_SIZE - sell * POSITION_SIZE
            action = "EXIT(rug)"
            remaining = 0

        # Rule 5: Holder declining → EXIT
        elif signals["declining_hours"] >= decline_exit and remaining > 0:
            sell = remaining
            realized += sell * pnl * POSITION_SIZE * (1 - SLIPPAGE)
            action = "EXIT(decline)"
            remaining = 0

        # Rule 4: Holder declining + volume shrinking → TP 40%
        elif signals["declining_hours"] >= decline_12h and signals["volume_trend"] < 0.5:
            sell = min(0.40, remaining)
            realized += sell * pnl * POSITION_SIZE * (1 - SLIPPAGE)
            remaining -= sell
            action = "TP40%(decline+vol)"

        # Rule 3: Holder declining → TP 25%
        elif signals["declining_hours"] >= decline_6h:
            sell = min(0.25, remaining)
            realized += sell * pnl * POSITION_SIZE * (1 - SLIPPAGE)
            remaining -= sell
            action = "TP25%(decline)"

        # Rule 2: Holder stagnating + profit → TP 20%
        elif abs(signals["holder_trend"]) < 0.001 and pnl > stagnant_profit:
            sell = min(0.20, remaining)
            realized += sell * pnl * POSITION_SIZE * (1 - SLIPPAGE)
            remaining -= sell
            action = "TP20%(stagnant)"

        # Rule 1: Holder growing → HOLD
        # (no action needed, default is hold)

        if action:
            actions.append((t, action, remaining, pnl))

    # Force close remaining
    if remaining > 0.001:
        final_pnl = (mcap[-1] - entry_price) / max(entry_price, 1)
        realized += remaining * final_pnl * POSITION_SIZE * (1 - SLIPPAGE)

    total_return = realized / POSITION_SIZE
    return total_return, actions


def run_fixed_strategy(mcap):
    """Fixed TP/SL for comparison."""
    FIXED_TP = [(0.30, 0.15), (0.80, 0.20), (2.00, 0.20), (5.00, 0.20), (10.00, 0.15)]
    FIXED_SL = [(-0.15, 0.30), (-0.30, 0.30), (-0.50, 1.00)]

    entry_price = mcap[0]
    remaining = 1.0
    realized = 0.0
    tp_triggered = [False] * len(FIXED_TP)
    sl_triggered = [False] * len(FIXED_SL)

    for t in range(len(mcap)):
        if remaining <= 0.001:
            break
        pnl = (mcap[t] - entry_price) / max(entry_price, 1)
        for j, (trigger, sell_pct) in enumerate(FIXED_TP):
            if not tp_triggered[j] and pnl >= trigger:
                s = min(sell_pct, remaining)
                realized += s * pnl * POSITION_SIZE * (1 - SLIPPAGE)
                remaining -= s
                tp_triggered[j] = True
        if pnl < 0:
            for j, (trigger, sell_pct) in enumerate(FIXED_SL):
                if not sl_triggered[j] and pnl <= trigger:
                    s = min(sell_pct, remaining)
                    realized += s * pnl * POSITION_SIZE * (1 - SLIPPAGE)
                    remaining -= s
                    sl_triggered[j] = True

    final_pnl = (mcap[-1] - entry_price) / max(entry_price, 1)
    realized += remaining * final_pnl * POSITION_SIZE * (1 - SLIPPAGE)
    return realized / POSITION_SIZE


# ── Backtest ─────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("Lifecycle Strategy — Holder-Based Position Management")
    print("=" * 60)
    print("Rules:")
    print("  Holder growing → HOLD (no matter what)")
    print("  Holder stagnant + profit → TP 20%")
    print("  Holder declining 6h → TP 25%")
    print("  Holder declining 12h + volume dying → TP 40%")
    print("  Holder declining 24h → EXIT")
    print("  -70% from peak + holder declining → EXIT (rug)")
    print()

    organic = load_organic_tokens()
    print(f"Organic tokens: {len(organic)}")

    lifecycle_rets = []
    fixed_rets = []
    hold_rets = []
    no_holder_data = 0

    for _, row in organic.iterrows():
        data = load_token_data(row["address"])
        if data is None:
            continue

        mcap = data["mcap"]
        volume = data["volume"]
        holders = data["holders"]

        # Skip if no holder data
        if holders.max() == 0:
            no_holder_data += 1
            continue

        # Lifecycle strategy
        lc_return, actions = run_lifecycle_strategy(mcap, volume, holders)
        lifecycle_rets.append(lc_return)

        # Fixed TP/SL
        fixed_ret = run_fixed_strategy(mcap)
        fixed_rets.append(fixed_ret)

        # Buy & Hold
        final_pnl = (mcap[-1] - mcap[0]) / max(mcap[0], 1)
        hold_rets.append(final_pnl * (1 - SLIPPAGE))

    print(f"Tokens with holder data: {len(lifecycle_rets)}")
    print(f"Tokens without holder data: {no_holder_data}")

    def stats(name, rets):
        r = np.array(rets)
        wins = r > 0
        gp = r[wins].sum() if wins.sum() > 0 else 0
        gl = abs(r[~wins].sum()) if (~wins).sum() > 0 else 1e-9
        sharpe = r.mean() / max(r.std(), 1e-9) * np.sqrt(252)

        print(f"\n  {name} ({len(r)} trades):")
        print(f"    Mean return:    {r.mean()*100:+.1f}%")
        print(f"    Median return:  {np.median(r)*100:+.1f}%")
        print(f"    Win rate:       {wins.mean()*100:.0f}%")
        if wins.sum() > 0:
            print(f"    Avg win:        {r[wins].mean()*100:+.1f}%")
        if (~wins).sum() > 0:
            print(f"    Avg loss:       {r[~wins].mean()*100:+.1f}%")
        print(f"    Profit Factor:  {gp/gl:.2f}")
        print(f"    Sharpe:         {sharpe:.2f}")
        return {"mean": r.mean(), "median": np.median(r), "winrate": wins.mean(),
                "pf": gp/gl, "sharpe": sharpe}

    print(f"\n{'='*60}")
    print(f"RESULTS: Full lifecycle (entry → end of data)")
    print(f"{'='*60}")

    m1 = stats("Lifecycle Strategy (Holder-Based)", lifecycle_rets)
    m2 = stats("Fixed TP/SL", fixed_rets)
    m3 = stats("Buy & Hold", hold_rets)

    print(f"\n{'='*70}")
    print(f"{'Metric':>20s} | {'Lifecycle':>12s} | {'Fixed TP/SL':>12s} | {'Buy & Hold':>12s}")
    print(f"{'-'*20}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    for key, label in [("mean", "Mean Return"), ("median", "Median Return"),
                        ("winrate", "Win Rate"), ("pf", "Profit Factor"), ("sharpe", "Sharpe")]:
        v1 = m1[key]; v2 = m2[key]; v3 = m3[key]
        if key in ("mean", "median"):
            print(f"  {label:>18s} | {v1*100:>+11.1f}% | {v2*100:>+11.1f}% | {v3*100:>+11.1f}%")
        elif key == "winrate":
            print(f"  {label:>18s} | {v1*100:>11.0f}% | {v2*100:>11.0f}% | {v3*100:>11.0f}%")
        else:
            print(f"  {label:>18s} | {v1:>12.2f} | {v2:>12.2f} | {v3:>12.2f}")


if __name__ == "__main__":
    main()
