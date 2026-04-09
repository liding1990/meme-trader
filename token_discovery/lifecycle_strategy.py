"""Lifecycle Strategy v3 — Phase-based holder position management.

Core principle: don't use price for stop loss. Use "life signals":
  - Holder still growing → HOLD (no matter how much price drops)
  - Holder stagnating → start taking profit
  - Holder declining → accelerate selling
  - Holder collapsing + volume dying → EXIT

3 MCap phases control how aggressively we manage:
  Early (<$500K): Pure hold, only catastrophic rug protection
  Growth ($500K-$2M): Relaxed thresholds (8h/15h/30h)
  Maturity (>$2M): Standard rules + trailing profit lock + momentum fading

Rich signals: holder trend/accel/velocity, volume trend/surge/dry,
  price ROC multi-timeframe, VWAP deviation, mcap/holder efficiency,
  composite sell pressure score.

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
    """Compute rich life signals at time t.

    Returns dict with holder, volume, price, and composite signals.
    """
    if t < window:
        return None

    # ── Holder signals ──
    h_window = holders[t-window:t+1]
    if h_window[0] > 0:
        holder_changes = np.diff(h_window) / np.maximum(h_window[:-1], 1)
        holder_trend = np.mean(holder_changes)
    else:
        holder_trend = 0

    # Holder acceleration (is growth speeding up or slowing down?)
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

    # Short-term holder trend (4h) — more responsive
    if t >= 4 and holders[t-4] > 0:
        holder_trend_4h = (holders[t] - holders[t-4]) / max(holders[t-4], 1)
    else:
        holder_trend_4h = 0

    # Consecutive declining hours (scan up to 48h back)
    declining_hours = 0
    for i in range(t, max(t-48, 0), -1):
        if i > 0 and holders[i] < holders[i-1]:
            declining_hours += 1
        else:
            break

    # Holder velocity: absolute new holders per hour (not just %)
    if t >= 4:
        holder_velocity = (holders[t] - holders[t-4]) / 4
    else:
        holder_velocity = 0

    # ── Volume signals ──
    vol_recent = volume[max(0,t-6):t+1].mean()
    vol_hist = volume[max(0,t-24):t+1].mean()
    volume_trend = vol_recent / max(vol_hist, 1)

    # Volume surge: is current hour volume spiking?
    vol_4h_avg = volume[max(0,t-4):t+1].mean()
    vol_24h_avg = volume[max(0,t-24):t+1].mean() if t >= 24 else vol_4h_avg
    volume_surge = volume[t] / max(vol_24h_avg, 1)

    # Volume drying: consecutive hours of below-average volume
    vol_dry_hours = 0
    if t >= 24:
        vol_ma = volume[max(0,t-24):t+1].mean()
        for i in range(t, max(t-24, 0), -1):
            if volume[i] < vol_ma * 0.5:
                vol_dry_hours += 1
            else:
                break

    # ── Price momentum signals ──
    # ROC at multiple timeframes
    roc_1h = (mcap[t] - mcap[t-1]) / max(mcap[t-1], 1) if t >= 1 else 0
    roc_4h = (mcap[t] - mcap[t-4]) / max(mcap[t-4], 1) if t >= 4 else 0
    roc_12h = (mcap[t] - mcap[t-12]) / max(mcap[t-12], 1) if t >= 12 else 0

    # VWAP deviation — price above/below volume-weighted average
    vwap_len = min(24, t+1)
    cum_vol = volume[t-vwap_len+1:t+1].sum()
    cum_vp = (mcap[t-vwap_len+1:t+1] * volume[t-vwap_len+1:t+1]).sum()
    vwap = cum_vp / max(cum_vol, 1)
    price_vs_vwap = mcap[t] / max(vwap, 1) - 1  # >0 means above VWAP

    # Drawdown from running peak
    peak = np.max(mcap[:t+1])
    drawdown = (mcap[t] - peak) / max(peak, 1)

    # Price position in 24h range (0=at low, 1=at high)
    recent = mcap[max(0,t-24):t+1]
    price_range = recent.max() - recent.min()
    price_position = (mcap[t] - recent.min()) / max(price_range, 1)

    # ── MCap/Holder efficiency ──
    mcap_per_holder = mcap[t] / max(holders[t], 1)
    if t >= 12 and holders[t-12] > 0:
        prev_mph = mcap[t-12] / max(holders[t-12], 1)
        mph_trend = (mcap_per_holder - prev_mph) / max(prev_mph, 1)
    else:
        mph_trend = 0

    return {
        # Holder
        "holder_trend": holder_trend,
        "holder_trend_4h": holder_trend_4h,
        "holder_accel": holder_accel,
        "holder_growing": holder_trend > 0.001,
        "declining_hours": declining_hours,
        "holder_velocity": holder_velocity,
        # Volume
        "volume_trend": volume_trend,
        "volume_surge": volume_surge,
        "vol_dry_hours": vol_dry_hours,
        # Price
        "roc_1h": roc_1h,
        "roc_4h": roc_4h,
        "roc_12h": roc_12h,
        "price_vs_vwap": price_vs_vwap,
        "drawdown": drawdown,
        "price_position": price_position,
        # Efficiency
        "mcap_per_holder": mcap_per_holder,
        "mph_trend": mph_trend,
    }


# ── Strategy ─────────────────────────────────────────────────────────────────


def get_mcap_phase(mcap_value):
    """Determine market phase based on current mcap.

    3 phases:
      Early (<$500K): Pure hold, only catastrophic rug protection
      Growth ($500K-$2M): Relaxed thresholds (1.5x standard)
      Maturity (>$2M): Full management with trailing locks + momentum
    """
    if mcap_value < 500_000:
        return "early"
    elif mcap_value < 2_000_000:
        return "growth"
    else:
        return "maturity"


def compute_sell_pressure(signals, pnl, drawdown_from_peak, phase):
    """Compute a composite sell pressure score [0, 1].

    Combines multiple weak signals into one strong signal.
    Higher = more urgency to sell.
    """
    pressure = 0.0

    # Holder decline — strongest signal
    dh = signals["declining_hours"]
    if dh >= 24:
        pressure += 0.5
    elif dh >= 12:
        pressure += 0.3
    elif dh >= 6:
        pressure += 0.15

    # Holder acceleration turning negative (growth decelerating)
    if signals["holder_accel"] < -0.005:
        pressure += 0.1
    elif signals["holder_accel"] < -0.002:
        pressure += 0.05

    # Volume drying up
    if signals["vol_dry_hours"] >= 12:
        pressure += 0.15
    elif signals["vol_dry_hours"] >= 6:
        pressure += 0.08
    elif signals["volume_trend"] < 0.3:
        pressure += 0.1

    # Price below VWAP — sellers in control
    if signals["price_vs_vwap"] < -0.15:
        pressure += 0.1
    elif signals["price_vs_vwap"] < -0.05:
        pressure += 0.05

    # Price momentum all negative — multi-timeframe confirmation
    neg_count = sum(1 for r in [signals["roc_1h"], signals["roc_4h"], signals["roc_12h"]] if r < -0.02)
    if neg_count == 3:
        pressure += 0.15
    elif neg_count >= 2:
        pressure += 0.08

    # MCap/holder efficiency declining (price falling faster than holders leaving = dump)
    if signals["mph_trend"] < -0.2:
        pressure += 0.1

    # Drawdown from peak — deeper drawdown adds urgency
    if drawdown_from_peak < -0.50:
        pressure += 0.1
    elif drawdown_from_peak < -0.30:
        pressure += 0.05

    return min(pressure, 1.0)


def run_lifecycle_strategy(mcap, volume, holders):
    """Run lifecycle strategy v3 — 3-phase mcap gates + rich signals.

    Phase gates control WHEN management activates:
      Early (<$500K): Pure hold, only catastrophic rug protection
      Growth ($500K-$2M): Relaxed thresholds + stagnant+profit TP
      Maturity (>$2M): Standard rules + trailing profit lock + momentum fading
    """
    n = len(mcap)
    entry_price = mcap[0]
    remaining = 1.0
    realized = 0.0
    peak_price = entry_price
    peak_pnl = 0.0
    actions = []

    for t in range(n):
        if remaining <= 0.001:
            break

        current = mcap[t]
        pnl = (current - entry_price) / max(entry_price, 1)
        peak_price = max(peak_price, current)
        peak_pnl = max(peak_pnl, pnl)
        drawdown_from_peak = (current - peak_price) / max(peak_price, 1)

        signals = compute_life_signals(mcap, volume, holders, t)
        if signals is None:
            continue

        action = None
        phase = get_mcap_phase(current)
        sp = compute_sell_pressure(signals, pnl, drawdown_from_peak, phase)

        # ── Early Stage (<$500K): Hold with rug protection + dead token cut ──
        if phase == "early":
            if drawdown_from_peak < -0.80 and signals["declining_hours"] >= 12:
                sell = remaining
                realized += sell * current * (1 - SLIPPAGE) / entry_price * POSITION_SIZE - sell * POSITION_SIZE
                action = "EXIT(rug-early)"
                remaining = 0

            # Dead token: holder declining 20h+ AND volume dead AND price tanking
            elif (signals["declining_hours"] >= 20
                    and signals["volume_trend"] < 0.3
                    and signals["roc_12h"] < -0.15):
                sell = remaining
                realized += sell * pnl * POSITION_SIZE * (1 - SLIPPAGE)
                action = "EXIT(dead-early)"
                remaining = 0

        # ── Growth Stage ($500K-$2M): Relaxed 1.5x thresholds ──
        elif phase == "growth":
            if drawdown_from_peak < -0.70 and signals["declining_hours"] >= 6:
                sell = remaining
                realized += sell * current * (1 - SLIPPAGE) / entry_price * POSITION_SIZE - sell * POSITION_SIZE
                action = "EXIT(rug-growth)"
                remaining = 0

            elif signals["declining_hours"] >= 30:
                sell = remaining
                realized += sell * pnl * POSITION_SIZE * (1 - SLIPPAGE)
                action = "EXIT(decline-growth)"
                remaining = 0

            elif signals["declining_hours"] >= 15 and signals["volume_trend"] < 0.5:
                sell = min(0.35, remaining)
                realized += sell * pnl * POSITION_SIZE * (1 - SLIPPAGE)
                remaining -= sell
                action = "TP35%(decline+vol-growth)"

            elif signals["declining_hours"] >= 8:
                sell = min(0.20, remaining)
                realized += sell * pnl * POSITION_SIZE * (1 - SLIPPAGE)
                remaining -= sell
                action = "TP20%(decline-growth)"

            # Holder stagnant + profit → TP 15% (core v2 rule)
            elif abs(signals["holder_trend"]) < 0.001 and pnl > 0.80:
                sell = min(0.15, remaining)
                realized += sell * pnl * POSITION_SIZE * (1 - SLIPPAGE)
                remaining -= sell
                action = "TP15%(stagnant-growth)"

            # Multi-signal fading: lower profit bar but needs momentum confirmation
            elif (abs(signals["holder_trend"]) < 0.001
                    and pnl > 0.40
                    and signals["roc_4h"] < -0.03
                    and signals["price_vs_vwap"] < -0.05):
                sell = min(0.15, remaining)
                realized += sell * pnl * POSITION_SIZE * (1 - SLIPPAGE)
                remaining -= sell
                action = "TP15%(fade-growth)"

        # ── Maturity Stage (>$2M): v2 base + enhanced signals ──
        else:
            # Rug protection
            if drawdown_from_peak < -0.70 and not signals["holder_growing"]:
                sell = remaining
                realized += sell * current * (1 - SLIPPAGE) / entry_price * POSITION_SIZE - sell * POSITION_SIZE
                action = "EXIT(rug-maturity)"
                remaining = 0

            # CORE: Holder declining 24h → EXIT (same as v2)
            elif signals["declining_hours"] >= 24:
                sell = remaining
                realized += sell * pnl * POSITION_SIZE * (1 - SLIPPAGE)
                action = "EXIT(decline-maturity)"
                remaining = 0

            # CORE: Holder declining 12h + volume dying → TP 40% (same as v2)
            elif signals["declining_hours"] >= 12 and signals["volume_trend"] < 0.5:
                sell = min(0.40, remaining)
                realized += sell * pnl * POSITION_SIZE * (1 - SLIPPAGE)
                remaining -= sell
                action = "TP40%(decline+vol-maturity)"

            # CORE: Holder declining 6h → TP 25% (same as v2)
            elif signals["declining_hours"] >= 6:
                sell = min(0.25, remaining)
                realized += sell * pnl * POSITION_SIZE * (1 - SLIPPAGE)
                remaining -= sell
                action = "TP25%(decline-maturity)"

            # NEW: Trailing profit lock — only after core rules pass
            # Protect big gains when composite pressure is high
            elif peak_pnl >= 5.0 and drawdown_from_peak < -0.35 and sp >= 0.35:
                sell = min(0.40, remaining)
                realized += sell * pnl * POSITION_SIZE * (1 - SLIPPAGE)
                remaining -= sell
                action = "TP40%(trailing-lock)"

            elif peak_pnl >= 2.0 and drawdown_from_peak < -0.45 and sp >= 0.35:
                sell = min(0.30, remaining)
                realized += sell * pnl * POSITION_SIZE * (1 - SLIPPAGE)
                remaining -= sell
                action = "TP30%(trailing-lock)"

            # NEW: Multi-signal momentum fading
            elif (pnl > 0.80
                    and signals["holder_accel"] < -0.003
                    and signals["roc_4h"] < -0.02
                    and signals["price_vs_vwap"] < -0.03):
                sell = min(0.20, remaining)
                realized += sell * pnl * POSITION_SIZE * (1 - SLIPPAGE)
                remaining -= sell
                action = "TP20%(fade-maturity)"

            # CORE: Holder stagnant + profit → TP 20% (same as v2)
            elif abs(signals["holder_trend"]) < 0.001 and pnl > 0.50:
                sell = min(0.20, remaining)
                realized += sell * pnl * POSITION_SIZE * (1 - SLIPPAGE)
                remaining -= sell
                action = "TP20%(stagnant-maturity)"

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
    print("Lifecycle Strategy v3 — Phase-Based Position Management")
    print("=" * 60)
    print("Phases:")
    print("  Early (<$500K):  Pure hold, rug protection only")
    print("  Growth ($500K-$2M): Relaxed thresholds (8h/15h/30h)")
    print("  Maturity (>$2M):  Standard rules + trailing lock + momentum")
    print("Signals:")
    print("  Holder trend/accel/velocity, Volume trend/surge/dry")
    print("  Price ROC(1/4/12h), VWAP deviation, Composite sell pressure")
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
