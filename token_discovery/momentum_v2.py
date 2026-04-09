"""Momentum Model v2 — Phase-based position management.

4 phases based on market cap level:
  Incubation ($100K-$500K): Pure hold, no action
  Confirmation ($500K-$2M): Hard stop only (-50%)
  Growth ($2M-$10M): Momentum signal starts managing
  Maturity (>$10M): Full momentum management, aggressive TP

Momentum signal requires sustained bearish (N consecutive hours)
instead of single-point triggers.

Usage:
    PYTHONPATH=. python token_discovery/momentum_v2.py
"""

import csv
import glob
import json
import os
import sys
import pickle
from collections import deque

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
SLIPPAGE = 0.03

# Phase thresholds (mcap levels)
PHASE_INCUBATION = 500_000      # $100K-$500K: pure hold
PHASE_CONFIRMATION = 2_000_000  # $500K-$2M: hard stop only
PHASE_GROWTH = 10_000_000       # $2M-$10M: momentum starts
# >$10M: full momentum

# Sustained signal: require N consecutive bearish hours
SUSTAINED_BEAR_HOURS = 3


# ── Reuse from v1 ────────────────────────────────────────────────────────────

from token_discovery.momentum_model import (
    load_organic_tokens, load_token_history, load_holders,
    compute_features_at_t, FEATURE_NAMES, build_dataset, train_model,
)


# ── Phase-Based Backtest ─────────────────────────────────────────────────────


def get_phase(mcap):
    if mcap < PHASE_INCUBATION:
        return "incubation"
    elif mcap < PHASE_CONFIRMATION:
        return "confirmation"
    elif mcap < PHASE_GROWTH:
        return "growth"
    else:
        return "maturity"


def backtest_v2(model):
    """Backtest with phase-based momentum management.

    Only measures from entry ($100K) to ATH period.
    """
    print("\nPhase 3: Backtesting phase-based momentum management...")
    print(f"  Phases: Incubation(<${PHASE_INCUBATION/1e6:.1f}M) → Confirmation(<${PHASE_CONFIRMATION/1e6:.0f}M) → Growth(<${PHASE_GROWTH/1e6:.0f}M) → Maturity")
    print(f"  Sustained bear: {SUSTAINED_BEAR_HOURS} consecutive hours")

    organic = load_organic_tokens()

    results_momentum = []
    results_fixed = []
    results_hold = []

    FIXED_TP = [(0.30, 0.15), (0.80, 0.20), (2.00, 0.20), (5.00, 0.20), (10.00, 0.15)]
    FIXED_SL = [(-0.15, 0.30), (-0.30, 0.30), (-0.50, 1.00)]

    for idx, (_, row) in enumerate(organic.iterrows()):
        mcap, volume, addr = load_token_history(row["address"])
        if mcap is None or len(mcap) < 30:
            continue
        holders = load_holders(row["address"], len(mcap))
        n = len(mcap)
        entry_price = mcap[0]

        # Find ATH index (only backtest entry → ATH region + some post-ATH)
        ath_idx = np.argmax(mcap)
        # Extend a bit past ATH to capture the decline
        end_idx = min(ath_idx + max(24, int((ath_idx - 0) * 0.3)), n)

        # ── Momentum v2 Strategy ──
        remaining = 1.0
        realized = 0.0
        bear_count = 0  # consecutive bearish signal hours
        prev_signal = 0.0
        peak_signal = 0.0

        for t in range(min(12, end_idx), end_idx):
            if remaining <= 0.001:
                break

            current_mcap = mcap[t]
            pnl = (current_mcap - entry_price) / max(entry_price, 1)
            phase = get_phase(current_mcap)

            # Compute momentum signal
            feat = compute_features_at_t(mcap, volume, holders, t)
            if feat:
                vec = np.array([[feat.get(f, 0) for f in FEATURE_NAMES]])
                vec = np.nan_to_num(vec, nan=0)
                signal = float(model.predict(vec)[0])
            else:
                signal = 0

            peak_signal = max(peak_signal, signal)

            # Track sustained bearish
            if signal < -0.03:
                bear_count += 1
            else:
                bear_count = 0

            # Phase-based decisions
            if phase == "incubation":
                # Pure hold — no action regardless of signal
                pass

            elif phase == "confirmation":
                # Only hard stop at -50%
                if pnl < -0.50:
                    realized += remaining * pnl * POSITION_SIZE * (1 - SLIPPAGE)
                    remaining = 0

            elif phase == "growth":
                # Momentum starts — but only on sustained signals
                if bear_count >= SUSTAINED_BEAR_HOURS and signal < -0.05:
                    # Sustained bearish in growth: reduce 30%
                    sell = min(0.30, remaining)
                    realized += sell * pnl * POSITION_SIZE * (1 - SLIPPAGE)
                    remaining -= sell

                elif pnl > 0.50 and signal < 0.01 and prev_signal > 0.05:
                    # Momentum fading with profit: TP 20%
                    sell = min(0.20, remaining)
                    realized += sell * pnl * POSITION_SIZE * (1 - SLIPPAGE)
                    remaining -= sell

            elif phase == "maturity":
                # Full momentum management — aggressive
                if bear_count >= SUSTAINED_BEAR_HOURS and signal < -0.08:
                    # Sustained strong bearish: exit most
                    sell = min(0.50, remaining)
                    realized += sell * pnl * POSITION_SIZE * (1 - SLIPPAGE)
                    remaining -= sell

                elif bear_count >= SUSTAINED_BEAR_HOURS and signal < -0.03:
                    # Sustained moderate bearish: reduce
                    sell = min(0.25, remaining)
                    realized += sell * pnl * POSITION_SIZE * (1 - SLIPPAGE)
                    remaining -= sell

                elif pnl > 1.0 and signal < 0.02 and prev_signal > 0.05:
                    # Big profit + momentum fading: TP
                    sell = min(0.25, remaining)
                    realized += sell * pnl * POSITION_SIZE * (1 - SLIPPAGE)
                    remaining -= sell

                elif pnl > 3.0 and signal < prev_signal * 0.3:
                    # Huge profit + momentum collapsing: large TP
                    sell = min(0.35, remaining)
                    realized += sell * pnl * POSITION_SIZE * (1 - SLIPPAGE)
                    remaining -= sell

            prev_signal = signal

        # Close remaining at end of backtest window
        final_pnl = (mcap[end_idx - 1] - entry_price) / max(entry_price, 1)
        realized += remaining * final_pnl * POSITION_SIZE * (1 - SLIPPAGE)
        mom_return = realized / POSITION_SIZE - 1

        # ── Fixed TP/SL ──
        rem_f = 1.0
        real_f = 0.0
        tp_f = [False] * len(FIXED_TP)
        sl_f = [False] * len(FIXED_SL)

        for t in range(end_idx):
            if rem_f <= 0.001:
                break
            pnl = (mcap[t] - entry_price) / max(entry_price, 1)
            for j, (trigger, sell_pct) in enumerate(FIXED_TP):
                if not tp_f[j] and pnl >= trigger:
                    s = min(sell_pct, rem_f)
                    real_f += s * pnl * POSITION_SIZE * (1 - SLIPPAGE)
                    rem_f -= s
                    tp_f[j] = True
            if pnl < 0:
                for j, (trigger, sell_pct) in enumerate(FIXED_SL):
                    if not sl_f[j] and pnl <= trigger:
                        s = min(sell_pct, rem_f)
                        real_f += s * pnl * POSITION_SIZE * (1 - SLIPPAGE)
                        rem_f -= s
                        sl_f[j] = True
        real_f += rem_f * final_pnl * POSITION_SIZE * (1 - SLIPPAGE)
        fixed_return = real_f / POSITION_SIZE - 1

        # ── Buy & Hold to end of window ──
        hold_return = final_pnl * (1 - SLIPPAGE)

        results_momentum.append(mom_return)
        results_fixed.append(fixed_return)
        results_hold.append(hold_return)

    # ── Results ──
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
        print(f"    Max single:     {r.max()*100:+.0f}%")
        print(f"    Min single:     {r.min()*100:+.0f}%")

        return {"mean": r.mean(), "median": np.median(r), "winrate": wins.mean(),
                "pf": gp/gl, "sharpe": sharpe}

    print(f"\n{'='*60}")
    print(f"RESULTS: Entry($100K) → ATH region ({len(results_momentum)} tokens)")
    print(f"{'='*60}")

    m1 = stats("Momentum v2 (Phase-Based)", results_momentum)
    m2 = stats("Fixed TP/SL", results_fixed)
    m3 = stats("Buy & Hold", results_hold)

    # Comparison table
    print(f"\n{'='*60}")
    print(f"{'Metric':>20s} | {'Momentum v2':>12s} | {'Fixed TP/SL':>12s} | {'Buy & Hold':>12s}")
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

    return results_momentum, results_fixed, results_hold


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("Momentum Model v2 — Phase-Based Position Management")
    print("=" * 60)

    # Phase 1+2: Reuse v1 training (same features, same model)
    df = build_dataset()
    model, y_pred, y_true = train_model(df)

    # Phase 3: Backtest with phase-based rules
    mom_rets, fixed_rets, hold_rets = backtest_v2(model)

    # Phase 4: Save
    path = os.path.join(OUTPUT_DIR, "momentum_model_v2.cbm")
    model.save_model(path)
    print(f"\nModel saved to {path}")


if __name__ == "__main__":
    main()
