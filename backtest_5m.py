"""5-minute bar backtesting: HMM regime + P4 indicators + P1 survival exit.

Integrates all signals:
  - HMM: regime detection (entry signal) — run on hourly-equivalent features
  - P4: ROC acceleration, MACD histogram slope, RVOL (trade management)
  - P1: Survival hazard rate (exit timing)
  - Trailing stop with adaptive tightening

Usage:
    python backtest_5m.py run [--params]
    python backtest_5m.py sweep
"""

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

from indicators import extract_5m_candles, compute_all_indicators, generate_trade_management_signals
from survival import load_models as load_survival_models, compute_hazard_rate
from space4d import load_token_list

DATA_DIR = "data"
MCAP_THRESHOLD = 50_000


class Strategy5m:
    """Combined 5-minute strategy with all signal layers."""

    def __init__(self,
                 stop_loss=15.0,
                 tight_stop_loss=8.0,
                 min_mcap=100_000,
                 min_bars=12,
                 roc_entry_threshold=3.0,
                 rvol_entry_threshold=1.5,
                 hazard_exit_threshold=0.15,
                 grace_period=6,
                 require_multi_bar_confirmation=True,
                 dynamic_hazard=True):
        """
        Args:
            stop_loss: default trailing stop %
            tight_stop_loss: tightened stop when momentum weakening
            min_mcap: minimum mcap to enter
            min_bars: minimum bars before entry (= min_bars * 5 minutes)
            roc_entry_threshold: ROC 30m must exceed this to enter
            rvol_entry_threshold: RVOL must exceed this to enter
            hazard_exit_threshold: exit when P(pump dies next bar) exceeds this
            grace_period: bars to hold before trailing stop activates
            require_multi_bar_confirmation: require 2+ consecutive positive bars before entry
            dynamic_hazard: relax hazard threshold when in profit, tighten when losing
        """
        self.stop_loss = stop_loss
        self.tight_stop_loss = tight_stop_loss
        self.min_mcap = min_mcap
        self.min_bars = min_bars
        self.roc_entry_threshold = roc_entry_threshold
        self.rvol_entry_threshold = rvol_entry_threshold
        self.hazard_exit_threshold = hazard_exit_threshold
        self.grace_period = grace_period
        self.require_multi_bar_confirmation = require_multi_bar_confirmation
        self.dynamic_hazard = dynamic_hazard

    def simulate(self, df, kmf=None):
        """Run strategy on a 5-min DataFrame with indicators.

        Args:
            df: DataFrame from compute_all_indicators + generate_trade_management_signals
            kmf: KaplanMeierFitter for hazard rate (optional)
        """
        if len(df) < self.min_bars + 10:
            return []

        trades = []
        in_trade = False
        entry_idx = None
        peak_mcap = 0
        bars_held = 0

        for idx in range(self.min_bars, len(df)):
            row = df.iloc[idx]
            mcap = row["mcap"]

            if not in_trade:
                # --- ENTRY CONDITIONS ---
                # min_mcap filter: skip if data looks like mcap (>100) and below threshold
                # Codex data is price (<1), GMGN data is mcap (>1000) — auto-detect
                if mcap > 100 and mcap < self.min_mcap:
                    continue

                roc = row.get("roc_30m", 0)
                rvol = row.get("rvol", 0)
                roc_accel = row.get("roc_accel_30m", 0)

                if pd.isna(roc) or pd.isna(rvol):
                    continue

                # Entry: positive ROC + above-average volume + acceleration
                entry_ok = (roc > self.roc_entry_threshold and
                            rvol > self.rvol_entry_threshold and
                            roc_accel > 0)

                # Ehlers EBSW regime filter: only trade in trending markets
                ebsw = row.get("ebsw", np.nan)
                if entry_ok and not pd.isna(ebsw) and ebsw < 0:
                    entry_ok = False  # cycling market = sit out

                # Ehlers price above ITrend = uptrend confirmation
                above_itrend = row.get("above_itrend", 1)
                if entry_ok and above_itrend == 0:
                    entry_ok = False  # below trendline = no long entry

                # OFI confirmation: if buy/sell data available, require net buying pressure
                ofi = row.get("ofi_30m", np.nan)
                if entry_ok and not pd.isna(ofi) and ofi < 0:
                    entry_ok = False  # no entry if selling pressure dominates

                # Multi-bar confirmation: previous bar also had positive ROC
                if entry_ok and self.require_multi_bar_confirmation and idx >= 1:
                    prev_roc = df.iloc[idx - 1].get("roc_30m", 0)
                    if pd.isna(prev_roc) or prev_roc <= 0:
                        entry_ok = False

                if entry_ok:
                    in_trade = True
                    entry_idx = idx
                    peak_mcap = mcap
                    bars_held = 0

            else:
                bars_held += 1
                entry_mcap = df.iloc[entry_idx]["mcap"]

                # Track peak
                if mcap > peak_mcap:
                    peak_mcap = mcap

                # --- EXIT CONDITIONS ---
                should_exit = False
                reason = None

                # 1. Survival hazard: P(pump dies) too high
                # Dynamic: relax threshold when profitable, tighten when losing
                if kmf is not None and bars_held >= 3:
                    hazard = compute_hazard_rate(kmf, bars_held)
                    pnl_pct = (mcap / entry_mcap - 1) * 100 if entry_mcap > 0 else 0

                    if self.dynamic_hazard:
                        if pnl_pct > 5:
                            # In profit >5%: relax hazard (let it run)
                            effective_threshold = self.hazard_exit_threshold * 1.5
                        elif pnl_pct < -3:
                            # Losing >3%: tighten hazard (get out faster)
                            effective_threshold = self.hazard_exit_threshold * 0.7
                        else:
                            effective_threshold = self.hazard_exit_threshold
                    else:
                        effective_threshold = self.hazard_exit_threshold

                    if hazard > effective_threshold:
                        should_exit = True
                        reason = "hazard"

                # 2. Trailing stop (with grace period and adaptive tightening)
                if not should_exit and bars_held >= self.grace_period:
                    tighten = row.get("tighten_stop", False)
                    current_stop = self.tight_stop_loss if tighten else self.stop_loss

                    if peak_mcap > 0:
                        drawdown = (1 - mcap / peak_mcap) * 100
                        if drawdown > current_stop:
                            should_exit = True
                            reason = "trailing_stop_tight" if tighten else "trailing_stop"

                # 3. Hard stop (catastrophic loss protection, always active)
                if not should_exit and entry_mcap > 0:
                    pnl = (mcap / entry_mcap - 1) * 100
                    if pnl < -self.stop_loss * 2:
                        should_exit = True
                        reason = "hard_stop"

                # 4. Momentum collapsed: ROC deeply negative
                if not should_exit and bars_held >= self.grace_period:
                    roc = row.get("roc_30m", 0)
                    if not pd.isna(roc) and roc < -self.roc_entry_threshold * 2:
                        should_exit = True
                        reason = "momentum_collapse"

                if should_exit:
                    ret = (mcap / entry_mcap - 1) * 100 if entry_mcap > 0 else 0
                    trades.append({
                        "entry_idx": entry_idx,
                        "exit_idx": idx,
                        "entry_mcap": entry_mcap,
                        "exit_mcap": mcap,
                        "peak_mcap": peak_mcap,
                        "return_pct": ret,
                        "peak_return_pct": (peak_mcap / entry_mcap - 1) * 100 if entry_mcap > 0 else 0,
                        "bars_held": bars_held,
                        "exit_reason": reason,
                    })
                    in_trade = False
                    entry_idx = None
                    peak_mcap = 0
                    bars_held = 0

        # Close open position
        if in_trade and entry_idx is not None:
            last = df.iloc[-1]
            entry_mcap = df.iloc[entry_idx]["mcap"]
            ret = (last["mcap"] / entry_mcap - 1) * 100 if entry_mcap > 0 else 0
            trades.append({
                "entry_idx": entry_idx,
                "exit_idx": len(df) - 1,
                "entry_mcap": entry_mcap,
                "exit_mcap": last["mcap"],
                "peak_mcap": peak_mcap,
                "return_pct": ret,
                "peak_return_pct": (peak_mcap / entry_mcap - 1) * 100 if entry_mcap > 0 else 0,
                "bars_held": bars_held,
                "exit_reason": "end_of_data",
            })

        return trades


def split_tokens(tokens, train_ratio=0.6):
    train, test = [], []
    for t in tokens:
        h = int(hashlib.md5(t["address"].encode()).hexdigest(), 16) % 100
        if h < train_ratio * 100:
            train.append(t)
        else:
            test.append(t)
    return train, test


def run_on_tokens(strategy, tokens, kmf=None):
    all_trades = []
    processed = 0
    for token in tokens:
        df = extract_5m_candles(token["address"])
        if df is None or len(df) < 30:
            continue

        df = compute_all_indicators(df)
        df = generate_trade_management_signals(df)

        trades = strategy.simulate(df, kmf=kmf)
        for t in trades:
            t["symbol"] = token["symbol"]
            t["address"] = token["address"]
        all_trades.extend(trades)
        processed += 1

    return all_trades, processed


def print_stats(trades, label=""):
    if not trades:
        print(f"  {label}: No trades")
        return {}

    rets = [t["return_pct"] for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    gp = sum(wins) if wins else 0
    gl = abs(sum(losses)) if losses else 1
    pf = gp / gl if gl > 0 else float("inf")
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    reasons = {}
    for t in trades:
        r = t.get("exit_reason", "unknown")
        reasons[r] = reasons.get(r, 0) + 1

    # Captured vs left on table
    peak_rets = [t["peak_return_pct"] for t in trades]
    capture_ratio = np.mean(rets) / np.mean(peak_rets) * 100 if np.mean(peak_rets) > 0 else 0

    stats = {
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100,
        "avg_return": np.mean(rets),
        "med_return": np.median(rets),
        "profit_factor": pf,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "wl_ratio": avg_win / abs(avg_loss) if avg_loss else 0,
        "avg_bars": np.mean([t["bars_held"] for t in trades]),
        "avg_peak": np.mean(peak_rets),
        "capture_ratio": capture_ratio,
    }

    print(f"\n  {label}")
    print(f"  {'─'*65}")
    print(f"  Trades:          {stats['trades']}")
    print(f"  Win rate:        {stats['win_rate']:.1f}%")
    print(f"  Avg return:      {stats['avg_return']:+.2f}%")
    print(f"  Med return:      {stats['med_return']:+.2f}%")
    print(f"  Profit factor:   {stats['profit_factor']:.2f}")
    print(f"  Avg win:         {stats['avg_win']:+.1f}%  |  Avg loss: {stats['avg_loss']:+.1f}%")
    print(f"  W/L ratio:       {stats['wl_ratio']:.2f}x")
    print(f"  Avg hold:        {stats['avg_bars']:.0f} bars ({stats['avg_bars']*5:.0f} min)")
    print(f"  Avg peak return: {stats['avg_peak']:+.1f}% (captured {stats['capture_ratio']:.0f}%)")
    print(f"  Exit reasons:    {reasons}")

    return stats


def cmd_run(args):
    tokens = load_token_list()
    train_tokens, test_tokens = split_tokens(tokens)

    # Load survival model
    try:
        kmf, _ = load_survival_models()
    except Exception:
        print("WARNING: Survival model not found, running without hazard exit")
        kmf = None

    strategy = Strategy5m(
        stop_loss=args.stop_loss,
        tight_stop_loss=args.tight_stop,
        min_mcap=args.min_mcap,
        roc_entry_threshold=args.roc_threshold,
        rvol_entry_threshold=args.rvol_threshold,
        hazard_exit_threshold=args.hazard_threshold,
        grace_period=args.grace_period,
    )

    print(f"\n{'='*70}")
    print(f"  5-Minute Combined Strategy Backtest")
    print(f"  SL={args.stop_loss}% TightSL={args.tight_stop}% ROC>{args.roc_threshold} RVOL>{args.rvol_threshold}")
    print(f"  Hazard>{args.hazard_threshold} Grace={args.grace_period} bars MinMCap=${args.min_mcap:,}")
    print(f"{'='*70}")

    train_trades, train_n = run_on_tokens(strategy, train_tokens, kmf=kmf)
    test_trades, test_n = run_on_tokens(strategy, test_tokens, kmf=kmf)

    print(f"  Train: {len(train_tokens)} tokens ({train_n} with 5m data)")
    print(f"  Test:  {len(test_tokens)} tokens ({test_n} with 5m data)")

    train_stats = print_stats(train_trades, "TRAIN SET")
    test_stats = print_stats(test_trades, "TEST SET")

    if train_stats and test_stats:
        wr_gap = abs(train_stats["win_rate"] - test_stats["win_rate"])
        pf_gap = abs(train_stats["profit_factor"] - test_stats["profit_factor"])
        print(f"\n  Overfit check: WR gap={wr_gap:.1f}pp, PF gap={pf_gap:.2f}")

    print(f"{'='*70}")

    all_trades = train_trades + test_trades
    if all_trades:
        pd.DataFrame(all_trades).to_csv(os.path.join(DATA_DIR, "backtest_5m_trades.csv"), index=False)


def cmd_sweep(args):
    tokens = load_token_list()
    train_tokens, test_tokens = split_tokens(tokens)

    try:
        kmf, _ = load_survival_models()
    except Exception:
        kmf = None

    configs = [
        # === BASELINE (previous best) ===
        {"roc_entry_threshold": 3.0, "rvol_entry_threshold": 1.5, "stop_loss": 15, "tight_stop_loss": 8, "hazard_exit_threshold": 0.10, "grace_period": 6, "min_mcap": 100000, "require_multi_bar_confirmation": False, "dynamic_hazard": False},
        # === Multi-bar confirmation (filter false entries) ===
        {"roc_entry_threshold": 3.0, "rvol_entry_threshold": 1.5, "stop_loss": 15, "tight_stop_loss": 8, "hazard_exit_threshold": 0.10, "grace_period": 6, "min_mcap": 100000, "require_multi_bar_confirmation": True, "dynamic_hazard": False},
        # === Dynamic hazard (let winners run) ===
        {"roc_entry_threshold": 3.0, "rvol_entry_threshold": 1.5, "stop_loss": 15, "tight_stop_loss": 8, "hazard_exit_threshold": 0.10, "grace_period": 6, "min_mcap": 100000, "require_multi_bar_confirmation": False, "dynamic_hazard": True},
        # === Both improvements ===
        {"roc_entry_threshold": 3.0, "rvol_entry_threshold": 1.5, "stop_loss": 15, "tight_stop_loss": 8, "hazard_exit_threshold": 0.10, "grace_period": 6, "min_mcap": 100000, "require_multi_bar_confirmation": True, "dynamic_hazard": True},
        # === Both + varying hazard threshold ===
        {"roc_entry_threshold": 3.0, "rvol_entry_threshold": 1.5, "stop_loss": 15, "tight_stop_loss": 8, "hazard_exit_threshold": 0.08, "grace_period": 6, "min_mcap": 100000, "require_multi_bar_confirmation": True, "dynamic_hazard": True},
        {"roc_entry_threshold": 3.0, "rvol_entry_threshold": 1.5, "stop_loss": 15, "tight_stop_loss": 8, "hazard_exit_threshold": 0.12, "grace_period": 6, "min_mcap": 100000, "require_multi_bar_confirmation": True, "dynamic_hazard": True},
        {"roc_entry_threshold": 3.0, "rvol_entry_threshold": 1.5, "stop_loss": 15, "tight_stop_loss": 8, "hazard_exit_threshold": 0.15, "grace_period": 6, "min_mcap": 100000, "require_multi_bar_confirmation": True, "dynamic_hazard": True},
        # === Both + varying entry thresholds ===
        {"roc_entry_threshold": 5.0, "rvol_entry_threshold": 1.5, "stop_loss": 15, "tight_stop_loss": 8, "hazard_exit_threshold": 0.10, "grace_period": 6, "min_mcap": 100000, "require_multi_bar_confirmation": True, "dynamic_hazard": True},
        {"roc_entry_threshold": 3.0, "rvol_entry_threshold": 2.0, "stop_loss": 15, "tight_stop_loss": 8, "hazard_exit_threshold": 0.10, "grace_period": 6, "min_mcap": 100000, "require_multi_bar_confirmation": True, "dynamic_hazard": True},
        # === Both + mcap variations ===
        {"roc_entry_threshold": 3.0, "rvol_entry_threshold": 1.5, "stop_loss": 15, "tight_stop_loss": 8, "hazard_exit_threshold": 0.10, "grace_period": 6, "min_mcap": 50000, "require_multi_bar_confirmation": True, "dynamic_hazard": True},
        {"roc_entry_threshold": 3.0, "rvol_entry_threshold": 1.5, "stop_loss": 15, "tight_stop_loss": 8, "hazard_exit_threshold": 0.10, "grace_period": 6, "min_mcap": 200000, "require_multi_bar_confirmation": True, "dynamic_hazard": True},
        # === Both + stop loss variations ===
        {"roc_entry_threshold": 3.0, "rvol_entry_threshold": 1.5, "stop_loss": 10, "tight_stop_loss": 5, "hazard_exit_threshold": 0.10, "grace_period": 6, "min_mcap": 100000, "require_multi_bar_confirmation": True, "dynamic_hazard": True},
        {"roc_entry_threshold": 3.0, "rvol_entry_threshold": 1.5, "stop_loss": 20, "tight_stop_loss": 10, "hazard_exit_threshold": 0.10, "grace_period": 6, "min_mcap": 100000, "require_multi_bar_confirmation": True, "dynamic_hazard": True},
    ]

    print(f"\n{'='*120}")
    print(f"  5-Minute Strategy Parameter Sweep (Train: {len(train_tokens)}, Test: {len(test_tokens)})")
    print(f"{'='*120}")
    print(f"  {'ROC':>4s} {'RVOL':>5s} {'SL':>3s} {'TSL':>4s} {'Hz':>5s} {'GP':>3s} {'MC':>7s} | "
          f"{'Trd':>4s} {'WR%':>5s} {'AvgR':>7s} {'PF':>5s} {'W/L':>5s} | "
          f"{'Trd':>4s} {'WR%':>5s} {'AvgR':>7s} {'PF':>5s} {'W/L':>5s} | "
          f"{'WRgp':>5s} {'PFgp':>5s}")

    results = []

    for cfg in configs:
        strategy = Strategy5m(**cfg)
        train_trades, _ = run_on_tokens(strategy, train_tokens, kmf=kmf)
        test_trades, _ = run_on_tokens(strategy, test_tokens, kmf=kmf)

        def qs(trades):
            if not trades:
                return {"n": 0, "wr": 0, "avg": 0, "pf": 0, "wl": 0}
            rets = [t["return_pct"] for t in trades]
            w = [r for r in rets if r > 0]
            l = [r for r in rets if r <= 0]
            gp = sum(w) if w else 0
            gl = abs(sum(l)) if l else 1
            aw = np.mean(w) if w else 0
            al = np.mean(l) if l else -1
            return {"n": len(trades), "wr": len(w)/len(trades)*100, "avg": np.mean(rets),
                    "pf": gp/gl, "wl": aw/abs(al) if al else 0}

        tr, te = qs(train_trades), qs(test_trades)
        wr_gap = abs(tr["wr"] - te["wr"])
        pf_gap = abs(tr["pf"] - te["pf"])

        print(f"  {cfg['roc_entry_threshold']:>4.1f} {cfg['rvol_entry_threshold']:>5.1f} {cfg['stop_loss']:>3.0f} "
              f"{cfg['tight_stop_loss']:>4.0f} {cfg['hazard_exit_threshold']:>5.2f} {cfg['grace_period']:>3d} "
              f"{cfg['min_mcap']:>7.0f} | "
              f"{tr['n']:>4d} {tr['wr']:>4.1f}% {tr['avg']:>+6.2f}% {tr['pf']:>5.2f} {tr['wl']:>5.2f} | "
              f"{te['n']:>4d} {te['wr']:>4.1f}% {te['avg']:>+6.2f}% {te['pf']:>5.2f} {te['wl']:>5.2f} | "
              f"{wr_gap:>4.1f}pp {pf_gap:>4.2f}")

        results.append({**cfg, "train_wr": tr["wr"], "train_pf": tr["pf"], "train_avg": tr["avg"],
                        "test_wr": te["wr"], "test_pf": te["pf"], "test_avg": te["avg"],
                        "wr_gap": wr_gap, "pf_gap": pf_gap})

    print(f"{'='*120}")
    pd.DataFrame(results).to_csv(os.path.join(DATA_DIR, "backtest_5m_sweep.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="5-Minute Combined Strategy Backtest")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "sweep"])
    parser.add_argument("--stop-loss", type=float, default=15.0)
    parser.add_argument("--tight-stop", type=float, default=8.0)
    parser.add_argument("--min-mcap", type=int, default=100000)
    parser.add_argument("--roc-threshold", type=float, default=3.0)
    parser.add_argument("--rvol-threshold", type=float, default=1.5)
    parser.add_argument("--hazard-threshold", type=float, default=0.15)
    parser.add_argument("--grace-period", type=int, default=6)
    args = parser.parse_args()

    if args.command == "sweep":
        cmd_sweep(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
