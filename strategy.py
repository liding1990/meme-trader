"""Momentum scalping strategy v2 — domain-informed entry/exit with train/test validation.

Key improvements over v1 (momentum.py baseline):
1. Entry requires STRONG momentum, not just positive trend
2. Exit uses RAW feature signals (faster than Kalman-smoothed CUSUM)
3. Filters based on domain knowledge from cluster analysis (C6/C7 patterns)
4. Train/test split to detect overfitting

Entry conditions (ALL must be true):
  - MCap log return (raw, not smoothed) positive for last 2 of 3 hours
  - Holder count is increasing (raw delta > 0 for last 2 of 3 hours)
  - Top10% is NOT rapidly declining (no large outflow by whales)
  - Token has been alive >= min_age hours (filter early noise)
  - MCap is above a minimum threshold (filter dead tokens)

Exit conditions (ANY triggers exit):
  - Holder count decreasing for 2 consecutive hours (earliest momentum death signal)
  - MCap drops more than stop_loss% from entry price
  - Held for max_hold hours (time stop)

Usage:
    python strategy.py run [--stop-loss 15 --max-hold 12 --min-age 3]
    python strategy.py validate  # train/test split validation
    python strategy.py sweep     # parameter sweep
"""

import argparse
import os
import sys
import hashlib

import numpy as np
import pandas as pd

from space4d import extract_trajectory, load_token_list

DATA_DIR = "data"


class MomentumStrategy:
    """Domain-informed momentum scalping strategy."""

    def __init__(self, stop_loss=15.0, max_hold=12, min_age=6,
                 entry_lookback=3, entry_min_positive=2,
                 exit_holder_decline_hours=2,
                 top10_max_decline=3.0,
                 min_mcap=200_000,
                 require_acceleration=True):
        """
        Args:
            stop_loss: max loss % before forced exit
            max_hold: max hours to hold a position
            min_age: minimum token age (hours from 50K) before considering entry
            entry_lookback: how many hours to look back for entry conditions
            entry_min_positive: how many of those hours must show positive momentum
            exit_holder_decline_hours: consecutive hours of holder decline to trigger exit
            top10_max_decline: max hourly top10% decline allowed at entry (whale dump filter)
            min_mcap: minimum mcap at entry (filter low-cap noise)
            require_acceleration: require mcap growth to be accelerating (2nd derivative > 0)
        """
        self.stop_loss = stop_loss
        self.max_hold = max_hold
        self.min_age = min_age
        self.entry_lookback = entry_lookback
        self.entry_min_positive = entry_min_positive
        self.exit_holder_decline_hours = exit_holder_decline_hours
        self.top10_max_decline = top10_max_decline
        self.min_mcap = min_mcap
        self.require_acceleration = require_acceleration

    def compute_raw_features(self, trajectory):
        """Compute raw (unsmoothed) features from trajectory.

        Input: (T, 3) array [mcap, holders, top10%]
        Output: DataFrame with raw momentum features
        """
        T = len(trajectory)
        if T < self.min_age + 2:
            return None

        mcap = trajectory[:, 0]
        holders = trajectory[:, 1]
        top10 = trajectory[:, 2]

        mcap_safe = np.clip(mcap, 1.0, None)
        holders_safe = np.clip(holders, 1.0, None)

        log_return = np.diff(np.log(mcap_safe))
        holder_delta = np.diff(holders_safe)
        holder_growth = holder_delta / holders_safe[:-1]
        top10_delta = np.diff(top10)

        df = pd.DataFrame({
            "hour": np.arange(1, T),
            "mcap": mcap[1:],
            "holders": holders[1:],
            "top10_pct": top10[1:],
            "log_return": log_return,
            "holder_delta": holder_delta,
            "holder_growth": holder_growth,
            "top10_delta": top10_delta,
        })

        return df

    def check_entry(self, df, idx):
        """Check if entry conditions are met at index idx."""
        if idx < max(self.entry_lookback + 1, 4):
            return False

        row = df.iloc[idx]
        hour = row["hour"]

        if hour < self.min_age:
            return False

        # Minimum mcap filter
        if row["mcap"] < self.min_mcap:
            return False

        # Look back entry_lookback hours
        window = df.iloc[idx - self.entry_lookback + 1: idx + 1]

        # MCap must be positive for at least entry_min_positive of lookback hours
        mcap_positive = (window["log_return"] > 0).sum()
        if mcap_positive < self.entry_min_positive:
            return False

        # Holders must be increasing for at least entry_min_positive of lookback hours
        holder_positive = (window["holder_delta"] > 0).sum()
        if holder_positive < self.entry_min_positive:
            return False

        # Top10% must NOT be rapidly declining (whale dump filter)
        top10_decline = window["top10_delta"].min()
        if top10_decline < -self.top10_max_decline:
            return False

        # Acceleration check: recent mcap returns > earlier mcap returns
        if self.require_acceleration and len(window) >= 3:
            recent_avg = window["log_return"].iloc[-2:].mean()
            earlier_avg = window["log_return"].iloc[:-2].mean() if len(window) > 2 else 0
            if recent_avg <= earlier_avg:
                return False

        return True

    def check_exit(self, df, idx, entry_idx, peak_mcap):
        """Check if exit conditions are met at index idx.

        Uses trailing stop: once price reaches a new high, the stop loss
        ratchets up to protect profits. This lets winners run while cutting
        losers quickly.

        Returns (should_exit, reason, updated_peak_mcap)
        """
        entry_mcap = df.iloc[entry_idx]["mcap"]
        current_mcap = df.iloc[idx]["mcap"]
        hours_held = df.iloc[idx]["hour"] - df.iloc[entry_idx]["hour"]

        # Track peak mcap since entry
        if current_mcap > peak_mcap:
            peak_mcap = current_mcap

        # Trailing stop: exit if dropped stop_loss% from peak (not from entry)
        if peak_mcap > 0:
            drawdown_from_peak = (1 - current_mcap / peak_mcap) * 100
            if drawdown_from_peak > self.stop_loss:
                return True, "trailing_stop", peak_mcap

        # Hard stop: never lose more than 2x stop_loss from entry
        if entry_mcap > 0:
            pnl_pct = (current_mcap / entry_mcap - 1) * 100
            if pnl_pct < -self.stop_loss * 2:
                return True, "hard_stop", peak_mcap

        # Time stop only if NOT profitable (let winners run)
        if hours_held >= self.max_hold:
            if entry_mcap > 0 and current_mcap <= entry_mcap:
                return True, "time_stop", peak_mcap

        # Holder decline for N consecutive hours
        if idx >= self.exit_holder_decline_hours:
            recent = df.iloc[idx - self.exit_holder_decline_hours + 1: idx + 1]
            if (recent["holder_delta"] < 0).all():
                return True, "holder_decline", peak_mcap

        return False, None, peak_mcap

    def simulate(self, trajectory):
        """Run strategy on a single token's trajectory. Returns list of trades."""
        df = self.compute_raw_features(trajectory)
        if df is None or len(df) < self.min_age + self.entry_lookback:
            return [], df

        trades = []
        in_trade = False
        entry_idx = None
        peak_mcap = 0

        for idx in range(len(df)):
            if not in_trade:
                if self.check_entry(df, idx):
                    in_trade = True
                    entry_idx = idx
                    peak_mcap = df.iloc[idx]["mcap"]
            else:
                should_exit, reason, peak_mcap = self.check_exit(df, idx, entry_idx, peak_mcap)
                if should_exit:
                    entry_row = df.iloc[entry_idx]
                    exit_row = df.iloc[idx]
                    ret = (exit_row["mcap"] / entry_row["mcap"] - 1) * 100 if entry_row["mcap"] > 0 else 0

                    trades.append({
                        "entry_hour": entry_row["hour"],
                        "exit_hour": exit_row["hour"],
                        "entry_mcap": entry_row["mcap"],
                        "exit_mcap": exit_row["mcap"],
                        "return_pct": ret,
                        "hold_hours": exit_row["hour"] - entry_row["hour"],
                        "exit_reason": reason,
                    })
                    in_trade = False
                    entry_idx = None

        # Close open position at end
        if in_trade and entry_idx is not None:
            entry_row = df.iloc[entry_idx]
            last_row = df.iloc[-1]
            ret = (last_row["mcap"] / entry_row["mcap"] - 1) * 100 if entry_row["mcap"] > 0 else 0
            trades.append({
                "entry_hour": entry_row["hour"],
                "exit_hour": last_row["hour"],
                "entry_mcap": entry_row["mcap"],
                "exit_mcap": last_row["mcap"],
                "return_pct": ret,
                "hold_hours": last_row["hour"] - entry_row["hour"],
                "exit_reason": "end_of_data",
            })

        return trades, df


def split_tokens(tokens, train_ratio=0.6, seed=42):
    """Deterministic train/test split based on address hash."""
    train, test = [], []
    for t in tokens:
        h = int(hashlib.md5(t["address"].encode()).hexdigest(), 16) % 100
        if h < train_ratio * 100:
            train.append(t)
        else:
            test.append(t)
    return train, test


def run_on_tokens(strategy, tokens):
    """Run strategy on a list of tokens, return aggregated results."""
    all_trades = []
    for token in tokens:
        traj = extract_trajectory(token["address"])
        if traj is None or len(traj) < 10:
            continue
        trades, _ = strategy.simulate(traj)
        for t in trades:
            t["symbol"] = token["symbol"]
            t["address"] = token["address"]
        all_trades.extend(trades)
    return all_trades


def print_stats(trades, label=""):
    """Print strategy statistics."""
    if not trades:
        print(f"  {label}: No trades")
        return {}

    returns = [t["return_pct"] for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    gp = sum(wins) if wins else 0
    gl = abs(sum(losses)) if losses else 1
    pf = gp / gl if gl > 0 else float("inf")

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        r = t.get("exit_reason", "unknown")
        reasons[r] = reasons.get(r, 0) + 1

    stats = {
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100,
        "avg_return": np.mean(returns),
        "med_return": np.median(returns),
        "profit_factor": pf,
        "avg_hold": np.mean([t["hold_hours"] for t in trades]),
        "total_return": sum(returns),
        "sharpe": np.mean(returns) / np.std(returns) * np.sqrt(len(returns)) if np.std(returns) > 0 else 0,
    }

    print(f"\n  {label}")
    print(f"  {'─'*60}")
    print(f"  Trades:       {stats['trades']}")
    print(f"  Win rate:     {stats['win_rate']:.1f}%")
    print(f"  Avg return:   {stats['avg_return']:+.2f}%")
    print(f"  Med return:   {stats['med_return']:+.2f}%")
    print(f"  Profit factor: {stats['profit_factor']:.2f}")
    print(f"  Avg hold:     {stats['avg_hold']:.1f}h")
    print(f"  Total return: {stats['total_return']:+.1f}%")
    print(f"  Exit reasons: {reasons}")

    return stats


def cmd_run(args):
    """Run strategy on all tokens."""
    strategy = MomentumStrategy(
        stop_loss=args.stop_loss,
        max_hold=args.max_hold,
        min_age=args.min_age,
    )
    tokens = load_token_list()
    trades = run_on_tokens(strategy, tokens)

    print(f"\n{'='*70}")
    print(f"  Strategy v2 Results")
    print(f"{'='*70}")
    stats = print_stats(trades, "All Tokens")
    print(f"{'='*70}")

    if trades:
        pd.DataFrame(trades).to_csv(os.path.join(DATA_DIR, "strategy_trades.csv"), index=False)


def cmd_validate(args):
    """Train/test split validation to check for overfitting."""
    tokens = load_token_list()
    train_tokens, test_tokens = split_tokens(tokens)

    print(f"\n{'='*70}")
    print(f"  Train/Test Validation (60/40 split)")
    print(f"  Train: {len(train_tokens)} tokens, Test: {len(test_tokens)} tokens")
    print(f"{'='*70}")

    strategy = MomentumStrategy(
        stop_loss=args.stop_loss,
        max_hold=args.max_hold,
        min_age=args.min_age,
    )

    train_trades = run_on_tokens(strategy, train_tokens)
    test_trades = run_on_tokens(strategy, test_tokens)

    train_stats = print_stats(train_trades, "TRAIN SET")
    test_stats = print_stats(test_trades, "TEST SET")

    # Overfitting check
    if train_stats and test_stats:
        wr_diff = abs(train_stats["win_rate"] - test_stats["win_rate"])
        pf_diff = abs(train_stats["profit_factor"] - test_stats["profit_factor"])
        print(f"\n  Overfitting Check:")
        print(f"    Win rate gap:     {wr_diff:.1f}pp {'OK' if wr_diff < 10 else 'WARNING'}")
        print(f"    Profit factor gap: {pf_diff:.2f} {'OK' if pf_diff < 0.3 else 'WARNING'}")

    print(f"{'='*70}")


def cmd_sweep(args):
    """Parameter sweep with train/test validation."""
    tokens = load_token_list()
    train_tokens, test_tokens = split_tokens(tokens)

    configs = [
        # Vary stop_loss with good defaults
        {"stop_loss": 10, "max_hold": 12, "min_age": 6, "min_mcap": 200_000},
        {"stop_loss": 15, "max_hold": 12, "min_age": 6, "min_mcap": 200_000},
        {"stop_loss": 20, "max_hold": 12, "min_age": 6, "min_mcap": 200_000},
        {"stop_loss": 25, "max_hold": 12, "min_age": 6, "min_mcap": 200_000},
        # Vary min_mcap
        {"stop_loss": 15, "max_hold": 12, "min_age": 6, "min_mcap": 100_000},
        {"stop_loss": 15, "max_hold": 12, "min_age": 6, "min_mcap": 500_000},
        {"stop_loss": 15, "max_hold": 12, "min_age": 6, "min_mcap": 1_000_000},
        # Vary min_age
        {"stop_loss": 15, "max_hold": 12, "min_age": 3, "min_mcap": 200_000},
        {"stop_loss": 15, "max_hold": 12, "min_age": 8, "min_mcap": 200_000},
        {"stop_loss": 15, "max_hold": 12, "min_age": 10, "min_mcap": 200_000},
        # Vary max_hold
        {"stop_loss": 15, "max_hold": 6, "min_age": 6, "min_mcap": 200_000},
        {"stop_loss": 15, "max_hold": 24, "min_age": 6, "min_mcap": 200_000},
        # No acceleration requirement
        {"stop_loss": 15, "max_hold": 12, "min_age": 6, "min_mcap": 200_000, "require_acceleration": False},
    ]

    print(f"\n{'='*110}")
    print(f"  Parameter Sweep with Train/Test Validation")
    print(f"  Train: {len(train_tokens)} tokens, Test: {len(test_tokens)} tokens")
    print(f"{'='*110}")
    print(f"  {'SL%':>4s} {'MaxH':>5s} {'MinA':>5s} | "
          f"{'Trades':>7s} {'WR%':>6s} {'AvgR':>7s} {'PF':>5s} | "
          f"{'Trades':>7s} {'WR%':>6s} {'AvgR':>7s} {'PF':>5s} | "
          f"{'WR_gap':>7s} {'PF_gap':>7s}")
    header = f"  {'':>16s} | {'───── TRAIN ─────':>28s} | {'────── TEST ──────':>28s} | {'─ Overfit ─':>15s}"
    print(header)

    results = []

    for cfg in configs:
        strategy = MomentumStrategy(**cfg)
        train_trades = run_on_tokens(strategy, train_tokens)
        test_trades = run_on_tokens(strategy, test_tokens)

        def quick_stats(trades):
            if not trades:
                return {"trades": 0, "wr": 0, "avg": 0, "pf": 0}
            rets = [t["return_pct"] for t in trades]
            wins = [r for r in rets if r > 0]
            losses = [r for r in rets if r <= 0]
            gp = sum(wins) if wins else 0
            gl = abs(sum(losses)) if losses else 1
            return {
                "trades": len(trades),
                "wr": len(wins) / len(trades) * 100,
                "avg": np.mean(rets),
                "pf": gp / gl if gl > 0 else 0,
            }

        tr = quick_stats(train_trades)
        te = quick_stats(test_trades)
        wr_gap = abs(tr["wr"] - te["wr"])
        pf_gap = abs(tr["pf"] - te["pf"])

        print(f"  {cfg['stop_loss']:>4d} {cfg['max_hold']:>5d} {cfg['min_age']:>5d} | "
              f"{tr['trades']:>7d} {tr['wr']:>5.1f}% {tr['avg']:>+6.2f}% {tr['pf']:>5.2f} | "
              f"{te['trades']:>7d} {te['wr']:>5.1f}% {te['avg']:>+6.2f}% {te['pf']:>5.2f} | "
              f"{wr_gap:>6.1f}pp {pf_gap:>6.2f}")

        results.append({**cfg, "train_wr": tr["wr"], "train_pf": tr["pf"], "train_avg": tr["avg"],
                        "test_wr": te["wr"], "test_pf": te["pf"], "test_avg": te["avg"],
                        "wr_gap": wr_gap, "pf_gap": pf_gap})

    print(f"{'='*110}")

    pd.DataFrame(results).to_csv(os.path.join(DATA_DIR, "strategy_sweep.csv"), index=False)
    print(f"Sweep results saved: data/strategy_sweep.csv")


def main():
    parser = argparse.ArgumentParser(description="Momentum Scalping Strategy v2")
    parser.add_argument("--stop-loss", type=float, default=15.0)
    parser.add_argument("--max-hold", type=int, default=12)
    parser.add_argument("--min-age", type=int, default=3)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run")
    sub.add_parser("validate")
    sub.add_parser("sweep")
    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "sweep":
        cmd_sweep(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
