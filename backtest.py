"""Vectorized backtesting of the Kalman + CUSUM momentum strategy.

For each historical token:
1. Run momentum signal engine → entry/exit signals
2. Simulate trades: enter on first entry signal, exit on first exit signal after entry
3. Compute per-trade returns
4. Aggregate across all tokens: win rate, profit factor, avg return, etc.

Usage:
    python backtest.py [--h 1.0] [--d 0.0] [--min-hours 10]
    python backtest.py sweep  # parameter sweep over h values
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from space4d import extract_trajectory, load_token_list
from momentum import generate_signals

DATA_DIR = "data"
PLOTS_DIR = "plots"


def simulate_trades(signals_df):
    """Simulate trades from entry/exit signals.

    Returns list of trade dicts: {entry_hour, exit_hour, entry_mcap, exit_mcap, return_pct, hold_hours}
    """
    trades = []
    in_trade = False
    entry_hour = None
    entry_mcap = None

    for _, row in signals_df.iterrows():
        if not in_trade:
            # Look for entry: first hour where entry=True
            if row["entry"]:
                in_trade = True
                entry_hour = row["hour"]
                entry_mcap = row["mcap"]
        else:
            # Look for exit
            if row["exit"] or not row["entry"]:
                # Exit on exit signal or when entry condition no longer holds
                if row["exit"]:
                    exit_mcap = row["mcap"]
                    exit_hour = row["hour"]
                    ret = (exit_mcap / entry_mcap - 1) * 100 if entry_mcap > 0 else 0
                    trades.append({
                        "entry_hour": entry_hour,
                        "exit_hour": exit_hour,
                        "entry_mcap": entry_mcap,
                        "exit_mcap": exit_mcap,
                        "return_pct": ret,
                        "hold_hours": exit_hour - entry_hour,
                    })
                    in_trade = False
                    entry_hour = None
                    entry_mcap = None

    # If still in trade at end, close at last price
    if in_trade and entry_mcap is not None:
        last = signals_df.iloc[-1]
        ret = (last["mcap"] / entry_mcap - 1) * 100 if entry_mcap > 0 else 0
        trades.append({
            "entry_hour": entry_hour,
            "exit_hour": last["hour"],
            "entry_mcap": entry_mcap,
            "exit_mcap": last["mcap"],
            "return_pct": ret,
            "hold_hours": last["hour"] - entry_hour,
        })

    return trades


def run_backtest(h=1.0, d=0.0, min_hours=10):
    """Run backtest across all historical tokens."""
    tokens = load_token_list()

    all_trades = []
    token_results = []
    skipped = 0

    for token in tokens:
        address = token["address"]
        symbol = token["symbol"]

        traj = extract_trajectory(address)
        if traj is None or len(traj) < min_hours:
            skipped += 1
            continue

        signals_df = generate_signals(traj, h=h, d=d)
        if signals_df is None:
            skipped += 1
            continue

        trades = simulate_trades(signals_df)
        if not trades:
            token_results.append({
                "symbol": symbol,
                "address": address,
                "n_trades": 0,
                "total_return": 0,
                "win_rate": 0,
                "avg_return": 0,
                "avg_hold": 0,
            })
            continue

        returns = [t["return_pct"] for t in trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        for t in trades:
            t["symbol"] = symbol
            t["address"] = address
        all_trades.extend(trades)

        token_results.append({
            "symbol": symbol,
            "address": address,
            "n_trades": len(trades),
            "total_return": sum(returns),
            "win_rate": len(wins) / len(trades) * 100,
            "avg_return": np.mean(returns),
            "max_return": max(returns),
            "min_return": min(returns),
            "avg_hold": np.mean([t["hold_hours"] for t in trades]),
        })

    return all_trades, token_results, skipped


def print_results(all_trades, token_results, skipped, h, d):
    """Print backtest summary."""
    if not all_trades:
        print("No trades generated.")
        return

    returns = [t["return_pct"] for t in all_trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    tokens_with_trades = [r for r in token_results if r["n_trades"] > 0]

    print(f"\n{'='*70}")
    print(f"  Backtest Results (h={h}, d={d})")
    print(f"{'='*70}")
    print(f"  Tokens analyzed:  {len(token_results)} ({skipped} skipped)")
    print(f"  Tokens w/ trades: {len(tokens_with_trades)}")
    print(f"  Total trades:     {len(all_trades)}")
    print(f"{'─'*70}")
    print(f"  Win rate:         {len(wins)}/{len(all_trades)} = {len(wins)/len(all_trades)*100:.1f}%")
    print(f"  Avg return:       {np.mean(returns):+.2f}%")
    print(f"  Median return:    {np.median(returns):+.2f}%")
    print(f"  Avg hold time:    {np.mean([t['hold_hours'] for t in all_trades]):.1f}h")
    print(f"{'─'*70}")
    print(f"  Gross profit:     {gross_profit:+.1f}%")
    print(f"  Gross loss:       {-gross_loss:.1f}%")
    print(f"  Profit factor:    {profit_factor:.2f}")
    print(f"{'─'*70}")
    print(f"  Best trade:       {max(returns):+.1f}%")
    print(f"  Worst trade:      {min(returns):+.1f}%")
    print(f"  Std dev:          {np.std(returns):.1f}%")

    # Top profitable tokens
    top = sorted(tokens_with_trades, key=lambda x: x["total_return"], reverse=True)[:10]
    print(f"\n  Top 10 Tokens by Total Return:")
    for r in top:
        print(f"    {r['symbol']:12s}  {r['n_trades']} trades  total={r['total_return']:+.1f}%  "
              f"win={r['win_rate']:.0f}%  avg={r['avg_return']:+.1f}%  hold={r['avg_hold']:.0f}h")

    # Worst tokens
    bottom = sorted(tokens_with_trades, key=lambda x: x["total_return"])[:5]
    print(f"\n  Bottom 5 Tokens:")
    for r in bottom:
        print(f"    {r['symbol']:12s}  {r['n_trades']} trades  total={r['total_return']:+.1f}%  "
              f"win={r['win_rate']:.0f}%  avg={r['avg_return']:+.1f}%")

    print(f"{'='*70}")


def run_sweep():
    """Parameter sweep over CUSUM threshold h."""
    h_values = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]

    print(f"\n{'='*90}")
    print(f"  Parameter Sweep: CUSUM threshold h")
    print(f"{'='*90}")
    print(f"  {'h':>5s}  {'Trades':>7s}  {'WinRate':>8s}  {'AvgRet':>8s}  {'MedRet':>8s}  "
          f"{'PF':>6s}  {'AvgHold':>8s}  {'Best':>8s}  {'Worst':>8s}")
    print(f"  {'─'*85}")

    sweep_results = []

    for h in h_values:
        all_trades, token_results, skipped = run_backtest(h=h, d=0.0)
        if not all_trades:
            continue

        returns = [t["return_pct"] for t in all_trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]
        gp = sum(wins) if wins else 0
        gl = abs(sum(losses)) if losses else 0
        pf = gp / gl if gl > 0 else float("inf")

        row = {
            "h": h,
            "trades": len(all_trades),
            "win_rate": len(wins) / len(all_trades) * 100,
            "avg_return": np.mean(returns),
            "med_return": np.median(returns),
            "profit_factor": pf,
            "avg_hold": np.mean([t["hold_hours"] for t in all_trades]),
            "best": max(returns),
            "worst": min(returns),
        }
        sweep_results.append(row)

        print(f"  {h:>5.1f}  {row['trades']:>7d}  {row['win_rate']:>7.1f}%  "
              f"{row['avg_return']:>+7.2f}%  {row['med_return']:>+7.2f}%  "
              f"{row['profit_factor']:>6.2f}  {row['avg_hold']:>7.1f}h  "
              f"{row['best']:>+7.1f}%  {row['worst']:>+7.1f}%")

    print(f"{'='*90}")

    # Save sweep results
    if sweep_results:
        pd.DataFrame(sweep_results).to_csv(
            os.path.join(DATA_DIR, "backtest_sweep.csv"), index=False
        )
        print(f"Sweep results saved: data/backtest_sweep.csv")


def main():
    parser = argparse.ArgumentParser(description="Backtest Kalman + CUSUM momentum strategy")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "sweep"])
    parser.add_argument("--h", type=float, default=1.0, help="CUSUM threshold")
    parser.add_argument("--d", type=float, default=0.0, help="CUSUM drift")
    parser.add_argument("--min-hours", type=int, default=10, help="Min trajectory length")
    args = parser.parse_args()

    if args.command == "sweep":
        run_sweep()
    else:
        all_trades, token_results, skipped = run_backtest(h=args.h, d=args.d, min_hours=args.min_hours)
        print_results(all_trades, token_results, skipped, args.h, args.d)

        # Save detailed results
        if all_trades:
            pd.DataFrame(all_trades).to_csv(
                os.path.join(DATA_DIR, "backtest_trades.csv"), index=False
            )
            pd.DataFrame(token_results).to_csv(
                os.path.join(DATA_DIR, "backtest_tokens.csv"), index=False
            )
            print(f"\nDetailed results saved to data/backtest_trades.csv and data/backtest_tokens.csv")


if __name__ == "__main__":
    main()
