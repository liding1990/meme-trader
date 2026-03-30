"""HMM-based momentum strategy: regime detection for entry/exit.

Uses a 2-state Gaussian HMM trained on pooled cross-token data:
  State 0: "growth" regime — positive mcap momentum, holder inflows
  State 1: "decay" regime  — negative momentum, holder outflows

The HMM detects regimes earlier than rule-based lookback because it
uses the full joint distribution of features, not just sign/direction.

Key advantages over v2 strategy:
  - Earlier entry: HMM can detect regime shift from 1-2 observations
  - Probabilistic: gives P(growth) not just binary signal
  - Multivariate: jointly models mcap, holder, and concentration dynamics
  - Trained on pooled data: learns from ALL tokens, not just current one

Usage:
    python hmm_strategy.py train              # train HMM on all historical data
    python hmm_strategy.py backtest           # run backtest
    python hmm_strategy.py sweep              # parameter sweep
    python hmm_strategy.py analyze <address>  # analyze single token
"""

import argparse
import hashlib
import os
import sys
import pickle

import numpy as np
import pandas as pd
from hmmlearn import hmm

from space4d import extract_trajectory, load_token_list

DATA_DIR = "data"
PLOTS_DIR = "plots"


def compute_features(trajectory):
    """Compute stationary features from raw trajectory.

    Input: (T, 3) [mcap, holders, top10%]
    Output: (T-1, 3) [log_return, holder_growth, delta_top10]
    """
    if len(trajectory) < 3:
        return None

    mcap = np.clip(trajectory[:, 0], 1.0, None)
    holders = np.clip(trajectory[:, 1], 1.0, None)
    top10 = trajectory[:, 2]

    log_ret = np.diff(np.log(mcap))
    holder_growth = np.diff(holders) / holders[:-1]
    delta_top10 = np.diff(top10)

    features = np.column_stack([log_ret, holder_growth, delta_top10])

    # Clip extreme outliers
    for col in range(features.shape[1]):
        std = np.std(features[:, col])
        if std > 0:
            features[:, col] = np.clip(features[:, col], -5 * std, 5 * std)

    return features


def build_training_data(tokens, min_hours=10):
    """Pool features from all tokens into one training dataset.

    Returns (concatenated_features, lengths) for hmmlearn's fit().
    """
    all_features = []
    lengths = []

    for token in tokens:
        traj = extract_trajectory(token["address"])
        if traj is None or len(traj) < min_hours:
            continue
        features = compute_features(traj)
        if features is None or len(features) < 5:
            continue
        all_features.append(features)
        lengths.append(len(features))

    if not all_features:
        return None, None

    X = np.concatenate(all_features, axis=0)
    return X, lengths


def train_hmm(tokens, n_states=2, n_iter=100, min_hours=10):
    """Train a 2-state Gaussian HMM on pooled cross-token data."""
    print(f"Building training data from {len(tokens)} tokens...")
    X, lengths = build_training_data(tokens, min_hours=min_hours)

    if X is None:
        print("ERROR: No valid training data", file=sys.stderr)
        return None

    print(f"Training data: {X.shape[0]} observations from {len(lengths)} tokens")
    print(f"  Feature means: {X.mean(axis=0)}")
    print(f"  Feature stds:  {X.std(axis=0)}")

    # Train HMM
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=42,
        verbose=False,
    )

    print(f"Training {n_states}-state GaussianHMM (n_iter={n_iter})...")
    model.fit(X, lengths)

    # Identify which state is "growth" and which is "decay"
    # Growth state has higher mean log_return (feature 0)
    growth_state = np.argmax(model.means_[:, 0])
    decay_state = 1 - growth_state

    print(f"\nHMM trained successfully:")
    print(f"  Growth state ({growth_state}): mean log_return={model.means_[growth_state, 0]:+.4f}, "
          f"holder_growth={model.means_[growth_state, 1]:+.4f}, "
          f"delta_top10={model.means_[growth_state, 2]:+.4f}")
    print(f"  Decay state  ({decay_state}): mean log_return={model.means_[decay_state, 0]:+.4f}, "
          f"holder_growth={model.means_[decay_state, 1]:+.4f}, "
          f"delta_top10={model.means_[decay_state, 2]:+.4f}")
    print(f"  Transition matrix:")
    print(f"    P(stay growth) = {model.transmat_[growth_state, growth_state]:.4f}")
    print(f"    P(stay decay)  = {model.transmat_[decay_state, decay_state]:.4f}")
    print(f"    P(growth→decay) = {model.transmat_[growth_state, decay_state]:.4f}")
    print(f"    P(decay→growth) = {model.transmat_[decay_state, growth_state]:.4f}")

    return model, growth_state


def save_model(model, growth_state):
    os.makedirs("models", exist_ok=True)
    with open("models/hmm.pkl", "wb") as f:
        pickle.dump({"model": model, "growth_state": growth_state}, f)
    print("Model saved: models/hmm.pkl")


def load_model():
    with open("models/hmm.pkl", "rb") as f:
        d = pickle.load(f)
    return d["model"], d["growth_state"]


class HMMStrategy:
    """Trading strategy using HMM regime detection."""

    def __init__(self, model, growth_state,
                 entry_threshold=0.7,
                 exit_threshold=0.3,
                 stop_loss=15.0,
                 min_mcap=200_000,
                 min_age=3):
        """
        Args:
            model: trained GaussianHMM
            growth_state: which HMM state corresponds to growth
            entry_threshold: P(growth) must exceed this to enter
            exit_threshold: P(growth) must drop below this to exit
            stop_loss: trailing stop loss %
            min_mcap: minimum mcap to consider entry
            min_age: minimum hours from 50K before entry
        """
        self.model = model
        self.growth_state = growth_state
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.min_mcap = min_mcap
        self.min_age = min_age

    def generate_signals(self, trajectory):
        """Run HMM on a trajectory and generate entry/exit signals."""
        features = compute_features(trajectory)
        if features is None or len(features) < 3:
            return None

        # Get per-timestep state probabilities (online filtering via forward algo)
        log_prob, posteriors = self.model.score_samples(features)
        growth_prob = posteriors[:, self.growth_state]

        T = len(features)
        df = pd.DataFrame({
            "hour": np.arange(1, T + 1),
            "mcap": trajectory[1:T + 1, 0],
            "holders": trajectory[1:T + 1, 1],
            "top10_pct": trajectory[1:T + 1, 2],
            "growth_prob": growth_prob,
            "log_return": features[:, 0],
            "holder_growth": features[:, 1],
        })

        return df

    def simulate(self, trajectory):
        """Simulate trades on a single token."""
        df = self.generate_signals(trajectory)
        if df is None:
            return [], df

        trades = []
        in_trade = False
        entry_idx = None
        peak_mcap = 0

        for idx in range(len(df)):
            row = df.iloc[idx]

            if not in_trade:
                # Entry: P(growth) > threshold, mcap filter, age filter
                if (row["growth_prob"] > self.entry_threshold and
                        row["mcap"] >= self.min_mcap and
                        row["hour"] >= self.min_age):
                    in_trade = True
                    entry_idx = idx
                    peak_mcap = row["mcap"]
            else:
                current_mcap = row["mcap"]
                entry_mcap = df.iloc[entry_idx]["mcap"]

                # Track peak
                if current_mcap > peak_mcap:
                    peak_mcap = current_mcap

                # Exit conditions
                should_exit = False
                reason = None

                # 1. HMM says regime changed to decay
                if row["growth_prob"] < self.exit_threshold:
                    should_exit = True
                    reason = "regime_change"

                # 2. Trailing stop
                if peak_mcap > 0:
                    drawdown = (1 - current_mcap / peak_mcap) * 100
                    if drawdown > self.stop_loss:
                        should_exit = True
                        reason = "trailing_stop"

                # 3. Hard stop (2x stop_loss from entry)
                if entry_mcap > 0:
                    pnl = (current_mcap / entry_mcap - 1) * 100
                    if pnl < -self.stop_loss * 2:
                        should_exit = True
                        reason = "hard_stop"

                if should_exit:
                    ret = (current_mcap / entry_mcap - 1) * 100 if entry_mcap > 0 else 0
                    trades.append({
                        "entry_hour": df.iloc[entry_idx]["hour"],
                        "exit_hour": row["hour"],
                        "entry_mcap": entry_mcap,
                        "exit_mcap": current_mcap,
                        "return_pct": ret,
                        "hold_hours": row["hour"] - df.iloc[entry_idx]["hour"],
                        "exit_reason": reason,
                        "entry_growth_prob": df.iloc[entry_idx]["growth_prob"],
                        "exit_growth_prob": row["growth_prob"],
                    })
                    in_trade = False
                    entry_idx = None
                    peak_mcap = 0

        # Close open position
        if in_trade and entry_idx is not None:
            last = df.iloc[-1]
            entry_mcap = df.iloc[entry_idx]["mcap"]
            ret = (last["mcap"] / entry_mcap - 1) * 100 if entry_mcap > 0 else 0
            trades.append({
                "entry_hour": df.iloc[entry_idx]["hour"],
                "exit_hour": last["hour"],
                "entry_mcap": entry_mcap,
                "exit_mcap": last["mcap"],
                "return_pct": ret,
                "hold_hours": last["hour"] - df.iloc[entry_idx]["hour"],
                "exit_reason": "end_of_data",
                "entry_growth_prob": df.iloc[entry_idx]["growth_prob"],
                "exit_growth_prob": last["growth_prob"],
            })

        return trades, df


def split_tokens(tokens, train_ratio=0.6, seed=42):
    """Deterministic train/test split."""
    train, test = [], []
    for t in tokens:
        h = int(hashlib.md5(t["address"].encode()).hexdigest(), 16) % 100
        if h < train_ratio * 100:
            train.append(t)
        else:
            test.append(t)
    return train, test


def run_on_tokens(strategy, tokens):
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
    if not trades:
        print(f"  {label}: No trades")
        return {}

    rets = [t["return_pct"] for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    gp = sum(wins) if wins else 0
    gl = abs(sum(losses)) if losses else 1
    pf = gp / gl if gl > 0 else float("inf")

    reasons = {}
    for t in trades:
        r = t.get("exit_reason", "unknown")
        reasons[r] = reasons.get(r, 0) + 1

    stats = {
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100,
        "avg_return": np.mean(rets),
        "med_return": np.median(rets),
        "profit_factor": pf,
        "avg_hold": np.mean([t["hold_hours"] for t in trades]),
        "avg_win": np.mean(wins) if wins else 0,
        "avg_loss": np.mean(losses) if losses else 0,
    }

    print(f"\n  {label}")
    print(f"  {'─'*60}")
    print(f"  Trades:        {stats['trades']}")
    print(f"  Win rate:      {stats['win_rate']:.1f}%")
    print(f"  Avg return:    {stats['avg_return']:+.2f}%")
    print(f"  Med return:    {stats['med_return']:+.2f}%")
    print(f"  Profit factor: {stats['profit_factor']:.2f}")
    print(f"  Avg win:       {stats['avg_win']:+.1f}%  |  Avg loss: {stats['avg_loss']:+.1f}%")
    print(f"  W/L ratio:     {stats['avg_win']/abs(stats['avg_loss']):.2f}x" if stats['avg_loss'] else "  W/L ratio: N/A")
    print(f"  Avg hold:      {stats['avg_hold']:.1f}h")
    print(f"  Exit reasons:  {reasons}")

    return stats


def cmd_train(args):
    tokens = load_token_list()
    model, growth_state = train_hmm(tokens, n_states=2, n_iter=100)
    if model:
        save_model(model, growth_state)


def cmd_backtest(args):
    tokens = load_token_list()
    train_tokens, test_tokens = split_tokens(tokens)

    # Train HMM only on train set
    print("Training HMM on train set only...")
    model, growth_state = train_hmm(train_tokens, n_states=2, n_iter=100)
    if not model:
        return

    strategy = HMMStrategy(
        model, growth_state,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        stop_loss=args.stop_loss,
        min_mcap=args.min_mcap,
        min_age=args.min_age,
    )

    print(f"\n{'='*70}")
    print(f"  HMM Strategy Backtest")
    print(f"  Train: {len(train_tokens)} tokens, Test: {len(test_tokens)} tokens")
    print(f"  Entry P(growth) > {args.entry_threshold}, Exit P(growth) < {args.exit_threshold}")
    print(f"  Trailing stop: {args.stop_loss}%, Min mcap: ${args.min_mcap:,}, Min age: {args.min_age}h")
    print(f"{'='*70}")

    train_trades = run_on_tokens(strategy, train_tokens)
    test_trades = run_on_tokens(strategy, test_tokens)

    train_stats = print_stats(train_trades, "TRAIN SET")
    test_stats = print_stats(test_trades, "TEST SET")

    if train_stats and test_stats:
        wr_gap = abs(train_stats["win_rate"] - test_stats["win_rate"])
        pf_gap = abs(train_stats["profit_factor"] - test_stats["profit_factor"])
        print(f"\n  Overfitting check: WR gap={wr_gap:.1f}pp, PF gap={pf_gap:.2f}")

    print(f"{'='*70}")

    # Save trades
    if train_trades or test_trades:
        all_trades = train_trades + test_trades
        pd.DataFrame(all_trades).to_csv(os.path.join(DATA_DIR, "hmm_trades.csv"), index=False)


def cmd_sweep(args):
    tokens = load_token_list()
    train_tokens, test_tokens = split_tokens(tokens)

    # Train HMM once on train set
    print("Training HMM on train set...")
    model, growth_state = train_hmm(train_tokens, n_states=2, n_iter=100)
    if not model:
        return

    configs = [
        {"entry_threshold": 0.6, "exit_threshold": 0.4, "stop_loss": 15, "min_mcap": 200000, "min_age": 3},
        {"entry_threshold": 0.7, "exit_threshold": 0.3, "stop_loss": 15, "min_mcap": 200000, "min_age": 3},
        {"entry_threshold": 0.8, "exit_threshold": 0.3, "stop_loss": 15, "min_mcap": 200000, "min_age": 3},
        {"entry_threshold": 0.7, "exit_threshold": 0.4, "stop_loss": 15, "min_mcap": 200000, "min_age": 3},
        {"entry_threshold": 0.7, "exit_threshold": 0.3, "stop_loss": 10, "min_mcap": 200000, "min_age": 3},
        {"entry_threshold": 0.7, "exit_threshold": 0.3, "stop_loss": 20, "min_mcap": 200000, "min_age": 3},
        {"entry_threshold": 0.7, "exit_threshold": 0.3, "stop_loss": 15, "min_mcap": 100000, "min_age": 3},
        {"entry_threshold": 0.7, "exit_threshold": 0.3, "stop_loss": 15, "min_mcap": 500000, "min_age": 3},
        {"entry_threshold": 0.7, "exit_threshold": 0.3, "stop_loss": 15, "min_mcap": 200000, "min_age": 1},
        {"entry_threshold": 0.7, "exit_threshold": 0.3, "stop_loss": 15, "min_mcap": 200000, "min_age": 5},
        # Tighter thresholds
        {"entry_threshold": 0.8, "exit_threshold": 0.4, "stop_loss": 15, "min_mcap": 200000, "min_age": 3},
        {"entry_threshold": 0.9, "exit_threshold": 0.3, "stop_loss": 15, "min_mcap": 200000, "min_age": 3},
        # Very early entry
        {"entry_threshold": 0.6, "exit_threshold": 0.3, "stop_loss": 15, "min_mcap": 100000, "min_age": 1},
        {"entry_threshold": 0.7, "exit_threshold": 0.3, "stop_loss": 15, "min_mcap": 100000, "min_age": 1},
    ]

    print(f"\n{'='*110}")
    print(f"  HMM Strategy Parameter Sweep")
    print(f"  Train: {len(train_tokens)}, Test: {len(test_tokens)}")
    print(f"{'='*110}")
    print(f"  {'EntP':>5s} {'ExtP':>5s} {'SL%':>4s} {'MinMC':>7s} {'Age':>4s} | "
          f"{'Trd':>4s} {'WR%':>5s} {'AvgR':>7s} {'PF':>5s} | "
          f"{'Trd':>4s} {'WR%':>5s} {'AvgR':>7s} {'PF':>5s} | "
          f"{'WRgap':>6s} {'PFgap':>6s}")

    results = []

    for cfg in configs:
        strategy = HMMStrategy(model, growth_state, **cfg)
        train_trades = run_on_tokens(strategy, train_tokens)
        test_trades = run_on_tokens(strategy, test_tokens)

        def qs(trades):
            if not trades:
                return {"n": 0, "wr": 0, "avg": 0, "pf": 0}
            rets = [t["return_pct"] for t in trades]
            w = [r for r in rets if r > 0]
            l = [r for r in rets if r <= 0]
            gp = sum(w) if w else 0
            gl = abs(sum(l)) if l else 1
            return {"n": len(trades), "wr": len(w)/len(trades)*100, "avg": np.mean(rets), "pf": gp/gl}

        tr, te = qs(train_trades), qs(test_trades)
        wr_gap = abs(tr["wr"] - te["wr"])
        pf_gap = abs(tr["pf"] - te["pf"])

        print(f"  {cfg['entry_threshold']:>5.1f} {cfg['exit_threshold']:>5.1f} {cfg['stop_loss']:>4.0f} "
              f"{cfg['min_mcap']:>7.0f} {cfg['min_age']:>4d} | "
              f"{tr['n']:>4d} {tr['wr']:>4.1f}% {tr['avg']:>+6.2f}% {tr['pf']:>5.2f} | "
              f"{te['n']:>4d} {te['wr']:>4.1f}% {te['avg']:>+6.2f}% {te['pf']:>5.2f} | "
              f"{wr_gap:>5.1f}pp {pf_gap:>5.2f}")

        results.append({**cfg, "train_wr": tr["wr"], "train_pf": tr["pf"],
                        "test_wr": te["wr"], "test_pf": te["pf"],
                        "test_avg": te["avg"], "wr_gap": wr_gap, "pf_gap": pf_gap})

    print(f"{'='*110}")
    pd.DataFrame(results).to_csv(os.path.join(DATA_DIR, "hmm_sweep.csv"), index=False)


def cmd_analyze(args):
    model, growth_state = load_model()
    strategy = HMMStrategy(model, growth_state,
                           entry_threshold=args.entry_threshold,
                           exit_threshold=args.exit_threshold,
                           stop_loss=args.stop_loss,
                           min_mcap=args.min_mcap,
                           min_age=args.min_age)

    traj = extract_trajectory(args.address)
    if traj is None:
        from gmgn_api import fetch_token_data
        from dna_extractor import extract_candle_series, extract_trend_series, build_dna_dataframe
        print(f"Fetching {args.address[:8]}...")
        _, loaded_data = fetch_token_data("sol", args.address)
        candle_df = extract_candle_series(loaded_data.get("token_mcap_candles", {}))
        trend_df = extract_trend_series(loaded_data.get("token_trends", {}))
        dna_df = build_dna_dataframe(candle_df, trend_df)
        mcap = dna_df["mcap"].values
        above = np.where(mcap >= 50000)[0]
        if len(above) == 0:
            print("Never reached 50K"); return
        trimmed = dna_df.iloc[above[0]:].reset_index(drop=True)
        traj = trimmed[["mcap", "holder_count", "top10_pct"]].values

    df = strategy.generate_signals(traj)
    trades, _ = strategy.simulate(traj)

    print(f"\nLast 10 hours:")
    print(df[["hour", "mcap", "holders", "growth_prob"]].tail(10).to_string(index=False))
    print(f"\nCurrent P(growth): {df['growth_prob'].iloc[-1]:.3f}")
    print(f"Trades: {len(trades)}")
    for t in trades[-3:]:
        print(f"  h{t['entry_hour']}→h{t['exit_hour']}: {t['return_pct']:+.1f}% ({t['exit_reason']})")


def main():
    parser = argparse.ArgumentParser(description="HMM Momentum Strategy")
    parser.add_argument("--entry-threshold", type=float, default=0.7)
    parser.add_argument("--exit-threshold", type=float, default=0.3)
    parser.add_argument("--stop-loss", type=float, default=15.0)
    parser.add_argument("--min-mcap", type=int, default=200000)
    parser.add_argument("--min-age", type=int, default=3)

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("train")
    sub.add_parser("backtest")
    sub.add_parser("sweep")
    a = sub.add_parser("analyze")
    a.add_argument("address")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "sweep":
        cmd_sweep(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
