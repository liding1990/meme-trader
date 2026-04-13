"""Scalper V3 Backtest — validates strategy on historical 5m data with state classifier."""

import csv
import hashlib
import json
import os
import glob
import sys

import numpy as np
import pandas as pd

from scalper import config
from scalper.state_classifier import (
    StateClassifier, build_point_cloud, build_point_cloud_legacy,
    train_classifier, save_artifacts, _load_5m_with_holders,
)

DATA_DIR = "data"
RADAR_CSV = os.path.join(DATA_DIR, "radar_tokens.csv")
MCAP_THRESHOLD = 50_000


def load_radar_tokens():
    tokens = []
    if not os.path.isfile(RADAR_CSV):
        return tokens
    with open(RADAR_CSV) as f:
        for row in csv.reader(f):
            if len(row) >= 4:
                tokens.append({"address": row[0], "chain": row[1], "name": row[2], "symbol": row[3]})
    return tokens


def split_tokens(tokens, train_ratio=config.TRAIN_RATIO, seed=42):
    train, test = [], []
    for t in tokens:
        h = int(hashlib.md5(t["address"].encode()).hexdigest(), 16) % 100
        if h < train_ratio * 100:
            train.append(t)
        else:
            test.append(t)
    return train, test


def load_5m_candles(address: str) -> pd.DataFrame | None:
    """Load 5m mcap candles for a token."""
    data_dir = os.path.join(DATA_DIR, address)
    if not os.path.isdir(data_dir):
        return None
    files = sorted(glob.glob(os.path.join(data_dir, "token_mcap_candles_5m_*.json")))
    if not files:
        return None
    with open(files[-1]) as f:
        data = json.load(f)
    data_section = data.get("data")
    if not data_section:
        return None
    candles = data_section.get("list", [])
    if not candles:
        return None

    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["time"].astype(int), unit="ms")
    df["mcap"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df = df[["datetime", "mcap", "volume"]].sort_values("datetime").reset_index(drop=True)
    return df


def load_hourly_trajectory(address: str) -> pd.DataFrame | None:
    """Load hourly trajectory with mcap + holders for state classification."""
    data_dir = os.path.join(DATA_DIR, address)
    if not os.path.isdir(data_dir):
        return None

    mcap_files = sorted([
        f for f in glob.glob(os.path.join(data_dir, "token_mcap_candles_[0-9]*.json"))
        if "5m" not in os.path.basename(f)
    ])
    trend_files = sorted(glob.glob(os.path.join(data_dir, "token_trends_*.json")))
    if not mcap_files or not trend_files:
        return None

    with open(mcap_files[-1]) as f:
        mcap_data = json.load(f)
    with open(trend_files[-1]) as f:
        trend_data = json.load(f)

    if not mcap_data or not trend_data:
        return None
    data_section = mcap_data.get("data")
    trend_section = trend_data.get("data")
    if not data_section or not trend_section:
        return None

    candles = data_section.get("list", [])
    holder_series = trend_section.get("trends", {}).get("holder_count", [])
    if not candles or not holder_series:
        return None

    mcap_df = pd.DataFrame(candles)
    mcap_df["datetime"] = pd.to_datetime(mcap_df["time"].astype(int), unit="ms")
    mcap_df["mcap"] = mcap_df["close"].astype(float)
    mcap_df["datetime"] = mcap_df["datetime"].dt.floor("h")
    mcap_df = mcap_df.groupby("datetime", as_index=False)["mcap"].last()

    holder_df = pd.DataFrame(holder_series)
    holder_df["datetime"] = pd.to_datetime(holder_df["timestamp"].astype(int), unit="s")
    holder_df["holders"] = holder_df["value"].astype(float)
    holder_df = holder_df.set_index("datetime").resample("1h").last().dropna().reset_index()
    holder_df["datetime"] = holder_df["datetime"].dt.floor("h")

    merged = pd.merge(mcap_df, holder_df, on="datetime", how="inner").sort_values("datetime")
    if merged.empty:
        return None

    above = merged[merged["mcap"] >= MCAP_THRESHOLD]
    if above.empty:
        return None

    start_idx = above.index[0]
    merged = merged.loc[start_idx:].reset_index(drop=True)
    t0 = merged["datetime"].iloc[0]
    merged["hours"] = (merged["datetime"] - t0).dt.total_seconds() / 3600

    return merged


def simulate_token(classifier: StateClassifier, address: str, symbol: str,
                   take_profit=config.TAKE_PROFIT_PCT,
                   hard_stop=config.HARD_STOP_PCT,
                   time_stop_bars=config.TIME_STOP_BARS,
                   roc_threshold=config.ROC_THRESHOLD,
                   slippage_pct=config.BACKTEST_SLIPPAGE_PCT):
    """Simulate the scalper strategy on one token using 5m mcap + Moralis holders."""
    df = _load_5m_with_holders(address)
    if df is None or len(df) < 60:
        return []

    mcap = df["mcap"].values
    hours = df["hours"].values
    holders = df["holders"].values
    volume = df["volume"].values if "volume" in df.columns else np.zeros(len(df))
    n = len(df)

    # Compute ROC and RVOL on 5m data
    roc_period = 6   # 30 min
    rvol_window = 48  # 4 hours

    roc_30m = np.full(n, np.nan)
    for i in range(roc_period, n):
        if mcap[i - roc_period] > 0:
            roc_30m[i] = (mcap[i] / mcap[i - roc_period] - 1) * 100

    # RVOL: use USD volume if available, else mcap volatility proxy
    rvol = np.full(n, np.nan)
    use_volume = volume.sum() > 0
    if use_volume:
        for i in range(rvol_window, n):
            avg = volume[i - rvol_window:i].mean()
            if avg > 0:
                rvol[i] = volume[i] / avg
    else:
        mcap_abs = np.abs(np.diff(mcap, prepend=mcap[0]))
        for i in range(rvol_window, n):
            avg = mcap_abs[i - rvol_window:i].mean()
            if avg > 0:
                rvol[i] = mcap_abs[i] / avg

    # Compute holder growth rate and whale flow (rolling 6-bar)
    net_change = df["net_holder_change"].values if "net_holder_change" in df.columns else np.zeros(n)
    whale_flow = df["whale_shark_flow"].values if "whale_shark_flow" in df.columns else np.zeros(n)

    holder_growth_arr = np.zeros(n)
    whale_flow_30m_arr = np.zeros(n)
    for i in range(6, n):
        if holders[i] > 0:
            holder_growth_arr[i] = net_change[max(0, i - 5):i + 1].sum() / holders[i] * 100
        whale_flow_30m_arr[i] = whale_flow[max(0, i - 5):i + 1].sum()

    # Classify state every 6 bars (30 min), carry forward
    state_favorable = np.zeros(n, dtype=bool)
    for i in range(roc_period, n, 6):
        if mcap[i] <= 0 or holders[i] <= 0:
            continue
        cid, is_fav, conf = classifier.classify(
            np.log1p(mcap[i]), hours[i], np.log1p(holders[i]),
            holder_growth_rate=holder_growth_arr[i],
            whale_shark_flow=whale_flow_30m_arr[i],
            mcap_roc_30m=roc_30m[i],
        )
        end = min(i + 6, n)
        state_favorable[i:end] = is_fav

    # Simulate
    trades = []
    in_trade = False
    entry_idx = None
    entry_price = 0
    peak_price = 0

    for i in range(rvol_window, n):
        price = mcap[i]
        if price <= 0:
            continue

        if not in_trade:
            if (state_favorable[i]
                    and not np.isnan(roc_30m[i]) and roc_30m[i] > roc_threshold
                    and not np.isnan(rvol[i]) and rvol[i] > config.RVOL_THRESHOLD):
                in_trade = True
                entry_idx = i
                entry_price = price * (1 + slippage_pct / 100)
                peak_price = price
        else:
            if price > peak_price:
                peak_price = price

            bars_held = i - entry_idx
            pnl_pct = (price / entry_price - 1) * 100
            drawdown = (1 - price / peak_price) * 100 if peak_price > 0 else 0

            exit_reason = None

            if pnl_pct >= take_profit:
                exit_reason = "take_profit"
            elif pnl_pct <= -hard_stop:
                exit_reason = "hard_stop"
            elif (pnl_pct >= config.TRAILING_ACTIVATE_PCT
                  and drawdown >= config.TRAILING_STOP_PCT):
                exit_reason = "trailing_stop"
            elif bars_held >= time_stop_bars:
                exit_reason = "time_stop"
            elif not np.isnan(roc_30m[i]) and roc_30m[i] < config.MOMENTUM_REVERSAL_ROC:
                exit_reason = "momentum_reversal"
            elif not state_favorable[i] and bars_held > 2:
                exit_reason = "state_exit"

            if exit_reason:
                exit_price = price * (1 - slippage_pct / 100)
                ret = (exit_price / entry_price - 1) * 100
                trades.append({
                    "symbol": symbol,
                    "address": address,
                    "entry_bar": entry_idx,
                    "exit_bar": i,
                    "bars_held": bars_held,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return_pct": ret,
                    "exit_reason": exit_reason,
                })
                in_trade = False
                entry_idx = None

    return trades


def run_backtest(classifier, tokens, **kwargs):
    """Run backtest on a set of tokens."""
    all_trades = []
    for t in tokens:
        trades = simulate_token(classifier, t["address"], t["symbol"], **kwargs)
        all_trades.extend(trades)
    return all_trades


def print_stats(trades, label=""):
    if not trades:
        print(f"  {label}: No trades")
        return {}

    returns = [t["return_pct"] for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    gp = sum(wins) if wins else 0
    gl = abs(sum(losses)) if losses else 1
    pf = gp / gl if gl > 0 else float("inf")

    reasons = {}
    for t in trades:
        r = t["exit_reason"]
        reasons[r] = reasons.get(r, 0) + 1

    n_tokens = len(set(t["address"] for t in trades))
    avg_bars = np.mean([t["bars_held"] for t in trades])

    stats = {
        "trades": len(trades),
        "tokens": n_tokens,
        "win_rate": len(wins) / len(trades) * 100,
        "avg_return": np.mean(returns),
        "med_return": np.median(returns),
        "profit_factor": pf,
        "avg_bars": avg_bars,
        "total_return": sum(returns),
    }

    print(f"\n  {label}")
    print(f"  {'─' * 60}")
    print(f"  Trades:        {stats['trades']} across {n_tokens} tokens")
    print(f"  Win rate:      {stats['win_rate']:.1f}%")
    print(f"  Avg return:    {stats['avg_return']:+.2f}%")
    print(f"  Med return:    {stats['med_return']:+.2f}%")
    print(f"  Profit factor: {stats['profit_factor']:.2f}")
    print(f"  Avg hold:      {stats['avg_bars']:.0f} bars ({stats['avg_bars']*5:.0f} min)")
    print(f"  Total return:  {stats['total_return']:+.1f}%")
    print(f"  Exit reasons:  {reasons}")

    return stats


def cmd_backtest():
    """Run backtest with train/test split."""
    print("Loading classifier...")
    classifier = StateClassifier.load()

    tokens = load_radar_tokens()
    train_tokens, test_tokens = split_tokens(tokens)
    print(f"Train: {len(train_tokens)} tokens, Test: {len(test_tokens)} tokens")

    print("\nRunning train set...")
    train_trades = run_backtest(classifier, train_tokens)
    train_stats = print_stats(train_trades, "TRAIN SET")

    print("\nRunning test set...")
    test_trades = run_backtest(classifier, test_tokens)
    test_stats = print_stats(test_trades, "TEST SET")

    if train_stats and test_stats:
        wr_gap = abs(train_stats["win_rate"] - test_stats["win_rate"])
        pf_gap = abs(train_stats["profit_factor"] - test_stats["profit_factor"])
        print(f"\n  Overfitting Check:")
        print(f"    Win rate gap:      {wr_gap:.1f}pp {'OK' if wr_gap < 10 else 'WARNING'}")
        print(f"    Profit factor gap: {pf_gap:.2f} {'OK' if pf_gap < 0.3 else 'WARNING'}")


def cmd_sweep():
    """Parameter sweep with train/test validation."""
    print("Loading classifier...")
    classifier = StateClassifier.load()

    tokens = load_radar_tokens()
    train_tokens, test_tokens = split_tokens(tokens)
    print(f"Train: {len(train_tokens)}, Test: {len(test_tokens)}\n")

    configs = []
    for tp in [2, 3, 4, 5]:
        for hs in [3, 5, 7]:
            for ts in [6, 9, 12, 18]:
                for roc in [1.0, 1.5, 2.0]:
                    configs.append({
                        "take_profit": tp, "hard_stop": hs,
                        "time_stop_bars": ts, "roc_threshold": roc,
                    })

    print(f"{'TP%':>4} {'HS%':>4} {'TS':>3} {'ROC':>4} | "
          f"{'#Tr':>4} {'WR%':>5} {'AvgR':>6} {'PF':>5} | "
          f"{'#Te':>4} {'WR%':>5} {'AvgR':>6} {'PF':>5} | "
          f"{'WRg':>4} {'PFg':>5}")
    print("─" * 85)

    results = []
    for cfg in configs:
        train_trades = run_backtest(classifier, train_tokens, **cfg)
        test_trades = run_backtest(classifier, test_tokens, **cfg)

        def qs(trades):
            if not trades:
                return {"n": 0, "wr": 0, "avg": 0, "pf": 0}
            rets = [t["return_pct"] for t in trades]
            w = [r for r in rets if r > 0]
            l = [r for r in rets if r <= 0]
            gp = sum(w) if w else 0
            gl = abs(sum(l)) if l else 1
            return {"n": len(trades), "wr": len(w)/len(trades)*100, "avg": np.mean(rets), "pf": gp/gl if gl > 0 else 0}

        tr, te = qs(train_trades), qs(test_trades)
        wr_gap = abs(tr["wr"] - te["wr"])
        pf_gap = abs(tr["pf"] - te["pf"])

        print(f"{cfg['take_profit']:>4} {cfg['hard_stop']:>4} {cfg['time_stop_bars']:>3} {cfg['roc_threshold']:>4.1f} | "
              f"{tr['n']:>4} {tr['wr']:>4.1f}% {tr['avg']:>+5.2f}% {tr['pf']:>5.2f} | "
              f"{te['n']:>4} {te['wr']:>4.1f}% {te['avg']:>+5.2f}% {te['pf']:>5.2f} | "
              f"{wr_gap:>3.1f} {pf_gap:>5.2f}")

        results.append({**cfg, **{f"train_{k}": v for k, v in tr.items()}, **{f"test_{k}": v for k, v in te.items()},
                        "wr_gap": wr_gap, "pf_gap": pf_gap})

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(DATA_DIR, "scalper_sweep.csv"), index=False)
    print(f"\nSaved: data/scalper_sweep.csv")

    # Show top configs by test PF
    top = df.nlargest(10, "test_pf")
    print(f"\nTop 10 by Test PF:")
    print(top[["take_profit", "hard_stop", "time_stop_bars", "roc_threshold",
               "train_n", "train_wr", "train_pf", "test_n", "test_wr", "test_pf", "wr_gap"]].to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "run":
            cmd_backtest()
        elif sys.argv[1] == "sweep":
            cmd_sweep()
        else:
            print("Usage: python -m scalper.backtest [run|sweep]")
    else:
        cmd_backtest()
