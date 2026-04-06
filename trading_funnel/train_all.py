"""Train all funnel layers + end-to-end backtest.

Step 1: Build L0 windows (historical scan)
Step 2: Apply L1 filter (outcome cluster + early warning)
Step 3: Train L2 entry model
Step 4: Train L3 position model
Step 5: End-to-end backtest

Usage:
    python trading_funnel/train_all.py
"""

import csv
import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report

DATA_DIR = "data"
RADAR_CSV = os.path.join(DATA_DIR, "radar_tokens.csv")
OUTPUT_DIR = "trading_funnel"

# L0 thresholds
MCAP_MIN = 100_000
HOLDER_MIN = 100
GAIN_4H_MIN = 0.15

# L2 thresholds
ENTRY_REWARD_THRESHOLD = 0.20  # 20% max gain in 48h = good entry
PREDICT_HOURS = 48

# L3 actions
ACTIONS = ["HOLD", "TP_25", "TP_50", "TP_100", "SL_25", "SL_50", "EXIT"]
ACTION_SELL = {"HOLD": 0, "TP_25": 0.25, "TP_50": 0.50, "TP_100": 1.0,
               "SL_25": 0.25, "SL_50": 0.50, "EXIT": 1.0}


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_radar_tokens():
    tokens = []
    with open(RADAR_CSV) as f:
        for row in csv.reader(f):
            if len(row) >= 4:
                tokens.append({"address": row[0], "chain": row[1], "symbol": row[3]})
    return tokens


def load_token_data(address):
    """Load all data for a token: hourly candles, holders, top10."""
    data_dir = os.path.join(DATA_DIR, address)

    # Hourly candles
    h_files = sorted([f for f in glob.glob(os.path.join(data_dir, "token_mcap_candles_[0-9]*.json"))
                       if "5m" not in os.path.basename(f)])
    if not h_files:
        return None
    try:
        with open(h_files[-1]) as f:
            data = json.load(f)
        candles = (data or {}).get("data", {}).get("list", [])
        if not candles or len(candles) < 10:
            return None
        df = pd.DataFrame(candles)
        df["mcap"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df["datetime"] = pd.to_datetime(df["time"].astype(int), unit="ms")
        df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    except Exception:
        return None

    mcap = df["mcap"].values
    volume = df["volume"].values
    n = len(mcap)

    # Holders
    holders = np.zeros(n)
    moralis_path = os.path.join(data_dir, "moralis_holders_1h.json")
    holder_loaded = False
    if os.path.isfile(moralis_path):
        try:
            with open(moralis_path) as f:
                hdata = json.load(f)
            if hdata:
                hdf = pd.DataFrame(hdata)
                hdf["datetime"] = pd.to_datetime(hdf["timestamp"], utc=True).dt.tz_localize(None)
                hdf["holders"] = hdf["totalHolders"].astype(float)
                hdf = hdf.set_index("datetime").resample("1h").last().ffill().reset_index()
                merged = pd.merge_asof(df[["datetime"]], hdf[["datetime", "holders"]],
                                        on="datetime", direction="backward")
                if "holders" in merged.columns:
                    holders = merged["holders"].fillna(0).values
                    holder_loaded = True
        except Exception:
            pass

    if not holder_loaded:
        t_files = sorted(glob.glob(os.path.join(data_dir, "token_trends_*.json")))
        if t_files:
            try:
                with open(t_files[-1]) as f:
                    tdata = json.load(f)
                series = (tdata or {}).get("data", {}).get("trends", {}).get("holder_count", [])
                if series:
                    hdf = pd.DataFrame(series)
                    hdf["datetime"] = pd.to_datetime(hdf["timestamp"].astype(int), unit="s")
                    hdf["holders"] = hdf["value"].astype(float)
                    hdf = hdf.set_index("datetime").resample("1h").last().ffill().reset_index()
                    merged = pd.merge_asof(df[["datetime"]], hdf[["datetime", "holders"]],
                                            on="datetime", direction="backward")
                    if "holders" in merged.columns:
                        holders = merged["holders"].fillna(0).values
            except Exception:
                pass

    # Top10
    top10 = np.full(n, np.nan)
    t_files = sorted(glob.glob(os.path.join(data_dir, "token_trends_*.json")))
    if t_files:
        try:
            with open(t_files[-1]) as f:
                tdata = json.load(f)
            t10 = (tdata or {}).get("data", {}).get("trends", {}).get("top10_holder_percent", [])
            if t10:
                t10df = pd.DataFrame(t10)
                t10df["datetime"] = pd.to_datetime(t10df["timestamp"].astype(int), unit="s")
                t10df["top10_pct"] = t10df["value"].astype(float)
                merged = pd.merge_asof(df[["datetime"]], t10df[["datetime", "top10_pct"]],
                                        on="datetime", direction="backward")
                if "top10_pct" in merged.columns:
                    top10 = merged["top10_pct"].values
        except Exception:
            pass

    return {"mcap": mcap, "volume": volume, "holders": holders, "top10": top10, "n": n}


# ── L0: Historical Scan ─────────────────────────────────────────────────────


def find_l0_windows(mcap, holders):
    """Find all hourly ticks where L0 conditions are met."""
    n = len(mcap)
    gain_4h = np.zeros(n)
    for i in range(4, n):
        if mcap[i - 4] > 0:
            gain_4h[i] = (mcap[i] - mcap[i - 4]) / mcap[i - 4]

    mask = (mcap >= MCAP_MIN) & (holders >= HOLDER_MIN) & (gain_4h >= GAIN_4H_MIN)
    return np.where(mask)[0]


# ── L1: Scam Filter ─────────────────────────────────────────────────────────


def l1_filter(mcap, holders, top10, tick_idx):
    """Simple rule-based scam filter (replaces full cluster model for speed).

    Returns True if token passes (not scam), False if filtered out.
    """
    i = tick_idx
    n = len(mcap)

    # Check for pump-dump pattern: mcap spike with no holders
    if holders[i] < 200:
        return False

    # Check for instant pump: ATH in first 2 hours of data
    if i >= 2:
        ath_so_far = mcap[:i+1].max()
        ath_idx = np.argmax(mcap[:i+1])
        if ath_idx <= 2 and mcap[i] < ath_so_far * 0.3:
            return False  # pumped and dumped already

    # Top10 concentration too high = whale manipulation
    if top10 is not None and not np.isnan(top10[i]) and top10[i] > 0.80:
        return False

    return True


# ── L2: Entry Model ─────────────────────────────────────────────────────────


def build_l2_dataset(tokens):
    """Build L2 training data: for each L0 window, compute features + entry label."""
    from trading_funnel.features import (extract_features_at_tick, FEATURE_NAMES,
                                          precompute_hmm_for_token, apply_hmm_to_features)

    samples = []
    n_tokens = 0

    for i, token in enumerate(tokens):
        data = load_token_data(token["address"])
        if data is None:
            continue

        mcap, volume, holders, top10, n = data["mcap"], data["volume"], data["holders"], data["top10"], data["n"]

        l0_ticks = find_l0_windows(mcap, holders)
        if len(l0_ticks) == 0:
            continue

        n_tokens += 1

        # Precompute HMM once for this token
        hmm_s, hmm_d, hmm_t = precompute_hmm_for_token(mcap)

        for tick in l0_ticks:
            if not l1_filter(mcap, holders, top10, tick):
                continue

            feat = extract_features_at_tick(mcap, volume, holders, top10, tick)
            apply_hmm_to_features(feat, hmm_s, hmm_d, hmm_t, tick)

            future_end = min(tick + PREDICT_HOURS, n)
            if future_end - tick < 6:
                continue
            future_max = mcap[tick:future_end].max()
            future_gain = (future_max - mcap[tick]) / max(mcap[tick], 1)
            label = 1 if future_gain >= ENTRY_REWARD_THRESHOLD else 0

            samples.append({
                "address": token["address"],
                "symbol": token["symbol"],
                "tick_idx": tick,
                "label": label,
                "future_gain": round(future_gain, 4),
                **feat,
            })

        if (i + 1) % 50 == 0:
            print(f"  L2: {i+1}/{len(tokens)}, {len(samples)} 样本, {n_tokens} token")

    print(f"L2 数据集: {len(samples)} 个样本, {n_tokens} 个 token")
    return pd.DataFrame(samples)


# ── L3: Position Model ───────────────────────────────────────────────────────


def compute_l3_label(mcap, tick, entry_price, remaining):
    """Hindsight-optimal action at this tick."""
    n = len(mcap)
    current = mcap[tick]
    pnl = (current - entry_price) / max(entry_price, 1)
    future = mcap[tick:min(tick + 6, n)]
    full_future = mcap[tick:min(tick + 48, n)]

    if len(future) < 2:
        return "HOLD"

    future_max = future.max()
    future_min = future.min()
    future_return = (future_max - current) / max(current, 1)
    near_drawdown = (future_min - current) / max(current, 1)
    long_return = (full_future.max() - current) / max(current, 1) if len(full_future) > 0 else future_return

    # Local peak detection
    is_peak = (tick + 3 < n and
               current >= mcap[tick+1] and current >= mcap[tick+2] and current >= mcap[tick+3])

    # TP at local peaks
    if is_peak and pnl > 0:
        if pnl > 1.5 and long_return < 0.20:
            return "TP_100" if remaining <= 0.5 else "TP_50"
        if pnl > 0.5 and near_drawdown < -0.20:
            return "TP_50" if remaining > 0.5 else "TP_25"
        if pnl > 0.2 and near_drawdown < -0.15:
            return "TP_25"

    # SL when clearly dying
    if pnl < -0.30 and long_return < 0.10:
        return "EXIT"
    if pnl < -0.20 and long_return < 0.05:
        return "SL_50" if remaining > 0.5 else "EXIT"
    if pnl < -0.15 and future_return < 0.03 and near_drawdown < -0.05:
        return "SL_25"

    return "HOLD"


def build_l3_dataset(tokens, l2_entries):
    """Build L3 training data from L2 entry points."""
    from trading_funnel.features import (extract_features_at_tick, precompute_hmm_for_token,
                                          apply_hmm_to_features)

    samples = []
    entry_groups = l2_entries.groupby("address")

    for token in tokens:
        addr = token["address"]
        if addr not in entry_groups.groups:
            continue

        data = load_token_data(addr)
        if data is None:
            continue

        mcap, volume, holders, top10, n = data["mcap"], data["volume"], data["holders"], data["top10"], data["n"]
        hmm_s, hmm_d, hmm_t = precompute_hmm_for_token(mcap)
        token_entries = entry_groups.get_group(addr)

        for _, entry_row in token_entries.iterrows():
            entry_tick = int(entry_row["tick_idx"])
            entry_price = mcap[entry_tick]
            remaining = 1.0
            peak = entry_price

            max_tick = min(entry_tick + 48, n)
            for t in range(entry_tick, max_tick):
                if remaining <= 0.01:
                    break

                peak = max(peak, mcap[t])
                label = compute_l3_label(mcap, t, entry_price, remaining)

                feat = extract_features_at_tick(
                    mcap, volume, holders, top10, t,
                    entry_price=entry_price,
                    holding_hours=t - entry_tick,
                    sold_pct=1.0 - remaining,
                    remaining_pct=remaining,
                    peak_price=peak,
                )
                apply_hmm_to_features(feat, hmm_s, hmm_d, hmm_t, t)

                samples.append({
                    "address": addr, "symbol": token["symbol"],
                    "entry_idx": entry_tick, "tick_idx": t,
                    "label": label, **feat,
                })

                sell = min(ACTION_SELL.get(label, 0), remaining)
                remaining -= sell

    print(f"L3 数据集: {len(samples)} 个样本")
    return pd.DataFrame(samples)


# ── End-to-End Backtest ──────────────────────────────────────────────────────


def backtest_e2e(tokens, l2_model, l3_model, l2_features, l3_features):
    """Run full funnel backtest: L0 → L1 → L2 → L3."""
    from trading_funnel.features import (extract_features_at_tick, precompute_hmm_for_token,
                                          apply_hmm_to_features)

    all_trades = []

    for token in tokens:
        data = load_token_data(token["address"])
        if data is None:
            continue
        mcap, volume, holders, top10, n = data["mcap"], data["volume"], data["holders"], data["top10"], data["n"]

        l0_ticks = find_l0_windows(mcap, holders)
        if len(l0_ticks) == 0:
            continue

        hmm_s, hmm_d, hmm_t = precompute_hmm_for_token(mcap)
        cooldown_until = -1

        for tick in l0_ticks:
            if tick < cooldown_until:
                continue

            if not l1_filter(mcap, holders, top10, tick):
                continue

            feat = extract_features_at_tick(mcap, volume, holders, top10, tick)
            apply_hmm_to_features(feat, hmm_s, hmm_d, hmm_t, tick)
            l2_vec = np.array([[feat.get(c, 0) for c in l2_features]])
            l2_vec = np.nan_to_num(l2_vec, nan=0)
            l2_pred = int(l2_model.predict(l2_vec).flatten()[0])

            if l2_pred != 1:
                continue

            entry_price = mcap[tick]
            remaining = 1.0
            realized = 0.0
            peak = entry_price
            exit_tick = min(tick + 48, n)

            for t in range(tick, min(tick + 48, n)):
                if remaining <= 0.01:
                    exit_tick = t
                    break

                peak = max(peak, mcap[t])
                pnl = (mcap[t] - entry_price) / max(entry_price, 1)

                l3_feat = extract_features_at_tick(
                    mcap, volume, holders, top10, t,
                    entry_price=entry_price, holding_hours=t - tick,
                    sold_pct=1.0 - remaining, remaining_pct=remaining, peak_price=peak,
                )
                apply_hmm_to_features(l3_feat, hmm_s, hmm_d, hmm_t, t)
                l3_vec = np.array([[l3_feat.get(c, 0) for c in l3_features]])
                l3_vec = np.nan_to_num(l3_vec, nan=0)
                l3_pred = str(l3_model.predict(l3_vec).flatten()[0])

                sell = min(ACTION_SELL.get(l3_pred, 0), remaining)
                if sell > 0:
                    realized += sell * pnl
                    remaining -= sell

            if remaining > 0.01:
                final_pnl = (mcap[min(exit_tick, n-1)] - entry_price) / max(entry_price, 1)
                realized += remaining * final_pnl

            all_trades.append({
                "symbol": token["symbol"], "entry_tick": tick,
                "return": realized, "hold_hours": exit_tick - tick,
                "max_unrealized": (peak - entry_price) / max(entry_price, 1),
            })
            cooldown_until = tick + 12

    return all_trades


def print_metrics(trades, name):
    """Print comprehensive metrics."""
    if not trades:
        print(f"\n--- {name}: 无交易 ---")
        return

    returns = np.array([t["return"] for t in trades])
    hold_hours = np.array([t["hold_hours"] for t in trades])
    wins = returns > 0
    losses = returns <= 0

    avg_ret = returns.mean() * 100
    win_rate = wins.mean() * 100
    avg_win = returns[wins].mean() * 100 if wins.sum() > 0 else 0
    avg_loss = returns[losses].mean() * 100 if losses.sum() > 0 else 0
    avg_win_h = hold_hours[wins].mean() if wins.sum() > 0 else 0
    avg_loss_h = hold_hours[losses].mean() if losses.sum() > 0 else 0

    gross_profit = returns[wins].sum() if wins.sum() > 0 else 0
    gross_loss = abs(returns[losses].sum()) if losses.sum() > 0 else 1e-9
    pf = gross_profit / gross_loss

    sharpe = returns.mean() / max(returns.std(), 1e-9) * np.sqrt(252 * 6)
    max_loss = abs(returns.min()) if len(returns) > 0 else 0
    calmar = (returns.mean() * 365) / max(max_loss, 1e-9)

    print(f"\n--- {name} ({len(trades)} 笔交易) ---")
    print(f"  平均收益:           {avg_ret:>+8.2f}%")
    print(f"  胜率:               {win_rate:>8.1f}%")
    print(f"  平均盈利:           {avg_win:>+8.2f}%  (持仓 {avg_win_h:.0f}h)")
    print(f"  平均亏损:           {avg_loss:>+8.2f}%  (持仓 {avg_loss_h:.0f}h)")
    print(f"  Profit Factor:      {pf:>8.2f}")
    print(f"  Sharpe Ratio:       {sharpe:>8.2f}")
    print(f"  Calmar Ratio:       {calmar:>8.2f}")
    print(f"  最大单笔亏损:       {max_loss*100:>8.2f}%")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    tokens = load_radar_tokens()
    print(f"加载 {len(tokens)} 个 token\n")

    # ── Step 1+2: Build L2 dataset (L0 scan + L1 filter + entry labels) ──
    print("=" * 60)
    print("Step 1+2: 构建 L2 入场模型数据集 (L0 扫描 + L1 过滤)")
    print("=" * 60)
    l2_df = build_l2_dataset(tokens)
    print(f"标签分布: {l2_df['label'].value_counts().to_dict()}")

    from trading_funnel.features import FEATURE_NAMES
    l2_features = [c for c in FEATURE_NAMES if c in l2_df.columns]
    X2 = l2_df[l2_features].values.astype(float)
    X2 = np.nan_to_num(X2, nan=0)
    y2 = l2_df["label"].values
    groups2 = l2_df["address"].values

    # ── Step 3: Train L2 ──
    print(f"\n{'='*60}")
    print("Step 3: 训练 L2 入场模型")
    print("=" * 60)

    gkf = GroupKFold(n_splits=5)
    y2_pred = np.full(len(y2), -1)
    l2_fold_models = {}

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X2, y2, groups2)):
        model = CatBoostClassifier(iterations=300, depth=5, learning_rate=0.05,
                                    auto_class_weights="Balanced", random_seed=42, verbose=0)
        model.fit(X2[train_idx], y2[train_idx])
        y2_pred[val_idx] = model.predict(X2[val_idx]).flatten().astype(int)
        for idx in val_idx:
            l2_fold_models[idx] = model

    print(f"\nL2 分类报告 (GroupKFold OOS):")
    print(classification_report(y2, y2_pred, target_names=["不买", "买入"], zero_division=0))

    # Entry precision: when model says buy, what % actually gain > 20%?
    buy_mask = y2_pred == 1
    if buy_mask.sum() > 0:
        actual_gains = l2_df.loc[buy_mask, "future_gain"].values
        entry_precision = (actual_gains >= ENTRY_REWARD_THRESHOLD).mean() * 100
        avg_gain_when_buy = actual_gains.mean() * 100
        print(f"  入场信号精度: {entry_precision:.1f}% (说买时实际涨>20%的占比)")
        print(f"  入场后平均最大涨幅: {avg_gain_when_buy:.1f}%")
        print(f"  日均信号数 (估): {buy_mask.sum() / max(l2_df['address'].nunique(), 1) * 2:.1f}")

    # Train final L2
    l2_final = CatBoostClassifier(iterations=300, depth=5, learning_rate=0.05,
                                   auto_class_weights="Balanced", random_seed=42, verbose=0)
    l2_final.fit(X2, y2)

    # ── Step 4: Build + Train L3 ──
    print(f"\n{'='*60}")
    print("Step 4: 训练 L3 持仓管理模型")
    print("=" * 60)

    # Use L2 positive entries as L3 training data
    l2_entries = l2_df[l2_df["label"] == 1].copy()
    print(f"L2 正例 entry 数: {len(l2_entries)}")

    l3_df = build_l3_dataset(tokens, l2_entries)
    if len(l3_df) < 100:
        print("L3 数据不足，跳过")
        return

    print(f"标签分布: {l3_df['label'].value_counts().to_dict()}")

    l3_features = [c for c in FEATURE_NAMES if c in l3_df.columns]
    X3 = l3_df[l3_features].values.astype(float)
    X3 = np.nan_to_num(X3, nan=0)
    y3 = l3_df["label"].values
    groups3 = l3_df["address"].values

    gkf3 = GroupKFold(n_splits=min(5, len(np.unique(groups3))))
    y3_pred = np.full(len(y3), "", dtype=object)

    for fold, (train_idx, val_idx) in enumerate(gkf3.split(X3, y3, groups3)):
        model = CatBoostClassifier(iterations=300, depth=5, learning_rate=0.05,
                                    auto_class_weights="Balanced", random_seed=42, verbose=0)
        model.fit(X3[train_idx], y3[train_idx])
        y3_pred[val_idx] = model.predict(X3[val_idx]).flatten()

    print(f"\nL3 分类报告 (GroupKFold OOS):")
    print(classification_report(y3, y3_pred, labels=ACTIONS, zero_division=0))

    l3_final = CatBoostClassifier(iterations=300, depth=5, learning_rate=0.05,
                                   auto_class_weights="Balanced", random_seed=42, verbose=0)
    l3_final.fit(X3, y3)

    # Save models
    l2_final.save_model(os.path.join(OUTPUT_DIR, "l2_entry.cbm"))
    l3_final.save_model(os.path.join(OUTPUT_DIR, "l3_position.cbm"))
    with open(os.path.join(OUTPUT_DIR, "feature_config.json"), "w") as f:
        json.dump({"l2_features": l2_features, "l3_features": l3_features}, f)
    print(f"\n模型已保存到 {OUTPUT_DIR}/")

    # ── Step 5: End-to-End Backtest ──
    print(f"\n{'='*60}")
    print("Step 5: 端到端回测")
    print("=" * 60)

    trades = backtest_e2e(tokens, l2_final, l3_final, l2_features, l3_features)
    print_metrics(trades, "完整漏斗 (L0→L1→L2→L3)")

    # Baseline: L0 + L1 only, 48h time exit
    baseline_trades = []
    for token in tokens:
        data = load_token_data(token["address"])
        if data is None:
            continue
        mcap, volume, holders, top10, n = data["mcap"], data["volume"], data["holders"], data["top10"], data["n"]
        l0_ticks = find_l0_windows(mcap, holders)
        cooldown = -1
        for tick in l0_ticks:
            if tick < cooldown:
                continue
            if not l1_filter(mcap, holders, top10, tick):
                continue
            exit_t = min(tick + 48, n - 1)
            ret = (mcap[exit_t] - mcap[tick]) / max(mcap[tick], 1)
            baseline_trades.append({"symbol": token["symbol"], "return": ret,
                                     "hold_hours": exit_t - tick, "entry_tick": tick, "max_unrealized": 0})
            cooldown = tick + 12

    print_metrics(baseline_trades, "Baseline (L0+L1, 48h 定时退出)")

    # Summary
    print(f"\n{'='*60}")
    print("逐层指标")
    print("=" * 60)
    print(f"  L2 数据集:         {len(l2_df)} (通过 L0+L1)")
    print(f"  L2 正例 (应买):    {(y2==1).sum()} ({(y2==1).mean()*100:.1f}%)")
    print(f"  L3 数据集:         {len(l3_df)}")
    print(f"  完整漏斗交易数:    {len(trades)}")
    print(f"  Baseline 交易数:   {len(baseline_trades)}")


if __name__ == "__main__":
    main()
