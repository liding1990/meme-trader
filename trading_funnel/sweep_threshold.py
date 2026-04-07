"""Fast threshold sweep: train models once per fold, sweep L2 probability cutoff.

Usage:
    PYTHONPATH=. python trading_funnel/sweep_threshold.py
"""

import json, os, sys
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

sys.path.insert(0, ".")
from trading_funnel.train_all import (
    load_radar_tokens, load_token_data, find_l0_windows, l1_filter,
    build_l2_dataset, build_l3_dataset, ACTION_SELL, SLIPPAGE_PCT, RUG_THRESHOLD,
)
from trading_funnel.features import (
    extract_features_at_tick, precompute_hmm_for_token,
    apply_hmm_to_features, FEATURE_NAMES,
)


def backtest_fold(test_tokens, l2_model, l3_model, l2_features, l3_features):
    """Run backtest and return per-entry data (proba + trade result) for threshold sweep."""
    entries = []

    for token in test_tokens:
        data = load_token_data(token["address"])
        if data is None:
            continue
        mcap, volume, holders, top10, n = data["mcap"], data["volume"], data["holders"], data["top10"], data["n"]
        l0_ticks = find_l0_windows(mcap, holders)
        if len(l0_ticks) == 0:
            continue

        hmm_s, hmm_d, hmm_t = precompute_hmm_for_token(mcap)
        cooldown = -1

        for tick in l0_ticks:
            if tick < cooldown:
                continue
            if not l1_filter(mcap, holders, top10, tick):
                continue

            feat = extract_features_at_tick(mcap, volume, holders, top10, tick)
            apply_hmm_to_features(feat, hmm_s, hmm_d, hmm_t, tick)
            l2_vec = np.array([[feat.get(c, 0) for c in l2_features]])
            l2_vec = np.nan_to_num(l2_vec, nan=0)
            l2_proba = float(l2_model.predict_proba(l2_vec)[0][1])

            # Simulate L3 trade (always, we'll filter by proba later)
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

                if t > tick:
                    tc = (mcap[t] - mcap[t-1]) / max(mcap[t-1], 1)
                    if tc < RUG_THRESHOLD:
                        realized += remaining * pnl
                        remaining = 0
                        exit_tick = t
                        break

                l3_feat = extract_features_at_tick(
                    mcap, volume, holders, top10, t,
                    entry_price=entry_price, holding_hours=t - tick,
                    sold_pct=1.0 - remaining, remaining_pct=remaining, peak_price=peak,
                )
                apply_hmm_to_features(l3_feat, hmm_s, hmm_d, hmm_t, t)
                l3_vec = np.array([[l3_feat.get(c, 0) for c in l3_features]])
                l3_vec = np.nan_to_num(l3_vec, nan=0)
                l3_pred = str(l3_model.predict(l3_vec).flatten()[0])

                if pnl < -0.35:
                    l3_pred = "EXIT"
                if pnl > 1.0 and l3_pred == "HOLD" and remaining > 0.25:
                    l3_pred = "TP_25"

                sell = min(ACTION_SELL.get(l3_pred, 0), remaining)
                if sell > 0:
                    realized += sell * pnl * (1 - SLIPPAGE_PCT)
                    remaining -= sell

            if remaining > 0.01:
                fp = (mcap[min(exit_tick, n-1)] - entry_price) / max(entry_price, 1)
                realized += remaining * fp * (1 - SLIPPAGE_PCT)

            entries.append({"l2_proba": l2_proba, "return": realized,
                           "hold_hours": exit_tick - tick, "symbol": token["symbol"]})
            cooldown = tick + 12

    return entries


def main():
    tokens = load_radar_tokens()
    print(f"加载 {len(tokens)} 个 token\n")

    l2_df = build_l2_dataset(tokens, verbose=False)
    l2_features = [c for c in FEATURE_NAMES if c in l2_df.columns]
    l3_features = l2_features

    unique_addrs = list(set(t["address"] for t in tokens if load_token_data(t["address"]) is not None))
    np.random.seed(42)
    np.random.shuffle(unique_addrs)
    fold_size = len(unique_addrs) // 5
    folds = [set(unique_addrs[i*fold_size:(i+1)*fold_size]) for i in range(5)]
    folds[-1].update(unique_addrs[5*fold_size:])

    # Train once per fold, collect all entry data
    all_entries = []
    for fold_idx, test_addrs in enumerate(folds):
        train_addrs = set(unique_addrs) - test_addrs
        train_tokens = [t for t in tokens if t["address"] in train_addrs]
        test_tokens = [t for t in tokens if t["address"] in test_addrs]

        print(f"Fold {fold_idx}: 训练 L2+L3...")
        l2_train = build_l2_dataset(train_tokens, verbose=False)
        if len(l2_train) < 50:
            continue
        X2 = np.nan_to_num(l2_train[l2_features].values.astype(float), nan=0)
        l2_model = CatBoostClassifier(iterations=300, depth=5, learning_rate=0.05,
                                       auto_class_weights="Balanced", random_seed=42, verbose=0)
        l2_model.fit(X2, l2_train["label"].values)

        l3_train = build_l3_dataset(train_tokens, l2_train[l2_train["label"] == 1])
        if len(l3_train) < 50:
            continue
        X3 = np.nan_to_num(l3_train[l3_features].values.astype(float), nan=0)
        l3_model = CatBoostClassifier(iterations=300, depth=5, learning_rate=0.05,
                                       auto_class_weights="Balanced", random_seed=42, verbose=0)
        l3_model.fit(X3, l3_train["label"].values)

        print(f"Fold {fold_idx}: 回测 {len(test_tokens)} 个 test token...")
        fold_entries = backtest_fold(test_tokens, l2_model, l3_model, l2_features, l3_features)
        all_entries.extend(fold_entries)
        print(f"Fold {fold_idx}: {len(fold_entries)} 笔潜在交易\n")

    df = pd.DataFrame(all_entries)
    print(f"总潜在交易: {len(df)}\n")

    # Sweep thresholds
    print(f"{'阈值':>6s}  {'交易':>5s}  {'平均':>8s}  {'中位':>8s}  {'胜率':>6s}  {'PF':>6s}  {'Sharpe':>7s}  {'P95亏':>8s}  {'avg亏':>8s}  {'avg赢':>8s}")
    print("-" * 90)

    for thresh in [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]:
        sub = df[df["l2_proba"] >= thresh]
        if len(sub) < 10:
            continue
        rets = sub["return"].values
        wins = rets > 0
        losses = rets <= 0
        n_t = len(rets)
        mean = rets.mean() * 100
        med = np.median(rets) * 100
        wr = wins.mean() * 100
        gp = rets[wins].sum() if wins.sum() > 0 else 0
        gl = abs(rets[losses].sum()) if losses.sum() > 0 else 1e-9
        pf = gp / gl
        sharpe = rets.mean() / max(rets.std(), 1e-9) * np.sqrt(252 * 6)
        p5_loss = np.percentile(rets, 5) * 100
        avg_l = rets[losses].mean() * 100 if losses.sum() > 0 else 0
        avg_w = rets[wins].mean() * 100 if wins.sum() > 0 else 0

        print(f"  {thresh:.2f}  {n_t:>5d}  {mean:>+7.1f}%  {med:>+7.1f}%  {wr:>5.1f}%  {pf:>5.2f}  {sharpe:>6.2f}  {p5_loss:>+7.1f}%  {avg_l:>+7.1f}%  {avg_w:>+7.1f}%")


if __name__ == "__main__":
    main()
