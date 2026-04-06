"""Train CatBoost model for position management + backtest evaluation.

Usage:
    python posbot_train/train.py
"""

import os
import sys
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report

DATASET_PATH = "posbot_train/dataset.parquet"
MODEL_PATH = "posbot_train/model.cbm"
LABELS = ["HOLD", "TP_25", "TP_50", "TP_100", "SL_25", "SL_50", "EXIT"]
META_COLS = ["address", "symbol", "entry_idx", "tick_idx", "label", "remaining_pct"]


# ── Training ─────────────────────────────────────────────────────────────────


def train_and_evaluate():
    df = pd.read_parquet(DATASET_PATH)
    print(f"数据集: {len(df):,} 个样本, {df['address'].nunique()} 个 token")
    print(f"标签分布:\n{df['label'].value_counts().to_string()}\n")

    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    y = df["label"].values
    groups = df["address"].values

    # GroupKFold: 5 folds, entire tokens held out
    gkf = GroupKFold(n_splits=5)
    y_pred_all = np.full(len(y), "", dtype=object)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"  Fold {fold}: train={len(train_idx):,}, val={len(val_idx):,}")

        model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=0,
        )
        model.fit(X[train_idx], y[train_idx])
        y_pred_all[val_idx] = model.predict(X[val_idx]).flatten()

    # Classification report
    print("\n" + "=" * 70)
    print("分类报告（GroupKFold 样本外）")
    print("=" * 70)
    print(classification_report(y, y_pred_all, target_names=LABELS, zero_division=0))

    # Train final model on all data
    print("训练最终模型...")
    model_final = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        auto_class_weights="Balanced", random_seed=42, verbose=0,
    )
    model_final.fit(X, y)
    model_final.save_model(MODEL_PATH)

    # Feature importance
    importances = model_final.get_feature_importance()
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False)
    print("\nTop 15 特征重要性:")
    for _, row in imp_df.head(15).iterrows():
        print(f"  {row['feature']:>25s}: {row['importance']:.1f}")

    return df, y_pred_all, feature_cols


# ── Backtest ─────────────────────────────────────────────────────────────────


def backtest(df, feature_cols):
    """Simulate trading with dynamic model re-prediction, compare vs baselines."""
    print("\n" + "=" * 70)
    print("回测评估（动态仓位模拟）")
    print("=" * 70)

    X = df[feature_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    y = df["label"].values
    groups = df["address"].values

    # Train per-fold models for proper OOS backtest
    gkf = GroupKFold(n_splits=5)
    fold_models = {}
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        model = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            auto_class_weights="Balanced", random_seed=42, verbose=0,
        )
        model.fit(X[train_idx], y[train_idx])
        for idx in val_idx:
            fold_models[idx] = model

    # Map each row to its fold model
    df = df.copy()
    df["_row_idx"] = range(len(df))

    trades_model = []
    trades_bnh = []
    trades_rules = []

    def _fixed_rules(pnl_series):
        """Fixed rules baseline: 2x TP, -20% SL."""
        for i, pnl in enumerate(pnl_series):
            if pnl >= 1.0:  # 2x
                return pnl, i + 1
            if pnl <= -0.20:
                return pnl, i + 1
        return pnl_series[-1], len(pnl_series)

    for (addr, entry_idx), group in df.groupby(["address", "entry_idx"]):
        group = group.sort_values("tick_idx")
        pnl_series = group["unrealized_pnl"].values
        symbol = group.iloc[0]["symbol"]

        # ── Model strategy (with dynamic feature update) ──
        remaining = 1.0
        realized = 0.0
        max_pnl = 0.0
        exit_hour = len(group)

        # Get the OOS model for this trade
        first_row_idx = group["_row_idx"].iloc[0]
        model_for_bt = fold_models.get(first_row_idx)
        if model_for_bt is None:
            # fallback: skip this trade for model strategy
            trades_model.append({"symbol": symbol, "return": pnl_series[-1], "hold_hours": len(group)})
            trades_bnh.append({"symbol": symbol, "return": pnl_series[-1], "hold_hours": len(group)})
            rules_return, rules_hours = _fixed_rules(pnl_series)
            trades_rules.append({"symbol": symbol, "return": rules_return, "hold_hours": rules_hours})
            continue

        trade_features = group[feature_cols].values.astype(float).copy()
        trade_features = np.nan_to_num(trade_features, nan=0, posinf=0, neginf=0)

        sold_pct_col = feature_cols.index("sold_pct") if "sold_pct" in feature_cols else None
        remaining_col = feature_cols.index("remaining_pct") if "remaining_pct" in feature_cols else None

        for i in range(len(group)):
            pnl = pnl_series[i]
            max_pnl = max(max_pnl, pnl)

            if remaining <= 0.01:
                break

            if sold_pct_col is not None:
                trade_features[i, sold_pct_col] = 1.0 - remaining
            if remaining_col is not None:
                trade_features[i, remaining_col] = remaining

            pred = str(model_for_bt.predict(trade_features[i:i+1]).flatten()[0])

            sell_pcts = {"TP_25": 0.25, "TP_50": 0.50, "TP_100": 1.0,
                         "SL_25": 0.25, "SL_50": 0.50, "EXIT": 1.0, "HOLD": 0.0}
            sell = min(sell_pcts.get(pred, 0), remaining)

            if sell > 0:
                realized += sell * pnl
                remaining -= sell
                if remaining <= 0.01:
                    exit_hour = i + 1
                    break

        # Force exit at end if still holding
        if remaining > 0.01:
            realized += remaining * pnl_series[-1]

        trades_model.append({
            "symbol": symbol, "return": realized,
            "max_unrealized": max_pnl,
            "hold_hours": exit_hour,
        })

        # ── Buy and Hold 48h ──
        trades_bnh.append({"symbol": symbol, "return": pnl_series[-1], "hold_hours": len(group)})

        # ── Fixed rules: 2x TP, -20% SL ──
        rules_return, rules_hours = _fixed_rules(pnl_series)
        trades_rules.append({"symbol": symbol, "return": rules_return, "hold_hours": rules_hours})

    # ── Compute metrics ──
    def metrics(trades, name):
        returns = np.array([t["return"] for t in trades])
        hold_hours = np.array([t.get("hold_hours", 48) for t in trades])
        wins = returns > 0
        losses = returns <= 0

        n_trades = len(trades)
        avg_return = returns.mean() * 100
        win_rate = wins.mean() * 100

        avg_win = returns[wins].mean() * 100 if wins.sum() > 0 else 0
        avg_loss = returns[losses].mean() * 100 if losses.sum() > 0 else 0
        avg_win_hours = hold_hours[wins].mean() if wins.sum() > 0 else 0
        avg_loss_hours = hold_hours[losses].mean() if losses.sum() > 0 else 0

        # Profit Factor = gross profit / gross loss
        gross_profit = returns[wins].sum() if wins.sum() > 0 else 0
        gross_loss = abs(returns[losses].sum()) if losses.sum() > 0 else 1e-9
        profit_factor = gross_profit / gross_loss

        # Sharpe Ratio (annualized, assuming ~6 trades per day)
        sharpe = returns.mean() / max(returns.std(), 1e-9) * np.sqrt(252 * 6)

        # Max drawdown: worst single-trade loss (not cumulative)
        max_dd = abs(returns.min()) if len(returns) > 0 else 0

        # Calmar Ratio = annualized return / max single-trade drawdown
        annualized_return = returns.mean() * 365
        calmar = annualized_return / max(max_dd, 1e-9)

        print(f"\n--- {name} ({n_trades} 笔交易) ---")
        print(f"  平均收益:         {avg_return:>+8.2f}%")
        print(f"  胜率:             {win_rate:>8.1f}%")
        print(f"  平均盈利:         {avg_win:>+8.2f}%  (平均持仓 {avg_win_hours:.0f}h)")
        print(f"  平均亏损:         {avg_loss:>+8.2f}%  (平均持仓 {avg_loss_hours:.0f}h)")
        print(f"  Profit Factor:    {profit_factor:>8.2f}")
        print(f"  Sharpe Ratio:     {sharpe:>8.2f}")
        print(f"  Calmar Ratio:     {calmar:>8.2f}")
        print(f"  最大单笔亏损:     {-max_dd*100:>8.2f}%")

        return {
            "name": name, "n_trades": n_trades,
            "avg_return": avg_return, "win_rate": win_rate,
            "avg_win": avg_win, "avg_loss": avg_loss,
            "avg_win_hours": avg_win_hours, "avg_loss_hours": avg_loss_hours,
            "profit_factor": profit_factor,
            "sharpe": sharpe, "calmar": calmar,
            "max_dd": max_dd * 100,
        }

    m1 = metrics(trades_model, "CatBoost 模型")
    m2 = metrics(trades_bnh, "48h 定时退出")
    m3 = metrics(trades_rules, "固定规则 (2x TP / -20% SL)")

    # Summary comparison table
    print(f"\n{'=' * 70}")
    print(f"{'指标':>20s} | {'CatBoost':>12s} | {'48h定时退出':>12s} | {'固定规则':>12s}")
    print(f"{'-'*20}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    for key, label in [
        ("avg_return", "平均收益 (%)"),
        ("win_rate", "胜率 (%)"),
        ("avg_win", "平均盈利 (%)"),
        ("avg_loss", "平均亏损 (%)"),
        ("avg_win_hours", "盈利持仓 (h)"),
        ("avg_loss_hours", "亏损持仓 (h)"),
        ("profit_factor", "Profit Factor"),
        ("sharpe", "Sharpe Ratio"),
        ("calmar", "Calmar Ratio"),
        ("max_dd", "最大单笔亏损 (%)"),
    ]:
        print(f"  {label:>18s} | {m1[key]:>12.2f} | {m2[key]:>12.2f} | {m3[key]:>12.2f}")

    # Known tokens only (action distribution removed since preds are per-tick dynamic now)

    # Per-token results for known tokens
    print(f"\n--- 已知 Token 表现 ---")
    known = ["PUNCH", "GOYIM", "WAR", "GORK", "BFS", "CAPTCHA"]
    for sym in known:
        sym_trades = [t for t in trades_model if t["symbol"] == sym]
        if not sym_trades:
            continue
        returns = [t["return"] for t in sym_trades]
        print(f"  {sym:>10s}: {len(sym_trades):>3d} 笔, "
              f"平均 {np.mean(returns)*100:>+6.1f}%, "
              f"最佳 {max(returns)*100:>+6.1f}%, "
              f"最差 {min(returns)*100:>+6.1f}%")


def main():
    df, y_pred, feature_cols = train_and_evaluate()
    backtest(df, feature_cols)
    print(f"\n模型已保存到 {MODEL_PATH}")


if __name__ == "__main__":
    main()
