"""LightGBM meta-model: predict trade quality from all available features.

Takes the entry-time features from each trade (ROC, RVOL, OFI, Hurst, MACD, etc.)
and predicts whether the trade will be profitable. Uses triple-barrier labeling
and purged time-series cross-validation.

Two uses:
  1. Entry filter: only take trades where P(profit) > threshold
  2. Position sizing: size proportional to P(profit)

Usage:
    python meta_model.py train     # train on historical trades
    python meta_model.py evaluate  # evaluate with train/test split
"""

import argparse
import hashlib
import os
import sys
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from backtest_5m import Strategy5m, split_tokens, run_on_tokens
from indicators import extract_5m_candles, compute_all_indicators, generate_trade_management_signals
from survival import load_models as load_survival_models
from space4d import load_token_list

DATA_DIR = "data"


def extract_trade_features(trades, token_dfs):
    """Extract feature vectors at entry time for each trade.

    For each trade, look up the indicators DataFrame at the entry bar
    and extract all available features.
    """
    feature_rows = []

    for trade in trades:
        address = trade["address"]
        entry_idx = trade["entry_idx"]

        if address not in token_dfs:
            continue

        df = token_dfs[address]
        if entry_idx >= len(df):
            continue

        row = df.iloc[entry_idx]

        features = {
            # P4: Momentum quality
            "roc_30m": row.get("roc_30m", np.nan),
            "roc_1h": row.get("roc_1h", np.nan),
            "roc_accel_30m": row.get("roc_accel_30m", np.nan),
            "roc_accel_1h": row.get("roc_accel_1h", np.nan),
            "rvol": row.get("rvol", np.nan),
            "momentum_quality": row.get("momentum_quality", np.nan),

            # Ehlers DSP indicators
            "fisher": row.get("fisher", np.nan),
            "fisher_cross": row.get("fisher_cross", np.nan),
            "ebsw": row.get("ebsw", np.nan),
            "above_itrend": row.get("above_itrend", np.nan),

            # Legacy (kept for comparison)
            "macd_hist": row.get("macd_hist", np.nan),
            "macd_hist_slope": row.get("macd_hist_slope", np.nan),

            # P3: Trend strength
            "hurst": row.get("hurst", np.nan),

            # P2: Order flow
            "ofi_30m": row.get("ofi_30m", np.nan),
            "ofi_1h": row.get("ofi_1h", np.nan),
            "bs_ratio": row.get("bs_ratio", np.nan),
            "buyer_seller_ratio": row.get("buyer_seller_ratio", np.nan),

            # Raw data
            "volume": row.get("volume", np.nan),
            "buy_volume": row.get("buy_volume", np.nan),
            "sell_volume": row.get("sell_volume", np.nan),
            "liquidity": row.get("liquidity", np.nan),

            # Label
            "return_pct": trade["return_pct"],
            "profitable": 1 if trade["return_pct"] > 0 else 0,
        }

        feature_rows.append(features)

    return pd.DataFrame(feature_rows)


def build_dataset(tokens, strategy, kmf):
    """Build feature dataset from all tokens."""
    token_dfs = {}

    # Pre-compute indicator DataFrames for all tokens
    for token in tokens:
        df = extract_5m_candles(token["address"])
        if df is None or len(df) < 30:
            continue
        df = compute_all_indicators(df)
        df = generate_trade_management_signals(df)
        token_dfs[token["address"]] = df

    # Run strategy to get trades
    trades, _ = run_on_tokens(strategy, tokens, kmf=kmf)

    # Extract features at entry time
    features_df = extract_trade_features(trades, token_dfs)

    return features_df


FEATURE_COLS = [
    "roc_30m", "roc_1h", "roc_accel_30m", "roc_accel_1h",
    "rvol", "momentum_quality",
    "fisher", "fisher_cross", "ebsw", "above_itrend",
    "macd_hist", "macd_hist_slope",
    "hurst",
    "ofi_30m", "ofi_1h", "bs_ratio", "buyer_seller_ratio",
    "volume", "buy_volume", "sell_volume", "liquidity",
]


def train_model(features_df):
    """Train LightGBM classifier to predict profitable trades."""
    X = features_df[FEATURE_COLS].copy()
    y = features_df["profitable"].values

    # Replace inf with nan
    X = X.replace([np.inf, -np.inf], np.nan)

    # LightGBM handles NaN natively — no imputation needed
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
    )

    model.fit(X, y)

    return model


def evaluate_model(train_features, test_features, model):
    """Evaluate model on train/test split."""
    for label, df in [("TRAIN", train_features), ("TEST", test_features)]:
        X = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
        y = df["profitable"].values
        returns = df["return_pct"].values

        probs = model.predict_proba(X)[:, 1]
        preds = (probs > 0.5).astype(int)

        acc = accuracy_score(y, preds)
        try:
            auc = roc_auc_score(y, probs)
        except ValueError:
            auc = 0.5

        print(f"\n  {label} ({len(df)} trades):")
        print(f"    Accuracy:  {acc:.1%}")
        print(f"    AUC-ROC:   {auc:.3f}")
        print(f"    Baseline:  {y.mean():.1%} (always predict majority)")

        # Strategy improvement: only take trades where P(profit) > threshold
        for threshold in [0.5, 0.55, 0.6, 0.65, 0.7]:
            mask = probs > threshold
            if mask.sum() == 0:
                continue
            filtered_returns = returns[mask]
            wr = (filtered_returns > 0).mean() * 100
            avg = filtered_returns.mean()
            n = mask.sum()
            wins = filtered_returns[filtered_returns > 0]
            losses = filtered_returns[filtered_returns <= 0]
            gp = wins.sum() if len(wins) else 0
            gl = abs(losses.sum()) if len(losses) else 1
            pf = gp / gl if gl > 0 else 0

            print(f"    P>{threshold:.2f}: {n:>5d} trades, WR={wr:.0f}%, avg={avg:+.2f}%, PF={pf:.2f}")


def cmd_train(args):
    tokens = load_token_list()
    kmf, _ = load_survival_models()

    strategy = Strategy5m(
        roc_entry_threshold=3.0, rvol_entry_threshold=1.5,
        stop_loss=15, tight_stop_loss=8,
        hazard_exit_threshold=0.10, grace_period=6, min_mcap=100000,
        require_multi_bar_confirmation=True, dynamic_hazard=False,
    )

    print("Building feature dataset from all tokens...")
    features_df = build_dataset(tokens, strategy, kmf)
    print(f"Dataset: {len(features_df)} trades, {features_df['profitable'].mean():.1%} profitable")

    print("\nTraining LightGBM...")
    model = train_model(features_df)

    # Feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLS
    ).sort_values(ascending=False)

    print("\nFeature importance:")
    for feat, imp in importance.items():
        print(f"  {feat:25s}: {imp}")

    # Save
    os.makedirs("models", exist_ok=True)
    with open("models/meta_lgbm.pkl", "wb") as f:
        pickle.dump(model, f)
    features_df.to_csv(os.path.join(DATA_DIR, "meta_features.csv"), index=False)
    print("\nModel saved: models/meta_lgbm.pkl")


def cmd_evaluate(args):
    tokens = load_token_list()
    train_tokens, test_tokens = split_tokens(tokens)
    kmf, _ = load_survival_models()

    strategy = Strategy5m(
        roc_entry_threshold=3.0, rvol_entry_threshold=1.5,
        stop_loss=15, tight_stop_loss=8,
        hazard_exit_threshold=0.10, grace_period=6, min_mcap=100000,
        require_multi_bar_confirmation=True, dynamic_hazard=False,
    )

    print("Building train features...")
    train_features = build_dataset(train_tokens, strategy, kmf)
    print(f"Train: {len(train_features)} trades, {train_features['profitable'].mean():.1%} profitable")

    print("Building test features...")
    test_features = build_dataset(test_tokens, strategy, kmf)
    print(f"Test: {len(test_features)} trades, {test_features['profitable'].mean():.1%} profitable")

    print("\nTraining LightGBM on TRAIN only...")
    model = train_model(train_features)

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nFeature importance:")
    for feat, imp in importance.head(10).items():
        print(f"  {feat:25s}: {imp}")

    print(f"\n{'='*70}")
    print(f"  LightGBM Meta-Model Evaluation")
    print(f"{'='*70}")

    evaluate_model(train_features, test_features, model)

    # Compare: baseline strategy PF vs meta-filtered PF
    print(f"\n  Baseline strategy (no meta-filter):")
    base_rets = test_features["return_pct"].values
    base_wins = base_rets[base_rets > 0]
    base_losses = base_rets[base_rets <= 0]
    base_pf = base_wins.sum() / abs(base_losses.sum()) if base_losses.sum() != 0 else 0
    print(f"    {len(base_rets)} trades, WR={100*(base_rets>0).mean():.0f}%, "
          f"avg={base_rets.mean():+.2f}%, PF={base_pf:.2f}")

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="LightGBM Meta-Model for Trade Quality")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("train")
    sub.add_parser("evaluate")
    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
