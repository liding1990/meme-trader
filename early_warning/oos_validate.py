"""Out-of-sample validation using Codex-discovered tokens.

Discovers tokens NOT in radar set, fetches hourly bars from Codex,
runs model predictions, then checks against actual future performance.
"""

import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codex_api import filter_tokens, CODEX_API_KEY
from gmgn_api import fetch_full_candles
from early_warning.train_v3 import (
    load_radar_tokens, generate_samples, extract_features,
    load_1h_candles, load_moralis_holders, load_gmgn_holders, load_top10,
    MCAP_THRESHOLD, LOOKBACK_HOURS, PREDICT_HOURS, SLIDE_STEP_HOURS,
)

DATA_DIR = "data"
RADAR_CSV = os.path.join(DATA_DIR, "radar_tokens.csv")


def load_radar_addresses():
    addrs = set()
    with open(RADAR_CSV) as f:
        for row in csv.reader(f):
            if len(row) >= 4:
                addrs.add(row[0].lower())
    return addrs


def discover_oos_tokens(radar_addrs, n_per_period=50):
    """Discover out-of-sample tokens from Codex."""
    all_tokens = []
    # 6 month windows
    periods = [
        (1727740800, 1730419200),  # Oct 2025
        (1730419200, 1733011200),  # Nov 2025
        (1733011200, 1735689600),  # Dec 2025
        (1735689600, 1738368000),  # Jan 2026
        (1738368000, 1740787200),  # Feb 2026
        (1740787200, 1743465600),  # Mar 2026
    ]
    for start, end in periods:
        try:
            tokens, count, _ = filter_tokens(
                created_after=start, created_before=end,
                min_mcap=100000, min_holders=500, limit=n_per_period,
            )
            new = [t for t in tokens if t["token"]["address"].lower() not in radar_addrs]
            all_tokens.extend(new)
            print(f"  Period {start}: {len(new)} new tokens")
            time.sleep(0.5)
        except Exception as e:
            print(f"  Period {start}: error — {e}")

    # Deduplicate
    seen = set()
    unique = []
    for t in all_tokens:
        addr = t["token"]["address"]
        if addr not in seen:
            seen.add(addr)
            unique.append(t)
    return unique


def fetch_gmgn_hourly(address, chain="sol"):
    """Fetch hourly mcap candles from GMGN (paginated, full history)."""
    try:
        candles = fetch_full_candles(chain, address, resolution="1h", max_pages=5)
        if not candles:
            return None
        df = pd.DataFrame(candles)
        df["time_ms"] = df["time"].astype(int)
        df["datetime"] = pd.to_datetime(df["time_ms"], unit="ms")
        df["mcap"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    except Exception:
        return None



def main():
    if not CODEX_API_KEY:
        print("ERROR: Set CODEX_API_KEY in .env")
        return

    # Step 1: Train model on radar tokens
    print("Training model on radar tokens...")
    tokens = load_radar_tokens()
    all_samples = []
    for token in tokens:
        all_samples.extend(generate_samples(token))
    df_train = pd.DataFrame(all_samples)

    all_feature_cols = [c for c in df_train.columns if c not in
                        ("address", "symbol", "t_start", "t_end", "label", "future_max_mult")]
    hist_feats = ["hist_ath", "hist_ath_ratio", "token_age_hours",
                  "hist_return_total", "hist_volatility", "hist_pump_count"]
    base_cols = [c for c in all_feature_cols if c not in hist_feats]

    X_base = df_train[base_cols].values.astype(float)
    X_all = df_train[all_feature_cols].values.astype(float)
    y = np.log1p(df_train["future_max_mult"].values)

    params = dict(n_estimators=500, max_depth=6, learning_rate=0.03, num_leaves=25,
                  subsample=0.8, colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=0.2,
                  random_state=42, verbose=-1)

    m_base = lgb.LGBMRegressor(**params)
    m_base.fit(X_base, y)
    m_hist = lgb.LGBMRegressor(**params)
    m_hist.fit(X_all, y)

    print(f"Model trained on {len(df_train)} samples, {len(base_cols)}+{len(all_feature_cols)} features")

    # Step 2: Discover OOS tokens
    print("\nDiscovering out-of-sample tokens from Codex...")
    radar_addrs = load_radar_addresses()
    oos_tokens = discover_oos_tokens(radar_addrs, n_per_period=50)
    print(f"Found {len(oos_tokens)} OOS tokens")

    # Step 3: For each OOS token, fetch bars, extract features, predict, check actual
    print("\nFetching bars and evaluating...")
    results = []
    errors = 0

    for i, tok in enumerate(oos_tokens):
        addr = tok["token"]["address"]
        symbol = tok["token"].get("symbol", "?")

        df = fetch_gmgn_hourly(addr)
        time.sleep(1.5)  # rate limit for GMGN

        if df is None or len(df) < LOOKBACK_HOURS + PREDICT_HOURS:
            errors += 1
            continue

        # Find windows where mcap > threshold
        # Take the window that ends ~48h before the data ends (so we have future to validate)
        t_data_end = df["datetime"].iloc[-1]
        t_end = t_data_end - pd.Timedelta(hours=PREDICT_HOURS)
        t_start = t_end - pd.Timedelta(hours=LOOKBACK_HOURS)

        if t_start < df["datetime"].iloc[0]:
            errors += 1
            continue

        feat = extract_features(df, t_start, t_end)
        if feat is None:
            errors += 1
            continue

        # Add historical context
        above_thresh = df[df["mcap"] >= MCAP_THRESHOLD]
        if not above_thresh.empty:
            t_origin = above_thresh["datetime"].iloc[0]
            history = df[(df["datetime"] >= t_origin) & (df["datetime"] < t_start)]
            if len(history) >= 2:
                hist_mcap = history["mcap"].values
                feat["hist_ath"] = hist_mcap.max()
                feat["hist_ath_ratio"] = feat["mcap_end"] / max(hist_mcap.max(), 1)
                feat["token_age_hours"] = (t_start - t_origin).total_seconds() / 3600
                feat["hist_return_total"] = (hist_mcap[-1] - hist_mcap[0]) / max(hist_mcap[0], 1)
                feat["hist_volatility"] = np.std(np.diff(hist_mcap) / np.maximum(hist_mcap[:-1], 1))
                running_min = np.minimum.accumulate(hist_mcap)
                feat["hist_pump_count"] = int((hist_mcap / np.maximum(running_min, 1) >= 2.0).sum())
            else:
                feat["hist_ath"] = feat["mcap_end"]; feat["hist_ath_ratio"] = 1.0
                feat["token_age_hours"] = 0; feat["hist_return_total"] = 0
                feat["hist_volatility"] = 0; feat["hist_pump_count"] = 0
        else:
            feat["hist_ath"] = feat["mcap_end"]; feat["hist_ath_ratio"] = 1.0
            feat["token_age_hours"] = 0; feat["hist_return_total"] = 0
            feat["hist_volatility"] = 0; feat["hist_pump_count"] = 0

        mcap_at_end = feat["mcap_end"]
        if mcap_at_end <= 0:
            errors += 1
            continue

        # Actual future performance
        future = df[(df["datetime"] > t_end) & (df["datetime"] <= t_data_end)]
        if len(future) < 2:
            errors += 1
            continue

        actual_max = future["mcap"].max()
        actual_mult = actual_max / max(mcap_at_end, 1)

        # Predict (ensemble)
        base_vec = np.array([[feat.get(c, 0) for c in base_cols]], dtype=float)
        all_vec = np.array([[feat.get(c, 0) for c in all_feature_cols]], dtype=float)

        pred_base = m_base.predict(base_vec)[0]
        pred_hist = m_hist.predict(all_vec)[0]
        pred_log = 0.5 * pred_base + 0.5 * pred_hist
        pred_mult = float(np.expm1(pred_log))

        results.append({
            "symbol": symbol,
            "address": addr,
            "pred_mult": pred_mult,
            "actual_mult": round(actual_mult, 2),
            "is_pump": actual_mult >= 2.0,
            "mcap": mcap_at_end,
        })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(oos_tokens)}, {len(results)} valid, {errors} errors")

    print(f"\nProcessed: {len(results)} valid predictions, {errors} errors")

    if not results:
        print("No valid results.")
        return

    df_results = pd.DataFrame(results)
    n_pump = df_results["is_pump"].sum()
    pump_rate = n_pump / len(df_results) * 100

    print(f"\n{'=' * 60}")
    print(f"OUT-OF-SAMPLE VALIDATION")
    print(f"{'=' * 60}")
    print(f"Total tokens evaluated: {len(df_results)}")
    print(f"Actual pumps (2x+): {n_pump} ({pump_rate:.1f}%)")

    # Precision@K
    df_results = df_results.sort_values("pred_mult", ascending=False).reset_index(drop=True)

    print(f"\nPrecision@K (sorted by predicted multiple):")
    for k in [10, 20, 50, 100]:
        if k > len(df_results):
            continue
        top = df_results.head(k)
        n_hit = top["is_pump"].sum()
        avg_mult = top["actual_mult"].mean()
        print(f"  Top {k:>3d}: {n_hit:>3d} pumps ({n_hit/k*100:.1f}% precision), avg actual = {avg_mult:.2f}x")

    # Show top 30 predictions
    print(f"\nTop 30 Predictions:")
    print(f"{'Rank':>4s}  {'Symbol':>12s}  {'Pred':>6s}  {'Actual':>7s}  {'Pump':>4s}  {'MCap':>10s}")
    print("-" * 55)
    for i, row in df_results.head(30).iterrows():
        pump_mark = " YES" if row["is_pump"] else ""
        print(f"{i+1:>4d}  {row['symbol']:>12s}  {row['pred_mult']:>5.1f}x  {row['actual_mult']:>6.1f}x  {pump_mark:>4s}  ${row['mcap']:>9,.0f}")

    # Bottom 30 (should be low actual)
    print(f"\nBottom 30 Predictions (should be non-pumps):")
    bottom = df_results.tail(30)
    n_bottom_pump = bottom["is_pump"].sum()
    print(f"  Pumps in bottom 30: {n_bottom_pump} ({n_bottom_pump/30*100:.0f}%)")


if __name__ == "__main__":
    main()
