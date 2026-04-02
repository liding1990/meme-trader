"""Early Warning — Token Prediction with SABCD Grading.

Usage:
    python early_warning/predict.py <address> [--chain sol|base]
"""

import argparse
import csv
import json
import os
import glob
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

# Add parent dir to path for gmgn_api import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DATA_DIR = "data"
RADAR_CSV = os.path.join(DATA_DIR, "radar_tokens.csv")
MCAP_THRESHOLD = 100_000
LOOKBACK_HOURS = 24
PREDICT_HOURS = 48
SLIDE_STEP_HOURS = 6
MODEL_PATH = os.path.join("early_warning", "model.txt")
FEATURE_COLS_PATH = os.path.join("early_warning", "feature_cols.json")


# ── Data Loading ─────────────────────────────────────────────────────────────

# Import all loaders and feature extraction from train_v3
from early_warning.train_v3 import (
    load_radar_tokens, load_1h_candles, load_moralis_holders,
    load_gmgn_holders, load_top10,
    extract_features, generate_samples,
    LOOKBACK_HOURS, PREDICT_HOURS, SLIDE_STEP_HOURS, MCAP_THRESHOLD,
)


def load_codex_5m(address: str) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, address, "codex_5m_bars.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if not data:
            return None
        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        for col in ["buy_volume", "sell_volume", "buyers", "sellers", "transactions", "liquidity"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df["net_volume"] = df["buy_volume"] - df["sell_volume"]
        return df.sort_values("datetime").reset_index(drop=True)
    except Exception:
        return None


def extract_codex_features(address: str, t_start, t_end) -> dict:
    """Extract Codex-specific features for a time window."""
    feat = {}
    codex_df = load_codex_5m(address)
    if codex_df is None or len(codex_df) < 2:
        return _codex_defaults()

    window = codex_df[(codex_df["datetime"] >= t_start) &
                      (codex_df["datetime"] <= t_end)]
    if len(window) < 2:
        return _codex_defaults()

    feat["net_volume_total"] = window["net_volume"].sum()
    feat["net_volume_mean"] = window["net_volume"].mean()
    feat["buy_sell_ratio"] = (window["buy_volume"].sum() /
                              max(window["sell_volume"].sum(), 1))
    feat["buyer_seller_ratio"] = (window["buyers"].sum() /
                                  max(window["sellers"].sum(), 1))
    feat["total_transactions"] = window["transactions"].sum()
    feat["avg_liquidity"] = window["liquidity"].mean()

    # Net volume trend: first half vs second half
    mid = len(window) // 2
    if mid > 0:
        nv1 = window["net_volume"].iloc[:mid].mean()
        nv2 = window["net_volume"].iloc[mid:].mean()
        feat["net_volume_trend"] = nv2 - nv1
    else:
        feat["net_volume_trend"] = 0

    # Buying pressure: % of bars with positive net volume
    feat["buy_pressure_pct"] = (window["net_volume"] > 0).mean()

    return feat


def _codex_defaults():
    return {
        "net_volume_total": 0, "net_volume_mean": 0,
        "buy_sell_ratio": 0, "buyer_seller_ratio": 0,
        "total_transactions": 0, "avg_liquidity": 0,
        "net_volume_trend": 0, "buy_pressure_pct": 0,
    }


# ── Model Training ───────────────────────────────────────────────────────────


HIST_FEATURES = ["hist_ath", "hist_ath_ratio", "token_age_hours",
                  "hist_return_total", "hist_volatility", "hist_pump_count"]
MODEL_HIST_PATH = os.path.join("early_warning", "model_hist.txt")
FEATURE_HIST_COLS_PATH = os.path.join("early_warning", "feature_hist_cols.json")
ENSEMBLE_WEIGHT = 0.5  # blend weight for history model


def train_model(force=False):
    """Train and save ensemble models (base + history)."""
    if (os.path.isfile(MODEL_PATH) and os.path.isfile(MODEL_HIST_PATH)
            and os.path.isfile(FEATURE_COLS_PATH) and not force):
        model_base = lgb.Booster(model_file=MODEL_PATH)
        model_hist = lgb.Booster(model_file=MODEL_HIST_PATH)
        with open(FEATURE_COLS_PATH) as f:
            feature_cols = json.load(f)
        with open(FEATURE_HIST_COLS_PATH) as f:
            feature_hist_cols = json.load(f)
        return model_base, model_hist, feature_cols, feature_hist_cols

    print("Training ensemble models...", file=sys.stderr)
    tokens = load_radar_tokens()
    all_samples = []
    for token in tokens:
        all_samples.extend(generate_samples(token))

    df = pd.DataFrame(all_samples)
    all_feature_cols = [c for c in df.columns if c not in
                        ("address", "symbol", "t_start", "t_end", "label", "future_max_mult")]
    base_cols = [c for c in all_feature_cols if c not in HIST_FEATURES]

    X_base = df[base_cols].values.astype(float)
    X_all = df[all_feature_cols].values.astype(float)
    y = np.log1p(df["future_max_mult"].values)

    params = dict(n_estimators=500, max_depth=6, learning_rate=0.03,
                  num_leaves=25, subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=0.2, reg_lambda=0.2, random_state=42, verbose=-1)

    model_base = lgb.LGBMRegressor(**params)
    model_base.fit(X_base, y)

    model_hist = lgb.LGBMRegressor(**params)
    model_hist.fit(X_all, y)

    # Save
    model_base.booster_.save_model(MODEL_PATH)
    model_hist.booster_.save_model(MODEL_HIST_PATH)
    with open(FEATURE_COLS_PATH, "w") as f:
        json.dump(base_cols, f)
    with open(FEATURE_HIST_COLS_PATH, "w") as f:
        json.dump(all_feature_cols, f)

    # Compute thresholds from ensemble predictions
    pred_base = model_base.predict(X_base)
    pred_hist = model_hist.predict(X_all)
    blended = (1 - ENSEMBLE_WEIGHT) * pred_base + ENSEMBLE_WEIGHT * pred_hist
    train_pred = np.expm1(blended)

    thresholds = {
        "p99": float(np.percentile(train_pred, 99)),
        "p95": float(np.percentile(train_pred, 95)),
        "p90": float(np.percentile(train_pred, 90)),
        "p80": float(np.percentile(train_pred, 80)),
        "p50": float(np.percentile(train_pred, 50)),
        "n_samples": len(train_pred),
    }
    with open(os.path.join("early_warning", "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"Models saved ({len(df)} samples, {len(base_cols)}+{len(all_feature_cols)} features)", file=sys.stderr)
    return model_base.booster_, model_hist.booster_, base_cols, all_feature_cols


def load_thresholds():
    path = os.path.join("early_warning", "thresholds.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {"p99": 4.0, "p95": 2.5, "p90": 2.0, "p80": 1.5, "p50": 1.1}


# ── Grading ──────────────────────────────────────────────────────────────────


def compute_grade(pred_mult: float, thresholds: dict) -> dict:
    """Assign SABCD grade based on predicted multiple and historical percentiles.

    S = top 1%  (historically 53% pump rate)
    A = top 5%  (historically ~40% pump rate)
    B = top 10% (historically ~35% pump rate)
    C = top 20% (above average)
    D = bottom 80% (below average)
    """
    if pred_mult >= thresholds["p99"]:
        grade = "S"
        label = "Strong Pump Signal"
        color = "\033[93m"  # yellow/gold
        desc = "Top 1% — historically 53% of tokens at this level pumped 2x+ in 48h"
    elif pred_mult >= thresholds["p95"]:
        grade = "A"
        label = "Pump Signal"
        color = "\033[92m"  # green
        desc = "Top 5% — historically ~40% pumped 2x+ in 48h"
    elif pred_mult >= thresholds["p90"]:
        grade = "B"
        label = "Bullish"
        color = "\033[94m"  # blue
        desc = "Top 10% — above average pump probability"
    elif pred_mult >= thresholds["p80"]:
        grade = "C"
        label = "Neutral-Bullish"
        color = "\033[95m"  # purple
        desc = "Top 20% — slightly above average"
    else:
        grade = "D"
        label = "Neutral / Bearish"
        color = "\033[90m"  # gray
        desc = "Bottom 80% — no significant pump signal"

    return {
        "grade": grade,
        "label": label,
        "color": color,
        "desc": desc,
    }


# ── Inference ────────────────────────────────────────────────────────────────


def predict_token(address: str, chain: str = "sol"):
    """Full prediction pipeline for a single token."""
    from gmgn_api import fetch_token_data

    # Train/load ensemble
    model_base, model_hist, base_cols, hist_cols = train_model()
    thresholds = load_thresholds()

    # Fetch data
    print(f"Fetching {address[:16]}...", file=sys.stderr)
    _, loaded_data = fetch_token_data(chain, address)

    mcap_data = loaded_data.get("token_mcap_candles", {})
    trend_data = loaded_data.get("token_trends", {})

    # Parse 1h candles
    candles = mcap_data.get("data", {}).get("list", [])
    if not candles:
        return {"error": "No candle data available"}

    df_1h = pd.DataFrame(candles)
    df_1h["time_ms"] = df_1h["time"].astype(int)
    df_1h["datetime"] = pd.to_datetime(df_1h["time_ms"], unit="ms")
    df_1h["mcap"] = df_1h["close"].astype(float)
    df_1h["volume"] = df_1h["volume"].astype(float)
    df_1h = df_1h.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

    # Parse holders
    holder_df = load_moralis_holders(address)
    if holder_df is None:
        holder_series = trend_data.get("data", {}).get("trends", {}).get("holder_count", [])
        if holder_series:
            holder_df = pd.DataFrame(holder_series)
            holder_df["datetime"] = pd.to_datetime(holder_df["timestamp"].astype(int), unit="s")
            holder_df["holders"] = holder_df["value"].astype(float)
            holder_df = holder_df[["datetime", "holders"]].sort_values("datetime").reset_index(drop=True)

    # Parse top10
    top10_df = None
    t10_series = trend_data.get("data", {}).get("trends", {}).get("top10_holder_percent", [])
    if t10_series:
        top10_df = pd.DataFrame(t10_series)
        top10_df["datetime"] = pd.to_datetime(top10_df["timestamp"].astype(int), unit="s")
        top10_df["top10_pct"] = top10_df["value"].astype(float)
        top10_df = top10_df[["datetime", "top10_pct"]].sort_values("datetime").reset_index(drop=True)

    # Extract features for latest 24h window
    t_end = df_1h["datetime"].iloc[-1]
    t_start = t_end - pd.Timedelta(hours=LOOKBACK_HOURS)

    feat = extract_features(df_1h, t_start, t_end, holder_df, top10_df)
    if feat is None:
        return {"error": "Insufficient data for feature extraction"}

    # Add historical context features
    # Use all 1h candle data from $100K crossing to now
    above_100k = df_1h[df_1h["mcap"] >= MCAP_THRESHOLD]
    if not above_100k.empty:
        t_origin = above_100k["datetime"].iloc[0]
        history = df_1h[(df_1h["datetime"] >= t_origin) & (df_1h["datetime"] < t_start)]
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

    # Add Codex features (for display only, not in model)
    codex_feat = extract_codex_features(address, t_start, t_end)

    # Build feature vectors for ensemble
    base_vec = np.array([[feat.get(c, 0) for c in base_cols]], dtype=float)
    hist_vec = np.array([[feat.get(c, 0) for c in hist_cols]], dtype=float)

    # Ensemble prediction
    pred_base = model_base.predict(base_vec)[0]
    pred_hist = model_hist.predict(hist_vec)[0]
    pred_log = (1 - ENSEMBLE_WEIGHT) * pred_base + ENSEMBLE_WEIGHT * pred_hist
    pred_mult = float(np.expm1(pred_log))

    # Grade
    grade_info = compute_grade(pred_mult, thresholds)

    # Compute rank
    # Quick estimate: use thresholds
    if pred_mult >= thresholds["p99"]:
        rank_pct = 1
    elif pred_mult >= thresholds["p95"]:
        rank_pct = 5
    elif pred_mult >= thresholds["p90"]:
        rank_pct = 10
    elif pred_mult >= thresholds["p80"]:
        rank_pct = 20
    else:
        rank_pct = 80

    return {
        "address": address,
        "chain": chain,
        "grade": grade_info,
        "prediction": {
            "predicted_max_multiple": round(pred_mult, 2),
            "horizon": f"{PREDICT_HOURS}h",
            "rank_percentile": f"top {rank_pct}%",
        },
        "current_state": {
            "mcap": feat.get("mcap_end", 0),
            "mcap_24h_return": feat.get("mcap_return", 0),
            "mcap_max_24h": feat.get("mcap_max", 0),
            "volatility": feat.get("volatility", 0),
            "max_drawdown": feat.get("max_drawdown", 0),
            "holders": feat.get("holders_end", 0),
            "holder_growth_24h": feat.get("holder_growth", 0),
            "mcap_per_holder": feat.get("mcap_per_holder", 0),
            "top10_pct": feat.get("top10_pct", None),
            "volume_total_24h": feat.get("volume_total", 0),
            "volume_trend": feat.get("volume_trend", 0),
        },
        "codex": codex_feat if any(v != 0 for v in codex_feat.values()) else None,
        "history": {
            "token_age_hours": feat.get("token_age_hours", 0),
            "hist_ath": feat.get("hist_ath", 0),
            "hist_ath_ratio": feat.get("hist_ath_ratio", 0),
            "hist_pump_count": feat.get("hist_pump_count", 0),
            "hist_volatility": feat.get("hist_volatility", 0),
        },
        "data_quality": {
            "candle_count": len(df_1h),
            "time_range": f"{df_1h['datetime'].iloc[0]} → {df_1h['datetime'].iloc[-1]}",
            "has_holders": holder_df is not None,
            "has_top10": top10_df is not None,
            "has_codex": codex_feat.get("total_transactions", 0) > 0,
        },
    }


def print_report(result: dict):
    """Pretty-print the prediction report."""
    if "error" in result:
        print(f"\nERROR: {result['error']}")
        return

    g = result["grade"]
    p = result["prediction"]
    s = result["current_state"]
    q = result["data_quality"]

    reset = "\033[0m"
    bold = "\033[1m"

    print(f"\n{'=' * 60}")
    print(f"{bold}TOKEN EARLY WARNING REPORT{reset}")
    print(f"{'=' * 60}")
    print(f"Address: {result['address']}")
    print(f"Chain:   {result['chain']}")

    print(f"\n{bold}┌─────────────────────────────────────────┐{reset}")
    print(f"{bold}│  {g['color']}Grade {g['grade']} — {g['label']}{reset}{bold}  │{reset}")
    print(f"{bold}└─────────────────────────────────────────┘{reset}")
    print(f"  {g['desc']}")
    print(f"  Predicted 48h max multiple: {bold}{p['predicted_max_multiple']}x{reset}")
    print(f"  Ranking: {p['rank_percentile']}")

    print(f"\n{bold}Current State (last 24h):{reset}")
    print(f"  MCap:           ${s['mcap']:>14,.0f}")
    print(f"  24h Return:     {s['mcap_24h_return']*100:>+13.1f}%")
    print(f"  24h High:       ${s['mcap_max_24h']:>14,.0f}")
    print(f"  Volatility:     {s['volatility']:>14.4f}")
    print(f"  Max Drawdown:   {s['max_drawdown']*100:>13.1f}%")
    if s['holders'] > 0:
        print(f"  Holders:        {s['holders']:>14,.0f}")
        print(f"  Holder Growth:  {s['holder_growth_24h']*100:>+13.1f}%")
        print(f"  MCap/Holder:    ${s['mcap_per_holder']:>14,.0f}")
    if s['top10_pct'] is not None and not np.isnan(s['top10_pct']):
        print(f"  Top10 Conc.:    {s['top10_pct']*100:>13.1f}%")
    print(f"  Volume (24h):   ${s['volume_total_24h']:>14,.0f}")
    print(f"  Volume Trend:   {s['volume_trend']:>+14.2f}")

    h = result.get("history", {})
    if h.get("token_age_hours", 0) > 0:
        print(f"\n{bold}Historical Context:{reset}")
        print(f"  Token Age:      {h['token_age_hours']:>14.0f}h")
        print(f"  Historical ATH: ${h['hist_ath']:>14,.0f}")
        print(f"  Current/ATH:    {h['hist_ath_ratio']*100:>13.1f}%")
        print(f"  Past 2x Pumps:  {h['hist_pump_count']:>14d}")
        print(f"  Hist Volatility:{h['hist_volatility']:>14.4f}")

    codex = result.get("codex")
    if codex:
        print(f"\n{bold}Codex On-Chain Data:{reset}")
        print(f"  Net Volume:     ${codex['net_volume_total']:>14,.0f}")
        print(f"  Buy/Sell Ratio: {codex['buy_sell_ratio']:>14.2f}")
        print(f"  Buyer/Seller:   {codex['buyer_seller_ratio']:>14.2f}")
        print(f"  Transactions:   {codex['total_transactions']:>14,.0f}")
        print(f"  Buy Pressure:   {codex['buy_pressure_pct']*100:>13.1f}%")
        print(f"  NV Trend:       {codex['net_volume_trend']:>+14.2f}")

    print(f"\n{bold}Data Quality:{reset}")
    print(f"  Candles: {q['candle_count']}, Range: {q['time_range']}")
    print(f"  Holders: {'Yes' if q['has_holders'] else 'No'} | "
          f"Top10: {'Yes' if q['has_top10'] else 'No'} | "
          f"Codex: {'Yes' if q['has_codex'] else 'No'}")


def main():
    parser = argparse.ArgumentParser(description="Early Warning Token Prediction")
    parser.add_argument("address", help="Token contract address")
    parser.add_argument("--chain", default="sol", choices=["sol", "base"])
    parser.add_argument("--retrain", action="store_true", help="Force model retrain")
    args = parser.parse_args()

    if args.retrain:
        train_model(force=True)

    result = predict_token(args.address, args.chain)
    print_report(result)


if __name__ == "__main__":
    main()
