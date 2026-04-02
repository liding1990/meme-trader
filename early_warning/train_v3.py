"""Early Warning System v3 — Sliding window on HOURLY data with future-label.

Uses 1h candles (full history coverage) instead of 5m candles.
Generates sliding windows: 24h lookback on 1h data, labels from future 48h.

This solves two problems:
1. 5m data doesn't cover early period → use 1h data which covers everything
2. Fixed "first 24h only" misses late bloomers → sliding window catches all phases
"""

import csv
import json
import os
import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb


DATA_DIR = "data"
RADAR_CSV = os.path.join(DATA_DIR, "radar_tokens.csv")
MCAP_THRESHOLD = 100_000
LOOKBACK_HOURS = 24
PREDICT_HOURS = 48
SLIDE_STEP_HOURS = 6
PUMP_THRESHOLD = 2.0     # 2x in 48h = pump signal
DUMP_THRESHOLD = 0.4      # drop to 40% = dump


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_radar_tokens():
    tokens = []
    with open(RADAR_CSV) as f:
        for row in csv.reader(f):
            if len(row) >= 4:
                tokens.append({"address": row[0], "chain": row[1],
                                "name": row[2], "symbol": row[3]})
    return tokens


def load_1h_candles(address):
    data_dir = os.path.join(DATA_DIR, address)
    h_files = sorted([
        f for f in glob.glob(os.path.join(data_dir, "token_mcap_candles_[0-9]*.json"))
        if "5m" not in os.path.basename(f)
    ])
    if not h_files:
        return None
    try:
        with open(h_files[-1]) as f:
            data = json.load(f)
        candles = (data or {}).get("data", {}).get("list", [])
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


def load_moralis_holders(address):
    path = os.path.join(DATA_DIR, address, "moralis_holders_1h.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if not data:
            return None
        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
        df["holders"] = df["totalHolders"].astype(float)
        return df[["datetime", "holders"]].sort_values("datetime").reset_index(drop=True)
    except Exception:
        return None


def load_gmgn_holders(address):
    data_dir = os.path.join(DATA_DIR, address)
    t_files = sorted(glob.glob(os.path.join(data_dir, "token_trends_*.json")))
    if not t_files:
        return None
    try:
        with open(t_files[-1]) as f:
            data = json.load(f)
        series = (data or {}).get("data", {}).get("trends", {}).get("holder_count", [])
        if not series:
            return None
        df = pd.DataFrame(series)
        df["datetime"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        df["holders"] = df["value"].astype(float)
        return df[["datetime", "holders"]].sort_values("datetime").reset_index(drop=True)
    except Exception:
        return None


def load_top10(address):
    data_dir = os.path.join(DATA_DIR, address)
    t_files = sorted(glob.glob(os.path.join(data_dir, "token_trends_*.json")))
    if not t_files:
        return None
    try:
        with open(t_files[-1]) as f:
            data = json.load(f)
        t10 = (data or {}).get("data", {}).get("trends", {}).get("top10_holder_percent", [])
        if not t10:
            return None
        df = pd.DataFrame(t10)
        df["datetime"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        df["top10_pct"] = df["value"].astype(float)
        return df[["datetime", "top10_pct"]].sort_values("datetime").reset_index(drop=True)
    except Exception:
        return None


# ── Feature Extraction (hourly window) ───────────────────────────────────────


def extract_features(df_1h, t_start, t_end, holder_df=None, top10_df=None):
    """Extract features from a 24h window of hourly candles."""
    window = df_1h[(df_1h["datetime"] >= t_start) & (df_1h["datetime"] <= t_end)].copy()
    if len(window) < 4:
        return None

    mcap = window["mcap"].values
    volume = window["volume"].values
    n = len(mcap)

    feat = {}

    # Price features
    feat["mcap_start"] = mcap[0]
    feat["mcap_end"] = mcap[-1]
    feat["mcap_max"] = mcap.max()
    feat["mcap_min"] = mcap.min()
    feat["mcap_return"] = (mcap[-1] - mcap[0]) / max(mcap[0], 1)
    feat["mcap_range"] = (mcap.max() - mcap.min()) / max(mcap[0], 1)

    # Sub-window returns
    for frac, label in [(0.25, "6h"), (0.5, "12h")]:
        idx = max(1, int(n * frac))
        feat[f"return_{label}"] = (mcap[idx] - mcap[0]) / max(mcap[0], 1)

    # Volatility (hourly returns std)
    returns = np.diff(mcap) / np.maximum(mcap[:-1], 1)
    feat["volatility"] = np.std(returns) if len(returns) > 1 else 0
    feat["mean_return"] = np.mean(returns) if len(returns) > 0 else 0

    # Drawdown
    running_max = np.maximum.accumulate(mcap)
    drawdowns = (mcap - running_max) / np.maximum(running_max, 1)
    feat["max_drawdown"] = drawdowns.min()

    # Momentum: last 6h vs first 6h
    q1 = max(1, n // 4)
    q3 = min(n - 1, 3 * n // 4)
    r_first = (mcap[q1] - mcap[0]) / max(mcap[0], 1)
    r_last = (mcap[-1] - mcap[q3]) / max(mcap[q3], 1)
    feat["momentum_shift"] = r_last - r_first

    # Peak position (0 = start, 1 = end)
    feat["peak_position"] = np.argmax(mcap) / max(n - 1, 1)

    # Volume features
    feat["volume_total"] = volume.sum()
    feat["volume_mean"] = volume.mean()

    mid = n // 2
    if mid > 0:
        v1 = volume[:mid].mean()
        v2 = volume[mid:].mean()
        feat["volume_trend"] = (v2 - v1) / max(v1, 1)
    else:
        feat["volume_trend"] = 0

    if volume.sum() > 0:
        feat["volume_concentration"] = volume.max() / volume.sum()
    else:
        feat["volume_concentration"] = 0

    # Holder features
    if holder_df is not None and len(holder_df) >= 2:
        h_window = holder_df[(holder_df["datetime"] >= t_start - pd.Timedelta(hours=1)) &
                             (holder_df["datetime"] <= t_end + pd.Timedelta(hours=1))]
        if len(h_window) >= 2:
            h_vals = h_window["holders"].values
            feat["holders_start"] = h_vals[0]
            feat["holders_end"] = h_vals[-1]
            feat["holder_growth"] = (h_vals[-1] - h_vals[0]) / max(h_vals[0], 1)
            feat["mcap_per_holder"] = mcap[-1] / max(h_vals[-1], 1)

            # Holder-mcap correlation
            if len(h_vals) >= 3 and len(mcap) >= 3:
                # Align lengths
                min_len = min(len(h_vals), len(mcap))
                corr = np.corrcoef(mcap[:min_len], h_vals[:min_len])[0, 1]
                feat["mcap_holder_corr"] = corr if not np.isnan(corr) else 0
            else:
                feat["mcap_holder_corr"] = 0
        else:
            _fill_holder_defaults(feat, mcap)
    else:
        _fill_holder_defaults(feat, mcap)

    # Top10 features
    if top10_df is not None and len(top10_df) >= 1:
        t10_w = top10_df[(top10_df["datetime"] >= t_start - pd.Timedelta(hours=12)) &
                         (top10_df["datetime"] <= t_end + pd.Timedelta(hours=12))]
        if len(t10_w) >= 1:
            feat["top10_pct"] = t10_w["top10_pct"].iloc[-1]
            feat["top10_change"] = t10_w["top10_pct"].iloc[-1] - t10_w["top10_pct"].iloc[0]
        else:
            feat["top10_pct"] = np.nan
            feat["top10_change"] = np.nan
    else:
        feat["top10_pct"] = np.nan
        feat["top10_change"] = np.nan

    return feat


def _fill_holder_defaults(feat, mcap):
    feat["holders_start"] = 0
    feat["holders_end"] = 0
    feat["holder_growth"] = 0
    feat["mcap_per_holder"] = mcap[-1]
    feat["mcap_holder_corr"] = 0


# ── Sample Generation ────────────────────────────────────────────────────────


def generate_samples(token):
    addr = token["address"]
    symbol = token["symbol"]

    df_1h = load_1h_candles(addr)
    if df_1h is None or len(df_1h) < LOOKBACK_HOURS + PREDICT_HOURS:
        return []

    holder_df = load_moralis_holders(addr)
    if holder_df is None:
        holder_df = load_gmgn_holders(addr)
    top10_df = load_top10(addr)

    # Find $100K crossing
    above = df_1h[df_1h["mcap"] >= MCAP_THRESHOLD]
    if above.empty:
        return []
    t_origin = above["datetime"].iloc[0]
    t_data_end = df_1h["datetime"].iloc[-1]

    samples = []
    t_start = t_origin

    while True:
        t_end = t_start + pd.Timedelta(hours=LOOKBACK_HOURS)
        t_predict_end = t_end + pd.Timedelta(hours=PREDICT_HOURS)

        if t_predict_end > t_data_end:
            break

        # Features from lookback window
        feat = extract_features(df_1h, t_start, t_end, holder_df, top10_df)
        if feat is None:
            t_start += pd.Timedelta(hours=SLIDE_STEP_HOURS)
            continue

        # Historical context features
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
            feat["hist_ath"] = feat["mcap_end"]
            feat["hist_ath_ratio"] = 1.0
            feat["token_age_hours"] = 0
            feat["hist_return_total"] = 0
            feat["hist_volatility"] = 0
            feat["hist_pump_count"] = 0

        # Label from future window
        mcap_now = feat["mcap_end"]
        if mcap_now < MCAP_THRESHOLD * 0.5:  # skip if mcap too low
            t_start += pd.Timedelta(hours=SLIDE_STEP_HOURS)
            continue

        future = df_1h[(df_1h["datetime"] > t_end) & (df_1h["datetime"] <= t_predict_end)]
        if len(future) < 2:
            t_start += pd.Timedelta(hours=SLIDE_STEP_HOURS)
            continue

        future_max = future["mcap"].max()
        future_min = future["mcap"].min()

        max_mult = future_max / max(mcap_now, 1)
        min_ratio = future_min / max(mcap_now, 1)

        if max_mult >= PUMP_THRESHOLD:
            label = "pump"
        elif min_ratio <= DUMP_THRESHOLD:
            label = "dump"
        else:
            label = "flat"

        samples.append({
            "address": addr, "symbol": symbol,
            "t_start": t_start.isoformat(), "t_end": t_end.isoformat(),
            "label": label, "future_max_mult": round(max_mult, 2),
            **feat,
        })

        t_start += pd.Timedelta(hours=SLIDE_STEP_HOURS)

    return samples


# ── Training ─────────────────────────────────────────────────────────────────


def main():
    tokens = load_radar_tokens()
    print(f"Loaded {len(tokens)} radar tokens")

    all_samples = []
    n_tokens_used = 0

    for i, token in enumerate(tokens):
        samples = generate_samples(token)
        if samples:
            all_samples.extend(samples)
            n_tokens_used += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(tokens)}, {len(all_samples)} samples, {n_tokens_used} tokens")

    df = pd.DataFrame(all_samples)
    print(f"\nDataset: {len(df)} samples from {n_tokens_used} tokens")
    print(f"Labels: {df['label'].value_counts().to_dict()}")

    if len(df) < 50:
        print("Too few samples.")
        return

    feature_cols = [c for c in df.columns if c not in
                    ("address", "symbol", "t_start", "t_end", "label", "future_max_mult")]
    X = df[feature_cols].values.astype(float)
    groups = df["address"].values

    # ── 3-class: dump / flat / pump ──
    label_map = {"dump": 0, "flat": 1, "pump": 2}
    y = df["label"].map(label_map).values

    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"GroupKFold: {n_splits} splits, {len(unique_groups)} unique tokens")

    print("\n" + "=" * 60)
    print(f"3-CLASS: dump / flat / pump  (pump={PUMP_THRESHOLD}x in {PREDICT_HOURS}h)")
    print("=" * 60)

    gkf = GroupKFold(n_splits=n_splits)
    y_pred = np.full(len(y), -1)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, random_state=42, verbose=-1,
            class_weight="balanced",
        )
        model.fit(X[train_idx], y[train_idx])
        y_pred[val_idx] = model.predict(X[val_idx])

    names = ["dump", "flat", "pump"]
    print(f"\n{classification_report(y, y_pred, target_names=names, zero_division=0)}")

    # ── Binary: pump vs rest ──
    print("=" * 60)
    print("BINARY: pump vs not-pump")
    print("=" * 60)

    y_bin = (y == 2).astype(int)
    y_pred_bin = np.full(len(y_bin), -1)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_bin, groups)):
        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, random_state=42, verbose=-1,
            is_unbalance=True,
        )
        model.fit(X[train_idx], y_bin[train_idx])
        y_pred_bin[val_idx] = model.predict(X[val_idx])

    print(f"\n{classification_report(y_bin, y_pred_bin, target_names=['not_pump', 'pump'], zero_division=0)}")

    # Feature importance
    model_full = lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        num_leaves=31, random_state=42, verbose=-1, is_unbalance=True,
    )
    model_full.fit(X, y_bin)
    importances = sorted(zip(feature_cols, model_full.feature_importances_),
                         key=lambda x: x[1], reverse=True)
    print("Top features (pump detection):")
    for name, imp in importances[:12]:
        print(f"  {name:>25s}: {imp}")

    # Known tokens
    print("\n--- Known Token Windows ---")
    known = ["PUNCH", "GOYIM", "CAPTCHA", "WAR", "GORK"]
    for sym in known:
        mask = df["symbol"] == sym
        if not mask.any():
            continue
        sub = df[mask]
        n_pump_actual = (sub["label"] == "pump").sum()
        n_pump_pred = (y_pred_bin[mask.values] == 1).sum()
        print(f"  {sym:>10s}: {len(sub):>3d} windows, "
              f"actual_pump={n_pump_actual}, pred_pump={n_pump_pred}, "
              f"max_mult={sub['future_max_mult'].max():.1f}x")

    print("\nDone.")


if __name__ == "__main__":
    main()
