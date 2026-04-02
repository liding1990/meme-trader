"""Early Warning System v2 — Sliding window, any-time prediction.

Instead of fixed "first 24h" features, generates samples at multiple
time points for each token. Each sample uses a 24h lookback window
and predicts whether mcap will 5x+ in the next 48h.

Usage:
    cd /Users/liding/code/meme-trader
    source .venv/bin/activate
    python early_warning/train_v2.py
"""

import csv
import json
import os
import glob
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb


DATA_DIR = "data"
RADAR_CSV = os.path.join(DATA_DIR, "radar_tokens.csv")
MCAP_THRESHOLD = 100_000
LOOKBACK_HOURS = 24       # feature window
PREDICT_HOURS = 48        # prediction horizon
SLIDE_STEP_HOURS = 12     # step between windows
PUMP_THRESHOLD = 5.0      # 5x = runner signal
DUMP_THRESHOLD = 0.5      # drop to 50% = dump signal


# ── Data Loading (same as v1) ────────────────────────────────────────────────


def load_radar_tokens():
    tokens = []
    with open(RADAR_CSV) as f:
        for row in csv.reader(f):
            if len(row) >= 4:
                tokens.append({"address": row[0], "chain": row[1],
                                "name": row[2], "symbol": row[3]})
    return tokens


def load_5m_candles(address):
    data_dir = os.path.join(DATA_DIR, address)
    full_path = os.path.join(data_dir, "token_mcap_candles_5m_full.json")
    paths = [full_path]
    m_files = sorted(glob.glob(os.path.join(data_dir, "token_mcap_candles_5m_*.json")))
    if m_files:
        paths.append(m_files[-1])
    for fpath in paths:
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath) as f:
                data = json.load(f)
            candles = (data or {}).get("data", {}).get("list", [])
            if not candles:
                continue
            df = pd.DataFrame(candles)
            df["time_ms"] = df["time"].astype(int)
            df["datetime"] = pd.to_datetime(df["time_ms"], unit="ms")
            df["mcap"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(float)
            return df.sort_values("datetime").reset_index(drop=True)
        except Exception:
            continue
    return None


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
        return df.sort_values("datetime").reset_index(drop=True)
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
        holder_series = (data or {}).get("data", {}).get("trends", {}).get("holder_count", [])
        if not holder_series:
            return None
        df = pd.DataFrame(holder_series)
        df["datetime"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        df["holders"] = df["value"].astype(float)
        return df[["datetime", "holders"]].sort_values("datetime").reset_index(drop=True)
    except Exception:
        return None


def load_top10_concentration(address):
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


# ── Feature Extraction (window-based) ────────────────────────────────────────


def extract_window_features(df_5m: pd.DataFrame, t_start: pd.Timestamp,
                            t_end: pd.Timestamp, holder_df: pd.DataFrame = None,
                            top10_df: pd.DataFrame = None) -> dict | None:
    """Extract features from a 24h window of 5-minute data."""
    window = df_5m[(df_5m["datetime"] >= t_start) & (df_5m["datetime"] <= t_end)].copy()

    if len(window) < 10:
        return None

    mcap = window["mcap"].values
    volume = window["volume"].values
    hours = (window["datetime"] - t_start).dt.total_seconds().values / 3600

    # Merge holders
    holders_at_points = None
    if holder_df is not None and len(holder_df) >= 2:
        h_window = holder_df[(holder_df["datetime"] >= t_start - pd.Timedelta(hours=1)) &
                             (holder_df["datetime"] <= t_end + pd.Timedelta(hours=1))]
        if len(h_window) >= 2:
            h_resampled = h_window.set_index("datetime").resample("5min").ffill().reset_index()
            merged = pd.merge_asof(window[["datetime"]], h_resampled, on="datetime", direction="backward")
            if "holders" in merged.columns:
                holders_at_points = merged["holders"].values

    feat = {}

    # A. Price momentum
    feat["mcap_start"] = mcap[0]
    feat["mcap_max"] = mcap.max()
    feat["mcap_end"] = mcap[-1]
    feat["mcap_return_total"] = (mcap[-1] - mcap[0]) / max(mcap[0], 1)

    for h in [4, 8, 12]:
        mask = hours <= h
        if mask.sum() >= 2:
            sub = mcap[mask]
            feat[f"mcap_return_{h}h"] = (sub[-1] - sub[0]) / max(sub[0], 1)
        else:
            feat[f"mcap_return_{h}h"] = 0

    returns_5m = np.diff(mcap) / np.maximum(mcap[:-1], 1)
    feat["volatility"] = np.std(returns_5m) if len(returns_5m) > 1 else 0

    running_max = np.maximum.accumulate(mcap)
    drawdowns = (mcap - running_max) / np.maximum(running_max, 1)
    feat["max_drawdown"] = drawdowns.min()

    peak_idx = np.argmax(mcap)
    feat["hours_to_peak"] = hours[peak_idx]

    mid = len(mcap) // 2
    if mid > 1:
        r1 = (mcap[mid] - mcap[0]) / max(mcap[0], 1)
        r2 = (mcap[-1] - mcap[mid]) / max(mcap[mid], 1)
        feat["price_acceleration"] = r2 - r1
    else:
        feat["price_acceleration"] = 0

    # B. Volume
    feat["volume_total"] = volume.sum()
    feat["volume_mean"] = volume.mean()
    feat["volume_std"] = volume.std() if len(volume) > 1 else 0

    if mid > 0:
        v1 = volume[:mid].mean()
        v2 = volume[mid:].mean()
        feat["volume_trend"] = (v2 - v1) / max(v1, 1)
    else:
        feat["volume_trend"] = 0

    hourly_vol = pd.Series(volume, index=window["datetime"]).resample("1h").sum()
    if len(hourly_vol) > 0 and hourly_vol.sum() > 0:
        feat["volume_concentration"] = hourly_vol.max() / hourly_vol.sum()
    else:
        feat["volume_concentration"] = 0

    # C. Holders
    if holders_at_points is not None and len(holders_at_points) >= 2:
        valid_h = holders_at_points[~np.isnan(holders_at_points)]
        if len(valid_h) >= 2:
            feat["holders_start"] = valid_h[0]
            feat["holders_end"] = valid_h[-1]
            feat["holder_growth_rate"] = (valid_h[-1] - valid_h[0]) / max(valid_h[0], 1)
            mid_h = len(valid_h) // 2
            if mid_h > 0:
                feat["holder_acceleration"] = (valid_h[-1] - valid_h[mid_h]) - (valid_h[mid_h] - valid_h[0])
            else:
                feat["holder_acceleration"] = 0
        else:
            _fill_holder_defaults(feat)
    else:
        _fill_holder_defaults(feat)

    # D. Structure
    h_start = feat.get("holders_start", 0)
    h_end = feat.get("holders_end", 0)
    feat["mcap_per_holder_start"] = mcap[0] / max(h_start, 1) if h_start > 0 else mcap[0]
    feat["mcap_per_holder_end"] = mcap[-1] / max(h_end, 1) if h_end > 0 else mcap[-1]

    if holders_at_points is not None:
        valid_mask = ~np.isnan(holders_at_points)
        if valid_mask.sum() >= 5:
            corr = np.corrcoef(mcap[valid_mask], holders_at_points[valid_mask])[0, 1]
            feat["mcap_holder_corr"] = corr if not np.isnan(corr) else 0
        else:
            feat["mcap_holder_corr"] = 0
    else:
        feat["mcap_holder_corr"] = 0

    feat["volume_holder_ratio"] = feat["volume_total"] / max(h_end, 1) if h_end > 0 else feat["volume_total"]

    # E. Top 10 concentration
    if top10_df is not None and len(top10_df) >= 1:
        t10_w = top10_df[(top10_df["datetime"] >= t_start - pd.Timedelta(hours=1)) &
                         (top10_df["datetime"] <= t_end + pd.Timedelta(hours=1))]
        if len(t10_w) >= 1:
            feat["top10_pct_start"] = t10_w["top10_pct"].iloc[0]
            feat["top10_pct_end"] = t10_w["top10_pct"].iloc[-1]
            feat["top10_pct_change"] = t10_w["top10_pct"].iloc[-1] - t10_w["top10_pct"].iloc[0]
        else:
            feat["top10_pct_start"] = np.nan
            feat["top10_pct_end"] = np.nan
            feat["top10_pct_change"] = np.nan
    else:
        feat["top10_pct_start"] = np.nan
        feat["top10_pct_end"] = np.nan
        feat["top10_pct_change"] = np.nan

    return feat


def _fill_holder_defaults(feat):
    feat["holders_start"] = 0
    feat["holders_end"] = 0
    feat["holder_growth_rate"] = 0
    feat["holder_acceleration"] = 0


# ── Sliding Window Label + Feature Generation ────────────────────────────────


def compute_window_label(df_5m: pd.DataFrame, df_1h: pd.DataFrame,
                         t_end: pd.Timestamp) -> str | None:
    """Compute label: what happens in the next 48h after t_end?

    Labels:
        pump   — mcap reaches 5x of t_end mcap within 48h
        dump   — mcap drops to <50% of t_end mcap and never recovers in 48h
        flat   — neither pump nor dump
    """
    # Get mcap at t_end from 5m data
    at_end = df_5m[df_5m["datetime"] <= t_end]
    if at_end.empty:
        return None
    mcap_at_end = at_end["mcap"].iloc[-1]
    if mcap_at_end < MCAP_THRESHOLD:
        return None

    # Future window: use 1h data (more likely to have coverage)
    future_start = t_end
    future_end = t_end + pd.Timedelta(hours=PREDICT_HOURS)

    future = df_1h[(df_1h["datetime"] > future_start) & (df_1h["datetime"] <= future_end)]
    if len(future) < 2:
        # Fallback to 5m data
        future = df_5m[(df_5m["datetime"] > future_start) & (df_5m["datetime"] <= future_end)]
    if len(future) < 2:
        return None

    future_mcap = future["mcap"].values
    future_max = future_mcap.max()
    future_min = future_mcap.min()

    max_multiple = future_max / max(mcap_at_end, 1)
    min_ratio = future_min / max(mcap_at_end, 1)

    if max_multiple >= PUMP_THRESHOLD:
        return "pump"
    elif min_ratio <= DUMP_THRESHOLD:
        return "dump"
    else:
        return "flat"


def generate_samples(token: dict) -> list[dict]:
    """Generate sliding window samples for a single token."""
    addr = token["address"]
    symbol = token["symbol"]

    df_5m = load_5m_candles(addr)
    df_1h = load_1h_candles(addr)
    if df_5m is None or df_1h is None:
        return []

    holder_df = load_moralis_holders(addr)
    if holder_df is None:
        holder_df = load_gmgn_holders(addr)

    top10_df = load_top10_concentration(addr)

    # Find when mcap first crosses $100K
    above = df_5m[df_5m["mcap"] >= MCAP_THRESHOLD]
    if above.empty:
        return []

    t_origin = above["datetime"].iloc[0]
    t_data_end = df_5m["datetime"].iloc[-1]

    # Generate windows: slide from origin, step by SLIDE_STEP_HOURS
    samples = []
    t_start = t_origin

    while True:
        t_end = t_start + pd.Timedelta(hours=LOOKBACK_HOURS)

        # Need room for prediction window after t_end
        if t_end + pd.Timedelta(hours=PREDICT_HOURS) > t_data_end + pd.Timedelta(hours=1):
            break  # not enough future data

        # Extract features
        feat = extract_window_features(df_5m, t_start, t_end, holder_df, top10_df)
        if feat is None:
            t_start += pd.Timedelta(hours=SLIDE_STEP_HOURS)
            continue

        # Compute label
        label = compute_window_label(df_5m, df_1h, t_end)
        if label is None:
            t_start += pd.Timedelta(hours=SLIDE_STEP_HOURS)
            continue

        samples.append({
            "address": addr,
            "symbol": symbol,
            "window_start": t_start.isoformat(),
            "window_end": t_end.isoformat(),
            "label": label,
            **feat,
        })

        t_start += pd.Timedelta(hours=SLIDE_STEP_HOURS)

    return samples


# ── Training Pipeline ────────────────────────────────────────────────────────


def main():
    tokens = load_radar_tokens()
    print(f"Loaded {len(tokens)} radar tokens")

    # Step 1: Generate all sliding window samples
    all_samples = []
    tokens_with_samples = 0

    for i, token in enumerate(tokens):
        samples = generate_samples(token)
        if samples:
            all_samples.extend(samples)
            tokens_with_samples += 1
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(tokens)} tokens, {len(all_samples)} samples so far...")

    df = pd.DataFrame(all_samples)
    print(f"\nDataset: {len(df)} samples from {tokens_with_samples} tokens")
    print(f"Labels: {df['label'].value_counts().to_dict()}")

    if len(df) < 30:
        print("Too few samples, aborting.")
        return

    feature_cols = [c for c in df.columns if c not in
                    ("address", "symbol", "window_start", "window_end", "label")]

    X = df[feature_cols].values.astype(float)
    # Keep NaN for LightGBM (it handles them natively)

    label_map = {"dump": 0, "flat": 1, "pump": 2}
    y = df["label"].map(label_map).values
    groups = df["address"].values  # for GroupKFold

    print(f"Features: {len(feature_cols)}")

    # Step 2: GroupKFold CV (no token leakage between folds)
    print("\n" + "=" * 60)
    print("3-CLASS PREDICTION: dump / flat / pump")
    print(f"(lookback={LOOKBACK_HOURS}h, predict={PREDICT_HOURS}h, pump={PUMP_THRESHOLD}x)")
    print("=" * 60)

    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))

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
    print(f"\n{classification_report(y, y_pred, target_names=names)}")

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(f"{'':>12s}  {'pred_dump':>10s}  {'pred_flat':>10s}  {'pred_pump':>10s}")
    for i, name in enumerate(names):
        print(f"  {name:>10s}  {cm[i][0]:>10d}  {cm[i][1]:>10d}  {cm[i][2]:>10d}")

    # Feature importance
    model_full = lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        num_leaves=31, random_state=42, verbose=-1,
        class_weight="balanced",
    )
    model_full.fit(X, y)
    importances = sorted(zip(feature_cols, model_full.feature_importances_),
                         key=lambda x: x[1], reverse=True)
    print("\nTop features:")
    for name, imp in importances[:12]:
        print(f"  {name:>30s}: {imp}")

    # Step 3: Binary pump detection (most actionable)
    print("\n" + "=" * 60)
    print("BINARY: pump vs not-pump")
    print("=" * 60)

    y_binary = (y == 2).astype(int)
    y_pred_binary = np.full(len(y_binary), -1)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_binary, groups)):
        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, random_state=42, verbose=-1,
            is_unbalance=True,
        )
        model.fit(X[train_idx], y_binary[train_idx])
        y_pred_binary[val_idx] = model.predict(X[val_idx])

    print(f"\n{classification_report(y_binary, y_pred_binary, target_names=['not_pump', 'pump'])}")

    # Per-token pump detection: show known tokens
    print("--- Known Token Samples ---")
    known = ["PUNCH", "GOYIM", "CAPTCHA", "TRUMP", "WAR", "GORK"]
    for sym in known:
        mask = df["symbol"] == sym
        if not mask.any():
            continue
        sub = df[mask]
        sub_pred = y_pred[mask.values]
        sub_labels = sub["label"].values
        n_pump = (sub_labels == "pump").sum()
        n_pump_pred = (sub_pred == 2).sum()
        print(f"  {sym:>10s}: {len(sub)} windows, "
              f"actual pump={n_pump}, predicted pump={n_pump_pred}")
        # Show individual windows
        for _, row in sub.iterrows():
            idx = sub.index.get_loc(row.name) if isinstance(row.name, int) else 0
            pred = names[y_pred[row.name]] if y_pred[row.name] >= 0 else "?"
            print(f"    {row['window_start'][:16]} → {row['label']:>4s} (pred: {pred})")

    print("\nDone.")


if __name__ == "__main__":
    main()
