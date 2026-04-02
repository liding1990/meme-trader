"""Early Warning System — Train & evaluate on first-24h features.

Usage:
    cd /Users/liding/code/meme-trader
    source .venv/bin/activate
    python early_warning/train.py
"""

import csv
import json
import os
import glob
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb


DATA_DIR = "data"
RADAR_CSV = os.path.join(DATA_DIR, "radar_tokens.csv")
MCAP_THRESHOLD = 100_000
TARGET_HOURS = 24
MAX_HOURS_FULL = 90 * 24  # for label generation (full history)


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_radar_tokens():
    tokens = []
    with open(RADAR_CSV) as f:
        for row in csv.reader(f):
            if len(row) >= 4:
                tokens.append({"address": row[0], "chain": row[1],
                                "name": row[2], "symbol": row[3]})
    return tokens


def load_5m_candles(address: str) -> pd.DataFrame | None:
    data_dir = os.path.join(DATA_DIR, address)
    # Prefer consolidated full file
    full_path = os.path.join(data_dir, "token_mcap_candles_5m_full.json")
    paths_to_try = [full_path]
    # Fallback: latest snapshot
    m_files = sorted(glob.glob(os.path.join(data_dir, "token_mcap_candles_5m_*.json")))
    if m_files:
        paths_to_try.append(m_files[-1])

    for fpath in paths_to_try:
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
            df["open_price"] = df["open"].astype(float)
            df["high_price"] = df["high"].astype(float)
            df["low_price"] = df["low"].astype(float)
            return df.sort_values("datetime").reset_index(drop=True)
        except Exception:
            continue
    return None


def load_1h_candles(address: str) -> pd.DataFrame | None:
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


def load_moralis_holders(address: str) -> pd.DataFrame | None:
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


def load_gmgn_holders(address: str) -> pd.DataFrame | None:
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
        for col in ["buyers", "sellers", "buys", "sells", "buy_volume", "sell_volume", "transactions"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df.sort_values("datetime").reset_index(drop=True)
    except Exception:
        return None


def load_top10_concentration(address: str) -> pd.DataFrame | None:
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


# ── Label Generation ─────────────────────────────────────────────────────────


def compute_label(address: str) -> str | None:
    """Compute outcome label from full history.

    Labels: scam, rug, mid, runner
    """
    df_1h = load_1h_candles(address)
    if df_1h is None or len(df_1h) < 2:
        return None

    mcap = df_1h["mcap"].values
    ath = mcap.max()
    ath_idx = np.argmax(mcap)
    current = mcap[-1]

    if ath < MCAP_THRESHOLD:
        return None  # never reached threshold

    # Hours to ATH
    times = df_1h["time_ms"].values
    hours_to_ath = (times[ath_idx] - times[0]) / 3600000

    # Current as % of ATH
    current_pct = current / max(ath, 1)

    # Holders (best available)
    holder_df = load_moralis_holders(address)
    if holder_df is None:
        holder_df = load_gmgn_holders(address)
    max_holders = holder_df["holders"].max() if holder_df is not None and len(holder_df) > 0 else 0

    # Scam: ATH in first 2 hours, collapsed, low holders
    if hours_to_ath <= 2 and current_pct < 0.01 and max_holders < 2000:
        return "scam"

    # Rug: collapsed regardless of timing
    if current_pct < 0.05:
        return "rug"

    # Runner: high ATH + significant community
    if ath >= 5_000_000 and max_holders >= 5000:
        return "runner"

    return "mid"


# ── Feature Extraction ───────────────────────────────────────────────────────


def extract_features(address: str) -> dict | None:
    """Extract features from first 24h of 5-minute data after $100K crossing."""
    df_5m = load_5m_candles(address)
    if df_5m is None:
        return None

    # Find $100K crossing
    above = df_5m[df_5m["mcap"] >= MCAP_THRESHOLD]
    if above.empty:
        return None

    t0 = above["datetime"].iloc[0]
    window = df_5m[(df_5m["datetime"] >= t0) &
                   (df_5m["datetime"] <= t0 + pd.Timedelta(hours=TARGET_HOURS))].copy()

    if len(window) < 10:  # need minimum data
        return None

    mcap = window["mcap"].values
    volume = window["volume"].values
    hours = (window["datetime"] - t0).dt.total_seconds().values / 3600

    # Load holder data
    holder_df = load_moralis_holders(address)
    if holder_df is None:
        holder_df = load_gmgn_holders(address)

    # Merge holders onto the 5m window (hourly granularity, forward fill)
    holders_at_points = None
    if holder_df is not None and len(holder_df) >= 2:
        holder_df = holder_df[(holder_df["datetime"] >= t0 - pd.Timedelta(hours=1)) &
                              (holder_df["datetime"] <= t0 + pd.Timedelta(hours=TARGET_HOURS + 1))]
        if len(holder_df) >= 2:
            # Resample to match 5m window
            holder_df = holder_df.set_index("datetime").resample("5min").ffill().reset_index()
            merged = pd.merge_asof(window[["datetime"]], holder_df, on="datetime", direction="backward")
            if "holders" in merged.columns:
                holders_at_points = merged["holders"].values

    feat = {}

    # A. Price momentum
    feat["mcap_start"] = mcap[0]
    feat["mcap_max_24h"] = mcap.max()
    feat["mcap_end_24h"] = mcap[-1]
    feat["mcap_return_total"] = (mcap[-1] - mcap[0]) / max(mcap[0], 1)

    # Returns at sub-windows
    for h in [4, 8, 12]:
        mask = hours <= h
        if mask.sum() >= 2:
            sub = mcap[mask]
            feat[f"mcap_return_{h}h"] = (sub[-1] - sub[0]) / max(sub[0], 1)
        else:
            feat[f"mcap_return_{h}h"] = 0

    # Volatility
    returns_5m = np.diff(mcap) / np.maximum(mcap[:-1], 1)
    feat["volatility"] = np.std(returns_5m) if len(returns_5m) > 1 else 0

    # Max drawdown
    running_max = np.maximum.accumulate(mcap)
    drawdowns = (mcap - running_max) / np.maximum(running_max, 1)
    feat["max_drawdown"] = drawdowns.min()

    # Time to peak (within 24h window)
    peak_idx = np.argmax(mcap)
    feat["hours_to_peak"] = hours[peak_idx]

    # Price acceleration: first 12h vs last 12h
    mid = len(mcap) // 2
    if mid > 1:
        first_half_return = (mcap[mid] - mcap[0]) / max(mcap[0], 1)
        second_half_return = (mcap[-1] - mcap[mid]) / max(mcap[mid], 1)
        feat["price_acceleration"] = second_half_return - first_half_return
    else:
        feat["price_acceleration"] = 0

    # B. Volume features
    feat["volume_total_24h"] = volume.sum()
    feat["volume_mean_5m"] = volume.mean()
    feat["volume_std_5m"] = volume.std() if len(volume) > 1 else 0

    # Volume trend: first half vs second half
    if mid > 0:
        v1 = volume[:mid].mean()
        v2 = volume[mid:].mean()
        feat["volume_trend"] = (v2 - v1) / max(v1, 1)
    else:
        feat["volume_trend"] = 0

    # Volume concentration: max hour vs total
    hourly_vol = pd.Series(volume, index=window["datetime"]).resample("1h").sum()
    if len(hourly_vol) > 0 and hourly_vol.sum() > 0:
        feat["volume_concentration"] = hourly_vol.max() / hourly_vol.sum()
    else:
        feat["volume_concentration"] = 0

    # C. Holder features
    if holders_at_points is not None and len(holders_at_points) >= 2:
        valid_h = holders_at_points[~np.isnan(holders_at_points)]
        if len(valid_h) >= 2:
            feat["holders_start"] = valid_h[0]
            feat["holders_end"] = valid_h[-1]
            feat["holder_growth_rate"] = (valid_h[-1] - valid_h[0]) / max(valid_h[0], 1)

            # Holders at sub-windows
            for h in [4, 8, 12]:
                mask = hours <= h
                sub_h = holders_at_points[mask]
                sub_h = sub_h[~np.isnan(sub_h)]
                feat[f"holders_at_{h}h"] = sub_h[-1] if len(sub_h) > 0 else valid_h[0]

            # Holder acceleration
            mid_h = len(valid_h) // 2
            if mid_h > 0:
                h1 = valid_h[mid_h] - valid_h[0]
                h2 = valid_h[-1] - valid_h[mid_h]
                feat["holder_acceleration"] = h2 - h1
            else:
                feat["holder_acceleration"] = 0
        else:
            _fill_holder_defaults(feat)
    else:
        _fill_holder_defaults(feat)

    # D. Structure features
    if feat.get("holders_start", 0) > 0:
        feat["mcap_per_holder_start"] = mcap[0] / feat["holders_start"]
    else:
        feat["mcap_per_holder_start"] = mcap[0]

    if feat.get("holders_end", 0) > 0:
        feat["mcap_per_holder_end"] = mcap[-1] / feat["holders_end"]
    else:
        feat["mcap_per_holder_end"] = mcap[-1]

    # MCap-holder correlation
    if holders_at_points is not None:
        valid_mask = ~np.isnan(holders_at_points)
        if valid_mask.sum() >= 5:
            corr = np.corrcoef(mcap[valid_mask], holders_at_points[valid_mask])[0, 1]
            feat["mcap_holder_corr"] = corr if not np.isnan(corr) else 0
        else:
            feat["mcap_holder_corr"] = 0
    else:
        feat["mcap_holder_corr"] = 0

    # Volume/holder ratio
    if feat.get("holders_end", 0) > 0:
        feat["volume_holder_ratio"] = feat["volume_total_24h"] / feat["holders_end"]
    else:
        feat["volume_holder_ratio"] = feat["volume_total_24h"]

    # E. Top 10 holder concentration (only if data available — no zero-filling)
    top10_df = load_top10_concentration(address)
    if top10_df is not None and len(top10_df) >= 2:
        t10_window = top10_df[(top10_df["datetime"] >= t0 - pd.Timedelta(hours=1)) &
                              (top10_df["datetime"] <= t0 + pd.Timedelta(hours=TARGET_HOURS + 1))]
        if len(t10_window) >= 1:
            feat["top10_pct_start"] = t10_window["top10_pct"].iloc[0]
            feat["top10_pct_end"] = t10_window["top10_pct"].iloc[-1]
            feat["top10_pct_change"] = t10_window["top10_pct"].iloc[-1] - t10_window["top10_pct"].iloc[0]
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
    for h in [4, 8, 12]:
        feat[f"holders_at_{h}h"] = 0
    feat["holder_acceleration"] = 0


def _fill_codex_defaults(feat):
    feat["buy_sell_volume_ratio"] = 0
    feat["buyer_seller_ratio"] = 0
    feat["buy_sell_tx_ratio"] = 0
    feat["total_transactions_24h"] = 0
    feat["buy_pressure_trend"] = 0


def _fill_top10_defaults(feat):
    feat["top10_pct_start"] = 0
    feat["top10_pct_end"] = 0
    feat["top10_pct_change"] = 0


# ── Training Pipeline ────────────────────────────────────────────────────────


def main():
    tokens = load_radar_tokens()
    print(f"Loaded {len(tokens)} radar tokens")

    # Step 1: Generate labels and features
    rows = []
    label_counts = {}
    skip_reasons = {}

    for token in tokens:
        addr = token["address"]
        symbol = token["symbol"]

        label = compute_label(addr)
        if label is None:
            skip_reasons["no_label"] = skip_reasons.get("no_label", 0) + 1
            continue

        features = extract_features(addr)
        if features is None:
            skip_reasons["no_features"] = skip_reasons.get("no_features", 0) + 1
            continue

        label_counts[label] = label_counts.get(label, 0) + 1
        rows.append({"address": addr, "symbol": symbol, "label": label, **features})

    print(f"\nDataset: {len(rows)} tokens")
    print(f"Labels: {label_counts}")
    print(f"Skipped: {skip_reasons}")

    if len(rows) < 20:
        print("Too few samples, aborting.")
        return

    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c not in ("address", "symbol", "label")]

    X = df[feature_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Binary labels for Stage 1: scam detection
    y_scam = (df["label"] == "scam").astype(int).values

    # Multi-class for Stage 2: runner detection (exclude scam)
    label_map = {"rug": 0, "mid": 1, "runner": 2}
    mask_non_scam = df["label"] != "scam"
    X_stage2 = X[mask_non_scam.values]
    y_runner = df.loc[mask_non_scam, "label"].map(label_map).values

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Feature names: {feature_cols}")

    # Step 2: Stage 1 — Scam detection (5-fold CV)
    print("\n" + "=" * 60)
    print("STAGE 1: SCAM DETECTION (binary)")
    print("=" * 60)

    if y_scam.sum() >= 5:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred_scam = np.zeros(len(y_scam))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_scam)):
            model = lgb.LGBMClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                num_leaves=15, random_state=42, verbose=-1,
                is_unbalance=True,
            )
            model.fit(X[train_idx], y_scam[train_idx])
            y_pred_scam[val_idx] = model.predict(X[val_idx])

        print("\nConfusion Matrix (rows=actual, cols=predicted):")
        print("             pred_clean  pred_scam")
        cm = confusion_matrix(y_scam, y_pred_scam)
        print(f"  actual_clean  {cm[0][0]:>6d}  {cm[0][1]:>9d}")
        print(f"  actual_scam   {cm[1][0]:>6d}  {cm[1][1]:>9d}")
        print(f"\n{classification_report(y_scam, y_pred_scam, target_names=['clean', 'scam'])}")

        # Feature importance (train on full data for importance)
        model_full = lgb.LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            num_leaves=15, random_state=42, verbose=-1, is_unbalance=True,
        )
        model_full.fit(X, y_scam)
        importances = sorted(zip(feature_cols, model_full.feature_importances_),
                             key=lambda x: x[1], reverse=True)
        print("Top features (scam detection):")
        for name, imp in importances[:10]:
            print(f"  {name:>30s}: {imp}")
    else:
        print(f"  Too few scam samples ({y_scam.sum()}), skipping Stage 1")

    # Step 3: Stage 2 — Runner detection (5-fold CV, exclude scam)
    print("\n" + "=" * 60)
    print("STAGE 2: RUNNER DETECTION (rug/mid/runner)")
    print("=" * 60)

    non_scam_labels = df.loc[mask_non_scam, "label"].values
    non_scam_symbols = df.loc[mask_non_scam, "symbol"].values
    print(f"Samples: {len(y_runner)}")
    print(f"Distribution: { {k: (y_runner == v).sum() for k, v in label_map.items()} }")

    if len(y_runner) >= 20 and len(set(y_runner)) >= 2:
        n_splits = min(5, min(np.bincount(y_runner)))
        n_splits = max(2, n_splits)
        skf2 = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_pred_runner = np.full(len(y_runner), -1)

        for fold, (train_idx, val_idx) in enumerate(skf2.split(X_stage2, y_runner)):
            model = lgb.LGBMClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                num_leaves=15, random_state=42, verbose=-1,
                class_weight="balanced",
            )
            model.fit(X_stage2[train_idx], y_runner[train_idx])
            y_pred_runner[val_idx] = model.predict(X_stage2[val_idx])

        names = ["rug", "mid", "runner"]
        print(f"\n{classification_report(y_runner, y_pred_runner, target_names=names)}")

        # Feature importance
        model_full2 = lgb.LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            num_leaves=15, random_state=42, verbose=-1, class_weight="balanced",
        )
        model_full2.fit(X_stage2, y_runner)
        importances2 = sorted(zip(feature_cols, model_full2.feature_importances_),
                              key=lambda x: x[1], reverse=True)
        print("Top features (runner detection):")
        for name, imp in importances2[:10]:
            print(f"  {name:>30s}: {imp}")

        # Show predictions for known tokens
        print("\n--- Known Token Predictions ---")
        known = ["PUNCH", "GOYIM", "CAPTCHA", "TRUMP", "WAR", "GORK"]
        for i, sym in enumerate(non_scam_symbols):
            if sym in known:
                actual = non_scam_labels[i]
                pred = names[y_pred_runner[i]] if y_pred_runner[i] >= 0 else "?"
                print(f"  {sym:>10s}: actual={actual:>6s}, predicted={pred:>6s}")
    else:
        print("  Too few samples for Stage 2")

    print("\nDone.")


if __name__ == "__main__":
    main()
