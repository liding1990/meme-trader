"""State Classifier — classifies token market states into favorable/unfavorable clusters.

Offline: builds point cloud from historical data, clusters in 3D feature space, labels clusters.
Online: classifies a new observation by nearest centroid lookup.
"""

import csv
import json
import os
import glob
import pickle
import sys

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.preprocessing import StandardScaler

from scalper import config

DATA_DIR = "data"
RADAR_CSV = os.path.join(DATA_DIR, "radar_tokens.csv")
MCAP_THRESHOLD = 50_000
MAX_HOURS = 90 * 24  # 3 months
FUTURE_HOURS = 4


# ── Data Loading (adapted from app_tsne.py, no Streamlit dependency) ─────────

def _load_radar_metadata() -> dict:
    meta = {}
    if not os.path.isfile(RADAR_CSV):
        return meta
    with open(RADAR_CSV) as f:
        for row in csv.reader(f):
            if len(row) >= 4:
                meta[row[0]] = {"name": row[2], "symbol": row[3]}
    return meta


def _load_from_cache(address: str):
    data_dir = os.path.join(DATA_DIR, address)
    if not os.path.isdir(data_dir):
        return None, None
    mcap_files = sorted([
        f for f in glob.glob(os.path.join(data_dir, "token_mcap_candles_[0-9]*.json"))
        if "5m" not in os.path.basename(f)
    ])
    trend_files = sorted(glob.glob(os.path.join(data_dir, "token_trends_*.json")))
    if not mcap_files or not trend_files:
        return None, None
    with open(mcap_files[-1]) as f:
        mcap_data = json.load(f)
    with open(trend_files[-1]) as f:
        trend_data = json.load(f)
    return mcap_data, trend_data


def _build_trajectory(mcap_data: dict, trend_data: dict):
    if not mcap_data or not trend_data:
        return None
    data_section = mcap_data.get("data")
    if not data_section:
        return None
    candles = data_section.get("list", [])
    if not candles:
        return None

    mcap_df = pd.DataFrame(candles)
    mcap_df["datetime"] = pd.to_datetime(mcap_df["time"].astype(int), unit="ms")
    mcap_df["mcap"] = mcap_df["close"].astype(float)
    mcap_df = mcap_df[["datetime", "mcap"]].sort_values("datetime").reset_index(drop=True)

    trend_section = trend_data.get("data")
    if not trend_section:
        return None
    trends = trend_section.get("trends", {})
    holder_series = trends.get("holder_count", [])
    if not holder_series:
        return None

    holder_df = pd.DataFrame(holder_series)
    holder_df["datetime"] = pd.to_datetime(holder_df["timestamp"].astype(int), unit="s")
    holder_df["holders"] = holder_df["value"].astype(float)
    holder_df = holder_df[["datetime", "holders"]].sort_values("datetime")
    holder_df = holder_df.set_index("datetime").resample("1h").last().dropna().reset_index()

    mcap_df["datetime"] = mcap_df["datetime"].dt.floor("h")
    mcap_df = mcap_df.groupby("datetime", as_index=False).last()
    holder_df["datetime"] = holder_df["datetime"].dt.floor("h")
    holder_df = holder_df.groupby("datetime", as_index=False).last()

    merged = pd.merge(mcap_df, holder_df, on="datetime", how="inner")
    if merged.empty:
        merged = pd.merge(mcap_df, holder_df, on="datetime", how="outer").sort_values("datetime")
        merged["mcap"] = merged["mcap"].ffill()
        merged["holders"] = merged["holders"].ffill()
        merged = merged.dropna()

    if merged.empty:
        return None

    merged = merged.sort_values("datetime").reset_index(drop=True)

    above = merged[merged["mcap"] >= MCAP_THRESHOLD]
    if above.empty:
        return None

    start_idx = above.index[0]
    merged = merged.loc[start_idx:].reset_index(drop=True)

    t0 = merged["datetime"].iloc[0]
    merged["hours"] = (merged["datetime"] - t0).dt.total_seconds() / 3600
    merged = merged[merged["hours"] <= MAX_HOURS].reset_index(drop=True)

    if len(merged) < FUTURE_HOURS + 1:
        return None

    return merged


# ── Point Cloud ──────────────────────────────────────────────────────────────

def _load_5m_with_holders(address: str) -> pd.DataFrame | None:
    """Load 5m candles + Moralis 5m holders, aligned and merged."""
    data_dir = os.path.join(DATA_DIR, address)
    if not os.path.isdir(data_dir):
        return None

    # 5m mcap candles
    mcap_files = sorted(glob.glob(os.path.join(data_dir, "token_mcap_candles_5m_*.json")))
    if not mcap_files:
        return None
    with open(mcap_files[-1]) as f:
        mcap_data = json.load(f)
    data_section = mcap_data.get("data")
    if not data_section:
        return None
    candles = data_section.get("list", [])
    if not candles:
        return None

    mcap_df = pd.DataFrame(candles)
    mcap_df["datetime"] = pd.to_datetime(mcap_df["time"].astype(int), unit="ms")
    mcap_df["mcap"] = mcap_df["close"].astype(float)
    if "amount" in mcap_df.columns:
        mcap_df["volume"] = mcap_df["amount"].astype(float)
    else:
        mcap_df["volume"] = 0.0
    mcap_df = mcap_df[["datetime", "mcap", "volume"]].sort_values("datetime").reset_index(drop=True)

    # Moralis 5m holders
    holders_path = os.path.join(data_dir, "moralis_holders_5m.json")
    if not os.path.isfile(holders_path):
        return None
    with open(holders_path) as f:
        holders_data = json.load(f)
    if not holders_data:
        return None

    holder_df = pd.DataFrame(holders_data)
    holder_df["datetime"] = pd.to_datetime(holder_df["timestamp"], utc=True).dt.tz_localize(None)
    holder_df["holders"] = holder_df["totalHolders"].astype(float)
    holder_df["net_holder_change"] = holder_df["netHolderChange"].astype(float)
    # Whale+shark flow: in minus out
    holder_df["whale_shark_in"] = holder_df["holdersIn"].apply(
        lambda x: x.get("whales", 0) + x.get("sharks", 0) if isinstance(x, dict) else 0
    )
    holder_df["whale_shark_out"] = holder_df["holdersOut"].apply(
        lambda x: x.get("whales", 0) + x.get("sharks", 0) if isinstance(x, dict) else 0
    )
    holder_df["whale_shark_flow"] = holder_df["whale_shark_in"] - holder_df["whale_shark_out"]
    holder_df = holder_df[["datetime", "holders", "net_holder_change", "whale_shark_flow"]].sort_values("datetime").reset_index(drop=True)

    # Align to 5-min floor
    mcap_df["datetime"] = mcap_df["datetime"].dt.floor("5min")
    holder_df["datetime"] = holder_df["datetime"].dt.floor("5min")
    mcap_df = mcap_df.groupby("datetime", as_index=False).agg({"mcap": "last", "volume": "sum"})
    holder_df = holder_df.groupby("datetime", as_index=False).agg({
        "holders": "last",
        "net_holder_change": "sum",
        "whale_shark_flow": "sum",
    })

    merged = pd.merge(mcap_df, holder_df, on="datetime", how="inner")
    if merged.empty:
        merged = pd.merge(mcap_df, holder_df, on="datetime", how="outer").sort_values("datetime")
        merged["mcap"] = merged["mcap"].ffill()
        merged["holders"] = merged["holders"].ffill()
        merged["net_holder_change"] = merged["net_holder_change"].fillna(0)
        merged["whale_shark_flow"] = merged["whale_shark_flow"].fillna(0)
        merged = merged.dropna(subset=["mcap", "holders"])

    if len(merged) < FUTURE_BARS_5M + 1:
        return None

    merged = merged.sort_values("datetime").reset_index(drop=True)

    # Find origin: first bar where mcap >= 50K
    above = merged[merged["mcap"] >= MCAP_THRESHOLD]
    if above.empty:
        return None
    start_idx = above.index[0]
    merged = merged.loc[start_idx:].reset_index(drop=True)

    t0 = merged["datetime"].iloc[0]
    merged["hours"] = (merged["datetime"] - t0).dt.total_seconds() / 3600

    return merged


# Future return window: 4 hours = 48 five-minute bars
FUTURE_BARS_5M = 48


def build_point_cloud() -> pd.DataFrame:
    """Build enriched feature matrix from 5-minute data (mcap + Moralis holders)."""
    meta = _load_radar_metadata()
    radar_addrs = set(meta.keys())

    rows = []
    tokens_used = 0
    roc_period = 6  # 30 min = 6 bars

    for addr in radar_addrs:
        df = _load_5m_with_holders(addr)
        if df is None:
            continue

        mcap_arr = df["mcap"].values
        hours_arr = df["hours"].values
        holders_arr = df["holders"].values
        net_change_arr = df["net_holder_change"].values
        whale_flow_arr = df["whale_shark_flow"].values
        n = len(df)
        tokens_used += 1

        # Precompute rolling features
        # ROC 30m
        roc_30m = np.full(n, 0.0)
        for i in range(roc_period, n):
            if mcap_arr[i - roc_period] > 0:
                roc_30m[i] = (mcap_arr[i] / mcap_arr[i - roc_period] - 1) * 100

        # Sample every 6 bars (30 min) for denser point cloud
        for i in range(roc_period, n - FUTURE_BARS_5M):
            if i % 6 != 0:
                continue
            current_mcap = mcap_arr[i]
            future_mcap = mcap_arr[i + FUTURE_BARS_5M]
            if current_mcap <= 0 or holders_arr[i] <= 0:
                continue
            future_return = (future_mcap - current_mcap) / current_mcap * 100
            future_return_clipped = np.clip(future_return, -100, 500)

            # Holder growth rate: sum of net changes over last 6 bars / total holders
            holder_growth = net_change_arr[max(0, i - 5):i + 1].sum() / holders_arr[i] * 100

            # Whale+shark net flow over last 6 bars
            whale_flow_30m = whale_flow_arr[max(0, i - 5):i + 1].sum()

            rows.append({
                "address": addr,
                "log_mcap": np.log1p(current_mcap),
                "hours": hours_arr[i],
                "log_holders": np.log1p(holders_arr[i]),
                "holder_growth_rate": holder_growth,
                "whale_shark_flow": float(whale_flow_30m),
                "mcap_roc_30m": roc_30m[i],
                "future_return_4h": future_return_clipped,
            })

    print(f"  Tokens with 5m+holders data: {tokens_used}")
    return pd.DataFrame(rows)


def build_point_cloud_legacy() -> pd.DataFrame:
    """Fallback: build from GMGN hourly data (for tokens without Moralis data)."""
    meta = _load_radar_metadata()
    radar_addrs = set(meta.keys())

    rows = []
    for addr in radar_addrs:
        mcap_data, trend_data = _load_from_cache(addr)
        if mcap_data is None:
            continue
        df = _build_trajectory(mcap_data, trend_data)
        if df is None:
            continue

        mcap_arr = df["mcap"].values
        hours_arr = df["hours"].values
        holders_arr = df["holders"].values
        n = len(df)

        for i in range(n - FUTURE_HOURS):
            current_mcap = mcap_arr[i]
            future_mcap = mcap_arr[i + FUTURE_HOURS]
            if current_mcap <= 0:
                continue
            future_return = (future_mcap - current_mcap) / current_mcap * 100
            future_return_clipped = np.clip(future_return, -100, 500)

            rows.append({
                "address": addr,
                "log_mcap": np.log1p(current_mcap),
                "hours": hours_arr[i],
                "log_holders": np.log1p(holders_arr[i]),
                "future_return_4h": future_return_clipped,
            })

    return pd.DataFrame(rows)


# ── Training ─────────────────────────────────────────────────────────────────

CLASSIFY_FEATURES = ["log_mcap", "hours", "log_holders", "holder_growth_rate", "whale_shark_flow", "mcap_roc_30m"]


def train_classifier(point_cloud: pd.DataFrame) -> dict:
    """Train state classifier on 3D feature space, label clusters by future return.

    Returns artifacts dict with: scaler, centroids, cluster_stats, favorable_ids.
    """
    X = point_cloud[CLASSIFY_FEATURES].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clusterer = HDBSCAN(min_cluster_size=100, min_samples=5, cluster_selection_method="leaf")
    labels = clusterer.fit_predict(X_scaled)

    point_cloud = point_cloud.copy()
    point_cloud["cluster"] = labels

    # Compute per-cluster stats
    cluster_stats = {}
    cluster_ids = sorted(set(labels))
    for cid in cluster_ids:
        if cid == -1:
            continue
        mask = labels == cid
        returns = point_cloud.loc[mask, "future_return_4h"]
        cluster_stats[cid] = {
            "count": int(mask.sum()),
            "median_return": float(returns.median()),
            "mean_return": float(returns.mean()),
            "win_rate": float((returns > 0).mean()),
            "centroid": X_scaled[mask].mean(axis=0).tolist(),
            "radius_std": float(np.linalg.norm(X_scaled[mask] - X_scaled[mask].mean(axis=0), axis=1).std()),
        }

    # Label favorable clusters
    favorable_ids = set()
    for cid, stats in cluster_stats.items():
        if (stats["win_rate"] >= config.FAVORABLE_MIN_WIN_RATE
                and stats["median_return"] >= config.FAVORABLE_MIN_MEDIAN_RETURN):
            favorable_ids.add(cid)

    artifacts = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "cluster_stats": cluster_stats,
        "favorable_ids": sorted(favorable_ids),
        "feature_names": CLASSIFY_FEATURES,
    }

    return artifacts


def save_artifacts(artifacts: dict, path: str = None):
    path = path or config.CLASSIFIER_MODEL_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"Saved classifier to {path}")


# ── Online Inference ─────────────────────────────────────────────────────────

class StateClassifier:
    """Classifies a token's current market state into a cluster."""

    def __init__(self, artifacts: dict):
        self.scaler_mean = np.array(artifacts["scaler_mean"])
        self.scaler_scale = np.array(artifacts["scaler_scale"])
        self.cluster_stats = artifacts["cluster_stats"]
        self.favorable_ids = set(artifacts["favorable_ids"])

        # Precompute centroid matrix for fast lookup
        self._cids = []
        self._centroids = []
        self._radii = []
        for cid_str, stats in self.cluster_stats.items():
            cid = int(cid_str) if isinstance(cid_str, str) else cid_str
            self._cids.append(cid)
            self._centroids.append(stats["centroid"])
            self._radii.append(stats["radius_std"])
        self._centroids = np.array(self._centroids)
        self._radii = np.array(self._radii)

    @classmethod
    def load(cls, path: str = None) -> "StateClassifier":
        path = path or config.CLASSIFIER_MODEL_PATH
        with open(path, "rb") as f:
            artifacts = pickle.load(f)
        return cls(artifacts)

    def classify(self, log_mcap: float, hours: float, log_holders: float,
                 holder_growth_rate: float = 0.0, whale_shark_flow: float = 0.0,
                 mcap_roc_30m: float = 0.0):
        """Classify a single observation.

        Returns (cluster_id, is_favorable, confidence).
        confidence is 1.0 - (distance / max_allowed_distance), clipped to [0, 1].
        Returns (None, False, 0.0) if no cluster matches within threshold.
        """
        x = np.array([log_mcap, hours, log_holders, holder_growth_rate, whale_shark_flow, mcap_roc_30m])
        x_scaled = (x - self.scaler_mean) / self.scaler_scale

        # Distance to each centroid
        dists = np.linalg.norm(self._centroids - x_scaled, axis=1)

        best_idx = dists.argmin()
        best_dist = dists[best_idx]
        best_cid = self._cids[best_idx]
        best_radius = self._radii[best_idx]

        max_dist = config.CLASSIFY_MAX_DISTANCE_SIGMA * best_radius
        if max_dist <= 0 or best_dist > max_dist:
            return None, False, 0.0

        confidence = max(0.0, 1.0 - best_dist / max_dist)
        is_favorable = best_cid in self.favorable_ids

        return best_cid, is_favorable, confidence

    def get_cluster_stats(self, cluster_id: int) -> dict:
        key = str(cluster_id) if str(cluster_id) in self.cluster_stats else cluster_id
        return self.cluster_stats.get(key, {})


# ── CLI ──────────────────────────────────────────────────────────────────────

def cmd_train():
    """Train the state classifier and save artifacts."""
    print("Building point cloud from 5m data (Moralis holders + enriched features)...")
    pc = build_point_cloud()
    if len(pc) < 100:
        print(f"  ERROR: Only {len(pc)} points. Need more Moralis holder data.")
        print("  Run: python -m scalper.fetch_holders")
        return
    print(f"  Points: {len(pc):,} from {pc['address'].nunique()} tokens")

    print("Training classifier...")
    artifacts = train_classifier(pc)

    n_clusters = len(artifacts["cluster_stats"])
    n_favorable = len(artifacts["favorable_ids"])
    print(f"  Clusters: {n_clusters}")
    print(f"  Favorable: {n_favorable}")

    print("\nCluster details:")
    for cid in sorted(artifacts["cluster_stats"], key=lambda x: int(x)):
        s = artifacts["cluster_stats"][cid]
        fav = "*" if int(cid) in set(artifacts["favorable_ids"]) else " "
        print(f"  {fav} C{str(cid):>3s}: n={s['count']:>5d}  "
              f"win_rate={s['win_rate']:.1%}  "
              f"med_ret={s['median_return']:+.1f}%  "
              f"mean_ret={s['mean_return']:+.1f}%")

    save_artifacts(artifacts)
    print("\nDone.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        cmd_train()
    else:
        print("Usage: python -m scalper.state_classifier train")
