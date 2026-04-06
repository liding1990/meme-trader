"""Phase 3: Predict which outcome cluster a token belongs to.

Given a token address, fetches its data, computes outcome features,
assigns it to the nearest KMeans cluster, and shows:
- Which label it matches (Organic Runner, Pump-Dump, etc.)
- Similar tokens in that cluster (runners highlighted)
- Median performance statistics of the cluster

Usage:
    python outcome_cluster_predict.py <address> [--chain sol|base]
"""

import argparse
import csv
import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd


DATA_DIR = "data"
OUTPUT_DIR = "outcome_cluster_data"
MCAP_THRESHOLD = 100_000


def load_model():
    pkl_path = os.path.join(OUTPUT_DIR, "model.pkl")
    csv_path = os.path.join(OUTPUT_DIR, "cluster_assignments.csv")
    if not os.path.isfile(pkl_path) or not os.path.isfile(csv_path):
        print("错误：未找到聚类模型。请先运行 python outcome_cluster.py")
        sys.exit(1)
    with open(pkl_path, "rb") as f:
        model_data = pickle.load(f)
    df = pd.read_csv(csv_path)
    return model_data, df


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


def load_holders(address):
    path = os.path.join(DATA_DIR, address, "moralis_holders_1h.json")
    if os.path.isfile(path):
        try:
            with open(path) as f:
                data = json.load(f)
            if data:
                df = pd.DataFrame(data)
                df["datetime"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
                df["holders"] = df["totalHolders"].astype(float)
                return df[["datetime", "holders"]].sort_values("datetime").reset_index(drop=True)
        except Exception:
            pass

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


def compute_features(address):
    """Compute the 6 outcome features for a token."""
    df = load_1h_candles(address)
    if df is None or len(df) < 2:
        return None

    mcap = df["mcap"].values
    ath = mcap.max()
    ath_idx = np.argmax(mcap)
    hours = df["time_ms"].values
    hours_to_ath = max((hours[ath_idx] - hours[0]) / 3600000, 1)

    price_roc = (ath - MCAP_THRESHOLD) / hours_to_ath
    vol_to_ath = df["volume"].iloc[:ath_idx + 1].sum()
    volume_roc = vol_to_ath / hours_to_ath

    holder_df = load_holders(address)
    if holder_df is not None and len(holder_df) >= 2:
        holders = holder_df["holders"].values
        max_holders = float(holders.max())
        ath_time = df["datetime"].iloc[ath_idx]
        h_before = holder_df[holder_df["datetime"] <= ath_time]
        holders_at_ath = h_before["holders"].iloc[-1] if len(h_before) > 0 else holders[0]
        holder_roc = (holders_at_ath - holders[0]) / hours_to_ath
        hours_post_ath = max((hours[-1] - hours[ath_idx]) / 3600000, 1)
        h_after = holder_df[holder_df["datetime"] >= ath_time]
        holders_end = h_after["holders"].iloc[-1] if len(h_after) > 0 else holders[-1]
        holder_decay_roc = (holders_end - holders_at_ath) / hours_post_ath
    else:
        max_holders = 0
        holder_roc = 0
        holder_decay_roc = 0

    return {
        "ath": ath,
        "price_roc": price_roc,
        "volume_roc": volume_roc,
        "holder_roc": holder_roc,
        "holder_decay_roc": holder_decay_roc,
        "max_holders": max_holders,
    }


def build_feature_vector(feat):
    return [
        np.log1p(max(feat["ath"], 0)),
        np.log1p(max(feat["price_roc"], 0)),
        np.log1p(max(feat["volume_roc"], 0)),
        np.log1p(max(feat["holder_roc"], 0)),
        -np.log1p(max(-feat["holder_decay_roc"], 0)),
        np.log1p(max(feat["max_holders"], 0)),
    ]


def predict_token(address, chain="sol"):
    """Predict cluster for a token and show analysis."""
    model_data, cluster_df = load_model()

    scaler = model_data["scaler"]
    km = model_data["kmeans"]
    cluster_label_map = model_data["cluster_label_map"]
    cluster_order = model_data["cluster_order"]
    rank_map = {cid: i + 1 for i, cid in enumerate(cluster_order)}

    # Try loading from cache first
    feat = compute_features(address)

    # If not cached, fetch via GMGN
    if feat is None:
        print("本地无数据，从 GMGN 获取...", file=sys.stderr)
        sys.path.insert(0, ".")
        from gmgn_api import fetch_token_data
        _, loaded_data = fetch_token_data(chain, address)

        mcap_data = loaded_data.get("token_mcap_candles", {})
        candles = mcap_data.get("data", {}).get("list", [])
        if not candles:
            print("错误：无法获取蜡烛图数据")
            return

        df = pd.DataFrame(candles)
        df["time_ms"] = df["time"].astype(int)
        df["datetime"] = pd.to_datetime(df["time_ms"], unit="ms")
        df["mcap"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df = df.sort_values("datetime").reset_index(drop=True)

        mcap = df["mcap"].values
        ath = mcap.max()
        ath_idx = np.argmax(mcap)
        hours = df["time_ms"].values
        hours_to_ath = max((hours[ath_idx] - hours[0]) / 3600000, 1)

        # Try to get holders from trends
        trend_data = loaded_data.get("token_trends", {})
        holder_series = (trend_data or {}).get("data", {}).get("trends", {}).get("holder_count", [])
        if holder_series:
            hdf = pd.DataFrame(holder_series)
            hdf["holders"] = hdf["value"].astype(float)
            holders = hdf["holders"].values
            max_holders = float(holders.max())
            holder_roc = (holders[min(ath_idx, len(holders)-1)] - holders[0]) / hours_to_ath
            holder_decay_roc = (holders[-1] - holders[min(ath_idx, len(holders)-1)]) / max((hours[-1] - hours[ath_idx]) / 3600000, 1)
        else:
            max_holders = 0
            holder_roc = 0
            holder_decay_roc = 0

        feat = {
            "ath": ath,
            "price_roc": (ath - MCAP_THRESHOLD) / hours_to_ath,
            "volume_roc": df["volume"].iloc[:ath_idx + 1].sum() / hours_to_ath,
            "holder_roc": holder_roc,
            "holder_decay_roc": holder_decay_roc,
            "max_holders": max_holders,
        }

    # Predict cluster
    vec = np.array([build_feature_vector(feat)])
    vec_scaled = scaler.transform(vec)
    cid = int(km.predict(vec_scaled)[0])
    label = cluster_label_map.get(cid, "Unknown")
    rank = rank_map.get(cid, 0)

    # Distance to all clusters
    dists = {}
    for c in cluster_order:
        dists[c] = float(np.linalg.norm(vec_scaled[0] - km.cluster_centers_[c]))

    # Get cluster members
    cluster_members = cluster_df[cluster_df["cluster_id"] == cid].copy()

    # Identify runners in this cluster (ATH >= $5M and max_holders >= 5000)
    runners = cluster_members[(cluster_members["ath"] >= 5_000_000) &
                               (cluster_members["max_holders"] >= 5000)]

    # Compute distances from query to each member for "most similar"
    member_vecs = np.array([build_feature_vector(row) for _, row in cluster_members.iterrows()])
    member_scaled = scaler.transform(member_vecs)
    member_dists = np.sqrt(((member_scaled - vec_scaled[0]) ** 2).sum(axis=1))
    cluster_members = cluster_members.copy()
    cluster_members["distance"] = member_dists
    similar = cluster_members.nsmallest(10, "distance")

    # ── Print Report ──
    from outcome_cluster import CLUSTER_LABELS

    bold = "\033[1m"
    reset = "\033[0m"
    green = "\033[92m"
    red = "\033[91m"
    yellow = "\033[93m"

    print(f"\n{'=' * 70}")
    print(f"{bold}OUTCOME CLUSTER 分析报告{reset}")
    print(f"{'=' * 70}")
    print(f"地址: {address}")

    print(f"\n{bold}┌────────────────────────────────────────────────┐{reset}")
    print(f"{bold}│  {yellow}#{rank} {label}{reset}{bold}  │{reset}")
    print(f"{bold}└────────────────────────────────────────────────┘{reset}")
    print(f"  {CLUSTER_LABELS.get(label, '')}")

    print(f"\n{bold}Token 当前特征:{reset}")
    print(f"  ATH:              ${feat['ath']:>14,.0f}")
    print(f"  价格增速:         ${feat['price_roc']:>14,.0f}/h")
    print(f"  成交量增速:       ${feat['volume_roc']:>14,.0f}/h")
    print(f"  持币人增速:        {feat['holder_roc']:>14,.1f}/h")
    print(f"  持币人衰减:        {feat['holder_decay_roc']:>14,.1f}/h")
    print(f"  最大持币人:        {feat['max_holders']:>14,.0f}")

    print(f"\n{bold}聚类统计（{len(cluster_members)} 个 token）:{reset}")
    print(f"  ATH 中位数:       ${cluster_members['ath'].median():>14,.0f}")
    print(f"  ATH 范围:         ${cluster_members['ath'].min():>12,.0f} — ${cluster_members['ath'].max():>12,.0f}")
    print(f"  持币人中位数:      {cluster_members['max_holders'].median():>14,.0f}")

    if len(runners) > 0:
        print(f"\n{bold}{green}该聚类中的 Runner（ATH >= $5M & holders >= 5K）:{reset}")
        for _, r in runners.nlargest(10, "ath").iterrows():
            print(f"  {green}{r['symbol']:>12s}{reset}  ATH ${r['ath']:>12,.0f}  "
                  f"holders {r['max_holders']:>8,.0f}  price_roc ${r['price_roc']:>10,.0f}/h")
        print(f"\n  Runner 占比: {len(runners)}/{len(cluster_members)} ({len(runners)/len(cluster_members)*100:.0f}%)")
        print(f"  Runner 中位 ATH: ${runners['ath'].median():>12,.0f}")
    else:
        print(f"\n{red}  该聚类中没有 Runner（ATH >= $5M 且 holders >= 5K）{reset}")

    print(f"\n{bold}最相似的历史 Token:{reset}")
    print(f"  {'Token':>12s}  {'ATH':>14s}  {'持币人':>8s}  {'价格增速':>12s}  {'距离':>6s}")
    print(f"  {'-'*60}")
    for _, s in similar.iterrows():
        is_runner = s["ath"] >= 5_000_000 and s["max_holders"] >= 5000
        color = green if is_runner else ""
        end = reset if is_runner else ""
        print(f"  {color}{s['symbol']:>12s}{end}  ${s['ath']:>13,.0f}  {s['max_holders']:>8,.0f}  "
              f"${s['price_roc']:>11,.0f}/h  {s['distance']:>6.2f}")

    # Performance summary
    all_ath = cluster_members["ath"].values
    print(f"\n{bold}该类 token 的中位数表现:{reset}")
    print(f"  中位 ATH:         ${np.median(all_ath):>14,.0f}")
    print(f"  75 分位 ATH:      ${np.percentile(all_ath, 75):>14,.0f}")
    print(f"  25 分位 ATH:      ${np.percentile(all_ath, 25):>14,.0f}")
    print(f"  中位持币人:        {cluster_members['max_holders'].median():>14,.0f}")

    # Distances to other clusters
    print(f"\n{bold}到各聚类的距离:{reset}")
    for c in cluster_order:
        d = dists[c]
        lbl = cluster_label_map.get(c, "?")
        r = rank_map[c]
        marker = " ← 当前" if c == cid else ""
        print(f"  #{r} {lbl:.<20s} {d:>6.2f}{marker}")


def main():
    parser = argparse.ArgumentParser(description="Outcome Cluster 预测")
    parser.add_argument("address", help="Token 合约地址")
    parser.add_argument("--chain", default="sol", choices=["sol", "base"])
    args = parser.parse_args()

    predict_token(args.address, args.chain)


if __name__ == "__main__":
    main()
