"""Baseline Clustering v2 — Outcome-based clustering with standardized time windows.

Data window per token:
  Start:  First hour mcap >= $100K
  Peak:   ATH (all-time high mcap)
  End:    First hour mcap drops to ATH × 0.3 (70% decline), or data end

Filters:
  - Token created after Jan 1, 2026
  - ATH >= $100K

11 features per token:
  Rise:   rise_hours, price_roc, volume_roc, holder_roc
  Peak:   ath, holders_at_ath
  Decay:  decay_hours, holder_decay_roc, price_decay_roc
  Time:   total_hours, rise_pct

Usage:
    python baseline_cluster_v2.py              # Run clustering
    python baseline_cluster_v2.py --show       # Show existing results
    python baseline_cluster_v2.py --predict <address>  # Classify a token
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


DATA_DIR = "data"
RADAR_CSV = os.path.join(DATA_DIR, "radar_tokens.csv")
MCAP_THRESHOLD = 100_000
DECAY_PCT = 0.70            # token lifecycle ends when mcap drops 70% from ATH
CREATED_AFTER_MS = 1735689600000  # Jan 1, 2026 in milliseconds
N_CLUSTERS = 6
OUTPUT_DIR = "baseline_cluster_v2_data"

FEATURE_NAMES = [
    "rise_hours", "price_roc", "volume_roc", "holder_roc",
    "ath", "holders_at_ath",
    "decay_hours", "holder_decay_roc", "price_decay_roc",
    "total_hours", "rise_pct",
]


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
        if not candles or len(candles) < 2:
            return None
        df = pd.DataFrame(candles)
        df["time_ms"] = df["time"].astype(int)
        df["datetime"] = pd.to_datetime(df["time_ms"], unit="ms").astype("datetime64[s]")
        df["mcap"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    except Exception:
        return None


def load_holders(address):
    # Moralis first
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
    # GMGN fallback
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


# ── Feature Extraction ───────────────────────────────────────────────────────


def compute_features(address):
    """Extract 11 features from standardized time window.

    Window: first $100K → ATH → ATH × 0.3 (or data end)
    """
    df = load_1h_candles(address)
    if df is None or len(df) < 2:
        return None

    # Filter: token must be from 2026+
    earliest_ts = df["time_ms"].min()
    if earliest_ts < CREATED_AFTER_MS:
        return None

    mcap = df["mcap"].values
    volume = df["volume"].values
    times = df["time_ms"].values
    n = len(mcap)

    # ATH check
    ath = mcap.max()
    if ath < MCAP_THRESHOLD:
        return None

    ath_idx = np.argmax(mcap)

    # Start: first $100K crossing
    start_idx = None
    for i in range(n):
        if mcap[i] >= MCAP_THRESHOLD:
            start_idx = i
            break
    if start_idx is None:
        return None

    # End: first time mcap drops to ATH × 0.3 after ATH
    decay_threshold = ath * (1 - DECAY_PCT)
    end_idx = n - 1  # default: data end
    for i in range(ath_idx, n):
        if mcap[i] <= decay_threshold:
            end_idx = i
            break

    # ── Rise phase: start → ATH ──
    rise_hours = max((times[ath_idx] - times[start_idx]) / 3600000, 1)
    price_roc = (ath - MCAP_THRESHOLD) / rise_hours

    vol_rise = volume[start_idx:ath_idx + 1].sum()
    volume_roc = vol_rise / rise_hours

    # Holders at start and ATH
    holder_df = load_holders(address)
    holders_at_start = 0
    holders_at_ath = 0
    holders_at_end = 0
    max_holders = 0

    if holder_df is not None and len(holder_df) >= 2:
        # Merge holders onto candle timeline
        h_merged = pd.merge_asof(
            df[["datetime"]].iloc[start_idx:end_idx + 1],
            holder_df, on="datetime", direction="backward"
        )
        if "holders" in h_merged.columns and h_merged["holders"].notna().sum() > 0:
            h_vals = h_merged["holders"].ffill().bfill().values
            holders_at_start = h_vals[0]
            ath_offset = ath_idx - start_idx
            if ath_offset < len(h_vals):
                holders_at_ath = h_vals[ath_offset]
            else:
                holders_at_ath = h_vals[-1]
            holders_at_end = h_vals[-1]
            max_holders = h_vals.max()

    holder_roc = (holders_at_ath - holders_at_start) / rise_hours

    # ── Decay phase: ATH → end ──
    decay_hours = max((times[end_idx] - times[ath_idx]) / 3600000, 1)
    holder_decay_roc = (holders_at_end - holders_at_ath) / decay_hours
    price_decay_roc = (mcap[end_idx] - ath) / decay_hours

    # ── Time features ──
    total_hours = max((times[end_idx] - times[start_idx]) / 3600000, 1)
    rise_pct = rise_hours / total_hours

    return {
        "rise_hours": rise_hours,
        "price_roc": price_roc,
        "volume_roc": volume_roc,
        "holder_roc": holder_roc,
        "ath": ath,
        "holders_at_ath": holders_at_ath,
        "decay_hours": decay_hours,
        "holder_decay_roc": holder_decay_roc,
        "price_decay_roc": price_decay_roc,
        "total_hours": total_hours,
        "rise_pct": rise_pct,
        # Extra for display (not used in clustering)
        "max_holders": max_holders,
        "mcap_at_end": mcap[end_idx],
        "start_idx": start_idx,
        "ath_idx": ath_idx,
        "end_idx": end_idx,
    }


# ── Clustering ───────────────────────────────────────────────────────────────


def build_feature_vector(feat):
    """Convert feature dict to log-scaled vector for KMeans."""
    return [
        np.log1p(max(feat["rise_hours"], 0)),
        np.log1p(max(feat["price_roc"], 0)),
        np.log1p(max(feat["volume_roc"], 0)),
        np.log1p(max(feat["holder_roc"], 0)),
        np.log1p(max(feat["ath"], 0)),
        np.log1p(max(feat["holders_at_ath"], 0)),
        np.log1p(max(feat["decay_hours"], 0)),
        -np.log1p(max(-feat["holder_decay_roc"], 0)),  # negative = faster decay
        -np.log1p(max(-feat["price_decay_roc"], 0)),
        np.log1p(max(feat["total_hours"], 0)),
        feat["rise_pct"],
    ]


def run_clustering(tokens):
    """Run KMeans on outcome features."""
    rows = []
    for i, token in enumerate(tokens):
        feat = compute_features(token["address"])
        if feat is None:
            continue
        rows.append({
            "address": token["address"], "symbol": token["symbol"],
            "name": token["name"], "chain": token["chain"], **feat
        })
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(tokens)}, {len(rows)} 有效")

    df = pd.DataFrame(rows)
    print(f"\n有效样本: {len(df)}")

    X = np.array([build_feature_vector(row) for _, row in df.iterrows()])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    df["cluster_id"] = labels

    # Rank by median ATH (descending)
    cluster_ath = df.groupby("cluster_id")["ath"].median()
    cluster_order = cluster_ath.sort_values(ascending=False).index.tolist()
    rank_map = {cid: i + 1 for i, cid in enumerate(cluster_order)}
    df["rank"] = df["cluster_id"].map(rank_map)

    return df, scaler, km, cluster_order


def print_clusters(df, cluster_order):
    """Print detailed cluster descriptions for human review."""
    rank_map = {cid: i + 1 for i, cid in enumerate(cluster_order)}

    print(f"\n{'=' * 80}")
    print(f"Baseline Clustering v2 ({N_CLUSTERS} clusters, {len(df)} tokens)")
    print(f"数据窗口: $100K起点 → ATH → 跌70%截止 | 仅2026年后token")
    print(f"{'=' * 80}")

    for cid in cluster_order:
        sub = df[df["cluster_id"] == cid]
        rank = rank_map[cid]

        print(f"\n{'─' * 70}")
        print(f"Cluster #{rank} ({len(sub)} tokens)")
        print(f"{'─' * 70}")

        # Key stats
        print(f"  ATH 中位:         ${sub['ath'].median():>14,.0f}   范围: ${sub['ath'].min():>12,.0f} — ${sub['ath'].max():>12,.0f}")
        print(f"  上升时长 中位:     {sub['rise_hours'].median():>14.0f}h   范围: {sub['rise_hours'].min():.0f}h — {sub['rise_hours'].max():.0f}h")
        print(f"  衰减时长 中位:     {sub['decay_hours'].median():>14.0f}h   范围: {sub['decay_hours'].min():.0f}h — {sub['decay_hours'].max():.0f}h")
        print(f"  总时长 中位:       {sub['total_hours'].median():>14.0f}h")
        print(f"  上升占比 中位:     {sub['rise_pct'].median():>14.1%}")
        print(f"  价格增速 中位:     ${sub['price_roc'].median():>14,.0f}/h")
        print(f"  Volume增速 中位:   ${sub['volume_roc'].median():>14,.0f}/h")
        print(f"  Holder增速 中位:    {sub['holder_roc'].median():>14.1f}/h")
        print(f"  Holder衰减 中位:    {sub['holder_decay_roc'].median():>14.1f}/h")
        print(f"  ATH时Holders中位:  {sub['holders_at_ath'].median():>14,.0f}")
        print(f"  最大Holders中位:   {sub['max_holders'].median():>14,.0f}")

        # Behavioral pattern summary
        med = {k: sub[k].median() for k in FEATURE_NAMES}
        patterns = []
        if med["rise_hours"] < 5:
            patterns.append("极速拉盘(上升<5h)")
        elif med["rise_hours"] < 24:
            patterns.append("快速上升(上升<24h)")
        elif med["rise_hours"] < 100:
            patterns.append("中速上升")
        else:
            patterns.append("慢速积累(上升>100h)")

        if med["holders_at_ath"] < 1000:
            patterns.append("低holders(<1K)")
        elif med["holders_at_ath"] < 5000:
            patterns.append("中holders")
        else:
            patterns.append("高holders(>5K)")

        if med["decay_hours"] < 5:
            patterns.append("极速崩盘")
        elif med["decay_hours"] < 24:
            patterns.append("快速衰减")
        else:
            patterns.append("缓慢衰减")

        if med["holder_roc"] < 1:
            patterns.append("无真实用户增长")
        elif med["holder_roc"] > 50:
            patterns.append("holder涌入")

        print(f"  行为模式:         {' | '.join(patterns)}")

        # Top tokens
        top = sub.nlargest(8, "ath")
        print(f"  代表Token:")
        for _, row in top.iterrows():
            print(f"    {row['symbol']:>12s}  ATH ${row['ath']:>12,.0f}  "
                  f"上升{row['rise_hours']:.0f}h  衰减{row['decay_hours']:.0f}h  "
                  f"holders@ATH={row['holders_at_ath']:,.0f}")


def save_results(df, scaler, km, cluster_order):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, "clusters.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "model.pkl"), "wb") as f:
        pickle.dump({"scaler": scaler, "kmeans": km,
                      "cluster_order": cluster_order}, f)
    print(f"\n结果已保存到 {OUTPUT_DIR}/")


# ── Prediction ───────────────────────────────────────────────────────────────


def predict_token(address):
    """Classify a single token."""
    pkl_path = os.path.join(OUTPUT_DIR, "model.pkl")
    csv_path = os.path.join(OUTPUT_DIR, "clusters.csv")
    if not os.path.isfile(pkl_path):
        print("请先运行 python baseline_cluster_v2.py 生成聚类模型")
        return

    with open(pkl_path, "rb") as f:
        model = pickle.load(f)
    cluster_df = pd.read_csv(csv_path)

    feat = compute_features(address)
    if feat is None:
        # Try fetching from GMGN
        print("本地无数据，尝试从 GMGN 获取...")
        from gmgn_api import fetch_token_data
        fetch_token_data("sol", address)
        feat = compute_features(address)

    if feat is None:
        print("无法计算特征（数据不足或 ATH < $100K）")
        return

    vec = np.array([build_feature_vector(feat)])
    vec_scaled = model["scaler"].transform(vec)
    cid = int(model["kmeans"].predict(vec_scaled)[0])

    cluster_order = model["cluster_order"]
    rank_map = {c: i + 1 for i, c in enumerate(cluster_order)}
    rank = rank_map.get(cid, 0)

    # Same cluster tokens
    same = cluster_df[cluster_df["cluster_id"] == cid]

    print(f"\n{'=' * 60}")
    print(f"Token: {address[:20]}...")
    print(f"Cluster: #{rank} ({len(same)} tokens)")
    print(f"{'=' * 60}")

    print(f"\n当前 Token 特征:")
    print(f"  ATH:              ${feat['ath']:>14,.0f}")
    print(f"  上升时长:          {feat['rise_hours']:>14.0f}h")
    print(f"  衰减时长:          {feat['decay_hours']:>14.0f}h")
    print(f"  价格增速:         ${feat['price_roc']:>14,.0f}/h")
    print(f"  Holder增速:        {feat['holder_roc']:>14.1f}/h")
    print(f"  Holder衰减:        {feat['holder_decay_roc']:>14.1f}/h")
    print(f"  ATH时Holders:      {feat['holders_at_ath']:>14,.0f}")
    print(f"  上升占比:          {feat['rise_pct']:>14.1%}")

    print(f"\n同 Cluster 统计:")
    print(f"  ATH 中位:         ${same['ath'].median():>14,.0f}")
    print(f"  上升时长 中位:     {same['rise_hours'].median():>14.0f}h")
    print(f"  衰减时长 中位:     {same['decay_hours'].median():>14.0f}h")

    # Nearest tokens
    scaler = model["scaler"]
    member_vecs = np.array([build_feature_vector(row) for _, row in same.iterrows()])
    member_scaled = scaler.transform(member_vecs)
    dists = np.sqrt(((member_scaled - vec_scaled[0]) ** 2).sum(axis=1))

    similar = same.copy()
    similar["distance"] = dists
    similar = similar.nsmallest(8, "distance")

    print(f"\n最相似 Token:")
    for _, s in similar.iterrows():
        print(f"  {s['symbol']:>12s}  ATH ${s['ath']:>12,.0f}  "
              f"上升{s['rise_hours']:.0f}h  衰减{s['decay_hours']:.0f}h  "
              f"dist={s['distance']:.2f}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Baseline Clustering v2")
    parser.add_argument("--show", action="store_true", help="显示已有结果")
    parser.add_argument("--predict", type=str, help="预测一个 token")
    args = parser.parse_args()

    if args.predict:
        predict_token(args.predict)
        return

    if args.show:
        csv_path = os.path.join(OUTPUT_DIR, "clusters.csv")
        pkl_path = os.path.join(OUTPUT_DIR, "model.pkl")
        if not os.path.isfile(csv_path):
            print("未找到结果。请先运行 python baseline_cluster_v2.py")
            return
        df = pd.read_csv(csv_path)
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
        print_clusters(df, model["cluster_order"])
        return

    tokens = load_radar_tokens()
    print(f"加载 {len(tokens)} 个 token")
    print(f"过滤: 2026年后 + ATH >= $100K")
    print(f"数据窗口: $100K → ATH → 跌70%截止")
    print()

    df, scaler, km, cluster_order = run_clustering(tokens)
    save_results(df, scaler, km, cluster_order)
    print_clusters(df, cluster_order)


if __name__ == "__main__":
    main()
