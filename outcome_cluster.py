"""Phase 3: Outcome-based KMeans clustering.

Clusters tokens by 6 outcome features using KMeans(k=6).
Saves cluster assignments and model for prediction.

Usage:
    python outcome_cluster.py              # Run clustering, save results
    python outcome_cluster.py --show       # Show cluster details
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
N_CLUSTERS = 6
OUTPUT_DIR = "outcome_cluster_data"


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


def compute_outcome_features(address):
    """Extract 6 outcome features from a token's full history."""
    df = load_1h_candles(address)
    if df is None or len(df) < 2:
        return None

    mcap = df["mcap"].values
    ath = mcap.max()
    if ath < MCAP_THRESHOLD:
        return None

    ath_idx = np.argmax(mcap)
    hours = df["time_ms"].values
    hours_to_ath = max((hours[ath_idx] - hours[0]) / 3600000, 1)

    # Price speed
    price_roc = (ath - MCAP_THRESHOLD) / hours_to_ath

    # Volume speed
    vol_to_ath = df["volume"].iloc[:ath_idx + 1].sum()
    volume_roc = vol_to_ath / hours_to_ath

    # Holder features
    holder_df = load_holders(address)
    if holder_df is not None and len(holder_df) >= 2:
        holders = holder_df["holders"].values
        max_holders = float(holders.max())

        # Find holders at ATH time
        ath_time = df["datetime"].iloc[ath_idx]
        h_before_ath = holder_df[holder_df["datetime"] <= ath_time]
        holders_at_ath = h_before_ath["holders"].iloc[-1] if len(h_before_ath) > 0 else holders[0]
        holder_roc = (holders_at_ath - holders[0]) / hours_to_ath

        # Decay
        hours_post_ath = max((hours[-1] - hours[ath_idx]) / 3600000, 1)
        h_after_ath = holder_df[holder_df["datetime"] >= ath_time]
        holders_end = h_after_ath["holders"].iloc[-1] if len(h_after_ath) > 0 else holders[-1]
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


# ── Clustering ───────────────────────────────────────────────────────────────


CLUSTER_LABELS = {
    "Organic Runner": "高ATH + 高持币人 + 中等增速。社区驱动的真实runner。",
    "Fast Organic": "中高ATH + 快速持币人涌入。爆发力强，有社区但来得快。",
    "Pump-Dump": "高ATH + 极快价格增速 + 极少持币人。机器人操盘，瞬间拉盘瞬间归零。",
    "Slow Grinder": "中ATH + 极慢增速。长期缓慢增长型，多见于Base链。",
    "Average": "各项指标中规中矩。普通token。",
    "Flash Crash": "低ATH + 极快持币人流失。快速崩盘型。",
}


def run_clustering(tokens):
    """Run KMeans clustering on outcome features."""
    rows = []
    for token in tokens:
        feat = compute_outcome_features(token["address"])
        if feat is None:
            continue
        rows.append({"address": token["address"], "symbol": token["symbol"],
                      "name": token["name"], "chain": token["chain"], **feat})

    df = pd.DataFrame(rows)
    print(f"可用 token: {len(df)}")

    feature_names = ["ath", "price_roc", "volume_roc", "holder_roc", "holder_decay_roc", "max_holders"]

    # Build feature matrix
    def build_feature_vector(row):
        return [
            np.log1p(max(row["ath"], 0)),
            np.log1p(max(row["price_roc"], 0)),
            np.log1p(max(row["volume_roc"], 0)),
            np.log1p(max(row["holder_roc"], 0)),
            -np.log1p(max(-row["holder_decay_roc"], 0)),
            np.log1p(max(row["max_holders"], 0)),
        ]

    X = np.array([build_feature_vector(row) for _, row in df.iterrows()])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    df["cluster_id"] = labels

    # Rank clusters by median ATH (descending)
    cluster_aths = df.groupby("cluster_id")["ath"].median()
    cluster_order = cluster_aths.sort_values(ascending=False).index.tolist()
    rank_map = {cid: i for i, cid in enumerate(cluster_order)}
    df["rank"] = df["cluster_id"].map(rank_map)

    # Auto-assign labels based on feature profile
    label_names = list(CLUSTER_LABELS.keys())
    global_med = {k: df[k].median() for k in feature_names}

    cluster_label_map = {}
    for cid in cluster_order:
        sub = df[df["cluster_id"] == cid]
        meds = {k: sub[k].median() for k in feature_names}
        rank = rank_map[cid]

        # Rank-based feature comparison
        ranks_by_feat = {}
        for feat_name in feature_names:
            vals = [(c, df[df["cluster_id"] == c][feat_name].median()) for c in cluster_order]
            if feat_name == "holder_decay_roc":
                vals.sort(key=lambda x: x[1])
            else:
                vals.sort(key=lambda x: x[1], reverse=True)
            for r, (c, _) in enumerate(vals):
                if c == cid:
                    ranks_by_feat[feat_name] = r

        high_ath = ranks_by_feat["ath"] <= 1
        high_holders = ranks_by_feat["max_holders"] <= 1
        fast_price = ranks_by_feat["price_roc"] <= 1
        no_holder_growth = ranks_by_feat["holder_roc"] >= N_CLUSTERS - 2
        fast_decay = ranks_by_feat["holder_decay_roc"] == 0
        slow_price = ranks_by_feat["price_roc"] >= N_CLUSTERS - 1

        if high_ath and high_holders and not fast_price:
            lbl = "Organic Runner"
        elif fast_price and no_holder_growth:
            lbl = "Pump-Dump"
        elif fast_decay:
            lbl = "Flash Crash"
        elif high_ath or (ranks_by_feat["holder_roc"] <= 1 and ranks_by_feat["ath"] <= 2):
            lbl = "Fast Organic"
        elif slow_price:
            lbl = "Slow Grinder"
        else:
            lbl = "Average"

        cluster_label_map[cid] = lbl

    df["label"] = df["cluster_id"].map(cluster_label_map)

    return df, scaler, km, feature_names, cluster_label_map, cluster_order


def save_results(df, scaler, km, feature_names, cluster_label_map, cluster_order):
    """Save clustering results for later prediction."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df.to_csv(os.path.join(OUTPUT_DIR, "cluster_assignments.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "model.pkl"), "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "kmeans": km,
            "feature_names": feature_names,
            "cluster_label_map": cluster_label_map,
            "cluster_order": cluster_order,
        }, f)

    print(f"结果已保存到 {OUTPUT_DIR}/")


def print_clusters(df, cluster_label_map, cluster_order):
    """Print cluster summary."""
    feature_names = ["ath", "price_roc", "volume_roc", "holder_roc", "holder_decay_roc", "max_holders"]
    rank_map = {cid: i for i, cid in enumerate(cluster_order)}

    print(f"\n{'=' * 80}")
    print(f"结果特征 KMeans 聚类（{N_CLUSTERS} 类，{len(df)} 个 token）")
    print(f"{'=' * 80}")

    for cid in cluster_order:
        sub = df[df["cluster_id"] == cid]
        label = cluster_label_map[cid]
        rank = rank_map[cid] + 1
        desc = CLUSTER_LABELS.get(label, "")

        print(f"\n--- #{rank} {label} ({len(sub)} tokens) ---")
        print(f"  {desc}")
        print(f"  ATH 中位数:       ${sub['ath'].median():>14,.0f}")
        print(f"  价格增速中位数:   ${sub['price_roc'].median():>14,.0f}/h")
        print(f"  成交量增速中位数: ${sub['volume_roc'].median():>14,.0f}/h")
        print(f"  持币人增速中位数:  {sub['holder_roc'].median():>14,.1f}/h")
        print(f"  持币人衰减中位数:  {sub['holder_decay_roc'].median():>14,.1f}/h")
        print(f"  最大持币人中位数:  {sub['max_holders'].median():>14,.0f}")

        # Show top tokens
        top = sub.nlargest(8, "ath")
        print(f"  代表 token:")
        for _, row in top.iterrows():
            print(f"    {row['symbol']:>12s}  ATH ${row['ath']:>12,.0f}  "
                  f"holders {row['max_holders']:>8,.0f}  "
                  f"price_roc ${row['price_roc']:>10,.0f}/h")


def main():
    parser = argparse.ArgumentParser(description="Outcome-based KMeans 聚类")
    parser.add_argument("--show", action="store_true", help="仅展示已有结果")
    args = parser.parse_args()

    if args.show:
        csv_path = os.path.join(OUTPUT_DIR, "cluster_assignments.csv")
        pkl_path = os.path.join(OUTPUT_DIR, "model.pkl")
        if not os.path.isfile(csv_path):
            print("未找到聚类结果。请先运行 python outcome_cluster.py")
            return
        df = pd.read_csv(csv_path)
        with open(pkl_path, "rb") as f:
            model_data = pickle.load(f)
        print_clusters(df, model_data["cluster_label_map"], model_data["cluster_order"])
        return

    tokens = load_radar_tokens()
    print(f"加载了 {len(tokens)} 个雷达 token")

    df, scaler, km, feature_names, cluster_label_map, cluster_order = run_clustering(tokens)
    save_results(df, scaler, km, feature_names, cluster_label_map, cluster_order)
    print_clusters(df, cluster_label_map, cluster_order)


if __name__ == "__main__":
    main()
