"""Token Discovery Pipeline — automated scan + classify + filter.

Runs one cycle:
1. Codex API → trending launchpad tokens (24h, Solana)
2. GMGN API → hourly candle history per token
3. Compute 11 lifecycle features
4. KMeans predict cluster
5. Save #5 + #6 to candidate_pool DB
6. Log everything

Usage:
    PYTHONPATH=. python -m token_discovery.pipeline          # run once
    PYTHONPATH=. python -m token_discovery.pipeline --loop   # run every 15 min
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codex_api import _query
from token_discovery import db

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

CLUSTER_DIR = "baseline_cluster_v2_data"
CLUSTER_META = {
    1: "Fake / Scam Token", 2: "Pump & Dump", 3: "Ghost Pump",
    4: "Slow Bleed", 5: "Organic Runner", 6: "Fast Organic",
}

LAUNCHPADS = [
    "Pump.fun", "Pump Mayhem", "Bonk", "Believe", "Moonshot",
    "Jupiter Studio", "boop", "Heaven", "LaunchLab", "Moonit",
    "TokenMill V2", "MeteoraDBC", "Zora Solana", "Cooking.City",
    "Circus", "BAGS", "time.fun", "Dealr",
]

SCAN_INTERVAL = 15 * 60  # 15 minutes


# ── Logging ──────────────────────────────────────────────────────────────────


def setup_logger():
    logger = logging.getLogger("token_discovery")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    # File: JSON structured, daily rotation
    fh = TimedRotatingFileHandler(
        os.path.join(LOG_DIR, "token_discovery.log"),
        when="midnight", backupCount=30, utc=True,
    )
    fh.setLevel(logging.DEBUG)

    class JsonFmt(logging.Formatter):
        def format(self, record):
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "msg": record.getMessage(),
            }
            if hasattr(record, "data"):
                entry["data"] = record.data
            return json.dumps(entry, ensure_ascii=False, default=str)

    fh.setFormatter(JsonFmt())
    logger.addHandler(fh)

    return logger


log = setup_logger()


def log_with_data(level, msg, data=None):
    record = log.makeRecord("token_discovery", level, "", 0, msg, (), None)
    record.data = data
    log.handle(record)


# ── Model Loading ────────────────────────────────────────────────────────────


def load_cluster_model():
    pkl_path = os.path.join(CLUSTER_DIR, "model.pkl")
    if not os.path.isfile(pkl_path):
        log.error(f"Cluster model not found at {pkl_path}")
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ── Step 1: Codex Scan ───────────────────────────────────────────────────────


def scan_codex():
    """Fetch trending launchpad tokens from Codex."""
    one_month_ago = int(time.time()) - 30 * 86400
    lp_list = ", ".join(f'"{lp}"' for lp in LAUNCHPADS)

    query = f'''
    query FilterTokens {{
      filterTokens(
        filters: {{
          network: [1399811149]
          marketCap: {{gte: 100000}}
          holders: {{gte: 200}}
          volume4: {{gte: 5000}}
          createdAt: {{gte: {one_month_ago}}}
          launchpadName: [{lp_list}]
          trendingIgnored: false
          potentialScam: false
        }}
        statsType: FILTERED
        rankings: [{{attribute: trendingScore24, direction: DESC}}]
        limit: 200
        offset: 0
      ) {{
        results {{
          marketCap
          holders
          liquidity
          volume4
          change4
          buyCount4
          sellCount4
          priceUSD
          createdAt
          token {{
            address
            name
            symbol
          }}
        }}
        count
      }}
    }}
    '''
    result = _query(query)
    ft = result.get("filterTokens", {})
    return ft.get("results", []), ft.get("count", 0)


# ── Step 2-4: GMGN Fetch + Feature Compute + Classify ────────────────────────


def classify_token(address, model):
    """Fetch GMGN data, compute features, predict cluster.

    Returns (rank, cluster_name, features_dict) or (None, None, None).
    """
    from baseline_cluster_v2 import compute_features, build_feature_vector

    feat = compute_features(address)
    if feat is None:
        # Try fetching first
        try:
            from gmgn_api import fetch_token_data
            fetch_token_data("sol", address)
            feat = compute_features(address)
        except Exception:
            pass

    if feat is None:
        return None, None, None

    scaler = model["scaler"]
    km = model["kmeans"]
    cluster_order = model["cluster_order"]
    rank_map = {c: i + 1 for i, c in enumerate(cluster_order)}

    vec = scaler.transform([build_feature_vector(feat)])
    cid = int(km.predict(vec)[0])
    rank = rank_map.get(cid, 0)
    name = CLUSTER_META.get(rank, "Unknown")

    return rank, name, feat


# ── Pipeline ─────────────────────────────────────────────────────────────────


def run_once():
    """Run one complete scan cycle."""
    log.info("=" * 50)
    log.info("Token Discovery pipeline starting")

    db.init_db()
    model = load_cluster_model()
    if model is None:
        log.error("Cannot load cluster model. Aborting.")
        return

    # Step 1: Codex scan
    log.info("Step 1: Scanning Codex trending tokens...")
    try:
        tokens, total_count = scan_codex()
    except Exception as e:
        log.error(f"Codex scan failed: {e}")
        return

    codex_count = len(tokens)
    log.info(f"  Codex returned {codex_count} tokens (total matching: {total_count})")

    # Step 2-4: Classify each token
    log.info("Step 2-4: Fetching GMGN data + classifying...")
    token_details = []
    gmgn_fetched = 0
    classified = 0
    new_candidates = 0

    for i, t in enumerate(tokens):
        tok = t["token"]
        addr = tok.get("address", "")
        sym = tok.get("symbol", "?")
        name = tok.get("name", "?")
        mcap = float(t.get("marketCap", 0) or 0)
        holders = int(t.get("holders", 0) or 0)
        vol4h = float(t.get("volume4", 0) or 0)
        change4h = float(t.get("change4", 0) or 0)
        liquidity = float(t.get("liquidity", 0) or 0)
        buy4h = int(t.get("buyCount4", 0) or 0)
        sell4h = int(t.get("sellCount4", 0) or 0)
        created_at = int(t.get("createdAt", 0) or 0)
        lifetime_hours = (time.time() - created_at) / 3600 if created_at > 0 else 0

        rank, cluster_name, feat = classify_token(addr, model)
        gmgn_fetched += 1
        time.sleep(1.5)  # GMGN rate limit

        detail = {
            "symbol": sym, "address": addr, "mcap": mcap,
            "holders": holders, "vol4h": vol4h, "change4h": change4h,
            "lifetime_hours": round(lifetime_hours, 1),
            "cluster_rank": rank, "cluster_name": cluster_name or "无法分类",
            "passed_l1": rank in [5, 6] if rank else False,
        }

        if feat:
            detail.update({
                "ath": feat.get("ath", 0),
                "rise_hours": feat.get("rise_hours", 0),
                "decay_hours": feat.get("decay_hours", 0),
                "price_roc": feat.get("price_roc", 0),
                "holders_at_ath": feat.get("holders_at_ath", 0),
            })
            classified += 1

        # Step 5: Save if organic
        if rank in [5, 6] and feat:
            is_new = db.add_candidate(
                address=addr, symbol=sym, name=name,
                cluster_rank=rank, cluster_name=cluster_name,
                mcap=mcap, holders=holders, vol4h=vol4h,
                change4h=change4h, liquidity=liquidity,
                buy4h=buy4h, sell4h=sell4h,
                ath=feat.get("ath", 0),
                rise_hours=feat.get("rise_hours", 0),
                decay_hours=feat.get("decay_hours", 0),
                price_roc=feat.get("price_roc", 0),
                volume_roc=feat.get("volume_roc", 0),
                holder_roc=feat.get("holder_roc", 0),
                holders_at_ath=feat.get("holders_at_ath", 0),
                total_hours=feat.get("total_hours", 0),
                rise_pct=feat.get("rise_pct", 0),
                lifetime_hours=lifetime_hours,
            )
            if is_new:
                new_candidates += 1
                detail["is_new"] = True
                log.info(f"  ✅ NEW candidate: {sym} → #{rank} {cluster_name} | MCap=${mcap:,.0f} | Holders={holders}")
            else:
                detail["is_new"] = False
        else:
            detail["is_new"] = False

        token_details.append(detail)
        log_with_data(logging.DEBUG, f"  {sym}: #{rank} {cluster_name or '?'} | MCap=${mcap:,.0f}",
                       data=detail)

        if (i + 1) % 10 == 0:
            log.info(f"  Progress: {i+1}/{codex_count}")

    # Step 6: Log summary
    total_pool = db.get_pool_size()

    log.info(f"\nPipeline complete:")
    log.info(f"  Codex scanned:    {codex_count}")
    log.info(f"  GMGN fetched:     {gmgn_fetched}")
    log.info(f"  Classified:       {classified}")
    log.info(f"  New candidates:   {new_candidates}")
    log.info(f"  Total pool:       {total_pool}")

    db.log_scan(codex_count, gmgn_fetched, classified, new_candidates, total_pool, token_details)

    log_with_data(logging.INFO, "Pipeline summary", data={
        "codex_count": codex_count, "gmgn_fetched": gmgn_fetched,
        "classified": classified, "new_candidates": new_candidates,
        "total_pool": total_pool,
    })

    return {
        "codex_count": codex_count, "new_candidates": new_candidates,
        "total_pool": total_pool,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Token Discovery Pipeline")
    parser.add_argument("--loop", action="store_true", help="Run continuously every 15 min")
    args = parser.parse_args()

    if args.loop:
        log.info(f"Starting continuous loop (interval: {SCAN_INTERVAL}s)")
        while True:
            try:
                run_once()
            except Exception as e:
                log.error(f"Pipeline error: {e}")
                import traceback
                traceback.print_exc()
            log.info(f"Sleeping {SCAN_INTERVAL}s until next scan...")
            time.sleep(SCAN_INTERVAL)
    else:
        run_once()


if __name__ == "__main__":
    main()
