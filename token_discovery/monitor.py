"""Candidate Monitor — hourly scoring of candidate pool tokens.

Re-computes lifecycle features for each candidate using latest GMGN data,
calculates z-scores against regression baselines, outputs composite score.

Usage:
    PYTHONPATH=. python -m token_discovery.monitor          # run once
    PYTHONPATH=. python -m token_discovery.monitor --loop   # run every hour
"""

import argparse
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timezone

import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from token_discovery import db
from token_discovery.pipeline import setup_logger

CLUSTER_DIR = "baseline_cluster_v2_data"
MONITOR_INTERVAL = 3600  # 1 hour

# Regression dimension pairs (same as in app_baseline_cluster.py)
REGRESSION_PAIRS = [
    ("volume_roc", "holder_roc", "holder吸引效率"),
    ("price_roc", "volume_roc", "成交量真实性"),
    ("ath", "holders_at_ath", "市值可持续性"),
    ("price_roc", "holder_roc", "价格-社区联动"),
    ("holder_roc", "holders_at_ath", "增长天花板"),
]

log = setup_logger()


def load_baseline_data():
    """Load cluster CSV for regression baseline computation."""
    import pandas as pd
    csv_path = os.path.join(CLUSTER_DIR, "clusters.csv")
    if not os.path.isfile(csv_path):
        return None
    df = pd.read_csv(csv_path)
    return df[df["rank"].isin([5, 6])]  # organic only


def compute_regression_zscore(baseline_df, x_col, y_col, token_x, token_y):
    """Compute z-score of a token against poly-2 regression of baseline."""
    sub = baseline_df[(baseline_df[x_col] > 0) & (baseline_df[y_col] > 0)]
    if len(sub) < 10 or token_x <= 0 or token_y <= 0:
        return 0.0

    log_x = np.log10(sub[x_col].values)
    log_y = np.log10(sub[y_col].values)

    coeffs = np.polyfit(log_x, log_y, 2)
    poly = np.poly1d(coeffs)

    y_pred = poly(log_x)
    residual_std = np.std(log_y - y_pred)
    if residual_std < 1e-9:
        return 0.0

    token_log_x = np.log10(token_x)
    token_log_y = np.log10(token_y)
    expected = poly(token_log_x)
    z = (token_log_y - expected) / residual_std

    return float(np.clip(z, -5, 5))


def score_candidate(address, baseline_df):
    """Re-compute features for a candidate and calculate z-scores."""
    from baseline_cluster_v2 import compute_features

    feat = compute_features(address)
    if feat is None:
        # Try fetching fresh data
        try:
            from gmgn_api import fetch_token_data
            fetch_token_data("sol", address)
            feat = compute_features(address)
        except Exception:
            pass

    if feat is None:
        return None

    z_scores = {}

    # Compute z-score for each regression pair
    z_scores["z_holder_efficiency"] = compute_regression_zscore(
        baseline_df, "volume_roc", "holder_roc",
        feat.get("volume_roc", 0), feat.get("holder_roc", 0))

    z_scores["z_volume_authenticity"] = compute_regression_zscore(
        baseline_df, "price_roc", "volume_roc",
        feat.get("price_roc", 0), feat.get("volume_roc", 0))

    z_scores["z_mcap_sustainability"] = compute_regression_zscore(
        baseline_df, "ath", "holders_at_ath",
        feat.get("ath", 0), feat.get("holders_at_ath", 0))

    z_scores["z_price_community"] = compute_regression_zscore(
        baseline_df, "price_roc", "holder_roc",
        feat.get("price_roc", 0), feat.get("holder_roc", 0))

    z_scores["z_growth_ceiling"] = compute_regression_zscore(
        baseline_df, "holder_roc", "holders_at_ath",
        feat.get("holder_roc", 0), feat.get("holders_at_ath", 0))

    # Weighted composite score based on empirical analysis:
    # - 增长天花板 (r=0.71 with ATH): highest predictive power
    # - holder吸引效率 (r=0.25): moderate, independent signal
    # - 成交量真实性 (r=-0.45): inverse signal, moderate
    # - 市值可持续性 (r=0.00): low predictive power
    # - 价格-社区联动 (r=-0.43): redundant with 成交量真实性 (r=0.61 correlation)
    weights = {
        "z_growth_ceiling": 0.35,        # strongest predictor of ATH
        "z_holder_efficiency": 0.25,     # independent, moderate signal
        "z_volume_authenticity": 0.20,   # moderate but inverse
        "z_mcap_sustainability": 0.10,   # weak predictor
        "z_price_community": 0.10,       # redundant with volume_authenticity
    }
    composite = sum(z_scores[k] * weights[k] for k in weights)

    return {
        "feat": feat,
        "z_scores": z_scores,
        "composite_score": composite,
    }


def run_once():
    """Score all candidates in the pool."""
    log.info("=" * 50)
    log.info("Candidate Monitor starting")

    db.init_db()
    baseline_df = load_baseline_data()
    if baseline_df is None:
        log.error("Cannot load baseline data")
        return

    candidates = db.get_candidates()
    log.info(f"Scoring {len(candidates)} candidates...")

    scored = 0
    failed = 0

    for c in candidates:
        addr = c["address"]
        sym = c["symbol"]

        result = score_candidate(addr, baseline_df)
        time.sleep(1.5)  # GMGN rate limit

        if result is None:
            failed += 1
            log.debug(f"  {sym}: failed to compute features")
            continue

        feat = result["feat"]
        zs = result["z_scores"]
        composite = result["composite_score"]

        db.upsert_score(
            address=addr, symbol=sym,
            current_mcap=feat.get("ath", 0),
            current_holders=feat.get("holders_at_ath", 0),
            current_ath=feat.get("ath", 0),
            current_rise_hours=feat.get("rise_hours", 0),
            current_price_roc=feat.get("price_roc", 0),
            current_volume_roc=feat.get("volume_roc", 0),
            current_holder_roc=feat.get("holder_roc", 0),
            current_holders_at_ath=feat.get("holders_at_ath", 0),
            **zs,
            composite_score=composite,
        )

        scored += 1
        direction = "+" if composite > 0 else ""
        log.info(f"  {sym:>12s}: score={direction}{composite:.2f} | "
                 f"holder效率={zs['z_holder_efficiency']:+.2f} "
                 f"成交量={zs['z_volume_authenticity']:+.2f} "
                 f"可持续={zs['z_mcap_sustainability']:+.2f}")

    log.info(f"\nMonitor complete: {scored} scored, {failed} failed")


def main():
    parser = argparse.ArgumentParser(description="Candidate Monitor")
    parser.add_argument("--loop", action="store_true", help="Run every hour")
    args = parser.parse_args()

    if args.loop:
        log.info(f"Starting monitor loop (interval: {MONITOR_INTERVAL}s)")
        while True:
            try:
                run_once()
            except Exception as e:
                log.error(f"Monitor error: {e}")
                import traceback
                traceback.print_exc()
            log.info(f"Sleeping {MONITOR_INTERVAL}s...")
            time.sleep(MONITOR_INTERVAL)
    else:
        run_once()


if __name__ == "__main__":
    main()
