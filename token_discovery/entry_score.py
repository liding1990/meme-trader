"""Entry Score — continuous scoring for candidate pool tokens.

Computes a 0-100 entry score based on 5 weighted dimensions:
  - Regression Quality (50%): from Candidate Monitor z-scores
  - Momentum (15%): 4h price change
  - Volume Surge (15%): recent vs average volume
  - Buy Pressure (10%): buy/sell ratio
  - Holder Momentum (10%): recent holder growth

Modifiers:
  - Freshness decay: newer tokens score higher (1.0 → 0.3 over 720h)
  - Health penalty: tokens that dumped from peak get penalized (drawdown + 24h crash)

Regression scores are refreshed from local data on every run (no API calls).
Health penalty uses Codex real-time marketCap to catch dumps between data refreshes.

Usage:
    PYTHONPATH=. python -m token_discovery.entry_score          # run once
    PYTHONPATH=. python -m token_discovery.entry_score --loop   # run every 15 min
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codex_api import _query, CODEX_API_KEY
from token_discovery import db
from token_discovery.monitor import score_candidate, load_baseline_data
from token_discovery.pipeline import setup_logger

ENTRY_INTERVAL = 15 * 60  # 15 minutes

# Weights
W_REGRESSION = 0.50
W_MOMENTUM = 0.15
W_VOLUME = 0.15
W_BUY_PRESSURE = 0.10
W_HOLDER_MOMENTUM = 0.10

log = setup_logger()


def fetch_realtime_stats(addresses):
    """Fetch 4h stats for multiple tokens from Codex in one call."""
    if not addresses or not CODEX_API_KEY:
        return {}

    # Codex filterTokens with specific token addresses
    addr_list = ", ".join(f'"{a}"' for a in addresses[:50])  # batch limit

    query = f'''
    query {{
      filterTokens(
        tokens: [{addr_list}]
        statsType: FILTERED
        limit: 200
      ) {{
        results {{
          marketCap
          volume4
          volume24
          change4
          change24
          buyCount4
          sellCount4
          holders
          token {{
            address
          }}
        }}
      }}
    }}
    '''

    try:
        result = _query(query)
        tokens = result.get("filterTokens", {}).get("results", [])
        stats = {}
        for t in tokens:
            addr = t.get("token", {}).get("address", "")
            if addr:
                stats[addr] = {
                    "marketCap": float(t.get("marketCap", 0) or 0),
                    "volume4": float(t.get("volume4", 0) or 0),
                    "volume24": float(t.get("volume24", 0) or 0),
                    "change4": float(t.get("change4", 0) or 0),
                    "change24": float(t.get("change24", 0) or 0),
                    "buyCount4": int(t.get("buyCount4", 0) or 0),
                    "sellCount4": int(t.get("sellCount4", 0) or 0),
                    "holders": int(t.get("holders", 0) or 0),
                }
        return stats
    except Exception as e:
        log.error(f"Codex batch fetch failed: {e}")
        return {}


def score_momentum(change4h):
    """4h price change → 0-1 score. 0%=0, 50%+=1, negative=0."""
    if change4h <= 0:
        return 0.0
    return min(change4h / 0.50, 1.0)  # 0-50% mapped to 0-1


def score_volume_surge(vol4h, vol24h):
    """Recent 4h volume vs average 4h volume → 0-1 score."""
    avg_4h = vol24h / 6  # 24h / 6 = average 4h block
    if avg_4h <= 0:
        return 0.0
    ratio = vol4h / avg_4h
    # ratio 1x=0, 3x+=1
    return min(max((ratio - 1) / 2, 0), 1.0)


def score_buy_pressure(buy_count, sell_count):
    """Buy/(buy+sell) ratio → 0-1 score. 0.5=0, 0.7+=1."""
    total = buy_count + sell_count
    if total == 0:
        return 0.0
    ratio = buy_count / total
    # 0.5=0, 0.7=1
    return min(max((ratio - 0.5) / 0.2, 0), 1.0)


def score_holder_momentum(current_holders, discovery_holders):
    """Holder growth since discovery → 0-1 score."""
    if discovery_holders <= 0:
        return 0.0
    growth = (current_holders - discovery_holders) / discovery_holders
    # 0%=0, 20%+=1
    return min(max(growth / 0.20, 0), 1.0)


def score_regression(composite_zscore):
    """Regression z-score → 0-1 score. z=-1→0, z=0→0.5, z=1→1."""
    return min(max((composite_zscore + 1) / 2, 0), 1.0)


def health_penalty(current_mcap, peak_mcap, change24):
    """Penalize tokens that have dumped from their peak.

    Returns a multiplier 0.0-1.0:
      Drawdown < 20%: no penalty (1.0)
      Drawdown 20-50%: linear decay to 0.5
      Drawdown 50-80%: linear decay to 0.1
      Drawdown > 80%: near-zero (0.05)
      24h change < -30%: additional penalty (floor at 0.3x)
    """
    if peak_mcap <= 0 or current_mcap <= 0:
        return 1.0

    drawdown = 1.0 - (current_mcap / peak_mcap)

    if drawdown < 0.20:
        dd_mult = 1.0
    elif drawdown < 0.50:
        dd_mult = 1.0 - (drawdown - 0.20) / 0.30 * 0.5  # 1.0 → 0.5
    elif drawdown < 0.80:
        dd_mult = 0.5 - (drawdown - 0.50) / 0.30 * 0.4  # 0.5 → 0.1
    else:
        dd_mult = 0.05

    # Additional 24h crash penalty
    if change24 < -0.30:
        crash_mult = max(0.3, 1.0 + change24)  # -50% → 0.5, -70% → 0.3
    else:
        crash_mult = 1.0

    return dd_mult * crash_mult


def freshness_decay(lifetime_hours):
    """Token age → 0.3-1.0 decay factor. Newer tokens get higher scores.

    < 48h:    1.0 (full score)
    48-168h:  linear decay to 0.7
    168-720h: linear decay to 0.3
    > 720h:   0.3 (floor)
    """
    if lifetime_hours <= 48:
        return 1.0
    elif lifetime_hours <= 168:
        return 1.0 - (lifetime_hours - 48) / (168 - 48) * 0.3  # 1.0 → 0.7
    elif lifetime_hours <= 720:
        return 0.7 - (lifetime_hours - 168) / (720 - 168) * 0.4  # 0.7 → 0.3
    else:
        return 0.3


def run_once():
    """Score all candidates for entry."""
    log.info("=" * 50)
    log.info("Entry Score computation starting")

    db.init_db()

    candidates = db.get_candidates()
    scores_data = db.get_scores()

    if not candidates:
        log.info("No candidates in pool")
        return

    # Build regression score lookup from DB (fallback)
    reg_scores_db = {}
    for s in scores_data:
        reg_scores_db[s["address"]] = s["composite_score"]

    # Re-compute regression z-scores from latest local data (fast, no API calls)
    baseline_df = load_baseline_data()
    reg_scores = {}
    if baseline_df is not None:
        refreshed = 0
        for _c in candidates:
            addr = _c["address"]
            result = score_candidate(addr, baseline_df, skip_fetch=True)
            if result is not None:
                reg_scores[addr] = result["composite_score"]
                # Also update candidate_scores DB
                feat = result["feat"]
                db.upsert_score(
                    address=addr, symbol=_c["symbol"],
                    current_mcap=feat.get("ath", 0),
                    current_holders=feat.get("holders_at_ath", 0),
                    current_ath=feat.get("ath", 0),
                    current_rise_hours=feat.get("rise_hours", 0),
                    current_price_roc=feat.get("price_roc", 0),
                    current_volume_roc=feat.get("volume_roc", 0),
                    current_holder_roc=feat.get("holder_roc", 0),
                    current_holders_at_ath=feat.get("holders_at_ath", 0),
                    **result["z_scores"],
                    composite_score=result["composite_score"],
                )
                refreshed += 1
            else:
                # Fallback to DB cached value
                reg_scores[addr] = reg_scores_db.get(addr, 0)
        log.info(f"Refreshed regression scores for {refreshed}/{len(candidates)} candidates")
    else:
        reg_scores = reg_scores_db
        log.warning("Baseline data not available, using cached regression scores")

    # Fetch real-time stats from Codex
    addresses = [c["address"] for c in candidates]

    # Batch in groups of 50
    all_stats = {}
    for i in range(0, len(addresses), 50):
        batch = addresses[i:i+50]
        stats = fetch_realtime_stats(batch)
        all_stats.update(stats)
        if i + 50 < len(addresses):
            time.sleep(0.5)

    log.info(f"Fetched real-time stats for {len(all_stats)}/{len(addresses)} tokens")

    # Compute entry scores
    results = []
    for _c in candidates:
        c = dict(_c)  # convert sqlite3.Row to dict for .get()
        addr = c["address"]
        sym = c["symbol"]
        stats = all_stats.get(addr, {})
        reg_z = reg_scores.get(addr, 0)

        # Individual dimension scores (0-1)
        s_regression = score_regression(reg_z)
        s_momentum = score_momentum(stats.get("change4", 0))
        s_volume = score_volume_surge(stats.get("volume4", 0), stats.get("volume24", 0))
        s_buy = score_buy_pressure(stats.get("buyCount4", 0), stats.get("sellCount4", 0))
        s_holder = score_holder_momentum(stats.get("holders", 0), c["holders_at_discovery"])

        # Weighted composite (0-100) with freshness decay + health penalty
        raw_score = (
            s_regression * W_REGRESSION +
            s_momentum * W_MOMENTUM +
            s_volume * W_VOLUME +
            s_buy * W_BUY_PRESSURE +
            s_holder * W_HOLDER_MOMENTUM
        ) * 100

        lifetime = c.get("lifetime_hours", 0) or 0
        decay = freshness_decay(lifetime)

        # Health penalty: penalize tokens that have dumped from peak
        current_mcap = stats.get("marketCap", 0)
        # Peak = max of: candidate_pool ATH, discovery mcap, candidate_scores ATH
        score_ath = 0
        for sc in scores_data:
            if sc["address"] == addr:
                try:
                    score_ath = sc["current_ath"] or 0
                except (KeyError, IndexError):
                    pass
                break
        peak_mcap = max(
            c.get("ath", 0) or 0,                    # from candidate_pool
            c.get("mcap_at_discovery", 0) or 0,      # at discovery
            score_ath,                                 # from candidate_scores
        )
        # If current_mcap available and peak known, use mcap-based drawdown
        if current_mcap > 0 and peak_mcap > 0:
            # Ensure peak is at least current (peak should always >= current)
            peak_mcap = max(peak_mcap, current_mcap)
        change24 = stats.get("change24", 0)
        hp = health_penalty(current_mcap, peak_mcap, change24)

        entry_score = raw_score * decay * hp

        results.append({
            "address": addr,
            "symbol": sym,
            "entry_score": round(entry_score, 1),
            "raw_score": round(raw_score, 1),
            "freshness": round(decay, 2),
            "health": round(hp, 2),
            "lifetime_hours": round(lifetime, 0),
            "s_regression": round(s_regression, 3),
            "s_momentum": round(s_momentum, 3),
            "s_volume": round(s_volume, 3),
            "s_buy": round(s_buy, 3),
            "s_holder": round(s_holder, 3),
            "change4h": stats.get("change4", 0),
            "change24h": change24,
            "current_mcap": current_mcap,
            "peak_mcap": peak_mcap,
            "vol4h": stats.get("volume4", 0),
            "holders": stats.get("holders", 0),
            "reg_z": reg_z,
        })

    # Sort by entry score
    results.sort(key=lambda x: -x["entry_score"])

    # Save to DB
    conn = db.get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entry_scores (
            address TEXT PRIMARY KEY,
            symbol TEXT,
            updated_at TEXT,
            entry_score REAL DEFAULT 0,
            raw_score REAL DEFAULT 0,
            freshness REAL DEFAULT 1,
            health REAL DEFAULT 1,
            lifetime_hours REAL DEFAULT 0,
            s_regression REAL DEFAULT 0,
            s_momentum REAL DEFAULT 0,
            s_volume REAL DEFAULT 0,
            s_buy REAL DEFAULT 0,
            s_holder REAL DEFAULT 0,
            change4h REAL DEFAULT 0,
            change24h REAL DEFAULT 0,
            current_mcap REAL DEFAULT 0,
            peak_mcap REAL DEFAULT 0,
            vol4h REAL DEFAULT 0,
            holders INTEGER DEFAULT 0,
            reg_z REAL DEFAULT 0
        )
    """)
    # Add new columns if table already exists (migration)
    for col, coltype in [("health", "REAL DEFAULT 1"), ("change24h", "REAL DEFAULT 0"),
                          ("current_mcap", "REAL DEFAULT 0"), ("peak_mcap", "REAL DEFAULT 0")]:
        try:
            conn.execute(f"ALTER TABLE entry_scores ADD COLUMN {col} {coltype}")
        except Exception:
            pass  # column already exists

    now = datetime.now(timezone.utc).isoformat()
    for r in results:
        conn.execute("""
            INSERT OR REPLACE INTO entry_scores (
                address, symbol, updated_at, entry_score, raw_score, freshness, health,
                lifetime_hours,
                s_regression, s_momentum, s_volume, s_buy, s_holder,
                change4h, change24h, current_mcap, peak_mcap, vol4h, holders, reg_z
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (r["address"], r["symbol"], now, r["entry_score"], r["raw_score"],
              r["freshness"], r["health"], r["lifetime_hours"],
              r["s_regression"], r["s_momentum"], r["s_volume"],
              r["s_buy"], r["s_holder"],
              r["change4h"], r["change24h"], r["current_mcap"], r["peak_mcap"],
              r["vol4h"], r["holders"], r["reg_z"]))
    conn.commit()
    conn.close()

    # Log results
    log.info(f"\nEntry Score Rankings ({len(results)} candidates):")
    for i, r in enumerate(results[:15]):
        log.info(f"  {i+1:>2d}. {r['symbol']:>12s}  score={r['entry_score']:>5.1f}  "
                 f"raw={r['raw_score']:.1f} fresh={r['freshness']:.0%} health={r['health']:.0%}  "
                 f"reg={r['s_regression']:.2f} mom={r['s_momentum']:.2f} "
                 f"vol={r['s_volume']:.2f} buy={r['s_buy']:.2f} hold={r['s_holder']:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Entry Score")
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    if args.loop:
        log.info(f"Starting entry score loop (interval: {ENTRY_INTERVAL}s)")
        while True:
            try:
                run_once()
            except Exception as e:
                log.error(f"Entry score error: {e}")
                import traceback
                traceback.print_exc()
            time.sleep(ENTRY_INTERVAL)
    else:
        run_once()


if __name__ == "__main__":
    main()
