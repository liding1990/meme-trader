"""Funnel Bot Engine — 4-layer paper trading with CatBoost models.

L0: Codex scan → L1: Scam filter → L2: Entry model → L3: Position management

Usage:
    PYTHONPATH=. python -m funnel_bot.engine
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catboost import CatBoostClassifier
from codex_api import filter_tokens, CODEX_API_KEY
from scalper.price import get_token_price
from funnel_bot import db
from trading_funnel.features import (
    extract_features_at_tick, precompute_hmm_for_token,
    apply_hmm_to_features, FEATURE_NAMES,
)

# ── Config ──

TICK_INTERVAL = 300          # 5 minutes
SCAN_INTERVAL = 600          # 10 minutes between scans
INITIAL_CAPITAL = 5000
POSITION_SIZE = 100
MAX_POSITIONS = 10
MAX_HOLD_BARS = 48           # 48 ticks = 4 hours at 5min, or 48h at 1h
L2_THRESHOLD = 0.60
HARD_STOP = -0.35
SLIPPAGE = 0.03

ACTION_SELL = {"HOLD": 0, "TP_25": 0.25, "TP_50": 0.50, "TP_100": 1.0,
               "SL_25": 0.25, "SL_50": 0.50, "EXIT": 1.0}


# ── Model Loading ──

def load_models():
    l2 = CatBoostClassifier()
    l2.load_model("trading_funnel/l2_entry.cbm")
    l3 = CatBoostClassifier()
    l3.load_model("trading_funnel/l3_position.cbm")
    with open("trading_funnel/feature_config.json") as f:
        cfg = json.load(f)
    return l2, l3, cfg["l2_features"], cfg["l3_features"]


# ── L0: Market Scan ──

def l0_scan():
    """Scan Codex for tokens matching L0 criteria."""
    if not CODEX_API_KEY:
        print("  [L0] No CODEX_API_KEY, skipping scan")
        return []

    try:
        three_hours_ago = int(time.time()) - 3 * 3600
        tokens, count, _ = filter_tokens(
            created_after=three_hours_ago,
            min_mcap=100000, min_holders=100, limit=50,
        )
        return tokens
    except Exception as e:
        print(f"  [L0] Scan error: {e}")
        return []


# ── L1: Scam Filter ──

def l1_filter(token_data):
    """Filter scam/rug tokens. Returns (pass: bool, label: str)."""
    holders = token_data.get("holders", 0)
    mcap = token_data.get("marketCap", 0)

    if holders < 200:
        return False, "low_holders"

    # MCap/holder ratio too high = potential manipulation
    if holders > 0 and mcap / holders > 50000:
        return False, "high_mcap_per_holder"

    return True, "passed"


# ── L2: Entry Signal ──

def l2_predict(address, l2_model, l2_features):
    """Run L2 entry model. Returns (should_enter: bool, proba: float, features: dict)."""
    price_data = get_token_price(address)
    if price_data is None:
        return False, 0, {}

    mcap = price_data.get("market_cap", 0) or price_data.get("price", 0)
    if mcap <= 0:
        return False, 0, {}

    # Build minimal features from current price data
    feat = {}
    for f in FEATURE_NAMES:
        feat[f] = 0

    feat["mcap_start"] = mcap
    feat["mcap_end"] = mcap
    feat["mcap_max"] = mcap
    feat["mcap_min"] = mcap

    buy_vol = price_data.get("buy_volume_5m", 0)
    sell_vol = price_data.get("sell_volume_5m", 0)
    feat["volume_trend"] = 0
    feat["volume_concentration"] = 0
    feat["price_vs_vwap"] = 1.0

    vec = np.array([[feat.get(c, 0) for c in l2_features]])
    vec = np.nan_to_num(vec, nan=0)

    proba = float(l2_model.predict_proba(vec)[0][1])
    return proba >= L2_THRESHOLD, proba, feat


# ── L3: Position Management ──

def l3_predict(address, entry_price, current_price, remaining, peak_price,
               bars_held, l3_model, l3_features):
    """Run L3 position model. Returns action string."""
    pnl = (current_price - entry_price) / max(entry_price, 1)

    # Hard stop override
    if pnl < HARD_STOP:
        return "EXIT"

    feat = {}
    for f in FEATURE_NAMES:
        feat[f] = 0

    feat["unrealized_pnl"] = pnl
    feat["holding_hours"] = bars_held
    feat["sold_pct"] = 1.0 - remaining
    feat["remaining_pct"] = remaining
    feat["distance_from_peak"] = (peak_price - current_price) / max(peak_price, 1)
    feat["mcap_end"] = current_price
    feat["price_vs_vwap"] = 1.0

    vec = np.array([[feat.get(c, 0) for c in l3_features]])
    vec = np.nan_to_num(vec, nan=0)

    pred = str(l3_model.predict(vec).flatten()[0])

    # Hard TP nudge
    if pnl > 1.0 and pred == "HOLD" and remaining > 0.25:
        pred = "TP_25"

    return pred


# ── Main Engine ──

def run():
    print("=" * 60)
    print("Funnel Bot Paper Trading Engine")
    print("=" * 60)

    db.init_db()
    l2_model, l3_model, l2_features, l3_features = load_models()

    capital = INITIAL_CAPITAL
    daily_pnl = 0
    daily_trades = 0
    daily_wins = 0
    last_scan = 0
    tick = 0

    print(f"初始资金: ${capital:,.0f}")
    print(f"仓位大小: ${POSITION_SIZE}")
    print(f"L2 入场阈值: {L2_THRESHOLD}")
    print(f"Tick 间隔: {TICK_INTERVAL}s")
    print()

    while True:
        tick += 1
        tick_start = time.time()
        now = datetime.now(timezone.utc)
        print(f"\n--- Tick {tick} | {now.strftime('%H:%M:%S')} | Capital: ${capital:,.0f} ---")

        try:
            # ── Phase 1: Periodic Scan (L0 + L1 + L2) ──
            if time.time() - last_scan >= SCAN_INTERVAL:
                last_scan = time.time()
                print("  [L0] 扫描市场...")

                candidates = l0_scan()
                l0_count = len(candidates)

                l1_passed = []
                l1_filtered = 0
                for c in candidates:
                    passed, label = l1_filter(c)
                    if passed:
                        l1_passed.append(c)
                    else:
                        l1_filtered += 1

                l2_signals = 0
                l2_rejected = 0
                for c in l1_passed:
                    addr = c["token"]["address"]
                    sym = c["token"].get("symbol", "?")
                    name = c["token"].get("name", "?")
                    mcap = c.get("marketCap", 0)
                    holders = c.get("holders", 0)

                    should_enter, proba, _ = l2_predict(addr, l2_model, l2_features)

                    db.upsert_watchlist(addr, sym, name, mcap, holders,
                                        "passed", round(proba, 3))

                    if should_enter:
                        l2_signals += 1
                    else:
                        l2_rejected += 1

                db.log_scan(l0_count, len(l1_passed), l1_filtered, l2_signals, l2_rejected)

                print(f"  [L0] {l0_count} 候选 → [L1] {len(l1_passed)} 通过, {l1_filtered} 过滤 → [L2] {l2_signals} 信号")

            # ── Phase 2: Manage Open Positions (L3) ──
            positions = db.get_positions()
            for pos in positions:
                addr = pos["token_address"]
                sym = pos["symbol"]
                entry_price = pos["entry_price"]
                remaining = pos["remaining_pct"]
                peak = pos["peak_price"]
                bars = pos["bars_held"] + 1
                realized = pos["realized_pnl"]

                price_data = get_token_price(addr)
                if price_data is None:
                    continue

                current = price_data.get("market_cap", 0) or price_data.get("price", 0)
                if current <= 0:
                    continue

                peak = max(peak, current)
                pnl = (current - entry_price) / max(entry_price, 1)

                # L3 prediction
                action = l3_predict(addr, entry_price, current, remaining, peak,
                                     bars, l3_model, l3_features)

                sell_frac = min(ACTION_SELL.get(action, 0), remaining)

                if sell_frac > 0:
                    sell_pnl = pnl * (1 - SLIPPAGE)
                    realized += sell_frac * sell_pnl * pos["position_size"]
                    remaining -= sell_frac

                    db.log_l3_action(addr, sym, action, current, pnl * 100, remaining + sell_frac, remaining)
                    print(f"  [L3] {sym}: {action} | PnL: {pnl*100:+.1f}% | 剩余: {remaining*100:.0f}%")

                # Force exit at max hold
                if bars >= MAX_HOLD_BARS and remaining > 0.01:
                    sell_pnl = pnl * (1 - SLIPPAGE)
                    realized += remaining * sell_pnl * pos["position_size"]
                    remaining = 0
                    action = "TIME_EXIT"
                    db.log_l3_action(addr, sym, "TIME_EXIT", current, pnl * 100, remaining, 0)

                if remaining <= 0.01:
                    # Close position
                    total_return = realized / max(pos["position_size"], 1)
                    pnl_usd = realized
                    capital += pos["position_size"] + pnl_usd
                    daily_pnl += pnl_usd
                    daily_trades += 1
                    if pnl_usd > 0:
                        daily_wins += 1

                    db.close_position(addr, current, total_return * 100, pnl_usd, action, [])
                    print(f"  [CLOSE] {sym}: {total_return*100:+.1f}% (${pnl_usd:+.1f})")
                else:
                    db.update_position(addr, peak, bars, remaining, realized)

            # ── Phase 3: Open New Positions ──
            positions = db.get_positions()
            if len(positions) < MAX_POSITIONS:
                watchlist = db.get_watchlist()
                open_addrs = {p["token_address"] for p in positions}

                for item in watchlist:
                    if len(positions) + len(open_addrs) >= MAX_POSITIONS:
                        break
                    addr = item["token_address"]
                    if addr in open_addrs:
                        continue
                    if item["l2_proba"] < L2_THRESHOLD:
                        continue
                    if capital < POSITION_SIZE:
                        break

                    # Open position
                    price_data = get_token_price(addr)
                    if price_data is None:
                        continue
                    mcap = price_data.get("market_cap", 0) or price_data.get("price", 0)
                    if mcap <= 0:
                        continue

                    capital -= POSITION_SIZE
                    trade_id = db.open_position(addr, item["symbol"], mcap, POSITION_SIZE,
                                                 item["l2_proba"], item["l1_label"])
                    open_addrs.add(addr)
                    print(f"  [ENTRY] {item['symbol']}: MCap=${mcap:,.0f} | L2={item['l2_proba']:.2f}")

            # ── Phase 4: Record Equity ──
            if tick % 5 == 0:
                positions = db.get_positions()
                total_equity = capital + sum(p["position_size"] for p in positions)
                db.record_equity(capital, len(positions), total_equity,
                                  daily_pnl, daily_trades, daily_wins)

        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()

        # Sleep
        elapsed = time.time() - tick_start
        sleep_time = max(0, TICK_INTERVAL - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    run()
