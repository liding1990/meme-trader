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
from funnel_bot.logger import FunnelLogger
from trading_funnel.features import FEATURE_NAMES

# ── Config ──

TICK_INTERVAL = 300          # 5 minutes
SCAN_INTERVAL = 600          # 10 minutes between scans
INITIAL_CAPITAL = 5000
POSITION_SIZE = 100
MAX_POSITIONS = 10
MAX_HOLD_BARS = 48
L2_THRESHOLD = 0.60
HARD_STOP = -0.35
SLIPPAGE = 0.03

ACTION_SELL = {"HOLD": 0, "TP_25": 0.25, "TP_50": 0.50, "TP_100": 1.0,
               "SL_25": 0.25, "SL_50": 0.50, "EXIT": 1.0}

log = FunnelLogger()


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
    if not CODEX_API_KEY:
        log.error("No CODEX_API_KEY")
        return []
    try:
        log.l0_scan_start()
        three_hours_ago = int(time.time()) - 3 * 3600
        tokens, count, _ = filter_tokens(
            created_after=three_hours_ago,
            min_mcap=100000, min_holders=100, limit=50,
        )
        log.l0_scan_result(len(tokens), tokens)
        return tokens
    except Exception as e:
        log.error(f"L0 scan failed", e)
        return []


# ── L1: Scam Filter ──

def l1_filter_token(token_data):
    holders = int(token_data.get("holders", 0) or 0)
    mcap = float(token_data.get("marketCap", 0) or 0)
    addr = token_data.get("token", {}).get("address", "")
    sym = token_data.get("token", {}).get("symbol", "?")

    if holders < 200:
        log.l1_filter(addr, sym, False, "low_holders", mcap, holders)
        return False, "low_holders"

    if holders > 0 and mcap / holders > 50000:
        log.l1_filter(addr, sym, False, "high_mcap_per_holder", mcap, holders)
        return False, "high_mcap_per_holder"

    log.l1_filter(addr, sym, True, "passed", mcap, holders)
    return True, "passed"


# ── L2: Entry Signal ──

def l2_predict(address, symbol, l2_model, l2_features):
    price_data = get_token_price(address)
    if price_data is None:
        log.l2_predict(address, symbol, 0, False)
        return False, 0, {}

    mcap = price_data.get("market_cap", 0) or price_data.get("price", 0)
    if mcap <= 0:
        log.l2_predict(address, symbol, 0, False)
        return False, 0, {}

    feat = {f: 0 for f in FEATURE_NAMES}
    feat["mcap_start"] = mcap
    feat["mcap_end"] = mcap
    feat["mcap_max"] = mcap
    feat["mcap_min"] = mcap
    feat["price_vs_vwap"] = 1.0

    # Add price data features
    buy_vol = price_data.get("buy_volume_5m", 0)
    sell_vol = price_data.get("sell_volume_5m", 0)
    if sell_vol > 0:
        feat["volume_trend"] = (buy_vol - sell_vol) / sell_vol

    vec = np.array([[feat.get(c, 0) for c in l2_features]])
    vec = np.nan_to_num(vec, nan=0)

    proba = float(l2_model.predict_proba(vec)[0][1])
    should_enter = proba >= L2_THRESHOLD

    log.l2_predict(address, symbol, proba, should_enter,
                    features={k: round(v, 4) if isinstance(v, float) else v
                             for k, v in feat.items() if v != 0})
    return should_enter, proba, feat


# ── L3: Position Management ──

def l3_predict(address, symbol, entry_price, current_price, remaining, peak_price,
               bars_held, l3_model, l3_features):
    pnl = (current_price - entry_price) / max(entry_price, 1)

    if pnl < HARD_STOP:
        log.l3_action(address, symbol, "EXIT(HARD_STOP)", current_price, pnl * 100,
                       remaining, 0)
        return "EXIT"

    feat = {f: 0 for f in FEATURE_NAMES}
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

    if pnl > 1.0 and pred == "HOLD" and remaining > 0.25:
        pred = "TP_25"

    if pred == "HOLD":
        log.l3_hold(address, symbol, current_price, pnl * 100, remaining)
    else:
        sell_frac = min(ACTION_SELL.get(pred, 0), remaining)
        log.l3_action(address, symbol, pred, current_price, pnl * 100,
                       remaining, remaining - sell_frac,
                       features={k: round(v, 4) if isinstance(v, float) else v
                                for k, v in feat.items() if v != 0})

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

    log.logger.info(f"Engine started: capital=${capital}, L2_threshold={L2_THRESHOLD}")

    while True:
        tick += 1
        tick_start = time.time()
        positions = db.get_positions()
        log.tick_start(tick, capital, len(positions))

        try:
            # ── Phase 1: Periodic Scan (L0 + L1 + L2) ──
            if time.time() - last_scan >= SCAN_INTERVAL:
                last_scan = time.time()

                candidates = l0_scan()
                l0_count = len(candidates)

                l1_passed = []
                l1_filtered = 0
                for c in candidates:
                    passed, reason = l1_filter_token(c)
                    if passed:
                        l1_passed.append(c)
                    else:
                        l1_filtered += 1
                log.l1_summary(len(l1_passed), l1_filtered)

                l2_signals = 0
                l2_rejected = 0
                for c in l1_passed:
                    addr = c["token"]["address"]
                    sym = c["token"].get("symbol", "?")
                    name = c["token"].get("name", "?")
                    mcap = c.get("marketCap", 0)
                    holders = c.get("holders", 0)

                    should_enter, proba, _ = l2_predict(addr, sym, l2_model, l2_features)

                    db.upsert_watchlist(addr, sym, name, mcap, holders, "passed", round(proba, 3))

                    if should_enter:
                        l2_signals += 1
                    else:
                        l2_rejected += 1

                log.l2_summary(l2_signals, l2_rejected)
                db.log_scan(l0_count, len(l1_passed), l1_filtered, l2_signals, l2_rejected)

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

                action = l3_predict(addr, sym, entry_price, current, remaining, peak,
                                     bars, l3_model, l3_features)

                sell_frac = min(ACTION_SELL.get(action, 0), remaining)

                if sell_frac > 0:
                    sell_pnl = pnl * (1 - SLIPPAGE)
                    realized += sell_frac * sell_pnl * pos["position_size"]
                    remaining -= sell_frac
                    db.log_l3_action(addr, sym, action, current, pnl * 100, remaining + sell_frac, remaining)

                # Force exit at max hold
                if bars >= MAX_HOLD_BARS and remaining > 0.01:
                    sell_pnl = pnl * (1 - SLIPPAGE)
                    realized += remaining * sell_pnl * pos["position_size"]
                    db.log_l3_action(addr, sym, "TIME_EXIT", current, pnl * 100, remaining, 0)
                    log.l3_action(addr, sym, "TIME_EXIT", current, pnl * 100, remaining, 0)
                    remaining = 0
                    action = "TIME_EXIT"

                if remaining <= 0.01:
                    total_return = realized / max(pos["position_size"], 1)
                    pnl_usd = realized
                    capital += pos["position_size"] + pnl_usd
                    daily_pnl += pnl_usd
                    daily_trades += 1
                    if pnl_usd > 0:
                        daily_wins += 1

                    db.close_position(addr, current, total_return * 100, pnl_usd, action, [])
                    log.exit(addr, sym, total_return * 100, pnl_usd, action, bars)
                else:
                    db.update_position(addr, peak, bars, remaining, realized)

            # ── Phase 3: Open New Positions ──
            positions = db.get_positions()
            if len(positions) < MAX_POSITIONS:
                watchlist = db.get_watchlist()
                open_addrs = {p["token_address"] for p in positions}

                for item in watchlist:
                    if len(open_addrs) >= MAX_POSITIONS:
                        break
                    addr = item["token_address"]
                    if addr in open_addrs:
                        continue
                    if item["l2_proba"] < L2_THRESHOLD:
                        continue
                    if capital < POSITION_SIZE:
                        break

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
                    log.entry(addr, item["symbol"], mcap, POSITION_SIZE, item["l2_proba"], item["l1_label"])

            # ── Phase 4: Record Equity ──
            if tick % 5 == 0:
                positions = db.get_positions()
                total_equity = capital + sum(p["position_size"] for p in positions)
                db.record_equity(capital, len(positions), total_equity,
                                  daily_pnl, daily_trades, daily_wins)
                log.equity_snapshot(capital, len(positions), total_equity,
                                     daily_pnl, daily_trades, daily_wins)

        except Exception as e:
            log.error(str(e), e)
            traceback.print_exc()

        elapsed = (time.time() - tick_start) * 1000
        log.tick_end(tick, elapsed)

        sleep_time = max(0, TICK_INTERVAL - elapsed / 1000)
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    run()
