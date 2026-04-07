"""Database for funnel bot paper trading."""

import json
import os
import sqlite3
from datetime import datetime, timezone


DB_PATH = os.path.join("data", "funnel_bot.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS funnel_scan (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        l0_candidates INTEGER DEFAULT 0,
        l1_passed INTEGER DEFAULT 0,
        l1_filtered INTEGER DEFAULT 0,
        l2_signals INTEGER DEFAULT 0,
        l2_rejected INTEGER DEFAULT 0,
        details TEXT DEFAULT '{}'
    );

    CREATE TABLE IF NOT EXISTS funnel_watchlist (
        token_address TEXT PRIMARY KEY,
        symbol TEXT,
        name TEXT,
        discovered_at TEXT,
        last_mcap REAL DEFAULT 0,
        last_holders INTEGER DEFAULT 0,
        l1_label TEXT DEFAULT '',
        l2_proba REAL DEFAULT 0,
        status TEXT DEFAULT 'watching'
    );

    CREATE TABLE IF NOT EXISTS funnel_positions (
        token_address TEXT PRIMARY KEY,
        symbol TEXT,
        entry_time TEXT,
        entry_price REAL,
        position_size REAL,
        remaining_pct REAL DEFAULT 1.0,
        peak_price REAL,
        bars_held INTEGER DEFAULT 0,
        realized_pnl REAL DEFAULT 0,
        trade_id INTEGER,
        l2_proba REAL DEFAULT 0,
        l1_label TEXT DEFAULT ''
    );

    CREATE TABLE IF NOT EXISTS funnel_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        token_address TEXT,
        symbol TEXT,
        entry_time TEXT,
        entry_price REAL,
        entry_mcap REAL,
        entry_holders INTEGER DEFAULT 0,
        position_size REAL,
        l1_label TEXT DEFAULT '',
        l2_proba REAL DEFAULT 0,
        exit_time TEXT,
        exit_price REAL,
        return_pct REAL,
        pnl_usd REAL,
        peak_price REAL,
        bars_held INTEGER DEFAULT 0,
        exit_reason TEXT DEFAULT '',
        l3_actions TEXT DEFAULT '[]'
    );

    CREATE TABLE IF NOT EXISTS funnel_equity (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        capital REAL,
        open_positions INTEGER,
        total_equity REAL,
        daily_pnl REAL,
        daily_trades INTEGER,
        daily_wins INTEGER
    );

    CREATE TABLE IF NOT EXISTS funnel_l3_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        token_address TEXT,
        symbol TEXT,
        action TEXT,
        price REAL,
        pnl_pct REAL,
        remaining_before REAL,
        remaining_after REAL,
        reason TEXT DEFAULT ''
    );
    """)
    conn.commit()
    conn.close()


# ── Scan Log ──

def log_scan(l0, l1_pass, l1_filter, l2_sig, l2_rej, details=None):
    conn = get_conn()
    conn.execute(
        "INSERT INTO funnel_scan (timestamp, l0_candidates, l1_passed, l1_filtered, l2_signals, l2_rejected, details) VALUES (?,?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), l0, l1_pass, l1_filter, l2_sig, l2_rej, json.dumps(details or {}))
    )
    conn.commit()
    conn.close()


# ── Watchlist ──

def upsert_watchlist(address, symbol, name, mcap, holders, l1_label, l2_proba):
    conn = get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO funnel_watchlist
        (token_address, symbol, name, discovered_at, last_mcap, last_holders, l1_label, l2_proba, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'watching')
    """, (address, symbol, name, datetime.now(timezone.utc).isoformat(), mcap, holders, l1_label, l2_proba))
    conn.commit()
    conn.close()


def get_watchlist():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM funnel_watchlist WHERE status = 'watching' ORDER BY l2_proba DESC").fetchall()
    conn.close()
    return rows


# ── Positions ──

def open_position(address, symbol, entry_price, position_size, l2_proba, l1_label):
    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO funnel_trades (token_address, symbol, entry_time, entry_price, entry_mcap, position_size, l1_label, l2_proba) VALUES (?,?,?,?,?,?,?,?)",
        (address, symbol, now, entry_price, entry_price, position_size, l1_label, l2_proba)
    )
    trade_id = cur.lastrowid
    conn.execute("""
        INSERT OR REPLACE INTO funnel_positions
        (token_address, symbol, entry_time, entry_price, position_size, remaining_pct, peak_price, bars_held, realized_pnl, trade_id, l2_proba, l1_label)
        VALUES (?,?,?,?,?,1.0,?,0,0,?,?,?)
    """, (address, symbol, now, entry_price, position_size, entry_price, trade_id, l2_proba, l1_label))
    conn.commit()
    conn.close()
    return trade_id


def get_positions():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM funnel_positions").fetchall()
    conn.close()
    return rows


def update_position(address, peak_price, bars_held, remaining_pct, realized_pnl):
    conn = get_conn()
    conn.execute(
        "UPDATE funnel_positions SET peak_price=?, bars_held=?, remaining_pct=?, realized_pnl=? WHERE token_address=?",
        (peak_price, bars_held, remaining_pct, realized_pnl, address)
    )
    conn.commit()
    conn.close()


def close_position(address, exit_price, return_pct, pnl_usd, exit_reason, l3_actions):
    conn = get_conn()
    pos = conn.execute("SELECT * FROM funnel_positions WHERE token_address=?", (address,)).fetchone()
    if pos:
        conn.execute("""
            UPDATE funnel_trades SET exit_time=?, exit_price=?, return_pct=?, pnl_usd=?,
            peak_price=?, bars_held=?, exit_reason=?, l3_actions=?
            WHERE id=?
        """, (datetime.now(timezone.utc).isoformat(), exit_price, return_pct, pnl_usd,
              pos["peak_price"], pos["bars_held"], exit_reason, json.dumps(l3_actions), pos["trade_id"]))
        conn.execute("DELETE FROM funnel_positions WHERE token_address=?", (address,))
    conn.commit()
    conn.close()


def log_l3_action(address, symbol, action, price, pnl_pct, remaining_before, remaining_after):
    conn = get_conn()
    conn.execute(
        "INSERT INTO funnel_l3_log (timestamp, token_address, symbol, action, price, pnl_pct, remaining_before, remaining_after) VALUES (?,?,?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), address, symbol, action, price, pnl_pct, remaining_before, remaining_after)
    )
    conn.commit()
    conn.close()


# ── Equity ──

def record_equity(capital, open_pos, total_equity, daily_pnl, daily_trades, daily_wins):
    conn = get_conn()
    conn.execute(
        "INSERT INTO funnel_equity (timestamp, capital, open_positions, total_equity, daily_pnl, daily_trades, daily_wins) VALUES (?,?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), capital, open_pos, total_equity, daily_pnl, daily_trades, daily_wins)
    )
    conn.commit()
    conn.close()


# ── Queries ──

def get_trades(limit=100):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM funnel_trades ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return rows


def get_equity_history(limit=500):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM funnel_equity ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return list(reversed(rows))


def get_scan_history(limit=100):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM funnel_scan ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return list(reversed(rows))


def get_l3_log(address=None, limit=50):
    conn = get_conn()
    if address:
        rows = conn.execute("SELECT * FROM funnel_l3_log WHERE token_address=? ORDER BY id DESC LIMIT ?", (address, limit)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM funnel_l3_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return rows
