"""Scalper V4 database layer — SQLite with WAL mode."""

import json
import os
import sqlite3
from datetime import datetime, timezone

from scalper import config

_conn = None


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
        _conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA busy_timeout=5000")
        _conn.row_factory = sqlite3.Row
        _init_schema(_conn)
    return _conn


def _init_schema(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS scalper_watchlist (
        token_address TEXT PRIMARY KEY,
        symbol TEXT,
        name TEXT,
        discovered_at TEXT NOT NULL,
        created_at_ts INTEGER,
        last_mcap REAL,
        last_holders INTEGER,
        scam_score REAL DEFAULT 0,
        scam_passed INTEGER DEFAULT 0,
        status TEXT DEFAULT 'active',
        removed_reason TEXT
    );

    CREATE TABLE IF NOT EXISTS scalper_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        token_address TEXT NOT NULL,
        symbol TEXT,
        entry_time TEXT NOT NULL,
        entry_price REAL NOT NULL,
        entry_mcap REAL,
        entry_holders INTEGER,
        entry_signals JSON,
        position_size REAL,
        exit_time TEXT,
        exit_price REAL,
        exit_mcap REAL,
        exit_reason TEXT,
        return_pct REAL,
        pnl_usd REAL,
        peak_mcap REAL,
        bars_held INTEGER,
        jupiter_entry_impact REAL,
        jupiter_exit_impact REAL,
        created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_scalper_trades_entry ON scalper_trades(entry_time);
    CREATE INDEX IF NOT EXISTS idx_scalper_trades_addr ON scalper_trades(token_address);

    CREATE TABLE IF NOT EXISTS scalper_positions (
        token_address TEXT PRIMARY KEY,
        symbol TEXT,
        entry_time TEXT NOT NULL,
        entry_price REAL NOT NULL,
        entry_mcap REAL,
        position_size REAL,
        peak_mcap REAL,
        bars_held INTEGER DEFAULT 0,
        trade_id INTEGER REFERENCES scalper_trades(id)
    );

    CREATE TABLE IF NOT EXISTS scalper_equity (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        capital REAL NOT NULL,
        open_positions INTEGER,
        unrealized_pnl REAL DEFAULT 0,
        total_equity REAL NOT NULL,
        daily_pnl REAL DEFAULT 0,
        daily_trades INTEGER DEFAULT 0,
        daily_wins INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS scalper_tick_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tick_time TEXT NOT NULL,
        tick_number INTEGER,
        watchlist_size INTEGER,
        tokens_scanned INTEGER,
        signals_generated INTEGER,
        positions_open INTEGER,
        daily_pnl REAL,
        tick_duration_ms INTEGER
    );

    CREATE TABLE IF NOT EXISTS scalper_discovery_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        tokens_found INTEGER,
        tokens_new INTEGER,
        tokens_passed_scam_filter INTEGER,
        tokens_blocked INTEGER
    );
    """)
    conn.commit()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# === Watchlist ===

def add_to_watchlist(address, symbol, name, created_at_ts, mcap=None, holders=None, scam_passed=False):
    conn = get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO scalper_watchlist
        (token_address, symbol, name, discovered_at, created_at_ts, last_mcap, last_holders, scam_passed, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active')
    """, (address, symbol, name, now_iso(), created_at_ts, mcap, holders, int(scam_passed)))
    conn.commit()


def get_active_watchlist():
    return get_conn().execute(
        "SELECT * FROM scalper_watchlist WHERE status = 'active' ORDER BY discovered_at DESC"
    ).fetchall()


def update_watchlist_token(address, mcap=None, holders=None):
    conn = get_conn()
    if mcap is not None:
        conn.execute("UPDATE scalper_watchlist SET last_mcap=? WHERE token_address=?", (mcap, address))
    if holders is not None:
        conn.execute("UPDATE scalper_watchlist SET last_holders=? WHERE token_address=?", (holders, address))
    conn.commit()


def remove_from_watchlist(address, reason="expired"):
    conn = get_conn()
    conn.execute("UPDATE scalper_watchlist SET status='removed', removed_reason=? WHERE token_address=?",
                 (reason, address))
    conn.commit()


def watchlist_count():
    return get_conn().execute("SELECT COUNT(*) FROM scalper_watchlist WHERE status='active'").fetchone()[0]


# === Trades ===

def open_trade(address, symbol, entry_price, entry_mcap, entry_holders, position_size, signals=None, jupiter_impact=None):
    conn = get_conn()
    cur = conn.execute("""
        INSERT INTO scalper_trades
        (token_address, symbol, entry_time, entry_price, entry_mcap, entry_holders,
         entry_signals, position_size, jupiter_entry_impact)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (address, symbol, now_iso(), entry_price, entry_mcap, entry_holders,
          json.dumps(signals) if signals else None, position_size, jupiter_impact))
    trade_id = cur.lastrowid

    conn.execute("""
        INSERT OR REPLACE INTO scalper_positions
        (token_address, symbol, entry_time, entry_price, entry_mcap, position_size, peak_mcap, trade_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (address, symbol, now_iso(), entry_price, entry_mcap, position_size, entry_mcap, trade_id))
    conn.commit()
    return trade_id


def close_trade(address, exit_price, exit_mcap, exit_reason, return_pct, pnl_usd,
                peak_mcap, bars_held, jupiter_impact=None):
    conn = get_conn()
    pos = conn.execute("SELECT trade_id FROM scalper_positions WHERE token_address=?", (address,)).fetchone()
    if not pos:
        return
    trade_id = pos["trade_id"]

    conn.execute("""
        UPDATE scalper_trades SET
            exit_time=?, exit_price=?, exit_mcap=?, exit_reason=?,
            return_pct=?, pnl_usd=?, peak_mcap=?, bars_held=?, jupiter_exit_impact=?
        WHERE id=?
    """, (now_iso(), exit_price, exit_mcap, exit_reason, return_pct, pnl_usd,
          peak_mcap, bars_held, jupiter_impact, trade_id))

    conn.execute("DELETE FROM scalper_positions WHERE token_address=?", (address,))
    conn.commit()


def get_open_positions():
    return get_conn().execute("SELECT * FROM scalper_positions").fetchall()


def get_position(address):
    return get_conn().execute("SELECT * FROM scalper_positions WHERE token_address=?", (address,)).fetchone()


def update_position(address, peak_mcap=None, bars_held=None):
    conn = get_conn()
    if peak_mcap is not None:
        conn.execute("UPDATE scalper_positions SET peak_mcap=? WHERE token_address=?", (peak_mcap, address))
    if bars_held is not None:
        conn.execute("UPDATE scalper_positions SET bars_held=? WHERE token_address=?", (bars_held, address))
    conn.commit()


def get_closed_trades(limit=100):
    return get_conn().execute("""
        SELECT * FROM scalper_trades WHERE exit_time IS NOT NULL
        ORDER BY exit_time DESC LIMIT ?
    """, (limit,)).fetchall()


def get_all_trades():
    return get_conn().execute("SELECT * FROM scalper_trades ORDER BY entry_time").fetchall()


def open_positions_count():
    return get_conn().execute("SELECT COUNT(*) FROM scalper_positions").fetchone()[0]


# === Equity ===

def record_equity(capital, open_positions, unrealized_pnl, total_equity, daily_pnl, daily_trades, daily_wins):
    conn = get_conn()
    conn.execute("""
        INSERT INTO scalper_equity
        (timestamp, capital, open_positions, unrealized_pnl, total_equity, daily_pnl, daily_trades, daily_wins)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (now_iso(), capital, open_positions, unrealized_pnl, total_equity, daily_pnl, daily_trades, daily_wins))
    conn.commit()


def get_equity_history(limit=1000):
    return get_conn().execute("SELECT * FROM scalper_equity ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()


# === Tick Log ===

def log_tick(tick_number, watchlist_size, tokens_scanned, signals, positions, daily_pnl, duration_ms):
    conn = get_conn()
    conn.execute("""
        INSERT INTO scalper_tick_log
        (tick_time, tick_number, watchlist_size, tokens_scanned, signals_generated,
         positions_open, daily_pnl, tick_duration_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (now_iso(), tick_number, watchlist_size, tokens_scanned, signals, positions, daily_pnl, duration_ms))
    conn.commit()


# === Discovery Log ===

def log_discovery(found, new, passed, blocked):
    conn = get_conn()
    conn.execute("""
        INSERT INTO scalper_discovery_log (timestamp, tokens_found, tokens_new, tokens_passed_scam_filter, tokens_blocked)
        VALUES (?, ?, ?, ?, ?)
    """, (now_iso(), found, new, passed, blocked))
    conn.commit()
