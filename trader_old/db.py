"""SQLite database layer for paper trading.

Tables: trades, watchlist, tick_log, positions
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

from trader.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token_address TEXT NOT NULL,
    symbol TEXT,
    entry_time TIMESTAMP NOT NULL,
    entry_price REAL NOT NULL,
    entry_signals JSON,
    position_size REAL NOT NULL,
    exit_time TIMESTAMP,
    exit_price REAL,
    exit_reason TEXT,
    exit_signals JSON,
    return_pct REAL,
    pnl_usd REAL,
    peak_price REAL,
    bars_held INTEGER,
    meta_model_score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS watchlist (
    token_address TEXT PRIMARY KEY,
    symbol TEXT,
    name TEXT,
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP,
    last_mcap REAL,
    last_volume REAL,
    status TEXT DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS tick_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tick_time TIMESTAMP NOT NULL,
    tick_number INTEGER,
    watchlist_size INTEGER,
    tokens_scanned INTEGER,
    signals_generated INTEGER,
    signals_filtered INTEGER,
    open_positions INTEGER,
    tick_duration_ms INTEGER,
    daily_pnl REAL,
    errors TEXT
);

CREATE TABLE IF NOT EXISTS positions (
    token_address TEXT PRIMARY KEY,
    symbol TEXT,
    entry_time TIMESTAMP NOT NULL,
    entry_price REAL NOT NULL,
    position_size REAL NOT NULL,
    peak_price REAL NOT NULL,
    bars_held INTEGER DEFAULT 0,
    trade_id INTEGER REFERENCES trades(id)
);

CREATE INDEX IF NOT EXISTS idx_trades_address ON trades(token_address);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time);
"""


class TradingDB:
    """SQLite database for paper trading records."""

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def _now(self):
        return datetime.now(timezone.utc).isoformat()

    # === Trades ===

    def open_trade(self, token_address, symbol, entry_price, position_size,
                   entry_signals=None, meta_score=None):
        """Record a new trade entry. Returns trade_id."""
        cur = self.conn.execute(
            """INSERT INTO trades (token_address, symbol, entry_time, entry_price,
               entry_signals, position_size, meta_model_score, peak_price, bars_held)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)""",
            (token_address, symbol, self._now(), entry_price,
             json.dumps(entry_signals) if entry_signals else None,
             position_size, meta_score, entry_price)
        )
        self.conn.commit()
        return cur.lastrowid

    def close_trade(self, trade_id, exit_price, exit_reason, exit_signals=None,
                    return_pct=None, pnl_usd=None, peak_price=None, bars_held=None):
        """Record trade exit."""
        self.conn.execute(
            """UPDATE trades SET exit_time=?, exit_price=?, exit_reason=?,
               exit_signals=?, return_pct=?, pnl_usd=?, peak_price=?, bars_held=?
               WHERE id=?""",
            (self._now(), exit_price, exit_reason,
             json.dumps(exit_signals) if exit_signals else None,
             return_pct, pnl_usd, peak_price, bars_held, trade_id)
        )
        self.conn.commit()

    def get_open_trades(self):
        """Get all trades without exit_time."""
        cur = self.conn.execute("SELECT * FROM trades WHERE exit_time IS NULL")
        return [dict(row) for row in cur.fetchall()]

    def get_trades_today(self):
        """Get all trades closed today."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cur = self.conn.execute(
            "SELECT * FROM trades WHERE exit_time IS NOT NULL AND date(exit_time) = ?",
            (today,)
        )
        return [dict(row) for row in cur.fetchall()]

    def get_daily_pnl(self):
        """Get total PnL for today's closed trades."""
        trades = self.get_trades_today()
        return sum(t["pnl_usd"] or 0 for t in trades)

    def get_recent_loss_time(self, token_address):
        """Get the most recent loss exit time for a token."""
        cur = self.conn.execute(
            """SELECT exit_time FROM trades WHERE token_address=?
               AND return_pct < 0 ORDER BY exit_time DESC LIMIT 1""",
            (token_address,)
        )
        row = cur.fetchone()
        return row["exit_time"] if row else None

    def get_all_trades(self):
        """Get all trades for analysis."""
        cur = self.conn.execute("SELECT * FROM trades ORDER BY entry_time")
        return [dict(row) for row in cur.fetchall()]

    # === Positions ===

    def save_position(self, token_address, symbol, entry_time, entry_price,
                      position_size, peak_price, bars_held, trade_id):
        self.conn.execute(
            """INSERT OR REPLACE INTO positions
               (token_address, symbol, entry_time, entry_price, position_size,
                peak_price, bars_held, trade_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (token_address, symbol, entry_time, entry_price,
             position_size, peak_price, bars_held, trade_id)
        )
        self.conn.commit()

    def remove_position(self, token_address):
        self.conn.execute("DELETE FROM positions WHERE token_address=?", (token_address,))
        self.conn.commit()

    def get_positions(self):
        cur = self.conn.execute("SELECT * FROM positions")
        return [dict(row) for row in cur.fetchall()]

    # === Watchlist ===

    def add_to_watchlist(self, token_address, symbol, name=None):
        self.conn.execute(
            """INSERT OR IGNORE INTO watchlist (token_address, symbol, name, discovered_at, status)
               VALUES (?, ?, ?, ?, 'active')""",
            (token_address, symbol, name, self._now())
        )
        self.conn.commit()

    def update_watchlist_token(self, token_address, mcap=None, volume=None):
        self.conn.execute(
            "UPDATE watchlist SET last_seen_at=?, last_mcap=?, last_volume=? WHERE token_address=?",
            (self._now(), mcap, volume, token_address)
        )
        self.conn.commit()

    def remove_from_watchlist(self, token_address, reason="removed"):
        self.conn.execute(
            "UPDATE watchlist SET status=? WHERE token_address=?",
            (reason, token_address)
        )
        self.conn.commit()

    def get_active_watchlist(self):
        cur = self.conn.execute("SELECT * FROM watchlist WHERE status='active'")
        return [dict(row) for row in cur.fetchall()]

    # === Tick Log ===

    def log_tick(self, tick_number, watchlist_size, tokens_scanned,
                 signals_generated, signals_filtered, open_positions,
                 tick_duration_ms, daily_pnl, errors=None):
        self.conn.execute(
            """INSERT INTO tick_log (tick_time, tick_number, watchlist_size,
               tokens_scanned, signals_generated, signals_filtered,
               open_positions, tick_duration_ms, daily_pnl, errors)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (self._now(), tick_number, watchlist_size, tokens_scanned,
             signals_generated, signals_filtered, open_positions,
             tick_duration_ms, daily_pnl, errors)
        )
        self.conn.commit()

    # === Stats ===

    def get_summary_stats(self):
        """Get overall paper trading statistics."""
        cur = self.conn.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN return_pct > 0 THEN 1 ELSE 0 END) as wins,
                AVG(return_pct) as avg_return,
                SUM(pnl_usd) as total_pnl,
                AVG(bars_held) as avg_bars
            FROM trades WHERE exit_time IS NOT NULL
        """)
        return dict(cur.fetchone())

    def close(self):
        self.conn.close()
