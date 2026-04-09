"""Database for Token Discovery pipeline."""

import json
import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.join("data", "token_discovery.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS candidate_pool (
        address TEXT PRIMARY KEY,
        symbol TEXT,
        name TEXT,
        cluster_rank INTEGER,
        cluster_name TEXT,
        first_seen TEXT,
        mcap_at_discovery REAL DEFAULT 0,
        holders_at_discovery INTEGER DEFAULT 0,
        vol4h_at_discovery REAL DEFAULT 0,
        change4h_at_discovery REAL DEFAULT 0,
        liquidity_at_discovery REAL DEFAULT 0,
        buy4h INTEGER DEFAULT 0,
        sell4h INTEGER DEFAULT 0,
        ath REAL DEFAULT 0,
        rise_hours REAL DEFAULT 0,
        decay_hours REAL DEFAULT 0,
        price_roc REAL DEFAULT 0,
        volume_roc REAL DEFAULT 0,
        holder_roc REAL DEFAULT 0,
        holders_at_ath REAL DEFAULT 0,
        total_hours REAL DEFAULT 0,
        rise_pct REAL DEFAULT 0,
        lifetime_hours REAL DEFAULT 0,
        status TEXT DEFAULT 'active'
    );

    CREATE TABLE IF NOT EXISTS scan_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        codex_count INTEGER DEFAULT 0,
        gmgn_fetched INTEGER DEFAULT 0,
        classified INTEGER DEFAULT 0,
        new_candidates INTEGER DEFAULT 0,
        total_pool INTEGER DEFAULT 0,
        token_details TEXT DEFAULT '[]'
    );
    """)
    conn.commit()
    conn.close()


def add_candidate(address, symbol, name, cluster_rank, cluster_name,
                   mcap, holders, vol4h, change4h, liquidity, buy4h, sell4h,
                   ath, rise_hours, decay_hours, price_roc, volume_roc,
                   holder_roc, holders_at_ath, total_hours, rise_pct,
                   lifetime_hours=0):
    """Add or update a candidate. Returns True if new, False if existing."""
    conn = get_conn()
    existing = conn.execute("SELECT address FROM candidate_pool WHERE address=?", (address,)).fetchone()

    if existing:
        conn.close()
        return False

    conn.execute("""
        INSERT INTO candidate_pool (
            address, symbol, name, cluster_rank, cluster_name, first_seen,
            mcap_at_discovery, holders_at_discovery, vol4h_at_discovery,
            change4h_at_discovery, liquidity_at_discovery, buy4h, sell4h,
            ath, rise_hours, decay_hours, price_roc, volume_roc,
            holder_roc, holders_at_ath, total_hours, rise_pct, lifetime_hours
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (address, symbol, name, cluster_rank, cluster_name,
          datetime.now(timezone.utc).isoformat(),
          mcap, holders, vol4h, change4h, liquidity, buy4h, sell4h,
          ath, rise_hours, decay_hours, price_roc, volume_roc,
          holder_roc, holders_at_ath, total_hours, rise_pct, lifetime_hours))
    conn.commit()
    conn.close()
    return True


def get_candidates(status="active"):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM candidate_pool WHERE status=? ORDER BY first_seen DESC",
        (status,)
    ).fetchall()
    conn.close()
    return rows


def get_pool_size():
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) FROM candidate_pool WHERE status='active'").fetchone()[0]
    conn.close()
    return n


def log_scan(codex_count, gmgn_fetched, classified, new_candidates, total_pool, token_details):
    conn = get_conn()
    conn.execute(
        "INSERT INTO scan_log (timestamp, codex_count, gmgn_fetched, classified, new_candidates, total_pool, token_details) VALUES (?,?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), codex_count, gmgn_fetched, classified,
         new_candidates, total_pool, json.dumps(token_details, ensure_ascii=False))
    )
    conn.commit()
    conn.close()


def get_scan_logs(limit=50):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM scan_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return rows
