"""Paper trading health check — runs every hour, reports status.

Usage:
    python -m trader.healthcheck          # one-time check
    python -m trader.healthcheck --loop   # continuous hourly check
"""

import argparse
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta

DB_PATH = "data/paper_trading.db"
LOG_PATH = "logs/trader.log"


def check_process():
    """Check if the trader process is running."""
    result = subprocess.run(
        ["pgrep", "-f", "python -m trader"],
        capture_output=True, text=True,
    )
    pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
    # Filter out this healthcheck process
    my_pid = str(os.getpid())
    pids = [p for p in pids if p != my_pid]
    return len(pids) > 0, pids


def check_database():
    """Check database for recent activity."""
    if not os.path.isfile(DB_PATH):
        return {"error": "DB not found"}

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Last tick
    cur = conn.execute("SELECT * FROM tick_log ORDER BY id DESC LIMIT 1")
    last_tick = dict(cur.fetchone()) if cur.fetchone() is None else None
    cur = conn.execute("SELECT * FROM tick_log ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    last_tick = dict(row) if row else None

    # Total ticks
    cur = conn.execute("SELECT COUNT(*) as n FROM tick_log")
    total_ticks = cur.fetchone()[0]

    # Ticks in last hour
    one_hour_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    cur = conn.execute("SELECT COUNT(*) as n FROM tick_log WHERE tick_time > ?", (one_hour_ago,))
    recent_ticks = cur.fetchone()[0]

    # Total signals ever
    cur = conn.execute("SELECT SUM(signals_generated) as n FROM tick_log")
    total_signals = cur.fetchone()[0] or 0

    # Open trades
    cur = conn.execute("SELECT COUNT(*) as n FROM trades WHERE exit_time IS NULL")
    open_trades = cur.fetchone()[0]

    # Closed trades
    cur = conn.execute("SELECT COUNT(*) as n FROM trades WHERE exit_time IS NOT NULL")
    closed_trades = cur.fetchone()[0]

    # Today's PnL
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cur = conn.execute(
        "SELECT SUM(pnl_usd) as pnl, COUNT(*) as n FROM trades WHERE exit_time IS NOT NULL AND date(exit_time) = ?",
        (today,)
    )
    row = cur.fetchone()
    today_pnl = row[0] or 0
    today_trades = row[1] or 0

    # All-time PnL
    cur = conn.execute("SELECT SUM(pnl_usd) as pnl FROM trades WHERE exit_time IS NOT NULL")
    total_pnl = cur.fetchone()[0] or 0

    # Win rate
    cur = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE exit_time IS NOT NULL AND return_pct > 0"
    )
    wins = cur.fetchone()[0]
    win_rate = (wins / closed_trades * 100) if closed_trades > 0 else 0

    # Watchlist
    cur = conn.execute("SELECT COUNT(*) FROM watchlist WHERE status='active'")
    watchlist_size = cur.fetchone()[0]

    # Errors in last hour
    cur = conn.execute(
        "SELECT COUNT(*) FROM tick_log WHERE tick_time > ? AND errors IS NOT NULL AND errors != ''",
        (one_hour_ago,)
    )
    recent_errors = cur.fetchone()[0]

    conn.close()

    return {
        "total_ticks": total_ticks,
        "recent_ticks_1h": recent_ticks,
        "total_signals": total_signals,
        "watchlist_size": watchlist_size,
        "open_trades": open_trades,
        "closed_trades": closed_trades,
        "today_trades": today_trades,
        "today_pnl": today_pnl,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "wins": wins,
        "recent_errors": recent_errors,
        "last_tick": last_tick,
    }


def check_logs():
    """Check recent log entries."""
    if not os.path.isfile(LOG_PATH):
        return {"error": "Log not found"}

    # Get last 5 lines
    result = subprocess.run(
        ["tail", "-5", LOG_PATH],
        capture_output=True, text=True,
    )
    return {"last_lines": result.stdout.strip()}


def run_check():
    """Run full health check and print report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*70}")
    print(f"  PAPER TRADING HEALTH CHECK — {now}")
    print(f"{'='*70}")

    # Process
    running, pids = check_process()
    status = f"RUNNING (PID: {', '.join(pids)})" if running else "NOT RUNNING"
    print(f"\n  Process:       {status}")

    if not running:
        print(f"  WARNING: Trader process not running! Restart with:")
        print(f"    tmux new-session -d -s trader 'source .venv/bin/activate && python -m trader'")

    # Database
    db = check_database()
    if "error" in db:
        print(f"  Database:      {db['error']}")
    else:
        print(f"  Watchlist:     {db['watchlist_size']} tokens")
        print(f"  Total ticks:   {db['total_ticks']} (last hour: {db['recent_ticks_1h']})")
        print(f"  Total signals: {db['total_signals']}")
        print(f"  Errors (1h):   {db['recent_errors']}")
        print(f"{'─'*70}")
        print(f"  Open trades:   {db['open_trades']}")
        print(f"  Closed trades: {db['closed_trades']} (today: {db['today_trades']})")
        print(f"  Win rate:      {db['win_rate']:.0f}% ({db['wins']}/{db['closed_trades']})")
        print(f"  Today PnL:     ${db['today_pnl']:+.2f}")
        print(f"  Total PnL:     ${db['total_pnl']:+.2f}")

        if db["recent_ticks_1h"] == 0 and running:
            print(f"\n  WARNING: No ticks in last hour but process is running — may be stuck")

        if db["recent_errors"] > 0:
            print(f"\n  WARNING: {db['recent_errors']} errors in the last hour")

    # Last log lines
    logs = check_logs()
    if "last_lines" in logs:
        print(f"{'─'*70}")
        print(f"  Recent log:")
        for line in logs["last_lines"].split("\n"):
            print(f"    {line}")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Paper trading health check")
    parser.add_argument("--loop", action="store_true", help="Run continuously every hour")
    parser.add_argument("--interval", type=int, default=3600, help="Check interval in seconds (default: 3600)")
    args = parser.parse_args()

    if args.loop:
        print(f"Starting hourly health check (every {args.interval}s)...")
        while True:
            try:
                run_check()
            except Exception as e:
                print(f"Health check error: {e}")
            time.sleep(args.interval)
    else:
        run_check()


if __name__ == "__main__":
    main()
