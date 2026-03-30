"""Paper trading dashboard — lightweight web UI for monitoring.

Serves a single-page dashboard showing:
- System status (process, ticks, errors)
- Open positions with real-time PnL
- Trade history with full signal snapshots
- Performance stats (win rate, PF, cumulative PnL chart)

Run: python -m trader.dashboard [--port 8050]
"""

import json
import os
import sqlite3
import subprocess
from datetime import datetime, timezone, timedelta
from http.server import HTTPServer, SimpleHTTPRequestHandler

DB_PATH = "data/paper_trading.db"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def query_system_status():
    """Get system health info."""
    # Process check
    result = subprocess.run(["pgrep", "-f", "python -m trader"], capture_output=True, text=True)
    pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
    running = len(pids) > 0

    conn = get_db()

    # Last tick
    cur = conn.execute("SELECT * FROM tick_log ORDER BY id DESC LIMIT 1")
    last_tick = dict(cur.fetchone()) if cur.fetchone() is None else None
    cur = conn.execute("SELECT * FROM tick_log ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    last_tick = dict(row) if row else {}

    # Tick stats
    cur = conn.execute("SELECT COUNT(*) as n FROM tick_log")
    total_ticks = cur.fetchone()[0]

    one_hour_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    cur = conn.execute("SELECT COUNT(*) FROM tick_log WHERE tick_time > ?", (one_hour_ago,))
    recent_ticks = cur.fetchone()[0]

    cur = conn.execute("SELECT COUNT(*) FROM tick_log WHERE tick_time > ? AND errors IS NOT NULL AND errors != ''", (one_hour_ago,))
    recent_errors = cur.fetchone()[0]

    cur = conn.execute("SELECT SUM(signals_generated) FROM tick_log")
    total_signals = cur.fetchone()[0] or 0

    cur = conn.execute("SELECT COUNT(*) FROM watchlist WHERE status='active'")
    watchlist = cur.fetchone()[0]

    conn.close()
    return {
        "running": running,
        "pids": pids,
        "total_ticks": total_ticks,
        "recent_ticks": recent_ticks,
        "recent_errors": recent_errors,
        "total_signals": total_signals,
        "watchlist": watchlist,
        "last_tick": last_tick,
    }


def query_positions():
    """Get open positions."""
    conn = get_db()
    cur = conn.execute("""
        SELECT t.symbol, t.token_address, t.entry_time, t.entry_price,
               t.position_size, t.entry_signals, t.meta_model_score,
               p.peak_price, p.bars_held
        FROM trades t
        JOIN positions p ON t.token_address = p.token_address
        WHERE t.exit_time IS NULL
        ORDER BY t.entry_time DESC
    """)
    positions = [dict(row) for row in cur.fetchall()]
    conn.close()
    return positions


def query_trades():
    """Get all closed trades."""
    conn = get_db()
    cur = conn.execute("""
        SELECT symbol, token_address, entry_time, exit_time,
               entry_price, exit_price, position_size,
               return_pct, pnl_usd, peak_price, bars_held,
               exit_reason, meta_model_score,
               entry_signals, exit_signals
        FROM trades
        WHERE exit_time IS NOT NULL
        ORDER BY exit_time DESC
    """)
    trades = [dict(row) for row in cur.fetchall()]
    conn.close()
    return trades


def query_performance():
    """Get performance summary."""
    conn = get_db()

    cur = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN return_pct > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN return_pct <= 0 THEN 1 ELSE 0 END) as losses,
            AVG(return_pct) as avg_return,
            SUM(pnl_usd) as total_pnl,
            AVG(bars_held) as avg_bars,
            MAX(return_pct) as best_trade,
            MIN(return_pct) as worst_trade
        FROM trades WHERE exit_time IS NOT NULL
    """)
    row = cur.fetchone()
    stats = dict(row) if row else {}

    # Cumulative PnL over time
    cur = conn.execute("""
        SELECT exit_time, pnl_usd, return_pct, symbol
        FROM trades WHERE exit_time IS NOT NULL
        ORDER BY exit_time
    """)
    pnl_history = []
    cumulative = 0
    for row in cur.fetchall():
        cumulative += (row[1] or 0)
        pnl_history.append({
            "time": row[0],
            "pnl": row[1],
            "cumulative": cumulative,
            "return_pct": row[2],
            "symbol": row[3],
        })

    # Exit reason breakdown
    cur = conn.execute("""
        SELECT exit_reason, COUNT(*) as n,
               AVG(return_pct) as avg_ret,
               SUM(CASE WHEN return_pct > 0 THEN 1 ELSE 0 END) as wins
        FROM trades WHERE exit_time IS NOT NULL
        GROUP BY exit_reason
    """)
    exit_reasons = [dict(row) for row in cur.fetchall()]

    # Today
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cur = conn.execute("""
        SELECT COUNT(*) as n, SUM(pnl_usd) as pnl,
               SUM(CASE WHEN return_pct > 0 THEN 1 ELSE 0 END) as wins
        FROM trades WHERE exit_time IS NOT NULL AND date(exit_time) = ?
    """, (today,))
    today_stats = dict(cur.fetchone())

    conn.close()

    total = stats.get("total", 0) or 0
    wins = stats.get("wins", 0) or 0
    losses = stats.get("losses", 0) or 0

    stats["win_rate"] = (wins / total * 100) if total > 0 else 0
    stats["profit_factor"] = 0
    if total > 0:
        conn2 = get_db()
        cur2 = conn2.execute("SELECT return_pct FROM trades WHERE exit_time IS NOT NULL")
        rets = [r[0] for r in cur2.fetchall() if r[0] is not None]
        conn2.close()
        gross_profit = sum(r for r in rets if r > 0)
        gross_loss = abs(sum(r for r in rets if r <= 0))
        stats["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else 0

    return {
        "stats": stats,
        "pnl_history": pnl_history,
        "exit_reasons": exit_reasons,
        "today": today_stats,
    }


def build_html():
    """Generate the dashboard HTML."""
    status = query_system_status()
    positions = query_positions()
    trades = query_trades()
    perf = query_performance()
    stats = perf["stats"]
    today = perf["today"]

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Status color
    status_color = "#2ca02c" if status["running"] else "#d62728"
    status_text = "RUNNING" if status["running"] else "STOPPED"

    # Build positions HTML
    positions_html = ""
    if positions:
        for p in positions:
            signals = json.loads(p["entry_signals"]) if p["entry_signals"] else {}
            entry_time = p["entry_time"][:19] if p["entry_time"] else ""
            positions_html += f"""
            <tr>
                <td><strong>{p['symbol']}</strong></td>
                <td>{entry_time}</td>
                <td>${p['entry_price']:.8f}</td>
                <td>${p['position_size']:.0f}</td>
                <td>{p['bars_held'] or 0} bars</td>
                <td>${p['peak_price']:.8f}</td>
                <td>{signals.get('roc_30m', 'N/A')}</td>
                <td>{signals.get('ofi_30m', 'N/A')}</td>
                <td>{signals.get('ebsw', 'N/A')}</td>
                <td>{signals.get('fisher', 'N/A')}</td>
            </tr>"""
    else:
        positions_html = '<tr><td colspan="10" style="text-align:center;color:#888">No open positions</td></tr>'

    # Build trades HTML
    trades_html = ""
    for t in trades[:50]:  # last 50
        pnl_color = "#2ca02c" if (t["return_pct"] or 0) > 0 else "#d62728"
        entry_time = t["entry_time"][:19] if t["entry_time"] else ""
        exit_time = t["exit_time"][:19] if t["exit_time"] else ""
        ret = t["return_pct"] or 0
        pnl = t["pnl_usd"] or 0

        signals = json.loads(t["entry_signals"]) if t["entry_signals"] else {}
        trades_html += f"""
        <tr>
            <td><strong>{t['symbol']}</strong></td>
            <td>{entry_time}</td>
            <td>{exit_time}</td>
            <td>${t['entry_price']:.8f}</td>
            <td>${t['exit_price']:.8f}</td>
            <td style="color:{pnl_color};font-weight:bold">{ret:+.1f}%</td>
            <td style="color:{pnl_color};font-weight:bold">${pnl:+.2f}</td>
            <td>{t['bars_held'] or 0}</td>
            <td>{t['exit_reason'] or ''}</td>
            <td title='{json.dumps(signals, indent=2)}' style="cursor:pointer">hover</td>
        </tr>"""

    if not trades:
        trades_html = '<tr><td colspan="10" style="text-align:center;color:#888">No closed trades yet</td></tr>'

    # Build PnL chart data
    pnl_data = perf["pnl_history"]
    chart_labels = [p["time"][:16] for p in pnl_data]
    chart_values = [p["cumulative"] for p in pnl_data]

    # Exit reasons
    exit_html = ""
    for er in perf["exit_reasons"]:
        n = er["n"]
        wins = er["wins"] or 0
        wr = (wins / n * 100) if n > 0 else 0
        exit_html += f'<tr><td>{er["exit_reason"]}</td><td>{n}</td><td>{wr:.0f}%</td><td>{er["avg_ret"]:+.1f}%</td></tr>'

    total_trades = stats.get("total", 0) or 0
    win_rate = stats.get("win_rate", 0) or 0
    total_pnl = stats.get("total_pnl", 0) or 0
    avg_return = stats.get("avg_return", 0) or 0
    pf = stats.get("profit_factor", 0) or 0
    best = stats.get("best_trade", 0) or 0
    worst = stats.get("worst_trade", 0) or 0
    today_n = today.get("n", 0) or 0
    today_pnl = today.get("pnl", 0) or 0

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Meme Trader Dashboard</title>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="60">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace; background: #0d1117; color: #c9d1d9; padding: 20px; }}
        h1 {{ color: #58a6ff; margin-bottom: 5px; }}
        h2 {{ color: #8b949e; font-size: 16px; margin: 20px 0 10px; text-transform: uppercase; letter-spacing: 1px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }}
        .card .label {{ color: #8b949e; font-size: 12px; }}
        .card .value {{ font-size: 24px; font-weight: bold; margin-top: 5px; }}
        .green {{ color: #2ea043; }}
        .red {{ color: #f85149; }}
        .yellow {{ color: #d29922; }}
        table {{ width: 100%; border-collapse: collapse; background: #161b22; border-radius: 8px; overflow: hidden; }}
        th {{ background: #21262d; color: #8b949e; text-align: left; padding: 8px 12px; font-size: 12px; text-transform: uppercase; }}
        td {{ padding: 8px 12px; border-top: 1px solid #21262d; font-size: 13px; }}
        tr:hover {{ background: #1c2128; }}
        .timestamp {{ color: #8b949e; font-size: 12px; margin-bottom: 20px; }}
        canvas {{ background: #161b22; border-radius: 8px; border: 1px solid #30363d; }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>Meme Trader Paper Trading</h1>
    <p class="timestamp">Last refresh: {now} (auto-refresh every 60s)</p>

    <h2>System Status</h2>
    <div class="grid">
        <div class="card">
            <div class="label">Process</div>
            <div class="value" style="color:{status_color}">{status_text}</div>
        </div>
        <div class="card">
            <div class="label">Watchlist</div>
            <div class="value">{status['watchlist']}</div>
        </div>
        <div class="card">
            <div class="label">Total Ticks</div>
            <div class="value">{status['total_ticks']}</div>
        </div>
        <div class="card">
            <div class="label">Ticks (1h)</div>
            <div class="value">{status['recent_ticks']}</div>
        </div>
        <div class="card">
            <div class="label">Total Signals</div>
            <div class="value">{status['total_signals']}</div>
        </div>
        <div class="card">
            <div class="label">Errors (1h)</div>
            <div class="value {'red' if status['recent_errors'] > 0 else 'green'}">{status['recent_errors']}</div>
        </div>
    </div>

    <h2>Performance</h2>
    <div class="grid">
        <div class="card">
            <div class="label">Total PnL</div>
            <div class="value {'green' if total_pnl >= 0 else 'red'}">${total_pnl:+.2f}</div>
        </div>
        <div class="card">
            <div class="label">Today PnL</div>
            <div class="value {'green' if today_pnl >= 0 else 'red'}">${today_pnl:+.2f}</div>
        </div>
        <div class="card">
            <div class="label">Win Rate</div>
            <div class="value">{win_rate:.0f}%</div>
        </div>
        <div class="card">
            <div class="label">Profit Factor</div>
            <div class="value {'green' if pf >= 1 else 'red'}">{pf:.2f}</div>
        </div>
        <div class="card">
            <div class="label">Total Trades</div>
            <div class="value">{total_trades}</div>
        </div>
        <div class="card">
            <div class="label">Avg Return</div>
            <div class="value {'green' if avg_return >= 0 else 'red'}">{avg_return:+.2f}%</div>
        </div>
        <div class="card">
            <div class="label">Best Trade</div>
            <div class="value green">{best:+.1f}%</div>
        </div>
        <div class="card">
            <div class="label">Worst Trade</div>
            <div class="value red">{worst:+.1f}%</div>
        </div>
    </div>

    {"<canvas id='pnlChart' height='80'></canvas>" if pnl_data else ""}

    <h2>Open Positions ({len(positions)})</h2>
    <table>
        <tr><th>Token</th><th>Entry Time</th><th>Entry Price</th><th>Size</th><th>Bars</th><th>Peak</th><th>ROC</th><th>OFI</th><th>EBSW</th><th>Fisher</th></tr>
        {positions_html}
    </table>

    <h2>Trade History ({len(trades)})</h2>
    <table>
        <tr><th>Token</th><th>Entry</th><th>Exit</th><th>Entry $</th><th>Exit $</th><th>Return</th><th>PnL</th><th>Bars</th><th>Reason</th><th>Signals</th></tr>
        {trades_html}
    </table>

    {"<h2>Exit Reasons</h2><table><tr><th>Reason</th><th>Count</th><th>Win Rate</th><th>Avg Return</th></tr>" + exit_html + "</table>" if exit_html else ""}

    <script>
    {"" if not pnl_data else f'''
    new Chart(document.getElementById('pnlChart'), {{
        type: 'line',
        data: {{
            labels: {json.dumps(chart_labels)},
            datasets: [{{
                label: 'Cumulative PnL ($)',
                data: {json.dumps(chart_values)},
                borderColor: '#58a6ff',
                backgroundColor: 'rgba(88,166,255,0.1)',
                fill: true,
                tension: 0.3,
            }}]
        }},
        options: {{
            plugins: {{ legend: {{ labels: {{ color: '#c9d1d9' }} }} }},
            scales: {{
                x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
                y: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }}
            }}
        }}
    }});
    '''}
    </script>
</body>
</html>"""
    return html


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            html = build_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # suppress access logs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print(f"Dashboard running at http://localhost:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
