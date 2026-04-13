"""Scalper V4 Dashboard — Streamlit visualization.

Layout:
  Header: Strategy status, capital, daily PnL
  Row 1: Pipeline funnel (discovered → filtered → signals → trades)
  Row 2: Live positions table + Equity curve
  Row 3: Trade history table + Performance stats
  Row 4: Per-exit-reason breakdown + Win/loss distribution
"""

import sys
import sqlite3
import os

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from scalper import config

DB_PATH = config.DB_PATH


def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def query_df(sql, params=None):
    conn = get_db()
    return pd.read_sql_query(sql, conn, params=params or [])


# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Scalper V4 Dashboard", layout="wide")

# ── Header: Strategy Status ──────────────────────────────────────────────────

st.title("Scalper V4 — Pump Rider")

# Key metrics
trades_df = query_df("SELECT * FROM scalper_trades ORDER BY entry_time")
closed_df = trades_df[trades_df["exit_time"].notna()]
open_df = query_df("SELECT * FROM scalper_positions")
equity_df = query_df("SELECT * FROM scalper_equity ORDER BY timestamp")
watchlist_df = query_df("SELECT * FROM scalper_watchlist WHERE status='active'")
discovery_df = query_df("SELECT * FROM scalper_discovery_log ORDER BY timestamp DESC LIMIT 50")
tick_df = query_df("SELECT * FROM scalper_tick_log ORDER BY tick_time DESC LIMIT 100")

# Compute stats
total_pnl = closed_df["pnl_usd"].sum() if len(closed_df) > 0 else 0
total_trades = len(closed_df)
win_rate = (closed_df["return_pct"] > 0).mean() * 100 if total_trades > 0 else 0
n_wins = (closed_df["return_pct"] > 0).sum() if total_trades > 0 else 0
capital = config.INITIAL_CAPITAL + total_pnl - open_df["position_size"].sum() if len(open_df) > 0 else config.INITIAL_CAPITAL + total_pnl
total_equity = capital + open_df["position_size"].sum() if len(open_df) > 0 else capital

# Profit factor
if total_trades > 0:
    gross_profit = closed_df[closed_df["return_pct"] > 0]["pnl_usd"].sum()
    gross_loss = abs(closed_df[closed_df["return_pct"] <= 0]["pnl_usd"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
else:
    pf = 0

# Header metrics
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.metric("Total Equity", f"${total_equity:,.2f}", f"${total_pnl:+,.2f}")
with col2:
    st.metric("Available Capital", f"${capital:,.2f}")
with col3:
    st.metric("Open Positions", f"{len(open_df)}/{config.MAX_POSITIONS}")
with col4:
    st.metric("Trades", f"{total_trades}", f"WR: {win_rate:.0f}%")
with col5:
    st.metric("Profit Factor", f"{pf:.2f}" if pf < 100 else "Inf")
with col6:
    st.metric("Watchlist", f"{len(watchlist_df)}")

st.divider()

# ── Row 1: Pipeline Funnel + Discovery Activity ─────────────────────────────

col_funnel, col_discovery = st.columns([1, 2])

with col_funnel:
    st.subheader("Pipeline Funnel")
    if len(discovery_df) > 0:
        latest = discovery_df.iloc[0]
        found = latest["tokens_found"]
        new = latest["tokens_new"]
        passed = latest["tokens_passed_scam_filter"]
        blocked = latest["tokens_blocked"]

        funnel_data = pd.DataFrame({
            "Stage": ["Discovered", "New (unseen)", "Passed Scam Filter", "Blocked"],
            "Count": [found, new, passed, blocked],
        })
        fig = go.Figure(go.Funnel(
            y=funnel_data["Stage"],
            x=funnel_data["Count"],
            textinfo="value+percent previous",
            marker=dict(color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]),
        ))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No discovery cycles yet. Waiting for engine to start...")

with col_discovery:
    st.subheader("Discovery History")
    if len(discovery_df) > 0:
        disc_plot = discovery_df.copy()
        disc_plot["timestamp"] = pd.to_datetime(disc_plot["timestamp"])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=disc_plot["timestamp"], y=disc_plot["tokens_passed_scam_filter"],
                             name="Passed", marker_color="#2ca02c"))
        fig.add_trace(go.Bar(x=disc_plot["timestamp"], y=disc_plot["tokens_blocked"],
                             name="Blocked", marker_color="#d62728"))
        fig.update_layout(barmode="stack", height=300, margin=dict(l=0, r=0, t=10, b=10),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Row 2: Live Positions + Equity Curve ─────────────────────────────────────

col_pos, col_equity = st.columns([1, 1])

with col_pos:
    st.subheader("Live Positions")
    if len(open_df) > 0:
        pos_display = open_df[["symbol", "entry_price", "position_size", "peak_mcap", "bars_held", "entry_time"]].copy()
        pos_display.columns = ["Symbol", "Entry MCap", "Size ($)", "Peak MCap", "Bars", "Entry Time"]
        pos_display["Hold (min)"] = pos_display["Bars"] * 5
        pos_display["Entry Time"] = pd.to_datetime(pos_display["Entry Time"]).dt.strftime("%H:%M:%S")
        st.dataframe(pos_display[["Symbol", "Entry MCap", "Peak MCap", "Bars", "Hold (min)", "Size ($)", "Entry Time"]],
                     use_container_width=True, hide_index=True)
    else:
        st.info("No open positions")

with col_equity:
    st.subheader("Equity Curve")
    if len(equity_df) > 0:
        eq = equity_df.copy()
        eq["timestamp"] = pd.to_datetime(eq["timestamp"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq["timestamp"], y=eq["total_equity"],
                                 mode="lines", name="Total Equity",
                                 line=dict(color="#1f77b4", width=2)))
        fig.add_hline(y=config.INITIAL_CAPITAL, line_dash="dash", line_color="gray",
                      annotation_text=f"Initial: ${config.INITIAL_CAPITAL:,.0f}")
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    elif len(closed_df) > 0:
        # Build equity from trades
        closed_sorted = closed_df.sort_values("exit_time")
        cum_pnl = closed_sorted["pnl_usd"].cumsum() + config.INITIAL_CAPITAL
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pd.to_datetime(closed_sorted["exit_time"]), y=cum_pnl,
                                 mode="lines+markers", name="Equity"))
        fig.add_hline(y=config.INITIAL_CAPITAL, line_dash="dash", line_color="gray")
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No equity data yet")

st.divider()

# ── Row 3: Trade History + Performance Breakdown ─────────────────────────────

col_trades, col_perf = st.columns([2, 1])

with col_trades:
    st.subheader("Trade History")
    if len(closed_df) > 0:
        display = closed_df[["symbol", "entry_time", "exit_time", "exit_reason",
                             "return_pct", "pnl_usd", "bars_held", "jupiter_entry_impact"]].copy()
        display.columns = ["Symbol", "Entry", "Exit", "Reason", "Return %", "PnL $", "Bars", "Jup Impact %"]
        display["Entry"] = pd.to_datetime(display["Entry"]).dt.strftime("%m/%d %H:%M")
        display["Exit"] = pd.to_datetime(display["Exit"]).dt.strftime("%m/%d %H:%M")
        display["Return %"] = display["Return %"].apply(lambda x: f"{x:+.1f}%" if x else "—")
        display["PnL $"] = display["PnL $"].apply(lambda x: f"${x:+.2f}" if x else "—")
        display["Hold"] = display["Bars"].apply(lambda x: f"{(x or 0)*5}m")

        st.dataframe(
            display[["Symbol", "Entry", "Exit", "Reason", "Return %", "PnL $", "Hold"]].sort_index(ascending=False),
            use_container_width=True, hide_index=True, height=400,
        )
    else:
        st.info("No closed trades yet")

with col_perf:
    st.subheader("Performance")
    if total_trades > 0:
        # Exit reason breakdown
        reason_counts = closed_df["exit_reason"].value_counts()
        fig = px.pie(values=reason_counts.values, names=reason_counts.index,
                     title="Exit Reasons", hole=0.4)
        fig.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Win/Loss distribution
        fig2 = go.Figure()
        wins = closed_df[closed_df["return_pct"] > 0]["return_pct"]
        losses = closed_df[closed_df["return_pct"] <= 0]["return_pct"]
        if len(wins) > 0:
            fig2.add_trace(go.Histogram(x=wins, name="Wins", marker_color="#2ca02c", opacity=0.7))
        if len(losses) > 0:
            fig2.add_trace(go.Histogram(x=losses, name="Losses", marker_color="#d62728", opacity=0.7))
        fig2.update_layout(title="Return Distribution", height=200, barmode="overlay",
                           margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig2, use_container_width=True)

        # Summary stats
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Total PnL | **${total_pnl:+,.2f}** |
        | Win Rate | **{win_rate:.1f}%** ({n_wins}/{total_trades}) |
        | Profit Factor | **{pf:.2f}** |
        | Avg Win | **{wins.mean():+.1f}%** |
        | Avg Loss | **{losses.mean():+.1f}%** |
        | Best Trade | **{closed_df['return_pct'].max():+.1f}%** |
        | Worst Trade | **{closed_df['return_pct'].min():+.1f}%** |
        | Avg Hold | **{closed_df['bars_held'].mean():.0f} bars** ({closed_df['bars_held'].mean()*5:.0f}m) |
        """)

st.divider()

# ── Row 4: Engine Activity ───────────────────────────────────────────────────

st.subheader("Engine Activity")

if len(tick_df) > 0:
    tick_plot = tick_df.copy()
    tick_plot["tick_time"] = pd.to_datetime(tick_plot["tick_time"])

    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tick_plot["tick_time"], y=tick_plot["signals_generated"],
                                 mode="lines", name="Signals", line=dict(color="#ff7f0e")))
        fig.add_trace(go.Scatter(x=tick_plot["tick_time"], y=tick_plot["positions_open"],
                                 mode="lines", name="Positions", line=dict(color="#1f77b4")))
        fig.update_layout(title="Signals & Positions Over Time", height=250,
                          margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tick_plot["tick_time"], y=tick_plot["daily_pnl"],
                                 mode="lines", fill="tozeroy", name="Daily PnL",
                                 line=dict(color="#2ca02c")))
        fig.update_layout(title="Daily PnL", height=250, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

# ── Strategy Rules Reference ─────────────────────────────────────────────────

with st.expander("Strategy Rules"):
    st.markdown(f"""
    ### Entry (ALL conditions must be true)
    1. **Token age < 6h** — early pump window only
    2. **Price at running ATH** (±5%) — pump in progress
    3. **Holders growing** — net change > 0 in last 30min
    4. **Organic buyers** — swap_in > 5 in last 30min (Moralis)
    5. **Price momentum** — ROC 30min > 2%
    6. **Liquidity OK** — Jupiter price impact < {config.MAX_PRICE_IMPACT_PCT}%

    ### Exit (ANY condition triggers)
    1. **Trailing stop** — {config.TRAILING_STOP_PCT}% from peak
    2. **Hard stop** — {config.HARD_STOP_PCT}% from entry
    3. **Time stop** — {config.TIME_STOP_BARS} bars ({config.TIME_STOP_BARS * 5}min)
    4. **Holder exit** — holders declining 3 consecutive checks

    ### Risk
    - Position size: ${config.POSITION_SIZE}
    - Max positions: {config.MAX_POSITIONS}
    - Daily loss limit: ${config.DAILY_LOSS_LIMIT}
    - Max trades/day: {config.MAX_TRADES_PER_DAY}
    - Initial capital: ${config.INITIAL_CAPITAL:,.0f}
    """)

# Auto-refresh
st.markdown(
    '<meta http-equiv="refresh" content="30">',
    unsafe_allow_html=True,
)
