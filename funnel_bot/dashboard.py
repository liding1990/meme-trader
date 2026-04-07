"""Funnel Bot Dashboard — Real-time visualization of the 4-layer trading funnel.

Usage:
    streamlit run funnel_bot/dashboard.py --server.port 8520
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from funnel_bot import db

st.set_page_config(page_title="Funnel Bot Dashboard", layout="wide")

# Auto refresh every 30s
st.markdown('<meta http-equiv="refresh" content="30">', unsafe_allow_html=True)

db.init_db()


# ── Header Metrics ──

st.title("Funnel Bot Paper Trading")

positions = db.get_positions()
trades = db.get_trades(limit=1000)
equity_hist = db.get_equity_history(limit=500)
scan_hist = db.get_scan_history(limit=100)

# Calculate metrics
total_trades = len(trades)
closed_trades = [t for t in trades if t["exit_time"]]
wins = [t for t in closed_trades if t["pnl_usd"] and t["pnl_usd"] > 0]
losses = [t for t in closed_trades if t["pnl_usd"] and t["pnl_usd"] <= 0]

total_pnl = sum(t["pnl_usd"] or 0 for t in closed_trades)
win_rate = len(wins) / max(len(closed_trades), 1) * 100
avg_win = np.mean([t["pnl_usd"] for t in wins]) if wins else 0
avg_loss = np.mean([t["pnl_usd"] for t in losses]) if losses else 0
pf = abs(sum(t["pnl_usd"] for t in wins)) / max(abs(sum(t["pnl_usd"] for t in losses)), 0.01) if losses else 0

latest_equity = equity_hist[-1] if equity_hist else None
capital = latest_equity["capital"] if latest_equity else 5000
total_equity = latest_equity["total_equity"] if latest_equity else 5000

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("总权益", f"${total_equity:,.0f}", f"${total_pnl:+,.0f}")
col2.metric("可用资金", f"${capital:,.0f}")
col3.metric("持仓数", f"{len(positions)}")
col4.metric("总交易", f"{len(closed_trades)}", f"胜率 {win_rate:.0f}%")
col5.metric("Profit Factor", f"{pf:.2f}" if pf else "N/A")
col6.metric("观察列表", f"{len(db.get_watchlist())}")

st.divider()

# ── Funnel Visualization ──

st.subheader("交易漏斗")

if scan_hist:
    # Aggregate last N scans
    recent = scan_hist[-min(20, len(scan_hist)):]
    avg_l0 = np.mean([s["l0_candidates"] for s in recent])
    avg_l1_pass = np.mean([s["l1_passed"] for s in recent])
    avg_l1_filter = np.mean([s["l1_filtered"] for s in recent])
    avg_l2_sig = np.mean([s["l2_signals"] for s in recent])
    avg_l2_rej = np.mean([s["l2_rejected"] for s in recent])

    # Funnel chart
    funnel_data = {
        "阶段": ["L0 市场扫描", "L1 Scam过滤", "L2 入场信号", "L3 实际交易"],
        "数量": [avg_l0, avg_l1_pass, avg_l2_sig, len(positions)],
        "颜色": ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444"],
    }

    fig_funnel = go.Figure(go.Funnel(
        y=funnel_data["阶段"],
        x=funnel_data["数量"],
        textinfo="value+percent initial",
        marker=dict(color=funnel_data["颜色"]),
    ))
    fig_funnel.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))

    col_funnel, col_scan = st.columns([1, 1])

    with col_funnel:
        st.plotly_chart(fig_funnel, use_container_width=True)

    with col_scan:
        # Scan history bar chart
        scan_df = pd.DataFrame([dict(s) for s in recent])
        scan_df["time"] = pd.to_datetime(scan_df["timestamp"]).dt.strftime("%H:%M")

        fig_scan = go.Figure()
        fig_scan.add_trace(go.Bar(x=scan_df["time"], y=scan_df["l0_candidates"], name="L0 候选", marker_color="#3b82f6"))
        fig_scan.add_trace(go.Bar(x=scan_df["time"], y=scan_df["l1_passed"], name="L1 通过", marker_color="#22c55e"))
        fig_scan.add_trace(go.Bar(x=scan_df["time"], y=scan_df["l2_signals"], name="L2 信号", marker_color="#f59e0b"))
        fig_scan.update_layout(barmode="group", height=300, title="扫描历史",
                                margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_scan, use_container_width=True)
else:
    st.info("暂无扫描数据。启动 engine 后数据将自动填充。")

st.divider()

# ── Live Positions ──

col_pos, col_equity = st.columns([1, 1])

with col_pos:
    st.subheader("当前持仓")
    if positions:
        pos_data = []
        for p in positions:
            pnl_pct = 0
            if p["entry_price"] and p["entry_price"] > 0:
                # Note: in paper mode we don't have real-time price update here
                pnl_pct = (p["peak_price"] - p["entry_price"]) / p["entry_price"] * 100

            pos_data.append({
                "Token": p["symbol"],
                "入场价": f"${p['entry_price']:,.0f}",
                "仓位": f"${p['position_size']:,.0f}",
                "剩余": f"{p['remaining_pct']*100:.0f}%",
                "峰值PnL": f"{pnl_pct:+.1f}%",
                "持仓": f"{p['bars_held']}t",
                "L2": f"{p['l2_proba']:.2f}",
                "已实现": f"${p['realized_pnl']:+.1f}",
            })
        st.dataframe(pd.DataFrame(pos_data), hide_index=True, use_container_width=True)
    else:
        st.info("无持仓")

with col_equity:
    st.subheader("权益曲线")
    if equity_hist:
        eq_df = pd.DataFrame([dict(e) for e in equity_hist])
        eq_df["time"] = pd.to_datetime(eq_df["timestamp"])

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=eq_df["time"], y=eq_df["total_equity"],
                                      mode="lines", name="总权益", line=dict(color="#3b82f6", width=2)))
        fig_eq.add_trace(go.Scatter(x=eq_df["time"], y=eq_df["capital"],
                                      mode="lines", name="可用资金", line=dict(color="#94a3b8", width=1)))
        fig_eq.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_eq, use_container_width=True)
    else:
        st.info("暂无权益数据")

st.divider()

# ── Trade History ──

st.subheader("交易记录")

if closed_trades:
    trade_data = []
    for t in closed_trades[:50]:
        trade_data.append({
            "Token": t["symbol"],
            "入场时间": t["entry_time"][:16] if t["entry_time"] else "",
            "退出时间": t["exit_time"][:16] if t["exit_time"] else "",
            "收益率": f"{t['return_pct']:+.1f}%" if t["return_pct"] else "",
            "PnL": f"${t['pnl_usd']:+.1f}" if t["pnl_usd"] else "",
            "退出原因": t["exit_reason"] or "",
            "L2分": f"{t['l2_proba']:.2f}",
            "持仓": f"{t['bars_held']}t",
        })
    st.dataframe(pd.DataFrame(trade_data), hide_index=True, use_container_width=True)

    # Performance charts
    col_ret, col_reason = st.columns(2)

    with col_ret:
        returns = [t["return_pct"] for t in closed_trades if t["return_pct"]]
        if returns:
            fig_ret = px.histogram(x=returns, nbins=30, title="收益率分布",
                                    labels={"x": "收益率 (%)", "count": "次数"},
                                    color_discrete_sequence=["#3b82f6"])
            fig_ret.add_vline(x=0, line_dash="dash", line_color="red")
            fig_ret.update_layout(height=300)
            st.plotly_chart(fig_ret, use_container_width=True)

    with col_reason:
        reasons = [t["exit_reason"] for t in closed_trades if t["exit_reason"]]
        if reasons:
            reason_counts = pd.Series(reasons).value_counts()
            fig_reason = px.pie(values=reason_counts.values, names=reason_counts.index,
                                 title="退出原因分布")
            fig_reason.update_layout(height=300)
            st.plotly_chart(fig_reason, use_container_width=True)
else:
    st.info("暂无已平仓交易")

st.divider()

# ── L3 Action Log ──

st.subheader("L3 操作日志")

l3_log = db.get_l3_log(limit=30)
if l3_log:
    log_data = []
    for l in l3_log:
        action_color = {"TP_25": "green", "TP_50": "green", "TP_100": "green",
                         "SL_25": "red", "SL_50": "red", "EXIT": "red",
                         "TIME_EXIT": "orange"}.get(l["action"], "gray")
        log_data.append({
            "时间": l["timestamp"][:19],
            "Token": l["symbol"],
            "动作": l["action"],
            "价格": f"${l['price']:,.0f}" if l["price"] else "",
            "PnL": f"{l['pnl_pct']:+.1f}%" if l["pnl_pct"] else "",
            "剩余": f"{l['remaining_before']*100:.0f}%→{l['remaining_after']*100:.0f}%",
        })
    st.dataframe(pd.DataFrame(log_data), hide_index=True, use_container_width=True)
else:
    st.info("暂无 L3 操作记录")

# ── Watchlist ──

with st.expander("观察列表"):
    watchlist = db.get_watchlist()
    if watchlist:
        wl_data = []
        for w in watchlist[:30]:
            wl_data.append({
                "Token": w["symbol"],
                "MCap": f"${w['last_mcap']:,.0f}" if w["last_mcap"] else "",
                "Holders": f"{w['last_holders']:,}" if w["last_holders"] else "",
                "L2分数": f"{w['l2_proba']:.3f}",
                "状态": w["status"],
            })
        st.dataframe(pd.DataFrame(wl_data), hide_index=True, use_container_width=True)

# Footer
st.caption(f"上次刷新: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | 自动刷新: 30秒")
