"""Baseline Clustering v2 — 交互式展示页面

展示 6 个 outcome cluster 的完整分析：数据选取方法、聚类结果、3D 可视化、Token 查询。

Usage:
    streamlit run app_baseline_cluster.py --server.port 8530
"""

import csv
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Baseline Clustering v2", layout="wide")

DATA_DIR = "data"
CLUSTER_DIR = "baseline_cluster_v2_data"
CLUSTER_CSV = os.path.join(CLUSTER_DIR, "clusters.csv")
MODEL_PKL = os.path.join(CLUSTER_DIR, "model.pkl")

CLUSTER_META = {
    1: {
        "name": "Fake / Scam Token",
        "name_cn": "假币 / 仿冒币",
        "color": "#6b7280",
        "desc": "ATH 数十亿但 holder 仅 0-1。全是 unicode 仿冒币（伪造 SOL、USDT 等），假 mcap，"
                "无真实交易量，无真实社区。应直接过滤。",
        "traits": ["ATH 虚高（中位 $2.6B）", "Holder 为 0-1", "Unicode 仿冒知名 token", "无真实交易"],
    },
    2: {
        "name": "Pump & Dump",
        "name_cn": "拉盘砸盘",
        "color": "#ef4444",
        "desc": "1 小时拉盘 + 1 小时崩盘的极端快进快出模式。有真人参与（~1.7K holder），"
                "但多数是追高被套。典型的事件驱动型，蹭热点后瞬间归零。",
        "traits": ["上升+衰减共 ~2h", "价格增速极高 ($1.8M/h)", "有真人参与 (~1.7K holders)", "事件驱动型"],
    },
    3: {
        "name": "Ghost Pump",
        "name_cn": "幽灵拉盘（无社区）",
        "color": "#a855f7",
        "desc": "有 mcap 增长但 holder = 0。纯合约操纵、LP 刷量或 holder 数据缺失。"
                "上升 14h 但衰减仅 3h，拉完就跑，没有任何社区基础。",
        "traits": ["Holder@ATH = 0", "纯合约 / LP 操纵", "上升 14h / 衰减 3h", "无社区基础"],
    },
    4: {
        "name": "Slow Bleed",
        "name_cn": "慢性归零",
        "color": "#f97316",
        "desc": "涨上去后花 20+ 天缓慢归零。上升仅占 4.6% 的生命周期，绝大部分时间在阴跌。"
                "有 holder 但社区停滞，增长速度为 0。最痛苦的一类——给人希望又慢慢磨灭。",
        "traits": ["衰减时长中位 479h (~20天)", "上升占比仅 4.6%", "Holder 增长停滞", "漫长阴跌"],
    },
    5: {
        "name": "Organic Runner",
        "name_cn": "社区驱动长线",
        "color": "#22c55e",
        "desc": "上升时长中位 372h（~15 天），慢慢积累型。PUNCH、WAR、WOJAK 都在这里。"
                "有真实持币社区，增长缓慢但持续。是真正由社区共识驱动的 token。",
        "traits": ["上升 ~15 天", "有真实社区 (1.7K+ holders)", "缓慢但持续的增长", "PUNCH/WAR/WOJAK"],
    },
    6: {
        "name": "Fast Organic",
        "name_cn": "快速爆发型",
        "color": "#3b82f6",
        "desc": "生命周期 ~1.5 天。Holder 增速 116/h 是所有 cluster 中最高——真实社区快速涌入但也快速散场。"
                "BFS、PENGUIN、CityBoy 在这里。爆发力强但持续性不足。",
        "traits": ["上升 16h / 衰减 18h", "Holder 增速最高 (116/h)", "真实社区 + 快速爆发", "BFS/PENGUIN/CityBoy"],
    },
}


# ── Data Loading ─────────────────────────────────────────────────────────────


@st.cache_data
def load_cluster_data():
    if not os.path.isfile(CLUSTER_CSV):
        return None, None
    df = pd.read_csv(CLUSTER_CSV)
    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)
    return df, model


@st.cache_data
def load_trajectories(addresses):
    """Load hourly mcap + holder trajectories for 3D visualization."""
    trajectories = {}
    for addr in addresses:
        data_dir = os.path.join(DATA_DIR, addr)
        h_files = sorted([f for f in glob.glob(os.path.join(data_dir, "token_mcap_candles_[0-9]*.json"))
                           if "5m" not in os.path.basename(f)])
        if not h_files:
            continue
        try:
            with open(h_files[-1]) as f:
                data = json.load(f)
            candles = (data or {}).get("data", {}).get("list", [])
            if not candles or len(candles) < 2:
                continue
            cdf = pd.DataFrame(candles)
            cdf["mcap"] = cdf["close"].astype(float)
            cdf["volume"] = cdf["volume"].astype(float)
            cdf["datetime"] = pd.to_datetime(cdf["time"].astype(int), unit="ms")
            cdf = cdf.sort_values("datetime").reset_index(drop=True)

            # Window: first $100K → end
            above = cdf[cdf["mcap"] >= 100000]
            if above.empty:
                continue
            start = above.index[0]
            cdf = cdf.loc[start:].reset_index(drop=True)
            t0 = cdf["datetime"].iloc[0]
            cdf["hours"] = (cdf["datetime"] - t0).dt.total_seconds() / 3600

            # Load holders
            moralis = os.path.join(data_dir, "moralis_holders_1h.json")
            cdf["holders"] = 0
            if os.path.isfile(moralis):
                try:
                    with open(moralis) as f:
                        hdata = json.load(f)
                    if hdata:
                        hdf = pd.DataFrame(hdata)
                        hdf["datetime"] = pd.to_datetime(hdf["timestamp"], utc=True).dt.tz_localize(None)
                        hdf["holders"] = hdf["totalHolders"].astype(float)
                        hdf = hdf.set_index("datetime").resample("1h").last().ffill().reset_index()
                        merged = pd.merge_asof(cdf[["datetime", "mcap", "volume", "hours"]],
                                                hdf[["datetime", "holders"]], on="datetime", direction="backward")
                        if "holders" in merged.columns:
                            cdf = merged
                except Exception:
                    pass

            trajectories[addr] = cdf
        except Exception:
            continue
    return trajectories


# ── Page Sections ────────────────────────────────────────────────────────────


def section_methodology():
    """数据选取方法论。"""
    st.markdown("## 数据选取方法")

    st.markdown("""
    ### 数据窗口定义

    对每个 token，我们定义了一个标准化的生命周期窗口：
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **起点**
        - 市值首次达到 **$100K** 的时刻
        - $100K 之前的数据丢弃（噪音阶段）
        """)
    with col2:
        st.markdown("""
        **峰值 (ATH)**
        - 窗口内市值最高点
        - 起点 → ATH = **上升阶段**
        """)
    with col3:
        st.markdown("""
        **截止点**
        - 从 ATH 下跌 **70%** 的第一个时刻
        - ATH → 截止 = **衰减阶段**
        - 若未跌 70% 则取数据末尾
        """)

    st.markdown("""
    ```
    $100K ════ 上升阶段 ════ ATH ════ 衰减阶段 ════ ATH × 30%
     起点                    峰值                   截止（跌70%）
    ```
    """)

    st.markdown("""
    ### 过滤条件
    - **Token 创建时间 > 2026 年 1 月 1 日**（排除旧周期 token）
    - **ATH >= $100K**（排除从未有过实质市值的 token）
    - **数据来源：GMGN 小时级蜡烛图**（close = 市值）
    """)

    st.markdown("""
    ### 11 个聚类特征

    | 阶段 | 特征 | 含义 |
    |---|---|---|
    | 上升 | `rise_hours` | 从 $100K 到 ATH 的时长 |
    | 上升 | `price_roc` | 价格增速 ($/h) |
    | 上升 | `volume_roc` | 成交量增速 ($/h) |
    | 上升 | `holder_roc` | 持币人增速 (/h) |
    | 峰值 | `ath` | 历史最高市值 |
    | 峰值 | `holders_at_ath` | ATH 时持币人数 |
    | 衰减 | `decay_hours` | 从 ATH 到跌 70% 的时长 |
    | 衰减 | `holder_decay_roc` | 持币人流失速度 (/h) |
    | 衰减 | `price_decay_roc` | 价格衰减速度 ($/h) |
    | 时间 | `total_hours` | 完整周期时长 |
    | 时间 | `rise_pct` | 上升阶段占比 |
    """)


def section_clusters(df):
    """6 个 Cluster 详细展示。"""
    st.markdown("## 聚类结果")

    cluster_order = sorted(df["rank"].unique())
    cid_by_rank = {int(df[df["rank"] == r]["cluster_id"].iloc[0]): r for r in cluster_order}
    rank_by_cid = {v: k for k, v in cid_by_rank.items()}

    st.markdown(f"**{len(df)} 个 token，6 个聚类**（KMeans, log-scaled, StandardScaler）")

    # Overview table
    overview_rows = []
    for rank in cluster_order:
        meta = CLUSTER_META.get(rank, {})
        sub = df[df["rank"] == rank]
        overview_rows.append({
            "#": rank,
            "名称": meta.get("name", "?"),
            "中文": meta.get("name_cn", "?"),
            "数量": len(sub),
            "ATH 中位": f"${sub['ath'].median():,.0f}",
            "上升时长": f"{sub['rise_hours'].median():.0f}h",
            "衰减时长": f"{sub['decay_hours'].median():.0f}h",
            "Holder@ATH": f"{sub['holders_at_ath'].median():,.0f}",
        })
    st.dataframe(pd.DataFrame(overview_rows), hide_index=True, use_container_width=True)

    # Detailed cards
    for rank in cluster_order:
        meta = CLUSTER_META.get(rank, {})
        sub = df[df["rank"] == rank]
        color = meta.get("color", "#888")

        st.markdown(f"""
        <div style="border-left: 5px solid {color}; padding: 12px 16px; margin: 16px 0; background: {color}10;">
            <h3 style="color: {color}; margin: 0;">#{rank} {meta.get('name', '?')} — {meta.get('name_cn', '?')} ({len(sub)} tokens)</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(meta.get("desc", ""))

        # Traits as tags
        traits = meta.get("traits", [])
        if traits:
            tags_html = " ".join(
                f'<span style="background:{color}20; color:{color}; padding:2px 8px; '
                f'border-radius:12px; font-size:0.85em; margin-right:4px;">{t}</span>'
                for t in traits
            )
            st.markdown(tags_html, unsafe_allow_html=True)

        # Stats columns
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ATH 中位", f"${sub['ath'].median():,.0f}")
        col2.metric("上升时长", f"{sub['rise_hours'].median():.0f}h")
        col3.metric("衰减时长", f"{sub['decay_hours'].median():.0f}h")
        col4.metric("Holder@ATH", f"{sub['holders_at_ath'].median():,.0f}")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("价格增速", f"${sub['price_roc'].median():,.0f}/h")
        col6.metric("Volume增速", f"${sub['volume_roc'].median():,.0f}/h")
        col7.metric("Holder增速", f"{sub['holder_roc'].median():.1f}/h")
        col8.metric("上升占比", f"{sub['rise_pct'].median():.0%}")

        # Sample tokens table
        with st.expander(f"查看 #{rank} 全部 {len(sub)} 个 token"):
            display = sub.sort_values("ath", ascending=False)[
                ["symbol", "ath", "rise_hours", "decay_hours", "holders_at_ath",
                 "price_roc", "holder_roc", "total_hours", "rise_pct"]
            ].copy()
            display.columns = ["Token", "ATH", "上升(h)", "衰减(h)", "Holder@ATH",
                               "价格增速($/h)", "Holder增速(/h)", "总时长(h)", "上升占比"]
            display["ATH"] = display["ATH"].apply(lambda x: f"${x:,.0f}")
            display["价格增速($/h)"] = display["价格增速($/h)"].apply(lambda x: f"${x:,.0f}")
            display["上升占比"] = display["上升占比"].apply(lambda x: f"{x:.0%}")
            st.dataframe(display, hide_index=True, use_container_width=True)


def section_3d_chart(df):
    """3D 轨迹可视化。"""
    st.markdown("## 3D 轨迹可视化")
    st.markdown("*每条线是一个 token 从 $100K 开始的生命轨迹，按 cluster 着色。*")

    # Cluster filter
    cluster_order = sorted(df["rank"].unique())
    options = {f"#{r} {CLUSTER_META.get(r, {}).get('name', '?')}": r for r in cluster_order}
    selected = st.multiselect("选择显示的 Cluster", list(options.keys()),
                               default=[k for k, v in options.items() if v not in [1]],  # hide fake by default
                               key="cluster_3d_filter")
    selected_ranks = [options[s] for s in selected]

    # Load trajectories
    addrs = df[df["rank"].isin(selected_ranks)]["address"].tolist()
    trajectories = load_trajectories(addrs[:200])  # cap at 200 for performance

    if not trajectories:
        st.warning("无轨迹数据")
        return

    # Build rank lookup
    addr_rank = dict(zip(df["address"], df["rank"]))

    fig = go.Figure()
    for addr, traj in trajectories.items():
        rank = addr_rank.get(addr, 0)
        if rank not in selected_ranks:
            continue

        meta = CLUSTER_META.get(rank, {})
        color = meta.get("color", "#999")
        name_label = meta.get("name", "?")
        symbol = df[df["address"] == addr]["symbol"].iloc[0] if addr in df["address"].values else "?"
        width = 3 if rank in [5, 6] else 2 if rank in [2, 4] else 1

        fig.add_trace(go.Scatter3d(
            x=traj["hours"], y=traj["mcap"],
            z=traj["holders"] if "holders" in traj.columns else [0] * len(traj),
            mode="lines",
            line=dict(color=color, width=width),
            text=[
                f"<b>{symbol}</b><br>#{rank} {name_label}<br>"
                f"T+{row['hours']:.0f}h<br>MCap: ${row['mcap']:,.0f}<br>"
                f"Holders: {row.get('holders', 0):,.0f}"
                for _, row in traj.iterrows()
            ],
            hoverinfo="text",
            name=f"{symbol}",
            showlegend=False,
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="时间（小时）",
            yaxis_title="市值 ($)",
            zaxis_title="持币人数",
            xaxis=dict(backgroundcolor="white", gridcolor="rgb(200,200,200)"),
            yaxis=dict(backgroundcolor="white", gridcolor="rgb(200,200,200)", type="log"),
            zaxis=dict(backgroundcolor="white", gridcolor="rgb(200,200,200)"),
            bgcolor="white",
        ),
        paper_bgcolor="white",
        font=dict(color="black"),
        height=700,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Legend
    legend_cols = st.columns(len(selected_ranks))
    for i, rank in enumerate(selected_ranks):
        meta = CLUSTER_META.get(rank, {})
        color = meta.get("color", "#999")
        n = len(df[df["rank"] == rank])
        legend_cols[i].markdown(
            f'<span style="color:{color}; font-weight:bold;">&#9632;</span> '
            f'#{rank} {meta.get("name", "?")} ({n})',
            unsafe_allow_html=True
        )


def section_scatter(df):
    """2D 散点图。"""
    st.markdown("## 特征散点分析")

    # Cluster filter — same level as X/Y
    cluster_order = sorted(df["rank"].unique())
    cluster_options = {f"#{r} {CLUSTER_META.get(r, {}).get('name', '?')}": r for r in cluster_order}

    selected_clusters = st.multiselect(
        "选择 Cluster",
        list(cluster_options.keys()),
        default=[k for k, v in cluster_options.items() if v in [5, 6]],
        key="scatter_clusters",
    )
    selected_ranks = [cluster_options[s] for s in selected_clusters]

    col_x, col_y = st.columns(2)
    feature_options = {
        "ATH ($)": "ath",
        "上升时长 (h)": "rise_hours",
        "衰减时长 (h)": "decay_hours",
        "价格增速 ($/h)": "price_roc",
        "Holder@ATH": "holders_at_ath",
        "Holder增速 (/h)": "holder_roc",
        "总时长 (h)": "total_hours",
        "上升占比": "rise_pct",
        "Volume增速 ($/h)": "volume_roc",
        "Holder衰减 (/h)": "holder_decay_roc",
        "价格衰减 ($/h)": "price_decay_roc",
    }
    x_label = col_x.selectbox("X 轴", list(feature_options.keys()), index=3, key="scatter_x")
    y_label = col_y.selectbox("Y 轴", list(feature_options.keys()), index=0, key="scatter_y")
    x_col = feature_options[x_label]
    y_col = feature_options[y_label]

    # Filter data
    plot_df = df[df["rank"].isin(selected_ranks)].copy()
    if plot_df.empty:
        st.info("请选择至少一个 Cluster")
        return

    plot_df["cluster_name"] = plot_df["rank"].map(
        lambda r: f"#{r} {CLUSTER_META.get(r, {}).get('name', '?')}")
    color_map = {f"#{r} {CLUSTER_META.get(r, {}).get('name', '?')}": CLUSTER_META.get(r, {}).get("color", "#999")
                 for r in selected_ranks}

    log_cols = {"ath", "price_roc", "volume_roc", "rise_hours", "decay_hours",
                "total_hours", "holders_at_ath"}
    log_x = x_col in log_cols
    log_y = y_col in log_cols

    # Handle negative values for log scale
    if log_x:
        plot_df = plot_df[plot_df[x_col] > 0]
    if log_y:
        plot_df = plot_df[plot_df[y_col] > 0]

    fig = px.scatter(plot_df, x=x_col, y=y_col, color="cluster_name",
                      hover_name="symbol", log_x=log_x, log_y=log_y,
                      color_discrete_map=color_map,
                      labels={x_col: x_label, y_col: y_label, "cluster_name": "Cluster"},
                      title=f"{x_label} vs {y_label}（{len(plot_df)} tokens）")
    fig.update_layout(height=550)
    fig.update_traces(marker=dict(size=7, opacity=0.8))
    st.plotly_chart(fig, use_container_width=True)

    # Per-cluster stats for selected
    if len(selected_ranks) >= 2:
        st.markdown("### 选中 Cluster 对比")
        compare_rows = []
        for r in selected_ranks:
            sub = df[df["rank"] == r]
            meta = CLUSTER_META.get(r, {})
            compare_rows.append({
                "Cluster": f"#{r} {meta.get('name', '?')}",
                "数量": len(sub),
                f"{x_label} 中位": f"{sub[x_col].median():,.1f}" if sub[x_col].median() < 1000
                                  else f"${sub[x_col].median():,.0f}" if "($" in x_label
                                  else f"{sub[x_col].median():,.0f}",
                f"{y_label} 中位": f"{sub[y_col].median():,.1f}" if sub[y_col].median() < 1000
                                  else f"${sub[y_col].median():,.0f}" if "($" in y_label
                                  else f"{sub[y_col].median():,.0f}",
            })
        st.dataframe(pd.DataFrame(compare_rows), hide_index=True)


def section_regression(df):
    """Regression 分析 — 针对 Organic cluster 的关键特征回归。"""
    st.markdown("## Regression 分析")
    st.markdown("针对 **#5 Organic Runner** 和 **#6 Fast Organic** 的关键特征对进行二次多项式回归，发现增长规律。")

    from scipy import stats as sp_stats

    # Only organic clusters
    organic = df[df["rank"].isin([5, 6])].copy()
    organic["cluster_name"] = organic["rank"].map(
        lambda r: f"#{r} {CLUSTER_META.get(r, {}).get('name', '?')}")

    # Predefined pairs with descriptions, ordered by R²
    REGRESSION_PAIRS = [
        ("volume_roc", "holder_roc", "Volume增速 ($/h)", "Holder增速 (/h)",
         "成交量越大的 token，holder 涌入速度越快。Volume 是吸引新用户的核心引擎。"),
        ("price_roc", "holder_roc", "价格增速 ($/h)", "Holder增速 (/h)",
         "价格涨得越快，holder 增长越快。价格上涨本身就是最好的营销。"),
        ("price_roc", "volume_roc", "价格增速 ($/h)", "Volume增速 ($/h)",
         "价格增速和成交量增速高度正相关。真正的上涨伴随着真实的成交量放大。"),
        ("rise_hours", "price_roc", "上升时长 (h)", "价格增速 ($/h)",
         "上升时间越长的 token，每小时价格增速越低。慢牛和快牛是两种完全不同的模式。"),
        ("ath", "holders_at_ath", "ATH ($)", "Holder@ATH",
         "ATH 越高的 token，峰值时持币人越多。大市值需要大社区支撑。"),
        ("price_roc", "holder_decay_roc", "价格增速 ($/h)", "Holder衰减 (/h)",
         "涨得越快的 token，衰减时 holder 流失也越快。来得快去得快。"),
        ("decay_hours", "holder_decay_roc", "衰减时长 (h)", "Holder衰减 (/h)",
         "衰减时间越长，每小时 holder 流失越慢。慢慢跌的 token 社区粘性更强。"),
        ("holder_roc", "holders_at_ath", "Holder增速 (/h)", "Holder@ATH",
         "每小时 holder 增速越快，最终 holder 峰值越高。增长动量决定了天花板。"),
        ("ath", "price_roc", "ATH ($)", "价格增速 ($/h)",
         "ATH 越高的 token 价格增速越快。大 runner 的特征是增速本身就快。"),
        ("price_roc", "holders_at_ath", "价格增速 ($/h)", "Holder@ATH",
         "价格增速越高，ATH 时 holder 越多。快速上涨吸引的不是投机客就是真信徒。"),
    ]

    # Precompute R² for all pairs
    pair_r2 = []
    for x_col, y_col, x_label, y_label, desc in REGRESSION_PAIRS:
        sub = organic[(organic[x_col] > 0) & (organic[y_col] > 0)]
        if len(sub) < 10:
            pair_r2.append(0)
            continue
        lx = np.log10(sub[x_col].values)
        ly = np.log10(sub[y_col].values)
        coeffs = np.polyfit(lx, ly, 2)
        poly = np.poly1d(coeffs)
        y_pred = poly(lx)
        ss_res = np.sum((ly - y_pred) ** 2)
        ss_tot = np.sum((ly - ly.mean()) ** 2)
        pair_r2.append(1 - ss_res / max(ss_tot, 1e-9))

    # Summary table (sorted by R²)
    st.markdown("### 分析维度总览（按 R² 排序）")
    summary_rows = []
    for i, (x_col, y_col, x_label, y_label, desc) in enumerate(REGRESSION_PAIRS):
        r2 = pair_r2[i]
        if r2 > 0.5:
            strength = "强"
            icon = "🟢"
        elif r2 > 0.3:
            strength = "中等"
            icon = "🟡"
        else:
            strength = "弱"
            icon = "🔴"
        summary_rows.append({
            "关联强度": f"{icon} {strength}",
            "R²": f"{r2:.3f}",
            "X 轴": x_label,
            "Y 轴": y_label,
            "解读": desc,
        })
    summary_rows.sort(key=lambda x: float(x["R²"]), reverse=True)
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    st.divider()

    # Pair selector
    pair_labels = [f"{x_label} vs {y_label}" for _, _, x_label, y_label, _ in REGRESSION_PAIRS]
    selected_pair = st.selectbox("选择分析维度", pair_labels, index=0, key="reg_pair")
    pair_idx = pair_labels.index(selected_pair)
    x_col, y_col, x_label, y_label, pair_desc = REGRESSION_PAIRS[pair_idx]

    degree = 2  # fixed polynomial degree

    # Filter valid data (positive values for log)
    plot_df = organic[(organic[x_col] > 0) & (organic[y_col] > 0)].copy()
    if len(plot_df) < 5:
        st.warning("有效数据点不足")
        return

    # Log-space polynomial regression
    log_x = np.log10(plot_df[x_col].values)
    log_y = np.log10(plot_df[y_col].values)

    # Fit polynomial in log space
    coeffs = np.polyfit(log_x, log_y, degree)
    poly = np.poly1d(coeffs)

    # R² calculation
    y_pred = poly(log_x)
    ss_res = np.sum((log_y - y_pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r_squared = 1 - ss_res / max(ss_tot, 1e-9)

    # Also compute linear R² for comparison
    from scipy import stats as sp_stats
    slope_lin, intercept_lin, r_lin, p_value, _ = sp_stats.linregress(log_x, log_y)
    r_squared_lin = r_lin ** 2

    # Regression curve points (smooth)
    x_range = np.linspace(log_x.min(), log_x.max(), 200)
    y_fit = poly(x_range)

    # Build chart
    color_map = {
        f"#{r} {CLUSTER_META.get(r, {}).get('name', '?')}": CLUSTER_META.get(r, {}).get("color", "#999")
        for r in [5, 6]
    }

    fig = px.scatter(plot_df, x=x_col, y=y_col, color="cluster_name",
                      hover_name="symbol", log_x=True, log_y=True,
                      color_discrete_map=color_map,
                      labels={x_col: x_label, y_col: y_label, "cluster_name": "Cluster"})

    # Add polynomial regression curve
    fig.add_trace(go.Scatter(
        x=10 ** x_range, y=10 ** y_fit,
        mode="lines",
        line=dict(color="#ef4444", width=3),
        name=f"Poly-{degree} (R²={r_squared:.3f})",
        showlegend=True,
    ))

    # Add confidence band (±1 std of residuals)
    residual_std = np.std(log_y - y_pred)
    fig.add_trace(go.Scatter(
        x=np.concatenate([10 ** x_range, 10 ** x_range[::-1]]),
        y=np.concatenate([10 ** (y_fit + residual_std), 10 ** (y_fit - residual_std)[::-1]]),
        fill="toself",
        fillcolor="rgba(239, 68, 68, 0.1)",
        line=dict(color="rgba(239, 68, 68, 0)"),
        name="±1σ 置信带",
        showlegend=True,
    ))

    degree_name = {1: "线性", 2: "二次", 3: "三次", 4: "四次"}[degree]
    fig.update_layout(
        title=f"{x_label} vs {y_label}（{len(plot_df)} tokens, {degree_name}回归 R²={r_squared:.3f}）",
        height=550,
    )
    fig.update_traces(marker=dict(size=7, opacity=0.8), selector=dict(mode="markers"))
    st.plotly_chart(fig, use_container_width=True)

    # Regression stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"R²（{degree_name}）", f"{r_squared:.3f}")
    col2.metric("R²（线性对比）", f"{r_squared_lin:.3f}",
                f"{r_squared - r_squared_lin:+.3f}" if degree > 1 else None)
    col3.metric("p-value（线性）", f"{p_value:.2e}")
    col4.metric("样本数", f"{len(plot_df)}")

    # Polynomial equation
    terms = []
    for i, c in enumerate(coeffs):
        power = degree - i
        if power == 0:
            terms.append(f"{c:.3f}")
        elif power == 1:
            terms.append(f"{c:.3f}·x")
        else:
            terms.append(f"{c:.3f}·x^{power}")
    equation = " + ".join(terms)

    # Interpretation
    if r_squared > 0.5:
        strength = "强"
    elif r_squared > 0.3:
        strength = "中等"
    else:
        strength = "弱"

    st.markdown(f"""
    ### 解读

    > {pair_desc}

    **R² = {r_squared:.3f}** — {x_label} 和 {y_label} 之间存在 **{strength}** 的非线性关系。
    {"（比线性回归提升 " + f"{r_squared - r_squared_lin:+.3f}" + "）" if degree > 1 and r_squared > r_squared_lin else ""}

    **回归方程（对数空间）：** `log(Y) = {equation}`

    **置信带** — 红色阴影区域表示 ±1 个标准差范围，约 68% 的 token 落在此区域内。
    落在置信带上方的 token 表现超预期，下方的则不及预期。
    """)

    # Per-cluster regression
    st.markdown("### 分 Cluster 回归")
    for rank in [5, 6]:
        sub = plot_df[plot_df["rank"] == rank]
        if len(sub) < 3:
            continue
        lx = np.log10(sub[x_col].values)
        ly = np.log10(sub[y_col].values)
        sub_coeffs = np.polyfit(lx, ly, degree)
        sub_poly = np.poly1d(sub_coeffs)
        sub_pred = sub_poly(lx)
        sub_ss_res = np.sum((ly - sub_pred) ** 2)
        sub_ss_tot = np.sum((ly - ly.mean()) ** 2)
        sub_r2 = 1 - sub_ss_res / max(sub_ss_tot, 1e-9)
        meta = CLUSTER_META.get(rank, {})
        st.markdown(f"- **#{rank} {meta.get('name', '?')}** ({len(sub)} tokens): "
                    f"R²={sub_r2:.3f}")


REGRESSION_PAIRS_FOR_QUERY = [
    ("volume_roc", "holder_roc", "Volume增速 ($/h)", "Holder增速 (/h)",
     "成交量越大的 token，holder 涌入速度越快。Volume 是吸引新用户的核心引擎。",
     "holder 吸引效率", "volume_roc"),
    ("price_roc", "volume_roc", "价格增速 ($/h)", "Volume增速 ($/h)",
     "价格增速和成交量增速高度正相关。真正的上涨伴随着真实的成交量放大。",
     "成交量真实性", "price_roc"),
    ("ath", "holders_at_ath", "ATH ($)", "Holder@ATH",
     "ATH 越高的 token，峰值时持币人越多。大市值需要大社区支撑。",
     "市值可持续性", "ath"),
    ("price_roc", "holder_roc", "价格增速 ($/h)", "Holder增速 (/h)",
     "价格涨得越快，holder 增长越快。价格上涨本身就是最好的营销。",
     "价格-社区联动", "price_roc"),
    ("holder_roc", "holders_at_ath", "Holder增速 (/h)", "Holder@ATH",
     "每小时 holder 增速越快，最终 holder 峰值越高。增长动量决定了天花板。",
     "增长天花板", "holder_roc"),
]


def _render_regression_position(df, feat, symbol, rank):
    """Render regression charts with the queried token's position highlighted."""
    from scipy import stats as sp_stats

    st.divider()
    st.markdown("### Regression 位置分析")
    st.markdown(f"以下展示 **{symbol}** 在各个关键维度上相对于 Organic 回归线的位置。"
                f"**回归线上方 = 超预期，下方 = 不及预期。**")

    organic = df[df["rank"].isin([5, 6])].copy()

    z_scores = []

    for x_col, y_col, x_label, y_label, desc, signal_name, _ in REGRESSION_PAIRS_FOR_QUERY:
        sub = organic[(organic[x_col] > 0) & (organic[y_col] > 0)].copy()
        if len(sub) < 10:
            continue

        token_x = feat.get(x_col, 0)
        token_y = feat.get(y_col, 0)
        if token_x <= 0 or token_y <= 0:
            continue

        log_x = np.log10(sub[x_col].values)
        log_y = np.log10(sub[y_col].values)

        # Poly-2 fit
        coeffs = np.polyfit(log_x, log_y, 2)
        poly = np.poly1d(coeffs)

        y_pred_all = poly(log_x)
        residual_std = np.std(log_y - y_pred_all)

        # Token's position
        token_log_x = np.log10(token_x)
        token_log_y = np.log10(token_y)
        expected_log_y = poly(token_log_x)
        z = (token_log_y - expected_log_y) / max(residual_std, 1e-9)
        z_scores.append((signal_name, z))

        # Chart
        x_range = np.linspace(log_x.min() - 0.2, log_x.max() + 0.2, 200)
        y_fit = poly(x_range)

        sub["cluster_name"] = sub["rank"].map(
            lambda r: f"#{r} {CLUSTER_META.get(r, {}).get('name', '?')}")
        color_map = {f"#{r} {CLUSTER_META.get(r, {}).get('name', '?')}": CLUSTER_META.get(r, {}).get("color", "#999")
                     for r in [5, 6]}

        fig = px.scatter(sub, x=x_col, y=y_col, color="cluster_name",
                          hover_name="symbol", log_x=True, log_y=True,
                          color_discrete_map=color_map,
                          labels={x_col: x_label, y_col: y_label, "cluster_name": "Cluster"})

        # Regression line
        fig.add_trace(go.Scatter(
            x=10 ** x_range, y=10 ** y_fit,
            mode="lines", line=dict(color="#ef4444", width=2.5),
            name="回归线", showlegend=True,
        ))

        # Confidence band
        fig.add_trace(go.Scatter(
            x=np.concatenate([10 ** x_range, 10 ** x_range[::-1]]),
            y=np.concatenate([10 ** (y_fit + residual_std), 10 ** (y_fit - residual_std)[::-1]]),
            fill="toself", fillcolor="rgba(239,68,68,0.08)",
            line=dict(color="rgba(239,68,68,0)"),
            name="±1σ", showlegend=True,
        ))

        # Token marker (large, prominent)
        marker_color = "#22c55e" if z > 0 else "#ef4444"
        fig.add_trace(go.Scatter(
            x=[token_x], y=[token_y],
            mode="markers+text",
            marker=dict(size=16, color=marker_color, symbol="star",
                         line=dict(width=2, color="white")),
            text=[symbol], textposition="top center",
            textfont=dict(size=13, color=marker_color, family="Arial Black"),
            name=symbol, showlegend=True,
        ))

        fig.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0))
        fig.update_traces(marker=dict(size=6, opacity=0.6), selector=dict(mode="markers"))
        st.plotly_chart(fig, use_container_width=True)

        # Description
        if z > 1:
            position_desc = f"**显著超预期**（z={z:+.2f}，top {(1-sp_stats.norm.cdf(z))*100:.0f}%）"
            interpretation = f"在同等 {x_label} 水平下，{symbol} 的 {y_label} 远高于大多数 organic token。"
        elif z > 0.3:
            position_desc = f"**略超预期**（z={z:+.2f}）"
            interpretation = f"{symbol} 的 {y_label} 高于回归预期值，表现不错。"
        elif z > -0.3:
            position_desc = f"**符合预期**（z={z:+.2f}）"
            interpretation = f"{symbol} 在这个维度上表现正常，符合 organic token 的典型增长关系。"
        elif z > -1:
            position_desc = f"**略低于预期**（z={z:+.2f}）"
            interpretation = f"{symbol} 的 {y_label} 低于回归预期，需要关注。"
        else:
            position_desc = f"**显著低于预期**（z={z:+.2f}，bottom {sp_stats.norm.cdf(z)*100:.0f}%）"
            interpretation = f"在同等 {x_label} 水平下，{symbol} 的 {y_label} 远低于多数 organic token，可能存在风险。"

        st.markdown(f"**{signal_name}：** {position_desc}")
        st.markdown(f"> {desc}")
        st.markdown(f"{interpretation}")
        st.divider()



def section_query(df, model):
    """Token 查询 — 支持数据集内 token 和任意外部 token 地址。"""
    st.markdown("## Token 查询")
    st.markdown("输入 token symbol（数据集内）或**任意合约地址**（自动从 GMGN 获取数据）。")

    query = st.text_input("Token Symbol 或合约地址",
                           placeholder="例如: PUNCH, BFS, 或 FaBXnb7UFBY81hi1Xg...",
                           key="cluster_query")
    if not query:
        return

    from baseline_cluster_v2 import build_feature_vector, compute_features

    query_lower = query.strip().lower()
    row = None
    feat = None
    symbol = "?"
    is_oos = False

    # Try in-sample first
    match = df[df["symbol"].str.lower() == query_lower]
    if match.empty:
        match = df[df["address"].str.lower() == query_lower]
    if match.empty:
        match = df[df["symbol"].str.lower().str.contains(query_lower, na=False)]

    if not match.empty:
        row = match.iloc[0]
        symbol = row["symbol"]
        feat = {k: row[k] for k in [
            "rise_hours", "price_roc", "volume_roc", "holder_roc",
            "ath", "holders_at_ath", "decay_hours", "holder_decay_roc",
            "price_decay_roc", "total_hours", "rise_pct",
        ]}
    else:
        # Out-of-sample: try as contract address
        address = query.strip()
        if len(address) < 20:
            st.warning(f"未找到「{query}」。请输入完整的合约地址来查询外部 token。")
            return

        is_oos = True
        with st.spinner(f"从 GMGN 获取 {address[:16]}... 的数据"):
            # First check if we already have data locally
            feat = compute_features(address)

            if feat is None:
                # Fetch from GMGN
                try:
                    from gmgn_api import fetch_token_data
                    fetch_token_data("sol", address)
                    feat = compute_features(address)
                except Exception as e:
                    st.error(f"获取数据失败: {e}")
                    return

        if feat is None:
            st.warning("无法计算特征。可能原因：token 市值未达 $100K，或数据不足。")
            return

        # Try to get symbol
        data_dir = os.path.join(DATA_DIR, address)
        h_files = sorted(glob.glob(os.path.join(data_dir, "token_mcap_candles_[0-9]*.json")))
        if h_files:
            try:
                with open(h_files[-1]) as f:
                    raw = json.load(f)
                s = (raw or {}).get("data", {}).get("symbol")
                if s:
                    symbol = s
            except Exception:
                pass
        if symbol == "?":
            symbol = address[:12] + "..."

    # Predict cluster
    scaler = model["scaler"]
    km = model["kmeans"]
    cluster_order = model["cluster_order"]
    rank_map = {c: i + 1 for i, c in enumerate(cluster_order)}

    vec = scaler.transform([build_feature_vector(feat)])
    cid = int(km.predict(vec)[0])
    rank = rank_map.get(cid, 0)
    meta = CLUSTER_META.get(rank, {})
    color = meta.get("color", "#888")
    same_cluster = df[df["cluster_id"] == cid]

    # Header
    oos_badge = ' <span style="background:#f59e0b; color:white; padding:2px 8px; border-radius:8px; font-size:0.75em;">OUT-OF-SAMPLE</span>' if is_oos else ''
    st.markdown(f"""
    <div style="border-left: 5px solid {color}; padding: 12px 16px; margin: 16px 0; background: {color}10;">
        <h3 style="margin:0;">{symbol} → <span style="color:{color};">#{rank} {meta.get('name', '?')}</span>{oos_badge}</h3>
        <p style="margin:4px 0 0 0; color: #666;">{meta.get('name_cn', '')} — {meta.get('desc', '')[:80]}...</p>
    </div>
    """, unsafe_allow_html=True)

    # Token features
    col1, col2, col3 = st.columns(3)
    col1.metric("ATH", f"${feat['ath']:,.0f}")
    col1.metric("价格增速", f"${feat['price_roc']:,.0f}/h")
    col2.metric("上升时长", f"{feat['rise_hours']:.0f}h")
    col2.metric("衰减时长", f"{feat['decay_hours']:.0f}h")
    col3.metric("Holder@ATH", f"{feat['holders_at_ath']:,.0f}")
    col3.metric("Holder增速", f"{feat['holder_roc']:.1f}/h")

    # Cluster comparison
    st.markdown(f"### 同 Cluster 对比（#{rank} {meta.get('name', '?')}, {len(same_cluster)} tokens）")
    compare = pd.DataFrame({
        "指标": ["ATH", "上升时长", "衰减时长", "Holder@ATH", "价格增速", "总时长", "上升占比"],
        "该 Token": [
            f"${feat['ath']:,.0f}", f"{feat['rise_hours']:.0f}h", f"{feat['decay_hours']:.0f}h",
            f"{feat['holders_at_ath']:,.0f}", f"${feat['price_roc']:,.0f}/h",
            f"{feat['total_hours']:.0f}h", f"{feat['rise_pct']:.0%}",
        ],
        "Cluster 中位": [
            f"${same_cluster['ath'].median():,.0f}", f"{same_cluster['rise_hours'].median():.0f}h",
            f"{same_cluster['decay_hours'].median():.0f}h",
            f"{same_cluster['holders_at_ath'].median():,.0f}",
            f"${same_cluster['price_roc'].median():,.0f}/h",
            f"{same_cluster['total_hours'].median():.0f}h",
            f"{same_cluster['rise_pct'].median():.0%}",
        ],
    })
    st.dataframe(compare, hide_index=True)

    # Distance to all clusters
    st.markdown("### 到各 Cluster 的距离")
    dist_rows = []
    for c in cluster_order:
        d = float(np.linalg.norm(vec[0] - km.cluster_centers_[c]))
        r = rank_map[c]
        m = CLUSTER_META.get(r, {})
        dist_rows.append({
            "Cluster": f"#{r} {m.get('name', '?')}",
            "距离": f"{d:.2f}",
            "": "← 当前" if c == cid else "",
        })
    st.dataframe(pd.DataFrame(dist_rows), hide_index=True)

    # Similar tokens
    member_vecs = scaler.transform([build_feature_vector(r) for _, r in same_cluster.iterrows()])
    dists = np.sqrt(((member_vecs - vec[0]) ** 2).sum(axis=1))
    same_cluster = same_cluster.copy()
    same_cluster["distance"] = dists
    similar = same_cluster.nsmallest(8, "distance")

    st.markdown("### 最相似的历史 Token")
    sim_display = similar[["symbol", "ath", "rise_hours", "decay_hours",
                            "holders_at_ath", "holder_roc", "distance"]].copy()
    sim_display.columns = ["Token", "ATH", "上升(h)", "衰减(h)", "Holder@ATH", "Holder增速(/h)", "距离"]
    sim_display["ATH"] = sim_display["ATH"].apply(lambda x: f"${x:,.0f}")
    sim_display["距离"] = sim_display["距离"].apply(lambda x: f"{x:.2f}")
    st.dataframe(sim_display, hide_index=True)

    # ── Regression Position Analysis (only for organic clusters #5, #6) ──
    if rank in [5, 6]:
        _render_regression_position(df, feat, symbol, rank)


# ── Token Discovery: L1 Cluster Filter ───────────────────────────────────────


def section_l1_filter(df, model):
    """L1 漏斗：展示 Pipeline 产出的 Candidate Pool。"""
    from token_discovery import db as disc_db

    disc_db.init_db()

    st.markdown("## L1: Cluster 筛选")
    st.markdown("""
    **自动化 Pipeline** 每 15 分钟运行一次：Codex Trending 扫描 → GMGN 数据获取 → Cluster 分类 → 仅保留 #5 Organic Runner 和 #6 Fast Organic。
    """)
    st.code("PYTHONPATH=. python -m token_discovery --loop", language="bash")

    # ── Candidate Pool ──
    candidates = disc_db.get_candidates()
    pool_size = len(candidates)

    st.markdown(f"### Candidate Pool（{pool_size} 个 token）")

    if pool_size == 0:
        st.info("Candidate Pool 为空。启动 Pipeline 后数据会自动填充。")
    else:
        # Metrics
        cdf = pd.DataFrame([dict(c) for c in candidates])

        n_organic = len(cdf[cdf["cluster_rank"] == 5])
        n_fast = len(cdf[cdf["cluster_rank"] == 6])
        avg_mcap = cdf["mcap_at_discovery"].mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("总 Candidates", f"{pool_size}")
        col2.metric("#5 Organic Runner", f"{n_organic}")
        col3.metric("#6 Fast Organic", f"{n_fast}")
        col4.metric("平均 MCap", f"${avg_mcap:,.0f}")

        # Table
        table_data = []
        for _, r in cdf.iterrows():
            meta = CLUSTER_META.get(r["cluster_rank"], {})
            table_data.append({
                "Token": r["symbol"],
                "Cluster": f"#{r['cluster_rank']} {meta.get('name', '?')}",
                "发现时间": r["first_seen"][:16],
                "MCap": f"${r['mcap_at_discovery']:,.0f}",
                "Vol 4h": f"${r['vol4h_at_discovery']:,.0f}",
                "Chg 4h": f"{r['change4h_at_discovery']*100:+.1f}%",
                "Holders": r["holders_at_discovery"],
                "ATH": f"${r['ath']:,.0f}",
                "上升时长": f"{r['rise_hours']:.0f}h",
            })
        st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)

        # ── 3D Scatter ──
        st.markdown("### 3D 可视化")
        st.markdown("*每个点 = 一个 candidate 在被发现时的状态。*")

        color_map = {5: "#22c55e", 6: "#3b82f6"}
        cdf["color"] = cdf["cluster_rank"].map(color_map)
        cdf["cluster_label"] = cdf["cluster_rank"].map(
            lambda r: f"#{r} {CLUSTER_META.get(r, {}).get('name', '?')}"
        )

        # Handle missing lifetime_hours column (old data)
        if "lifetime_hours" not in cdf.columns:
            cdf["lifetime_hours"] = 0

        fig = go.Figure()
        for rank in [5, 6]:
            sub = cdf[cdf["cluster_rank"] == rank]
            if sub.empty:
                continue
            meta = CLUSTER_META.get(rank, {})
            fig.add_trace(go.Scatter3d(
                x=sub["mcap_at_discovery"],
                y=sub["holders_at_discovery"],
                z=sub["lifetime_hours"].clip(lower=0.1),
                mode="markers",
                marker=dict(size=6, color=color_map[rank], opacity=0.8),
                text=[
                    f"<b>{r['symbol']}</b><br>"
                    f"#{r['cluster_rank']} {meta.get('name', '?')}<br>"
                    f"MCap: ${r['mcap_at_discovery']:,.0f}<br>"
                    f"Holders: {r['holders_at_discovery']:,}<br>"
                    f"Lifetime: {r.get('lifetime_hours', 0):.0f}h<br>"
                    f"发现: {r['first_seen'][:16]}"
                    for _, r in sub.iterrows()
                ],
                hoverinfo="text",
                name=f"#{rank} {meta.get('name', '?')} ({len(sub)})",
            ))

        fig.update_layout(
            scene=dict(
                xaxis_title="MCap at Discovery ($)",
                yaxis_title="Holders",
                zaxis_title="Token Lifetime (hours)",
                xaxis=dict(type="log", backgroundcolor="white", gridcolor="rgb(200,200,200)"),
                yaxis=dict(type="log", backgroundcolor="white", gridcolor="rgb(200,200,200)"),
                zaxis=dict(backgroundcolor="white", gridcolor="rgb(200,200,200)"),
                bgcolor="white",
            ),
            paper_bgcolor="white",
            font=dict(color="black"),
            height=600,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Pipeline Logs ──
    st.divider()
    st.markdown("### Pipeline 运行日志")

    logs = disc_db.get_scan_logs(limit=20)
    if logs:
        log_data = []
        for l in logs:
            log_data.append({
                "时间": l["timestamp"][:19],
                "Codex 扫描": l["codex_count"],
                "GMGN 获取": l["gmgn_fetched"],
                "已分类": l["classified"],
                "新增 Candidate": l["new_candidates"],
                "Pool 总量": l["total_pool"],
            })
        st.dataframe(pd.DataFrame(log_data), hide_index=True, use_container_width=True)

        # Expandable: latest scan token details
        latest = logs[0]
        details = json.loads(latest["token_details"]) if latest["token_details"] else []
        if details:
            with st.expander(f"最近一次扫描详情（{latest['timestamp'][:19]}，{len(details)} tokens）"):
                det_data = []
                for d in details:
                    passed = "✅" if d.get("passed_l1") else "❌"
                    new = "🆕" if d.get("is_new") else ""
                    det_data.append({
                        "": f"{passed} {new}",
                        "Token": d.get("symbol", "?"),
                        "Cluster": f"#{d['cluster_rank']} {d['cluster_name']}" if d.get("cluster_rank") else "无法分类",
                        "MCap": f"${d.get('mcap', 0):,.0f}",
                        "Holders": d.get("holders", 0),
                        "Vol 4h": f"${d.get('vol4h', 0):,.0f}",
                        "ATH": f"${d.get('ath', 0):,.0f}" if d.get("ath") else "N/A",
                    })
                st.dataframe(pd.DataFrame(det_data), hide_index=True, use_container_width=True)
    else:
        st.info("暂无运行日志。启动 Pipeline 后自动生成。")


# ── Main ─────────────────────────────────────────────────────────────────────


df, model = load_cluster_data()

if df is None:
    st.error("聚类数据未找到。请先运行 `python baseline_cluster_v2.py`。")
    st.stop()

MONITOR_REGRESSION_PAIRS = [
    ("volume_roc", "holder_roc", "Volume增速 ($/h)", "Holder增速 (/h)", "holder吸引效率", "z_holder_efficiency"),
    ("price_roc", "volume_roc", "价格增速 ($/h)", "Volume增速 ($/h)", "成交量真实性", "z_volume_authenticity"),
    ("ath", "holders_at_ath", "ATH ($)", "Holder@ATH", "市值可持续性", "z_mcap_sustainability"),
    ("price_roc", "holder_roc", "价格增速 ($/h)", "Holder增速 (/h)", "价格-社区联动", "z_price_community"),
    ("holder_roc", "holders_at_ath", "Holder增速 (/h)", "Holder@ATH", "增长天花板", "z_growth_ceiling"),
]


def section_candidate_monitor(df):
    """Candidate Monitor — Entry Score + Regression analysis, merged view."""
    from token_discovery import db as disc_db
    disc_db.init_db()

    st.markdown("## Candidate Monitor")
    st.markdown("持续监测 Candidate Pool，基于 Regression 质量 + 实时量价信号进行综合评分。")

    with st.expander("权重与评分说明"):
        st.markdown("""
        **Entry Score (0-100) = 加权综合分**

        | 维度 | 权重 | 计算方式 | 性质 |
        |---|---|---|---|
        | **Regression 质量** | 50% | 5 个 regression z-score 加权综合 | 结构性（全周期增长质量） |
        | **动量强度** | 15% | 4h 涨幅归一化 (0%→0, 50%→1) | 短期趋势 |
        | **放量程度** | 15% | 近4h vol / 24h均值4h vol (1x→0, 3x→1) | 短期成交量 |
        | **买卖压力** | 10% | buy/(buy+sell) (0.5→0, 0.7→1) | 短期方向 |
        | **Holder 动量** | 10% | 当前holders vs发现时增长率 (0%→0, 20%→1) | 短期社区 |

        **Regression 质量内部权重:**
        增长天花板 35% | Holder吸引效率 25% | 成交量真实性 20% | 市值可持续性 10% | 价格-社区联动 10%
        """)

    # ── Load Entry Scores ──
    conn = disc_db.get_conn()
    try:
        entry_rows = conn.execute("SELECT * FROM entry_scores ORDER BY entry_score DESC").fetchall()
    except Exception:
        entry_rows = []
    conn.close()

    scores = disc_db.get_scores()
    sdf = pd.DataFrame([dict(s) for s in scores]) if scores else pd.DataFrame()

    if not entry_rows:
        st.info("暂无评分数据。运行以下命令生成：")
        st.code("PYTHONPATH=. python -m token_discovery.monitor && PYTHONPATH=. python -m token_discovery.entry_score", language="bash")
        return

    edf = pd.DataFrame([dict(r) for r in entry_rows])

    # ── Top Metrics ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Candidates", f"{len(edf)}")
    col2.metric("最高分", f"{edf['entry_score'].max():.1f}")
    col3.metric("中位分", f"{edf['entry_score'].median():.1f}")
    col4.metric("平均分", f"{edf['entry_score'].mean():.1f}")

    # ── Ranked Table ──
    st.markdown("### Entry Score 排名")

    table_rows = []
    for i, (_, r) in enumerate(edf.iterrows()):
        raw = r.get("raw_score", r["entry_score"])
        fresh = r.get("freshness", 1.0)
        lifetime = r.get("lifetime_hours", 0)
        table_rows.append({
            "排名": i + 1,
            "Token": r["symbol"],
            "Entry Score": f"{r['entry_score']:.1f}",
            "原始分": f"{raw:.1f}",
            "新鲜度": f"{fresh:.0%}",
            "Token Age": f"{lifetime:.0f}h" if lifetime else "",
            "Reg (50%)": f"{r['s_regression']:.2f}",
            "动量 (15%)": f"{r['s_momentum']:.2f}",
            "放量 (15%)": f"{r['s_volume']:.2f}",
            "买压 (10%)": f"{r['s_buy']:.2f}",
            "Holder (10%)": f"{r['s_holder']:.2f}",
            "4h涨幅": f"{r['change4h']*100:+.1f}%",
        })
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)

    # ── Top 10 Dimension Breakdown ──
    st.divider()
    st.markdown("### Top 10 维度分解")

    top10 = edf.head(10)
    if len(top10) > 0:
        fig = go.Figure()
        dims = [
            ("s_regression", "Regression (50%)", "#ef4444"),
            ("s_momentum", "动量 (15%)", "#f59e0b"),
            ("s_volume", "放量 (15%)", "#3b82f6"),
            ("s_buy", "买压 (10%)", "#22c55e"),
            ("s_holder", "Holder (10%)", "#a855f7"),
        ]
        weights = [0.50, 0.15, 0.15, 0.10, 0.10]
        for (col, name, color), w in zip(dims, weights):
            fig.add_trace(go.Bar(
                y=top10["symbol"], x=top10[col] * w * 100,
                name=name, orientation="h", marker_color=color,
            ))
        fig.update_layout(barmode="stack", title="Entry Score 组成（加权）",
                           xaxis_title="Score", yaxis=dict(autorange="reversed"),
                           height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    # ── Regression Plots ──
    if len(sdf) > 0:
        st.divider()
        st.markdown("### Regression 维度分析")
        st.markdown("*灰色 = 历史 baseline，彩色 = 当前 candidate（绿>0.3, 黄中间, 红<-0.3）*")

        organic = df[df["rank"].isin([5, 6])].copy()

        for x_col, y_col, x_label, y_label, dim_name, z_col in MONITOR_REGRESSION_PAIRS:
            sub = organic[(organic[x_col] > 0) & (organic[y_col] > 0)]
            if len(sub) < 10:
                continue

            log_x = np.log10(sub[x_col].values)
            log_y = np.log10(sub[y_col].values)
            coeffs = np.polyfit(log_x, log_y, 2)
            poly = np.poly1d(coeffs)
            x_range = np.linspace(log_x.min() - 0.3, log_x.max() + 0.3, 200)
            y_fit = poly(x_range)
            residual_std = np.std(log_y - poly(log_x))

            col_map = {
                "volume_roc": "current_volume_roc", "price_roc": "current_price_roc",
                "holder_roc": "current_holder_roc", "ath": "current_ath",
                "holders_at_ath": "current_holders_at_ath",
            }
            sx, sy = col_map.get(x_col, x_col), col_map.get(y_col, y_col)
            valid = sdf[(sdf[sx] > 0) & (sdf[sy] > 0)].copy()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sub[x_col], y=sub[y_col], mode="markers",
                                      marker=dict(size=4, color="#d1d5db", opacity=0.4),
                                      name="Baseline", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=10**x_range, y=10**y_fit, mode="lines",
                                      line=dict(color="#ef4444", width=2), name="回归线"))
            fig.add_trace(go.Scatter(
                x=np.concatenate([10**x_range, 10**x_range[::-1]]),
                y=np.concatenate([10**(y_fit+residual_std), 10**(y_fit-residual_std)[::-1]]),
                fill="toself", fillcolor="rgba(239,68,68,0.06)",
                line=dict(color="rgba(0,0,0,0)"), showlegend=False))

            if len(valid) > 0:
                colors = ["#22c55e" if z > 0.3 else "#ef4444" if z < -0.3 else "#f59e0b"
                           for z in valid[z_col]]
                fig.add_trace(go.Scatter(
                    x=valid[sx], y=valid[sy], mode="markers+text",
                    marker=dict(size=10, color=colors, line=dict(width=1, color="white")),
                    text=valid["symbol"], textposition="top center", textfont=dict(size=9),
                    hovertext=[f"<b>{r['symbol']}</b><br>z={r[z_col]:+.2f}" for _, r in valid.iterrows()],
                    hoverinfo="text", name="Candidates"))

            fig.update_layout(title=f"{dim_name}: {x_label} vs {y_label}",
                               xaxis_title=x_label, yaxis_title=y_label,
                               xaxis_type="log", yaxis_type="log",
                               height=400, margin=dict(l=50, r=20, t=40, b=40))
            st.plotly_chart(fig, use_container_width=True)

    # ── Score Distribution ──
    st.divider()
    st.markdown("### Entry Score 分布")
    fig_hist = px.histogram(edf, x="entry_score", nbins=20,
                             title=f"Entry Score 分布（{len(edf)} candidates）",
                             labels={"entry_score": "Entry Score"},
                             color_discrete_sequence=["#3b82f6"])
    fig_hist.update_layout(height=300)
    st.plotly_chart(fig_hist, use_container_width=True)


def section_position_management():
    """持仓管理 — 分层止盈止损策略设计。"""
    st.markdown("## 持仓管理")
    st.markdown("基于 244 个 Organic token 的历史数据，设计分层止盈止损策略。初始建仓 **$5,000**。")

    # ── Historical Data Summary ──
    st.markdown("### 历史数据基础")

    col1, col2, col3 = st.columns(3)
    col1.markdown("""
    **涨幅分布（244 tokens）**
    - 小涨 (<50%): 20%
    - 中涨 (50-200%): 16%
    - 大涨 (200-500%): 23%
    - 暴涨 (500%+): **40%**
    """)
    col2.markdown("""
    **上升期最大回调**
    - <10% 回调: 16%
    - 10-20%: 7%
    - 20-40%: 18%
    - **>40% 回调: 59%**
    """)
    col3.markdown("""
    **ATH 后暴跌**
    - 中位跌幅: -74% (#5), -99% (#6)
    - 不卖 = 大概率归零
    - 必须分批止盈
    """)

    st.divider()

    # ── Strategy Definition ──
    POSITION_SIZE = 5000

    TP_LEVELS = [
        (0.30, 0.15, "+30%: 先回一部分本金，降低心理压力"),
        (0.80, 0.20, "+80%: 接近翻倍，锁定第一波利润"),
        (2.00, 0.20, "+200%: 3x，已卖出超过一半本金"),
        (5.00, 0.20, "+500%: 6x，大部分利润已锁定"),
        (10.00, 0.15, "+1000%: 10x，留最后 10% 搏更大收益"),
    ]

    SL_LEVELS = [
        (-0.15, 0.30, "-15%: 初步减仓，但不全清（59% 的 runner 回调 >40%）"),
        (-0.30, 0.30, "-30%: 大幅减仓，控制风险"),
        (-0.50, 1.00, "-50%: 硬止损，腰斩 = 大概率不回来"),
    ]

    TRAILING_STOP_PCT = 0.30  # 最后 10% 仓位的 trailing stop

    st.markdown("### 止盈策略（5 档递进）")

    tp_data = []
    remaining = 1.0
    cumulative_sold = 0.0
    for gain, sell_pct, desc in TP_LEVELS:
        actual_sell = sell_pct
        remaining -= actual_sell
        cumulative_sold += actual_sell
        tp_data.append({
            "涨幅触发": f"+{gain*100:.0f}%",
            "卖出比例": f"{actual_sell*100:.0f}%",
            "剩余仓位": f"{remaining*100:.0f}%",
            "累计已卖": f"{cumulative_sold*100:.0f}%",
            "说明": desc,
        })
    tp_data.append({
        "涨幅触发": "Trailing Stop",
        "卖出比例": f"{remaining*100:.0f}% (全部)",
        "剩余仓位": "0%",
        "累计已卖": "100%",
        "说明": f"最后 {remaining*100:.0f}% 仓位：从最高点回撤 {TRAILING_STOP_PCT*100:.0f}% 触发",
    })
    st.dataframe(pd.DataFrame(tp_data), hide_index=True, use_container_width=True)

    st.markdown("### 止损策略（3 档递进）")

    sl_data = []
    sl_remaining = 1.0
    for loss, sell_pct, desc in SL_LEVELS:
        actual = min(sell_pct, sl_remaining)
        sl_remaining -= actual
        sl_data.append({
            "跌幅触发": f"{loss*100:.0f}%",
            "卖出比例": f"{actual*100:.0f}%" + (" (全部)" if sl_remaining <= 0 else ""),
            "剩余仓位": f"{max(sl_remaining, 0)*100:.0f}%",
            "说明": desc,
        })
    st.dataframe(pd.DataFrame(sl_data), hide_index=True, use_container_width=True)

    st.divider()

    # ── Visualization with real token data ──
    st.markdown("### 可视化：实际 Token 回测")

    # Token selector from cluster #5 + #6
    cluster_csv = os.path.join(CLUSTER_DIR, "clusters.csv")
    if os.path.isfile(cluster_csv):
        cdf_all = pd.read_csv(cluster_csv)
        organic_tokens = cdf_all[cdf_all["rank"].isin([5, 6])].sort_values("ath", ascending=False)
        token_options = [f"{r['symbol']} (ATH ${r['ath']:,.0f}, #{r['rank']})"
                         for _, r in organic_tokens.iterrows()]
        selected_idx = st.selectbox("选择 Token 回测", range(len(token_options)),
                                     format_func=lambda i: token_options[i],
                                     index=0, key="pos_token_select")
        selected_token = organic_tokens.iloc[selected_idx]
    else:
        st.warning("聚类数据未找到")
        return

    # Load actual price data
    addr = selected_token["address"]
    h_files = sorted([f for f in glob.glob(os.path.join(DATA_DIR, addr, "token_mcap_candles_[0-9]*.json"))
                       if "5m" not in os.path.basename(f)])
    if not h_files:
        st.warning("无价格数据")
        return

    with open(h_files[-1]) as f:
        raw = json.load(f)
    candles = (raw or {}).get("data", {}).get("list", [])
    if not candles:
        st.warning("无蜡烛图数据")
        return

    mcap_arr = np.array([float(c["close"]) for c in sorted(candles, key=lambda x: int(x["time"]))])
    start_idx = next((i for i in range(len(mcap_arr)) if mcap_arr[i] >= 100000), None)
    if start_idx is None:
        st.warning("Token 市值未达到 $100K")
        return

    mcap_arr = mcap_arr[start_idx:]
    entry_price = mcap_arr[0]
    price_pcts = (mcap_arr - entry_price) / entry_price

    st.caption(f"**{selected_token['symbol']}** — 从市值 ${entry_price:,.0f} 入场，共 {len(price_pcts)} 小时数据")

    # Strategy simulation
    remaining_strat = 1.0
    peak_gain = 0.0
    trailing_active = False
    realized_strat = 0.0
    strat_value = []
    strat_remaining = []
    tp_markers = []
    sl_markers = []

    # Track which TP/SL levels have been triggered
    tp_triggered = [False] * len(TP_LEVELS)
    sl_triggered = [False] * len(SL_LEVELS)
    trailing_triggered = False

    for i, pct in enumerate(price_pcts):
        if remaining_strat <= 0.001:
            strat_value.append(realized_strat)
            strat_remaining.append(0)
            continue

        current_value = remaining_strat * POSITION_SIZE * (1 + pct) + realized_strat
        peak_gain = max(peak_gain, pct)

        # Check TP levels
        for j, (trigger, sell_pct, _) in enumerate(TP_LEVELS):
            if not tp_triggered[j] and pct >= trigger and remaining_strat > 0.001:
                actual_sell = min(sell_pct, remaining_strat)
                realized_strat += actual_sell * POSITION_SIZE * (1 + pct)
                remaining_strat -= actual_sell
                tp_triggered[j] = True
                tp_markers.append((i, pct, f"TP {trigger*100:.0f}%\n卖{actual_sell*100:.0f}%"))

        # Check trailing stop for last portion
        if all(tp_triggered) and not trailing_triggered and remaining_strat > 0.001:
            trailing_active = True
            if pct < peak_gain * (1 - TRAILING_STOP_PCT):
                realized_strat += remaining_strat * POSITION_SIZE * (1 + pct)
                tp_markers.append((i, pct, f"Trailing Stop\n卖{remaining_strat*100:.0f}%"))
                remaining_strat = 0
                trailing_triggered = True

        # Check SL levels (only if price is negative from entry)
        if pct < 0:
            for j, (trigger, sell_pct, _) in enumerate(SL_LEVELS):
                if not sl_triggered[j] and pct <= trigger and remaining_strat > 0.001:
                    actual_sell = min(sell_pct, remaining_strat)
                    realized_strat += actual_sell * POSITION_SIZE * (1 + pct)
                    remaining_strat -= actual_sell
                    sl_triggered[j] = True
                    sl_markers.append((i, pct, f"SL {trigger*100:.0f}%\n卖{actual_sell*100:.0f}%"))

        total = remaining_strat * POSITION_SIZE * (1 + pct) + realized_strat
        strat_value.append(total)
        strat_remaining.append(remaining_strat)

    # Hold-to-end comparison
    hold_value = [POSITION_SIZE * (1 + p) for p in price_pcts]

    # Build chart
    fig = go.Figure()

    # Price path (secondary y axis reference)
    fig.add_trace(go.Scatter(
        x=list(range(len(price_pcts))),
        y=[POSITION_SIZE * (1 + p) for p in price_pcts],
        mode="lines", line=dict(color="#d1d5db", width=1, dash="dot"),
        name="持有不卖 (Buy & Hold)", yaxis="y",
    ))

    # Strategy value
    fig.add_trace(go.Scatter(
        x=list(range(len(strat_value))),
        y=strat_value,
        mode="lines", line=dict(color="#22c55e", width=3),
        name="分批止盈止损策略", yaxis="y",
        fill="tonexty", fillcolor="rgba(34,197,94,0.1)",
    ))

    # TP markers
    for idx, pct, label in tp_markers:
        fig.add_trace(go.Scatter(
            x=[idx], y=[strat_value[idx]], mode="markers+text",
            marker=dict(size=12, color="#22c55e", symbol="triangle-up"),
            text=[label], textposition="top center", textfont=dict(size=9, color="#22c55e"),
            showlegend=False,
        ))

    # SL markers
    for idx, pct, label in sl_markers:
        fig.add_trace(go.Scatter(
            x=[idx], y=[strat_value[idx]], mode="markers+text",
            marker=dict(size=12, color="#ef4444", symbol="triangle-down"),
            text=[label], textposition="bottom center", textfont=dict(size=9, color="#ef4444"),
            showlegend=False,
        ))

    # Entry line
    fig.add_hline(y=POSITION_SIZE, line_dash="dash", line_color="#94a3b8",
                   annotation_text=f"Entry: ${POSITION_SIZE:,}", annotation_position="left")

    fig.update_layout(
        title=f"{selected_token['symbol']} — 分批止盈止损 vs 持有不卖",
        yaxis_title="账户价值 ($)",
        xaxis_title="时间 (小时)",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Final PnL summary
    final_strat = strat_value[-1] if strat_value else POSITION_SIZE
    final_hold = POSITION_SIZE * (1 + price_pcts[-1])
    peak_hold = POSITION_SIZE * (1 + price_pcts.max())

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("分批策略最终", f"${final_strat:,.0f}",
                   f"{(final_strat/POSITION_SIZE - 1)*100:+.0f}%")
    col_s2.metric("持有不卖最终", f"${final_hold:,.0f}",
                   f"{(final_hold/POSITION_SIZE - 1)*100:+.0f}%")
    col_s3.metric("策略优势", f"${final_strat - final_hold:+,.0f}",
                   f"峰值时持有价值 ${peak_hold:,.0f}")

    # ── Scenario Analysis ──
    st.divider()
    st.markdown("### 场景模拟")
    st.markdown("不同涨幅下，分批策略 vs 持有不卖的对比：")

    scenarios = [
        ("小涨后跌回", 0.5, -0.80),
        ("中涨后跌回", 2.0, -0.75),
        ("大涨后跌回", 5.0, -0.90),
        ("暴涨后跌回", 10.0, -0.95),
        ("直接下跌", 0.0, -0.50),
    ]

    scenario_rows = []
    for name, peak, crash in scenarios:
        # Strategy simulation
        rem = 1.0
        real = 0.0

        # TP
        for trigger, sell_pct, _ in TP_LEVELS:
            if peak >= trigger and rem > 0:
                s = min(sell_pct, rem)
                real += s * POSITION_SIZE * (1 + trigger)
                rem -= s

        # After crash
        final_pct = peak * (1 + crash) if peak > 0 else crash
        if final_pct < 0:
            # SL
            for trigger, sell_pct, _ in SL_LEVELS:
                if final_pct <= trigger and rem > 0:
                    s = min(sell_pct, rem)
                    real += s * POSITION_SIZE * (1 + trigger)
                    rem -= s

        strat_total = real + rem * POSITION_SIZE * (1 + final_pct)
        strat_pnl = strat_total - POSITION_SIZE

        hold_total = POSITION_SIZE * (1 + final_pct)
        hold_pnl = hold_total - POSITION_SIZE

        advantage = strat_pnl - hold_pnl

        scenario_rows.append({
            "场景": name,
            "峰值涨幅": f"+{peak*100:.0f}%",
            "最终跌幅": f"{crash*100:.0f}%",
            "策略收益": f"${strat_pnl:+,.0f} ({strat_pnl/POSITION_SIZE*100:+.0f}%)",
            "持有收益": f"${hold_pnl:+,.0f} ({hold_pnl/POSITION_SIZE*100:+.0f}%)",
            "策略优势": f"${advantage:+,.0f}",
        })

    st.dataframe(pd.DataFrame(scenario_rows), hide_index=True, use_container_width=True)

    st.markdown("""
    ### 关键结论

    1. **分批止盈在所有"先涨后跌"的场景中都显著优于持有不卖**
    2. 涨幅越大 + 崩盘越深 = 策略优势越大（暴涨后跌回场景差 $40K+）
    3. 唯一劣势：直接单边下跌时策略比持有多亏一点（因为止损卖出时已确认亏损）
    4. 但这个劣势极小（<$200），而上涨时的优势巨大
    """)

    st.divider()
    st.info("🚧 持仓管理的自动化执行将在下一阶段实现。当前为策略设计阶段。")


# Navigation via styled buttons as menu items
MENU = {
    "Baseline Cluster v2": {
        "caption": f"{len(df)} tokens · 6 clusters · 2026年后",
        "pages": ["数据方法论", "聚类结果", "3D 轨迹图", "散点分析", "Regression 分析", "Token 查询"],
    },
    "Token Discovery": {
        "caption": "漏斗筛选 → 量化信号 → 进场",
        "pages": ["L1: Cluster 筛选", "Candidate Monitor", "持仓管理"],
    },
}

if "page" not in st.session_state:
    st.session_state.page = "数据方法论"

# Custom CSS for menu buttons
st.markdown("""
<style>
div[data-testid="stSidebar"] .stButton > button {
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 4px 12px !important;
    font-size: 0.9em !important;
    border: none !important;
    background: transparent !important;
    color: #333 !important;
}
div[data-testid="stSidebar"] .stButton > button:hover {
    background: #f0f0f0 !important;
}
div[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    color: #ef4444 !important;
    font-weight: 600 !important;
    background: #fef2f2 !important;
    border-left: 3px solid #ef4444 !important;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("Meme Trading System")

    for section, info in MENU.items():
        st.divider()
        st.markdown(f"**{section}**")
        for p in info["pages"]:
            is_active = st.session_state.page == p
            if st.button(
                p,
                key=f"menu_{p}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.page = p
                st.rerun()

page = st.session_state.page

# Content routing
CLUSTER_PAGES = MENU["Baseline Cluster v2"]["pages"]
DISCOVERY_PAGES = MENU["Token Discovery"]["pages"]

if page in CLUSTER_PAGES:
    st.title("Baseline Clustering v2")
    st.caption("基于结果特征的 Memecoin 生命周期聚类 | 500 tokens · 6 clusters · 11 features")

    if page == "数据方法论":
        section_methodology()
    elif page == "聚类结果":
        section_clusters(df)
    elif page == "3D 轨迹图":
        section_3d_chart(df)
    elif page == "散点分析":
        section_scatter(df)
    elif page == "Regression 分析":
        section_regression(df)
    elif page == "Token 查询":
        section_query(df, model)

elif page in DISCOVERY_PAGES:
    st.title("Token Discovery")
    st.caption("市场扫描 → Cluster 筛选 → 量化信号 → 进场")

    if page == "L1: Cluster 筛选":
        section_l1_filter(df, model)
    elif page == "Candidate Monitor":
        section_candidate_monitor(df)
    elif page == "持仓管理":
        section_position_management()
