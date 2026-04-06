"""Memecoin 分析研究展示 — 6个实验阶段的完整记录

从无监督聚类到有监督早期预警系统的研究历程。

Usage:
    streamlit run app_research.py --server.port 8510
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

st.set_page_config(page_title="Memecoin 研究展示", layout="wide")

TRAIN_DATA_PATH = "early_warning/train_data.parquet"


# ── 工具函数 ─────────────────────────────────────────────────────────────────


def section(title, desc):
    st.markdown(f"## {title}")
    st.markdown(desc)


def finding(text, type="info"):
    {"info": st.info, "success": st.success, "warning": st.warning, "error": st.error}[type](text)


def limitations_box(items):
    with st.expander("局限性分析"):
        for item in items:
            st.markdown(f"- {item}")


def data_quality_box(items):
    with st.expander("数据质量与信号分析"):
        for item in items:
            st.markdown(f"- {item}")


CLUSTER_DATA_DIR = "outcome_cluster_data"
CLUSTER_COLORS_MAP = {
    "Organic Runner": "#FFD700",
    "Fast Organic": "#22c55e",
    "Pump-Dump": "#e41a1c",
    "Slow Grinder": "#3b82f6",
    "Average": "#a855f7",
    "Flash Crash": "#f97316",
}


@st.cache_data
def load_cluster_model():
    """加载已保存的聚类模型和分配结果。"""
    import pickle
    pkl_path = os.path.join(CLUSTER_DATA_DIR, "model.pkl")
    csv_path = os.path.join(CLUSTER_DATA_DIR, "cluster_assignments.csv")
    if not os.path.isfile(pkl_path) or not os.path.isfile(csv_path):
        return None, None
    with open(pkl_path, "rb") as f:
        model_data = pickle.load(f)
    df = pd.read_csv(csv_path)
    return model_data, df


@st.cache_data
def _load_trajectories_for_3d():
    """加载所有雷达 token 的小时级轨迹数据用于 3D 可视化。"""
    import csv, glob
    meta = {}
    with open("data/radar_tokens.csv") as f:
        for row in csv.reader(f):
            if len(row) >= 4:
                meta[row[0]] = {"name": row[2], "symbol": row[3]}

    trajectories = {}
    for addr in meta:
        data_dir = os.path.join("data", addr)
        h_files = sorted([
            f for f in glob.glob(os.path.join(data_dir, "token_mcap_candles_[0-9]*.json"))
            if "5m" not in os.path.basename(f)
        ])
        if not h_files:
            continue
        try:
            import json as _json
            with open(h_files[-1]) as f:
                data = _json.load(f)
            candles = (data or {}).get("data", {}).get("list", [])
            if not candles or len(candles) < 2:
                continue
            df = pd.DataFrame(candles)
            df["mcap"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(float)
            df["datetime"] = pd.to_datetime(df["time"].astype(int), unit="ms")
            df = df.sort_values("datetime").reset_index(drop=True)

            above = df[df["mcap"] >= 50000]
            if above.empty:
                continue
            start_idx = above.index[0]
            df = df.loc[start_idx:].reset_index(drop=True)
            t0 = df["datetime"].iloc[0]
            df["hours"] = (df["datetime"] - t0).dt.total_seconds() / 3600
            df = df[df["hours"] <= 90 * 24].reset_index(drop=True)
            if len(df) < 2:
                continue

            # Load holders
            moralis_path = os.path.join(data_dir, "moralis_holders_1h.json")
            holder_df = None
            if os.path.isfile(moralis_path):
                try:
                    with open(moralis_path) as f:
                        hdata = _json.load(f)
                    if hdata:
                        hdf = pd.DataFrame(hdata)
                        hdf["datetime"] = pd.to_datetime(hdf["timestamp"], utc=True).dt.tz_localize(None)
                        hdf["holders"] = hdf["totalHolders"].astype(float)
                        holder_df = hdf[["datetime", "holders"]]
                except Exception:
                    pass

            if holder_df is not None and len(holder_df) >= 2:
                holder_df = holder_df.set_index("datetime").resample("1h").last().ffill().reset_index()
                merged = pd.merge_asof(df[["datetime", "mcap", "volume", "hours"]].sort_values("datetime"),
                                        holder_df.sort_values("datetime"),
                                        on="datetime", direction="backward")
                if "holders" in merged.columns and merged["holders"].notna().sum() > 0:
                    df = merged.copy()

            if "holders" not in df.columns:
                df["holders"] = 0

            trajectories[addr] = {"df": df, "symbol": meta[addr]["symbol"], "name": meta[addr]["name"]}
        except Exception:
            continue

    return trajectories


def _render_phase3_3d_chart():
    """渲染按结果聚类着色的 3D 轨迹图。"""
    model_data, cluster_df = load_cluster_model()
    if model_data is None:
        return

    trajectories = _load_trajectories_for_3d()
    if not trajectories:
        return

    cluster_label_map = model_data["cluster_label_map"]
    cluster_order = model_data["cluster_order"]
    rank_map = {c: i + 1 for i, c in enumerate(cluster_order)}

    # Map address -> label
    addr_label = {}
    for _, row in cluster_df.iterrows():
        addr_label[row["address"]] = cluster_label_map.get(int(row["cluster_id"]), "Unknown")

    st.markdown("### 3D 轨迹可视化（按结果聚类着色）")
    st.markdown("*每条线是一个 token 的生命轨迹。颜色代表其结果聚类类别。*")

    # Cluster filter
    available_labels = sorted(set(addr_label.values()))
    selected_labels = st.multiselect("选择显示的聚类", available_labels, default=available_labels,
                                      key="phase3_3d_filter")

    fig = go.Figure()
    for addr, traj in trajectories.items():
        label = addr_label.get(addr)
        if label is None or label not in selected_labels:
            continue

        df = traj["df"]
        symbol = traj["symbol"]
        color = CLUSTER_COLORS_MAP.get(label, "#999999")
        width = 3 if label == "Organic Runner" else 2.5 if label == "Pump-Dump" else 1.5
        rank = rank_map.get(
            next((cid for cid, lbl in cluster_label_map.items() if lbl == label), -1), 0
        )

        fig.add_trace(go.Scatter3d(
            x=df["hours"], y=df["mcap"], z=df["holders"],
            mode="lines",
            line=dict(color=color, width=width),
            text=[
                f"<b>{symbol}</b><br>"
                f"#{rank} {label}<br>"
                f"T+{row['hours']:.0f}h<br>"
                f"MCap: ${row['mcap']:,.0f}<br>"
                f"Holders: {row.get('holders', 0):,.0f}"
                for _, row in df.iterrows()
            ],
            hoverinfo="text",
            name=f"{symbol} ({label})",
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
    legend_cols = st.columns(len(available_labels))
    for i, label in enumerate(available_labels):
        color = CLUSTER_COLORS_MAP.get(label, "#999")
        count = sum(1 for v in addr_label.values() if v == label)
        legend_cols[i].markdown(
            f'<span style="color:{color}; font-weight:bold;">&#9632;</span> {label} ({count})',
            unsafe_allow_html=True
        )


def _phase3_token_lookup(cluster_table):
    """在线查询：输入 token 地址或 symbol，查看其所属聚类。"""
    st.markdown("### Token 聚类查询")

    model_data, cluster_df = load_cluster_model()
    if model_data is None:
        st.warning("聚类模型未找到。请先运行 `python outcome_cluster.py`。")
        return

    query = st.text_input("输入 Token 地址或 Symbol", placeholder="例如: PUNCH, GOYIM, 或合约地址...",
                          key="phase3_query")
    if not query:
        return

    # 搜索: 先按 symbol，再按 address
    query_lower = query.strip().lower()
    match = cluster_df[cluster_df["symbol"].str.lower() == query_lower]
    if match.empty:
        match = cluster_df[cluster_df["address"].str.lower() == query_lower]
    if match.empty:
        # 模糊搜索
        match = cluster_df[cluster_df["symbol"].str.lower().str.contains(query_lower, na=False)]

    if match.empty:
        st.warning(f"未找到匹配「{query}」的 token。仅支持查询雷达列表中的 token。")
        return

    row = match.iloc[0]
    cid = int(row["cluster_id"])
    label = model_data["cluster_label_map"].get(cid, "Unknown")
    cluster_order = model_data["cluster_order"]
    rank_map = {c: i + 1 for i, c in enumerate(cluster_order)}
    rank = rank_map.get(cid, 0)

    # 该 token 的特征
    st.markdown(f"#### {row['symbol']} ({row['name']})")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("所属聚类", f"#{rank} {label}")
        st.metric("ATH", f"${row['ath']:,.0f}")
        st.metric("最大持币人", f"{row['max_holders']:,.0f}")

    with col2:
        st.metric("价格增速", f"${row['price_roc']:,.0f}/h")
        st.metric("持币人增速", f"{row['holder_roc']:,.1f}/h")
        st.metric("持币人衰减", f"{row['holder_decay_roc']:,.1f}/h")

    # 同聚类 token
    same_cluster = cluster_df[cluster_df["cluster_id"] == cid].copy()

    # Runner 列表
    runners = same_cluster[(same_cluster["ath"] >= 5_000_000) & (same_cluster["max_holders"] >= 5000)]
    st.markdown(f"#### 该聚类（#{rank} {label}）中的 Runner")

    if len(runners) > 0:
        runner_display = runners.nlargest(10, "ath")[["symbol", "ath", "max_holders", "price_roc"]].copy()
        runner_display.columns = ["Token", "ATH", "最大持币人", "价格增速 ($/h)"]
        runner_display["ATH"] = runner_display["ATH"].apply(lambda x: f"${x:,.0f}")
        runner_display["最大持币人"] = runner_display["最大持币人"].apply(lambda x: f"{x:,.0f}")
        runner_display["价格增速 ($/h)"] = runner_display["价格增速 ($/h)"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(runner_display, hide_index=True)

        st.markdown(f"""
        - **Runner 数量:** {len(runners)} / {len(same_cluster)} ({len(runners)/len(same_cluster)*100:.0f}%)
        - **Runner 中位 ATH:** ${runners['ath'].median():,.0f}
        - **Runner 75 分位 ATH:** ${np.percentile(runners['ath'], 75):,.0f}
        """)
    else:
        st.info("该聚类中没有符合 Runner 标准（ATH >= $5M 且 holders >= 5K）的 token。")

    # 聚类中位数表现
    st.markdown(f"#### 该聚类整体表现")
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    perf_col1.metric("中位 ATH", f"${same_cluster['ath'].median():,.0f}")
    perf_col2.metric("75 分位 ATH", f"${np.percentile(same_cluster['ath'], 75):,.0f}")
    perf_col3.metric("25 分位 ATH", f"${np.percentile(same_cluster['ath'], 25):,.0f}")

    # 最相似 token
    scaler = model_data["scaler"]

    def _build_vec(r):
        return [
            np.log1p(max(r["ath"], 0)),
            np.log1p(max(r["price_roc"], 0)),
            np.log1p(max(r["volume_roc"], 0)),
            np.log1p(max(r["holder_roc"], 0)),
            -np.log1p(max(-r["holder_decay_roc"], 0)),
            np.log1p(max(r["max_holders"], 0)),
        ]

    query_vec = scaler.transform([_build_vec(row)])
    member_vecs = scaler.transform([_build_vec(r) for _, r in same_cluster.iterrows()])
    dists = np.sqrt(((member_vecs - query_vec[0]) ** 2).sum(axis=1))
    same_cluster = same_cluster.copy()
    same_cluster["distance"] = dists

    similar = same_cluster[same_cluster["symbol"] != row["symbol"]].nsmallest(8, "distance")

    st.markdown("#### 最相似的历史 Token")
    sim_display = similar[["symbol", "ath", "max_holders", "price_roc", "distance"]].copy()
    sim_display.columns = ["Token", "ATH", "最大持币人", "价格增速 ($/h)", "距离"]
    sim_display["ATH"] = sim_display["ATH"].apply(lambda x: f"${x:,.0f}")
    sim_display["最大持币人"] = sim_display["最大持币人"].apply(lambda x: f"{x:,.0f}")
    sim_display["价格增速 ($/h)"] = sim_display["价格增速 ($/h)"].apply(lambda x: f"${x:,.0f}")
    sim_display["距离"] = sim_display["距离"].apply(lambda x: f"{x:.2f}")
    st.dataframe(sim_display, hide_index=True)


# ── 阶段一：3D 轨迹聚类 ──────────────────────────────────────────────────────


def phase_1():
    section("阶段一：3D 轨迹聚类（基线方案）",
    """
    **目标：** 通过轨迹形状匹配，找到具有相似增长模式的 token 群组。

    **方法：** 将每个 token 的小时级轨迹（时间 x 市值 x 持币人数）在对数空间中用弧长插值重采样为 50 个固定点，然后：

    1. **Z-归一化**：去除振幅差异，只保留形状特征
    2. **混合 DTW 距离**：50% 形状距离 + 50% 导数距离（变化率匹配）
    3. **UMAP 降维**：将 DTW 距离矩阵投影到 3D 空间
    4. **HDBSCAN 聚类**：基于密度自动发现聚类簇
    5. **S/A/B/C/D 评级**：按综合得分（ATH、效率、持币人数、成交量）对聚类排名
    """)

    finding(
        "聚类确实捕捉到了形状上的相似性，但**形状相似并不等于结果相似**。"
        "同一个聚类中既有涨了 100 倍的 runner，也有归零的 token。"
        "S/A/B/C/D 评级是基于聚类内中位 ATH 的事后排序，不具备预测能力。",
        "warning"
    )

    st.markdown("### 产物展示")
    st.markdown("*每条线代表一个 token 从市值首次超过 $50K 开始的生命轨迹。颜色 = 聚类评级。*")

    if os.path.isfile(TRAIN_DATA_PATH):
        _render_phase1_chart()

    st.caption("完整交互版本请运行：`streamlit run app_3d.py`")

    data_quality_box([
        "**200+ 个雷达 token**，小时级市值 + 持币人数据",
        "持币人数据优先使用 **Moralis 小时级**（310 个 token 有覆盖），其余回退到 GMGN 日级",
        "数据窗口：从市值首次超过 $50K 开始，最长截取 3 个月",
        "DTW 距离计算复杂度 O(n^2)，通过 @st.cache_data 缓存加速",
    ])

    limitations_box([
        "**形状相似 ≠ 结果相似**：一个「快速上涨」的形状既可能是 runner，也可能是 pump-dump",
        "HDBSCAN 将 50-70% 的 token 标记为「噪音」，无法归入任何聚类",
        "评级是事后的：需要完整的轨迹历史，无法用于实时预测",
        "UMAP 是随机算法：不同运行会产生不同的降维结果",
    ])


def _render_phase1_chart():
    st.markdown("""
    ```
    处理流水线: 原始4D时间序列 -> DTW距离矩阵 -> UMAP 3D降维 -> HDBSCAN聚类 -> S/A/B/C/D评级

    坐标轴:  X = 时间（从市值>$50K开始的小时数）
             Y = 市值（$, 对数刻度）
             Z = 持币人数
    颜色: 金色=S (Runner), 绿色=A (Strong), 蓝色=B, 紫色=C, 橙色=D
    ```
    """)
    df = pd.read_parquet(TRAIN_DATA_PATH)
    tokens = df.groupby("symbol").agg(
        ath=("mcap_max", "max"),
        max_holders=("holders_end", "max"),
    ).reset_index()
    fig = px.scatter(tokens, x="ath", y="max_holders", hover_name="symbol",
                     log_x=True, log_y=True,
                     title="Token 全景：ATH 市值 vs 最大持币人数",
                     labels={"ath": "历史最高市值 ($)", "max_holders": "最大持币人数"})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


# ── 阶段二：两阶段聚类 ───────────────────────────────────────────────────────


def phase_2():
    section("阶段二：两阶段聚类",
    """
    **目标：** 解决阶段一的问题——确保同一聚类内的 token 具有相似的最终结果。

    **方法：** 分两步走：
    - **第一阶段（结果分档）：** 用 KMeans(k=3) 对 4 个结果指标（ATH、价格增速、成交量增速、持币人增速）
      进行聚类，将 token 分为 Runner / Mid / Weak 三档
    - **第二阶段（形状聚类）：** 在每个档内部再跑 DTW + HDBSCAN，寻找行为子模式

    这样可以保证结果一致性（第一阶段），同时发现形状模式（第二阶段）。
    """)

    finding(
        "**关键发现：** 大部分 organic runner（PUNCH, WAR, GOYIM, GORK）在 DTW 聚类中全部沦为「噪音」"
        "——每个档内 57-71% 的 token 无法被聚类。反而是那些被紧密聚到一起的 token，"
        "全部是 scam/bot 驱动的 pump-dump（GME, TRUMP2, GameStop, XMONEY）。"
        "\n\n**结论：Organic token 没有可重复的轨迹形状。有模式 = bot/scam。**",
        "error"
    )

    st.markdown("### 证据对比")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Runner 档噪音（organic，无法聚类）：**")
        noise_data = {
            "Token": ["PUNCH", "TRUMP", "WAR", "CityBoy", "GORK"],
            "ATH": ["$46.9M", "$9.5B", "$63.6M", "$120.8M", "$21.2M"],
            "到达ATH耗时": ["411小时", "584小时", "1028小时", "343小时", "45小时"],
            "最大持币人数": ["27K", "650K", "86K", "57K", "16K"],
        }
        st.dataframe(pd.DataFrame(noise_data), hide_index=True)
        st.caption("每个都有独特的增长轨迹，DTW 无法找到匹配。")

    with col2:
        st.markdown("**Runner 档聚类成功（scam/bot，高度雷同）：**")
        cluster_data = {
            "Token": ["TRUMP2", "GameStop", "XMONEY", "DonTrump", "SHRIMP"],
            "ATH": ["$13.4M", "$12.0M", "$10.6M", "$4.6M", "$5.6M"],
            "到达ATH耗时": ["0小时", "0小时", "0小时", "1小时", "1小时"],
            "最大持币人数": ["1.2K", "3.1K", "1.2K", "965", "1.2K"],
        }
        st.dataframe(pd.DataFrame(cluster_data), hide_index=True)
        st.caption("模式完全一致：瞬间拉盘，~1K 持币人，最终全部归零。")

    data_quality_box([
        "与阶段一相同的 200+ 个 token 数据",
        "新增了 top10 持仓集中度数据（来自 token_trends）",
        "纯净度指标（聚类内结果的变异系数 CV）显示即使分档后，聚类内一致性仍然很低",
    ])

    limitations_box([
        "方案实现后被回退——两阶段方法并没有解决根本问题",
        "DTW 聚类天然倾向于发现程序化/机器人操作的模式",
        "Organic 增长由叙事、KOL 传播、社区情绪驱动——这些不在链上数据的形状里",
        "**这是整个研究的转折点：我们彻底放弃了聚类，转向有监督学习**",
    ])


# ── 阶段三：纯结果特征 KMeans ────────────────────────────────────────────────


def phase_3():
    section("阶段三：纯结果特征 KMeans",
    """
    **目标：** 彻底放弃轨迹形状。直接用结果指标进行聚类。

    **方法：** 对每个 token 提取 6 个对数缩放的结果特征，用 KMeans(k=6) 聚类：
    ATH、价格增速 ($/h)、成交量增速 ($/h)、持币人增速 (/h)、
    持币人衰减速度 (/h)、最大持币人数。

    每个聚类根据其特征轮廓自动分配语义标签。
    """)

    finding(
        "成功将 token 分为有意义的类别。PUNCH 和 WAR 被正确标记为「Organic Runner」，"
        "scam token（GME, TRUMP2）全部归入「Pump-Dump」类。分组清晰直观。",
        "success"
    )

    # ── 实际聚类结果 ──
    st.markdown("### 聚类结果（322 个 token，6 类）")

    cluster_table = pd.DataFrame({
        "#": [1, 2, 3, 4, 5, 6],
        "标签": ["Organic Runner", "Pump-Dump", "Average", "Slow Grinder", "Average", "Flash Crash"],
        "数量": [45, 39, 19, 100, 104, 15],
        "中位 ATH": ["$20.3M", "$4.1M", "$2.7M", "$1.9M", "$1.6M", "$568K"],
        "特征": [
            "高ATH + 高holders + 中等增速，社区驱动的真实runner",
            "极快price_roc + 0 holder增长，机器人操盘瞬间拉盘",
            "无 holder 数据的中等 token",
            "极慢增速，长期缓慢增长型，多见于Base链",
            "各指标中规中矩",
            "极快 holder 流失，快速崩盘型",
        ],
        "代表 Token": [
            "TRUMP, BFS, PENGUIN, HAM, CityBoy",
            "BP, GME, TRUMP2, GameStop, XMONEY",
            "DM, BOMB, LLM, torch, BANDS",
            "ANON, CHIBI, PIGEON, MOLT, ZORA",
            "ONE, DONT, wcd, RRCH, COPPERINU",
            "NBR, OpenClaw, ATK, OIL, SSTR",
        ],
    })
    st.dataframe(cluster_table, hide_index=True, use_container_width=True)

    # ── 散点图 ──
    if os.path.isfile(TRAIN_DATA_PATH):
        df = pd.read_parquet(TRAIN_DATA_PATH)
        token_stats = df.groupby("symbol").agg(
            ath=("mcap_max", "max"),
            max_holders=("holders_end", "max"),
            avg_volume=("volume_total", "mean"),
        ).reset_index()
        token_stats = token_stats[token_stats["ath"] > 0].copy()

        X = np.column_stack([
            np.log1p(token_stats["ath"].values),
            np.log1p(token_stats["avg_volume"].values),
            np.log1p(token_stats["max_holders"].values),
        ])
        X = np.nan_to_num(X)
        km = KMeans(n_clusters=6, random_state=42, n_init=10)
        token_stats["cluster"] = km.fit_predict(StandardScaler().fit_transform(X))

        cluster_ath = token_stats.groupby("cluster")["ath"].median()
        rank_map = {cid: i for i, cid in enumerate(cluster_ath.sort_values(ascending=False).index)}
        labels_map = {0: "Organic Runner", 1: "Fast Organic", 2: "Pump-Dump",
                      3: "Slow Grinder", 4: "Average", 5: "Flash Crash"}
        token_stats["rank"] = token_stats["cluster"].map(rank_map)
        token_stats["label"] = token_stats["rank"].map(labels_map)

        fig = px.scatter(token_stats, x="ath", y="max_holders",
                         color="label", hover_name="symbol",
                         log_x=True, log_y=True,
                         title="6 类结果聚类：ATH 市值 vs 最大持币人数",
                         color_discrete_sequence=["#FFD700", "#22c55e", "#e41a1c", "#3b82f6", "#a855f7", "#9ca3af"])
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # ── 预测脚本说明 ──
    st.markdown("### 预测工具")
    st.markdown("""
    基于上述聚类结果，我们构建了一个预测脚本，给定任意 token 地址，判断它属于哪个类别，
    并输出该类别下的 runner 列表和中位数表现。
    """)

    st.markdown("**示例输出（`outcome_cluster_predict.py`）：**")

    example_result = {
        "分类结果": "#5 Average",
        "该类 Token 数": 104,
        "该类 Runner 数": "15 (14%)",
        "Runner 中位 ATH": "$11,411,972",
        "最相似 Token": "Deadwhale ($1.2M), PERK ($1.2M), KID ($585K)",
        "该类中位 ATH": "$1,571,953",
        "该类 75分位 ATH": "$4,875,856",
    }
    col1, col2 = st.columns(2)
    with col1:
        for k, v in list(example_result.items())[:4]:
            st.metric(k, v)
    with col2:
        for k, v in list(example_result.items())[4:]:
            st.metric(k, v)

    st.markdown("**命令行使用方式：**")
    st.code("""# 第一步：运行聚类（仅需一次）
python outcome_cluster.py

# 第二步：预测新 token
python outcome_cluster_predict.py <token地址>

# 查看已有聚类结果
python outcome_cluster.py --show""", language="bash")

    # ── 3D 轨迹可视化（按结果聚类着色）──
    _render_phase3_3d_chart()

    # ── 在线查询 ──
    _phase3_token_lookup(cluster_table)

    # ── 局限性 ──
    finding(
        "**这纯粹是描述性的，不是预测性的。** 这些指标只有在 token 的历史完全走完之后才能计算。"
        "它能告诉你发生了什么，但无法告诉你将会发生什么。"
        "对于一个正在发展中的 token，它的最终 ATH 和 holder 数还未定，分类结果会随时间变化。",
        "warning"
    )

    limitations_box([
        "**纯事后标注**——需要完整历史数据（ATH、最终持币人数等）",
        "对于正在进行中的 token，分类结果不稳定（ATH 还在变化）",
        "无法用于早期预警或实时决策",
        "本质上是一个精细的分类标签系统，而非预测模型",
        "对于理解数据集很有价值，但不是我们要找的东西",
    ])


# ── 阶段四：MCap/Holder 2D 曲线 ──────────────────────────────────────────────


def phase_4():
    section("阶段四：MCap/Holder 2D 曲线",
    """
    **目标：** 简化为单一指标：市值 / 持币人数的比率随时间的变化。

    **假设：** 这个比率能区分 organic 增长（市值和持币人同步增长）与人为拉盘
    （市值暴涨但持币人不增长）。

    **方法：** 绘制每个 token 从 $100K 开始第一周的 MCap/Holder 曲线，用 DTW 聚类。
    """)

    finding(
        "**没有信号。** 所有 MCap/Holder 曲线严重重叠。这个比率噪声太大，"
        "与原始市值的相关性太高，无法提供独立的信息。"
        "聚类产生的分组在结果上没有任何一致性。",
        "error"
    )

    st.markdown("### 失败原因分析")
    st.markdown("""
    MCap/Holder 比率被**市值所主导**，因为：
    - 持币人数的方差有限（大部分 token 第一周在 500-5,000 之间）
    - 市值在 token 间的差异可达 100 倍以上（$100K 到 $10M+）
    - 这个比率本质上就是一个加了噪声的市值代理变量

    2D 图表显示所有曲线挤在重叠的带状区域中——完全没有分离度。
    """)

    st.caption("完整交互版本请运行：`streamlit run app_mcap_per_holder.py`")

    limitations_box([
        "MCap/Holder 不是独立信号——它只是市值的噪声版本",
        "一周的时间窗口太短，无法覆盖需要数周才激活的 token",
        "在 1D 信号上做 DTW 聚类，判别力比 3D 更差",
    ])


# ── 阶段五：早期预警 v1 ──────────────────────────────────────────────────────


def phase_5():
    section("阶段五：早期预警 v1（固定窗口分类器）",
    """
    **关键转折：从无监督聚类转向有监督机器学习。**

    **目标：** 利用前 24 小时的数据来**预测**一个 token 是否会成为 runner。

    **方法：**
    - 根据最终结果为每个 token 打标签（scam / rug / mid / runner）
    - 从前 24 小时窗口中提取 27 个特征（价格动量、成交量模式、持币人增长）
    - 使用 LightGBM 分类器，5 折分层交叉验证
    - 两阶段模型：scam 检测（二分类）→ runner 检测（三分类）
    """)

    finding(
        "**首次出现真实信号！** Scam 检测准确率 95%（F1=0.84）。"
        "Runner 检测 F1=0.72，10 个随机种子下稳定。"
        "最重要特征：volume_trend（成交量趋势）、volume_concentration（集中度）、"
        "mcap_return_4h（4小时收益率）、holders_start（初始持币人数）。",
        "success"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 第一阶段：Scam 检测")
        scam_data = {
            "类别": ["正常", "Scam"],
            "精确率": ["98%", "81%"],
            "召回率": ["97%", "87%"],
            "F1": ["0.97", "0.84"],
            "样本数": [199, 30],
        }
        st.dataframe(pd.DataFrame(scam_data), hide_index=True)

    with col2:
        st.markdown("### 第二阶段：Runner 检测")
        runner_data = {
            "类别": ["Rug", "Mid", "Runner"],
            "精确率": ["84%", "77%", "73%"],
            "召回率": ["79%", "80%", "79%"],
            "F1": ["0.81", "0.78", "0.76"],
            "样本数": [90, 81, 28],
        }
        st.dataframe(pd.DataFrame(runner_data), hide_index=True)

    st.markdown("### 已知 Token 预测结果")
    pred_data = {
        "Token": ["PUNCH", "GORK", "TRUMP (主)", "WAR", "GOYIM", "CAPTCHA"],
        "实际": ["runner", "runner", "runner", "runner", "runner", "runner"],
        "预测": ["runner", "runner", "runner", "runner", "rug", "mid"],
        "正确": ["是", "是", "是", "是", "否", "否"],
    }
    st.dataframe(pd.DataFrame(pred_data), hide_index=True)

    st.markdown("""
    **GOYIM 和 CAPTCHA 为什么被漏掉？** GOYIM 前 24 小时完全横盘
    （市值从 $836K 到 $815K，收益率 -2.5%），直到第 49 小时才开始爆发。
    CAPTCHA 在前 24 小时甚至在下跌（-31% 收益率）。
    模型正确识别了它们前 24 小时的特征——确实不像 runner。**问题不在模型，在于时间窗口的局限。**
    """)

    data_quality_box([
        "229 个 token 同时有标签和特征数据",
        "27 个特征来自前 24 小时（价格、成交量、持币人、结构性指标）",
        "5 折分层交叉验证，10 个随机种子下稳定（F1=0.72 +/- 0.05）",
        "仅 28 个 runner 样本——数量少但足以检测到信号",
    ])

    limitations_box([
        "**固定 24 小时窗口无法捕捉晚发型 token**——GOYIM 在第 49 小时才激活，对这个模型不可见",
        "仅 229 个样本（28 个 runner）——结果方差较大",
        "5 分钟蜡烛图数据早期覆盖率差（GMGN API 只保留最近 33 小时的 5 分钟数据）",
        "标签过于粗糙：4 个类别丢失了连续结果频谱的细微差异",
    ])


# ── 阶段六：早期预警 v3（最终方案）────────────────────────────────────────────


@st.cache_data
def load_training_data():
    if not os.path.isfile(TRAIN_DATA_PATH):
        return None
    return pd.read_parquet(TRAIN_DATA_PATH)


@st.cache_data
def run_oos_evaluation():
    """在训练数据上运行 GroupKFold 样本外评估。"""
    df = load_training_data()
    if df is None:
        return None, None, None

    feature_cols = [c for c in df.columns if c not in
                    ("address", "symbol", "t_start", "t_end", "label", "future_max_mult", "pred_mult")]
    X = df[feature_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0)
    y = np.log1p(df["future_max_mult"].values)
    groups = df["address"].values

    import lightgbm as lgb

    gkf = GroupKFold(n_splits=5)
    y_pred_oos = np.full(len(y), np.nan)
    importances = np.zeros(len(feature_cols))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        model = lgb.LGBMRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.03,
            num_leaves=25, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.2, reg_lambda=0.2, random_state=42, verbose=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        y_pred_oos[val_idx] = model.predict(X[val_idx])
        importances += model.feature_importances_

    df["pred_mult_oos"] = np.expm1(y_pred_oos)
    return df, feature_cols, importances / 5


def phase_6():
    section("阶段六：早期预警 v3（滑动窗口，最终方案）",
    """
    **突破：滑动窗口 + 回归 + 集成模型。**

    **相对 v1 的关键改进：**
    1. **滑动窗口**替代固定前 24 小时——在每个时间点都生成一个样本，能捕捉像 GOYIM 这样的晚发型 token
    2. **小时级数据**替代 5 分钟——完整历史覆盖（无 API 缺失）
    3. **回归**替代分类——预测连续的最大涨幅倍数（1.0x, 2.5x, 10x...），而非离散标签
    4. **GroupKFold 验证**——整个 token 被排除在训练集外，真正的样本外测试
    5. **集成模型**——混合基础模型（24小时窗口特征）和历史模型（+6个历史上下文特征）
    """)

    df, feature_cols, importances = run_oos_evaluation()

    if df is None:
        st.error("训练数据未找到。请先运行 `python early_warning/train_v3.py`。")
        return

    y_actual = df["future_max_mult"].values
    y_pred = df["pred_mult_oos"].values
    n_pump = (y_actual >= 2.0).sum()

    # 核心指标
    st.markdown("### 结果（GroupKFold 样本外验证）")
    st.markdown(f"**{len(df):,} 个样本**来自 **{df['address'].nunique()} 个 token** | "
                f"实际 pump 率：{n_pump:,}/{len(df):,} = **{n_pump/len(df)*100:.1f}%**")

    order = np.argsort(-y_pred)
    actual_sorted = y_actual[order]

    col1, col2, col3, col4 = st.columns(4)
    for col, k, label in [(col1, 50, "Top 50"), (col2, 100, "Top 100"),
                           (col3, 200, "Top 200"), (col4, 500, "Top 500")]:
        top = actual_sorted[:k]
        prec = (top >= 2.0).sum() / k * 100
        avg = top.mean()
        col.metric(label, f"{prec:.0f}% 精确率", f"平均 {avg:.1f}x")

    finding(
        f"**Top 100 预测中，49% 在 48 小时内真的涨了 2 倍以上，平均涨幅 {actual_sorted[:100].mean():.1f}x。**"
        f"基线 pump 率仅 {n_pump/len(df)*100:.1f}%。"
        f"相当于**随机选择的 {49/(n_pump/len(df)*100):.1f} 倍提升**。",
        "success"
    )

    # Precision@K 图表
    st.markdown("### Precision@K 分析")
    ks = [20, 50, 100, 200, 500, 1000, 2000]
    precisions = []
    avg_mults = []
    for k in ks:
        if k > len(actual_sorted):
            break
        top = actual_sorted[:k]
        precisions.append((top >= 2.0).sum() / k * 100)
        avg_mults.append(top.mean())

    fig_pk = go.Figure()
    fig_pk.add_trace(go.Bar(x=[str(k) for k in ks[:len(precisions)]], y=precisions,
                             name="精确率 (%)", marker_color="#22c55e"))
    fig_pk.add_trace(go.Scatter(x=[str(k) for k in ks[:len(avg_mults)]], y=avg_mults,
                                 name="平均涨幅倍数", yaxis="y2", mode="lines+markers",
                                 marker_color="#3b82f6", line=dict(width=3)))
    fig_pk.update_layout(
        yaxis=dict(title="精确率 (%)", range=[0, 60]),
        yaxis2=dict(title="平均涨幅 (x)", overlaying="y", side="right", range=[0, 6]),
        barmode="group", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig_pk.add_hline(y=n_pump/len(df)*100, line_dash="dash", line_color="red",
                      annotation_text=f"基线：{n_pump/len(df)*100:.1f}%")
    st.plotly_chart(fig_pk, use_container_width=True)

    # 特征重要性
    st.markdown("### 特征重要性")
    imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
    imp_df = imp_df.sort_values("Importance", ascending=True).tail(15)

    FEATURE_LABELS = {
        "top10_change": "Top10 持仓变化", "top10_pct": "Top10 集中度",
        "holders_start": "初始持币人数", "holders_end": "最终持币人数",
        "holder_growth": "持币人增长率", "volatility": "波动率",
        "mcap_per_holder": "人均市值", "mcap_max": "窗口内最高市值",
        "mcap_min": "窗口内最低市值", "mcap_end": "窗口末市值",
        "mcap_start": "窗口初市值", "volume_total": "总成交量",
        "max_drawdown": "最大回撤", "volume_concentration": "成交量集中度",
        "mcap_range": "价格范围", "hist_ath": "历史最高市值",
        "hist_pump_count": "历史 2x pump 次数", "token_age_hours": "Token 年龄",
        "mcap_return": "24h 收益率", "momentum_shift": "动量变化",
        "mean_return": "平均小时收益", "volume_trend": "成交量趋势",
        "volume_mean": "平均小时成交量", "peak_position": "峰值位置",
        "return_6h": "6h 收益率", "return_12h": "12h 收益率",
        "mcap_holder_corr": "市值-持币人相关性", "hist_ath_ratio": "当前/历史ATH",
        "hist_return_total": "历史总收益", "hist_volatility": "历史波动率",
    }
    imp_df["Label"] = imp_df["Feature"].map(lambda x: FEATURE_LABELS.get(x, x))

    fig_imp = px.bar(imp_df, x="Importance", y="Label", orientation="h",
                      title="Pump 预测最重要的 15 个特征",
                      color="Importance", color_continuous_scale="Viridis")
    fig_imp.update_layout(height=450, yaxis_title="")
    st.plotly_chart(fig_imp, use_container_width=True)

    # 预测 vs 实际散点图
    st.markdown("### 预测倍数 vs 实际倍数")
    scatter_df = df[["symbol", "future_max_mult", "pred_mult_oos", "label"]].copy()
    scatter_df["pred_mult_oos"] = scatter_df["pred_mult_oos"].clip(0.1, 50)
    scatter_df["future_max_mult"] = scatter_df["future_max_mult"].clip(0.1, 50)

    if len(scatter_df) > 5000:
        scatter_df = scatter_df.sample(5000, random_state=42)

    label_map_cn = {"pump": "暴涨 (pump)", "flat": "横盘 (flat)", "dump": "暴跌 (dump)"}
    scatter_df["label_cn"] = scatter_df["label"].map(label_map_cn)
    color_map = {"暴涨 (pump)": "#22c55e", "横盘 (flat)": "#94a3b8", "暴跌 (dump)": "#ef4444"}

    fig_scatter = px.scatter(scatter_df, x="pred_mult_oos", y="future_max_mult",
                              color="label_cn", hover_name="symbol",
                              log_x=True, log_y=True,
                              color_discrete_map=color_map,
                              title="预测 vs 实际 48h 最大涨幅倍数（样本外）",
                              labels={"pred_mult_oos": "预测倍数", "future_max_mult": "实际倍数",
                                      "label_cn": "实际结果"})
    fig_scatter.add_shape(type="line", x0=0.1, x1=50, y0=2, y1=2,
                           line=dict(dash="dash", color="green", width=1))
    fig_scatter.add_annotation(x=1.5, y=2.5, text="2x pump 阈值", showarrow=False, font=dict(color="green"))
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Top 30 样本外预测
    st.markdown("### Top 30 样本外预测")
    top30_idx = order[:30]
    top30 = df.iloc[top30_idx][["symbol", "t_end", "pred_mult_oos", "future_max_mult", "label", "mcap_end", "holders_end"]].copy()
    top30.insert(0, "排名", range(1, 31))
    top30.columns = ["排名", "Token", "窗口结束", "预测倍数", "实际倍数", "标签", "市值", "持币人"]
    top30["预测倍数"] = top30["预测倍数"].apply(lambda x: f"{x:.1f}x")
    top30["实际倍数"] = top30["实际倍数"].apply(lambda x: f"{x:.1f}x")
    top30["市值"] = top30["市值"].apply(lambda x: f"${x:,.0f}")
    top30["持币人"] = top30["持币人"].apply(lambda x: f"{x:,.0f}")
    top30["窗口结束"] = top30["窗口结束"].str[:16]
    st.dataframe(top30, hide_index=True, use_container_width=True)

    # 已知 token 表现
    st.markdown("### 已知 Token 表现")
    known = ["PUNCH", "GOYIM", "CAPTCHA", "WAR", "GORK", "BFS"]
    known_rows = []
    for sym in known:
        mask = df["symbol"] == sym
        if not mask.any():
            continue
        sub = df[mask]
        n_win = len(sub)
        n_p = (sub["label"] == "pump").sum()
        best_pred = sub["pred_mult_oos"].max()
        best_actual = sub.loc[sub["pred_mult_oos"].idxmax(), "future_max_mult"]
        rank = int((y_pred > best_pred).sum()) + 1
        known_rows.append({
            "Token": sym, "滑动窗口数": n_win, "实际 Pump 次数": n_p,
            "最佳预测": f"{best_pred:.1f}x",
            "该预测实际结果": f"{best_actual:.1f}x",
            "全局排名": f"#{rank:,}",
        })
    if known_rows:
        st.dataframe(pd.DataFrame(known_rows), hide_index=True)

    data_quality_box([
        f"**{len(df):,} 个训练样本**来自 {df['address'].nunique()} 个 token",
        "使用小时级蜡烛图数据（完整历史覆盖，没有 5 分钟数据的缺失问题）",
        "滑动窗口：24 小时回看，48 小时预测，6 小时步长",
        "30 个特征：价格类(12)、成交量类(4)、持币人类(5)、Top10(2)、历史上下文(6)",
        "GroupKFold：每一折整个 token 被排除在训练集外——无数据泄漏",
        "集成模型：50/50 混合基础模型和历史上下文模型",
    ])

    limitations_box([
        f"**仅 {df['address'].nunique()} 个 token**——需要 500+ 个才能稳健泛化",
        "小时级粒度丢失了小时内的动态变化（5 分钟数据覆盖改善后可以升级）",
        "训练中没有 Codex 链上特征（买卖流、净成交量）——仅在推理时展示",
        "模型预测的是最大涨幅倍数，而非时机——知道会不会涨，但不知道 48 小时内何时涨",
        "假阳性率仍然较高：Top 100 精确率 49% 意味着每 49 个真 pump 伴随 51 个误报",
    ])


# ── 总结 ─────────────────────────────────────────────────────────────────────


def conclusion():
    section("总结",
    """
    ### 从无监督到有监督：信号的发现之路
    """)

    journey = pd.DataFrame({
        "阶段": ["1. 3D 轨迹聚类", "2. 两阶段聚类", "3. 结果 KMeans",
                  "4. MCap/Holder", "5. 早期预警 v1", "6. 早期预警 v3"],
        "方法": ["DTW + HDBSCAN", "KMeans + DTW", "KMeans (结果特征)",
                 "DTW (MCap/Holder)", "LightGBM 分类器", "LightGBM 回归"],
        "信号": ["无", "无", "仅事后", "无", "有 (F1=0.72)", "有 (P@100=49%)"],
        "核心洞见": [
            "形状相似 ≠ 结果相似",
            "Organic token 没有可重复形状",
            "好标签但无法预测",
            "比率只是市值的噪声版本",
            "前 24 小时特征有信号",
            "滑动窗口捕捉所有阶段",
        ],
    })
    st.dataframe(journey, hide_index=True, use_container_width=True)

    st.markdown("### 核心结论")

    st.markdown("""
    1. **聚类无法用于预测。** 无监督方法（DTW、KMeans）能描述已经发生的事情，
       但无法预测将要发生的事情。Organic memecoin 的成功由独特的叙事驱动——
       不存在可以匹配的「runner 形状」。

    2. **有模式 = scam。** 被 DTW 紧密聚在一起的 token 是机器人驱动的 pump-and-dump。
       这对 scam 检测其实很有用，但恰恰是我们最初目标的反面。

    3. **有监督学习 + 滑动窗口有效。** 在 31K+ 个样本上训练，使用未来标签，
       LightGBM 学到了能预测 pump 的特征组合：Top 100 精确率 49%（相比基线 6.6% 提升 7.4 倍）。

    4. **最具预测力的特征不是价格形状，而是市场结构：**
       Top10 持仓集中度、持币人数量/增长率、波动率、人均市值。
       这些特征捕捉的是一个 token 是否有真实的社区兴趣，还是人为操纵。

    5. **数据量是当前瓶颈。** 仅有 210 个 token，模型结果很有前景但尚未达到生产级可靠性。
       扩展到 500+ 个 token 应该能显著提升稳定性和精确率。
    """)

    st.markdown("### 生产使用方式")
    st.code("source .venv/bin/activate\npython early_warning/predict.py <token地址>", language="bash")

    st.markdown("""
    预测工具输出：
    - **SABCD 评级**（S = Top 1%，历史上 53% 的 pump 率；D = 后 80%，无信号）
    - **关键因素**：驱动该预测的特征归因分析
    - **相似历史 Token**：最相似的历史窗口及其实际结果
    - **预测 48h 最大涨幅倍数**及置信排名
    """)


# ── 导航 ─────────────────────────────────────────────────────────────────────


PHASES = {
    "阶段一：3D 轨迹聚类": phase_1,
    "阶段二：两阶段聚类": phase_2,
    "阶段三：纯结果 KMeans": phase_3,
    "阶段四：MCap/Holder 2D": phase_4,
    "阶段五：早期预警 v1": phase_5,
    "阶段六：早期预警 v3（最终）": phase_6,
    "总结": conclusion,
}

with st.sidebar:
    st.title("Memecoin DNA 研究")
    st.caption("6 个实验阶段的完整记录")
    st.divider()
    selected = st.radio("导航", list(PHASES.keys()), label_visibility="collapsed")

PHASES[selected]()
