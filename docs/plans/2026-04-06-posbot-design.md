# Position Manager Bot (posbot) 设计方案

## 概述

一个手动入场、自动管理的持仓管理机器人。用户给定 token 地址和 entry price 后，bot 基于 CatBoost 模型自动做出持有/部分止盈/部分止损/全部退出的决策，通过 Jupiter swap 执行链上交易。

## 核心参数

- **持仓周期:** 几小时到 1-2 天
- **Tick 频率:** 5 分钟
- **执行:** Jupiter swap（先 paper，后 live）
- **模型:** CatBoost 7 类分类器

## 7 个决策动作

| 动作 | 含义 |
|---|---|
| HOLD | 继续持有，不操作 |
| TP_25 | 止盈卖出 25% |
| TP_50 | 止盈卖出 50% |
| TP_100 | 止盈全部卖出 |
| SL_25 | 止损减仓 25% |
| SL_50 | 止损减仓 50% |
| EXIT | 全部止损退出 |

## 文件结构

```
posbot/
  __main__.py    — CLI 入口: python -m posbot open <address> --size 100
  engine.py      — 主循环（5min tick + 数据采集 + 模型预测 + 执行）
  strategy.py    — CatBoost 模型加载和预测
  executor.py    — Jupiter swap 执行（paper / live 模式）
  monitor.py     — 数据采集（GMGN价格 + Codex量价 + Moralis holders）
  features.py    — 特征计算（VWAP、HMM、技术指标）
  db.py          — SQLite 持仓和交易记录
  
posbot_train/
  build_dataset.py  — 从历史数据构建训练集
  train.py          — CatBoost 训练 + GroupKFold 验证
  backtest.py       — 回测模拟器 + 量化评估
  label.py          — 标签生成逻辑（基于未来数据回标最优动作）
```

## 训练数据构建

### 数据来源

305 个雷达 token 的完整小时级历史（data/ 目录已有）。

### 样本生成

```
对每个 token:
  对每个可能的 entry point (mcap >= $100K):
    entry_price = mcap at entry
    
    对 entry 之后的每个 tick (每小时):
      计算特征向量 (25-30 维)
      计算标签 (基于未来 6h 数据回标最优动作)
      
      → 生成一个训练样本
```

预估样本量: 305 tokens × ~100 entries × ~50 ticks ≈ 百万级

### 标签生成逻辑

每个 tick 看未来 6 小时的 max 和 min:

```python
unrealized_pnl = (current - entry) / entry
future_max = max(future_6h_prices)
future_min = min(future_6h_prices)
future_return = (future_max - current) / current
future_drawdown = (future_min - current) / current

# 止盈类
if unrealized_pnl > 1.0 and future_return < 0.05:
    label = "TP_100"   # 已翻倍，未来不怎么涨了
elif unrealized_pnl > 0.5 and future_drawdown < -0.20:
    label = "TP_50"    # 涨50%+，未来会回撤>20%
elif unrealized_pnl > 0.2 and future_drawdown < -0.15:
    label = "TP_25"    # 涨20%+，未来会回撤>15%

# 止损类
elif unrealized_pnl < -0.20 and future_return < 0.10:
    label = "EXIT"     # 亏20%+，未来不会回升
elif unrealized_pnl < -0.10 and future_return < 0.05:
    label = "SL_50"    # 亏10%+，未来继续跌
elif unrealized_pnl < -0.05 and future_return < 0.03:
    label = "SL_25"    # 亏5%+，未来继续跌

else:
    label = "HOLD"
```

## 特征工程 (25-30 维)

### A. 持仓状态
- `unrealized_pnl` — 浮盈浮亏 %
- `holding_hours` — 持仓时长
- `sold_pct` — 已卖出比例 (0-100%)
- `distance_from_peak` — 当前价 vs 持仓最高价

### B. 量价特征
- `roc_1h / roc_4h / roc_12h` — 多窗口涨跌幅
- `volatility_1h / volatility_4h` — 波动率
- `volume_trend` — 成交量趋势
- `buy_sell_ratio` — 买卖压力比 (Codex)
- `volume_concentration` — 成交量集中度

### C. VWAP
- `vwap_24h` — 24h rolling VWAP
- `price_vs_vwap` — 当前价格 / VWAP
- `vwap_slope` — VWAP 斜率

### D. HMM Regime
- `hmm_state` — 当前 HMM 状态 (0=积蓄, 1=上涨, 2=下跌, 3=崩盘)
- `hmm_state_duration` — 当前状态持续时间
- `hmm_transition_prob` — 转移到下跌状态的概率

### E. 链上特征
- `holder_growth_1h / holder_growth_4h` — 持币人变化
- `mcap_per_holder` — 人均市值
- `top10_pct` — Top10 集中度
- `top10_change` — Top10 变化

### F. 模型信心
- `ew_predict_mult` — 早期预警预测倍数
- `ew_grade` — SABCD 评级 (编码 1-5)

## 模型训练

- **算法:** CatBoost 多分类 (7 类)
- **类别平衡:** class_weights="balanced"
- **验证:** GroupKFold (按 token), 5 折
- **超参:** 先用默认，后续可以 Optuna 调优

## 回测评估

### 模拟流程

```
对每个 fold 的测试 token:
  从 entry point 开始，初始仓位 100%
  每个 tick:
    模型预测 → 执行动作
    TP_25/50/100 → 卖出对应比例，记录已实现利润
    SL_25/50 → 减仓
    EXIT → 清仓
  直到 EXIT 或 48h 超时强制平仓
```

### 量化指标

| 指标 | 含义 |
|---|---|
| 总收益率 | 所有交易平均收益 |
| 胜率 | 盈利交易 / 总交易 |
| 盈亏比 | 平均盈利 / 平均亏损 |
| 最大回撤 | 单笔最大浮亏 |
| Sharpe Ratio | 收益 / 波动率 |
| vs Buy-and-Hold | 对比傻持 48h |
| vs 固定规则 | 对比 2x TP / -20% SL |
| 平均持仓时间 | entry 到完全退出 |
| TP/SL 触发分布 | 各动作触发次数和比例 |

### Baseline 对比

1. **Buy and Hold 48h** — 入场后持有 48 小时
2. **固定规则** — 2x 止盈，-20% 止损（现有 scalper 逻辑）

## 复用的现有组件

- `scalper/price.py` — GMGN 实时价格
- `scalper/jupiter_quote.py` — Jupiter 报价和执行
- `codex_api.py` — 5min bars with buy/sell volume
- `scalper/moralis_api.py` — holder 历史
- `early_warning/predict.py` — 模型信心分
- `scalper/db.py` 架构 — SQLite 模式参考

## 实施路径

1. **Phase 1: 训练数据 + 模型** — build_dataset.py + train.py + backtest.py
2. **Phase 2: 回测验证** — 量化评估，确认模型 vs baseline 有优势
3. **Phase 3: Paper Trading** — posbot engine + paper executor
4. **Phase 4: Live** — 接入 Jupiter 真实交易
