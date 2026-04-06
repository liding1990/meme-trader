# 交易漏斗系统设计

## 概述

一个 4 层漏斗式自动交易系统：从全市场扫描到持仓管理，逐层过滤，最终输出交易收益。

## 漏斗结构

```
Layer 0: 全市场扫描 (Codex 实时 / GMGN 历史回测)
  条件: MCap > $100K, Holders > 100, 4h 涨幅 > 15%
  输出: ~50-200 tokens/次
        ↓
Layer 1: Scam/Rug 过滤
  模型: Outcome Cluster + Early Warning grade
  过滤: Pump-Dump 类、Flash Crash 类、Grade D
  输出: ~20-50 tokens
        ↓
Layer 2: 入场信号 (Entry Model)
  模型: CatBoost 二分类 (买 vs 不买)
  标签: 从这个点买入后 48h 最大收益 > 20% → 买
  输出: ~5-10 entries/天
        ↓
Layer 3: 持仓管理 (Position Model)
  模型: CatBoost 7 类分类
  动作: HOLD / TP_25 / TP_50 / TP_100 / SL_25 / SL_50 / EXIT
  输出: 交易收益
```

## Layer 0: 全市场扫描

### 实盘模式
- 数据源: Codex `filterTokens` API
- 频率: 每 10 分钟扫描一次
- 条件:
  - MCap > $100K
  - Holders > 100
  - 4h 涨幅 > 15% (从 Codex hourly bars 计算)

### 回测模式
- 数据源: 454 个 radar token 的 GMGN 完整历史 (小时级)
- 方法: 遍历每个 token 的每个小时, 检查是否满足 L0 条件
- 覆盖: 262 个 token 至少通过一次, 共 14,354 个 L0 窗口

## Layer 1: Scam/Rug 过滤

### 模型
复用已有的两个模型:
1. **Outcome Cluster** (`outcome_cluster_predict.py`): 判断 token 属于哪个结果类别
   - 过滤: Pump-Dump 类, Flash Crash 类
2. **Early Warning** (`early_warning/predict.py`): 输出 SABCD grade
   - 过滤: Grade D (bottom 80%, 无 pump 信号)

### 训练标签
- 正例 (通过): 最终 ATH > entry price × 1.5 且 max_holders > 500
- 负例 (过滤): scam (ATH 在前 2h, 归零, <2K holders) 或 rug (归零且无恢复)

### 评估指标
- 过滤精度: 被过滤掉的 token 中确实是 scam/rug 的占比 (目标 > 80%)
- 漏放率: 通过过滤但实际是 scam 的比例 (目标 < 5%)
- 误杀率: 被过滤掉但实际是 runner 的比例 (目标 < 10%)

## Layer 2: 入场信号 (Entry Model)

### 模型
CatBoost 二分类: 在这个时间点买入是否值得?

### 训练标签
对每个通过 L1 的 token, 在每个时间点:
- **买入 (1)**: 从当前点开始 48h 内最大涨幅 > 20%
- **不买 (0)**: 48h 内最大涨幅 < 20%

### 特征 (与 Layer 3 共享特征体系)
- 量价: roc_1h/4h/12h, volatility, volume_trend, buy_sell_ratio
- VWAP: price_vs_vwap, vwap_slope
- HMM: hmm_state, hmm_duration, hmm_trans_to_down
- 链上: holder_growth, mcap_per_holder, top10_pct
- 早期预警: ew_predict_mult, ew_grade

### 评估指标
- 入场胜率: 入场后 48h 内最大涨幅 > 20% 的占比 (目标 > 50%)
- 信号精度: 模型说"买"时, 买了确实赚钱的比例 (目标 > 45%)
- 日均信号数: 每天产生多少个买入信号 (目标 3-10)

## Layer 3: 持仓管理 (Position Model)

### 模型
CatBoost 7 类分类

### 7 个决策动作
| 动作 | 含义 |
|---|---|
| HOLD | 继续持有 |
| TP_25 | 止盈卖 25% |
| TP_50 | 止盈卖 50% |
| TP_100 | 止盈全卖 |
| SL_25 | 止损减 25% |
| SL_50 | 止损减 50% |
| EXIT | 全部止损退出 |

### 训练标签
使用 hindsight-optimal 策略 (上帝视角) 回标:
- 在局部峰值处 TP (看未来确认即将回调)
- 在确认无恢复时 SL/EXIT
- 仓位状态 (sold_pct, remaining_pct) 随操作动态变化

### 特征 (30 维)
A. 持仓状态: unrealized_pnl, holding_hours, sold_pct, remaining_pct, distance_from_peak
B. 量价: roc_1h/4h/12h, volatility_1h/4h, volume_trend, volume_concentration
C. VWAP: vwap_24h, price_vs_vwap, vwap_slope
D. HMM: hmm_state, hmm_state_duration, hmm_trans_to_down
E. 链上: holder_growth_1h/4h, mcap_per_holder, top10_pct, top10_change
F. 模型: ew_predict_mult, ew_grade

### 评估指标
- 捕获率: 实际收益 / 完美操作收益 (目标 > 30%)
- 止损有效性: 止损后 token 确实继续跌的比例 (目标 > 60%)

## 端到端评估维度

### 整体指标
| 指标 | 含义 | 目标 |
|---|---|---|
| 日均收益率 | 所有交易平均每日 PnL | > 0 |
| Sharpe Ratio | 风险调整收益 | > 1.5 |
| Calmar Ratio | 年化收益 / 最大单笔亏损 | > 5 |
| Profit Factor | 总盈利 / 总亏损 | > 1.5 |
| 胜率 | 盈利交易占比 | > 40% |
| 平均盈利 | 盈利交易的平均收益 | 追踪 |
| 平均亏损 | 亏损交易的平均损失 | < 20% |
| 平均盈利持仓时间 | 盈利交易从 entry 到完全退出 | 追踪 |
| 平均亏损持仓时间 | 亏损交易从 entry 到完全退出 | 越短越好 |
| 最大单笔亏损 | 最差一笔 | < 30% |
| 日均交易数 | 每天开几仓 | 3-10 |

### 逐层指标
| 层 | 指标 | 目标 |
|---|---|---|
| L0 | 候选 token 数/天 | 50-200 |
| L0 | 候选中最终盈利的占比 | 追踪 |
| L1 | scam 过滤精度 | > 80% |
| L1 | runner 误杀率 | < 10% |
| L2 | 入场胜率 | > 50% |
| L2 | 入场信号精度 | > 45% |
| L2 | 日均信号数 | 3-10 |
| L3 | 收益捕获率 | > 30% |
| L3 | 止损有效性 | > 60% |

## 训练方法

### 原则
- 分层独立训练, 串联端到端评估
- GroupKFold 按 token 拆分, 确保无数据泄漏
- 各层模型互不依赖, 但共享特征体系

### 训练数据来源
454 个 radar token 的完整 GMGN 历史数据:
- 小时级 OHLCV (token_mcap_candles)
- Moralis/GMGN holder 趋势
- Top10 holder concentration
- 共 14,354 个 L0 窗口 (262 个 token)

### 训练流程

```
Step 1: 构建 L0 窗口数据集
  遍历所有 token 所有小时, 标记哪些时刻满足 L0 条件

Step 2: 训练 L1 (过滤模型)
  输入: L0 通过的 token + 该时刻的特征
  标签: 最终结果是 scam/rug (过滤) vs 正常 (通过)
  → 已有: outcome_cluster + early_warning, 直接复用

Step 3: 训练 L2 (入场模型)
  输入: 通过 L1 的 token + 每个时间点的特征
  标签: 未来 48h 最大涨幅 > 20% → 买, 否则 → 不买
  模型: CatBoost 二分类
  验证: GroupKFold, 报告入场胜率和信号精度

Step 4: 训练 L3 (持仓管理模型)
  输入: L2 入场后的每个 tick + 动态仓位状态
  标签: hindsight-optimal 动作 (7 类)
  模型: CatBoost 多分类
  验证: GroupKFold, 回测模拟完整交易

Step 5: 端到端回测
  串联 L0→L1→L2→L3, 模拟完整漏斗
  计算所有端到端指标
  识别瓶颈层 → 针对性优化
```

### 迭代优化流程

```
跑端到端回测 → 看整体 Sharpe/Calmar/PF
                ↓
          哪一层拖后腿?
         ↓          ↓          ↓
     L1 误杀多    L2 入场差   L3 管理差
         ↓          ↓          ↓
   调宽过滤阈值  加特征/调标签  调标签逻辑
         ↓          ↓          ↓
      重训 L1    重训 L2     重训 L3
                ↓
         再跑端到端回测
```

## 文件结构

```
trading_funnel/
  __init__.py
  l0_scanner.py       — L0 全市场扫描 (Codex 实盘 / GMGN 回测)
  l1_filter.py        — L1 scam/rug 过滤 (复用 outcome_cluster + early_warning)
  l2_entry.py         — L2 入场模型 (CatBoost 二分类)
  l3_position.py      — L3 持仓管理 (CatBoost 7 类)
  features.py         — 共享特征计算 (VWAP, HMM, 技术指标)
  backtest.py         — 端到端回测引擎
  train_all.py        — 一键训练所有层
  evaluate.py         — 端到端评估 + 逐层指标
  live_engine.py      — 实盘引擎 (L0 Codex扫描 → L1→L2→L3 → Jupiter执行)
```

## 实施路径

1. **Phase 1**: 构建 L0 窗口数据集 + L2 入场模型训练
2. **Phase 2**: 端到端回测引擎 (串联 L0→L1→L2→L3)
3. **Phase 3**: 评估 + 迭代优化各层
4. **Phase 4**: 实盘 paper trading
5. **Phase 5**: Live trading (Jupiter)

## 复用组件

- `outcome_cluster.py` / `outcome_cluster_predict.py` → L1
- `early_warning/predict.py` → L1 + L2 特征
- `posbot_train/build_dataset.py` → L3 训练数据
- `posbot_train/train.py` → L3 模型训练
- `codex_api.py` → L0 实盘扫描
- `gmgn_api.py` → 历史数据 + 实时价格
- `scalper/jupiter_quote.py` → 实盘执行
- `scalper/moralis_api.py` → holder 数据
