"""Scalper V3 configuration — independent from trader/ config."""

import os

# === System ===
TICK_INTERVAL = 90              # 90-second scan cycle
DISCOVER_INTERVAL = 1800        # 30 minutes — discover new tokens
WATCHLIST_MAX = 500             # wider funnel than trader (200)
SCAN_BATCH_SIZE = 50            # tokens per tick (rotate through watchlist)

# === Discovery Filters (looser than trader) ===
DISCOVER_MIN_MCAP = 30000       # lower than trader's 50K
DISCOVER_MIN_HOLDERS = 300      # lower than trader's 1000

# === Data Source ===
CODEX_BAR_COUNTBACK = 36        # 36 bars × 5min = 3 hours (enough for ROC/RVOL)
CODEX_RESOLUTION = "5"
SOLANA_NETWORK_ID = 1399811149
HOLDER_CACHE_TTL = 300          # 5 minutes
SCAN_RATE_LIMIT = 0.3           # seconds between API calls per token

# === State Classifier ===
CLASSIFIER_MODEL_PATH = os.path.join("models", "state_classifier.pkl")
FAVORABLE_MIN_WIN_RATE = 0.50   # cluster must have > 50% win rate
FAVORABLE_MIN_MEDIAN_RETURN = 0.0  # cluster median 4h return > 0% (positive)
CLASSIFY_MAX_DISTANCE_SIGMA = 2.0  # max distance from centroid in std devs

# === Entry Conditions ===
ROC_THRESHOLD = 1.5             # ROC 30m > 1.5% (lower than trader's 3%)
RVOL_THRESHOLD = 1.0            # RVOL > 1.0 (lower than trader's 1.5)

# === Exit Conditions ===
TAKE_PROFIT_PCT = 5.0           # +5% from entry
TRAILING_STOP_PCT = 3.0         # -3% from peak (after +2% profit)
TRAILING_ACTIVATE_PCT = 2.0     # trailing stop activates after this profit
HARD_STOP_PCT = 3.0             # -3% from entry (tight, fast cut)
TIME_STOP_BARS = 18             # 90 minutes max hold
MOMENTUM_REVERSAL_ROC = -3.0    # ROC 30m < -3% triggers exit

# === Risk Management ===
POSITION_SIZE = 50.0            # $50 per trade
MAX_POSITIONS = 10              # max simultaneous positions
INITIAL_CAPITAL = 5000.0        # dedicated scalper capital
DAILY_LOSS_LIMIT = 200.0        # pause new entries if daily loss exceeds
MAX_TRADES_PER_DAY = 30         # circuit breaker
COOLDOWN_MINUTES = 30           # per-token re-entry cooldown
MAX_PER_CLUSTER = 2             # max positions in same state cluster
WIN_RATE_PAUSE_THRESHOLD = 0.40 # pause 1h if win rate < 40% after 10 trades

# === Safety ===
MAX_TOP10_PCT = 50.0            # block if top10 holders > 50%
MIN_HOLDERS = 100               # block if holders < 100
MAX_PRICE_IMPACT_PCT = 3.0      # Jupiter quote: max price impact

# === Backtest ===
BACKTEST_SLIPPAGE_PCT = 1.0     # fixed slippage estimate for backtest
TRAIN_RATIO = 0.6               # 60/40 train/test split

# === Logging & DB ===
LOG_DIR = "logs"
DB_PATH = os.path.join("data", "paper_trading.db")  # shared with trader
