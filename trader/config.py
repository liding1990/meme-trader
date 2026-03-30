"""Centralized configuration for the paper trading system."""

import os

# === System ===
TICK_INTERVAL = 300             # 5 minutes (seconds)
DISCOVER_INTERVAL = 3600        # 1 hour — discover new tokens
WATCHLIST_MAX = 200             # max tokens to monitor
WATCHLIST_REMOVE_BELOW_MCAP = 30000  # remove token if mcap drops below this
WATCHLIST_REMOVE_INACTIVE_HOURS = 24  # remove if no volume for this long

# === Data Source ===
CODEX_BAR_COUNTBACK = 100       # bars per getTokenBars call
CODEX_RESOLUTION = "5"          # 5-minute bars
SOLANA_NETWORK_ID = 1399811149

# === Discovery Filters ===
DISCOVER_MIN_MCAP = 50000
DISCOVER_MIN_HOLDERS = 1000

# === Entry Conditions ===
ROC_THRESHOLD = 3.0             # ROC 30m must exceed this %
RVOL_THRESHOLD = 1.5            # relative volume must exceed this
ENTRY_REQUIRE_OFI = True        # require OFI >= 0
ENTRY_REQUIRE_EBSW = True       # require EBSW > 0
ENTRY_REQUIRE_ITREND = True     # require price > ITrend
META_MODEL_THRESHOLD = 0.55     # LightGBM P(profit), None to disable
META_MODEL_PATH = "models/meta_lgbm.pkl"
SURVIVAL_MODEL_PATH = "models/survival.pkl"

# === Exit Conditions ===
TRAILING_STOP_PCT = 15.0        # trailing stop from peak
TIGHT_STOP_PCT = 8.0            # tightened stop when momentum weakens
HARD_STOP_PCT = 30.0            # catastrophic stop from entry
HAZARD_THRESHOLD = 0.10         # survival hazard > this triggers exit
GRACE_PERIOD_BARS = 6           # bars before trailing stop activates

# === Risk Management ===
MAX_POSITIONS = 5               # max simultaneous positions
POSITION_SIZE = 100.0           # $ per trade (paper)
INITIAL_CAPITAL = 10000.0       # starting paper balance
DAILY_LOSS_LIMIT = 500.0        # max daily loss before pause
COOLDOWN_HOURS = 2              # hours before re-entry on same token after loss

# === Logging ===
LOG_DIR = "logs"
DB_PATH = os.path.join("data", "paper_trading.db")
LOG_LEVEL = "INFO"
LOG_ROTATE_DAYS = 30
