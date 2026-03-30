# Paper Trading System Design

## Overview

A long-running Python process (tmux on MacBook) that monitors Solana memecoins, generates entry/exit signals using the backtested strategy, and simulates trades with detailed logging for iterative optimization.

## Architecture

```
trader/
  engine.py       — Main loop + tick scheduling
  scanner.py      — Token discovery + signal generation
  portfolio.py    — Position management + exit signals
  executor.py     — Paper trade execution + PnL tracking
  db.py           — SQLite read/write
  config.py       — All parameters
```

## Data Flow

Every 5 minutes (tick):
1. **Discover** (hourly): Codex filterTokens → new tokens to watchlist
2. **Scan**: Codex getTokenBars (5m, countback=100) → compute indicators → entry signals → meta-model filter
3. **Manage**: Update positions → check exit conditions → close positions
4. **Execute**: Open new positions (paper), log everything

## Key Parameters

- Max 5 simultaneous positions, $100 each, $10K initial capital
- Entry: ROC>3%, RVOL>1.5, acceleration>0, OFI>=0, EBSW>0, price>ITrend, meta P>0.55
- Exit: survival hazard>10%, trailing stop 15%, tight stop 8%, hard stop 30%
- Daily loss limit: $500

## Logging

- SQLite: trades, watchlist, tick_log, positions tables
- Text: INFO (daily rotate) + DEBUG (indicator details)
- entry_signals/exit_signals JSON snapshots for every trade

## Error Handling

- API failures: skip token, continue tick
- Process crash: restore from SQLite positions table
- Network down: log warning, don't panic-close positions
