"""Unified Token Discovery daemon — runs all pipelines on schedule.

Single process that orchestrates:
  - L1 Cluster Scan:    every 15 min (Codex trending → cluster filter → candidate pool)
  - Candidate Monitor:  every 60 min (regression z-score scoring)
  - Entry Score:        every 15 min (real-time scoring with freshness decay)

Usage:
    PYTHONPATH=. python -m token_discovery.run
"""

import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from token_discovery.pipeline import run_once as run_discovery, setup_logger
from token_discovery.monitor import run_once as run_monitor
from token_discovery.entry_score import run_once as run_entry_score

DISCOVERY_INTERVAL = 15 * 60   # 15 min
MONITOR_INTERVAL = 60 * 60     # 60 min
ENTRY_INTERVAL = 15 * 60       # 15 min

log = setup_logger()


def main():
    log.info("=" * 60)
    log.info("Token Discovery Unified Daemon Starting")
    log.info(f"  L1 Scan:          every {DISCOVERY_INTERVAL // 60} min")
    log.info(f"  Candidate Monitor: every {MONITOR_INTERVAL // 60} min")
    log.info(f"  Entry Score:       every {ENTRY_INTERVAL // 60} min")
    log.info("=" * 60)

    last_discovery = 0
    last_monitor = 0
    last_entry = 0

    # Run everything once at startup
    tick = 0

    while True:
        tick += 1
        now = time.time()
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")

        ran_something = False

        # L1 Discovery
        if now - last_discovery >= DISCOVERY_INTERVAL:
            log.info(f"\n[{now_str}] Running L1 Cluster Scan...")
            try:
                run_discovery()
                last_discovery = time.time()
                ran_something = True
            except Exception as e:
                log.error(f"L1 Scan error: {e}")
                traceback.print_exc()

        # Candidate Monitor (less frequent)
        if now - last_monitor >= MONITOR_INTERVAL:
            log.info(f"\n[{now_str}] Running Candidate Monitor...")
            try:
                run_monitor()
                last_monitor = time.time()
                ran_something = True
            except Exception as e:
                log.error(f"Monitor error: {e}")
                traceback.print_exc()

        # Entry Score (after discovery + monitor)
        if now - last_entry >= ENTRY_INTERVAL:
            log.info(f"\n[{now_str}] Running Entry Score...")
            try:
                run_entry_score()
                last_entry = time.time()
                ran_something = True
            except Exception as e:
                log.error(f"Entry Score error: {e}")
                traceback.print_exc()

        if ran_something:
            next_discovery = max(0, DISCOVERY_INTERVAL - (time.time() - last_discovery))
            next_monitor = max(0, MONITOR_INTERVAL - (time.time() - last_monitor))
            next_entry = max(0, ENTRY_INTERVAL - (time.time() - last_entry))
            next_any = min(next_discovery, next_monitor, next_entry)
            log.info(f"\nNext: discovery={next_discovery/60:.0f}m, monitor={next_monitor/60:.0f}m, entry={next_entry/60:.0f}m")
            log.info(f"Sleeping {next_any/60:.1f} min...")

        # Sleep in short intervals so we can check more precisely
        time.sleep(60)


if __name__ == "__main__":
    main()
