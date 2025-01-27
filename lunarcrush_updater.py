#!/usr/bin/env python3
# =============================================================================
# FILE: lunarcrush_updater.py
# =============================================================================
"""
lunarcrush_updater.py

A background updater that periodically fetches time-series data for each
coin in 'lunarcrush_data', storing results in 'lunarcrush_timeseries'.
Respects the ~10 calls per minute rule by sleeping ~6s per coin.

Usage:
    python lunarcrush_updater.py [--interval=1800]

Where:
    --interval=SECONDS is how many seconds to wait between full passes
    (defaults to 1800s = 30 minutes).

Process Flow:
    1) On startup, connect to DB and list all coin_ids from 'lunarcrush_data'.
    2) For each coin_id, call fetch_time_series_for_tracked_coins (or a direct
       _fetch_lunarcrush_time_series if you prefer).
    3) Sleep 6 seconds per coin to avoid more than ~10 requests/min.
    4) Sleep 'interval' after finishing a full pass, then repeat.

You can incorporate incremental logic if you want, by calling
`fetch_last_increment_of_time()` from your `LunarCrushFetcher` and passing
'start' timestamps. For brevity, we do a simpler approach here.
"""

import time
import logging
import argparse
import sqlite3
import yaml
import os
from typing import Optional

from fetch_lunarcrush import LunarCrushFetcher, REQUEST_SLEEP_SECONDS, DB_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_PASS_INTERVAL = 1800  # 30 minutes

def parse_args():
    parser = argparse.ArgumentParser(description="Background job to update LunarCrush timeseries.")
    parser.add_argument("--interval", type=int, default=DEFAULT_PASS_INTERVAL,
                        help=f"Seconds to wait between fetch cycles (default={DEFAULT_PASS_INTERVAL}).")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    pass_interval = args.interval

    # Read config if needed, or skip:
    config_file = "config.yaml"
    if os.path.exists(config_file):
        with open(config_file,"r") as f:
            config = yaml.safe_load(f)
        # You might parse certain config keys, e.g. 'bucket' or 'interval' or 'months'
    else:
        logger.warning("No config.yaml found => using default approach.")

    fetcher = LunarCrushFetcher(db_file=DB_FILE)
    logger.info("[LunarCrush Updater] Starting indefinite background loop. Press Ctrl+C to exit.")

    try:
        while True:
            # (A) Possibly fetch the list of coin_ids from 'lunarcrush_data'
            # We'll just do fetch_time_series_for_tracked_coins, which
            # calls the internal logic for each coin in the DB.
            logger.info("[LunarCrush Updater] Starting a new pass => fetch_time_series_for_tracked_coins(...)")

            # If you want a completely incremental approach:
            #  - you might do something like:
            #    last_increments = fetcher.fetch_last_increment_of_time()
            #    then for each coin => fetch from last_increments[coin]+1 onward
            # or if you want a simpler approach for demonstration:
            fetcher.fetch_time_series_for_tracked_coins(
                bucket="hour",
                interval="1w",
                start=None,
                end=None
            )

            # (B) Sleep pass_interval
            logger.info(f"[LunarCrush Updater] Sleep {pass_interval}s before next pass...")
            time.sleep(pass_interval)

    except KeyboardInterrupt:
        logger.info("[LunarCrush Updater] user exit => stopping.")
    except Exception as e:
        logger.exception(f"[LunarCrush Updater] main error => {e}")
    finally:
        logger.info("[LunarCrush Updater] Exiting.")

if __name__ == "__main__":
    main()
