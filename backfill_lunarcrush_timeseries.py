#!/usr/bin/env python3
# =============================================================================
# FILE: backfill_lunarcrush_timeseries.py
# =============================================================================
"""
A script to back-populate 'lunarcrush_timeseries' for up to 1 year (or user-specified months),
chunked in ~30-day intervals, respecting ~10 calls/min by sleeping 6 seconds per chunk.

Enhancements:
    - We import `LunarCrushFetcher` from `fetch_lunarcrush.py` for time-series fetch logic.
    - We optionally fetch for either:
        --all-coins => read all coin_ids from `lunarcrush_data` and backfill them
        or
        => only "top coins" from `spot_check_high_performers()`.
    - We add a **new function** `backfill_configured_coins(...)` to backfill only
      those coins found in your config.yaml "traded_pairs" if they exist in `lunarcrush_data`.
      This can skip certain coins or focus on ones your config references.

Usage:
    python backfill_lunarcrush_timeseries.py [--all-coins] [--months=12] [--configured-coins]

Examples:
    # backfill top 10 coins
    python backfill_lunarcrush_timeseries.py --months=6

    # backfill all coins in 'lunarcrush_data'
    python backfill_lunarcrush_timeseries.py --all-coins

    # backfill only config.yaml "traded_pairs" if they exist
    python backfill_lunarcrush_timeseries.py --configured-coins
"""

import os
import sys
import time
import math
import sqlite3
import logging
import argparse
import yaml
from datetime import datetime, timedelta
from typing import Optional, List, Tuple

from fetch_lunarcrush import LunarCrushFetcher

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DB_FILE = "trades.db"   # your DB path
CHUNK_SECONDS = 30 * 86400  # ~30 days in seconds
REQUEST_SLEEP_SECONDS = 6   # ~10 calls/min

def parse_args():
    parser = argparse.ArgumentParser(description="Backfill LunarCrush time-series data.")
    parser.add_argument("--all-coins", action="store_true",
                        help="Backfill all coin_ids from 'lunarcrush_data'.")
    parser.add_argument("--months", type=int, default=12,
                        help="Number of months to backfill (default=12 => 1 year).")
    parser.add_argument("--configured-coins", action="store_true",
                        help="Backfill only the coins from config.yaml 'traded_pairs' (that exist in 'lunarcrush_data').")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    months = args.months

    # We'll read config if present to see traded_pairs
    config_file = "config.yaml"
    traded_pairs = []
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        traded_pairs = config.get("traded_pairs", [])
    else:
        logger.warning("No config.yaml found => 'configured-coins' approach won't have pairs to read.")

    fetcher = LunarCrushFetcher(db_file=DB_FILE)
    logger.info("Initialized LunarCrushFetcher for time-series backfill.")

    if args.all_coins:
        # 1) We'll read all coin_ids
        coin_ids = _fetch_all_coin_ids()
        logger.info(f"[BACKFILL] Will backfill ALL coins => {len(coin_ids)} found.")
    elif args.configured_coins:
        # 2) We'll only pick the coin_ids corresponding to your config.yaml 'traded_pairs'
        coin_ids = backfill_configured_coins(traded_pairs)
        logger.info(f"[BACKFILL] Will backfill coins from config => {coin_ids}")
    else:
        # 3) We'll pick the top coins from fetcher.spot_check_high_performers
        top_n = 10
        top_coins = fetcher.spot_check_entire_market(top_n=top_n)
        if not top_coins:
            logger.warning("No top coins found => aborting.")
            return
        # top_coins => e.g. [(coin_id, rank_score), ...]
        coin_ids = [tc[0] for tc in top_coins]
        logger.info(f"[BACKFILL] Will backfill top {top_n} coins => {coin_ids}")

    if not coin_ids:
        logger.info("[BACKFILL] No coin IDs => nothing to do.")
        return

    now_ts = int(time.time())
    total_back_seconds = months * CHUNK_SECONDS
    oldest_ts = now_ts - total_back_seconds

    # We'll do a function for chunk logic
    def backfill_one_coin(coin_id: str, months_count: int):
        logger.info(f"[BACKFILL] Starting coin_id={coin_id}, months={months_count}")
        now_ts2 = int(time.time())
        chunk_start = now_ts2 - (months_count * CHUNK_SECONDS)

        while chunk_start < now_ts2:
            chunk_end = chunk_start + CHUNK_SECONDS
            if chunk_end > now_ts2:
                chunk_end = now_ts2

            logger.info(f"[BACKFILL] chunk => start={chunk_start}, end={chunk_end}, coin_id={coin_id}")
            # real call => fetch time-series
            fetcher._fetch_lunarcrush_time_series(
                coin_id=coin_id,
                bucket="hour",
                interval="1w",
                start=chunk_start,
                end=chunk_end
            )

            time.sleep(REQUEST_SLEEP_SECONDS)
            chunk_start = chunk_end
            if chunk_start >= now_ts2:
                break

    # Actually backfill each coin
    for idx, cid in enumerate(coin_ids, start=1):
        logger.info(f"\n=== Backfilling coin_id={cid} ({idx}/{len(coin_ids)}) ===\n")
        backfill_one_coin(str(cid), months)

    logger.info("[BACKFILL] Done with all coins.")

def _fetch_all_coin_ids() -> List[str]:
    """
    Reads all distinct coin_ids from 'lunarcrush_data' => returns them as a list of strings.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            SELECT DISTINCT lunarcrush_id
            FROM lunarcrush_data
            WHERE lunarcrush_id IS NOT NULL
        """)
        rows = c.fetchall()
        coin_ids = [str(r[0]) for r in rows if r[0]]
        return coin_ids
    except Exception as e:
        logger.exception(f"Error retrieving all coin IDs => {e}")
        return []
    finally:
        conn.close()

def backfill_configured_coins(traded_pairs: List[str]) -> List[str]:
    """
    For each pair in traded_pairs, we:
     1) parse the symbol (split by '/')
     2) find the coin_id in 'lunarcrush_data' if it exists
     3) return that list of coin_ids
    We'll skip pairs that don't exist in the DB or don't have a coin_id.

    Example => traded_pairs = ["ETH/USD","XRP/USD","SOL/USD",...]
    """
    if not traded_pairs:
        logger.warning("[ConfiguredCoins] => no traded_pairs => skipping.")
        return []

    # Build a map symbol => lunarcrush_id from DB
    symbol_map = {}
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        q = """
            SELECT UPPER(symbol), lunarcrush_id
            FROM lunarcrush_data
            WHERE lunarcrush_id IS NOT NULL
        """
        rows = c.execute(q).fetchall()
        for sym, cid in rows:
            if sym and cid:
                symbol_map[sym.upper()] = str(cid)
    except Exception as e:
        logger.exception(f"[ConfiguredCoins] error => {e}")
    finally:
        conn.close()

    coin_ids = []
    for pair in traded_pairs:
        symbol = pair.split("/")[0].upper()
        cid = symbol_map.get(symbol)
        if cid:
            coin_ids.append(cid)
        else:
            logger.warning(f"[ConfiguredCoins] No coin_id found in DB for symbol={symbol}. Skipping.")

    logger.info(f"[ConfiguredCoins] Found {len(coin_ids)} coin_ids => {coin_ids}")
    return list(set(coin_ids))  # unique if duplicates

if __name__ == "__main__":
    main()
