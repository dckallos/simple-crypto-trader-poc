#!/usr/bin/env python3
# =============================================================================
# FILE: backfill_lunarcrush_timeseries.py
# =============================================================================
"""
A script to back-populate 'lunarcrush_timeseries' for up to 1 year of history
(assuming it's available), chunked by monthly intervals. Respects the limit of
10 calls/min by sleeping 6 seconds between calls.

Usage:
    python backfill_lunarcrush_timeseries.py

Requirements:
    - You must have the 'LunarCrushFetcher' class or an equivalent time-series fetch method
      that can accept start/end timestamps, along with the coin ID.
    - DB schema for 'lunarcrush_data' and 'lunarcrush_timeseries' must already exist.
    - The script uses a naive monthly chunk approach (30 days each).
    - If you want an exact approach, you'd handle calendar months precisely.

This script doesn't skip or omit code for brevity. Everything is shown in full.
"""

import os
import time
import math
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv

# If you have a "lunar_crush_fetcher.py" containing "LunarCrushFetcher", import it:
# from lunar_crush_fetcher import LunarCrushFetcher

# For demonstration, we'll inline a minimal fetcher snippet that you can adapt.
# If you already have a "fetch_lunarcrush_time_series" method, just reference it.

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DB_FILE = "trades.db"  # or your actual DB path

class MinimalLunarCrushFetcher:
    """
    A minimal version of your time-series fetcher code with the method:
      `_fetch_lunarcrush_time_series(coin_id: str, start=..., end=..., bucket=..., etc.)`
    We'll just show a direct approach: storing or updating the table with each chunk.
    """
    def __init__(self, db_path=DB_FILE):
        self.db_path = db_path
        load_dotenv()

        self.BEARER_TOKEN = os.getenv("LUNARCRUSH_BEARER_TOKEN", "")
        if not self.BEARER_TOKEN:
            logger.warning("No LUNARCRUSH_BEARER_TOKEN set. If your time-series endpoint requires it, calls may fail.")

    def _init_timeseries_table(self):
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS lunarcrush_timeseries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin_id TEXT,
                    timestamp INTEGER,
                    open_price REAL,
                    close_price REAL,
                    high_price REAL,
                    low_price REAL,
                    volume_24h REAL,
                    market_cap REAL,
                    sentiment REAL,
                    spam REAL,
                    galaxy_score REAL,
                    alt_rank INTEGER,
                    volatility REAL,
                    interactions REAL,
                    social_dominance REAL
                )
            """)
            conn.commit()
        except Exception as e:
            logger.exception(f"Error creating 'lunarcrush_timeseries' table: {e}")
        finally:
            conn.close()

    def _fetch_lunarcrush_time_series(self, coin_id: str, start: int, end: int,
                                      bucket: str = "hour", interval: str = "1w"):
        """
        Single call to time-series endpoint for the given [start, end].
        We'll store the results in 'lunarcrush_timeseries'.
        """
        import requests

        base_url = f"https://lunarcrush.com/api4/public/coins/{coin_id}/time-series/v2"
        headers = {}
        if self.BEARER_TOKEN:
            headers["Authorization"] = f"Bearer {self.BEARER_TOKEN}"

        params = {
            "bucket": bucket,
            "interval": interval,
            "start": start,
            "end": end
        }

        try:
            logger.info(f"Requesting time-series for coin_id={coin_id}, start={start}, end={end}, bucket={bucket}")
            resp = requests.get(base_url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()

            data = resp.json()
            records = data.get("data", [])
            if not records:
                logger.info(f"No time-series data returned for coin_id={coin_id} in chunk {start}..{end}")
                return

            self._init_timeseries_table()
            self._store_timeseries_records(coin_id, records)
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error => coin_id={coin_id}, start={start}, end={end}, error={e}")
        except Exception as e:
            logger.exception(f"Error fetching/storing coin_id={coin_id} timeseries: {e}")

    def _store_timeseries_records(self, coin_id: str, records: list):
        """
        Insert each data record into 'lunarcrush_timeseries'.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            for rec in records:
                timestamp = rec.get("time", 0)
                open_price = rec.get("open", 0.0)
                close_price = rec.get("close", 0.0)
                high_price = rec.get("high", 0.0)
                low_price = rec.get("low", 0.0)
                volume_24h = rec.get("volume_24h", 0.0)
                market_cap = rec.get("market_cap", 0.0)
                sentiment = rec.get("sentiment", 0.0)
                spam = rec.get("spam", 0.0)
                galaxy_score = rec.get("galaxy_score", 0.0)
                alt_rank = rec.get("alt_rank", 0)
                volatility = rec.get("volatility", 0.0)
                interactions = rec.get("interactions", 0.0)
                social_dominance = rec.get("social_dominance", 0.0)

                c.execute("""
                    INSERT INTO lunarcrush_timeseries (
                        coin_id, timestamp,
                        open_price, close_price, high_price, low_price,
                        volume_24h, market_cap, sentiment, spam,
                        galaxy_score, alt_rank, volatility,
                        interactions, social_dominance
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    coin_id,
                    timestamp,
                    open_price,
                    close_price,
                    high_price,
                    low_price,
                    volume_24h,
                    market_cap,
                    sentiment,
                    spam,
                    galaxy_score,
                    alt_rank,
                    volatility,
                    interactions,
                    social_dominance
                ])
            conn.commit()
            logger.info(f"Inserted {len(records)} rows for coin_id={coin_id} into 'lunarcrush_timeseries'.")
        except Exception as e:
            logger.exception(f"Error storing time-series records for coin_id={coin_id}: {e}")
        finally:
            conn.close()

def backfill_one_coin(fetcher: MinimalLunarCrushFetcher, coin_id: str, months: int = 12):
    """
    Backfill up to 'months' months in monthly intervals (30-day lumps)
    for coin_id, pacing calls at 10 calls/min => 6s sleep after each chunk.

    :param fetcher: The MinimalLunarCrushFetcher (or your real LunarCrushFetcher).
    :param coin_id: The numeric ID or symbol used by the LC time-series endpoint.
    :param months: how many months to go back. default=12 => ~1 year.
    """

    now_ts = int(time.time())  # end boundary = "now"
    # approximate 1 month as 30 days => 30*86400 = 2592000
    chunk_seconds = 30*86400

    # We'll do intervals like: [start, end], each ~1 month
    # start from oldest => up to now
    # e.g. 12 months => 12 calls => each call => 6s sleep => total 72s for 1 coin
    # If you have 30 coins => ~36 minutes.

    # Calculate oldest needed => ~ months * chunk_seconds
    total_back_seconds = months * chunk_seconds
    oldest_ts = now_ts - total_back_seconds

    # We'll chunk from oldest -> newest
    current_start = oldest_ts

    # Because we might not always match exactly, we do while current_start < now_ts
    while current_start < now_ts:
        current_end = current_start + chunk_seconds
        if current_end > now_ts:
            current_end = now_ts

        logger.info(f"[BACKFILL] coin_id={coin_id} => chunk start={current_start}, end={current_end}")
        fetcher._fetch_lunarcrush_time_series(
            coin_id=coin_id,
            start=current_start,
            end=current_end,
            bucket="hour",   # or "hour" if you want hourly data
            interval="1w"   # interval is overshadowed if we explicitly pass start/end
        )

        # respect 10 calls/min => sleep 6s
        logger.info("[RATE-LIMIT] Sleeping 6s to keep calls under 10/minute.")
        time.sleep(6)

        current_start = current_end
        if current_start >= now_ts:
            break

def main():
    """
    Main script entry:
      1) Gather coin IDs from 'lunarcrush_data'.
      2) For each coin => backfill up to 12 months of data in monthly chunks.
    """
    fetcher = MinimalLunarCrushFetcher(DB_FILE)

    # Retrieve coin_ids from your 'lunarcrush_data' table
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            SELECT DISTINCT lunarcrush_id, symbol
            FROM lunarcrush_data
            WHERE lunarcrush_id IS NOT NULL
        """)
        rows = c.fetchall()
    except Exception as e:
        logger.exception(f"Error retrieving coin IDs: {e}")
        return
    finally:
        conn.close()

    if not rows:
        logger.info("No coin IDs found in 'lunarcrush_data'. Exiting.")
        return

    # We'll do a 1-year backfill => months=12
    months = 12

    for idx, (lc_id, sym) in enumerate(rows, start=1):
        if not lc_id:
            continue
        logger.info(f"\n=== Backfilling coin_id={lc_id}, symbol={sym} ({idx}/{len(rows)}) ===\n")
        # For each coin => run the backfill
        backfill_one_coin(fetcher, str(lc_id), months=months)

        # If you have 10 calls/min total for the entire script,
        # you might consider an additional sleep here after each coin if needed.

if __name__ == "__main__":
    main()
