#!/usr/bin/env python3
# =============================================================================
# FILE: fetch_lunarcrush.py
# =============================================================================
"""
fetch_lunarcrush.py

An enhanced, class-based script to:
  1) Fetch snapshot data from /public/coins/list/v2, storing in 'lunarcrush_data'.
  2) Fetch time-series data from /public/coins/<coin_id>/time-series/v2, storing
     in 'lunarcrush_timeseries'.

Usage:
    python fetch_lunarcrush.py

Environment:
    - LUNARCRUSH_API_KEY: For snapshot endpoint.
    - LUNARCRUSH_BEARER_TOKEN: If time-series endpoint requires Bearer Auth.

Performance / Data-Volume Plan:
- Time-series data can grow large quickly. Consider archiving older data.
"""

import os
import requests
import logging
import sqlite3
import time
import math
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List

# If your db.py has a function like store_lunarcrush_data(...) for the snapshot,
# you might also add store_lunarcrush_timeseries(...) for the time-series or do it here inline.
# For demonstration, we'll show "INSERT" logic inside this file.

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Example Toggles
filter_symbols = [
    "ETH", "XBT", "SOL", "ADA", "XRP", "LTC", "DOT", "MATIC", "LINK", "ATOM",
    "XMR", "TRX", "SHIB", "AVAX", "UNI", "APE", "FIL", "AAVE", "SAND", "CHZ",
    "GRT", "CRV", "COMP", "ALGO", "NEAR", "LDO", "XTZ", "EGLD", "KSM"
]

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

DB_FILE = "trades.db"  # Adjust as needed


class LunarCrushFetcher:
    """
    A class that fetches LunarCrush data (both snapshot & time-series) and stores it in local DB.
    """

    def __init__(self, db_file: str = DB_FILE):
        self.db_file = db_file
        load_dotenv()

        self.API_KEY = os.getenv("LUNARCRUSH_API_KEY")
        if not self.API_KEY:
            logger.warning("No LUNARCRUSH_API_KEY set for snapshot fetch. Snapshot calls may fail.")

        self.BEARER_TOKEN = os.getenv("LUNARCRUSH_BEARER_TOKEN")
        if not self.BEARER_TOKEN:
            logger.warning("No LUNARCRUSH_BEARER_TOKEN set for time-series fetch. If required, calls may fail.")

    # --------------------------------------------------------------------------
    # 1) Snapshot: /public/coins/list/v2
    # --------------------------------------------------------------------------
    def fetch_snapshot_data(self):
        """
        Fetch data from /public/coins/list/v2 => store in 'lunarcrush_data' table.
        """
        if not self.API_KEY:
            logger.error("No LUNARCRUSH_API_KEY found. Aborting snapshot fetch.")
            return

        base_url = "https://lunarcrush.com/api4/public/coins/list/v2"
        params = {"key": self.API_KEY, "limit": 100}

        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                logger.info(f"[SNAPSHOT] Requesting data from {base_url} with params={params}. Attempt {attempt+1}.")
                resp = requests.get(base_url, params=params, timeout=10)
                resp.raise_for_status()

                data = resp.json()
                coins_list = data.get("data", [])
                logger.info(f"[SNAPSHOT] Fetched {len(coins_list)} coins from LunarCrush snapshot.")

                # Filter symbols if needed
                if filter_symbols:
                    filter_symbols_upper = [s.upper() for s in filter_symbols]
                    coins_list = [
                        c for c in coins_list
                        if c.get("symbol", "").upper() in filter_symbols_upper
                    ]
                logger.info(f"[SNAPSHOT] After filtering, we have {len(coins_list)} coins to store in DB.")

                self._init_lunarcrush_data_table()  # ensure table
                self._store_snapshot_records(coins_list)
                break  # success
            except requests.exceptions.HTTPError as e:
                attempt += 1
                logger.error(f"[SNAPSHOT] HTTP Error: {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                time.sleep(RETRY_DELAY_SECONDS)
            except Exception as e:
                attempt += 1
                logger.exception(f"[SNAPSHOT] Error fetching or storing snapshot: {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                time.sleep(RETRY_DELAY_SECONDS)
        else:
            logger.error(f"[SNAPSHOT] All {MAX_RETRIES} attempts to fetch LunarCrush data have failed.")

    def _init_lunarcrush_data_table(self):
        """
        Creates an updated 'lunarcrush_data' table for snapshot data if not exists.
        Update or alter columns as needed for your schema.
        """
        conn = sqlite3.connect(self.db_file)
        try:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS lunarcrush_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lunarcrush_id INTEGER,         -- numeric coin ID from LC
                    symbol TEXT,
                    name TEXT,
                    price REAL,
                    price_btc REAL,
                    volume_24h REAL,
                    volatility REAL,
                    circulating_supply REAL,
                    max_supply REAL,
                    percent_change_1h REAL,
                    percent_change_24h REAL,
                    percent_change_7d REAL,
                    percent_change_30d REAL,
                    market_cap REAL,
                    market_cap_rank INTEGER,
                    market_dominance REAL,
                    market_dominance_prev REAL,
                    social_volume_24h REAL,
                    interactions_24h REAL,
                    social_dominance REAL,
                    galaxy_score REAL,
                    galaxy_score_previous REAL,
                    alt_rank INTEGER,
                    alt_rank_previous INTEGER,
                    sentiment REAL,
                    categories TEXT,
                    topic TEXT,
                    logo TEXT,
                    inserted_at INTEGER
                )
            """)
            conn.commit()
        except Exception as e:
            logger.exception(f"[SNAPSHOT] Error creating or altering 'lunarcrush_data': {e}")
        finally:
            conn.close()

    def _store_snapshot_records(self, coins_list: List[dict]):
        """
        Insert each coin record from the snapshot into 'lunarcrush_data'.
        """
        now_ts = int(time.time())
        conn = sqlite3.connect(self.db_file)
        try:
            c = conn.cursor()
            for coin in coins_list:
                lunarcrush_id = coin.get("id", None)
                symbol = coin.get("symbol", "Unknown")
                name = coin.get("name", "Unknown")

                price = coin.get("price", 0.0)
                price_btc = coin.get("price_btc", 0.0)

                volume_24h = coin.get("volume_24h", 0.0)
                volatility = coin.get("volatility", 0.0)
                circulating_supply = coin.get("circulating_supply", 0.0)
                max_supply = coin.get("max_supply", 0.0)

                percent_change_1h  = coin.get("percent_change_1h", 0.0)
                percent_change_24h = coin.get("percent_change_24h", 0.0)
                percent_change_7d  = coin.get("percent_change_7d", 0.0)
                percent_change_30d = coin.get("percent_change_30d", 0.0)

                market_cap = coin.get("market_cap", 0.0)
                market_cap_rank = coin.get("market_cap_rank", 999999)
                market_dominance = coin.get("market_dominance", 0.0)
                market_dominance_prev = coin.get("market_dominance_prev", 0.0)

                social_volume_24h = coin.get("social_volume_24h", 0.0)
                interactions_24h  = coin.get("interactions_24h", 0.0)
                social_dominance  = coin.get("social_dominance", 0.0)

                galaxy_score          = coin.get("galaxy_score", 0.0)
                galaxy_score_previous = coin.get("galaxy_score_previous", 0.0)
                alt_rank              = coin.get("alt_rank", 0)
                alt_rank_previous     = coin.get("alt_rank_previous", 0)

                sentiment = coin.get("sentiment", 0.0)
                categories = coin.get("categories", "")
                topic = coin.get("topic", "")
                logo = coin.get("logo", "")

                c.execute("""
                    INSERT INTO lunarcrush_data (
                        lunarcrush_id, symbol, name,
                        price, price_btc,
                        volume_24h, volatility,
                        circulating_supply, max_supply,
                        percent_change_1h, percent_change_24h, percent_change_7d, percent_change_30d,
                        market_cap, market_cap_rank, market_dominance, market_dominance_prev,
                        social_volume_24h, interactions_24h, social_dominance,
                        galaxy_score, galaxy_score_previous,
                        alt_rank, alt_rank_previous,
                        sentiment, categories, topic, logo,
                        inserted_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    lunarcrush_id, symbol, name,
                    price, price_btc,
                    volume_24h, volatility,
                    circulating_supply, max_supply,
                    percent_change_1h, percent_change_24h, percent_change_7d, percent_change_30d,
                    market_cap, market_cap_rank, market_dominance, market_dominance_prev,
                    social_volume_24h, interactions_24h, social_dominance,
                    galaxy_score, galaxy_score_previous,
                    alt_rank, alt_rank_previous,
                    sentiment, categories, topic, logo,
                    now_ts
                ])
            conn.commit()
            logger.info(f"[SNAPSHOT] Inserted {len(coins_list)} rows into 'lunarcrush_data'.")
        except Exception as e:
            logger.exception(f"[SNAPSHOT] Error storing snapshot records: {e}")
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # 2) Time-series: /public/coins/:coin/time-series/v2
    # --------------------------------------------------------------------------
    def fetch_time_series_for_tracked_coins(
            self,
            bucket: str = "hour",
            interval: str = "1w",
            start: Optional[int] = None,
            end: Optional[int] = None
    ):
        """
        Queries the 'lunarcrush_data' table for all coins we have saved, retrieves
        their `lunarcrush_id`, and calls the time-series endpoint with pacing to
        5 requests/second. Stores the results in 'lunarcrush_timeseries'.
        """
        conn = sqlite3.connect(self.db_file)
        try:
            c = conn.cursor()
            # retrieve distinct lunarcrush_id => ignoring None
            c.execute("""
                SELECT DISTINCT lunarcrush_id, symbol
                FROM lunarcrush_data
                WHERE lunarcrush_id IS NOT NULL
            """)
            rows = c.fetchall()
        except Exception as e:
            logger.exception(f"[TIMESERIES] Error retrieving coin IDs from 'lunarcrush_data': {e}")
            return
        finally:
            conn.close()

        if not rows:
            logger.info("[TIMESERIES] No valid coin IDs found in 'lunarcrush_data'. Nothing to fetch.")
            return

        logger.info(f"[TIMESERIES] Starting time-series fetch for {len(rows)} coins.")

        for idx, (coin_id, symbol) in enumerate(rows, start=1):
            # Pace the requests to 5 calls/second => 1 call every 0.2s
            time.sleep(6)

            logger.info(
                f"[TIMESERIES] ({idx}/{len(rows)}) Fetching time-series for coin_id={coin_id}, "
                f"symbol={symbol}, bucket={bucket}, interval={interval}, start={start}, end={end}"
            )
            # Each call below fetches + logs what is parsed inside _fetch_lunarcrush_time_series(...)
            self._fetch_lunarcrush_time_series(
                str(coin_id),
                bucket=bucket,
                interval=interval,
                start=start,
                end=end
            )

    def _fetch_lunarcrush_time_series(
        self,
        coin_id: str,
        bucket: str = "hour",
        interval: str = "1w",
        start: Optional[int] = None,
        end: Optional[int] = None
    ):
        """
        Low-level function to fetch time-series for one coin, then store in 'lunarcrush_timeseries'.
        Repeats a retry loop just like the snapshot approach.
        """
        base_url = f"https://lunarcrush.com/api4/public/coins/{coin_id}/time-series/v2"
        headers = {}
        if self.BEARER_TOKEN:
            headers["Authorization"] = f"Bearer {self.BEARER_TOKEN}"

        params = {
            "bucket": bucket,
            "interval": interval
        }
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end

        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                logger.info(f"[TIMESERIES] GET {base_url} params={params}. Attempt {attempt+1}.")
                resp = requests.get(base_url, headers=headers, params=params, timeout=10)
                resp.raise_for_status()

                data = resp.json()
                records = data.get("data", [])
                logger.info(f"[TIMESERIES] Fetched {len(records)} records for coin_id={coin_id}.")

                if records:
                    self._init_lunarcrush_timeseries_table()
                    self._store_timeseries_records(coin_id, records)
                break  # success
            except requests.exceptions.HTTPError as e:
                attempt += 1
                logger.error(f"[TIMESERIES] HTTP Error: {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                time.sleep(RETRY_DELAY_SECONDS)
            except Exception as e:
                attempt += 1
                logger.exception(f"[TIMESERIES] Error fetching/storing coin_id={coin_id} time-series: {e}. "
                                 f"Retrying in {RETRY_DELAY_SECONDS}s...")
                time.sleep(RETRY_DELAY_SECONDS)
        else:
            logger.error(f"[TIMESERIES] All {MAX_RETRIES} attempts for coin_id={coin_id} time-series have failed.")

    def _init_lunarcrush_timeseries_table(self):
        """
        Creates the 'lunarcrush_timeseries' table if it doesn't exist, to store
        historical data from the time-series endpoint.
        """
        conn = sqlite3.connect(self.db_file)
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
            logger.exception(f"[TIMESERIES] Error creating table 'lunarcrush_timeseries': {e}")
        finally:
            conn.close()

    def _store_timeseries_records(self, coin_id: str, records: list):
        """
        Insert each data record from time-series into 'lunarcrush_timeseries'.
        Each record typically includes: time, open, close, volume_24h, sentiment, galaxy_score, alt_rank, etc.
        """
        conn = sqlite3.connect(self.db_file)
        try:
            c = conn.cursor()
            for rec in records:
                timestamp       = rec.get("time", 0)
                open_price      = rec.get("open", 0.0)
                close_price     = rec.get("close", 0.0)
                high_price      = rec.get("high", 0.0)
                low_price       = rec.get("low", 0.0)
                volume_24h      = rec.get("volume_24h", 0.0)
                market_cap      = rec.get("market_cap", 0.0)
                sentiment       = rec.get("sentiment", 0.0)
                spam            = rec.get("spam", 0.0)  # if you want to store spam too, you can
                galaxy_score    = rec.get("galaxy_score", 0.0)
                alt_rank        = rec.get("alt_rank", 0)
                volatility      = rec.get("volatility", 0.0)
                interactions    = rec.get("interactions", 0.0)
                social_dom      = rec.get("social_dominance", 0.0)

                c.execute("""
                    INSERT INTO lunarcrush_timeseries (
                        coin_id, timestamp, open_price, close_price, high_price, low_price,
                        volume_24h, market_cap, sentiment, spam, galaxy_score, alt_rank,
                        volatility, interactions, social_dominance
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
                    social_dom
                ])
            conn.commit()
            logger.info(f"[TIMESERIES] Inserted {len(records)} rows for coin_id={coin_id} into 'lunarcrush_timeseries'.")
        except Exception as e:
            logger.exception(f"[TIMESERIES] Error storing time-series records for coin_id={coin_id}: {e}")
        finally:
            conn.close()


# ------------------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    fetcher = LunarCrushFetcher()

    # 1) Fetch snapshot data => stored in 'lunarcrush_data'
    fetcher.fetch_snapshot_data()

    # 2) Then fetch time-series data for all coin IDs found in 'lunarcrush_data'.
    #    By default: bucket=hour, interval=1w => last 1 week of hourly data
    fetcher.fetch_time_series_for_tracked_coins(bucket="hour", interval="1w")
