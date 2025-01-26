#!/usr/bin/env python3
# =============================================================================
# FILE: fetch_lunarcrush.py
# =============================================================================
"""
fetch_lunarcrush.py

An enhanced, class-based script that:
  1) Offers two snapshot methods:
     a) fetch_snapshot_data_filtered(...) => uses the 'filter_symbols' approach
     b) fetch_snapshot_data_all_coins(...) => fetches all coins (no filter)
  2) Fetches time-series data from /public/coins/<coin_id>/time-series/v2
     in 'lunarcrush_timeseries'.
  3) Spot-check entire 'lunarcrush_data' table for top N coins across the market.
  4) A separate method to backfill time-series specifically for those top N coins.

Usage:
    python fetch_lunarcrush.py

Environment:
    - LUNARCRUSH_API_KEY: For snapshot endpoints
    - LUNARCRUSH_BEARER_TOKEN: For time-series if required.

Performance / Data-Volume Plan:
- Time-series data can grow quickly. Consider archiving older data or limiting to best coins.
"""

import os
import requests
import logging
import sqlite3
import time
import math
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------------------
# Example Toggles / Filters
# ------------------------------------------------------------------------------
filter_symbols = [
    "ETH", "XBT", "SOL", "ADA", "XRP", "LTC", "DOT", "MATIC", "LINK", "ATOM",
    "XMR", "TRX", "SHIB", "AVAX", "UNI", "APE", "FIL", "AAVE", "SAND", "CHZ",
    "GRT", "CRV", "COMP", "ALGO", "NEAR", "LDO", "XTZ", "EGLD", "KSM"
]

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
DB_FILE = "trades.db"

# Rate-limit ~10 calls/min => 6s each
REQUEST_SLEEP_SECONDS = 6

class LunarCrushFetcher:
    """
    A class that fetches LunarCrush data (snapshot & time-series) into local DB.
    We have separate methods for (a) all coins vs. (b) filtered coins.
    """

    def __init__(self, db_file: str = DB_FILE):
        self.db_file = db_file
        load_dotenv()

        self.API_KEY = os.getenv("LUNARCRUSH_API_KEY", "")
        if not self.API_KEY:
            logger.warning("No LUNARCRUSH_API_KEY set. Snapshot calls may fail.")

        self.BEARER_TOKEN = os.getenv("LUNARCRUSH_BEARER_TOKEN", "")
        if not self.BEARER_TOKEN:
            logger.warning("No LUNARCRUSH_BEARER_TOKEN set. Time-series calls may fail if required.")

    # --------------------------------------------------------------------------
    # (A) Snapshot: "filtered"
    # --------------------------------------------------------------------------
    def fetch_snapshot_data_filtered(self, limit: int = 100):
        """
        Uses /public/coins/list/v2 => includes only 'filter_symbols'.
        This replicates your original logic that references 'filter_symbols'.
        """
        if not self.API_KEY:
            logger.error("No LUNARCRUSH_API_KEY => abort snapshot fetch.")
            return

        base_url = "https://lunarcrush.com/api4/public/coins/list/v2"
        params = {
            "key": self.API_KEY,
            "limit": limit
        }

        attempt=0
        while attempt<MAX_RETRIES:
            try:
                logger.info(f"[SNAPSHOT-FILTERED] GET => {base_url}, params={params}, attempt={attempt+1}")
                resp = requests.get(base_url, params=params, timeout=10)
                resp.raise_for_status()

                data = resp.json()
                coins_list = data.get("data", [])
                logger.info(f"[SNAPSHOT-FILTERED] Fetched {len(coins_list)} coins total.")

                # filter by 'filter_symbols'
                filter_upper = [s.upper() for s in filter_symbols]
                coins_list = [
                    c for c in coins_list
                    if c.get("symbol","").upper() in filter_upper
                ]
                logger.info(f"[SNAPSHOT-FILTERED] After filter, we have {len(coins_list)} coins.")

                self._init_lunarcrush_data_table()
                self._store_snapshot_records(coins_list)
                break
            except requests.exceptions.HTTPError as e:
                attempt+=1
                logger.error(f"[SNAPSHOT-FILTERED] HTTP Error => {e}, sleep={RETRY_DELAY_SECONDS}")
                time.sleep(RETRY_DELAY_SECONDS)
            except Exception as e:
                attempt+=1
                logger.exception(f"[SNAPSHOT-FILTERED] Error => {e}, sleep={RETRY_DELAY_SECONDS}")
                time.sleep(RETRY_DELAY_SECONDS)
        else:
            logger.error("[SNAPSHOT-FILTERED] all attempts failed => gave up.")

    # --------------------------------------------------------------------------
    # (B) Snapshot: "all_coins"
    # --------------------------------------------------------------------------
    def fetch_snapshot_data_all_coins(self, limit: int = 500):
        """
        Another method to fetch ALL coins from /public/coins/list/v2, ignoring filter_symbols.
        For a truly wide net, you might set limit=500 or 1000 if your plan allows.
        """
        if not self.API_KEY:
            logger.error("No LUNARCRUSH_API_KEY => abort snapshot fetch.")
            return

        base_url = "https://lunarcrush.com/api4/public/coins/list/v2"
        params = {
            "key": self.API_KEY,
            "limit": limit
        }

        attempt=0
        while attempt<MAX_RETRIES:
            try:
                logger.info(f"[SNAPSHOT-ALL] GET => {base_url}, params={params}, attempt={attempt+1}")
                resp = requests.get(base_url, params=params, timeout=10)
                resp.raise_for_status()

                data = resp.json()
                coins_list = data.get("data", [])
                logger.info(f"[SNAPSHOT-ALL] Fetched {len(coins_list)} coins (unfiltered).")

                self._init_lunarcrush_data_table()
                self._store_snapshot_records(coins_list)
                break
            except requests.exceptions.HTTPError as e:
                attempt+=1
                logger.error(f"[SNAPSHOT-ALL] HTTP Error => {e}, sleep={RETRY_DELAY_SECONDS}")
                time.sleep(RETRY_DELAY_SECONDS)
            except Exception as e:
                attempt+=1
                logger.exception(f"[SNAPSHOT-ALL] Error => {e}, sleep={RETRY_DELAY_SECONDS}")
                time.sleep(RETRY_DELAY_SECONDS)
        else:
            logger.error("[SNAPSHOT-ALL] all attempts failed => gave up.")


    def _init_lunarcrush_data_table(self):
        """
        Creates or ensures 'lunarcrush_data' if not exists.
        """
        conn = sqlite3.connect(DB_FILE)
        try:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS lunarcrush_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lunarcrush_id INTEGER,
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
            logger.exception(f"[SNAPSHOT] Error creating 'lunarcrush_data': {e}")
        finally:
            conn.close()

    def _store_snapshot_records(self, coins_list: List[dict]):
        """
        Insert into 'lunarcrush_data'.
        """
        now_ts = int(time.time())
        conn = sqlite3.connect(self.db_file)
        try:
            c=conn.cursor()
            for coin in coins_list:
                # parse all fields
                lc_id = coin.get("id",None)
                symbol = coin.get("symbol","")
                name   = coin.get("name","")

                price = coin.get("price",0.0)
                price_btc = coin.get("price_btc",0.0)

                volume_24h = coin.get("volume_24h",0.0)
                volatility = coin.get("volatility",0.0)
                circ_supply= coin.get("circulating_supply",0.0)
                max_supply= coin.get("max_supply",0.0)

                pct1h= coin.get("percent_change_1h",0.0)
                pct24= coin.get("percent_change_24h",0.0)
                pct7d= coin.get("percent_change_7d",0.0)
                pct30= coin.get("percent_change_30d",0.0)

                mc= coin.get("market_cap",0.0)
                mc_rank= coin.get("market_cap_rank",999999)
                mdom= coin.get("market_dominance",0.0)
                mdom_prev= coin.get("market_dominance_prev",0.0)

                socvol_24 = coin.get("social_volume_24h",0.0)
                inter_24  = coin.get("interactions_24h",0.0)
                soc_dom   = coin.get("social_dominance",0.0)

                gal_score = coin.get("galaxy_score",0.0)
                gal_prev  = coin.get("galaxy_score_previous",0.0)
                alt_rank  = coin.get("alt_rank",0)
                alt_prev  = coin.get("alt_rank_previous",0)

                sentiment= coin.get("sentiment",0.0)
                cats     = coin.get("categories","")
                topic    = coin.get("topic","")
                logo     = coin.get("logo","")

                c.execute("""
                    INSERT INTO lunarcrush_data (
                      lunarcrush_id, symbol, name,
                      price, price_btc,
                      volume_24h, volatility,
                      circulating_supply, max_supply,
                      percent_change_1h, percent_change_24h, percent_change_7d, percent_change_30d,
                      market_cap, market_cap_rank,
                      market_dominance, market_dominance_prev,
                      social_volume_24h, interactions_24h, social_dominance,
                      galaxy_score, galaxy_score_previous,
                      alt_rank, alt_rank_previous,
                      sentiment, categories, topic, logo,
                      inserted_at
                    )
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, [
                    lc_id, symbol, name,
                    price, price_btc,
                    volume_24h, volatility,
                    circ_supply, max_supply,
                    pct1h, pct24, pct7d, pct30,
                    mc, mc_rank,
                    mdom, mdom_prev,
                    socvol_24, inter_24, soc_dom,
                    gal_score, gal_prev,
                    alt_rank, alt_prev,
                    sentiment, cats, topic, logo,
                    now_ts
                ])
            conn.commit()
            logger.info(f"[SNAPSHOT] Inserted {len(coins_list)} rows => 'lunarcrush_data'")
        except Exception as e:
            logger.exception(f"[SNAPSHOT] store_snapshot => {e}")
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # Time-series fetch
    # --------------------------------------------------------------------------
    def fetch_time_series_for_tracked_coins(
        self,
        bucket="hour",
        interval="1w",
        start: Optional[int]=None,
        end: Optional[int]=None
    ):
        """
        For each coin in 'lunarcrush_data' => calls _fetch_lunarcrush_time_series =>
        you can remove the 'filter_symbols' approach here, we want all from the DB.
        """
        conn=sqlite3.connect(self.db_file)
        try:
            c=conn.cursor()
            q="""
            SELECT DISTINCT lunarcrush_id, symbol
            FROM lunarcrush_data
            WHERE lunarcrush_id IS NOT NULL
            """
            rows=c.execute(q).fetchall()
        except Exception as e:
            logger.exception(f"[TIMESERIES] => {e}")
            return
        finally:
            conn.close()

        if not rows:
            logger.info("[TIMESERIES] No coins => skip.")
            return

        logger.info(f"[TIMESERIES] => found {len(rows)} coin IDs => let's fetch")

        for idx, (cid, sym) in enumerate(rows, start=1):
            time.sleep(REQUEST_SLEEP_SECONDS)
            logger.info(
                f"[TIMESERIES] => {idx}/{len(rows)} => coin_id={cid}, sym={sym}, bucket={bucket}, interval={interval}"
            )
            self._fetch_lunarcrush_time_series(
                str(cid),
                bucket=bucket,
                interval=interval,
                start=start,
                end=end
            )

    def _fetch_lunarcrush_time_series(
        self,
        coin_id: str,
        bucket="hour",
        interval="1w",
        start: Optional[int]=None,
        end: Optional[int]=None
    ):
        base_url=f"https://lunarcrush.com/api4/public/coins/{coin_id}/time-series/v2"
        headers={}
        if self.BEARER_TOKEN:
            headers["Authorization"] = f"Bearer {self.BEARER_TOKEN}"

        params={
            "bucket": bucket,
            "interval": interval
        }
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"]   = end

        attempt=0
        while attempt<MAX_RETRIES:
            try:
                logger.info(f"[TIMESERIES] GET => {base_url}, params={params}, attempt={attempt+1}")
                resp = requests.get(base_url, headers=headers, params=params, timeout=10)
                resp.raise_for_status()

                data = resp.json()
                recs=data.get("data",[])
                logger.info(f"[TIMESERIES] coin_id={coin_id}, got {len(recs)} records")

                if recs:
                    self.init_lunarcrush_timeseries_table()
                    self._store_timeseries_records(coin_id, recs)
                break
            except requests.exceptions.HTTPError as e:
                attempt+=1
                logger.error(f"[TIMESERIES] HTTP => {e}. sleep={RETRY_DELAY_SECONDS}")
                time.sleep(RETRY_DELAY_SECONDS)
            except Exception as e:
                attempt+=1
                logger.exception(f"[TIMESERIES] => {e}. sleep={RETRY_DELAY_SECONDS}")
                time.sleep(RETRY_DELAY_SECONDS)
        else:
            logger.error(f"[TIMESERIES] All attempts fail => coin_id={coin_id}")

    @staticmethod
    def init_lunarcrush_timeseries_table():
        conn = sqlite3.connect(DB_FILE)
        try:
            c=conn.cursor()
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
            logger.exception(f"[TIMESERIES] create timeseries => {e}")
        finally:
            conn.close()

    def _store_timeseries_records(self, coin_id:str, records: list):
        conn=sqlite3.connect(self.db_file)
        try:
            c=conn.cursor()
            for rec in records:
                timestamp=rec.get("time",0)
                open_p = rec.get("open",0.0)
                close_p= rec.get("close",0.0)
                high_p = rec.get("high",0.0)
                low_p  = rec.get("low",0.0)
                vol24  = rec.get("volume_24h",0.0)
                mc     = rec.get("market_cap",0.0)
                sent   = rec.get("sentiment",0.0)
                sp     = rec.get("spam",0.0)
                gal_s  = rec.get("galaxy_score",0.0)
                alt_r  = rec.get("alt_rank",0)
                volty  = rec.get("volatility",0.0)
                inter  = rec.get("interactions",0.0)
                socdom = rec.get("social_dominance",0.0)

                c.execute("""
                INSERT INTO lunarcrush_timeseries (
                    coin_id, timestamp,
                    open_price, close_price, high_price, low_price,
                    volume_24h, market_cap,
                    sentiment, spam,
                    galaxy_score, alt_rank, volatility,
                    interactions, social_dominance
                )
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, [
                    coin_id, timestamp,
                    open_p, close_p, high_p, low_p,
                    vol24, mc,
                    sent, sp,
                    gal_s, alt_r, volty,
                    inter, socdom
                ])
            conn.commit()
            logger.info(f"[TIMESERIES] coin_id={coin_id}, inserted {len(records)} rows.")
        except Exception as e:
            logger.exception(f"[TIMESERIES] store => coin_id={coin_id}, {e}")
        finally:
            conn.close()

    # ==========================================================================
    # (A) Spot-check entire market: "spot_check_entire_market"
    # ==========================================================================
    @staticmethod
    def spot_check_entire_market(top_n: int=10) -> List[Tuple[str,float]]:
        """
        This method does NOT rely on 'filter_symbols'. It expects you to have
        run fetch_snapshot_data_all_coins(...) so 'lunarcrush_data' has all coins.
        Then we pick top N by a rank_score formula.

        rank_score = (0.5*gscore) + (0.2*(market_cap/1e9)) + (0.2*%24h) + (0.05*senti) - (0.05*alt_rank)
        The higher => the better.

        Returns => list of (coin_id, rank_score).
        """

        conn = sqlite3.connect(DB_FILE)
        out=[]
        try:
            c=conn.cursor()
            # reading entire market:
            q="""
              SELECT
                symbol, lunarcrush_id,
                galaxy_score, alt_rank,
                market_cap, percent_change_24h,
                sentiment
              FROM lunarcrush_data
              WHERE lunarcrush_id IS NOT NULL
            """
            rows = c.execute(q).fetchall()
            if not rows:
                logger.warning("[SPOTCHECK-ALL] no rows => can't proceed.")
                return []

            temp=[]
            for (sym, cid, gscore, arank, mcap, pc24, senti) in rows:
                if not cid:
                    continue
                gscore = gscore or 0.0
                arank  = arank or 999999
                mcap   = mcap or 0.0
                pc24   = pc24 or 0.0
                senti  = senti or 0.0

                scaled_senti = senti*0.01  # if your sentiment is 0..100
                rank_score = (
                    (0.5*gscore)
                    + (0.2*(mcap/1e9))
                    + (0.2*pc24)
                    + (0.05*scaled_senti)
                    - (0.05*arank)
                )
                temp.append((str(cid), rank_score, sym))

            temp.sort(key=lambda x: x[1], reverse=True)
            top_temp = temp[:top_n]
            logger.info(f"[SPOTCHECK-ALL] top_n={top_n} => {top_temp}")
            out = [(c[0], c[1]) for c in top_temp]
        except Exception as e:
            logger.exception(f"[SPOTCHECK-ALL] => {e}")
        finally:
            conn.close()
        return out

    # ==========================================================================
    # (B) Backfill the top N from entire market
    # ==========================================================================
    def backfill_top_n_timeseries(
        self,
        top_coins: List[Tuple[str,float]],
        bucket="hour",
        interval="1w",
        start: Optional[int]=None,
        end: Optional[int]=None
    ):
        """
        We explicitly only do time-series for these top_coins.
        Each item => (coin_id, rank_score).
        """
        if not top_coins:
            logger.info("[BACKFILL-TOPN] no top coins => skip.")
            return

        self.init_lunarcrush_timeseries_table()

        for idx, (cid, rs) in enumerate(top_coins, start=1):
            logger.info(f"[BACKFILL-TOPN] {idx}/{len(top_coins)} => coin_id={cid}, rank_score={rs}")
            time.sleep(REQUEST_SLEEP_SECONDS)
            self._fetch_lunarcrush_time_series(
                coin_id=cid,
                bucket=bucket,
                interval=interval,
                start=start,
                end=end
            )

# ------------------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------------------
if __name__=="__main__":
    fetcher = LunarCrushFetcher(DB_FILE)

    # 1) Option A => fetch snapshot data with filters
    # fetcher.fetch_snapshot_data_filtered(limit=100)
    fetcher.init_lunarcrush_timeseries_table()

    # or Option B => fetch entire market
    fetcher.fetch_snapshot_data_all_coins(limit=100)

    # 2) Now we have 'lunarcrush_data' filled. Let's spot-check entire market:
    top_n_coins = fetcher.spot_check_entire_market(top_n=30)
    logger.info(f"Top 10 across entire market => {top_n_coins}")

    # 3) We'll backfill just those top 10 coins:
    fetcher.backfill_top_n_timeseries(top_n_coins, bucket="hour", interval="1w")
