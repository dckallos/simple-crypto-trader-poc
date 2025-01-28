#!/usr/bin/env python3
# =============================================================================
# FILE: fetch_lunarcrush.py
# =============================================================================
"""
fetch_lunarcrush.py

A single script that:
  1) Fetches snapshot data (/public/coins/list/v2) => upserts into the existing 'lunarcrush_data' table,
     adding columns if needed. The table structure extends what's defined in db.py with extra columns:
       - price_btc, circulating_supply, max_supply, market_dominance, market_dominance_prev,
         galaxy_score_previous, alt_rank_previous, blockchains (stored as JSON), ...
  2) Fetches time-series data (/public/coins/<symbol or ID>/time-series/v2) => upserts into
     the existing 'lunarcrush_timeseries' table, with additional columns if needed, like:
       - market_dominance, circulating_supply, contributors_active, etc.
  3) Chunk-based backfill from (now - N months) to present in ~30-day intervals.
  4) "Spot-check" a naive rank => top N => partial backfill if desired.

Environment:
    - LUNARCRUSH_API_KEY: For snapshot endpoints
    - LUNARCRUSH_BEARER_TOKEN: For time-series (if required)
"""

import os
import sys
import time
import logging
import requests
import sqlite3
import argparse
import json
from typing import Optional, List, Tuple

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DB_FILE = "trades.db"    # Reuses your db.py path
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
REQUEST_SLEEP_SECONDS = 6

# If you want to snapshot only certain symbols:
filter_symbols = [
    "ETH", "XBT", "SOL", "ADA", "XRP", "LTC", "DOT", "MATIC", "LINK", "ATOM",
    "XMR", "TRX", "SHIB", "AVAX", "UNI", "APE", "FIL", "AAVE", "SAND", "CHZ",
    "GRT", "CRV", "COMP", "ALGO", "NEAR", "LDO", "XTZ", "EGLD", "KSM"
]


class LunarCrushFetcher:
    """
    Fetches LunarCrush snapshot & time-series data, storing into the existing
    'lunarcrush_data' and 'lunarcrush_timeseries' tables. We extend them with new columns
    if needed for additional data (like price_btc, circulating_supply, etc.).
    """

    def __init__(self, db_file: str = DB_FILE):
        self.db_file = db_file
        load_dotenv()

        self.API_KEY = os.getenv("LUNARCRUSH_API_KEY", "")
        if not self.API_KEY:
            logger.warning("No LUNARCRUSH_API_KEY => snapshot calls may fail or skip.")

        self.BEARER_TOKEN = os.getenv("LUNARCRUSH_BEARER_TOKEN", "")
        if not self.BEARER_TOKEN:
            logger.warning("No LUNARCRUSH_BEARER_TOKEN => time-series calls may fail.")

        # Attempt to add new columns if they do not exist
        self._add_new_columns_to_lunarcrush_data()
        self._add_new_columns_to_lunarcrush_timeseries()

    # --------------------------------------------------------------------------
    # Extend existing tables if needed
    # --------------------------------------------------------------------------
    def _add_new_columns_to_lunarcrush_data(self):
        """
        We unify the table from db.py with new columns from the snapshot JSON:
        - price_btc, circulating_supply, max_supply, market_dominance, market_dominance_prev
        - galaxy_score_previous, alt_rank_previous
        - blockchains => stored as JSON text
        - etc.
        We'll do naive try/except for each new column.
        """
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        new_columns = [
            ("price_btc", "REAL"),
            ("circulating_supply", "REAL"),
            ("max_supply", "REAL"),
            ("market_dominance", "REAL"),
            ("market_dominance_prev", "REAL"),
            ("galaxy_score_previous", "REAL"),
            ("alt_rank_previous", "INTEGER"),
            ("blockchains", "TEXT"),    # storing as JSON text
        ]
        for col, col_type in new_columns:
            try:
                c.execute(f"ALTER TABLE lunarcrush_data ADD COLUMN {col} {col_type}")
                logger.info(f"[Schema] Added column={col} to lunarcrush_data.")
            except sqlite3.OperationalError:
                # likely means it already exists
                pass
        conn.commit()
        conn.close()

    def _add_new_columns_to_lunarcrush_timeseries(self):
        """
        We unify the table from db.py with new columns from the timeseries JSON:
        - market_dominance, circulating_supply, contributors_active, etc.
        """
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        new_columns = [
            ("market_dominance", "REAL"),
            ("circulating_supply", "REAL"),
            ("contributors_active", "REAL"),
            ("contributors_created", "REAL"),
            ("posts_active", "REAL"),
            ("posts_created", "REAL"),
        ]
        for col, col_type in new_columns:
            try:
                c.execute(f"ALTER TABLE lunarcrush_timeseries ADD COLUMN {col} {col_type}")
                logger.info(f"[Schema] Added column={col} to lunarcrush_timeseries.")
            except sqlite3.OperationalError:
                pass
        conn.commit()
        conn.close()

    # --------------------------------------------------------------------------
    # Snapshot
    # --------------------------------------------------------------------------
    def fetch_snapshot_data_filtered(self, limit: int=100):
        """
        /public/coins/list/v2 => only those symbols in filter_symbols (uppercase).
        We'll store or upsert in 'lunarcrush_data' for each coin found, with new columns where needed.
        """
        if not self.API_KEY:
            logger.error("No LUNARCRUSH_API_KEY => cannot fetch snapshots.")
            return

        base_url = "https://lunarcrush.com/api4/public/coins/list/v2"
        params = {"key": self.API_KEY, "limit": limit}
        attempt=0
        while attempt<MAX_RETRIES:
            try:
                logger.info(f"[SNAPSHOT-FILTERED] attempt={attempt+1}, limit={limit}")
                resp = requests.get(base_url, params=params, timeout=10)
                resp.raise_for_status()

                data = resp.json()
                all_coins = data.get("data", [])
                logger.info(f"[SNAPSHOT-FILTERED] => {len(all_coins)} coins total from endpoint.")

                # local filter
                filter_up = [s.upper() for s in filter_symbols]
                filtered_coins = [
                    coin for coin in all_coins
                    if coin.get("symbol","").upper() in filter_up
                ]
                logger.info(f"[SNAPSHOT-FILTERED] => after filtering => {len(filtered_coins)} remain.")
                self._store_snapshot_records(filtered_coins)
                break
            except requests.exceptions.RequestException as e:
                attempt +=1
                logger.error(f"[SNAPSHOT-FILTERED] => {e}, sleeping={RETRY_DELAY_SECONDS}")
                time.sleep(RETRY_DELAY_SECONDS)
        else:
            logger.error("[SNAPSHOT-FILTERED] all attempts failed => giving up.")

    def fetch_snapshot_data_all_coins(self, limit: int=500):
        """
        /public/coins/list/v2 => fetch up to 'limit' coins, ignoring filter_symbols.
        We'll store them in 'lunarcrush_data'.
        """
        if not self.API_KEY:
            logger.error("No LUNARCRUSH_API_KEY => cannot fetch snapshots.")
            return

        base_url = "https://lunarcrush.com/api4/public/coins/list/v2"
        params = {"key": self.API_KEY, "limit": limit}
        attempt=0
        while attempt<MAX_RETRIES:
            try:
                logger.info(f"[SNAPSHOT-ALL] attempt={attempt+1}, limit={limit}")
                resp = requests.get(base_url, params=params, timeout=10)
                resp.raise_for_status()

                data = resp.json()
                coins_list = data.get("data", [])
                logger.info(f"[SNAPSHOT-ALL] => {len(coins_list)} coins from endpoint, unfiltered.")
                self._store_snapshot_records(coins_list)
                break
            except requests.exceptions.RequestException as e:
                attempt +=1
                logger.error(f"[SNAPSHOT-ALL] => {e}, sleeping={RETRY_DELAY_SECONDS}")
                time.sleep(RETRY_DELAY_SECONDS)
        else:
            logger.error("[SNAPSHOT-ALL] all attempts failed => giving up.")

    def _store_snapshot_records(self, coins_list: List[dict]):
        """
        Insert or update rows in 'lunarcrush_data', extending columns with new fields:
          (price_btc, circulating_supply, max_supply, market_dominance, market_dominance_prev,
           galaxy_score_previous, alt_rank_previous, blockchains as JSON, etc.)
        We'll do an 'INSERT OR REPLACE' approach if you'd like, or just an INSERT.
        """
        if not coins_list:
            logger.info("[SNAPSHOT] => no coins => skip.")
            return

        now_ts = int(time.time())
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()

        inserted=0
        for coin in coins_list:
            lunar_id = coin.get("id", None)
            symbol   = coin.get("symbol","")
            name     = coin.get("name","")
            price    = coin.get("price",0.0)
            price_btc= coin.get("price_btc", 0.0)
            vol24    = coin.get("volume_24h",0.0)
            volty    = coin.get("volatility",0.0)
            circ     = coin.get("circulating_supply",0.0)
            maxs     = coin.get("max_supply",0.0)
            pct1h    = coin.get("percent_change_1h",0.0)
            pct24    = coin.get("percent_change_24h",0.0)
            pct7d    = coin.get("percent_change_7d",0.0)
            pct30    = coin.get("percent_change_30d",0.0)  # or 0 if missing
            mcap     = coin.get("market_cap",0.0)
            mcap_rank= coin.get("market_cap_rank",999999)
            inter24  = coin.get("interactions_24h",0.0)
            socvol24 = coin.get("social_volume_24h",0.0)
            soc_dom  = coin.get("social_dominance",0.0)
            mk_dom   = coin.get("market_dominance",0.0)
            mk_dom_prv=coin.get("market_dominance_prev",0.0)
            gal_s    = coin.get("galaxy_score",0.0)
            gal_s_prv= coin.get("galaxy_score_previous",0.0)
            alt_r    = coin.get("alt_rank",999999)
            alt_r_prv= coin.get("alt_rank_previous",999999)
            senti    = coin.get("sentiment",0.0)

            cats      = coin.get("categories","")  # comma-delimited
            topic     = coin.get("topic","")
            logo      = coin.get("logo","")

            # blockchains => store as JSON text
            blocks = coin.get("blockchains",[])
            blocks_json = json.dumps(blocks, separators=(",",":"))  # compact

            # We'll do an upsert approach:
            # If you want to handle collisions by symbol or something else, adapt accordingly.
            c.execute("""
                INSERT OR REPLACE INTO lunarcrush_data (
                    id, timestamp, lunarcrush_id, symbol, name,
                    price, price_btc, volume_24h, volatility,
                    circulating_supply, max_supply,
                    percent_change_1h, percent_change_24h, percent_change_7d, percent_change_30d,
                    market_cap, market_cap_rank,
                    interactions_24h, social_volume_24h, social_dominance,
                    market_dominance, market_dominance_prev,
                    galaxy_score, galaxy_score_previous,
                    alt_rank, alt_rank_previous,
                    sentiment, categories, topic, logo,
                    blockchains
                )
                VALUES (
                    NULL, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?, ?, ?,
                    ?
                )
            """, [
                now_ts, lunar_id, symbol, name,
                price, price_btc, vol24, volty,
                circ, maxs,
                pct1h, pct24, pct7d, pct30,
                mcap, mcap_rank,
                inter24, socvol24, soc_dom,
                mk_dom, mk_dom_prv,
                gal_s, gal_s_prv,
                alt_r, alt_r_prv,
                senti, cats, topic, logo,
                blocks_json
            ])
            inserted+=1

        conn.commit()
        conn.close()
        logger.info(f"[SNAPSHOT] Inserted/Upserted {inserted} rows in 'lunarcrush_data'.")

    # --------------------------------------------------------------------------
    # Time-series
    # --------------------------------------------------------------------------
    def fetch_time_series_for_tracked_coins(self, bucket="hour", interval="1w",
                                            start: Optional[int]=None, end: Optional[int]=None):
        symbols = self._load_symbols_from_db()
        if not symbols:
            logger.info("[TIMESERIES] no symbols => skip.")
            return

        logger.info(f"[TIMESERIES] => found {len(symbols)} distinct symbols => fetching time-series.")
        for idx, sym in enumerate(symbols, start=1):
            time.sleep(REQUEST_SLEEP_SECONDS)
            logger.info(f"[TIMESERIES] => {idx}/{len(symbols)} => symbol={sym}, bucket={bucket}, interval={interval}")
            self._fetch_lunarcrush_time_series(
                coin_id=sym,
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
        if not self.BEARER_TOKEN:
            logger.warning("No LUNARCRUSH_BEARER_TOKEN => cannot fetch time-series.")
            return

        base_url = f"https://lunarcrush.com/api4/public/coins/{coin_id}/time-series/v2"
        headers = {"Authorization": f"Bearer {self.BEARER_TOKEN}"}
        params = {"bucket": bucket, "interval": interval}
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end

        attempt=0
        while attempt<MAX_RETRIES:
            try:
                logger.info(f"[TIMESERIES] GET => {base_url}, coin_id={coin_id}, attempt={attempt+1}")
                r = requests.get(base_url, headers=headers, params=params, timeout=10)
                r.raise_for_status()

                j = r.json()
                recs = j.get("data", [])
                logger.info(f"[TIMESERIES] symbol={coin_id}, got {len(recs)} records")
                if recs:
                    self._store_timeseries_records(coin_id, recs)
                break
            except requests.exceptions.RequestException as e:
                attempt+=1
                logger.error(f"[TIMESERIES] => {e} => sleep={RETRY_DELAY_SECONDS}")
                time.sleep(RETRY_DELAY_SECONDS)
        else:
            logger.error(f"[TIMESERIES] all attempts failed => coin_id={coin_id}")

    def _store_timeseries_records(self, coin_id: str, records: List[dict]):
        """
        Insert (or upsert) each row into 'lunarcrush_timeseries', with new columns:
          - market_dominance, circulating_supply, contributors_active, contributors_created
          - posts_active, posts_created
        """
        if not records:
            return

        conn=sqlite3.connect(self.db_file)
        c = conn.cursor()
        inserted=0

        for row in records:
            ts         = row.get("time", 0)
            open_p     = row.get("open", 0.0)
            close_p    = row.get("close", 0.0)
            high_p     = row.get("high", 0.0)
            low_p      = row.get("low", 0.0)
            vol24      = row.get("volume_24h", 0.0)
            mcap       = row.get("market_cap", 0.0)
            mk_dom     = row.get("market_dominance", 0.0)
            circ_sup   = row.get("circulating_supply", 0.0)
            senti      = row.get("sentiment", 0.0)
            spam       = row.get("spam", 0.0)
            gal_s      = row.get("galaxy_score", 0.0)
            volty      = row.get("volatility", 0.0)
            alt_r      = row.get("alt_rank", 999999)
            contrib_act= row.get("contributors_active", 0.0)
            contrib_cre= row.get("contributors_created", 0.0)
            posts_act  = row.get("posts_active", 0.0)
            posts_cre  = row.get("posts_created", 0.0)
            inter      = row.get("interactions", 0.0)
            soc_dom    = row.get("social_dominance", 0.0)

            # Insert or replace
            c.execute("""
              INSERT OR REPLACE INTO lunarcrush_timeseries (
                id, coin_id, timestamp,
                open_price, close_price, high_price, low_price,
                volume_24h, market_cap, market_dominance, circulating_supply,
                sentiment, spam, galaxy_score, volatility, alt_rank,
                contributors_active, contributors_created, posts_active, posts_created,
                interactions, social_dominance
              )
              VALUES (
                NULL, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?
              )
            """, [
                coin_id, ts,
                open_p, close_p, high_p, low_p,
                vol24, mcap, mk_dom, circ_sup,
                senti, spam, gal_s, volty, alt_r,
                contrib_act, contrib_cre, posts_act, posts_cre,
                inter, soc_dom
            ])
            inserted+=1

        conn.commit()
        conn.close()
        logger.info(f"[TIMESERIES] coin_id={coin_id} => upserted {inserted} rows.")

    def _load_symbols_from_db(self) -> List[str]:
        out=[]
        conn=sqlite3.connect(self.db_file)
        try:
            c=conn.cursor()
            q="""
            SELECT DISTINCT UPPER(symbol)
            FROM lunarcrush_data
            WHERE symbol IS NOT NULL AND symbol != ''
            """
            rows = c.execute(q).fetchall()
            out=[r[0] for r in rows if r[0]]
        except Exception as e:
            logger.exception(f"[TIMESERIES] load_symbols => {e}")
        finally:
            conn.close()
        return out

    # --------------------------------------------------------------------------
    # chunk-based backfill
    # --------------------------------------------------------------------------
    def backfill_coins(
        self,
        coin_ids: List[str],
        months: int=12,
        bucket="hour",
        interval="1w"
    ):
        if not coin_ids:
            logger.info("[BACKFILL] => no coin_ids => skip.")
            return
        if not self.BEARER_TOKEN:
            logger.warning("[BACKFILL] => no token => cannot fetch time-series.")
            return

        logger.info(f"[BACKFILL] => chunk-based => {len(coin_ids)} symbols, months={months}")
        now_ts = int(time.time())
        chunk_seconds = 30 * 86400
        total_back_secs = months*chunk_seconds

        for idx, sym in enumerate(coin_ids, start=1):
            logger.info(f"[BACKFILL] => symbol={sym} => {idx}/{len(coin_ids)}, months={months}")
            start_ts = now_ts - total_back_secs
            while start_ts < now_ts:
                chunk_end = min(start_ts + chunk_seconds, now_ts)
                logger.info(f"[BACKFILL] chunk => {sym}, start={start_ts}, end={chunk_end}")
                self._fetch_lunarcrush_time_series(
                    coin_id=sym,
                    bucket=bucket,
                    interval=interval,
                    start=start_ts,
                    end=chunk_end
                )
                time.sleep(REQUEST_SLEEP_SECONDS)
                start_ts = chunk_end

    # --------------------------------------------------------------------------
    # Spot-check entire market => pick top N for partial backfill
    # --------------------------------------------------------------------------
    def spot_check_entire_market(self, top_n: int=10) -> List[str]:
        """
        We'll read from 'lunarcrush_data' => compute a naive rank_score => return top N symbols
        so we can do partial backfill. The formula is an example.
        """
        out=[]
        conn=sqlite3.connect(self.db_file)
        try:
            c=conn.cursor()
            rows=c.execute("""
            SELECT UPPER(symbol),
                   galaxy_score,
                   alt_rank,
                   market_cap,
                   percent_change_24h,
                   sentiment
            FROM lunarcrush_data
            WHERE symbol IS NOT NULL
            """).fetchall()
            if not rows:
                logger.warning("[SPOTCHECK] => no rows => can't proceed.")
                return []

            temp=[]
            for (sym, gsc, ar, mc, pc24, sent) in rows:
                if not sym:
                    continue
                gsc = gsc or 0.0
                ar  = ar or 999999
                mc  = mc or 0.0
                pc24= pc24 or 0.0
                sent= sent or 0.0

                # example rank => (0.5*gscore + 0.2*(mc/1e9) + 0.2*pc24 + 0.05*(sent/100) - 0.05*ar)
                scaled_senti = sent*0.01
                rank_score = (
                    0.5*gsc
                    + 0.2*(mc/1e9)
                    + 0.2*(pc24)
                    + 0.05*scaled_senti
                    - 0.05*ar
                )
                temp.append((sym, rank_score))

            temp.sort(key=lambda x: x[1], reverse=True)
            top_temp = temp[:top_n]
            out = [t[0] for t in top_temp]
            logger.info(f"[SPOTCHECK] => top_n={top_n}, {top_temp}")
        except Exception as e:
            logger.exception(f"[SPOTCHECK] => {e}")
        finally:
            conn.close()
        return out


def parse_args():
    p = argparse.ArgumentParser(description="Fetch from LunarCrush => existing db.py tables.")
    p.add_argument("--snapshot-filtered", action="store_true",
                   help="Fetch only filter_symbols from /coins/list/v2 => store in 'lunarcrush_data'.")
    p.add_argument("--snapshot-all", action="store_true",
                   help="Fetch up to 'limit' coins from /coins/list/v2 => store in 'lunarcrush_data'.")
    p.add_argument("--time-series", action="store_true",
                   help="Fetch time-series for each distinct symbol in 'lunarcrush_data'.")
    p.add_argument("--backfill-all", action="store_true",
                   help="Chunk-based backfill for all symbols found in 'lunarcrush_data'.")
    p.add_argument("--backfill-top", action="store_true",
                   help="Spot-check rank => pick top N => chunk-based backfill them.")
    p.add_argument("--backfill-configured", action="store_true",
                   help="Load config.yaml traded_pairs => parse the symbol => chunk-based backfill.")
    p.add_argument("--top-n", type=int, default=10,
                   help="Number of coins for --backfill-top. Default=10.")
    p.add_argument("--months", type=int, default=12,
                   help="Number of months to chunk-based backfill. Default=12 => ~1 year.")
    p.add_argument("--limit", type=int, default=100,
                   help="Limit for snapshot. Default=100.")
    return p.parse_args()

def main():
    args = parse_args()
    fetcher = LunarCrushFetcher(DB_FILE)

    # If you need to read config
    import os
    config_file = "config.yaml"
    traded_pairs=[]
    if os.path.exists(config_file):
        import yaml
        with open(config_file,"r") as f:
            c = yaml.safe_load(f)
        traded_pairs = c.get("traded_pairs",[])

    # Snapshots
    if args.snapshot_filtered:
        fetcher.fetch_snapshot_data_filtered(limit=args.limit)
    if args.snapshot_all:
        fetcher.fetch_snapshot_data_all_coins(limit=args.limit)

    # Time-series => no chunk => all symbols
    if args.time_series:
        fetcher.fetch_time_series_for_tracked_coins(bucket="hour", interval="1w")

    # chunk-based backfill
    if any([args.backfill_all, args.backfill_top, args.backfill_configured]):
        def load_all_symbols() -> List[str]:
            conn=sqlite3.connect(DB_FILE)
            out=[]
            try:
                c=conn.cursor()
                q="""
                SELECT DISTINCT UPPER(symbol)
                FROM lunarcrush_data
                WHERE symbol IS NOT NULL
                  AND symbol != ''
                """
                rows=c.execute(q).fetchall()
                out=[r[0] for r in rows if r[0]]
            except Exception as e:
                logger.exception(f"[BACKFILL-ALL] => {e}")
            finally:
                conn.close()
            return out

        def load_configured_symbols(tpairs:List[str]) -> List[str]:
            syms=[]
            for pair in tpairs:
                syms.append(pair.split("/")[0].upper())
            syms = list(set(syms))
            logger.info(f"[ConfiguredSymbols] => {syms}")
            return syms

        coin_ids=[]
        if args.backfill_all:
            coin_ids=load_all_symbols()
        elif args.backfill_top:
            coin_ids=fetcher.spot_check_entire_market(top_n=args.top_n)
        elif args.backfill_configured:
            coin_ids=load_configured_symbols(traded_pairs)

        if not coin_ids:
            logger.info("[BACKFILL] => no coin_ids => skip.")
        else:
            fetcher.backfill_coins(coin_ids, months=args.months, bucket="hour", interval="1w")


if __name__=="__main__":
    main()
