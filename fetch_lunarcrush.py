#!/usr/bin/env python3
# =============================================================================
# FILE: fetch_lunarcrush.py
# =============================================================================
"""
fetch_lunarcrush.py

A single script that:
  1) Fetches snapshot data (coins/list/v2) => stores in 'lunarcrush_data' (matching db.py columns).
  2) Fetches time-series data (/public/coins/<id>/time-series/v2) => stores in 'lunarcrush_timeseries'.
  3) Supports chunk-based backfill for older historical data, if you treat the 'coin_id' as a symbol
     or a numeric ID. Here, we store the symbol in 'coin_id' by default, matching the new db.py schema.
  4) Offers a "spot-check" rank to pick top N symbols if you want partial backfill.

Usage examples:
    python fetch_lunarcrush.py --snapshot-all
    python fetch_lunarcrush.py --snapshot-filtered
    python fetch_lunarcrush.py --time-series
    python fetch_lunarcrush.py --backfill-all --months=12
    python fetch_lunarcrush.py --backfill-top --top-n=10
    python fetch_lunarcrush.py --backfill-configured --months=6

Environment:
    - LUNARCRUSH_API_KEY: For snapshot endpoints
    - LUNARCRUSH_BEARER_TOKEN: For time-series endpoints, if required
"""

import os
import sys
import time
import logging
import requests
import sqlite3
import argparse
import yaml
from typing import Optional, List, Tuple

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DB_FILE = "trades.db"  # matches your db.py
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
    Fetches LunarCrush snapshot & time-series data, storing into 'lunarcrush_data'
    and 'lunarcrush_timeseries' as defined in db.py. Also provides chunk-based
    backfill and a spot-check rank function.
    """

    def __init__(self, db_file: str = DB_FILE):
        self.db_file = db_file
        load_dotenv()

        self.API_KEY = os.getenv("LUNARCRUSH_API_KEY", "")
        if not self.API_KEY:
            logger.warning("No LUNARCRUSH_API_KEY => snapshot calls may fail or skip.")

        self.BEARER_TOKEN = os.getenv("LUNARCRUSH_BEARER_TOKEN", "")
        if not self.BEARER_TOKEN:
            logger.warning("No LUNARCRUSH_BEARER_TOKEN => time-series calls may fail or skip.")

    # --------------------------------------------------------------------------
    # Snapshot: filtered vs all
    # --------------------------------------------------------------------------
    def fetch_snapshot_data_filtered(self, limit: int=100):
        """
        /public/coins/list/v2 => only those symbols in filter_symbols (uppercase).
        We'll store a row in 'lunarcrush_data' for each coin found.
        """
        if not self.API_KEY:
            logger.error("No LUNARCRUSH_API_KEY => cannot fetch snapshots.")
            return

        base_url = "https://lunarcrush.com/api4/public/coins/list/v2"
        params = {"key": self.API_KEY}
        attempt=0
        while attempt<MAX_RETRIES:
            try:
                logger.info(f"[SNAPSHOT-FILTERED] attempt={attempt+1}, limit={limit}")
                resp = requests.get(base_url, params=params, timeout=10)
                resp.raise_for_status()

                data = resp.json()
                all_coins = data.get("data", [])
                logger.info(f"[SNAPSHOT-FILTERED] Endpoint returned {len(all_coins)} coins total.")

                # local filter
                filter_up = [s.upper() for s in filter_symbols]
                coins_list = [
                    c for c in all_coins
                    if c.get("symbol","").upper() in filter_up
                ]
                logger.info(f"[SNAPSHOT-FILTERED] after local filter => {len(coins_list)} coins remain.")
                self._store_snapshot_records(coins_list)
                break
            except requests.exceptions.HTTPError as e:
                attempt +=1
                logger.error(f"[SNAPSHOT-FILTERED] HTTP => {e} => sleeping={RETRY_DELAY_SECONDS}")
                time.sleep(RETRY_DELAY_SECONDS)
            except Exception as e:
                attempt +=1
                logger.exception(f"[SNAPSHOT-FILTERED] => {e}")
                time.sleep(RETRY_DELAY_SECONDS)
        else:
            logger.error("[SNAPSHOT-FILTERED] all attempts failed => gave up.")

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
                logger.info(f"[SNAPSHOT-ALL] Endpoint returned {len(coins_list)} coins unfiltered.")
                self._store_snapshot_records(coins_list)
                break
            except requests.exceptions.HTTPError as e:
                attempt +=1
                logger.error(f"[SNAPSHOT-ALL] HTTP => {e} => sleeping={RETRY_DELAY_SECONDS}")
                time.sleep(RETRY_DELAY_SECONDS)
            except Exception as e:
                attempt +=1
                logger.exception(f"[SNAPSHOT-ALL] => {e}")
                time.sleep(RETRY_DELAY_SECONDS)
        else:
            logger.error("[SNAPSHOT-ALL] all attempts failed => gave up.")


    def _store_snapshot_records(self, coins_list: list):
        """
        Insert rows into `lunarcrush_data` using the columns from db.py:

        (timestamp, symbol, name, price, market_cap, volume_24h, volatility,
         percent_change_1h, percent_change_24h, percent_change_7d, percent_change_30d,
         social_volume_24h, interactions_24h, social_dominance, galaxy_score,
         alt_rank, sentiment, categories, topic, logo)
        """

        if not coins_list:
            logger.info("[SNAPSHOT] no coins => skip storing.")
            return

        now_ts = int(time.time())
        conn = sqlite3.connect(self.db_file)
        try:
            c = conn.cursor()
            inserted=0
            for coin in coins_list:
                symbol = coin.get("symbol","")
                name   = coin.get("name","")
                price  = coin.get("price",0.0)
                mcap   = coin.get("market_cap",0.0)
                vol24  = coin.get("volume_24h",0.0)
                volty  = coin.get("volatility",0.0)

                pct1h  = coin.get("percent_change_1h",0.0)
                pct24  = coin.get("percent_change_24h",0.0)
                pct7d  = coin.get("percent_change_7d",0.0)
                pct30  = coin.get("percent_change_30d",0.0)

                socvol_24 = coin.get("social_volume_24h",0.0)
                inter_24  = coin.get("interactions_24h",0.0)
                soc_dom   = coin.get("social_dominance",0.0)
                gal_score = coin.get("galaxy_score",0.0)
                alt_r     = coin.get("alt_rank",999999)
                senti     = coin.get("sentiment",0.0)

                cats      = coin.get("categories","")
                topic     = coin.get("topic","")
                logo      = coin.get("logo","")

                c.execute("""
                    INSERT INTO lunarcrush_data (
                        timestamp, symbol, name, price, market_cap, volume_24h, volatility,
                        percent_change_1h, percent_change_24h, percent_change_7d, percent_change_30d,
                        social_volume_24h, interactions_24h, social_dominance, galaxy_score,
                        alt_rank, sentiment, categories, topic, logo
                    )
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, [
                    now_ts,
                    symbol, name, price, mcap, vol24, volty,
                    pct1h, pct24, pct7d, pct30,
                    socvol_24, inter_24, soc_dom, gal_score,
                    alt_r, senti, cats, topic, logo
                ])
                inserted+=1

            conn.commit()
            logger.info(f"[SNAPSHOT] Inserted {inserted} rows into 'lunarcrush_data'.")
        except Exception as e:
            logger.exception(f"[SNAPSHOT] error => {e}")
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # Time-series => 'lunarcrush_timeseries'
    # --------------------------------------------------------------------------
    def fetch_time_series_for_tracked_coins(self, bucket="hour", interval="1w",
                                            start: Optional[int]=None, end: Optional[int]=None):
        """
        We'll interpret the 'symbol' column in lunarcrush_data as the unique coin ID
        to pass to /public/coins/<symbol>/time-series/v2.
        If the API expects numeric IDs, you'd adapt. We'll store the returned data in 'lunarcrush_timeseries'.
        """
        coin_symbols = self._load_symbols_from_db()
        if not coin_symbols:
            logger.info("[TIMESERIES] no symbols => skip.")
            return

        logger.info(f"[TIMESERIES] => found {len(coin_symbols)} distinct symbols => fetch time-series.")
        for idx, sym in enumerate(coin_symbols, start=1):
            time.sleep(REQUEST_SLEEP_SECONDS)
            logger.info(f"[TIMESERIES] => {idx}/{len(coin_symbols)} => symbol={sym}, bucket={bucket}, interval={interval}")
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
                logger.info(f"[TIMESERIES] GET => {base_url}, attempt={attempt+1}, coin_id={coin_id}")
                resp = requests.get(base_url, headers=headers, params=params, timeout=10)
                resp.raise_for_status()

                data = resp.json()
                recs = data.get("data", [])
                logger.info(f"[TIMESERIES] symbol={coin_id}, got {len(recs)} records")

                if recs:
                    self._store_timeseries_records(coin_id, recs)
                break
            except requests.exceptions.HTTPError as e:
                attempt+=1
                logger.error(f"[TIMESERIES] HTTP => {e}, sleep={RETRY_DELAY_SECONDS}")
                time.sleep(RETRY_DELAY_SECONDS)
            except Exception as e:
                attempt+=1
                logger.exception(f"[TIMESERIES] => {e}, sleep={RETRY_DELAY_SECONDS}")
                time.sleep(RETRY_DELAY_SECONDS)
        else:
            logger.error(f"[TIMESERIES] All attempts fail => symbol={coin_id}")

    def _store_timeseries_records(self, coin_id: str, records: list):
        """
        Insert into 'lunarcrush_timeseries', matching the db.py columns:
          (coin_id, timestamp, open_price, close_price, high_price, low_price, volume_24h, market_cap,
           sentiment, spam, galaxy_score, alt_rank, volatility, interactions, social_dominance)
        We'll skip 'market_dominance', 'circulating_supply', etc. not in db.py.
        """
        if not records:
            logger.info(f"[TIMESERIES] symbol={coin_id} => no records => skip.")
            return

        conn = sqlite3.connect(self.db_file)
        try:
            c = conn.cursor()
            inserted=0
            for r in records:
                # typical fields from the time-series JSON
                ts = r.get("time",0)
                open_p = r.get("open",0.0)
                close_p= r.get("close",0.0)
                high_p = r.get("high",0.0)
                low_p  = r.get("low",0.0)
                vol24  = r.get("volume_24h",0.0)
                mc     = r.get("market_cap",0.0)
                senti  = r.get("sentiment",0.0)
                sp     = r.get("spam",0.0)
                gal_s  = r.get("galaxy_score",0.0)
                alt_r  = r.get("alt_rank",0)
                volty  = r.get("volatility",0.0)
                inter  = r.get("interactions",0.0)
                socdom = r.get("social_dominance",0.0)

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
                    coin_id, ts,
                    open_p, close_p, high_p, low_p,
                    vol24, mc,
                    senti, sp,
                    gal_s, alt_r, volty,
                    inter, socdom
                ])
                inserted+=1
            conn.commit()
            logger.info(f"[TIMESERIES] symbol={coin_id} => inserted {inserted} rows.")
        except Exception as e:
            logger.exception(f"[TIMESERIES] store => {e}")
        finally:
            conn.close()

    def _load_symbols_from_db(self) -> List[str]:
        """
        Return distinct uppercase symbols from 'lunarcrush_data' => we assume to pass them to /public/coins/<symbol>/time-series.
        If your API usage requires numeric IDs, adapt accordingly.
        """
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
        """
        For chunk-based historical data. We'll interpret 'coin_ids' as the symbols. We'll
        chunk from (now - months*30days) to present in ~30-day increments.
        """
        if not self.BEARER_TOKEN:
            logger.warning("No LUNARCRUSH_BEARER_TOKEN => cannot backfill timeseries.")
            return
        if not coin_ids:
            logger.info("[BACKFILL] no coin_ids => skip.")
            return

        logger.info(f"[BACKFILL] chunk-based => {len(coin_ids)} symbols, months={months}")
        now_ts = int(time.time())
        chunk_seconds = 30*86400
        total_back_secs = months*chunk_seconds

        for idx, sym in enumerate(coin_ids, start=1):
            logger.info(f"[BACKFILL] => symbol={sym} => {idx}/{len(coin_ids)} => months={months}")
            start_ts = now_ts - total_back_secs
            while start_ts < now_ts:
                chunk_end = start_ts + chunk_seconds
                if chunk_end > now_ts:
                    chunk_end = now_ts

                logger.info(f"[BACKFILL] chunk => symbol={sym}, start={start_ts}, end={chunk_end}")
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
            SELECT UPPER(symbol), galaxy_score, alt_rank, market_cap, percent_change_24h, sentiment
            FROM lunarcrush_data
            WHERE symbol IS NOT NULL
            """).fetchall()
            if not rows:
                logger.warning("[SPOTCHECK] no rows => can't proceed.")
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

                # example rank => (0.5*gscore + 0.2*(mc/1e9) + 0.2*pc24 + 0.05*(sent/1=100?) - 0.05*ar
                # We'll just do a naive approach:
                scaled_senti = sent*0.01
                rank_score = (0.5*gsc) + (0.2*(mc/1e9)) + (0.2*pc24) + (0.05*scaled_senti) - (0.05*ar)
                temp.append((sym, rank_score))

            temp.sort(key=lambda x:x[1], reverse=True)
            top_temp = temp[:top_n]
            out = [t[0] for t in top_temp]
            logger.info(f"[SPOTCHECK] => top_n={top_n}, {top_temp}")
        except Exception as e:
            logger.exception(f"[SPOTCHECK] => {e}")
        finally:
            conn.close()
        return out


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Fetch from LunarCrush into your db.py schema.")
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

    # if you need config
    config_file = "config.yaml"
    traded_pairs = []
    if os.path.exists(config_file):
        import yaml
        with open(config_file,"r") as f:
            config = yaml.safe_load(f)
        traded_pairs = config.get("traded_pairs",[])

    # (A) Snapshots
    if args.snapshot_filtered:
        fetcher.fetch_snapshot_data_filtered(limit=args.limit)
    if args.snapshot_all:
        fetcher.fetch_snapshot_data_all_coins(limit=args.limit)

    # (B) Time-series => no chunk => just fetch for all symbols
    if args.time_series:
        fetcher.fetch_time_series_for_tracked_coins(bucket="hour", interval="1w")

    # (C) chunk-based backfill
    if any([args.backfill_all, args.backfill_top, args.backfill_configured]):
        coin_ids=[]

        def load_all_symbols() -> List[str]:
            out=[]
            conn=sqlite3.connect(DB_FILE)
            try:
                c=conn.cursor()
                rows=c.execute("""
                SELECT DISTINCT UPPER(symbol)
                FROM lunarcrush_data
                WHERE symbol IS NOT NULL AND symbol != ''
                """).fetchall()
                out=[r[0] for r in rows if r[0]]
            except Exception as e:
                logger.exception(f"[BACKFILL-ALL] => {e}")
            finally:
                conn.close()
            return out

        def load_configured_symbols(tpairs:List[str]) -> List[str]:
            # parse symbol from each "ETH/USD" => "ETH"
            syms=[]
            for pair in tpairs:
                syms.append(pair.split("/")[0].upper())
            syms = list(set(syms))
            logger.info(f"[ConfiguredSymbols] => {syms}")
            return syms

        if args.backfill_all:
            coin_ids = load_all_symbols()

        elif args.backfill_top:
            top_syms = fetcher.spot_check_entire_market(top_n=args.top_n)
            coin_ids = top_syms

        elif args.backfill_configured:
            coin_ids = load_configured_symbols(traded_pairs)

        if not coin_ids:
            logger.info("[BACKFILL] no coin_ids => skip.")
        else:
            fetcher.backfill_coins(coin_ids, months=args.months, bucket="hour", interval="1w")


if __name__=="__main__":
    main()
