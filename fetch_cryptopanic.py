#!/usr/bin/env python3
# =============================================================================
# FILE: fetch_cryptopanic.py
# =============================================================================
"""
fetch_cryptopanic.py

Enhancements:
1) Multi-filter iteration (e.g. "rising", "hot", "bullish") in a single run.
2) Robust 'cryptopanic_posts' table storing advanced fields (metadata, domain, panic_score, etc.).
3) Additional methods to:
   - Choose top coins from 'lunarcrush_data' to specifically fetch cryptopanic news for (spot-check synergy).
   - Recommend new pairs after synergy analysis from both cryptopanic + lunarcrush.

Usage:
    python fetch_cryptopanic.py

Environment:
    - CRYPTO_PANIC_API_KEY: your token
"""

import os
import time
import requests
import logging
import sqlite3
from typing import Dict, Any, Optional, List, Tuple
from dotenv import load_dotenv
import math

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DB_FILE = "trades.db"
TABLE_NAME = "cryptopanic_posts"

# Multi-filter iteration in one run
MULTI_FILTERS = ["rising", "hot", "bullish", "bearish"]  # or whatever filters you want

# Example toggles
use_public   = False
use_following= False
use_metadata = True
use_approved = True
use_panic    = False   # "panic_score=true" if your plan supports it
kind_param   = "news"  # e.g. "news" or "media"
regions_param= None    # e.g. "en,de"

# Currencies to pass
coin_list = [
    "ETH", "XBT", "SOL", "ADA", "XRP", "LTC", "DOT", "MATIC", "LINK", "ATOM",
    "XMR", "TRX", "SHIB", "AVAX", "UNI", "APE", "FIL", "AAVE", "SAND", "CHZ",
    "GRT", "CRV", "COMP", "ALGO", "NEAR", "LDO", "XTZ", "EGLD", "KSM"
]

# How many pages per filter to fetch
max_pages = 2

# Sleep for rate-limiting.
# CryptoPanic doc => 5 or 10 requests/sec depending on plan.
# We'll do 0.2s => 5 calls/sec for each page fetch.
REQUEST_SLEEP = 0.2

# For synergy-based pair recommendation
RECOMMEND_THRESHOLD = 0.3


def main():
    """
    By default =>
      1) Create table if missing
      2) For each filter in MULTI_FILTERS => fetch multiple pages
      3) Possibly backfill for top coins from lunarcrush
      4) synergy-based recommended pairs
    """
    fetcher = CryptoPanicFetcher(db_file=DB_FILE)

    # 1) Ensure table
    fetcher.init_cryptopanic_table()

    # 2) Multi-filter iteration => fetch posts for each filter
    for fparam in MULTI_FILTERS:
        logger.info(f"[MultiFilter] Attempting filter={fparam}")
        fetcher.fetch_posts_with_params(filter_param=fparam, max_pages=max_pages)
    logger.info("[MultiFilter] Done fetching for multiple filters.")

    # 3) Possibly do a 'spot-check' => pick top coins from lunarcrush => fetch cryptopanic for them
    fetcher.backfill_for_top_lunarcrush_coins(limit_coins=5)

    # 4) synergy-based recommended pairs
    new_pairs = fetcher.recommend_new_pairs()
    logger.info(f"Recommended new pairs => {new_pairs}")


class CryptoPanicFetcher:
    """
    A robust class that fetches CryptoPanic posts with advanced usage:
    - multi-filter iteration in a single run
    - metadata/approved/panic_score toggles
    - synergy with 'lunarcrush_data' for top coins
    - synergy-based recommended pairs
    """

    def __init__(self, db_file: str = DB_FILE):
        self.db_file = db_file
        load_dotenv()
        self.api_key = os.getenv("CRYPTO_PANIC_API_KEY", "")
        if not self.api_key:
            logger.error("No CRYPTO_PANIC_API_KEY in env. Some calls may fail or be incomplete.")

    def init_cryptopanic_table(self):
        """
        Creates cryptopanic_posts if missing, with advanced fields.
        """
        conn = sqlite3.connect(self.db_file)
        try:
            c = conn.cursor()
            c.execute(f"""
                CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_id TEXT UNIQUE,
                    title TEXT,
                    url TEXT,
                    domain TEXT,
                    published_at TEXT,
                    tags TEXT,
                    negative_votes INTEGER,
                    positive_votes INTEGER,
                    liked_votes INTEGER,
                    disliked_votes INTEGER,
                    sentiment_score REAL,
                    image TEXT,
                    description TEXT,
                    panic_score REAL,
                    created_at INTEGER
                )
            """)
            conn.commit()
        except Exception as e:
            logger.exception(f"Error creating {TABLE_NAME}: {e}")
        finally:
            conn.close()

    def fetch_posts_with_params(self, filter_param: str = None, max_pages: int = 2):
        """
        Fetch up to max_pages from CryptoPanic using your toggles + a given 'filter_param'.
        This supports multiple 'filter_param' calls in a single run if needed.
        """
        if not self.api_key:
            logger.warning("No CRYPTO_PANIC_API_KEY => skip fetch.")
            return

        base_url = "https://cryptopanic.com/api/v1/posts/"
        params = {"auth_token": self.api_key}

        if use_public:
            params["public"] = "true"
        if use_following:
            params["following"] = "true"
        if use_metadata:
            params["metadata"] = "true"
        if use_approved:
            params["approved"] = "true"
        if use_panic:
            params["panic_score"] = "true"
        if kind_param:
            params["kind"] = kind_param
        if regions_param:
            params["regions"] = regions_param

        if filter_param:
            params["filter"] = filter_param

        if coin_list:
            joined = ",".join(coin_list)
            params["currencies"] = joined

        logger.info(f"[CryptoPanic] Start fetch => filter={filter_param}, pages={max_pages}, params={params}")

        page_count=0
        next_url=None

        def do_get(url, p):
            logger.debug(f"GET => {url}, p={p}")
            r = requests.get(url, params=p, timeout=10)
            r.raise_for_status()
            time.sleep(REQUEST_SLEEP)
            return r.json()

        # 1) first page
        try:
            data = do_get(base_url, params)
        except Exception as e:
            logger.exception(f"[CryptoPanic] first page => {e}")
            return

        self._parse_and_store_posts(data)
        page_count+=1
        logger.info(f"[CryptoPanic] page=1 => stored {len(data.get('results',[]))} posts")

        next_url = data.get("next")
        while next_url and page_count<max_pages:
            page_count+=1
            logger.info(f"[CryptoPanic] next => {next_url}, page={page_count}")
            try:
                r2 = requests.get(next_url, timeout=10)
                r2.raise_for_status()
                time.sleep(REQUEST_SLEEP)
                data2 = r2.json()
            except Exception as e:
                logger.exception(f"[CryptoPanic] next fetch => {e}")
                break

            self._parse_and_store_posts(data2)
            logger.info(f"[CryptoPanic] page={page_count} => stored {len(data2.get('results',[]))} posts")
            next_url = data2.get("next")
            if not next_url:
                logger.info("[CryptoPanic] no more pages => stop.")
                break

    def _parse_and_store_posts(self, json_data: dict):
        posts = json_data.get("results", [])
        if not posts:
            logger.info("No results in this response chunk.")
            return

        conn = sqlite3.connect(self.db_file)
        try:
            c = conn.cursor()
            now_ts = int(time.time())

            for post in posts:
                post_id = str(post.get("id",""))
                title   = post.get("title","No Title")
                url_val = post.get("url","")
                domain  = post.get("domain","")
                pub_at  = post.get("published_at","")

                votes = post.get("votes",{})
                neg_v= votes.get("negative",0)
                pos_v= votes.get("positive",0)
                liked= votes.get("liked",0)
                disliked= votes.get("disliked",0)

                tags_list = post.get("tags",[])
                tags_str  = ",".join(tags_list)

                sentiment_score = compute_enhanced_sentiment(post)

                image_val=""
                desc_val=""
                if "metadata" in post and isinstance(post["metadata"],dict):
                    image_val = post["metadata"].get("image","")
                    desc_val  = post["metadata"].get("description","")

                p_score=None
                if "panic_score" in post:
                    p_score = post["panic_score"]

                # skip if we already have post_id
                c.execute(f"SELECT 1 FROM {TABLE_NAME} WHERE post_id=?",[post_id])
                dupe_row = c.fetchone()
                if dupe_row:
                    continue

                c.execute(f"""
                    INSERT INTO {TABLE_NAME} (
                        post_id, title, url, domain, published_at,
                        tags, negative_votes, positive_votes, liked_votes, disliked_votes,
                        sentiment_score, image, description, panic_score, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    post_id, title, url_val, domain, pub_at,
                    tags_str, neg_v, pos_v, liked, disliked,
                    sentiment_score, image_val, desc_val, p_score, now_ts
                ])
            conn.commit()
            logger.info(f"[CryptoPanic] Inserted {len(posts)} new posts into {TABLE_NAME}")
        except Exception as e:
            logger.exception(f"[CryptoPanic] parse/store => {e}")
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # New methods for synergy with lunarcrush
    # --------------------------------------------------------------------------
    def backfill_for_top_lunarcrush_coins(self, limit_coins=5):
        """
        We find the top 'limit_coins' from 'lunarcrush_data' by a simple approach
        (like highest galaxy_score). Then fetch cryptopanic with that subset of coin_list
        (currencies=...) in a single request.
        """
        conn = sqlite3.connect(self.db_file)
        best_symbols=[]
        try:
            c = conn.cursor()
            q = """
                SELECT symbol, galaxy_score
                FROM lunarcrush_data
                WHERE galaxy_score>0
                ORDER BY galaxy_score DESC
                LIMIT ?
            """
            rows = c.execute(q, [limit_coins]).fetchall()
            for (sym, gsc) in rows:
                best_symbols.append(sym.upper())
        except Exception as e:
            logger.exception(f"[CryptoPanic] error picking top coins => {e}")
        finally:
            conn.close()

        if not best_symbols:
            logger.info("[CryptoPanic] No top coins => skip.")
            return

        logger.info(f"[CryptoPanic] backfill_for_top_lunarcrush_coins => {best_symbols}")

        base_url = "https://cryptopanic.com/api/v1/posts/"
        params = {
            "auth_token": self.api_key
        }
        if use_public:
            params["public"] = "true"
        if use_following:
            params["following"] = "true"
        if use_metadata:
            params["metadata"] = "true"
        if use_approved:
            params["approved"] = "true"
        if use_panic:
            params["panic_score"] = "true"
        if kind_param:
            params["kind"] = kind_param

        # We'll do a single call with &currencies= the top coins
        joined = ",".join(best_symbols)
        params["currencies"] = joined

        try:
            resp = requests.get(base_url, params=params, timeout=10)
            resp.raise_for_status()
            time.sleep(REQUEST_SLEEP)
            data = resp.json()
            self._parse_and_store_posts(data)
        except Exception as e:
            logger.exception(f"[CryptoPanic] error fetch for top coins => {e}")

    def recommend_new_pairs(self) -> List[str]:
        """
        A synergy-based approach =>
          1) read average cryptopanic sentiment from last 7 days
          2) read galaxy_score from lunarcrush
          3) synergy => if synergy >= RECOMMEND_THRESHOLD => add pair

        We do a naive approach => doesn't tie posts to coin symbol 1:1.
        For a robust approach, you'd parse the 'tags' or store symbol references in cryptopanic.
        """
        synergy_symbols=[]
        overall_cp_score = self._compute_overall_cryptopanic_sentiment(days=7)

        # read symbol => galaxy_score from 'lunarcrush_data'
        conn=sqlite3.connect(self.db_file)
        sym_to_gsc={}
        try:
            c = conn.cursor()
            q="""
                SELECT UPPER(symbol), AVG(galaxy_score)
                FROM lunarcrush_data
                WHERE galaxy_score>0
                GROUP BY UPPER(symbol)
            """
            for row in c.execute(q):
                sy = row[0]
                gs = float(row[1]) if row[1] else 0.0
                sym_to_gsc[sy]=gs
        except Exception as e:
            logger.exception(f"recommend_new_pairs => {e}")
        finally:
            conn.close()

        # synergy => synergy = ( (gscore/100)+ (overall_cp_score+1)/2 ) / 2
        # if synergy >= RECOMMEND_THRESHOLD => add
        # or any formula you like
        for s,gsc in sym_to_gsc.items():
            norm_g = gsc/100.0
            norm_cp = (overall_cp_score+1.0)/2.0
            synergy = (norm_g + norm_cp)/2.0
            if synergy>=RECOMMEND_THRESHOLD:
                synergy_symbols.append(s)

        # produce new pairs => e.g. "ETH/USD"
        pairs=[]
        for sy in synergy_symbols:
            pairs.append(f"{sy}/USD")

        return pairs

    def _compute_overall_cryptopanic_sentiment(self, days: int=7) -> float:
        """
        A naive approach => average sentiment_score from last X days.
        Range => [-1..+1].
        If no posts, return 0.0
        """
        cutoff_ts= int(time.time()) - (days*86400)
        conn=sqlite3.connect(self.db_file)
        try:
            c=conn.cursor()
            q=f"""
                SELECT AVG(sentiment_score)
                FROM {TABLE_NAME}
                WHERE created_at >= ?
            """
            c.execute(q,[cutoff_ts])
            row=c.fetchone()
            if row and row[0]:
                return float(row[0])
            else:
                return 0.0
        except Exception as e:
            logger.exception(f"Error computing cryptopanic sentiment => {e}")
            return 0.0
        finally:
            conn.close()


def compute_enhanced_sentiment(post: Dict[str, Any]) -> float:
    """
    Incorporates post tags + votes => sentiment in [-1..+1].
    """
    sentiment=0.0
    tags = post.get("tags",[])
    if "bullish" in tags:
        sentiment+=0.5
    if "bearish" in tags:
        sentiment-=0.5

    votes = post.get("votes",{})
    pos_v = votes.get("positive",0)
    neg_v = votes.get("negative",0)
    liked= votes.get("liked",0)
    disliked= votes.get("disliked",0)

    # weighting
    sentiment += pos_v*0.02
    sentiment -= neg_v*0.02
    sentiment += liked*0.01
    sentiment -= disliked*0.01

    # clamp
    if sentiment>1.0: sentiment=1.0
    if sentiment<-1.0:sentiment=-1.0

    return sentiment


if __name__=="__main__":
    main()
