#!/usr/bin/env python3
# =============================================================================
# FILE: fetch_cryptopanic.py
# =============================================================================
"""
fetch_cryptopanic.py

A script to fully maximize data retrieval from CryptoPanic's API, including:
1) Multiple pages of results (using the 'next' field in the JSON).
2) Advanced fields like 'domain', 'published_at', 'image', 'description' (metadata).
3) Original link if your plan includes 'approved=true'.
4) Optional panic_score if available.
5) Advanced sentiment logic from post["votes"] and post["tags"].

We store each post into a 'cryptopanic_posts' table with columns for post ID, title,
domain, published_at, tags, sentiment_score, etc. We also apply a "backfill" approach:
keep following the 'next' URL up to a maximum number of pages or until no more data.

Usage:
    python fetch_cryptopanic.py

Environment:
    - CRYPTO_PANIC_API_KEY: your token
    - For advanced usage, ensure you have a plan that allows metadata, approved, panic_score, etc.
"""

import os
import time
import requests
import logging
import sqlite3
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
DB_FILE = "trades.db"  # or wherever you want
TABLE_NAME = "cryptopanic_posts"   # The robust table that stores all post info

# Example toggles for the query
use_public     = False        # => public feed if True
use_following  = True         # => "following=true" if True
use_metadata   = True         # => "metadata=true" if True
use_approved   = True         # => "approved=true" if True
use_panic      = False        # => "panic_score=true" if you have the plan
filter_param   = "rising"     # e.g. "rising", "hot", "bullish", "bearish", ...
kind_param     = "news"       # e.g. "news" or "media"
regions_param  = None         # e.g. "en,de"
coin_list      = [
    "ETH", "XBT", "SOL", "ADA", "XRP", "LTC", "DOT", "MATIC", "LINK", "ATOM",
    "XMR", "TRX", "SHIB", "AVAX", "UNI", "APE", "FIL", "AAVE", "SAND", "CHZ",
    "GRT", "CRV", "COMP", "ALGO", "NEAR", "LDO", "XTZ", "EGLD", "KSM"
]
max_pages      = 5   # number of pages to fetch in a single run (for "backfill")

# Rate-limit => e.g. 5 requests/second => let's do a small sleep after each call to be safe
REQUEST_SLEEP_SECONDS = 0.3

# ------------------------------------------------------------------------------
# Main fetch method
# ------------------------------------------------------------------------------
def fetch_cryptopanic_data():
    """
    Fetches multiple pages of CryptoPanic data, storing each post in 'cryptopanic_posts'.
    We incorporate advanced fields with &metadata=..., &approved=..., &panic_score=..., etc.
    """
    load_dotenv()
    CRYPTO_PANIC_API_KEY = os.getenv("CRYPTO_PANIC_API_KEY")
    if not CRYPTO_PANIC_API_KEY:
        logger.error("No CRYPTO_PANIC_API_KEY found in environment! Aborting.")
        return

    base_url = "https://cryptopanic.com/api/v1/posts/"
    # Build request params
    params = {"auth_token": CRYPTO_PANIC_API_KEY}

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
    if filter_param:
        params["filter"] = filter_param
    if kind_param:
        params["kind"] = kind_param
    if regions_param:
        params["regions"] = regions_param

    # Convert coin_list => e.g. "XBT,ETH,SOL"
    if coin_list:
        params["currencies"] = ",".join(coin_list)

    logger.info(f"Starting CryptoPanic fetch => base={base_url}, params={params}")

    # 1) Ensure DB table
    _init_cryptopanic_posts_table()

    # 2) We'll fetch multiple pages. The first call is base_url + params.
    next_url = None
    page_count = 0

    # We'll define a local helper
    def get_data(url: str, p: dict) -> dict:
        logger.debug(f"GET => {url}, p={p}")
        resp = requests.get(url, params=p, timeout=10)
        time.sleep(REQUEST_SLEEP_SECONDS)
        resp.raise_for_status()
        return resp.json()

    # fetch the first page
    try:
        data = get_data(base_url, params)
    except Exception as e:
        logger.exception(f"Error fetching first page => {e}")
        return

    # parse and store
    parse_and_store_posts(data)
    page_count += 1
    logger.info(f"Done page=1 => stored {len(data.get('results',[]))} posts.")

    next_url = data.get("next")  # e.g. a fully qualified URL or None
    while next_url and page_count < max_pages:
        page_count += 1
        logger.info(f"Fetching next page => {next_url}, page_count={page_count}")
        # we'll do a separate approach => next_url is full => no need for base_url or initial params
        try:
            # next_url might not require 'params' if it's a full query string
            # but for safety, we parse it
            data2 = requests.get(next_url, timeout=10).json()
            time.sleep(REQUEST_SLEEP_SECONDS)
        except Exception as e:
            logger.exception(f"Error fetching next page => {e}")
            break

        parse_and_store_posts(data2)
        logger.info(f"Done page={page_count} => stored {len(data2.get('results',[]))} posts.")

        next_url = data2.get("next")
        if not next_url:
            logger.info("No more pages left from the API => stopping.")
            break

    logger.info(f"Finished fetching => total pages visited={page_count}")


# ------------------------------------------------------------------------------
# Table creation and storing logic
# ------------------------------------------------------------------------------
def _init_cryptopanic_posts_table():
    """
    Creates a robust table 'cryptopanic_posts' with columns for:
      - post_id => from CryptoPanic
      - title, url, domain
      - published_at
      - negative_votes, positive_votes, liked_votes, disliked_votes, ...
      - tags => comma-delimited
      - sentiment_score => our computed float from  [-1..+1]
      - created_at => local insertion time
      - image, description => from metadata if available
      - panic_score => if your plan includes it
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id TEXT,
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
        logger.exception(f"Error creating table {TABLE_NAME}: {e}")
    finally:
        conn.close()

def parse_and_store_posts(json_data: dict):
    """
    Takes the JSON from a CryptoPanic API response => parse each post => store in DB.
    """
    results = json_data.get("results", [])
    if not results:
        logger.info("No results found in this response. Possibly no posts.")
        return

    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        for post in results:
            post_id = str(post.get("id",""))
            title   = post.get("title","No Title")
            url_val = post.get("url","")
            domain  = post.get("domain","")
            published_at = post.get("published_at","")

            # parse votes
            votes = post.get("votes",{})
            neg = votes.get("negative",0)
            pos = votes.get("positive",0)
            liked= votes.get("liked",0)
            disliked= votes.get("disliked",0)

            # parse tags => store as comma-delimited
            tags = post.get("tags",[])
            tags_str = ",".join(tags)

            # compute advanced sentiment
            sentiment_score = compute_enhanced_sentiment(post)

            # parse image, description if metadata => in "metadata" param or top-level
            # per doc => 'metadata' is only if we used &metadata=true
            image_val = ""
            description_val = ""
            if "metadata" in post and isinstance(post["metadata"], dict):
                image_val = post["metadata"].get("image","")
                description_val = post["metadata"].get("description","")

            # parse panic_score if your plan includes it
            # doc => "Include in your request following parameter &panic_score=true
            panic_val = None
            if "panic_score" in post:
                panic_val = post["panic_score"]

            # insertion
            now_ts = int(time.time())
            c.execute(f"""
                INSERT INTO {TABLE_NAME} (
                    post_id, title, url, domain, published_at,
                    tags,
                    negative_votes, positive_votes, liked_votes, disliked_votes,
                    sentiment_score,
                    image, description,
                    panic_score,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                post_id, title, url_val, domain, published_at,
                tags_str,
                neg, pos, liked, disliked,
                sentiment_score,
                image_val, description_val,
                panic_val,
                now_ts
            ])
        conn.commit()
        logger.info(f"Inserted {len(results)} posts into {TABLE_NAME}")
    except Exception as e:
        logger.exception(f"Error parsing/storing cryptopanic posts => {e}")
    finally:
        conn.close()

def compute_enhanced_sentiment(post: Dict[str, Any]) -> float:
    """
    A robust approach:
      1) Checks post["tags"] for "bullish","bearish"
      2) Looks at votes => positive, negative, liked, disliked
      3) Possibly handle "panic_score" if it's numeric
      Return => [-1..+1].
    """
    sentiment = 0.0

    tags = post.get("tags",[])
    if "bullish" in tags:
        sentiment+=0.5
    if "bearish" in tags:
        sentiment-=0.5

    # parse votes
    votes = post.get("votes",{})
    pos_v = votes.get("positive",0)
    neg_v = votes.get("negative",0)
    liked= votes.get("liked",0)
    disliked= votes.get("disliked",0)

    # weighting => naive
    sentiment += pos_v*0.02
    sentiment -= neg_v*0.02
    sentiment += liked*0.01
    sentiment -= disliked*0.01

    # clamp [-1..1]
    if sentiment>1.0: sentiment=1.0
    if sentiment<-1.0:sentiment=-1.0

    return sentiment

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
if __name__=="__main__":
    fetch_cryptopanic_data()
