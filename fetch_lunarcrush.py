#!/usr/bin/env python3
# =============================================================================
# FILE: fetch_lunarcrush.py
# =============================================================================
"""
fetch_lunarcrush.py

A script that fetches data from LunarCrush, storing it in your DB for AI usage.

Features:
1) Loads LUNARCRUSH_API_KEY from environment (e.g., .env).
2) Calls the "/public/coins/list/v2" endpoint (as an example) to retrieve
   social/market metrics for multiple coins.
3) Iterates over each coin's metrics, then calls store_lunarcrush_data(...)
   to insert into your 'lunarcrush' table (you'll create or expand it in db.py).
4) Optionally logs or manipulates (e.g., naive sentiment from "galaxy_score").

Usage:
  python fetch_lunarcrush.py
"""

import os
import requests
import logging
from typing import Dict, Any
from dotenv import load_dotenv

# If your db.py has a function like "store_lunarcrush_data(symbol, galaxy_score, alt_rank, price, etc...)"
from db import store_lunarcrush_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------------------
# Example Config Toggles
# ------------------------------------------------------------------------------
# For instance, you can set which symbols you want to filter or just fetch them all.
# (The LunarCrush docs say 'public/coins/list/v2' returns snapshot for many coins)
filter_symbols = ["BTC", "ETH", "SOL", "ADA", "XRP"]
# Adjust as needed. You can remove filtering if you want everything.

def fetch_lunarcrush_data():
    """
    Fetch data from LunarCrush's coins/list/v2 endpoint to get a snapshot
    of multiple coins' metrics (Galaxy Score, AltRank, etc.).
    Store each coin's record into the DB using store_lunarcrush_data(...).
    """
    # 1) Load your API key
    load_dotenv()
    LUNARCRUSH_API_KEY = os.getenv("LUNARCRUSH_API_KEY")
    if not LUNARCRUSH_API_KEY:
        logger.error("No LUNARCRUSH_API_KEY found in environment! Aborting.")
        return

    # 2) Build your base URL and params
    base_url = "https://lunarcrush.com/api4/public/coins/list/v2"

    # LunarCrush usually expects "?key=YOUR_KEY" or "Authorization" header. We'll do param style.
    params = {
        "key": LUNARCRUSH_API_KEY,
        "limit": 100,  # or more if you want a bigger snapshot
        # You can also pass 'symbol=BTC,ETH' to limit results, etc.
    }

    try:
        logger.info(f"Requesting data from {base_url} with params={params}")
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()

        data = resp.json()  # Expect JSON data
        # Usually, the structure might have something like "data": [...]
        coins_list = data.get("data", [])
        logger.info(f"Fetched {len(coins_list)} coins from LunarCrush.")

        # 3) If you want to filter by certain symbols
        if filter_symbols:
            # Convert them to uppercase for matching
            filter_symbols_upper = [s.upper() for s in filter_symbols]
            coins_list = [c for c in coins_list if c.get("symbol", "").upper() in filter_symbols_upper]

        logger.info(f"After filtering, we have {len(coins_list)} coins to store.")

        # 4) For each coin, parse relevant fields and store
        for coin in coins_list:
            symbol = coin.get("symbol", "Unknown")
            name = coin.get("name", "Unknown")

            # Some metrics you might want
            galaxy_score = coin.get("gs", 0.0)       # e.g. "gs" => Galaxy Score
            alt_rank = coin.get("acr", 0)           # "acr" => AltRank
            price = coin.get("price", 0.0)
            # etc. Possibly: market_cap, social_impact_score, news_count, volume, etc.

            logger.debug(f"Coin => symbol={symbol}, galaxy_score={galaxy_score}, alt_rank={alt_rank}, price={price}")

            # Insert into your DB
            store_lunarcrush_data(symbol, name, galaxy_score, alt_rank, price)

        logger.info(f"Done storing LunarCrush data for {len(coins_list)} coins.")

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error: {e}")
    except Exception as e:
        logger.exception(f"Error fetching or storing LunarCrush data: {e}")

if __name__ == "__main__":
    fetch_lunarcrush_data()
