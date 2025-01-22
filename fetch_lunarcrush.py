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
   to insert into your 'lunarcrush_data' table (you'll create or expand it in db.py).
4) Optionally logs or manipulates these metrics for advanced usage.

Usage:
  python fetch_lunarcrush.py
"""

import os
import requests
import logging
from typing import Dict, Any
from dotenv import load_dotenv
import json

# If your db.py has a function like:
# store_lunarcrush_data(
#    symbol, name, price, market_cap, volume_24h, volatility,
#    percent_change_1h, percent_change_24h, percent_change_7d, percent_change_30d,
#    social_volume_24h, interactions_24h, social_dominance, galaxy_score, alt_rank, sentiment,
#    categories, topic, logo
# )
from db import store_lunarcrush_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------------------
# Example Config Toggles
# ------------------------------------------------------------------------------
# For instance, you can set which symbols you want to filter or just fetch them all.
# (The LunarCrush docs say '/public/coins/list/v2' returns snapshot for many coins)
filter_symbols = [
    "ETH", "XBT", "SOL", "ADA", "XRP", "LTC", "DOT", "MATIC", "LINK", "ATOM",
    "XMR", "TRX", "SHIB", "AVAX", "UNI", "APE", "FIL", "AAVE", "SAND", "CHZ",
    "GRT", "CRV", "COMP", "ALGO", "NEAR", "LDO", "XTZ", "EGLD", "KSM"
]
# Adjust as needed. You can remove filtering if you want everything.

def fetch_lunarcrush_data():
    """
    Fetch data from LunarCrush's coins/list/v2 endpoint to get a snapshot
    of multiple coins' metrics (Galaxy Score, AltRank, price, etc.).
    Store each coin's record into the DB using store_lunarcrush_data(...).

    The following fields are parsed and sent to db.py for storage:
      - symbol, name
      - price, market_cap
      - volume_24h, volatility
      - percent_change_1h, percent_change_24h, percent_change_7d, percent_change_30d
      - social_volume_24h, interactions_24h, social_dominance
      - galaxy_score, alt_rank, sentiment
      - categories, topic, logo
    """
    # 1) Load your API key
    load_dotenv()
    LUNARCRUSH_API_KEY = os.getenv("LUNARCRUSH_API_KEY")
    if not LUNARCRUSH_API_KEY:
        logger.error("No LUNARCRUSH_API_KEY found in environment! Aborting.")
        return

    # 2) Build your base URL and params
    base_url = "https://lunarcrush.com/api4/public/coins/list/v2"

    # LunarCrush usually expects "?key=YOUR_KEY". We'll do param style.
    params = {
        "key": LUNARCRUSH_API_KEY,
        "limit": 100,  # or more if you want a bigger snapshot
        # You can also pass 'symbol=BTC,ETH' if you want to limit results on the server side.
    }

    try:
        logger.info(f"Requesting data from {base_url} with params={params}")
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()

        data = resp.json()  # Expect JSON data
        print(f'data: {json.dumps(data, indent=4)}')
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

            # The snippet from your example code:
            #   "price", "market_cap", "volume_24h", "volatility",
            #   "percent_change_1h", "percent_change_24h", "percent_change_7d", "percent_change_30d",
            #   "social_volume_24h", "interactions_24h", "social_dominance",
            #   "galaxy_score", "alt_rank", "sentiment",
            #   "categories", "topic", "logo"
            # In the actual JSON, the field names differ from your snippet. Some are short codes:
            #   e.g. "gs" for galaxy_score, "acr" for alt_rank, etc.
            #
            # We'll do our best to map them:

            price                = coin.get("price", 0.0)
            market_cap           = coin.get("market_cap", 0.0)
            volume_24h           = coin.get("volume_24h", 0.0)
            volatility           = coin.get("volatility", 0.0)

            percent_change_1h    = coin.get("percent_change_1h", 0.0)
            percent_change_24h   = coin.get("percent_change_24h", 0.0)
            percent_change_7d    = coin.get("percent_change_7d", 0.0)
            percent_change_30d   = coin.get("percent_change_30d", 0.0)

            social_volume_24h    = coin.get("social_volume_24h", 0.0)
            interactions_24h     = coin.get("interactions_24h", 0.0)
            social_dominance     = coin.get("social_dominance", 0.0)

            # "gs" => galaxy_score, "acr" => alt_rank, "sentiment" => 0..100
            galaxy_score         = coin.get("gs", 0.0)
            alt_rank             = coin.get("acr", 0)
            sentiment            = coin.get("sentiment", 0.0)

            # categories might be a string like "nft,gaming"
            categories           = coin.get("categories", "")

            # topic might be something like "sand the sandbox"
            topic                = coin.get("topic", "")

            # e.g. "https://cdn.lunarcrush.com/the-sandbox.png"
            logo                 = coin.get("logo", "")

            logger.debug(
                f"Coin => symbol={symbol}, name={name}, galaxy_score={galaxy_score}, "
                f"alt_rank={alt_rank}, price={price}, sentiment={sentiment}"
            )

            # Insert into your DB.
            store_lunarcrush_data(
                symbol,
                name,
                price,
                market_cap,
                volume_24h,
                volatility,
                percent_change_1h,
                percent_change_24h,
                percent_change_7d,
                percent_change_30d,
                social_volume_24h,
                interactions_24h,
                social_dominance,
                galaxy_score,
                alt_rank,
                sentiment,
                categories,
                topic,
                logo
            )

        logger.info(f"Done storing LunarCrush data for {len(coins_list)} coins.")

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error: {e}")
    except Exception as e:
        logger.exception(f"Error fetching or storing LunarCrush data: {e}")


if __name__ == "__main__":
    fetch_lunarcrush_data()
