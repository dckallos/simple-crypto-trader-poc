#!/usr/bin/env python3
# =============================================================================
# FILE: fetch_cryptopanic.py
# =============================================================================
"""
fetch_cryptopanic.py

Enhancements over the basic version:
1) Parses post["votes"] to incorporate positive/negative counts into the sentiment score.
2) Checks for "bullish"/"bearish" in post["tags"] as before.
3) Optionally logs or stores 'published_at', 'domain', or 'metadata' fields if you want them in your DB.
4) More granular logging for debugging.

Usage:
    python fetch_cryptopanic.py
"""

import os
import requests
import logging
from typing import Dict, Any
from dotenv import load_dotenv
import json

from db import store_cryptopanic_data
# If you want to store more fields in DB, you can add a new function in db.py or expand store_cryptopanic_data().

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------------------
# Global Toggles (same idea as before, adapt to your plan)
# ------------------------------------------------------------------------------
use_public = False
use_following = True
use_metadata = True
use_approved = True
filter_param = "rising"
kind_param = "news"
# If you'd like to specify multiple currencies, put them comma-separated like "ETH,BTC"
currencies_param = "ETH,XBT,SOL,ADA,XRP,LTC,DOT,MATIC,LINK,ATOM,XMR,TRX,SHIB,AVAX,UNI,APE,FIL,AAVE,SAND,CHZ,GRT,CRV,COMP,ALGO,NEAR,LDO,XTZ,EGLD,KSM"
regions_param = None


def fetch_cryptopanic_data():
    """
    Fetches posts from CryptoPanic using your chosen parameters and stores them via db.py.
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
    if filter_param:
        params["filter"] = filter_param
    if kind_param:
        params["kind"] = kind_param
    if currencies_param:
        params["currencies"] = currencies_param
    if regions_param:
        params["regions"] = regions_param

    try:
        logger.info(f"Fetching CryptoPanic data with params={params}")
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()

        data = resp.json()
        # data might have "count", "next", "previous", "results"
        posts = data.get("results", [])
        logger.info(f"Fetched {len(posts)} posts from CryptoPanic.")

        # Optional: log the entire response if you want deeper debugging
        # logger.debug("Full response data: %s", data)

        # For each post, parse sentiment and store
        for post in posts:
            # Title
            title = post.get("title", "No Title")
            # URL
            url = post.get("url", "")

            # NEW: If you want domain, published_at, etc.
            domain = post.get("domain", "")
            published_at = post.get("published_at", "")
            # You can do something like storing them in DB if you have the columns:
            # store_cryptopanic_data_extended(title, url, domain, published_at, sentiment_score)

            # Our improved sentiment
            sentiment_score = compute_enhanced_sentiment(post)

            logger.debug(
                f"POST => Title: {title} | domain={domain} | published_at={published_at} "
                f"| votes={post.get('votes')} | tags={post.get('tags')} | sentiment={sentiment_score}"
            )

            # For now, we’ll keep storing them with store_cryptopanic_data
            # which expects just (title, url, sentiment_score).
            store_cryptopanic_data(title, url, sentiment_score)

        logger.info(f"Stored {len(posts)} posts in DB.")
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error: {e}")
    except Exception as e:
        logger.exception(f"Error fetching or storing CryptoPanic data: {e}")


def compute_enhanced_sentiment(post: Dict[str, Any]) -> float:
    """
    A more robust approach that:
    1) Checks post["tags"] for "bullish", "bearish", etc.
    2) Looks at post["votes"] (positive, negative, etc.).
    3) Possibly extended to parse metadata or do basic text-based sentiment.

    Return in [-1, +1] range, or beyond if you prefer bigger weighting.
    """
    sentiment = 0.0

    # --- Step A: parse tags
    tags = post.get("tags", [])
    # If you see "bullish" => +0.5, "bearish" => -0.5
    if "bullish" in tags:
        sentiment += 0.5
    if "bearish" in tags:
        sentiment -= 0.5
    # If you want, parse "lol", "hot", "rising" or something else

    # --- Step B: parse "votes"
    votes = post.get("votes", {})
    # Some typical fields: negative, positive, important, liked, disliked, lol, toxic, saved, comments
    positive_votes = votes.get("positive", 0)
    negative_votes = votes.get("negative", 0)
    liked_votes = votes.get("liked", 0)
    disliked_votes = votes.get("disliked", 0)

    # Weight them as you like:
    # e.g. +0.02 for each positive, -0.02 for each negative, etc.
    # This is naive — tweak as needed.
    sentiment += positive_votes * 0.02
    sentiment -= negative_votes * 0.02
    sentiment += liked_votes * 0.01
    sentiment -= disliked_votes * 0.01

    # If you want to clamp the final result to [-1, +1], do this:
    sentiment = max(-1.0, min(1.0, sentiment))

    return sentiment

if __name__ == "__main__":
    fetch_cryptopanic_data()
