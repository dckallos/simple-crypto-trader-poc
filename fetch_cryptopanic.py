#!/usr/bin/env python3
# =============================================================================
# FILE: fetch_cryptopanic.py
# =============================================================================
"""
fetch_cryptopanic.py

Enhancements over the basic version:
1) Parses post["votes"] to incorporate positive/negative counts into the sentiment score.
2) Checks post["tags"] for "bullish"/"bearish".
3) Optionally logs 'published_at', 'domain', etc. for potential DB storage.
4) Allows specifying a Python list of coin symbols, which are joined for the API call.

Usage:
    python fetch_cryptopanic.py
"""

import os
import requests
import logging
from typing import Dict, Any
from dotenv import load_dotenv

from db import store_cryptopanic_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------------------
# Configuration Toggles
# ------------------------------------------------------------------------------
use_public = False        # e.g. True => public feed
use_following = True      # e.g. True => uses your personal 'Following' feed
use_metadata = True       # e.g. True => include metadata fields from CryptoPanic
use_approved = True       # e.g. True => get "approved" links
filter_param = "rising"   # "rising", "hot", or other
kind_param = "news"       # "news" or "media"
regions_param = None      # e.g. "en,de"

# ------------------------------------------------------------------------------
# New: coin_list => create a Python list
# ------------------------------------------------------------------------------
coin_list = [
    "ETH", "XBT", "SOL", "ADA", "XRP", "LTC", "DOT", "MATIC", "LINK", "ATOM",
    "XMR", "TRX", "SHIB", "AVAX", "UNI", "APE", "FIL", "AAVE", "SAND", "CHZ",
    "GRT", "CRV", "COMP", "ALGO", "NEAR", "LDO", "XTZ", "EGLD", "KSM"
]

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
    if regions_param:
        params["regions"] = regions_param

    # NEW: Convert Python coin_list into comma-separated string
    # If it's empty, we skip passing 'currencies'
    if coin_list:
        joined_currencies = ",".join(coin_list)  # e.g. "ETH,XBT,SOL,ADA,..."
        params["currencies"] = joined_currencies

    try:
        logger.info(f"Fetching CryptoPanic data with params={params}")
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()

        data = resp.json()
        # data might have "count", "next", "previous", "results"
        posts = data.get("results", [])
        logger.info(f"Fetched {len(posts)} posts from CryptoPanic.")

        # For each post, parse sentiment and store
        for post in posts:
            # Title
            title = post.get("title", "No Title")
            # URL
            url = post.get("url", "")

            # domain and published_at if you want them
            domain = post.get("domain", "")
            published_at = post.get("published_at", "")

            # Our improved sentiment
            sentiment_score = compute_enhanced_sentiment(post)

            logger.debug(
                f"POST => Title: {title} | domain={domain} | published_at={published_at} "
                f"| votes={post.get('votes')} | tags={post.get('tags')} "
                f"| sentiment={sentiment_score}"
            )

            # Store minimal fields in DB for now
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

    # A) parse tags
    tags = post.get("tags", [])
    # If you see "bullish" => +0.5, "bearish" => -0.5
    if "bullish" in tags:
        sentiment += 0.5
    if "bearish" in tags:
        sentiment -= 0.5
    # optionally handle "lol", "hot", or others

    # B) parse "votes"
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

    # clamp to [-1, 1]
    sentiment = max(-1.0, min(1.0, sentiment))

    return sentiment

if __name__ == "__main__":
    fetch_cryptopanic_data()
