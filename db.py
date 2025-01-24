# =============================================================================
# FILE: db.py
# =============================================================================
"""
db.py

SQLite database utilities for:
1) Recording executed trades in the 'trades' table.
2) Storing price history in 'price_history'.
3) Collecting aggregator data in 'cryptopanic_news' and 'lunarcrush_data'.
4) Maintaining GPT context in 'ai_context'.

Note: The new sub-positions approach is now handled entirely in
      `risk_manager.py` and its `RiskManagerDB` class, which creates and manages
      the 'sub_positions' table. We've removed all references to the old
      'open_positions' table logic, as that no longer "fits" the multi-position
      design.

Usage:
    - Call init_db() once on startup to ensure main tables exist.
    - For sub-positions, see `RiskManagerDB.initialize()` in risk_manager.py.
"""

import sqlite3
import time
import logging

logger = logging.getLogger(__name__)

DB_FILE = "trades.db"

def init_db():
    """
    Creates the 'trades', 'price_history', 'cryptopanic_news', and 'lunarcrush_data'
    tables if they do not exist. Also creates the 'ai_context' table.

    If you are using the multi-sub-position approach, you should initialize the
    'sub_positions' table from `risk_manager.py` via RiskManagerDB.initialize().
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()

        # ----------------------------------------------------------------------
        # 1) 'trades' table
        # ----------------------------------------------------------------------
        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                pair TEXT,
                side TEXT,       -- 'BUY' or 'SELL'
                quantity REAL,
                price REAL,
                order_id TEXT
            )
        """)

        # ----------------------------------------------------------------------
        # 2) 'price_history' table
        # ----------------------------------------------------------------------
        c.execute("""
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                pair TEXT,
                bid_price REAL,
                ask_price REAL,
                last_price REAL,
                volume REAL
            )
        """)

        # ----------------------------------------------------------------------
        # 3) 'cryptopanic_news' table
        # ----------------------------------------------------------------------
        c.execute("""
            CREATE TABLE IF NOT EXISTS cryptopanic_news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                title TEXT,
                url TEXT,
                sentiment_score REAL
            )
        """)

        # ----------------------------------------------------------------------
        # 4) 'lunarcrush_data' table
        # ----------------------------------------------------------------------
        c.execute("""
            CREATE TABLE IF NOT EXISTS lunarcrush_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                symbol TEXT,
                name TEXT,
                price REAL,
                market_cap REAL,
                volume_24h REAL,
                volatility REAL,
                percent_change_1h REAL,
                percent_change_24h REAL,
                percent_change_7d REAL,
                percent_change_30d REAL,
                social_volume_24h REAL,
                interactions_24h REAL,
                social_dominance REAL,
                galaxy_score REAL,
                alt_rank INTEGER,
                sentiment REAL,
                categories TEXT,
                topic TEXT,
                logo TEXT
            )
        """)

        # ----------------------------------------------------------------------
        # 5) 'ai_context' table
        # ----------------------------------------------------------------------
        c.execute("""
            CREATE TABLE IF NOT EXISTS ai_context (
                id INTEGER PRIMARY KEY,
                context TEXT
            )
        """)

        conn.commit()
    except Exception as e:
        logger.exception(f"Error creating DB: {e}")
    finally:
        conn.close()


def record_trade_in_db(side: str, quantity: float, price: float, order_id: str, pair="ETH/USD"):
    """
    Inserts a new record into the 'trades' table.

    :param side: 'BUY' or 'SELL'.
    :param quantity: The size of the trade in base units (e.g. BTC quantity).
    :param price: Fill price (float).
    :param order_id: The ID returned by place_order (mock or real).
    :param pair: Which trading pair was traded.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            INSERT INTO trades (timestamp, pair, side, quantity, price, order_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            int(time.time()),
            pair,
            side,
            quantity,
            price,
            order_id
        ))
        conn.commit()
    except Exception as e:
        logger.exception(f"Error recording trade in DB: {e}")
    finally:
        conn.close()


def store_price_history(pair: str, bid: float, ask: float, last: float, volume: float):
    """
    Inserts a new record into the 'price_history' table for real-time price data.

    :param pair: e.g. "XBT/USD"
    :param bid: Current bid price
    :param ask: Current ask price
    :param last: Last traded price
    :param volume: Volume in the last period or since last update
    """
    logger.debug(f"Inserting price for {pair}: last={last}, volume={volume}")
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            INSERT INTO price_history (timestamp, pair, bid_price, ask_price, last_price, volume)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            int(time.time()),
            pair,
            bid,
            ask,
            last,
            volume
        ))
        conn.commit()
    except Exception as e:
        logger.exception(f"Error storing price in DB: {e}")
    finally:
        conn.close()


def store_cryptopanic_data(title: str, url: str, sentiment_score: float):
    """
    Inserts a row into 'cryptopanic_news'.

    :param title: The news title or headline.
    :param url: Link to the article.
    :param sentiment_score: A numeric sentiment measure (e.g. from -1 to +1).
    """
    logger.info(f"Storing CryptoPanic news: title={title}, sentiment={sentiment_score}")
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            INSERT INTO cryptopanic_news (timestamp, title, url, sentiment_score)
            VALUES (?, ?, ?, ?)
        """, (
            int(time.time()),
            title,
            url,
            sentiment_score
        ))
        conn.commit()
    except Exception as e:
        logger.exception(f"Error storing cryptopanic data: {e}")
    finally:
        conn.close()


def store_lunarcrush_data(
    symbol: str,
    name: str,
    price: float,
    market_cap: float,
    volume_24h: float,
    volatility: float,
    percent_change_1h: float,
    percent_change_24h: float,
    percent_change_7d: float,
    percent_change_30d: float,
    social_volume_24h: float,
    interactions_24h: float,
    social_dominance: float,
    galaxy_score: float,
    alt_rank: int,
    sentiment: float,
    categories: str,
    topic: str,
    logo: str
):
    """
    Insert a row into 'lunarcrush_data' table with the specified metrics.

    :param symbol: e.g. "ETH"
    :param name: e.g. "Ethereum"
    :param price: Current price for the asset.
    :param market_cap: Current market cap.
    :param volume_24h: Trading volume in the last 24 hours.
    :param volatility: Volatility metric from LunarCrush.
    :param percent_change_1h: Price % change in last hour.
    :param percent_change_24h: Price % change in last 24h.
    :param percent_change_7d: Price % change in last 7 days.
    :param percent_change_30d: Price % change in last 30 days.
    :param social_volume_24h: Social mentions in last 24h.
    :param interactions_24h: Interactions (likes, shares, etc.) in 24h.
    :param social_dominance: The ratio of social volume vs. total across all assets.
    :param galaxy_score: LunarCrush metric (0..100).
    :param alt_rank: LunarCrush alt rank (lower is better).
    :param sentiment: 0..100 measure or a custom sentiment from LunarCrush.
    :param categories: Comma-separated categories, e.g. "defi,nft".
    :param topic: Additional text about the asset, if provided.
    :param logo: URL to the coin's logo or an image resource.
    """
    import sqlite3
    import time
    from db import DB_FILE, logger

    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            INSERT INTO lunarcrush_data (
                timestamp,
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
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(time.time()),
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
        ))
        conn.commit()
    except Exception as e:
        logger.exception(f"Error storing LunarCrush data: {e}")
    finally:
        conn.close()


def create_ai_context_table():
    """
    A simple table to store GPT conversation memory or context.
    Usually called within init_db() or manually if you want it separate.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS ai_context (
                id INTEGER PRIMARY KEY,
                context TEXT
            )
        """)
        conn.commit()
    except Exception as e:
        logger.exception(f"Error creating ai_context table: {e}")
    finally:
        conn.close()


def load_gpt_context_from_db():
    """
    Returns the 'context' field from ai_context WHERE id=1, or an empty string if none found.

    :return: str - the stored GPT context, or "" if not found.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("SELECT context FROM ai_context WHERE id=1")
        row = c.fetchone()
        if row and row[0]:
            return row[0]
        return ""
    except Exception as e:
        logger.exception(f"Error loading GPT context: {e}")
        return ""
    finally:
        conn.close()


def save_gpt_context_to_db(context_str: str):
    """
    Upsert the single row in 'ai_context' with id=1, storing context_str.

    :param context_str: The string representing GPT memory or conversation context.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            INSERT INTO ai_context (id, context)
            VALUES (1, ?)
            ON CONFLICT(id) DO UPDATE SET context=excluded.context
        """, (context_str,))
        conn.commit()
    except Exception as e:
        logger.exception(f"Error saving GPT context: {e}")
    finally:
        conn.close()
