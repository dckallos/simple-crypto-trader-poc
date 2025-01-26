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
5) Storing advanced time-series data in 'lunarcrush_timeseries'.
6) Logging final AI decisions in 'ai_decisions'.

For multi-sub-position logic, sub_positions is handled in `risk_manager.py` (RiskManagerDB).

Usage:
    - Call init_db() once on startup (or run this file directly) to ensure all tables exist.
    - Use store_*() helpers to insert or update data as needed.
    - For sub-positions, see `RiskManagerDB.initialize()` in risk_manager.py.
"""

import sqlite3
import time
import logging

logger = logging.getLogger(__name__)

DB_FILE = "trades.db"

def init_db():
    """
    Creates all needed tables:
      - trades
      - price_history
      - cryptopanic_news
      - lunarcrush_data
      - ai_context
      - ai_decisions
      - lunarcrush_timeseries
    If sub_positions are needed, see `risk_manager.py => RiskManagerDB.initialize()`.
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
            CREATE TABLE IF NOT EXISTS cryptopanic_posts (
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
                sentiment_score REAL
                image TEXT,
                description TEXT,
                panic_score REAL,
                created_at INTEGER
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

        # ----------------------------------------------------------------------
        # 6) 'ai_decisions' table
        # ----------------------------------------------------------------------
        c.execute("""
            CREATE TABLE IF NOT EXISTS ai_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                pair TEXT,
                action TEXT,
                size REAL,
                rationale TEXT
            )
        """)

        # ----------------------------------------------------------------------
        # 7) 'lunarcrush_timeseries' table
        # ----------------------------------------------------------------------
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

    import fetch_lunarcrush.init_lunarcrush as init_lunarcrush


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
    Returns the 'context' field from ai_context WHERE id=1, or "" if none.

    :return: str
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


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger.info("Running init_db() from db.py directly.")
    init_db()
    logger.info("DB initialization complete.")
