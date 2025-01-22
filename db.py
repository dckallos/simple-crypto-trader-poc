# =============================================================================
# FILE: db.py
# =============================================================================
"""
db.py

Minimal SQLite database usage to record trades and now price data.
We add a 'store_price_history' function for the WebSocket feed,
and a new table for CryptoPanic data with a function to insert rows.
"""

import sqlite3
import time
import logging

logger = logging.getLogger(__name__)

DB_FILE = "trades.db"

def init_db():
    """
    Creates the 'trades' table and the 'price_history' table if they don't exist.
    Also creates the 'cryptopanic_news' table to store news data.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        # Original 'trades' table
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

        # 'price_history' table
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

        # 'cryptopanic_news' table for storing aggregated or raw news data
        c.execute("""
            CREATE TABLE IF NOT EXISTS cryptopanic_news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                title TEXT,
                url TEXT,
                sentiment_score REAL
            )
        """)

        # 4) NEW: 'lunarcrush_data' table for storing metrics from LunarCrush
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

        conn.commit()
    except Exception as e:
        logger.exception(f"Error creating DB: {e}")
    finally:
        conn.close()

    create_open_positions_table()
    create_ai_context_table()


def record_trade_in_db(side: str, quantity: float, price: float, order_id: str, pair="ETH/USD"):
    """
    Inserts a new record into the 'trades' table.

    :param side: 'BUY' or 'SELL'.
    :param quantity: The size of the trade in base units (e.g., BTC quantity).
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
    Insert a row into 'lunarcrush_data' table.

    You must create (or update) this table in your DB schema, for example:

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
    );
    """
    import time
    import sqlite3
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

def create_open_positions_table():
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS open_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT UNIQUE,
                position_size REAL,
                avg_entry_price REAL,
                updated_at INTEGER
            )
        """)
        conn.commit()
    except Exception as e:
        logger.exception(f"Error creating open_positions table: {e}")
    finally:
        conn.close()


def load_positions_from_db():
    """
    Returns a dict: { "ETH/USD": {"size": 0.5, "basis": 1500.0}, ... }
    for all pairs found in 'open_positions'.
    """
    results = {}
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("SELECT pair, position_size, avg_entry_price FROM open_positions")
        rows = c.fetchall()
        for row in rows:
            pair, size, basis = row
            results[pair] = {"size": float(size), "basis": float(basis)}
    except Exception as e:
        logger.exception(f"Error loading positions from DB: {e}")
    finally:
        conn.close()
    return results

def save_position_to_db(pair: str, position_size: float, avg_entry_price: float):
    """
    Upserts a row in 'open_positions' for the given pair. If position_size=0 => remove row.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        ts = int(time.time())
        if abs(position_size) < 1e-12:
            # remove the row if position is effectively zero
            c.execute("DELETE FROM open_positions WHERE pair=?", (pair,))
        else:
            # upsert approach
            c.execute("""
                INSERT INTO open_positions (pair, position_size, avg_entry_price, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(pair) DO UPDATE SET
                  position_size=excluded.position_size,
                  avg_entry_price=excluded.avg_entry_price,
                  updated_at=excluded.updated_at
            """, (pair, position_size, avg_entry_price, ts))
        conn.commit()
    except Exception as e:
        logger.exception(f"Error saving position for {pair}: {e}")
    finally:
        conn.close()

def create_ai_context_table():
    """
    A simple table to store a single row of GPT conversation memory (or you can store multiple rows).
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
        # We can store just 1 row with id=1 if we want a single memory, or do a bigger approach
        conn.commit()
    except Exception as e:
        logger.exception(f"Error creating ai_context table: {e}")
    finally:
        conn.close()

def load_gpt_context_from_db():
    """
    Returns the 'context' field from ai_context WHERE id=1, or empty if none found.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("SELECT context FROM ai_context WHERE id=1")
        row = c.fetchone()
        if row and row[0]:
            return row[0]  # this might be a string with all prior GPT memory
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

