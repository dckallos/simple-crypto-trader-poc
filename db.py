# db.py

"""
db.py

Minimal SQLite database usage to record trades and now price data.
We add a 'store_price_history' function for the WebSocket feed.
"""

import sqlite3
import time
import logging

logger = logging.getLogger(__name__)

DB_FILE = "trades.db"

def init_db():
    """
    Creates the 'trades' table and the 'price_history' table if they don't exist.
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

        # New 'price_history' table
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

        conn.commit()
    except Exception as e:
        logger.exception(f"Error creating DB: {e}")
    finally:
        conn.close()

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
