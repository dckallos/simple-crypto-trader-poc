# =============================================================================
# FILE: db.py
# =============================================================================
"""
db.py

SQLite database utilities for:
1) Recording executed trades in the 'trades' table.
2) Storing price history in 'price_history'.
3) Collecting aggregator data in 'cryptopanic_posts' and 'lunarcrush_data'.
4) Maintaining GPT context in 'ai_context'.
5) Logging final AI decisions in 'ai_decisions'.
6) Storing advanced time-series data in 'lunarcrush_timeseries'.

For multi-sub-position logic, see `RiskManagerDB` in risk_manager.py.

Extended Columns for partial fill logic:
    - In the 'trades' table, columns have been added so we can track pending orders,
      partial fills, rejections, or final fills:
        status TEXT        -- 'pending','open','part_filled','closed','rejected'
        filled_size REAL   -- total filled so far
        avg_fill_price REAL-- weighted average fill price for the portion filled
        fee REAL           -- cumulative fee so far

Usage:
    - Call init_db() once on startup (or run this file directly) to ensure all tables exist.
    - Use record_trade_in_db(), store_price_history(), store_cryptopanic_data(), etc.
    - GPT context stored in 'ai_context' => load_gpt_context_from_db() & save_gpt_context_to_db().
    - For sub-positions, see risk_manager.py => RiskManagerDB.
    - (NEW) create_pending_trade(...), update_trade_fill(...), set_trade_rejected(...)
      can help you manage partial-fills or rejections in 'trades'.
"""

import sqlite3
import time
import logging

logger = logging.getLogger(__name__)

DB_FILE = "trades.db"


def init_db():
    """
    Creates all needed tables in the SQLite DB specified by DB_FILE, if not exist:
      - trades (with extended columns for partial-fill tracking)
      - price_history
      - cryptopanic_posts
      - lunarcrush_data
      - ai_context
      - ai_decisions
      - lunarcrush_timeseries

    If sub_positions are needed, see `risk_manager.py => RiskManagerDB.initialize()`.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()

        # 1) trades
        #    Now includes columns for partial fill logic: status, filled_size, avg_fill_price, fee
        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                pair TEXT,
                side TEXT,
                quantity REAL,
                price REAL,
                order_id TEXT,
                status TEXT,
                filled_size REAL DEFAULT 0.0, 
                avg_fill_price REAL DEFAULT 0.0,
                fee REAL DEFAULT 0.0
            )
        """)

        # 2) price_history
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
        # 3) 'cryptopanic_posts' table
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
                sentiment_score REAL,
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
                lunarcrush_id INTEGER,
                symbol TEXT,
                name TEXT,
                price REAL,
                price_btc REAL,
                volume_24h REAL,
                volatility REAL,
                circulating_supply REAL,
                max_supply REAL,
                percent_change_1h REAL,
                percent_change_24h REAL,
                percent_change_7d REAL,
                percent_change_30d REAL,
                market_cap REAL,
                market_cap_rank INTEGER,
                interactions_24h REAL,
                social_volume_24h REAL,
                social_dominance REAL,
                market_dominance REAL,
                market_dominance_prev REAL,
                galaxy_score REAL,
                galaxy_score_previous REAL,
                alt_rank INTEGER,
                alt_rank_previous INTEGER,
                sentiment REAL,
                categories TEXT,
                topic TEXT,
                logo TEXT,
                blockchains TEXT
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
        logger.info("All DB tables ensured in init_db().")
    except Exception as e:
        logger.exception(f"Error creating DB: {e}")
    finally:
        conn.close()


def record_trade_in_db(side: str, quantity: float, price: float, order_id: str, pair="ETH/USD"):
    """
    Inserts a new record into the 'trades' table. This was originally used to
    record a final trade after it's been executed. By default, you might treat
    this as a fully filled trade (status='closed').

    :param side: 'BUY' or 'SELL'.
    :param quantity: The final filled size in base units (e.g. BTC quantity).
    :param price: Fill price (float).
    :param order_id: The ID returned by place_order (mock or real).
    :param pair: Which trading pair was traded.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            INSERT INTO trades (
                timestamp, pair, side, quantity, price, order_id,
                status, filled_size, avg_fill_price, fee
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(time.time()),
            pair,
            side,
            quantity,
            price,
            order_id,
            'closed',     # For this legacy function, we treat it as fully filled
            quantity,     # filled_size = quantity
            price,        # avg_fill_price = final fill price
            0.0           # fee = 0 by default here
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
    Inserts a row into 'cryptopanic_posts'.

    :param title: The news title or headline.
    :param url: Link to the article.
    :param sentiment_score: A numeric sentiment measure (e.g. from -1 to +1).
    """
    logger.info(f"Storing CryptoPanic news: title={title}, sentiment={sentiment_score}")
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            INSERT INTO cryptopanic_posts (created_at, title, url, sentiment_score)
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

    Returns: None. Data is inserted into 'lunarcrush_data'.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            INSERT INTO lunarcrush_data (
                timestamp, symbol, name, price, market_cap, volume_24h, volatility,
                percent_change_1h, percent_change_24h, percent_change_7d, percent_change_30d,
                social_volume_24h, interactions_24h, social_dominance, galaxy_score,
                alt_rank, sentiment, categories, topic, logo
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            int(time.time()), symbol, name, price, market_cap, volume_24h, volatility,
            percent_change_1h, percent_change_24h, percent_change_7d, percent_change_30d,
            social_volume_24h, interactions_24h, social_dominance, galaxy_score,
            alt_rank, sentiment, categories, topic, logo
        ])
        conn.commit()
        logger.info(f"Inserted row into lunarcrush_data for symbol={symbol}.")
    except Exception as e:
        logger.exception(f"Error storing lunarcrush data: {e}")
    finally:
        conn.close()


def load_gpt_context_from_db() -> str:
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

    :param context_str: The conversation context to store.
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


# =============================================================================
# New or Extended Functions for Partial Fill Logic
# =============================================================================

def create_pending_trade(side: str, requested_qty: float, pair: str, order_id: str) -> int:
    """
    Creates a new row in 'trades' with status='pending' (useful if we're
    waiting on Kraken fill). The 'price' can remain 0 or indicative if we want
    to store the intended limit or last known price. For a market order, you
    can pass 0 or the approximate quote.

    Returns: The newly inserted row's ID (primary key).
    """
    conn = sqlite3.connect(DB_FILE)
    row_id = None
    try:
        ts_now = int(time.time())
        c = conn.cursor()
        c.execute("""
            INSERT INTO trades (
                timestamp, pair, side, quantity, price, order_id,
                status, filled_size, avg_fill_price, fee
            )
            VALUES (?, ?, ?, ?, ?, ?, 'pending', 0.0, 0.0, 0.0)
        """, (
            ts_now, pair, side, requested_qty, 0.0, order_id
        ))
        conn.commit()
        row_id = c.lastrowid
    except Exception as e:
        logger.exception(f"Error creating pending trade row: {e}")
    finally:
        conn.close()

    return row_id


def update_trade_fill(order_id: str, filled_size: float, avg_fill_price: float, fee: float, status: str):
    """
    Updates an existing 'trades' row identified by order_id, setting:
      - filled_size (cumulative quantity executed so far)
      - avg_fill_price (weighted average fill price so far)
      - fee (cumulative fees so far)
      - status (e.g. 'part_filled','closed')

    For partial fills over time, call this each time new fill data arrives from
    the private feed. For example, if new_filled_size increments from 0.02 -> 0.03,
    you'll store the updated total and any updated fee.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            UPDATE trades
            SET filled_size = ?,
                avg_fill_price = ?,
                fee = ?,
                status = ?
            WHERE order_id = ?
        """, (filled_size, avg_fill_price, fee, status, order_id))
        conn.commit()
        if c.rowcount < 1:
            logger.warning(f"No trade row found for order_id={order_id} when updating fill.")
    except Exception as e:
        logger.exception(f"Error updating trade fill for order_id={order_id}: {e}")
    finally:
        conn.close()


def set_trade_rejected(order_id: str):
    """
    If Kraken says the order is invalid or insufficient funds, we mark
    the 'trades' row as status='rejected'. Optionally you can also fill
    in an error message in another column if you want to store more detail.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            UPDATE trades
            SET status = 'rejected'
            WHERE order_id = ?
        """, (order_id,))
        conn.commit()
        if c.rowcount < 1:
            logger.warning(f"No trade row found for order_id={order_id} to set rejected.")
    except Exception as e:
        logger.exception(f"Error setting trade rejected for order_id={order_id}: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger.info("Running init_db() from db.py directly.")
    init_db()
    logger.info("DB initialization complete.")
