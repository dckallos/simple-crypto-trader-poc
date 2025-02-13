# =============================================================================
# FILE: db.py
# =============================================================================
"""
db.py

SQLite database utilities for:
1) Recording final (completed) trades in the 'trades' table. (No pending states here!)
2) Storing ephemeral/pending trades in the new 'pending_trades' table.
3) Storing kraken asset-pair info in 'kraken_asset_pairs' for minOrder and other fields.
4) Logging ledger entries, price history, cryptopanic data, lunarcrush data, AI context, etc.

NOTE: This file has been updated to reflect the new design where 'trades' only
holds finalized/closed trades, while 'pending_trades' holds ephemeral states
('pending','open','rejected','closed'), but we are *not* tracking partial fills
there. We also added 'kraken_asset_pairs' table with expanded columns to store
Kraken /AssetPairs data for each pair.

All other existing functionality not relevant to the newly introduced logic is
left intact, but references to "pending" or partial-fill logic in 'trades' have
been removed. Sub-position logic is handled in risk_manager.py or by your app.

CHANGES summary (search for "## NEW or ## CHANGED" comments below for modifications).
"""

import sqlite3
import time
import logging
import os
from typing import List, Dict, Any
from config_loader import ConfigLoader

logger = logging.getLogger(__name__)

# Path to your SQLite DB file
DB_FILE = "trades.db"

# ------------------------------------------------------------------------------
# Basic init / create tables
# ------------------------------------------------------------------------------
def init_db():
    """
    Creates all needed tables in the SQLite DB specified by DB_FILE, if not exist:
      - trades          (for final, completed trades only)
      - pending_trades  (new table for ephemeral pending trades)
      - price_history
      - cryptopanic_posts
      - lunarcrush_data
      - ai_context
      - ai_decisions
      - lunarcrush_timeseries
      - kraken_asset_pairs (new comprehensive table for /AssetPairs)
      - ledger_entries  (for ledger data)
    """
    init_ledger_table(DB_FILE)
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()

        # ----------------------------------------------------------------------
        # 1) TRADES (FINAL ONLY)
        # ----------------------------------------------------------------------
        ## CHANGED: Removing 'pending','open','part_filled','rejected' statuses, or partial fill columns.
        ## We'll create a simpler trades table for final records only.
        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    kraken_trade_id TEXT,
                    pair TEXT,
                    side TEXT,         -- 'BUY' or 'SELL'
                    quantity REAL,
                    price REAL,
                    order_id TEXT,
                    fee REAL DEFAULT 0,
                    realized_pnl REAL
                )
        """)

        # ----------------------------------------------------------------------
        # 2) PENDING_TRADES (NEW)
        # ----------------------------------------------------------------------
        create_pending_trades_table(conn)  # calls a helper function below

        # ----------------------------------------------------------------------
        # 3) PRICE_HISTORY
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
        # 4) CRYPTOPANIC_POSTS
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
        # 5) LUNARCRUSH_DATA
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
        # 6) AI_CONTEXT
        # ----------------------------------------------------------------------
        c.execute("""
            CREATE TABLE IF NOT EXISTS ai_context (
                id INTEGER PRIMARY KEY,
                context TEXT
            )
        """)

        # ----------------------------------------------------------------------
        # 7) AI_DECISIONS
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
        # 8) KRAKEN_ASSET_PAIRS (NEW or EXPANDED)
        # ----------------------------------------------------------------------
        create_kraken_asset_pairs_table(conn)

        conn.commit()
        logger.info("All DB tables ensured in init_db().")

    except Exception as e:
        logger.exception(f"Error creating DB: {e}")
    finally:
        conn.close()

# ------------------------------------------------------------------------------
# LEDGER ENTRIES
# ------------------------------------------------------------------------------
def init_ledger_table(db_path: str = DB_FILE):
    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS ledger_entries (
                ledger_id TEXT PRIMARY KEY,  -- e.g. L4UESK-KG3EQ-UFO4T5
                refid TEXT,
                time REAL,
                type TEXT,
                subtype TEXT,
                asset TEXT,
                amount REAL,
                fee REAL,
                balance REAL
            )
        """)
        conn.commit()
    except Exception as e:
        logger.exception(f"Error creating ledger_entries table: {e}")
    finally:
        conn.close()

# ------------------------------------------------------------------------------
# PRICE HISTORY & CRYPTOPANIC
# ------------------------------------------------------------------------------
def store_price_history(pair: str, bid: float, ask: float, last: float, volume: float):
    """
    Inserts a new record into the 'price_history' table for real-time price data.
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

# ------------------------------------------------------------------------------
# TRADES (FINAL) - We keep record_trade_in_db for legacy, but emphasize it's final
# ------------------------------------------------------------------------------
def record_trade_in_db(
    side: str,
    quantity: float,
    price: float,
    order_id: str,
    pair: str = "ETH/USD",
    fee: float = 0.0,
    kraken_trade_id: str = None
):
    """
    Inserts a final, *fully executed* record into the 'trades' table.
    Extended to include 'kraken_trade_id' so that we can store the unique
    trade ID from ownTrades. 'order_id' typically refers to the Kraken
    order ID for the overall order, while 'kraken_trade_id' can store the
    fill-level ID.

    :param side: 'BUY' or 'SELL'
    :param quantity: final filled quantity
    :param price: final average fill price (or fill price)
    :param order_id: The Kraken order ID or some local reference
    :param pair: e.g. 'ETH/USD'
    :param fee: total fee paid for this fill
    :param kraken_trade_id: The unique trade fill ID from ownTrades
    """
    import time
    import sqlite3

    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute(f"""
            INSERT INTO trades (
                timestamp,
                pair,
                side,
                quantity,
                price,
                order_id,
                fee,
                kraken_trade_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(time.time()),  # or pass in an explicit fill timestamp if needed
            pair,
            side,
            quantity,
            price,
            order_id,
            fee,
            kraken_trade_id
        ))
        conn.commit()
    except Exception as e:
        logger.exception(f"Error recording trade in DB: {e}")
    finally:
        conn.close()

# ------------------------------------------------------------------------------
# GPT context loading/saving
# ------------------------------------------------------------------------------
def load_gpt_context_from_db() -> str:
    """
    Returns the 'context' field from ai_context WHERE id=1, or "" if none.
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

# =============================================================================
# CREATE PENDING_TRADES TABLE
# =============================================================================
## NEW: We'll store ephemeral/pending trades here, with minimal columns. No partial fill logic.
def create_pending_trades_table(conn=None):
    close_conn = False
    if conn is None:
        conn = sqlite3.connect(DB_FILE)
        close_conn = True

    try:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS pending_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at INTEGER,
                pair TEXT,
                side TEXT,
                requested_qty REAL,
                status TEXT,               -- 'pending','open','closed','rejected'
                kraken_order_id TEXT,
                reason TEXT
            )
        """)
        conn.commit()
        logger.info("[DB] pending_trades table ensured.")
    except Exception as e:
        logger.exception(f"Error creating pending_trades table: {e}")
    finally:
        if close_conn:
            conn.close()

# ------------------------------------------------------------------------------
# CRUD for PENDING_TRADES
# ------------------------------------------------------------------------------
## CHANGED: No partial fill columns or partial fill logic is included.
def create_pending_trade(side: str, requested_qty: float, pair: str, reason: str = None) -> int:
    """
    Insert a new row in 'pending_trades' with status='pending'.
    Returns newly inserted row ID.
    """
    conn = sqlite3.connect(DB_FILE)
    row_id = None
    try:
        c = conn.cursor()
        c.execute("""
            INSERT INTO pending_trades (
                created_at, pair, side, requested_qty, status, kraken_order_id, reason
            )
            VALUES (?, ?, ?, ?, 'pending', NULL, ?)
        """, (int(time.time()), pair, side, requested_qty, reason))
        conn.commit()
        row_id = c.lastrowid
    except Exception as e:
        logger.exception(f"Error creating pending trade: {e}")
    finally:
        conn.close()
    return row_id


def mark_pending_trade_open(pending_id: int, kraken_order_id: str):
    """
    If Kraken accepts the order, set status='open' and store kraken_order_id.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            UPDATE pending_trades
            SET status='open', kraken_order_id=?
            WHERE id=?
        """, (kraken_order_id, pending_id))
        conn.commit()
        if c.rowcount < 1:
            logger.warning(f"No pending_trades row found with id={pending_id} to mark open.")
    except Exception as e:
        logger.exception(f"Error marking pending trade open: {e}")
    finally:
        conn.close()


def mark_pending_trade_rejected(pending_id: int, reason: str = None):
    """
    If Kraken rejects the order for any reason, set status='rejected'.
    Optionally store a reason string.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            UPDATE pending_trades
            SET status='rejected', reason=?
            WHERE id=?
        """, (reason, pending_id))
        conn.commit()
        if c.rowcount < 1:
            logger.warning(f"No pending_trades row found with id={pending_id} to reject.")
    except Exception as e:
        logger.exception(f"Error marking pending trade rejected: {e}")
    finally:
        conn.close()


# def set_pending_trade_kraken_id(pending_id: int, kraken_order_id: str):
#     """
#     Sets kraken_order_id on an existing pending trade row by its local 'id'.
#     This is used right after we receive 'status':'ok' with a new Kraken txid,
#     so future updates can match by kraken_order_id.
#     """
#     conn = sqlite3.connect(DB_FILE)
#     try:
#         c = conn.cursor()
#         c.execute("""
#             UPDATE pending_trades
#             SET kraken_order_id = ?
#             WHERE id = ?
#         """, (kraken_order_id, pending_id))
#         conn.commit()
#         if c.rowcount < 1:
#             logger.warning(f"No pending_trades row found with id={pending_id} to set kraken_order_id.")
#         else:
#             logger.info(f"Set kraken_order_id={kraken_order_id} for pending_id={pending_id}.")
#     except Exception as e:
#         logger.exception(f"Error setting pending trade kraken_order_id: {e}")
#     finally:
#         conn.close()


def set_kraken_order_id_for_pending_trade(pending_id: int, kraken_order_id: str):
    """
    Assigns the kraken_order_id (txid from Kraken) to our local pending_trades row
    once we know which pending_id it belongs to.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            UPDATE pending_trades
               SET kraken_order_id = ?
             WHERE id = ?
        """, (kraken_order_id, pending_id))
        conn.commit()
        if c.rowcount < 1:
            logger.warning(f"[DB] No pending_trades row found with id={pending_id} to set kraken_order_id.")
        else:
            logger.info(f"[DB] Assigned kraken_order_id={kraken_order_id} to pending_id={pending_id}.")
    except Exception as e:
        logger.exception(f"Error setting kraken_order_id: {e}")
    finally:
        conn.close()


def mark_pending_trade_closed(pending_id: int, reason: str = None):
    """
    If the trade is fully filled, mark status='closed'.
    We'll also create a final row in 'trades' outside (ws_data_feed or whichever).
    This just updates the status in pending_trades for traceability.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            UPDATE pending_trades
            SET status='closed', reason=?
            WHERE id=?
        """, (reason, pending_id))
        conn.commit()
        if c.rowcount < 1:
            logger.warning(f"No pending_trades row found with id={pending_id} to close.")
    except Exception as e:
        logger.exception(f"Error marking pending trade closed: {e}")
    finally:
        conn.close()


# =============================================================================
# CREATE KRAKEN_ASSET_PAIRS TABLE (expanded)
# =============================================================================
## NEW: We store a wide range of columns from the /AssetPairs response
def create_kraken_asset_pairs_table(conn=None):
    close_conn = False
    if conn is None:
        conn = sqlite3.connect(DB_FILE)
        close_conn = True

    try:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS kraken_asset_pairs (
                pair_name TEXT PRIMARY KEY,    -- e.g. 'XXBTZUSD','XETHXXBT'
                altname TEXT,
                wsname TEXT,
                aclass_base TEXT,
                base TEXT,
                aclass_quote TEXT,
                quote TEXT,
                lot TEXT,
                cost_decimals INTEGER,
                pair_decimals INTEGER,
                lot_decimals INTEGER,
                lot_multiplier INTEGER,
                leverage_buy TEXT,  -- stored as JSON string
                leverage_sell TEXT, -- stored as JSON
                fees TEXT,          -- stored as JSON
                fees_maker TEXT,    -- stored as JSON
                fee_volume_currency TEXT,
                margin_call INTEGER,
                margin_stop INTEGER,
                ordermin TEXT,       -- store as string or float if you prefer
                costmin TEXT,        -- store as string or float
                tick_size TEXT,      -- store as string or float
                status TEXT,
                long_position_limit INTEGER,
                short_position_limit INTEGER,
                last_updated INTEGER
            )
        """)
        conn.commit()
        logger.info("[DB] kraken_asset_pairs table ensured.")
    except Exception as e:
        logger.exception(f"Error creating kraken_asset_pairs table: {e}")
    finally:
        if close_conn:
            conn.close()

def store_kraken_asset_pair_info(pair_name: str, pair_info: Dict[str, Any]):
    """
    Insert or replace the row in kraken_asset_pairs with the data from pair_info.
    We'll store arrays (leverage, fees) as JSON strings for simplicity.
    `pair_info` is the dict for that pair from the /AssetPairs result.
    """
    import json
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        now_ts = int(time.time())
        # Extract fields
        altname = pair_info.get("altname","")
        wsname = pair_info.get("wsname","")
        aclass_base = pair_info.get("aclass_base","")
        base = pair_info.get("base","")
        aclass_quote = pair_info.get("aclass_quote","")
        quote = pair_info.get("quote","")
        lot = pair_info.get("lot","")
        cost_decimals = pair_info.get("cost_decimals",0)
        pair_decimals = pair_info.get("pair_decimals",0)
        lot_decimals = pair_info.get("lot_decimals",0)
        lot_multiplier = pair_info.get("lot_multiplier",0)
        # Convert arrays to JSON
        leverage_buy = json.dumps(pair_info.get("leverage_buy", []))
        leverage_sell = json.dumps(pair_info.get("leverage_sell", []))
        fees = json.dumps(pair_info.get("fees", []))
        fees_maker = json.dumps(pair_info.get("fees_maker", []))
        fee_volume_currency = pair_info.get("fee_volume_currency","")
        margin_call = pair_info.get("margin_call",0)
        margin_stop = pair_info.get("margin_stop",0)
        ordermin = pair_info.get("ordermin","")
        costmin = pair_info.get("costmin","")
        tick_size = pair_info.get("tick_size","")
        status = pair_info.get("status","")
        long_pos_limit = pair_info.get("long_position_limit",0)
        short_pos_limit = pair_info.get("short_position_limit",0)

        c.execute("""
            INSERT OR REPLACE INTO kraken_asset_pairs (
                pair_name, altname, wsname, aclass_base, base, aclass_quote, quote,
                lot, cost_decimals, pair_decimals, lot_decimals, lot_multiplier,
                leverage_buy, leverage_sell, fees, fees_maker, fee_volume_currency,
                margin_call, margin_stop, ordermin, costmin, tick_size, status,
                long_position_limit, short_position_limit, last_updated
            )
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            pair_name,
            altname,
            wsname,
            aclass_base,
            base,
            aclass_quote,
            quote,
            lot,
            cost_decimals,
            pair_decimals,
            lot_decimals,
            lot_multiplier,
            leverage_buy,
            leverage_sell,
            fees,
            fees_maker,
            fee_volume_currency,
            margin_call,
            margin_stop,
            ordermin,
            costmin,
            tick_size,
            status,
            long_pos_limit,
            short_pos_limit,
            now_ts
        ))
        conn.commit()
        logger.info(f"Upserted asset pair info for {pair_name} into kraken_asset_pairs.")
    except Exception as e:
        logger.exception(f"Error storing asset pair info for {pair_name}: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()
