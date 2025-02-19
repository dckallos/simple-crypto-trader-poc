# =============================================================================
# FILE: db.py
# =============================================================================
"""
db.py

SQLite database utilities for:

1) Recording final (completed) trades in the 'trades' table. (No pending states here!)
2) Storing ephemeral/pending trades in the 'pending_trades' table.
3) Storing kraken asset-pair info in 'kraken_asset_pairs' for minOrder and other fields.
4) Logging ledger entries, price history, cryptopanic data, lunarcrush data, AI context, etc.
5) NEW: Storing AI snapshot data for ML usage (the 'ai_snapshots' table).

Additional updates:
-------------------
 - The 'ai_snapshots' table stores a wide variety of numeric and textual data:
   * The vector of price changes used in the AI prompt (JSON)
   * The last_price or other numeric stats
   * A JSON field capturing LunarCrush or volatility data
   * Some columns for “market conditions,” “bullish/bearish,” “risk_of_investment,” etc.
   * Plus references to interval durations, stop_loss_pct, and so forth

 - We have also added a short function store_ai_snapshot(...) to handle inserts.
 - This schema is designed with a future Postgres migration in mind, using simple
   typed columns (INTEGER/REAL/TEXT). Additional indexing can be done later.
"""

import sqlite3
import time
import logging
import os
from typing import List, Dict, Any, Tuple
import asyncio
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

      - trades (for final, completed trades only)
      - pending_trades (ephemeral pending trades)
      - price_history
      - cryptopanic_posts
      - lunarcrush_data
      - ai_context
      - ai_decisions
      - lunarcrush_timeseries
      - kraken_asset_pairs
      - ledger_entries
      - NEW: ai_snapshots (stores extended info for ML)

    Then runs migrations on 'trades' to ensure extended analytics columns exist.
    """
    init_ledger_table(DB_FILE)
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()

        # ----------------------------------------------------------------------
        # 1) TRADES (FINAL ONLY)
        # ----------------------------------------------------------------------
        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                kraken_trade_id TEXT,
                pair TEXT,
                side TEXT,
                quantity REAL,
                price REAL,
                order_id TEXT,
                fee REAL DEFAULT 0,
                realized_pnl REAL,
                source TEXT,
                rationale TEXT
            )
        """)

        # Run a short migration after creating trades:
        _migrate_trades_table_for_analytics(conn)

        # ----------------------------------------------------------------------
        # 2) PENDING_TRADES
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
        # 8) KRAKEN_ASSET_PAIRS
        # ----------------------------------------------------------------------
        create_kraken_asset_pairs_table(conn)

        # ----------------------------------------------------------------------
        # 9) KRAKEN_BALANCE_HISTORY
        # ----------------------------------------------------------------------
        create_kraken_balance_table(conn)

        # ----------------------------------------------------------------------
        # 10) AI_SNAPSHOTS (NEW FOR ML)
        # ----------------------------------------------------------------------
        create_ai_snapshots_table(conn)

        conn.commit()
        logger.info("All DB tables ensured in init_db().")

    except Exception as e:
        logger.exception(f"Error creating DB: {e}")
    finally:
        conn.close()


def _migrate_trades_table_for_analytics(conn: sqlite3.Connection):
    """
    Checks if certain analytic columns exist in 'trades' and adds them if missing:

      - lot_id (INTEGER) => link to holding_lots.id
      - ai_model (TEXT)
      - source_config (TEXT) => store JSON risk config
      - ai_rationale (TEXT) => store full GPT rationale
      - exchange_fill_time (INTEGER) => actual exchange fill time
      - trade_metrics_json (TEXT) => store peak adverse/favorable excursion, etc.
    """
    try:
        c = conn.cursor()
        c.execute("PRAGMA table_info(trades)")
        rows = c.fetchall()
        existing_cols = [row[1] for row in rows]

        add_columns = []

        if "lot_id" not in existing_cols:
            add_columns.append(("lot_id", "INTEGER"))
        if "ai_model" not in existing_cols:
            add_columns.append(("ai_model", "TEXT"))
        if "source_config" not in existing_cols:
            add_columns.append(("source_config", "TEXT"))
        if "ai_rationale" not in existing_cols:
            add_columns.append(("ai_rationale", "TEXT"))
        if "exchange_fill_time" not in existing_cols:
            add_columns.append(("exchange_fill_time", "INTEGER"))
        if "trade_metrics_json" not in existing_cols:
            add_columns.append(("trade_metrics_json", "TEXT"))

        for col_name, col_type in add_columns:
            logger.info(f"[DB MIGRATION] Adding column '{col_name}' to trades (type={col_type}).")
            c.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")

        if add_columns:
            conn.commit()
            logger.info(f"[DB MIGRATION] trades => added {len(add_columns)} new columns => {add_columns}")
    except Exception as e:
        logger.exception(f"[DB MIGRATION] Error migrating trades => {e}")


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


# =============================================================================
# CREATE AI_SNAPSHOTS TABLE (NEW)
# =============================================================================

def create_ai_snapshots_table(conn=None):
    """
    Creates the 'ai_snapshots' table. This table is used to store extensive
    contextual data for each AI inference or aggregator cycle, facilitating
    future machine learning and analytics.

    We store:
      - id (PK)
      - created_at (INT epoch)
      - pair (TEXT) -> for multi-coin calls, you can store 'ALL' or repeated inserts
      - aggregator_interval_secs (INT) -> from config
      - stop_loss_pct, take_profit_pct, daily_drawdown_limit (REAL) -> risk configs
      - price_changes_json (TEXT) -> the vector of minute-based % changes used in prompt
      - last_price (REAL) -> the last known price
      - lunarcrush_data_json (TEXT) -> wide JSON snapshot of LunarCrush metrics if needed
      - coin_volatility (REAL) -> from LunarCrush or aggregator
      - is_market_bullish (TEXT or boolean) -> user-labeled
      - risk_estimate (REAL) -> an example numeric score
      - notes (TEXT) -> for anything else

    (You can expand as needed!)
    """
    close_conn = False
    if conn is None:
        conn = sqlite3.connect(DB_FILE)
        close_conn = True

    try:
        c = conn.cursor()
        # If you want typed columns for easier Postgres migration, we do so here:
        c.execute("""
            CREATE TABLE IF NOT EXISTS ai_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at INTEGER,
                pair TEXT,
                aggregator_interval_secs INTEGER,
                stop_loss_pct REAL,
                take_profit_pct REAL,
                daily_drawdown_limit REAL,
                price_changes_json TEXT,
                last_price REAL,
                lunarcrush_data_json TEXT,
                coin_volatility REAL,
                is_market_bullish TEXT,
                risk_estimate REAL,
                notes TEXT
            )
        """)
        conn.commit()
        logger.info("[DB] ai_snapshots table ensured.")
    except Exception as e:
        logger.exception(f"[DB] Error creating ai_snapshots table: {e}")
    finally:
        if close_conn:
            conn.close()


def store_ai_snapshot(
    pair: str,
    price_changes: List[float],
    last_price: float,
    aggregator_interval_secs: int,
    stop_loss_pct: float,
    take_profit_pct: float,
    daily_drawdown_limit: float,
    coin_volatility: float,
    is_market_bullish: str,
    risk_estimate: float,
    lunarcrush_data: Dict[str, Any] = None,
    notes: str = None,
    db_path: str = DB_FILE
):
    """
    Insert a single row into ai_snapshots with a wide variety of data.
    This can be called each time we run the aggregator or do an AI inference.

    :param pair: e.g. "ETH/USD"
    :param price_changes: The array of minute-based % changes used in the prompt
    :param last_price: e.g. 1234.56
    :param aggregator_interval_secs: from config
    :param stop_loss_pct: from config
    :param take_profit_pct: from config
    :param daily_drawdown_limit: from config or risk_manager
    :param coin_volatility: from LunarCrush or aggregator
    :param is_market_bullish: e.g. "YES","NO","MIXED" or similar
    :param risk_estimate: a numeric measure if you have one
    :param lunarcrush_data: a dict of lunarcrush metrics to store as JSON
    :param notes: free-form text
    :param db_path: path to the SQLite DB
    """
    now_ts = int(time.time())
    price_changes_json_str = "[]"
    if price_changes:
        import json
        price_changes_json_str = json.dumps(price_changes)

    lunarcrush_data_json_str = "{}"
    if lunarcrush_data:
        import json
        lunarcrush_data_json_str = json.dumps(lunarcrush_data)

    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("""
            INSERT INTO ai_snapshots (
                created_at,
                pair,
                aggregator_interval_secs,
                stop_loss_pct,
                take_profit_pct,
                daily_drawdown_limit,
                price_changes_json,
                last_price,
                lunarcrush_data_json,
                coin_volatility,
                is_market_bullish,
                risk_estimate,
                notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now_ts,
            pair,
            aggregator_interval_secs,
            stop_loss_pct,
            take_profit_pct,
            daily_drawdown_limit,
            price_changes_json_str,
            last_price,
            lunarcrush_data_json_str,
            coin_volatility,
            is_market_bullish,
            risk_estimate,
            notes
        ))
        conn.commit()
        logger.info(f"[DB] Inserted ai_snapshot => pair={pair}, last_price={last_price}")
    except Exception as e:
        logger.exception(f"[DB] Error storing ai_snapshot => {e}")
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


def fetch_minute_spaced_prices(
    pair: str,
    db_path: str = "trades.db",
    num_points: int = 15,
    max_rows_to_search: int = 300
) -> List[Tuple[int, float]]:
    """
    Returns up to `num_points` data points from 'price_history' spaced ~1 minute apart,
    with the most recent data as the last entry in chronological order.

    Each returned item is (timestamp, last_price).
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        c = conn.cursor()
        c.execute("""
            SELECT timestamp, last_price
            FROM price_history
            WHERE pair = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (pair, max_rows_to_search))
        rows_desc = c.fetchall()
        if not rows_desc:
            return []

        # Reverse to ascending chronological order
        rows_asc = list(reversed(rows_desc))

        out = []
        last_picked_ts = 0
        for row in rows_asc:
            ts = int(row["timestamp"])
            price = float(row["last_price"])
            if not out:
                out.append((ts, price))
                last_picked_ts = ts
            else:
                # only pick a row if >=60 seconds from the prior
                if (ts - last_picked_ts) >= 60:
                    out.append((ts, price))
                    last_picked_ts = ts
            if len(out) >= num_points:
                break
        return out
    except Exception as e:
        logger.exception(f"[fetch_minute_spaced_prices] error => {e}")
        return []
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
# TRADES (FINAL) - We keep record_trade_in_db for legacy usage
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
    :param price: final average fill price
    :param order_id: The Kraken order ID or some local reference
    :param pair: e.g. 'ETH/USD'
    :param fee: total fee paid for this fill
    :param kraken_trade_id: The unique trade fill ID from ownTrades
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
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
            int(time.time()),
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
def create_pending_trades_table(conn=None):
    """
    We'll store ephemeral/pending trades here, with minimal columns. No partial fill logic.
    """
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
                status TEXT,  -- 'pending','open','closed','rejected'
                kraken_order_id TEXT,
                reason TEXT,
                lot_id INTEGER,
                source TEXT,
                rationale TEXT
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
def create_pending_trade(
    side: str,
    requested_qty: float,
    pair: str,
    reason: str = None,
    source: str = None,
    rationale: str = None
) -> int:
    """
    Inserts a new row into pending_trades with status='pending'.
    """
    conn = sqlite3.connect(DB_FILE)
    row_id = None
    try:
        c = conn.cursor()
        c.execute("""
            INSERT INTO pending_trades (
                created_at, pair, side, requested_qty, status,
                kraken_order_id, reason, source, rationale
            )
            VALUES (?, ?, ?, ?, 'pending', NULL, ?, ?, ?)
        """, (
            int(time.time()),
            pair,
            side,
            requested_qty,
            reason,
            source,
            rationale
        ))
        conn.commit()
        row_id = c.lastrowid
        logger.info(f"[DB] Successfully created pending trade ID {row_id} for {side} {requested_qty} {pair}")
    except Exception as e:
        logger.exception(f"[DB] Error creating pending trade: {e}")
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


def set_kraken_order_id_for_pending_trade(pending_id: int, kraken_order_id: str):
    """
    Assigns the kraken_order_id (txid from Kraken) to our local pending_trades row
    once we know which local 'id' it belongs to.
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


# ------------------------------------------------------------------------------
# KRAKEN BALANCE HISTORY
# ------------------------------------------------------------------------------
def create_kraken_balance_table(conn=None):
    """
    Ensures a table 'kraken_balance_history' exists for storing snapshots
    of fetched Kraken balances over time.

    Columns:
      - id
      - timestamp
      - asset
      - balance
    """
    close_conn = False
    if conn is None:
        conn = sqlite3.connect(DB_FILE)
        close_conn = True

    try:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS kraken_balance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                asset TEXT,
                balance REAL
            )
        """)
        conn.commit()
        logger.info("[DB] kraken_balance_history table ensured.")
    except Exception as e:
        logger.exception(f"[DB] Error creating kraken_balance_history table: {e}")
    finally:
        if close_conn:
            conn.close()


def store_kraken_balances(balances: Dict[str, float], db_path: str = DB_FILE):
    """
    Appends a snapshot of balances to the 'kraken_balance_history' table.
    Each asset gets one row, so if you have 5 assets, you’ll insert 5 rows.
    """
    if not balances:
        logger.warning("[DB] store_kraken_balances called with empty dict => skipping.")
        return

    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        now_ts = int(time.time())

        rows_to_insert = []
        for asset, bal in balances.items():
            rows_to_insert.append((now_ts, asset, bal))

        c.executemany("""
            INSERT INTO kraken_balance_history (timestamp, asset, balance)
            VALUES (?, ?, ?)
        """, rows_to_insert)

        conn.commit()
        logger.info(f"[DB] Inserted {len(rows_to_insert)} asset balances at ts={now_ts}.")
    except Exception as e:
        logger.exception(f"[DB] Error inserting kraken balances: {e}")
    finally:
        conn.close()


async def background_balance_updater(
    kraken_rest_manager,
    refresh_interval: float = 60.0,
    db_path: str = DB_FILE
):
    """
    A background coroutine that loops indefinitely:

      1) Fetches current Kraken balances via kraken_rest_manager.fetch_balance()
      2) Stores them in the DB
      3) Sleeps for refresh_interval
      4) Repeats
    """
    logger.info("[BalanceUpdater] Starting background coroutine to poll balances.")
    while True:
        try:
            # 1) fetch
            current_balances = kraken_rest_manager.fetch_balance()
            # 2) store in DB
            store_kraken_balances(current_balances, db_path=db_path)
        except Exception as e:
            logger.exception(f"[BalanceUpdater] Error in fetch/store loop => {e}")

        await asyncio.sleep(refresh_interval)


def mark_pending_trade_closed(pending_id: int, reason: str = None):
    """
    If the trade is fully filled, mark status='closed'.
    We'll also create a final row in 'trades' outside (ws_data_feed or whichever).
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
def create_kraken_asset_pairs_table(conn=None):
    """
    Stores a wide range of columns from /AssetPairs. We'll do an 'INSERT OR REPLACE'
    when upserting each pair. This function just ensures the table exists.
    """
    close_conn = False
    if conn is None:
        conn = sqlite3.connect(DB_FILE)
        close_conn = True

    try:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS kraken_asset_pairs (
                pair_name TEXT PRIMARY KEY,
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
                leverage_buy TEXT,   -- stored as JSON string
                leverage_sell TEXT,  -- stored as JSON
                fees TEXT,           -- stored as JSON
                fees_maker TEXT,     -- stored as JSON
                fee_volume_currency TEXT,
                margin_call INTEGER,
                margin_stop INTEGER,
                ordermin TEXT,
                costmin TEXT,
                tick_size TEXT,
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


def fetch_price_history_desc(pair: str, limit: int = 10):
    """
    Fetch the most recent `limit` rows of price history for a given pair,
    sorted by timestamp DESC (newest first).
    """
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        c = conn.cursor()
        c.execute("""
            SELECT
                timestamp,
                bid_price,
                ask_price,
                last_price
            FROM price_history
            WHERE pair = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (pair, limit))
        rows = c.fetchall()
        return rows
    except Exception as e:
        print(f"Error fetching price history: {e}")
        return []
    finally:
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
        altname = pair_info.get("altname", "")
        wsname = pair_info.get("wsname", "")
        aclass_base = pair_info.get("aclass_base", "")
        base = pair_info.get("base", "")
        aclass_quote = pair_info.get("aclass_quote", "")
        quote = pair_info.get("quote", "")
        lot = pair_info.get("lot", "")
        cost_decimals = pair_info.get("cost_decimals", 0)
        pair_decimals = pair_info.get("pair_decimals", 0)
        lot_decimals = pair_info.get("lot_decimals", 0)
        lot_multiplier = pair_info.get("lot_multiplier", 0)

        # Convert arrays to JSON
        leverage_buy = json.dumps(pair_info.get("leverage_buy", []))
        leverage_sell = json.dumps(pair_info.get("leverage_sell", []))
        fees = json.dumps(pair_info.get("fees", []))
        fees_maker = json.dumps(pair_info.get("fees_maker", []))
        fee_volume_currency = pair_info.get("fee_volume_currency", "")
        margin_call = pair_info.get("margin_call", 0)
        margin_stop = pair_info.get("margin_stop", 0)
        ordermin = pair_info.get("ordermin", "")
        costmin = pair_info.get("costmin", "")
        tick_size = pair_info.get("tick_size", "")
        status = pair_info.get("status", "")
        long_pos_limit = pair_info.get("long_position_limit", 0)
        short_pos_limit = pair_info.get("short_position_limit", 0)

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
