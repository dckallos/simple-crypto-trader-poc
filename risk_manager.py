#!/usr/bin/env python3
# =============================================================================
# FILE: risk_manager.py
# =============================================================================
"""
Production version of risk_manager.py with new logic:
 - Replaces has_sold_in_between with lot_status text column (ACTIVE, PENDING_SELL, PARTIAL_SOLD, CLOSED).
 - Persists additional fields (origin_source, strategy_version, risk_params_json, time_closed).
 - Adds a new 'stop_loss_events' table to log each SL trigger.
 - Removes the unconditional DELETE in rebuild_lots_from_ledger_entries, supporting a more "incremental" approach.
 - Adds lightweight migrations for both holding_lots and trades to store new columns.
"""

import logging
import os
import sqlite3
import time
import datetime
from typing import Tuple, Optional, List, Dict, Any

from dotenv import load_dotenv

from config_loader import ConfigLoader
import db_lookup
from db import create_pending_trade

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DB_FILE = "trades.db"  # Path to your SQLite DB file

# ------------------------------------------------------------------------------
# Utility: Initialize or migrate 'holding_lots' and 'stop_loss_events'
# ------------------------------------------------------------------------------

def init_holding_lots_table(db_path: str = DB_FILE) -> None:
    """
    Ensures the holding_lots table exists, and adds new columns if missing:
      - lot_status (TEXT) => 'ACTIVE', 'PENDING_SELL', 'PARTIAL_SOLD', 'CLOSED'
      - origin_source (TEXT) => e.g. 'gpt', 'manual'
      - strategy_version (TEXT) => e.g. 'v1.0'
      - risk_params_json (TEXT) => JSON snapshot of risk settings
      - time_closed (INTEGER) => epoch when lot closed

    We also remove references to has_sold_in_between in the schema,
    or we keep it if it already exists. We'll do a migration approach.
    """
    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        # Create table if not exists (the old version):
        c.execute("""
        CREATE TABLE IF NOT EXISTS holding_lots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT,
            purchase_price REAL,
            initial_quantity REAL,
            quantity REAL,
            date_purchased INTEGER
            -- old column has_sold_in_between might exist from prior versions
        )
        """)

        # Check columns, add them if missing:
        existing_cols = _get_existing_columns(c, "holding_lots")

        if "lot_status" not in existing_cols:
            logger.info("[RiskManager] MIGRATION: adding column 'lot_status' to holding_lots.")
            c.execute("ALTER TABLE holding_lots ADD COLUMN lot_status TEXT DEFAULT 'ACTIVE'")

        if "origin_source" not in existing_cols:
            logger.info("[RiskManager] MIGRATION: adding column 'origin_source' to holding_lots.")
            c.execute("ALTER TABLE holding_lots ADD COLUMN origin_source TEXT")

        if "strategy_version" not in existing_cols:
            logger.info("[RiskManager] MIGRATION: adding column 'strategy_version' to holding_lots.")
            c.execute("ALTER TABLE holding_lots ADD COLUMN strategy_version TEXT")

        if "risk_params_json" not in existing_cols:
            logger.info("[RiskManager] MIGRATION: adding column 'risk_params_json' to holding_lots.")
            c.execute("ALTER TABLE holding_lots ADD COLUMN risk_params_json TEXT")

        if "time_closed" not in existing_cols:
            logger.info("[RiskManager] MIGRATION: adding column 'time_closed' to holding_lots.")
            c.execute("ALTER TABLE holding_lots ADD COLUMN time_closed INTEGER")

        conn.commit()

        logger.info("[RiskManager] holding_lots table ensured with new columns.")
    except Exception as exc:
        logger.exception(f"[RiskManager] Error creating/migrating holding_lots => {exc}")
    finally:
        conn.close()


def create_stop_loss_events_table(db_path: str = DB_FILE) -> None:
    """
    Creates a new table 'stop_loss_events' to log each time a stop-loss is triggered.
    This can help you do analytics on how often your SL rules are firing.

    Columns:
      - id (PK)
      - lot_id => references which lot triggered
      - triggered_at => epoch time
      - current_price => price at moment of trigger
      - reason => free text
      - risk_params_json => snapshot of relevant risk parameters
    """
    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS stop_loss_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lot_id INTEGER,
            triggered_at INTEGER,
            current_price REAL,
            reason TEXT,
            risk_params_json TEXT
        )
        """)
        conn.commit()
        logger.info("[RiskManager] stop_loss_events table ensured.")
    except Exception as e:
        logger.exception(f"[RiskManager] Error creating stop_loss_events => {e}")
    finally:
        conn.close()


def _get_existing_columns(cursor: sqlite3.Cursor, table_name: str) -> List[str]:
    """
    Utility: return a list of column names for a given table.
    """
    cursor.execute(f"PRAGMA table_info({table_name})")
    rows = cursor.fetchall()
    return [row[1] for row in rows]


# ------------------------------------------------------------------------------
# Utility: Migrate trades table for extended columns
# ------------------------------------------------------------------------------

def _maybe_migrate_trades_table(db_path: str) -> None:
    """
    Checks for new columns in 'trades' and adds them if missing:
      - lot_id (INTEGER)
      - ai_model (TEXT)
      - source_config (TEXT) => to store JSON risk config
      - ai_rationale (TEXT)
      - exchange_fill_time (INTEGER) => actual fill time from the exchange
      - trade_metrics_json (TEXT) => e.g. storing peak adverse excursion, etc.
    """
    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        existing_cols = _get_existing_columns(c, "trades")

        if "lot_id" not in existing_cols:
            logger.info("[RiskManager] MIGRATION: adding column 'lot_id' to trades.")
            c.execute("ALTER TABLE trades ADD COLUMN lot_id INTEGER")

        if "ai_model" not in existing_cols:
            logger.info("[RiskManager] MIGRATION: adding column 'ai_model' to trades.")
            c.execute("ALTER TABLE trades ADD COLUMN ai_model TEXT")

        if "source_config" not in existing_cols:
            logger.info("[RiskManager] MIGRATION: adding column 'source_config' to trades.")
            c.execute("ALTER TABLE trades ADD COLUMN source_config TEXT")

        if "ai_rationale" not in existing_cols:
            logger.info("[RiskManager] MIGRATION: adding column 'ai_rationale' to trades.")
            c.execute("ALTER TABLE trades ADD COLUMN ai_rationale TEXT")

        if "exchange_fill_time" not in existing_cols:
            logger.info("[RiskManager] MIGRATION: adding column 'exchange_fill_time' to trades.")
            c.execute("ALTER TABLE trades ADD COLUMN exchange_fill_time INTEGER")

        if "trade_metrics_json" not in existing_cols:
            logger.info("[RiskManager] MIGRATION: adding column 'trade_metrics_json' to trades.")
            c.execute("ALTER TABLE trades ADD COLUMN trade_metrics_json TEXT")

        conn.commit()
    except Exception as e:
        logger.exception(f"[RiskManager] Error migrating trades table => {e}")
    finally:
        conn.close()


# ------------------------------------------------------------------------------
# Primary Class: RiskManagerDB
# ------------------------------------------------------------------------------

class RiskManagerDB:
    """
    The core class for stop-loss / take-profit logic, plus the new rules:
      1. Force all BUY orders to exceed the exchange's minimum cost in USD
      2. If partial SELL leftover is < 110% of exchange min order, SELL it all
      3. Replaces old integer-based 'has_sold_in_between' with 'lot_status' text
      4. Logs stop-loss triggers to 'stop_loss_events'
      5. Single final PnL on fully closed lots
    """

    def __init__(
        self,
        db_path: str = DB_FILE,
        max_position_size: float = 3.0,
        max_daily_drawdown: float = None,
        initial_spending_account: float = 100.0,
        private_ws_client=None,
        place_live_orders: bool = False
    ):
        """
        :param db_path: Path to SQLite DB
        :param max_position_size: Hard clamp on number of coins per trade
        :param max_daily_drawdown: e.g. -0.02 => if realized daily PnL < -2%, skip new BUYs
        :param initial_spending_account: A simple budget cap if desired
        :param private_ws_client: e.g. KrakenPrivateWSClient
        :param place_live_orders: if True => we call private_ws_client.send_order(...) after new SL/TP SELL
        """
        self.db_path = db_path
        self.max_position_size = max_position_size
        self.max_daily_drawdown = max_daily_drawdown
        self.initial_spending_account = initial_spending_account

        # store references just like AIStrategy
        self.private_ws_client = private_ws_client
        self.place_live_orders = place_live_orders

    def initialize(self) -> None:
        """
        Ensures the trades + holding_lots + stop_loss_events are up-to-date.
        Also runs migrations to add new columns if they don’t exist.
        """
        # 1) Ensure trades table
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
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
                realized_pnl REAL
                -- new columns get added below with migration
            )
            """)
            conn.commit()
            logger.info("[RiskManager] 'trades' table ensured.")
        except Exception as e:
            logger.exception(f"[RiskManager] Error ensuring 'trades' => {e}")
        finally:
            conn.close()

        # 2) Migrate trades for new columns (lot_id, ai_model, etc.)
        _maybe_migrate_trades_table(self.db_path)

        # 3) holding_lots => ensure + migrate
        init_holding_lots_table(self.db_path)

        # 4) create stop_loss_events
        create_stop_loss_events_table(self.db_path)

    # ----------------------------------------------------------------------
    # 1) Price-check cycle
    # ----------------------------------------------------------------------
    async def start_db_price_check_cycle(
        self,
        pairs: List[str]
    ):
        """
        An asynchronous loop that:
         - Waits some time for private feed
         - For each pair, reads the last price from DB
         - Calls on_price_update => check SL/TP
         - Sleeps for 'interval' seconds
         - Repeats until canceled
        """
        import asyncio

        dynamic_interval = ConfigLoader.get_value("risk_manager_interval_seconds", 30.0)
        logger.debug(f"[RiskManager] Sleeping for {dynamic_interval} seconds.")

        # Delay to ensure private feed is connected
        logger.info("[RiskManager] Delaying 30 seconds after init before first price check.")
        initial_risk_manager_sleep = ConfigLoader.get_value("initial_risk_manager_sleep_seconds", 30.0)
        await asyncio.sleep(initial_risk_manager_sleep)
        logger.info("[RiskManager] Proceeding with price check cycle now.")

        logger.info(f"[RiskManager] Starting DB-based price check cycle for {pairs}, interval={dynamic_interval}s")
        while True:
            try:
                for pair in pairs:
                    latest_price = self._fetch_latest_price_for_pair(pair)
                    if latest_price > 0:
                        dummy_balances = {}
                        self.on_price_update(
                            pair=pair,
                            current_price=latest_price,
                            kraken_balances=dummy_balances
                        )
            except asyncio.CancelledError:
                logger.info("[RiskManager] price_check_cycle task is cancelled => exiting cleanly.")
                break
            except Exception as e:
                logger.exception(f"[RiskManager] DB price check cycle => error => {e}")

            dynamic_interval = ConfigLoader.get_value("risk_manager_interval_seconds", 30.0)
            logger.debug(f"[RiskManager] Sleeping for {dynamic_interval} seconds.")
            await asyncio.sleep(dynamic_interval)

        logger.info("[RiskManager] price_check_cycle has exited.")

    def _fetch_latest_price_for_pair(self, pair: str) -> float:
        """
        Utility: returns newest last_price from 'price_history' for the given pair, or 0 if not found.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            c = conn.cursor()
            c.execute("""
            SELECT last_price
            FROM price_history
            WHERE pair=?
            ORDER BY timestamp DESC
            LIMIT 1
            """, (pair,))
            row = c.fetchone()
            if row is None:
                return 0.0
            return float(row["last_price"])
        except Exception as e:
            logger.exception(f"[RiskManager] _fetch_latest_price_for_pair => pair={pair}, err={e}")
            return 0.0
        finally:
            conn.close()

    # ----------------------------------------------------------------------
    # 2) on_price_update => check SL/TP => create SELL if triggered
    # ----------------------------------------------------------------------
    def on_price_update(
        self,
        pair: str,
        current_price: float,
        kraken_balances: Dict[str, float],
        stop_loss_pct: float = ConfigLoader.get_value("stop_loss_percent", 0.05),
        take_profit_pct: float = ConfigLoader.get_value("take_profit_percent", 0.02)
    ):
        """
        Checks each open lot => if SL or TP triggered => place SELL => set lot_status='PENDING_SELL'.
        Also logs to stop_loss_events if a SL is triggered.
        """
        lots = self._load_lots_for_pair(pair)
        if not lots:
            return

        for lot in lots:
            lot_id = lot["id"]
            lot_price = float(lot["purchase_price"])
            lot_qty = float(lot["quantity"])
            init_qty = float(lot["initial_quantity"])
            lot_status = lot.get("lot_status", "ACTIVE")

            if lot_qty <= 0 or lot_status == "PENDING_SELL" or lot_status == "CLOSED":
                # skip if empty or already pending sell or closed
                continue

            # STOP LOSS
            if stop_loss_pct > 0.0 and current_price <= lot_price * (1.0 - stop_loss_pct):
                reason = f"STOP_LOSS triggered at price={current_price:.4f}"
                logger.info(f"[RiskManager] Stop-loss => lot_id={lot_id}, SELL all {lot_qty:.4f} of {pair}")

                self._record_stop_loss_event(lot_id, current_price, reason)

                min_order_size = db_lookup.get_ordermin(pair)
                if lot_qty < min_order_size:
                    logger.warning(
                        f"[RiskManager] lot_id={lot_id} => SELL qty={lot_qty:.4f} below exch min={min_order_size:.4f}. Skipping..."
                    )
                    # optionally mark as dust or do some other fallback
                    continue

                pending_id = create_pending_trade(
                    side="SELL",
                    requested_qty=lot_qty,
                    pair=pair,
                    reason=reason,
                    source="risk_manager",
                    rationale="stop-loss"
                )
                success = self._maybe_place_kraken_order(pair, "SELL", lot_qty, pending_id)
                if success:
                    self._mark_lot_as_pending_sell(lot_id)
                else:
                    logger.warning("[RiskManager] Private feed not open => skipping => will retry next cycle.")
                continue

            # TAKE PROFIT
            if take_profit_pct > 0.0 and current_price >= lot_price * (1.0 + take_profit_pct):
                # Proposed partial sale
                sell_size = min(init_qty, lot_qty)
                min_order_size = db_lookup.get_ordermin(pair)

                # If leftover < 1.1 x min_order => sell entire lot
                leftover = lot_qty - sell_size
                if leftover > 0 and leftover < (1.1 * min_order_size):
                    logger.info(
                        f"[RiskManager] leftover={leftover:.4f} < 1.1 x min_order_size={min_order_size:.4f} => selling entire lot."
                    )
                    sell_size = lot_qty

                if sell_size < min_order_size:
                    logger.warning(
                        f"[RiskManager] lot_id={lot_id} => SELL qty={sell_size:.4f} below exch min={min_order_size:.4f}. Skipping..."
                    )
                    continue

                reason = f"TAKE_PROFIT triggered at price={current_price:.4f}"
                logger.info(f"[RiskManager] Take-profit => lot_id={lot_id}, SELL {sell_size:.4f} of {pair}")

                pending_id = create_pending_trade(
                    side="SELL",
                    requested_qty=sell_size,
                    pair=pair,
                    reason=reason,
                    source="risk_manager",
                    rationale="take-profit"
                )
                success = self._maybe_place_kraken_order(pair, "SELL", sell_size, pending_id)
                if success:
                    self._mark_lot_as_pending_sell(lot_id)
                else:
                    logger.warning("[RiskManager] Private feed not open => skipping => will retry next cycle.")

    def _record_stop_loss_event(self, lot_id: int, current_price: float, reason: str):
        """
        Insert a row in stop_loss_events table for analytics.
        Potentially store risk_params_json with the current SL/TP config, if desired.
        """
        risk_params_dict = {
            "stop_loss_percent": ConfigLoader.get_value("stop_loss_percent", 0.05),
            "take_profit_percent": ConfigLoader.get_value("take_profit_percent", 0.02),
            "daily_drawdown_limit": self.max_daily_drawdown
        }
        import json
        rp_json = json.dumps(risk_params_dict)

        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("""
            INSERT INTO stop_loss_events (lot_id, triggered_at, current_price, reason, risk_params_json)
            VALUES (?, ?, ?, ?, ?)
            """, (lot_id, int(time.time()), current_price, reason, rp_json))
            conn.commit()
            logger.info(f"[RiskManager] stop_loss_events => recorded SL event => lot_id={lot_id}, reason={reason}")
        except Exception as e:
            logger.exception(f"[RiskManager] error in _record_stop_loss_event => {e}")
        finally:
            conn.close()


    # ----------------------------------------------------------------------
    # 3) On SELL fill => on_pending_trade_closed => update leftover or close
    # ----------------------------------------------------------------------
    def on_pending_trade_closed(
        self,
        lot_id: int,
        fill_size: float,
        side: str,
        fill_price: float
    ):
        """
        Called once the private feed says SELL (or BUY) is fully closed on Kraken.
        If SELL => remove or partially leftover. If leftover remains, set lot_status='PARTIAL_SOLD'.
        If zero leftover => set lot_status='CLOSED', time_closed=Now.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            c = conn.cursor()
            c.execute("SELECT * FROM holding_lots WHERE id=?", (lot_id,))
            lot = c.fetchone()
            if not lot:
                logger.warning(f"[RiskManager] on_pending_trade_closed => no lot_id={lot_id} found.")
                return

            lot_qty = float(lot["quantity"])
            lot_status = lot.get("lot_status", "ACTIVE")

            if side.upper() == "SELL":
                leftover = lot_qty - fill_size
                if leftover <= 0:
                    # fully sold
                    c.execute("""
                    UPDATE holding_lots
                       SET quantity=0,
                           lot_status='CLOSED',
                           time_closed=?
                     WHERE id=?
                    """, (int(time.time()), lot_id))
                    logger.info(f"[RiskManager] Fully sold lot_id={lot_id} => CLOSED.")
                else:
                    # partial leftover
                    c.execute("""
                    UPDATE holding_lots
                       SET quantity=?,
                           lot_status='PARTIAL_SOLD'
                     WHERE id=?
                    """, (leftover, lot_id))
                    logger.info(f"[RiskManager] Partial leftover => lot_id={lot_id}, leftover={leftover:.4f}")
                conn.commit()

            else:
                # side=BUY => merges are handled by add_or_combine_lot.
                logger.debug("[RiskManager] side=BUY => no special logic in on_pending_trade_closed.")
        except Exception as e:
            logger.exception(f"[RiskManager] on_pending_trade_closed => {e}")
        finally:
            conn.close()

    # ----------------------------------------------------------------------
    # 4) Add or Combine new BUY
    # ----------------------------------------------------------------------
    def add_or_combine_lot(
        self,
        pair: str,
        buy_quantity: float,
        buy_price: float,
        origin_source: str = "gpt",
        strategy_version: str = "v1.0",
        risk_params_json: str = None
    ):
        """
        Called when a BUY is final. Merges with an ACTIVE lot or creates new.
        We also store origin_source, strategy_version, etc. for data science.
        """
        if buy_quantity <= 0 or buy_price <= 0:
            return

        existing_lot = self._find_open_lot_without_sells(pair)
        if existing_lot:
            lot_id = existing_lot["id"]
            old_qty = float(existing_lot["quantity"])
            old_price = float(existing_lot["purchase_price"])
            total_qty = old_qty + buy_quantity
            new_price = ((old_qty * old_price) + (buy_quantity * buy_price)) / total_qty

            logger.info(
                f"[RiskManager] Combine-lot => lot_id={lot_id}, old_qty={old_qty:.4f}, "
                f"new_qty={buy_quantity:.4f}, total={total_qty:.4f}, px={new_price:.4f}"
            )
            self._update_combined_lot(lot_id, new_price, total_qty, old_qty, buy_quantity)
        else:
            now_ts = int(time.time())
            self._insert_new_lot(
                pair,
                buy_price,
                buy_quantity,
                now_ts,
                origin_source=origin_source,
                strategy_version=strategy_version,
                risk_params_json=risk_params_json
            )

    def adjust_trade(
        self,
        signal: str,
        suggested_size: float,
        pair: str,
        current_price: float,
        kraken_balances: Dict[str, float]
    ) -> Tuple[str, float]:
        """
        Check daily drawdown, clamp final_size, ensure cost >= minCost, ensure balance is enough, etc.
        Return (action, final_size).
        """
        # daily drawdown block
        if self.max_daily_drawdown is not None:
            daily_pnl = self.get_daily_realized_pnl()
            if daily_pnl <= self.max_daily_drawdown and signal == "BUY":
                logger.warning(
                    f"[RiskManager] dailyPnL={daily_pnl:.4f} <= {self.max_daily_drawdown} => no new buys => HOLD"
                )
                return ("HOLD", 0.0)

        if signal not in ("BUY", "SELL"):
            return ("HOLD", 0.0)

        final_size = min(suggested_size, self.max_position_size)
        if final_size <= 0:
            return ("HOLD", 0.0)

        # If BUY => check cost and user capacity
        if signal == "BUY":
            cost = final_size * current_price
            min_cost = db_lookup.get_minimum_cost_in_usd(pair)
            if cost < min_cost:
                logger.info(
                    f"[RiskManager] Proposed BUY cost={cost:.2f} < min cost={min_cost:.2f} => skip."
                )
                return ("HOLD", 0.0)

            if cost > self.initial_spending_account:
                logger.info(
                    f"[RiskManager] cost={cost:.2f} > initial_spending_account={self.initial_spending_account:.2f}"
                )
                return ("HOLD", 0.0)

            usd_symbol = self._get_quote_symbol(pair)
            free_usd = kraken_balances.get(usd_symbol, 0.0)
            if cost > free_usd:
                logger.info(f"[RiskManager] insufficient {usd_symbol}, cost={cost:.2f}, have={free_usd:.2f}")
                return ("HOLD", 0.0)

        # If SELL => ensure enough base coins
        else:
            base_sym = self._get_base_symbol(pair)
            free_coins = kraken_balances.get(base_sym, 0.0)
            if final_size > free_coins:
                logger.info(
                    f"[RiskManager] insufficient {base_sym} => requested={final_size:.4f}, have={free_coins:.4f}"
                )
                return ("HOLD", 0.0)

        return (signal, final_size)

    def get_daily_realized_pnl(self, date_str: Optional[str] = None) -> float:
        """
        Returns sum of realized_pnl from 'trades' for the given UTC date.
        """
        if not date_str:
            date_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")

        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("""
            SELECT IFNULL(SUM(realized_pnl), 0.0)
            FROM trades
            WHERE realized_pnl IS NOT NULL
              AND date(timestamp, 'unixepoch') = ?
            """, (date_str,))
            row = c.fetchone()
            if row and row[0] is not None:
                return float(row[0])
            return 0.0
        except Exception as e:
            logger.exception(f"[RiskManager] get_daily_realized_pnl => {e}")
            return 0.0
        finally:
            conn.close()

    # ----------------------------------------------------------------------
    # Possibly place real order
    # ----------------------------------------------------------------------
    def _maybe_place_kraken_order(
        self,
        pair: str,
        action: str,
        volume: float,
        pending_id: int = None
    ) -> bool:
        """
        If place_live_orders=True and private_ws_client is available => place a market order.
        Returns True if handed off, else False.
        """
        if not self.place_live_orders:
            return True  # "Pretend" success in test mode
        if not self.private_ws_client or not self.private_ws_client.running:
            logger.warning("[RiskManager] Private feed not open => cannot place real order => returning False.")
            return False

        side_for_kraken = "buy" if action.upper() == "BUY" else "sell"
        logger.info(
            f"[RiskManager] Sending real order => pair={pair}, side={side_for_kraken}, volume={volume}, pending_id={pending_id}"
        )
        self.private_ws_client.send_order(
            pair=pair,
            side=side_for_kraken,
            ordertype="market",
            volume=volume,
            userref=str(pending_id) if pending_id else None
        )
        return True

    # ----------------------------------------------------------------------
    # Internals: mark a lot as PENDING_SELL
    # ----------------------------------------------------------------------
    def _mark_lot_as_pending_sell(self, lot_id: int):
        """
        Mark the lot as lot_status='PENDING_SELL' so we skip re-triggers.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("""
            UPDATE holding_lots
               SET lot_status='PENDING_SELL'
             WHERE id=?
            """, (lot_id,))
            conn.commit()
            logger.info(f"[RiskManager] Marked lot_id={lot_id} => PENDING_SELL.")
        except Exception as e:
            logger.exception(f"[RiskManager] _mark_lot_as_pending_sell => {e}")
        finally:
            conn.close()

    # ----------------------------------------------------------------------
    # Insert new lot (ACTIVE)
    # ----------------------------------------------------------------------
    def _insert_new_lot(
        self,
        pair: str,
        purchase_price: float,
        quantity: float,
        date_purchased: int,
        origin_source: str = "gpt",
        strategy_version: str = "v1.0",
        risk_params_json: str = None
    ):
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("""
            INSERT INTO holding_lots (
                pair,
                purchase_price,
                initial_quantity,
                quantity,
                date_purchased,
                lot_status,
                origin_source,
                strategy_version,
                risk_params_json,
                time_closed
            )
            VALUES (?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?, NULL)
            """, (
                pair,
                purchase_price,
                quantity,
                quantity,
                date_purchased,
                origin_source,
                strategy_version,
                risk_params_json
            ))
            conn.commit()
            logger.info(
                f"[RiskManager] Inserted new lot => pair={pair}, qty={quantity:.4f}, px={purchase_price:.4f}, status=ACTIVE"
            )
        except Exception as e:
            logger.exception(f"[RiskManager] _insert_new_lot => {e}")
        finally:
            conn.close()

    # ----------------------------------------------------------------------
    # Weighted cost basis update
    # ----------------------------------------------------------------------
    def _update_combined_lot(
        self,
        lot_id: int,
        new_price: float,
        new_qty: float,
        old_qty: float,
        add_qty: float
    ):
        """
        Weighted cost basis => update purchase_price, initial_quantity, quantity.
        We keep the lot_status as is unless you want to reset it to ACTIVE.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("SELECT initial_quantity, lot_status FROM holding_lots WHERE id=?", (lot_id,))
            row = c.fetchone()
            if not row:
                logger.warning(f"[RiskManager] No lot found with id={lot_id} to combine.")
                return
            old_init_qty = float(row[0])
            new_init_qty = old_init_qty + add_qty

            c.execute("""
            UPDATE holding_lots
               SET purchase_price=?,
                   initial_quantity=?,
                   quantity=?,
                   lot_status='ACTIVE' -- after a new buy, it should be ACTIVE
             WHERE id=?
            """, (new_price, new_init_qty, new_qty, lot_id))
            conn.commit()
        except Exception as e:
            logger.exception(f"[RiskManager] _update_combined_lot => {e}")
        finally:
            conn.close()

    # ----------------------------------------------------------------------
    # Find an open lot => we only merge with lot_status='ACTIVE'
    # ----------------------------------------------------------------------
    def _find_open_lot_without_sells(self, pair: str) -> Optional[dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            c = conn.cursor()
            c.execute("""
            SELECT *
              FROM holding_lots
             WHERE pair=?
               AND quantity>0
               AND lot_status='ACTIVE'
             ORDER BY id ASC
             LIMIT 1
            """, (pair,))
            row = c.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.exception(f"[RiskManager] _find_open_lot_without_sells => {e}")
            return None
        finally:
            conn.close()

    # ----------------------------------------------------------------------
    # Load all open lots for a pair => 'quantity>0' => not closed
    # ----------------------------------------------------------------------
    def _load_lots_for_pair(self, pair: str) -> List[dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        results = []
        try:
            c = conn.cursor()
            c.execute("""
            SELECT *
              FROM holding_lots
             WHERE pair=?
               AND quantity>0
               AND lot_status!='CLOSED'
            """, (pair,))
            rows = c.fetchall()
            results = [dict(r) for r in rows]
        except Exception as e:
            logger.exception(f"[RiskManager] _load_lots_for_pair => {e}")
        finally:
            conn.close()
        return results

    # ----------------------------------------------------------------------
    # Helpers for asset naming
    # ----------------------------------------------------------------------
    def _get_quote_symbol(self, pair: str) -> str:
        return db_lookup.get_asset_value_for_pair(pair, value="quote")

    def _get_base_symbol(self, pair: str) -> str:
        return db_lookup.get_base_asset(pair)

    # ----------------------------------------------------------------------
    # Rebuild lots from ledger - now incremental approach
    # ----------------------------------------------------------------------
    def rebuild_lots_from_ledger_entries(self):
        """
        Example incremental rebuild approach:
         1) We do NOT delete holding_lots unconditionally.
         2) We read ledger_entries for new trades that we haven't applied yet.
         3) For each new buy => add_or_combine_lot, for each sell => apply partial leftover.

        If you prefer approach C (real-time hooking), you can run a narrower approach to
        only add new lines. The code below is an example snippet that used to do a full rebuild.
        It's left as a reference, but the unconditional DELETE is removed.

        Adjust to your liking for partial or incremental merges.
        """
        logger.info("[RiskManager] Attempting incremental rebuild from ledger => no unconditional DELETE.")
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            # EXAMPLE: fetch all "trade" ledger rows in ascending order
            c.execute("""
            SELECT ledger_id, refid, time, asset, amount, fee, balance
              FROM ledger_entries
             WHERE type='trade'
             ORDER BY time ASC, ledger_id ASC
            """)
            rows = c.fetchall()
            from collections import defaultdict
            trades_by_refid = defaultdict(list)
            for (ledger_id, refid, tstamp, asset, amt, fee, bal) in rows:
                trades_by_refid[refid].append({
                    "ledger_id": ledger_id,
                    "time": tstamp,
                    "asset": asset,
                    "amount": amt,
                    "fee": fee,
                    "balance": bal
                })

            refids_sorted = sorted(trades_by_refid.keys(), key=lambda r: trades_by_refid[r][0]["time"])
            for refid in refids_sorted:
                ledger_group = trades_by_refid[refid]
                if len(ledger_group) < 2:
                    logger.warning(f"[RiskManager] Ledger refid={refid} has <2 rows => skipping.")
                    continue

                maybe_zusd = [x for x in ledger_group if x["asset"].upper().startswith("Z")]
                maybe_coin = [x for x in ledger_group if not x["asset"].upper().startswith("Z")]
                if not maybe_zusd or not maybe_coin:
                    logger.warning(f"[RiskManager] refid={refid} => missing Z + coin => skipping.")
                    continue

                zusd_row = maybe_zusd[0]
                coin_row = maybe_coin[0]
                coin_amt = coin_row["amount"]
                zusd_amt = zusd_row["amount"]

                if coin_amt > 0:
                    buy_qty = coin_amt
                    cost_usd = abs(zusd_amt)
                    if abs(buy_qty) > 0:
                        avg_price = cost_usd / buy_qty
                        # Simply do an add_or_combine with no forced delete
                        self.add_or_combine_lot(
                            pair=self._get_pair_name(coin_row["asset"]),
                            buy_quantity=buy_qty,
                            buy_price=avg_price,
                            origin_source="external_ledger"
                        )
                else:
                    sell_qty = abs(coin_amt)
                    proceeds_usd = zusd_amt
                    if sell_qty > 0:
                        avg_price = proceeds_usd / sell_qty if sell_qty else 0
                        self._apply_historical_sell_ledger(
                            pair=self._get_pair_name(coin_row["asset"]),
                            sell_qty=sell_qty,
                            sell_price=avg_price
                        )

            logger.info("[RiskManager] Completed incremental rebuilding from ledger entries.")
        except Exception as e:
            logger.exception(f"[RiskManager] Error in rebuild_lots_from_ledger_entries => {e}")
        finally:
            conn.close()

    def _apply_historical_sell_ledger(self, pair: str, sell_qty: float, sell_price: float):
        """
        Example partial leftover approach for an external SELL recognized from ledger.
        We find any lot that is 'ACTIVE' or 'PARTIAL_SOLD' and decrement quantity.
        """
        if sell_qty <= 0:
            return
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            c = conn.cursor()
            c.execute("""
            SELECT id, quantity, lot_status
              FROM holding_lots
             WHERE pair=?
               AND quantity>0
               AND lot_status!='CLOSED'
             ORDER BY id ASC
            """, (pair,))
            rows = c.fetchall()

            qty_to_sell = sell_qty
            for row in rows:
                lot_id = row["id"]
                lot_qty = float(row["quantity"])
                lot_status = row["lot_status"] or "ACTIVE"

                if qty_to_sell <= 0:
                    break

                if lot_qty <= qty_to_sell:
                    # fully remove
                    c.execute("""
                    UPDATE holding_lots
                       SET quantity=0,
                           lot_status='CLOSED',
                           time_closed=?
                     WHERE id=?
                    """, (int(time.time()), lot_id))
                    logger.debug(f"[RiskManager] Ledger SELL => closed lot_id={lot_id}")
                    qty_to_sell -= lot_qty
                else:
                    leftover = lot_qty - qty_to_sell
                    new_status = 'PARTIAL_SOLD' if leftover > 0 else 'CLOSED'
                    c.execute("""
                    UPDATE holding_lots
                       SET quantity=?,
                           lot_status=?,
                           time_closed=CASE WHEN ?='CLOSED' THEN ? ELSE time_closed END
                     WHERE id=?
                    """, (leftover, new_status, new_status, int(time.time()), lot_id))
                    logger.debug(f"[RiskManager] Ledger SELL => lot_id={lot_id}, leftover={leftover:.4f}")
                    qty_to_sell = 0

            conn.commit()
            if qty_to_sell > 0:
                logger.warning(f"[RiskManager] Ledger SELL leftover={qty_to_sell} not allocated.")
        except Exception as e:
            logger.exception(f"[RiskManager] _apply_historical_sell_ledger => {e}")
        finally:
            conn.close()

    def _get_pair_name(self, asset_name: str) -> str:
        return db_lookup.get_websocket_name_from_base_asset(asset_name)
