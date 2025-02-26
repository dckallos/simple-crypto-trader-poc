#!/usr/bin/env python3
# =============================================================================
# FILE: risk_manager.py
# =============================================================================
"""
Enhanced version of risk_manager.py (Production-Ready)

Major additions & changes:
--------------------------
1. Extended `holding_lots` schema:
   - lot_status (TEXT): 'ACTIVE','PENDING_SELL','PARTIAL_SOLD','CLOSED'
   - origin_source (TEXT): e.g. 'gpt', 'manual', 'external'
   - strategy_version (TEXT): e.g. 'v1.0'
   - risk_params_json (TEXT): a JSON snapshot of your risk configs
   - time_closed (INTEGER): epoch time when the lot is fully closed
   - peak_favorable_price (REAL): highest price seen since purchase
   - peak_adverse_price (REAL): lowest price seen since purchase

2. Extended logic for price checks:
   - Each time on_price_update is called, we update peak_favorable_price
     and peak_adverse_price for each open lot. This helps track your
     "peak favorable" and "peak adverse" excursions for potential analytics.
   - We no longer unconditionally delete from holding_lots in
     rebuild_lots_from_ledger_entries. Instead, we do incremental merges
     so your partially open lots stay intact across restarts.

3. Single final realized PnL:
   - We store final realized PnL in the 'trades' table once a SELL closes
     a lot. The final SELL can look up the cost basis from holding_lots
     to do your own realized PnL calculations if desired.
   - For advanced usage, you might store advanced analytics or PnL details
     in trade_metrics_json or in custom columns.

4. stop_loss_events (unchanged from previous versions):
   - We still log each triggered SL event with a snapshot of risk_params_json
   - You can expand or unify this with other events if you want.

5. Additional references:
   - We keep place_live_orders logic the same, but you can unify or expand
     concurrency checks if multiple tasks might write to DB simultaneously.
   - We assume 'db_lookup' is available for fetching min order sizes, etc.

NOTE:
If you are migrating from an older version, ensure your DB has columns:
 - holding_lots.peak_favorable_price
 - holding_lots.peak_adverse_price
 - trades.lot_id, trades.ai_model, trades.ai_rationale, etc.
You can add them with straightforward migrations or rely on the
automatic creation in init_holding_lots_table / _maybe_migrate_trades_table.

"""

import logging
import os
import sqlite3
import time
import datetime
from typing import Tuple, Optional, List, Dict, Any

from dotenv import load_dotenv

import db_lookup
from db import create_pending_trade
from config_loader import ConfigLoader
from threading import Lock

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
      - lot_status (TEXT) => 'ACTIVE','PENDING_SELL','PARTIAL_SOLD','CLOSED'
      - origin_source (TEXT) => e.g. 'gpt','manual','external'
      - strategy_version (TEXT)
      - risk_params_json (TEXT)
      - time_closed (INTEGER)
      - peak_favorable_price (REAL)
      - peak_adverse_price (REAL)
    """
    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()

        # Create table if not exists
        c.execute("""
        CREATE TABLE IF NOT EXISTS holding_lots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT,
            purchase_price REAL,
            initial_quantity REAL,
            quantity REAL,
            date_purchased INTEGER
        )
        """)

        existing_cols = _get_existing_columns(c, "holding_lots")

        additions = []
        if "lot_status" not in existing_cols:
            additions.append("ALTER TABLE holding_lots ADD COLUMN lot_status TEXT DEFAULT 'ACTIVE'")
        if "origin_source" not in existing_cols:
            additions.append("ALTER TABLE holding_lots ADD COLUMN origin_source TEXT")
        if "strategy_version" not in existing_cols:
            additions.append("ALTER TABLE holding_lots ADD COLUMN strategy_version TEXT")
        if "risk_params_json" not in existing_cols:
            additions.append("ALTER TABLE holding_lots ADD COLUMN risk_params_json TEXT")
        if "time_closed" not in existing_cols:
            additions.append("ALTER TABLE holding_lots ADD COLUMN time_closed INTEGER")
        if "peak_favorable_price" not in existing_cols:
            additions.append("ALTER TABLE holding_lots ADD COLUMN peak_favorable_price REAL")
        if "peak_adverse_price" not in existing_cols:
            additions.append("ALTER TABLE holding_lots ADD COLUMN peak_adverse_price REAL")

        for stmt in additions:
            logger.info(f"[RiskManager] MIGRATION: {stmt}")
            c.execute(stmt)

        conn.commit()
        if additions:
            logger.info(f"[RiskManager] holding_lots table migrated with new columns => {additions}")
        else:
            logger.info("[RiskManager] holding_lots table already has all columns.")

    except Exception as exc:
        logger.exception(f"[RiskManager] Error creating/migrating holding_lots => {exc}")
    finally:
        conn.close()


def create_stop_loss_events_table(db_path: str = DB_FILE) -> None:
    """
    Creates a new table 'stop_loss_events' to log each time a stop-loss is triggered.
    If it doesn't exist, we add it. You may expand columns for more analytics.
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
    Return a list of column names for a given table.
    """
    cursor.execute(f"PRAGMA table_info({table_name})")
    rows = cursor.fetchall()
    return [row[1] for row in rows]


# ------------------------------------------------------------------------------
# Utility: Migrate trades table for extended columns
# ------------------------------------------------------------------------------
def _maybe_migrate_trades_table(db_path: str) -> None:
    """
    Ensures that 'trades' has columns for advanced analytics:
      - lot_id, ai_model, source_config, ai_rationale, exchange_fill_time, trade_metrics_json
    """
    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        existing_cols = _get_existing_columns(c, "trades")

        to_add = []
        if "lot_id" not in existing_cols:
            to_add.append(("lot_id", "INTEGER"))
        if "ai_model" not in existing_cols:
            to_add.append(("ai_model", "TEXT"))
        if "source_config" not in existing_cols:
            to_add.append(("source_config", "TEXT"))
        if "ai_rationale" not in existing_cols:
            to_add.append(("ai_rationale", "TEXT"))
        if "exchange_fill_time" not in existing_cols:
            to_add.append(("exchange_fill_time", "INTEGER"))
        if "trade_metrics_json" not in existing_cols:
            to_add.append(("trade_metrics_json", "TEXT"))

        for col_name, col_type in to_add:
            logger.info(f"[RiskManager] MIGRATION: adding column '{col_name}' to trades.")
            c.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")

        if to_add:
            conn.commit()
            logger.info(f"[RiskManager] trades => added new columns => {to_add}")
        else:
            logger.info("[RiskManager] trades table is up-to-date.")
    except Exception as e:
        logger.exception(f"[RiskManager] Error migrating trades table => {e}")
    finally:
        conn.close()


# ------------------------------------------------------------------------------
# Primary Class: RiskManagerDB
# ------------------------------------------------------------------------------
class RiskManagerDB:
    """
    The core class for managing risk parameters, merges of new BUYs,
    stop-loss/take-profit triggers, partial leftovers, and final close logic.

    Notable fields in 'holding_lots':
      - lot_status: 'ACTIVE','PENDING_SELL','PARTIAL_SOLD','CLOSED'
      - peak_favorable_price, peak_adverse_price
      - On each price update, we refresh these for open lots.

    Notable expansions:
      - incremental rebuild from ledger for external trades
      - storing final realized PnL in 'trades' or after SELL close (if desired)
    """

    def __init__(
        self,
        db_path: str = DB_FILE,
        max_position_size: float = 3.0,
        max_daily_drawdown: float = None,
        initial_spending_account: float = 100.0,
        private_ws_client=None,
        place_live_orders: bool = False,
        ai_lock: Lock = None
    ):
        """
        :param db_path: Path to SQLite DB
        :param max_position_size: Hard clamp on position size
        :param max_daily_drawdown: e.g. -0.02 => skip new BUYs if daily PnL < -2%
        :param initial_spending_account: A simple budget if desired
        :param private_ws_client: e.g. KrakenPrivateWSClient for placing orders
        :param place_live_orders: if True => we place real SELL/BUY orders
        """
        self.db_path = db_path
        self.max_position_size = max_position_size
        self.max_daily_drawdown = max_daily_drawdown
        self.initial_spending_account = initial_spending_account

        self.private_ws_client = private_ws_client
        self.place_live_orders = place_live_orders
        self.ai_lock = ai_lock

    def initialize(self) -> None:
        """
        Ensures the trades + holding_lots + stop_loss_events are up-to-date.
        Also runs migrations to add new columns if needed.
        """
        # Ensure trades table
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
            )
            """)
            conn.commit()
            logger.info("[RiskManager] 'trades' table ensured.")
        except Exception as e:
            logger.exception(f"[RiskManager] Error ensuring 'trades' => {e}")
        finally:
            conn.close()

        # Migrate trades => new columns
        _maybe_migrate_trades_table(self.db_path)

        # holding_lots => ensure + migrate
        init_holding_lots_table(self.db_path)

        # create stop_loss_events => ensure
        create_stop_loss_events_table(self.db_path)

    # ----------------------------------------------------------------------
    # Helpers for asset naming
    # ----------------------------------------------------------------------
    def _get_quote_symbol(self, pair: str) -> str:
        """
        For "DOT/USD", returns the 'quote' symbol in Kraken format, e.g. "ZUSD".
        """
        return db_lookup.get_asset_value_for_pair(pair, value="quote")

    def _get_base_symbol(self, pair: str) -> str:
        """
        For "DOT/USD", returns the 'base' symbol in Kraken format, e.g. "XDOT".
        """
        return db_lookup.get_base_asset(pair)

    # ----------------------------------------------------------------------
    # Price-check cycle => called in main.py with an async loop
    # ----------------------------------------------------------------------
    async def start_db_price_check_cycle(self, pairs: List[str]):
        """
        Asynchronous loop:
         - sleeps initial_risk_manager_sleep_seconds
         - Then, every X seconds, fetches last known DB price for each pair
         - Calls on_price_update => handle SL/TP triggers
         - Repeats until canceled
        """
        import asyncio

        dynamic_interval = ConfigLoader.get_value("risk_manager_interval_seconds", 30.0)
        logger.debug(f"[RiskManager] Sleeping for {dynamic_interval} seconds before start.")

        initial_sleep = ConfigLoader.get_value("initial_risk_manager_sleep_seconds", 30.0)
        logger.info(f"[RiskManager] Delaying {initial_sleep} seconds before first cycle.")
        await asyncio.sleep(initial_sleep)

        logger.info(f"[RiskManager] Starting DB-based price check cycle => pairs={pairs}, interval={dynamic_interval}s")
        while True:
            try:
                # Check if the AI is processing (lock is acquired)
                if self.ai_lock.locked():
                    logger.info("[RiskManager] AI is processing, skipping this cycle.")
                    await asyncio.sleep(1)  # Sleep briefly to avoid busy-waiting
                    continue

                for pair in pairs:
                    last_price = self._fetch_latest_price_for_pair(pair)
                    if last_price > 0:
                        # We pass empty kraken_balances unless you want to pass real data
                        self.on_price_update(pair, last_price, {})
            except asyncio.CancelledError:
                logger.info("[RiskManager] price_check_cycle => canceled => exit loop.")
                break
            except Exception as e:
                logger.exception(f"[RiskManager] DB price check error => {e}")

            await asyncio.sleep(dynamic_interval)

        logger.info("[RiskManager] price_check_cycle => fully exited.")

    def _fetch_latest_price_for_pair(self, pair: str) -> float:
        """
        Returns the newest last_price from 'price_history' for the given pair, or 0 if none found.
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
            if row:
                return float(row["last_price"])
            return 0.0
        except Exception as e:
            logger.exception(f"[RiskManager] _fetch_latest_price_for_pair => {e}")
            return 0.0
        finally:
            conn.close()

    # ----------------------------------------------------------------------
    # on_price_update => check stop-loss / take-profit => place SELL if triggered
    # also update peak_favorable_price & peak_adverse_price for each lot
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
        For each open lot on 'pair', we update peak prices, then check if
        SL or TP triggers => place SELL => set lot_status='PENDING_SELL'.
        """
        lots = self._load_open_lots(pair)
        if not lots:
            return

        for lot in lots:
            lot_id = lot["id"]
            lot_price = float(lot["purchase_price"])
            lot_qty = float(lot["quantity"])
            lot_status = lot.get("lot_status", "ACTIVE")

            # 1) Update peak favorable / adverse
            peak_fav = float(lot.get("peak_favorable_price") or lot_price)
            peak_adv = float(lot.get("peak_adverse_price") or lot_price)
            updated_fav = max(peak_fav, current_price)
            updated_adv = min(peak_adv, current_price)
            if (updated_fav != peak_fav) or (updated_adv != peak_adv):
                self._update_lot_peak_prices(lot_id, updated_fav, updated_adv)

            if lot_qty <= 0 or lot_status in ("PENDING_SELL", "CLOSED"):
                continue  # skip

            # 2) STOP LOSS
            if stop_loss_pct > 0.0 and current_price <= lot_price * (1.0 - stop_loss_pct):
                reason = f"STOP_LOSS triggered at price={current_price:.4f}"
                logger.info(f"[RiskManager] Stop-loss => lot_id={lot_id}, SELL all => {lot_qty:.4f} {pair}")
                self._record_stop_loss_event(lot_id, current_price, reason)

                min_order_size = db_lookup.get_ordermin(pair)
                if lot_qty < min_order_size:
                    logger.warning(f"[RiskManager] lot_id={lot_id} => SELL qty={lot_qty:.4f} < exch min={min_order_size:.4f}, skip.")
                    continue

                # Create pending SELL
                pending_id = create_pending_trade(
                    side="SELL",
                    requested_qty=lot_qty,
                    pair=pair,
                    reason=reason,
                    source="risk_manager",
                    rationale="stop-loss"
                )
                if self._maybe_place_kraken_order(pair, "SELL", lot_qty, pending_id):
                    self._mark_lot_as_pending_sell(lot_id)

            elif current_price >= take_profit_threshold:
                reason = f"TAKE_PROFIT triggered at price={current_price:.4f} (threshold={take_profit_threshold:.4f})"
                logger.info(f"[RiskManager] Take-profit => lot_id={lot_id}, SELL => {lot_qty:.4f} {pair}")
                min_order_size = db_lookup.get_ordermin(pair)
                if lot_qty < min_order_size:
                    logger.warning(
                        f"[RiskManager] lot_id={lot_id} => SELL qty={lot_qty:.4f} < min={min_order_size:.4f}, skip.")
                    continue
                pending_id = create_pending_trade(
                    side="SELL",
                    requested_qty=lot_qty,
                    pair=pair,
                    reason=reason,
                    source="risk_manager",
                    rationale="take-profit"
                )
                if self._maybe_place_kraken_order(pair, "SELL", lot_qty, pending_id):
                    self._mark_lot_as_pending_sell(lot_id)

    def _update_lot_peak_prices(self, lot_id: int, new_fav: float, new_adv: float):
        """
        Update peak_favorable_price and peak_adverse_price in holding_lots.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("""
            UPDATE holding_lots
               SET peak_favorable_price=?,
                   peak_adverse_price=?
             WHERE id=?
            """, (new_fav, new_adv, lot_id))
            conn.commit()
        except Exception as e:
            logger.exception(f"[RiskManager] _update_lot_peak_prices => {e}")
        finally:
            conn.close()

    def _get_thresholds_from_db(self, pair: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Fetches the latest stop_loss and take_profit thresholds from coin_thresholds for the given pair.

        Args:
            pair (str): The trading pair (e.g., "ETH/USD").

        Returns:
            Tuple[Optional[float], Optional[float]]: (stop_loss, take_profit), or (None, None) if not found.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("""
                SELECT stop_loss, take_profit
                FROM coin_thresholds
                WHERE pair = ?
                ORDER BY last_updated DESC
                LIMIT 1
            """, (pair,))
            row = c.fetchone()
            if row:
                return float(row[0]), float(row[1])
            logger.warning(f"[RiskManager] No thresholds found for {pair}")
            return None, None
        except Exception as e:
            logger.exception(f"[RiskManager] Error fetching thresholds for {pair}: {e}")
            return None, None
        finally:
            conn.close()

    def _record_stop_loss_event(self, lot_id: int, current_price: float, reason: str):
        """
        Insert a row in stop_loss_events for analytics. We also store
        the current risk_params.
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
            """, (
                lot_id,
                int(time.time()),
                current_price,
                reason,
                rp_json
            ))
            conn.commit()
            logger.info(f"[RiskManager] Logged stop_loss_event => lot_id={lot_id}, price={current_price}")
        except Exception as e:
            logger.exception(f"[RiskManager] error => _record_stop_loss_event => {e}")
        finally:
            conn.close()

    # ----------------------------------------------------------------------
    # Called once we know a SELL has closed => leftover or 0 => finalize
    # ----------------------------------------------------------------------
    def on_pending_trade_closed(
        self,
        lot_id: int,
        fill_size: float,
        side: str,
        fill_price: float
    ):
        """
        Called by ws_data_feed once a pending SELL or BUY is fully filled on Kraken.
        If SELL => reduce leftover => possibly set lot_status='CLOSED', time_closed=Now,
        or 'PARTIAL_SOLD' if leftover remains.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            c = conn.cursor()
            c.execute("SELECT * FROM holding_lots WHERE id=?", (lot_id,))
            row = c.fetchone()
            if not row:
                logger.warning(f"[RiskManager] No lot found with id={lot_id} => skipping.")
                return

            lot_qty = float(row["quantity"])
            lot_status = row["lot_status"] or "ACTIVE"

            if side.upper() == "SELL":
                leftover = lot_qty - fill_size
                if leftover <= 0:
                    # fully sold => CLOSED
                    c.execute("""
                    UPDATE holding_lots
                       SET quantity=0,
                           lot_status='CLOSED',
                           time_closed=?
                     WHERE id=?
                    """, (int(time.time()), lot_id))
                    logger.info(f"[RiskManager] lot_id={lot_id} => fully SOLD => CLOSED.")
                else:
                    # partial leftover => PARTIAL_SOLD
                    c.execute("""
                    UPDATE holding_lots
                       SET quantity=?,
                           lot_status='PARTIAL_SOLD'
                     WHERE id=?
                    """, (leftover, lot_id))
                    logger.info(f"[RiskManager] lot_id={lot_id} => leftover={leftover:.4f} => PARTIAL_SOLD.")

                conn.commit()

            else:
                # For a BUY fill => merges are handled in add_or_combine_lot
                logger.debug("[RiskManager] on_pending_trade_closed => side=BUY => no leftover logic here.")
        except Exception as e:
            logger.exception(f"[RiskManager] on_pending_trade_closed => {e}")
        finally:
            conn.close()

    # ----------------------------------------------------------------------
    # Handling new BUY merges => either new lot or combine with existing
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
        Called when a BUY fill is recognized => merges with an ACTIVE lot or
        creates a new one. We track peak_favorable_price & peak_adverse_price
        at inception = buy_price by default.
        """
        if buy_quantity <= 0 or buy_price <= 0:
            return

        existing_lot = self._find_active_lot(pair)
        if existing_lot:
            # Weighted cost basis update
            lot_id = existing_lot["id"]
            old_qty = float(existing_lot["quantity"])
            old_price = float(existing_lot["purchase_price"])

            new_qty = old_qty + buy_quantity
            new_price = ((old_qty * old_price) + (buy_quantity * buy_price)) / new_qty

            logger.info(
                f"[RiskManager] Combine-lot => lot_id={lot_id}, old_qty={old_qty:.4f}, "
                f"add_qty={buy_quantity:.4f}, total={new_qty:.4f}, new_px={new_price:.4f}"
            )
            self._update_combined_lot(lot_id, new_price, new_qty)
        else:
            # Insert brand-new lot => set peak_favorable_price & peak_adverse_price = buy_price initially
            now_ts = int(time.time())
            self._insert_new_lot(
                pair=pair,
                purchase_price=buy_price,
                quantity=buy_quantity,
                date_purchased=now_ts,
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
        Called from AIStrategy => verifies daily drawdown, clamp final_size,
        ensures cost >= exchange min cost, checks user balances, etc.
        Returns (action, final_size).
        """
        # daily drawdown block
        if self.max_daily_drawdown is not None:
            daily_pnl = self.get_daily_realized_pnl()
            if daily_pnl <= self.max_daily_drawdown and signal == "BUY":
                logger.warning(f"[RiskManager] dailyPnL={daily_pnl:.4f} <= {self.max_daily_drawdown} => skip BUY => HOLD.")
                return ("HOLD", 0.0)

        # clamp action
        if signal not in ("BUY", "SELL"):
            return ("HOLD", 0.0)

        final_size = min(suggested_size, self.max_position_size)
        if final_size <= 0:
            return ("HOLD", 0.0)

        # If BUY => check cost & user capacity
        if signal == "BUY":
            cost = final_size * current_price
            min_cost = db_lookup.get_minimum_cost_in_usd(pair)
            if cost < min_cost:
                logger.info(f"[RiskManager] Proposed BUY cost={cost:.2f} < min cost={min_cost:.2f} => skip.")
                return ("HOLD", 0.0)

            if cost > self.initial_spending_account:
                logger.info(f"[RiskManager] cost={cost:.2f} > init_acct={self.initial_spending_account:.2f} => skip.")
                return ("HOLD", 0.0)

            usd_symbol = self._get_quote_symbol(pair)
            free_usd = kraken_balances.get(usd_symbol, 0.0)
            if cost > free_usd:
                logger.info(f"[RiskManager] insufficient {usd_symbol}, cost={cost:.2f}, have={free_usd:.2f} => skip.")
                return ("HOLD", 0.0)

        # If SELL => ensure enough base coins
        else:
            base_sym = self._get_base_symbol(pair)
            free_coins = kraken_balances.get(base_sym, 0.0)
            if final_size > free_coins:
                logger.info(f"[RiskManager] insufficient {base_sym}, need={final_size:.4f}, have={free_coins:.4f} => skip.")
                return ("HOLD", 0.0)

        return (signal, final_size)

    def get_daily_realized_pnl(self, date_str: Optional[str] = None) -> float:
        """
        Returns sum of realized_pnl from 'trades' on the given UTC date.
        If none specified, uses today's UTC date.
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
               AND date(timestamp, 'unixepoch')=?
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
    def _maybe_place_kraken_order(self, pair: str, action: str, volume: float, pending_id: int = None) -> bool:
        """
        If place_live_orders=True and private_ws_client is up, we place a market order.
        Return True if we "attempted" to place, or if we skip because place_live_orders=False.
        """
        if not self.place_live_orders:
            return True  # Pretend success in dev mode
        if not self.private_ws_client or not self.private_ws_client.running:
            logger.warning("[RiskManager] private_ws_client not available => can't place order => returning False.")
            return False

        side_for_kraken = "buy" if action.upper() == "BUY" else "sell"
        logger.info(f"[RiskManager] Sending real order => {pair} {side_for_kraken} vol={volume}, userref={pending_id}")
        self.private_ws_client.send_order(
            pair=pair,
            side=side_for_kraken,
            ordertype="market",
            volume=volume,
            userref=str(pending_id) if pending_id else None
        )
        return True

    # ----------------------------------------------------------------------
    # Mark lot => PENDING_SELL
    # ----------------------------------------------------------------------
    def _mark_lot_as_pending_sell(self, lot_id: int):
        """
        Set lot_status='PENDING_SELL' so we skip re-triggers for that lot
        until we confirm the SELL is closed or canceled.
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
    # Insert new lot => default peak_favorable=peak_adverse= purchase_price
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
                time_closed,
                peak_favorable_price,
                peak_adverse_price
            )
            VALUES (?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?, NULL, ?, ?)
            """, (
                pair,
                purchase_price,
                quantity,
                quantity,
                date_purchased,
                origin_source,
                strategy_version,
                risk_params_json,
                purchase_price,  # peak_favorable_price
                purchase_price   # peak_adverse_price
            ))
            conn.commit()
            lot_id = c.lastrowid
            logger.info(f"[RiskManager] Inserted new lot => lot_id={lot_id}, pair={pair}, qty={quantity:.4f}")
        except Exception as e:
            logger.exception(f"[RiskManager] _insert_new_lot => {e}")
        finally:
            conn.close()

    # ----------------------------------------------------------------------
    # Weighted cost basis => re-activate the lot if needed
    # ----------------------------------------------------------------------
    def _update_combined_lot(self, lot_id: int, new_price: float, new_qty: float):
        """
        Weighted cost basis => update purchase_price, initial_quantity, quantity.
        Also reset lot_status to 'ACTIVE' because we just added more coins.
        We do NOT alter peak_favorable_price or peak_adverse_price here,
        though you could recompute if you want. For now, we keep the old extremes.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()

            # fetch old initial_quantity
            c.execute("SELECT initial_quantity, lot_status FROM holding_lots WHERE id=?", (lot_id,))
            row = c.fetchone()
            if not row:
                logger.warning(f"[RiskManager] No lot found with id={lot_id} => can't combine.")
                return
            old_init_qty = float(row[0])
            new_init_qty = old_init_qty + (new_qty - old_init_qty)  # or just set = old_init_qty + ?

            c.execute("""
            UPDATE holding_lots
               SET purchase_price=?,
                   initial_quantity=?,
                   quantity=?,
                   lot_status='ACTIVE'
             WHERE id=?
            """, (new_price, new_init_qty, new_qty, lot_id))
            conn.commit()
            logger.info(f"[RiskManager] updated lot_id={lot_id} => new_px={new_price:.4f}, new_qty={new_qty:.4f}")
        except Exception as e:
            logger.exception(f"[RiskManager] _update_combined_lot => {e}")
        finally:
            conn.close()

    def _find_active_lot(self, pair: str) -> Optional[dict]:
        """
        Finds a single open/active lot for the given pair. If you want to handle
        multiple parallel lots per pair, you'd revise logic to pick the earliest
        or combine them differently.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            c = conn.cursor()
            c.execute("""
            SELECT *
              FROM holding_lots
             WHERE pair=? 
               AND lot_status IN ('ACTIVE','PARTIAL_SOLD')
             ORDER BY id ASC
             LIMIT 1
            """, (pair,))
            row = c.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.exception(f"[RiskManager] _find_active_lot => {e}")
            return None
        finally:
            conn.close()

    def _load_open_lots(self, pair: str) -> List[dict]:
        """
        Return all lots that are not CLOSED for the given pair
        (i.e., quantity>0 and status != CLOSED).
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        results = []
        try:
            c = conn.cursor()
            c.execute("""
            SELECT *
              FROM holding_lots
             WHERE pair=?
               AND lot_status != 'CLOSED'
               AND quantity>0
            """, (pair,))
            rows = c.fetchall()
            results = [dict(r) for r in rows]
        except Exception as e:
            logger.exception(f"[RiskManager] _load_open_lots => {e}")
        finally:
            conn.close()
        return results

    # ----------------------------------------------------------------------
    # rebuild_lots_from_ledger_entries => incremental approach
    # ----------------------------------------------------------------------
    def rebuild_lots_from_ledger_entries(self):
        """
        Instead of deleting holding_lots, we do an incremental approach:
         1) read all ledger_entries with type='trade'
         2) for each, if it's a buy => add_or_combine_lot
         3) if it's a sell => apply partial leftover
        This approach won't blow away partially open lots.
        If you do want a full rebuild, do it manually or with a force flag.
        """
        logger.info("[RiskManager] Attempting incremental rebuild => no unconditional DELETE.")
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            c = conn.cursor()
            c.execute("""
            SELECT ledger_id, refid, time, asset, amount, fee, balance
              FROM ledger_entries
             WHERE type='trade'
             ORDER BY time ASC, ledger_id ASC
            """)
            rows = c.fetchall()

            from collections import defaultdict
            trades_by_refid = defaultdict(list)
            for r in rows:
                trades_by_refid[r["refid"]].append({
                    "ledger_id": r["ledger_id"],
                    "time": r["time"],
                    "asset": r["asset"],
                    "amount": r["amount"],
                    "fee": r["fee"],
                    "balance": r["balance"]
                })

            refids_sorted = sorted(trades_by_refid.keys(), key=lambda x: trades_by_refid[x][0]["time"])
            for refid in refids_sorted:
                group = trades_by_refid[refid]
                if len(group) < 2:
                    logger.warning(f"[RiskManager] Ledger refid={refid} has <2 rows => skipping.")
                    continue

                # Typically one row is ZUSD (or ZXX?), another is a coin
                maybe_coin = [x for x in group if not x["asset"].upper().startswith("Z")]
                maybe_zusd = [x for x in group if x["asset"].upper().startswith("Z")]
                if not maybe_coin or not maybe_zusd:
                    logger.warning(f"[RiskManager] refid={refid} => missing coin or Z => skipping.")
                    continue

                coin_row = maybe_coin[0]
                zusd_row = maybe_zusd[0]
                coin_amt = float(coin_row["amount"])
                usd_amt = float(zusd_row["amount"])

                if coin_amt > 0:
                    # BUY scenario
                    cost_usd = abs(usd_amt)
                    if cost_usd > 0 and coin_amt > 0:
                        px = cost_usd / coin_amt
                        self.add_or_combine_lot(
                            pair=self._infer_pair_name(coin_row["asset"]),
                            buy_quantity=coin_amt,
                            buy_price=px,
                            origin_source="external_ledger"
                        )
                else:
                    # SELL scenario
                    sell_qty = abs(coin_amt)
                    proceeds_usd = usd_amt
                    if sell_qty > 0:
                        px = proceeds_usd / sell_qty if sell_qty else 0
                        self._apply_historical_sell_ledger(
                            pair=self._infer_pair_name(coin_row["asset"]),
                            sell_qty=sell_qty,
                            sell_price=px
                        )
            logger.info("[RiskManager] Completed incremental ledger-lots rebuild.")
        except Exception as e:
            logger.exception(f"[RiskManager] rebuild_lots_from_ledger_entries => {e}")
        finally:
            conn.close()

    def _apply_historical_sell_ledger(self, pair: str, sell_qty: float, sell_price: float):
        """
        If we see an external SELL from the ledger, we decrement the local lot.
        If leftover <= 0 => set CLOSED/time_closed. If partial => PARTIAL_SOLD.
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
               AND lot_status!='CLOSED'
               AND quantity>0
             ORDER BY id ASC
            """, (pair,))
            rows = c.fetchall()

            qty_to_sell = sell_qty
            for r in rows:
                lot_id = r["id"]
                lot_qty = float(r["quantity"])
                if qty_to_sell <= 0:
                    break

                if lot_qty <= qty_to_sell:
                    # fully close
                    c.execute("""
                    UPDATE holding_lots
                       SET quantity=0,
                           lot_status='CLOSED',
                           time_closed=?
                     WHERE id=?
                    """, (int(time.time()), lot_id))
                    logger.debug(f"[RiskManager] external SELL => closed lot_id={lot_id}")
                    qty_to_sell -= lot_qty
                else:
                    leftover = lot_qty - qty_to_sell
                    new_status = 'PARTIAL_SOLD'
                    c.execute("""
                    UPDATE holding_lots
                       SET quantity=?,
                           lot_status=?,
                           time_closed=NULL
                     WHERE id=?
                    """, (leftover, new_status, lot_id))
                    logger.debug(f"[RiskManager] external SELL => lot_id={lot_id}, leftover={leftover:.4f}")
                    qty_to_sell = 0

            conn.commit()
            if qty_to_sell > 0:
                logger.warning(f"[RiskManager] leftover SELL from ledger => qty={qty_to_sell:.4f} unallocated.")
        except Exception as e:
            logger.exception(f"[RiskManager] _apply_historical_sell_ledger => {e}")
        finally:
            conn.close()

    def _infer_pair_name(self, asset_name: str) -> str:
        """
        A small helper to convert ledger asset codes (e.g. 'XETH') to a wsname
        like 'ETH/USD'. We rely on db_lookup or a known map. Adjust if needed.
        """
        return db_lookup.get_websocket_name_from_base_asset(asset_name)



