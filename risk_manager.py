# =============================================================================
# FILE: risk_manager.py
# =============================================================================
"""
Production version of risk_manager.py

Implements a "lots-based" risk model for managing stop-loss and take-profit:

1) Each BUY finalization either creates a new row in `holding_lots` or combines with
   an existing lot if no partial sells have occurred since the last purchase of that pair.

2) The `on_price_update(...)` method checks each open lot for:
   - Stop-Loss: If current_price <= (lot_price * (1 - stop_loss_percent)) => SELL entire lot
   - Take-Profit: If current_price >= (lot_price * (1 + take_profit_percent)) => SELL quantity
     equal to the lot’s original purchased size (“initial_quantity”).

3) If a partial take-profit occurs, the leftover quantity remains in the same row, but is
   flagged `has_sold_in_between=1`, ensuring that any new buy does not merge cost basis
   with that partial position.

4) A daily drawdown check remains in `adjust_trade(...)`, halting new buys if the user’s
   daily realized PnL is below the configured threshold.

Database Tables:
- `trades`: for final fill records (inserts come from the private feed)
- `pending_trades`: ephemeral states (for new SELL actions placed by stop-loss or take-profit)
- `holding_lots`: new table that tracks each open lot of a purchased coin. Columns:
    id, pair, purchase_price, initial_quantity, quantity, date_purchased, has_sold_in_between
"""

import logging
import os
import sqlite3
import time
import datetime
from typing import Tuple, Optional

from dotenv import load_dotenv
from config_loader import ConfigLoader
from kraken_rest_manager import KrakenRestManager
import db_lookup
from db import create_pending_trade

load_dotenv()

# Configure logger for production usage
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DB_FILE = "trades.db"  # Path to your trades DB file


# --------------------------------------------------------------------------
# Utility to initialize holding_lots
# --------------------------------------------------------------------------
def init_holding_lots_table(db_path: str = DB_FILE) -> None:
    """
    Ensures the holding_lots table exists in the SQLite database.
    Each row represents a distinct “lot” of an asset purchased, with:
      - purchase_price: Weighted cost basis (including fees) for this lot
      - initial_quantity: Original quantity bought (used to limit partial take-profit sells)
      - quantity: Current quantity of this lot still open
      - has_sold_in_between: 0 => future buys can combine with this lot's cost basis;
                             1 => some portion sold, so no further cost-basis merges.
    """
    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS holding_lots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT,
                purchase_price REAL,
                initial_quantity REAL,
                quantity REAL,
                date_purchased INTEGER,
                has_sold_in_between INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        logger.info("[RiskManager] holding_lots table ensured.")
    except Exception as exc:
        logger.exception(f"Error creating holding_lots table => {exc}")
    finally:
        conn.close()


# --------------------------------------------------------------------------
# RiskManagerDB Class
# --------------------------------------------------------------------------
class RiskManagerDB:
    """
    The primary class for risk management and stop-loss / take-profit logic in production.
    - Each buy is tracked as a lot, or combined with a previous lot if we haven't sold any portion
      in between.
    - on_price_update(...) checks each open lot for SL/TP triggers and creates SELL orders
      in 'pending_trades' if triggered.
    - adjust_trade(...) enforces a daily drawdown limit and ensures the user has sufficient
      USD/coins for each new buy/sell.
    """

    def __init__(
            self,
            db_path: str = DB_FILE,
            max_position_size: float = 3.0,
            max_daily_drawdown: float = None,
            initial_spending_account: float = 100.0
    ):
        """
        :param db_path: SQLite DB path. Must contain 'trades', 'pending_trades', 'holding_lots' tables.
        :param max_position_size: Hard limit on the per-trade size (e.g., 3.0 coins).
        :param max_daily_drawdown: If the daily realized PnL is below this threshold (e.g. -0.02 => -2%),
                                   do not allow new buys for that day.
        :param initial_spending_account: A per-trade budget in USD. For instance, 100 => no single buy
                                         can exceed $100 total cost.
        """
        self.db_path = db_path
        self.max_position_size = max_position_size
        self.max_daily_drawdown = max_daily_drawdown
        self.initial_spending_account = initial_spending_account

    # ----------------------------------------------------------------------
    # Initialize required tables
    # ----------------------------------------------------------------------
    def initialize(self) -> None:
        """
        Creates 'trades' if missing, plus ensures 'holding_lots' for the lots-based approach.
        'pending_trades' is assumed to be created in db.py.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
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
            conn.commit()
            logger.info("[RiskManager] 'trades' table ensured.")
        except Exception as e:
            logger.exception(f"[RiskManager] Error ensuring 'trades' table => {e}")
        finally:
            conn.close()

        # Also ensure holding_lots
        init_holding_lots_table(self.db_path)

    # ----------------------------------------------------------------------
    # 1) Stop-Loss / Take-Profit: Called on each price update
    # ----------------------------------------------------------------------
    def on_price_update(
            self,
            pair: str,
            current_price: float,
            kraken_balances: dict,
            stop_loss_pct: float = ConfigLoader.get_value("stop_loss_percent", 0.05),
            take_profit_pct: float = ConfigLoader.get_value("take_profit_percent", 0.02)
    ):
        """
        For each open lot of `pair`, check if the price triggers:
          - Stop-Loss => SELL entire lot
          - Take-Profit => SELL 'initial_quantity'

        If triggered, we create a SELL row in 'pending_trades' and either remove or update
        that lot to reflect the partial or full sale.

        :param pair: The trading pair (e.g. "ETH/USD")
        :param current_price: Current market price from a feed
        :param kraken_balances: Dict with user’s real-time balances from the exchange
        :param stop_loss_pct: e.g. 0.05 => 5% stop
        :param take_profit_pct: e.g. 0.02 => 2% take-profit
        """
        lots = self._load_lots_for_pair(pair)
        if not lots:
            return

        for lot in lots:
            lot_id = lot["id"]
            lot_price = lot["purchase_price"]
            lot_qty = lot["quantity"]
            init_qty = lot["initial_quantity"]

            # Skip empty or closed positions
            if lot_qty <= 0:
                continue

            # -------------- STOP-LOSS --------------
            if current_price <= lot_price * (1.0 - stop_loss_pct):
                reason = f"STOP_LOSS triggered: px={current_price:.2f}"
                logger.info(
                    f"[RiskManager] Stop-loss => lot_id={lot_id}, sell all {lot_qty:.4f} of {pair}"
                )
                # Create SELL pending trade
                create_pending_trade(
                    side="SELL",
                    requested_qty=lot_qty,
                    pair=pair,
                    reason=reason
                )
                # Remove or zero out the lot
                self._remove_lot(lot_id)
                continue

            # -------------- TAKE-PROFIT --------------
            if current_price >= lot_price * (1.0 + take_profit_pct):
                # Sell up to 'init_qty'
                sell_size = min(init_qty, lot_qty)
                if sell_size <= 0:
                    continue

                reason = f"TAKE_PROFIT triggered: px={current_price:.2f}"
                logger.info(
                    f"[RiskManager] Take-profit => lot_id={lot_id}, sell {sell_size:.4f} of {pair}"
                )
                create_pending_trade(
                    side="SELL",
                    requested_qty=sell_size,
                    pair=pair,
                    reason=reason
                )

                leftover_qty = lot_qty - sell_size
                if leftover_qty > 0:
                    # Partial leftover => update it, mark has_sold_in_between=1
                    self._update_lot_after_partial_sale(lot_id, leftover_qty)
                else:
                    # Entire lot closed
                    self._remove_lot(lot_id)

    # ----------------------------------------------------------------------
    # 2) Combine or Insert new "lot" upon a BUY
    # ----------------------------------------------------------------------
    def add_or_combine_lot(self, pair: str, buy_quantity: float, buy_price: float):
        """
        Called once a new BUY is finalized.
        If there's an existing lot for `pair` with has_sold_in_between=0, combine cost basis.
        Otherwise, create a new row.

        :param pair: e.g. "ETH/USD"
        :param buy_quantity: final filled quantity from the buy
        :param buy_price: actual fill price (including fees if you prefer)
        """
        if buy_quantity <= 0 or buy_price <= 0:
            return

        existing_lot = self._find_open_lot_without_sells(pair)
        if existing_lot:
            # Weighted average cost basis update
            lot_id = existing_lot["id"]
            old_qty = existing_lot["quantity"]
            old_price = existing_lot["purchase_price"]

            total_qty = old_qty + buy_quantity
            new_price = ((old_qty * old_price) + (buy_quantity * buy_price)) / total_qty
            logger.info(
                f"[RiskManager] Combine-lot => lot_id={lot_id}, old_qty={old_qty:.4f}, "
                f"new_qty={buy_quantity:.4f}, total={total_qty:.4f}, price={new_price:.4f}"
            )
            self._update_combined_lot(lot_id, new_price, total_qty, old_qty, buy_quantity)
        else:
            # Insert a new lot row
            now_ts = int(time.time())
            self._insert_new_lot(pair, buy_price, buy_quantity, now_ts)

    # ----------------------------------------------------------------------
    # 3) Daily Drawdown & Basic Pre-Trade Checks
    # ----------------------------------------------------------------------
    def adjust_trade(
            self,
            signal: str,
            suggested_size: float,
            pair: str,
            current_price: float,
            kraken_balances: dict
    ) -> Tuple[str, float]:
        """
        Ensures trades abide by daily drawdown and user capacity:
          - If daily drawdown is below threshold => no new BUYs
          - If user doesn't have enough USD/coin => skip
          - Clamps final size to self.max_position_size
          - If cost is > initial_spending_account => skip

        Returns (final_signal, final_size).
        Typically called just before placing an order in AIStrategy.
        """
        # 1) daily drawdown check
        if self.max_daily_drawdown is not None:
            daily_pnl = self.get_daily_realized_pnl()
            if daily_pnl <= self.max_daily_drawdown and signal == "BUY":
                logger.warning(
                    f"[RiskManager] daily drawdown => realizedPnL={daily_pnl:.4f} "
                    f"<= {self.max_daily_drawdown} => no new buys => HOLD."
                )
                return ("HOLD", 0.0)

        # 2) If not BUY/SELL => do nothing
        if signal not in ("BUY", "SELL"):
            return ("HOLD", 0.0)

        # 3) clamp the size
        final_size = min(suggested_size, self.max_position_size)
        if final_size <= 0:
            return ("HOLD", 0.0)

        # 4) Check cost or coin availability
        if signal == "BUY":
            cost = final_size * current_price
            if cost > self.initial_spending_account:
                logger.info(
                    f"[RiskManager] cost={cost:.2f} > initial_spending_account={self.initial_spending_account:.2f}"
                )
                return ("HOLD", 0.0)
            usd_symbol = self._get_quote_symbol(pair)
            free_usd = kraken_balances.get(usd_symbol, 0.0)
            if cost > free_usd:
                logger.info(
                    f"[RiskManager] insufficient {usd_symbol} => cost={cost:.2f}, free={free_usd:.2f}"
                )
                return ("HOLD", 0.0)
        else:
            # SELL => check base coin
            base_sym = self._get_base_symbol(pair)
            free_coins = kraken_balances.get(base_sym, 0.0)
            if final_size > free_coins:
                logger.info(
                    f"[RiskManager] insufficient {base_sym} => requested={final_size:.4f}, have={free_coins:.4f}"
                )
                return ("HOLD", 0.0)

        return (signal, final_size)

    # ----------------------------------------------------------------------
    # 4) Daily PnL Calculation
    # ----------------------------------------------------------------------
    def get_daily_realized_pnl(self, date_str: Optional[str] = None) -> float:
        """
        Sums realized_pnl from 'trades' for the specified UTC date.
        If none, uses today's date in UTC. Returns 0.0 if none found or error.
        """
        if not date_str:
            date_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            q = """
                SELECT IFNULL(SUM(realized_pnl), 0.0)
                FROM trades
                WHERE realized_pnl IS NOT NULL
                  AND date(timestamp, 'unixepoch') = ?
            """
            c.execute(q, (date_str,))
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
    # Utility: Insert new lot
    # ----------------------------------------------------------------------
    def _insert_new_lot(self, pair: str, purchase_price: float, quantity: float, date_purchased: int):
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("""
                INSERT INTO holding_lots (
                    pair, purchase_price, initial_quantity, quantity, date_purchased, has_sold_in_between
                )
                VALUES (?, ?, ?, ?, ?, 0)
            """, (pair, purchase_price, quantity, quantity, date_purchased))
            conn.commit()
            logger.info(
                f"[RiskManager] Inserted new lot => pair={pair}, qty={quantity:.4f}, px={purchase_price:.4f}"
            )
        except Exception as e:
            logger.exception(f"[RiskManager] _insert_new_lot => {e}")
        finally:
            conn.close()

    # ----------------------------------------------------------------------
    # Utility: Combine cost basis
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
        Weighted cost basis update. Also updates 'initial_quantity' by the newly purchased amount.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("""
                SELECT initial_quantity
                FROM holding_lots
                WHERE id=?
            """, (lot_id,))
            row = c.fetchone()
            if not row:
                logger.warning(f"[RiskManager] No lot found with id={lot_id} to combine.")
                return

            old_init_qty = float(row[0])
            new_init_qty = old_init_qty + add_qty  # expand "initial_quantity" by the new buy

            c.execute("""
                UPDATE holding_lots
                SET purchase_price=?,
                    initial_quantity=?,
                    quantity=?
                WHERE id=?
            """, (new_price, new_init_qty, new_qty, lot_id))
            conn.commit()
        except Exception as e:
            logger.exception(f"[RiskManager] _update_combined_lot => {e}")
        finally:
            conn.close()

    # ----------------------------------------------------------------------
    # Utility: Mark partial leftover
    # ----------------------------------------------------------------------
    def _update_lot_after_partial_sale(self, lot_id: int, new_quantity: float):
        """
        Mark leftover quantity and set has_sold_in_between=1 so future buys won't merge cost basis.
        """
        if new_quantity < 0:
            new_quantity = 0.0
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("""
                UPDATE holding_lots
                SET quantity=?,
                    has_sold_in_between=1
                WHERE id=?
            """, (new_quantity, lot_id))
            conn.commit()
            logger.info(f"[RiskManager] partial sale => lot_id={lot_id}, leftover={new_quantity:.4f}")
        except Exception as e:
            logger.exception(f"[RiskManager] _update_lot_after_partial_sale => {e}")
        finally:
            conn.close()

    # ----------------------------------------------------------------------
    # Utility: Remove lot
    # ----------------------------------------------------------------------
    def _remove_lot(self, lot_id: int):
        """
        Deletes the lot row from holding_lots after a full stop-loss or a total take-profit exit.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("DELETE FROM holding_lots WHERE id=?", (lot_id,))
            conn.commit()
            logger.info(f"[RiskManager] removed lot_id={lot_id}")
        except Exception as e:
            logger.exception(f"[RiskManager] _remove_lot => {e}")
        finally:
            conn.close()

    # ----------------------------------------------------------------------
    # Utility: Find an existing open lot for merging
    # ----------------------------------------------------------------------
    def _find_open_lot_without_sells(self, pair: str) -> Optional[dict]:
        """
        Returns the first open lot for this pair where has_sold_in_between=0
        and quantity>0, or None if not found. We'll combine cost basis with that lot if found.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            c = conn.cursor()
            c.execute("""
                SELECT id, pair, purchase_price, initial_quantity, quantity, date_purchased, has_sold_in_between
                FROM holding_lots
                WHERE pair=? AND quantity>0 AND has_sold_in_between=0
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
    # Utility: load lots for a pair
    # ----------------------------------------------------------------------
    def _load_lots_for_pair(self, pair: str) -> list:
        """
        Return all open lots for this pair (quantity>0).
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        results = []
        try:
            c = conn.cursor()
            c.execute("""
                SELECT
                    id, pair, purchase_price, initial_quantity,
                    quantity, date_purchased, has_sold_in_between
                FROM holding_lots
                WHERE pair=? AND quantity>0
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
        """
        For "ETH/USD", returns the 'quote' symbol in Kraken format, e.g. "ZUSD".
        """
        return db_lookup.get_asset_value_for_pair(pair, value="quote")

    def _get_base_symbol(self, pair: str) -> str:
        """
        For "ETH/USD", returns the 'base' symbol in Kraken format, e.g. "XETH".
        """
        return db_lookup.get_base_asset(pair)
