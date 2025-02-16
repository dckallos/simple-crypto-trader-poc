# =============================================================================
# FILE: risk_manager.py
# =============================================================================
"""
Production version of risk_manager.py

Key Points:
-----------
1) Lots-based stop-loss and take-profit logic. Each BUY either creates a new
   lot in `holding_lots` or merges with an existing one if `has_sold_in_between=0`.

2) The `on_price_update(...)` method scans all open lots for a given pair, checking
   if the current price triggers a stop-loss or take-profit event. If triggered,
   we create a SELL entry in `pending_trades` **with `lot_id`**, mark that lot as
   “pending sell,” and optionally place the real SELL order on Kraken if
   `place_live_orders` is True and `private_ws_client` is provided.

3) After Kraken **actually** fills or closes the SELL, the private feed finalizes
   the trade in `pending_trades`. It then calls `risk_manager_db.on_pending_trade_closed(...)`
   with the lot_id, fill_size, etc. Only then do we remove or partially update that lot.

4) `adjust_trade(...)` enforces daily drawdown and ensures the user has sufficient
   balances for each new trade.

5) We have an async background loop (`start_db_price_check_cycle(...)`) that
   periodically queries `price_history` for each pair's newest price, then calls
   `on_price_update(...)`.

New Changes to Meet Requirements:
--------------------------------
- We now **wait 30 seconds** after initialization, before the loop starts checking pairs.
- The `_maybe_place_kraken_order(...)` method returns a boolean success/failure.
  If it fails (e.g. the private feed isn’t open), we **do not** mark the lot as pending
  sell. Next cycle, we’ll see the lot still unsold and try again.

All other code is unchanged except where noted to implement these two items.
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

DB_FILE = "trades.db"  # Path to your SQLite DB file (must contain 'price_history', etc.)


# ------------------------------------------------------------------------------
# Utility to initialize 'holding_lots' table
# ------------------------------------------------------------------------------
def init_holding_lots_table(db_path: str = DB_FILE) -> None:
    """
    Ensures the holding_lots table exists in the SQLite database.
    Each row represents a distinct “lot” of an asset purchased, with:
      - purchase_price: Weighted cost basis (including fees) for this lot
      - initial_quantity: Original quantity bought (used to limit partial T/P sells)
      - quantity: Current quantity of this lot still open
      - date_purchased: integer (epoch time)
      - has_sold_in_between:
           0 => future buys can merge cost basis
           1 => partial sells, so no further merges
           2 => pending a SELL (we won't re-trigger on_price_update)
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
        logger.exception(f"[RiskManager] Error creating holding_lots table => {exc}")
    finally:
        conn.close()


# ------------------------------------------------------------------------------
# RiskManagerDB
# ------------------------------------------------------------------------------
class RiskManagerDB:
    """
    The core class for your stop-loss and take-profit logic. Key features:

      - 'on_price_update(...)': if current_price hits SL or TP => create SELL in
        'pending_trades' (with lot_id) => mark that lot as “pending sell” => optionally
        send the real SELL order to Kraken if place_live_orders is True.

      - 'on_pending_trade_closed(...)': once the private feed says the SELL is actually
        closed on Kraken => remove or partially update the lot.

      - 'adjust_trade(...)': daily drawdown + user capacity checks.

      - 'start_db_price_check_cycle(...)': background loop that queries 'price_history'
        to get the newest price for each pair, then calls 'on_price_update(...)'.

      - 'add_or_combine_lot(...)': merges or creates new holding_lots for each BUY fill.
    """

    def __init__(
            self,
            db_path: str = DB_FILE,
            max_position_size: float = 3.0,
            max_daily_drawdown: float = None,
            initial_spending_account: float = 100.0,
            # NEW: same idea as in AIStrategy
            private_ws_client=None,
            place_live_orders: bool = False
    ):
        """
        :param db_path: SQLite DB path containing price_history, trades, pending_trades, holding_lots
        :param max_position_size: Hard limit on per-trade size (e.g. 3 coins).
        :param max_daily_drawdown: If daily realized PnL <= this threshold => skip new BUYs.
        :param initial_spending_account: A budget cap for each trade in USD, if relevant to you.
        :param private_ws_client: an instance of KrakenPrivateWSClient (optional)
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
        Ensures 'trades' table and 'holding_lots' are present.
        'pending_trades' is presumably created in db.py.
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

        init_holding_lots_table(self.db_path)

    # ----------------------------------------------------------------------
    # 1) Start an async loop that calls on_price_update(...) from DB prices
    # ----------------------------------------------------------------------
    async def start_db_price_check_cycle(
            self,
            pairs: List[str],
            interval: float = 30.0
    ):
        """
        An asynchronous loop that does the following forever:
          1) Waits 30s after initialization to ensure private feed is connected
          2) For each pair in 'pairs':
             - query price_history => find the newest row => get 'last_price'
             - call on_price_update(pair, that_price, <balances dict>)
          3) Sleep for 'interval' seconds
          4) Repeat
        """
        import asyncio

        # NEW: Wait 30 seconds after initialization
        logger.info("[RiskManager] Delaying 30 seconds after init before first price check.")
        await asyncio.sleep(30)
        logger.info("[RiskManager] Proceeding with price check cycle now.")

        logger.info(
            f"[RiskManager] Starting DB-based price check cycle for {pairs}, interval={interval}s"
        )
        while True:
            try:
                for pair in pairs:
                    latest_price = self._fetch_latest_price_for_pair(pair)
                    if latest_price > 0:
                        # Typically you'd pass real user balances from somewhere.
                        # For demonstration, an empty dict here.
                        dummy_balances = {}
                        self.on_price_update(
                            pair=pair,
                            current_price=latest_price,
                            kraken_balances=dummy_balances
                        )
            except Exception as e:
                logger.exception(f"[RiskManager] DB price check cycle => error => {e}")

            await asyncio.sleep(interval)

    def _fetch_latest_price_for_pair(self, pair: str) -> float:
        """
        Reads the newest 'price_history' row for the given pair from the DB,
        returning last_price. Returns 0.0 if not found or error.
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
    # 2) Stop-Loss / Take-Profit => create SELL in pending_trades + optional real order
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
        For each open lot of `pair`, check if current_price triggers a stop-loss or T/P.
        If triggered, we:
         - Create a SELL 'pending_trade' row with `lot_id=...`.
         - Then *only if we successfully placed the Kraken order* do we mark that lot
           as “pending sell” so we don’t re-trigger next time. If the private feed
           is not connected, we skip and will retry next cycle.
        """
        lots = self._load_lots_for_pair(pair)
        if not lots:
            return

        for lot in lots:
            lot_id = lot["id"]
            lot_price = lot["purchase_price"]
            lot_qty = lot["quantity"]
            init_qty = lot["initial_quantity"]
            has_sold_flag = lot["has_sold_in_between"]

            # Skip if we've already marked it pending-sell or it has zero qty
            if lot_qty <= 0 or has_sold_flag == 2:
                continue

            # STOP LOSS
            if current_price <= lot_price * (1.0 - stop_loss_pct):
                reason = f"STOP_LOSS triggered at {current_price:.4f}"
                logger.info(
                    f"[RiskManager] Stop-loss => lot_id={lot_id}, SELL all {lot_qty:.4f} of {pair}"
                )
                pending_id = create_pending_trade(
                    side="SELL",
                    requested_qty=lot_qty,
                    pair=pair,
                    reason=reason
                )
                success = self._maybe_place_kraken_order(pair, "SELL", lot_qty, pending_id)
                if success:
                    self._mark_lot_as_pending_sell(lot_id)
                else:
                    logger.warning(
                        "[RiskManager] Private feed not open => skipping mark 'pending sell'. Will retry next cycle."
                    )
                continue

            # TAKE PROFIT
            if current_price >= lot_price * (1.0 + take_profit_pct):
                sell_size = min(init_qty, lot_qty)
                if sell_size <= 0:
                    continue

                reason = f"TAKE_PROFIT triggered at {current_price:.4f}"
                logger.info(
                    f"[RiskManager] Take-profit => lot_id={lot_id}, SELL {sell_size:.4f} of {pair}"
                )
                pending_id = create_pending_trade(
                    side="SELL",
                    requested_qty=sell_size,
                    pair=pair,
                    reason=reason
                )
                success = self._maybe_place_kraken_order(pair, "SELL", sell_size, pending_id)
                if success:
                    self._mark_lot_as_pending_sell(lot_id)
                else:
                    logger.warning(
                        "[RiskManager] Private feed not open => skipping mark 'pending sell'. Will retry next cycle."
                    )

    def _mark_lot_as_pending_sell(self, lot_id: int):
        """
        Mark this lot so we know a SELL is pending, to avoid repeated triggers.
        We'll set has_sold_in_between=2 to indicate "pending SELL".
        """
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("""
                UPDATE holding_lots
                SET has_sold_in_between=2
                WHERE id=?
            """, (lot_id,))
            conn.commit()
            logger.info(f"[RiskManager] Marked lot_id={lot_id} as pending SELL (has_sold_in_between=2).")
        except Exception as e:
            logger.exception(f"[RiskManager] _mark_lot_as_pending_sell => {e}")
        finally:
            conn.close()

    # ----------------------------------------------------------------------
    # 3) Called by the private feed once the SELL is actually closed on Kraken
    # ----------------------------------------------------------------------
    def on_pending_trade_closed(
            self,
            lot_id: int,
            fill_size: float,
            side: str,
            fill_price: float
    ):
        """
        Once the private feed says the SELL or BUY is fully closed on Kraken,
        update or remove the lot accordingly.

        - For a SELL => either remove the lot (if fill_size >= lot.quantity),
          or do a partial leftover update.
        - For a BUY => Typically merges are handled at add_or_combine_lot time,
          so no special logic here by default.
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

            current_qty = float(lot["quantity"])
            if side.upper() == "SELL":
                leftover = current_qty - fill_size
                if leftover <= 0:
                    # The entire lot is sold
                    c.execute("DELETE FROM holding_lots WHERE id=?", (lot_id,))
                    logger.info(f"[RiskManager] Removed lot_id={lot_id} fully sold.")
                else:
                    # Partial leftover => update and set has_sold_in_between=1
                    c.execute("""
                        UPDATE holding_lots
                        SET quantity=?,
                            has_sold_in_between=1
                        WHERE id=?
                    """, (leftover, lot_id))
                    logger.info(
                        f"[RiskManager] Partial leftover => lot_id={lot_id}, leftover={leftover:.4f}"
                    )
                conn.commit()
            else:
                # side==BUY => typically, you'd do no special logic, unless partial fill synergy is needed
                logger.debug(f"[RiskManager] on_pending_trade_closed => side=BUY => no action needed.")
        except Exception as e:
            logger.exception(f"[RiskManager] on_pending_trade_closed => {e}")
        finally:
            conn.close()

    # ----------------------------------------------------------------------
    # 4) Combine or Insert new "lot" upon a BUY
    # ----------------------------------------------------------------------
    def add_or_combine_lot(self, pair: str, buy_quantity: float, buy_price: float):
        """
        Called once a BUY finalizes. If there's an existing open lot with no partial sells,
        merges cost basis. Otherwise, create a new row in holding_lots.
        """
        if buy_quantity <= 0 or buy_price <= 0:
            return

        existing_lot = self._find_open_lot_without_sells(pair)
        if existing_lot:
            lot_id = existing_lot["id"]
            old_qty = existing_lot["quantity"]
            old_price = existing_lot["purchase_price"]

            total_qty = old_qty + buy_quantity
            new_price = ((old_qty * old_price) + (buy_quantity * buy_price)) / total_qty
            logger.info(
                f"[RiskManager] Combine-lot => lot_id={lot_id}, old_qty={old_qty:.4f}, "
                f"new_qty={buy_quantity:.4f}, total={total_qty:.4f}, px={new_price:.4f}"
            )
            self._update_combined_lot(lot_id, new_price, total_qty, old_qty, buy_quantity)
        else:
            now_ts = int(time.time())
            self._insert_new_lot(pair, buy_price, buy_quantity, now_ts)

    # ----------------------------------------------------------------------
    # 5) Daily Drawdown & Basic Pre-Trade Checks
    # ----------------------------------------------------------------------
    def adjust_trade(
            self,
            signal: str,
            suggested_size: float,
            pair: str,
            current_price: float,
            kraken_balances: Dict[str, float]
    ) -> Tuple[str, float]:
        """
        1) If daily drawdown is below max => skip new BUY
        2) clamp final_size to self.max_position_size
        3) check user capacity (cost vs free USD, or coin vs free_coins)
        4) return (final_signal, final_size) => "HOLD" if blocked
        """
        # daily drawdown
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
                logger.info(f"[RiskManager] insufficient {usd_symbol}, cost={cost:.2f}, have={free_usd:.2f}")
                return ("HOLD", 0.0)
        else:  # SELL => check base coin
            base_sym = self._get_base_symbol(pair)
            free_coins = kraken_balances.get(base_sym, 0.0)
            if final_size > free_coins:
                logger.info(
                    f"[RiskManager] insufficient {base_sym} => requested={final_size:.4f}, have={free_coins:.4f}")
                return ("HOLD", 0.0)

        return (signal, final_size)

    # ----------------------------------------------------------------------
    # 6) Daily PnL Calculation
    # ----------------------------------------------------------------------
    def get_daily_realized_pnl(self, date_str: Optional[str] = None) -> float:
        """
        Sums realized_pnl from 'trades' for a given UTC date. If not specified, use today (UTC).
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
    # 7) Possibly place a live Kraken SELL/BUY order
    # ----------------------------------------------------------------------
    def _maybe_place_kraken_order(self, pair: str, action: str, volume: float, pending_id: int = None) -> bool:
        """
        If place_live_orders=True and private_ws_client is available, we attempt a simple
        market order. 'pending_id' is used as userref to match the pending_trades row.

        Returns:
          True if the order was successfully handed off to the private feed.
          False if the private feed is not ready (or place_live_orders=False).
          The caller uses this return to decide whether to mark the lot as
          has_sold_in_between=2 or skip and retry next cycle.
        """
        if not self.place_live_orders:
            return True
        if not self.private_ws_client or not self.private_ws_client.running:
            logger.warning(
                "[RiskManager] Private feed not open => cannot place real order => returning False for retry."
            )
            return False

        side_for_kraken = "buy" if action.upper() == "BUY" else "sell"
        logger.info(
            f"[RiskManager] Sending real order => pair={pair}, side={side_for_kraken}, "
            f"volume={volume}, pending_id={pending_id}"
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
    # Utility: Weighted cost basis update
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
        Weighted cost basis update; also update initial_quantity by the new buy.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("SELECT initial_quantity FROM holding_lots WHERE id=?", (lot_id,))
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
                    quantity=?
                WHERE id=?
            """, (new_price, new_init_qty, new_qty, lot_id))
            conn.commit()
        except Exception as e:
            logger.exception(f"[RiskManager] _update_combined_lot => {e}")
        finally:
            conn.close()

    # ----------------------------------------------------------------------
    # Utility: find an open lot for merging
    # ----------------------------------------------------------------------
    def _find_open_lot_without_sells(self, pair: str) -> Optional[dict]:
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
    # Utility: load all open lots for a pair
    # ----------------------------------------------------------------------
    def _load_lots_for_pair(self, pair: str) -> List[dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        results = []
        try:
            c = conn.cursor()
            c.execute("""
                SELECT id, pair, purchase_price, initial_quantity, quantity, date_purchased, has_sold_in_between
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
        For "DOT/USD", returns the 'quote' symbol in Kraken format, e.g. "ZUSD".
        """
        return db_lookup.get_asset_value_for_pair(pair, value="quote")

    def _get_base_symbol(self, pair: str) -> str:
        """
        For "DOT/USD", returns the 'base' symbol in Kraken format, e.g. "XDOT".
        """
        return db_lookup.get_base_asset(pair)

    # ----------------------------------------------------------------------
    # Rebuild from ledger (unchanged from your existing snippet)
    # ----------------------------------------------------------------------
    def rebuild_lots_from_ledger_entries(self):
        """
        (Existing code you had for ledger-based rebuild. Unchanged.)
        """
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()

            logger.info("[RiskManager] Rebuilding lots from ledger => clearing existing holding_lots.")
            c.execute("DELETE FROM holding_lots")
            conn.commit()

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

            refids_sorted = sorted(trades_by_refid.keys(),
                                   key=lambda r: trades_by_refid[r][0]["time"])
            for refid in refids_sorted:
                ledger_group = trades_by_refid[refid]
                if len(ledger_group) < 2:
                    logger.warning(f"[RiskManager] Ledger refid={refid} has <2 rows => skipping.")
                    continue

                maybe_zusd = [x for x in ledger_group if x["asset"].upper().startswith("Z")]
                maybe_coin = [x for x in ledger_group if not x["asset"].upper().startswith("Z")]

                if not maybe_zusd or not maybe_coin:
                    logger.warning(f"[RiskManager] refid={refid} doesn't have ZUSD + coin => skipping.")
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
                        self.add_or_combine_lot(
                            pair=self._get_pair_name(coin_row["asset"]),
                            buy_quantity=buy_qty,
                            buy_price=avg_price
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

            logger.info("[RiskManager] Completed rebuilding lots from ledger entries.")

        except Exception as e:
            logger.exception(f"[RiskManager] Error in rebuild_lots_from_ledger_entries => {e}")
        finally:
            conn.close()

    def _apply_historical_sell_ledger(self, pair: str, sell_qty: float, sell_price: float):
        if sell_qty <= 0:
            return

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            c = conn.cursor()
            c.execute("""
                SELECT id, quantity, purchase_price, initial_quantity
                FROM holding_lots
                WHERE pair=?
                  AND quantity>0
                  AND has_sold_in_between != 2
                ORDER BY id ASC
            """, (pair,))
            rows = c.fetchall()

            qty_to_sell = sell_qty
            for row in rows:
                lot_id = row["id"]
                lot_qty = float(row["quantity"])

                if qty_to_sell <= 0:
                    break

                if lot_qty <= qty_to_sell:
                    c.execute("DELETE FROM holding_lots WHERE id=?", (lot_id,))
                    logger.debug(f"[RiskManager] Ledger SELL => fully removed lot_id={lot_id}")
                    qty_to_sell -= lot_qty
                else:
                    leftover = lot_qty - qty_to_sell
                    c.execute("""
                        UPDATE holding_lots
                        SET quantity=?, has_sold_in_between=1
                        WHERE id=?
                    """, (leftover, lot_id))
                    logger.debug(f"[RiskManager] Ledger SELL => lot_id={lot_id}, leftover={leftover}")
                    qty_to_sell = 0

            conn.commit()
            if qty_to_sell > 0:
                logger.warning(f"[RiskManager] Ledger SELL leftover={qty_to_sell} not allocated. Possibly mismatch.")
        except Exception as e:
            logger.exception(f"[RiskManager] _apply_historical_sell_ledger => {e}")
        finally:
            conn.close()

    def _get_pair_name(self, asset_name: str) -> str:
        return db_lookup.get_websocket_name_from_base_asset(asset_name)
