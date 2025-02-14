# =============================================================================
# FILE: risk_manager.py
# =============================================================================
"""
risk_manager.py

Now includes on_price_update(...) which handles each new price from the
public feed, checks stop-loss / take-profit thresholds, and, if triggered,
places a SELL. This approach does not rely on local sub-positions; it
derives net position and cost basis from the 'trades' table for each pair.
"""

import logging
import os
import sqlite3
import datetime
from typing import Tuple
from dotenv import load_dotenv

import db_lookup
from config_loader import ConfigLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from kraken_rest_manager import KrakenRestManager
load_dotenv()
rest_manager = KrakenRestManager(
    api_key=os.getenv("KRAKEN_API_KEY"),
    api_secret=os.getenv("KRAKEN_SECRET_API_KEY")
)

DB_FILE = "trades.db"  # Ensure this matches your actual DB path

class RiskManagerDB:
    def __init__(
        self,
        db_path: str = DB_FILE,
        max_position_size: float = 3.0,
        max_daily_drawdown: float = None,
        initial_spending_account: float = 100.0
    ):
        """
        :param db_path: path to your trades.db
        :param max_position_size: clamp for per-trade size (e.g. 3.0 means 3 coins).
        :param max_daily_drawdown: e.g. -0.02 => skip trades if daily realized < -2%
        :param initial_spending_account: max cost (USD) allowed per buy trade
        """
        self.db_path = db_path
        self.max_position_size = max_position_size
        self.max_daily_drawdown = max_daily_drawdown
        self.initial_spending_account = initial_spending_account

    def initialize(self) -> None:
        """
        Ensure 'trades' table exists. No sub-positions creation is done.
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
            logger.info("RiskManagerDB => 'trades' table ensured.")
        except Exception as e:
            logger.exception(f"[RiskManager] Error ensuring trades table => {e}")
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # STOP-LOSS / TAKE-PROFIT TRIGGER VIA PRICE UPDATE
    # --------------------------------------------------------------------------
    def on_price_update(
        self,
        pair: str,
        current_price: float,
        kraken_balances: dict,
        stop_loss_pct: float = ConfigLoader.get_value("stop_loss_percent"),
        take_profit_pct: float = ConfigLoader.get_value("take_profit_percent")
    ):
        """
        Called each time a new price for `pair` arrives from the public feed.
        1) We see if user has a net position in `pair` (via get_net_position_and_cost_basis).
        2) If net_position > 0, we check if price <= cost_basis*(1-SL%) or >= cost_basis*(1+TP%).
        3) If triggered, we attempt a SELL for the net_position (or partial).

        :param pair: e.g. "ETH/USD"
        :param current_price: market price from the feed
        :param kraken_balances: dict from fetch_kraken_balance(...) => e.g. {"ZUSD":500.0,"XETH":2.0}
        :param stop_loss_pct: e.g. 0.05 => -5% stop
        :param take_profit_pct: e.g. 0.10 => +10% TP
        """
        net_position, cost_basis = self.get_net_position_and_cost_basis(pair)
        if net_position <= 0:
            # no net holdings => nothing to do
            return

        # If price crosses stop-loss or take-profit threshold, we place SELL
        stop_trigger = (current_price <= cost_basis * (1.0 - stop_loss_pct))
        tp_trigger = (current_price >= cost_basis * (1.0 + take_profit_pct))

        if stop_trigger or tp_trigger:
            # SELL everything we hold, or partial if you prefer
            logger.info(
                f"[RiskManager] on_price_update => SL/TP triggered for {pair}, "
                f"net_pos={net_position:.4f}, cost_basis={cost_basis:.2f}, px={current_price:.2f}"
            )
            final_signal, final_size = self.adjust_trade(
                "SELL", net_position, pair, current_price, kraken_balances
            )
            if final_signal == "SELL" and final_size > 0:
                # Record pending trade or place actual order
                reason = "STOP_LOSS" if stop_trigger else "TAKE_PROFIT"
                pending_id = self._create_pending_trade(final_signal, final_size, pair, reason)
                logger.info(f"[RiskManager] => Created SELL pending trade for {final_size} {pair}. reason={reason}")

    def _create_pending_trade(self, side: str, qty: float, pair: str, reason: str) -> int:
        """
        Minimal placeholder method. You can call create_pending_trade(...) from db.py or
        your own logic if you prefer. This just returns the row ID.
        """
        import time
        conn = sqlite3.connect(self.db_path)
        new_id = None
        try:
            c = conn.cursor()
            c.execute("""
                INSERT INTO pending_trades (
                    created_at, pair, side, requested_qty, status, kraken_order_id, reason
                )
                VALUES (?, ?, ?, ?, 'pending', NULL, ?)
            """, (int(time.time()), pair, side, qty, reason))
            conn.commit()
            new_id = c.lastrowid
        except Exception as e:
            logger.exception(f"[RiskManager] Error creating pending trade => {e}")
        finally:
            conn.close()
        return new_id if new_id else 0

    # --------------------------------------------------------------------------
    # NET POSITION & COST BASIS (Weighed Approach)
    # --------------------------------------------------------------------------
    def get_net_position_and_cost_basis(self, pair: str) -> Tuple[float, float]:
        """
        Sums up all trades for this pair from the 'trades' table to find net_position
        and approximate cost basis. If net_position=0 => cost_basis=0.
        """
        conn = sqlite3.connect(self.db_path)
        rows = []
        try:
            c = conn.cursor()
            c.execute("""
                SELECT side, quantity, price
                FROM trades
                WHERE pair=?
                ORDER BY timestamp ASC
            """, (pair,))
            rows = c.fetchall()
        except Exception as e:
            logger.exception(f"[RiskManager] get_net_position_and_cost_basis => {e}")
        finally:
            conn.close()

        net_position = 0.0
        weighted_cost_sum = 0.0

        for (side, qty, px) in rows:
            if side.upper() == "BUY":
                net_position += qty
                weighted_cost_sum += (qty * px)
            elif side.upper() == "SELL":
                net_position_before = net_position
                net_position -= qty
                if net_position_before > 0:
                    fraction_sold = min(qty, net_position_before) / net_position_before
                    weighted_cost_sum -= fraction_sold * weighted_cost_sum

                if net_position <= 0:
                    net_position = 0.0
                    weighted_cost_sum = 0.0

        if net_position <= 0:
            return (0.0, 0.0)
        cost_basis = weighted_cost_sum / net_position
        return (net_position, cost_basis)

    # --------------------------------------------------------------------------
    # EXISTING adjust_trade(...) (unchanged from your new version)
    # --------------------------------------------------------------------------
    def adjust_trade(
        self,
        signal: str,
        suggested_size: float,
        pair: str,
        current_price: float,
        kraken_balances: dict
    ) -> (str, float):
        """
        Main pre-trade check from your updated risk_manager.
        ...
        """
        # 1) daily drawdown check
        if self.max_daily_drawdown is not None:
            daily_pnl = self.get_daily_realized_pnl()
            if daily_pnl <= self.max_daily_drawdown:
                logger.warning(
                    f"[RiskManager] daily drawdown => realizedPnL={daily_pnl:.4f}"
                    f" <= {self.max_daily_drawdown} => forcing HOLD."
                )
                return ("HOLD", 0.0)

        # 2) If the signal is not BUY/SELL => hold
        if signal not in ("BUY", "SELL"):
            return ("HOLD", 0.0)

        # 3) clamp trade size
        final_size = min(suggested_size, self.max_position_size)
        if final_size <= 0:
            return ("HOLD", 0.0)

        # BUY checks
        if signal == "BUY":
            cost = final_size * current_price
            # check initial_spending_account limit
            if cost > self.initial_spending_account:
                logger.info(
                    f"[RiskManager] cost={cost:.2f} > initial_spending_account={self.initial_spending_account:.2f}"
                )
                return ("HOLD", 0.0)

            # check real-time USD
            quote_asset = self._get_quote_symbol(pair)
            balances = rest_manager.fetch_balance()
            free_usd = balances.get(quote_asset, 0.0)
            if cost > free_usd:
                logger.info(
                    f"[RiskManager] Not enough {quote_asset} => cost={cost:.2f}, free={free_usd:.2f}"
                )
                return ("HOLD", 0.0)

        # SELL checks
        if signal == "SELL":
            base_symbol = self._get_base_symbol(pair)
            balances = rest_manager.fetch_balance()
            free_coins = balances.get(base_symbol, 0.0)
            if final_size > free_coins:
                logger.info(
                    f"[RiskManager] Not enough {base_symbol} => requested={final_size}, have={free_coins}"
                )
                return ("HOLD", 0.0)

        return (signal, final_size)

    def get_daily_realized_pnl(self, date_str: str = None) -> float:
        """
        Sums realized_pnl from 'trades' for the specified UTC date.
        If no date_str => use today's date (UTC).
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
                  AND DATE(timestamp, 'unixepoch') = ?
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

    # # Helpers for buy/sell checks
    # def _guess_quote_symbol(self, pair: str) -> str:
    #     import db_lookup
    #     quote_raw = db_lookup.get_asset_value_for_pair(pair, value="quote")
    #     if quote_raw == "USD":
    #         return "ZUSD"
    #     if quote_raw == "EUR":
    #         return "ZEUR"
    #     return quote_raw

    def _get_quote_symbol(self, pair: str) -> str:
        return db_lookup.get_asset_value_for_pair(pair, value="quote")

    # def _guess_base_symbol(self, base_symbol: str) -> str:
    #     if base_symbol in ("XBT", "BTC"):
    #         return "XXBT"
    #     return "X" + base_symbol

    def _get_base_symbol(self, pair: str) -> str:
        return db_lookup.get_base_asset(pair)
