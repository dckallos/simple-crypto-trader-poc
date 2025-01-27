# =============================================================================
# FILE: risk_manager.py
# =============================================================================
"""
risk_manager.py

Encapsulates a DB-based approach to risk management with multiple sub-positions
per pair in 'sub_positions'. Each sub-position has:
  - side ("long" or "short")
  - entry_price
  - size
  - created_at
  - closed_at (NULL if open)
  - exit_price, realized_pnl when closed

We also track daily realized PnL; if it's below 'max_daily_drawdown', then new
trades are forced to HOLD. Additionally, 'max_position_size' clamps the
per-trade size, an optional 'max_position_value' can clamp cost in USD, and now
`initial_spending_account` ensures we don't exceed a total allocated capital.
"""

import logging
import sqlite3
import time
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SUB_POSITIONS_TABLE = "sub_positions"


class RiskManagerDB:
    """
    RiskManagerDB stores multiple sub-positions in the 'sub_positions' table and
    applies constraints:
      - daily drawdown limit => if daily realized PnL < max_daily_drawdown => force HOLD
      - clamp trade size by max_position_size
      - optionally clamp cost (size * price) by max_position_value
      - ensures total used capital doesn't exceed `initial_spending_account`
      - optionally auto-close sub-positions in check_stop_loss_take_profit() if
        stop_loss_pct or take_profit_pct are set.

    A typical flow:
      - AIStrategy calls: final_signal, final_size = adjust_trade("BUY", 0.01, "ETH/USD", 2000)
      - We do daily check => if okay, clamp size => ensure we don't exceed capital => insert sub-position => return final.
    """

    def __init__(
        self,
        db_path: str,
        max_position_size: float,
        stop_loss_pct: float = None,
        take_profit_pct: float = None,
        max_daily_drawdown: float = None,
        max_position_value: float = None,
        initial_spending_account: float = 40.0
    ):
        """
        :param db_path: path to your trades.db
        :param max_position_size: clamp for per-trade size (e.g. 0.001 BTC).
        :param stop_loss_pct: e.g. 0.05 => auto-close if sub-position is -5% from entry.
        :param take_profit_pct: e.g. 0.10 => auto-close if +10%.
        :param max_daily_drawdown: e.g. -0.02 => skip new trades if daily realized < -2%.
        :param max_position_value: optional clamp on cost => trade_size * price <= this.
        :param initial_spending_account: total allowed capital for all open buy positions.
        """
        self.db_path = db_path
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_daily_drawdown = max_daily_drawdown
        self.max_position_value = max_position_value
        self.initial_spending_account = initial_spending_account

    def initialize(self) -> None:
        """
        Ensures the 'sub_positions' table exists, storing open/closed sub-positions
        with (side, entry_price, size, created_at, closed_at, exit_price, realized_pnl).
        """
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute(f"""
                CREATE TABLE IF NOT EXISTS {SUB_POSITIONS_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT,
                    side TEXT,       -- "long" or "short"
                    entry_price REAL,
                    size REAL,
                    created_at INTEGER,
                    closed_at INTEGER,
                    exit_price REAL,
                    realized_pnl REAL
                )
            """)
            conn.commit()
            logger.info(f"Table '{SUB_POSITIONS_TABLE}' ensured in DB.")
        except Exception as e:
            logger.exception(f"Error creating '{SUB_POSITIONS_TABLE}' table: {e}")
        finally:
            conn.close()

    def adjust_trade(
        self,
        signal: str,
        suggested_size: float,
        pair: str,
        current_price: float
    ) -> (str, float):
        """
        Adjust an AI-proposed trade to comply with risk constraints, and if valid,
        open a sub-position in the DB.

        Steps:
          1) If daily realized PnL <= max_daily_drawdown => force HOLD
          2) If signal not in ("BUY","SELL") => return HOLD
          3) clamp size by self.max_position_size
          4) if signal=BUY => check total spent so far, ensure (spent + cost) <= initial_spending_account
          5) if self.max_position_value => clamp cost in USD
          6) insert sub-position row => side= "long" or "short"
          7) return final_signal, final_size
        """
        # 1) daily drawdown check
        if self.max_daily_drawdown is not None:
            daily_pnl = self.get_daily_realized_pnl()
            if daily_pnl <= self.max_daily_drawdown:
                logger.warning(
                    f"Daily drawdown limit => realizedPnL={daily_pnl:.4f} <= {self.max_daily_drawdown}, forcing HOLD."
                )
                return ("HOLD", 0.0)

        # 2) If not BUY/SELL => we hold
        if signal not in ("BUY", "SELL"):
            return ("HOLD", 0.0)

        # 3) clamp size
        final_size = min(suggested_size, self.max_position_size)
        if final_size <= 0:
            return ("HOLD", 0.0)

        cost = final_size * current_price

        # 4) If signal is BUY => check total spent
        if signal == "BUY":
            spent_so_far = self._sum_open_buy_positions()
            if spent_so_far + cost > self.initial_spending_account:
                logger.info(
                    f"[RiskManager] Not enough capital => spent_so_far={spent_so_far:.2f}, "
                    f"new_cost={cost:.2f}, limit={self.initial_spending_account:.2f}"
                )
                return ("HOLD", 0.0)

        # 5) clamp cost in USD if max_position_value is set
        if self.max_position_value and self.max_position_value > 0:
            if cost > self.max_position_value:
                new_size = self.max_position_value / max(current_price, 1e-9)
                logger.info(
                    f"[RiskManager] Clamping trade => cost {cost:.2f} > max_position_value={self.max_position_value}, "
                    f"new size={new_size:.6f}"
                )
                final_size = new_size
                if final_size <= 0:
                    return ("HOLD", 0.0)
                cost = final_size * current_price

        # insert sub-position => side= "long" or "short"
        side_str = "long" if signal == "BUY" else "short"
        self._insert_sub_position(pair, side_str, current_price, final_size)
        logger.info(
            f"[RiskManager] Opened new sub-position => pair={pair}, side={side_str}, "
            f"entry_price={current_price:.2f}, size={final_size:.6f}, cost={cost:.2f}"
        )

        return (signal, final_size)

    def check_stop_loss_take_profit(self, pair: str, current_price: float) -> None:
        """
        Optionally close sub-positions if they cross stop_loss_pct or take_profit_pct.
        For each open sub-position:
           - if side=long => unrealized % = (current_price - entry_price)/entry_price
           - if side=short => unrealized % = (entry_price - current_price)/entry_price
           If unrealized <= -stop_loss_pct => close
           If unrealized >= take_profit_pct => close
        """
        if not self.stop_loss_pct and not self.take_profit_pct:
            return

        open_positions = self._load_open_positions_for_pair(pair)
        if not open_positions:
            return

        for pos in open_positions:
            pos_id = pos["id"]
            side = pos["side"]
            entry = pos["entry_price"]
            size = pos["size"]

            # compute unrealized
            if side == "long":
                pct = (current_price - entry) / (entry + 1e-9)
            else:
                pct = (entry - current_price) / (entry + 1e-9)

            triggered = False
            reason = ""
            if self.stop_loss_pct and pct <= -abs(self.stop_loss_pct):
                triggered = True
                reason = f"Stop-loss => {pct*100:.2f}%"
            elif self.take_profit_pct and pct >= self.take_profit_pct:
                triggered = True
                reason = f"Take-profit => {pct*100:.2f}%"

            if triggered:
                realized = self._compute_realized_pnl(side, entry, current_price, size)
                self._close_sub_position(pos_id, current_price, realized)
                logger.info(
                    f"[RiskManager] Closed sub-position id={pos_id}, {pair} => {reason}, realized_pnl={realized:.4f}"
                )

    def get_daily_realized_pnl(self, date_str: str = None) -> float:
        """
        Sums realized_pnl from sub_positions *closed* on 'date_str' (UTC). If None,
        uses today's UTC date. Return 0.0 if none closed that day.
        """
        import datetime
        if not date_str:
            date_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")

        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            query = f"""
                SELECT IFNULL(SUM(realized_pnl), 0.0)
                FROM {SUB_POSITIONS_TABLE}
                WHERE closed_at IS NOT NULL
                  AND DATE(closed_at, 'unixepoch') = ?
            """
            c.execute(query, (date_str,))
            row = c.fetchone()
            if row and row[0] is not None:
                return float(row[0])
            return 0.0
        except Exception as e:
            logger.exception(f"Error computing daily realized pnl: {e}")
            return 0.0
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # DB sub-positions
    # --------------------------------------------------------------------------
    def _sum_open_buy_positions(self) -> float:
        """
        Returns the total cost of all open BUY sub-positions, i.e. sum(entry_price * size).
        This helps ensure we don't exceed initial_spending_account.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            query = f"""
                SELECT IFNULL(SUM(entry_price * size), 0.0)
                FROM {SUB_POSITIONS_TABLE}
                WHERE closed_at IS NULL
                  AND side IN ('long','BUY')
            """
            c.execute(query)
            row = c.fetchone()
            if row and row[0]:
                return float(row[0])
            return 0.0
        except Exception as e:
            logger.exception(f"Error summing open buy positions => {e}")
            return 0.0
        finally:
            conn.close()

    def _insert_sub_position(self, pair: str, side: str, entry_price: float, size: float):
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            now_ts = int(time.time())
            c.execute(f"""
                INSERT INTO {SUB_POSITIONS_TABLE} (
                    pair, side, entry_price, size, created_at, closed_at, exit_price, realized_pnl
                )
                VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL)
            """, (pair, side, entry_price, size, now_ts))
            conn.commit()
        except Exception as e:
            logger.exception(f"Error inserting sub-position => {e}")
        finally:
            conn.close()

    def _close_sub_position(self, pos_id: int, exit_price: float, realized_pnl: float):
        """
        Mark sub-position as closed => sets closed_at, exit_price, realized_pnl
        for the row with ID=pos_id.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            now_ts = int(time.time())
            c.execute(f"""
                UPDATE {SUB_POSITIONS_TABLE}
                SET closed_at=?, exit_price=?, realized_pnl=?
                WHERE id=?
            """, (now_ts, exit_price, realized_pnl, pos_id))
            conn.commit()
        except Exception as e:
            logger.exception(f"Error closing sub-position => {e}")
        finally:
            conn.close()

    def _load_open_positions_for_pair(self, pair: str) -> list:
        """
        Returns a list of open sub-positions for 'pair', each as a dict:
         {
           "id": int, "side": str, "entry_price": float, "size": float, "created_at": int
         }
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            c = conn.cursor()
            q = f"""
                SELECT id, side, entry_price, size, created_at
                FROM {SUB_POSITIONS_TABLE}
                WHERE pair=? AND closed_at IS NULL
            """
            c.execute(q, (pair,))
            rows = c.fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.exception(f"Error loading open sub-positions for {pair}: {e}")
            return []
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # Realized PnL Computation
    # --------------------------------------------------------------------------
    def _compute_realized_pnl(self, side: str, entry_price: float, exit_price: float, size: float) -> float:
        """
        For a single closed position, realized PnL in base currency terms:
          if side=long => (exit_price - entry_price) * size
          if side=short => (entry_price - exit_price) * size
        """
        if side.lower() in ("long","buy"):
            return (exit_price - entry_price) * size
        else:
            return (entry_price - exit_price) * size
