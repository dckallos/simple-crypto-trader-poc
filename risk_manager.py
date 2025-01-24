# =============================================================================
# FILE: risk_manager.py
# =============================================================================
"""
risk_manager.py

A RiskManagerDB class that stores sub-positions in an SQLite database rather
than in memory, allowing multiple sub-positions per pair (long or short). Each
sub-position has:
    - side ("long" or "short")
    - entry_price
    - size
    - created_at (timestamp)
    - optionally closed_at, exit_price, realized_pnl if the position is closed

Additionally, it checks for daily drawdown by summing up the realized PnL for
closed positions each day. If your daily realized PnL is below 'max_daily_drawdown',
no new trades are allowed. (They become HOLD.)

Usage:
    python risk_manager.py

This will run a small test scenario in the `if __name__ == "__main__":` block
which you can observe to confirm that DB-based sub-positions are functioning.

Requires:
    - SQLite DB (trades.db) or whatever path you specify.
    - 'sub_positions' table, automatically created if missing.
"""

import logging
import sqlite3
import time
import datetime
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Name of the table where sub-positions are stored
SUB_POSITIONS_TABLE = "sub_positions"


class RiskManagerDB:
    """
    RiskManagerDB handles the storage and enforcement of risk constraints
    using a database table 'sub_positions'. It supports:

    1) Multiple sub-positions per pair, each with:
       - side ("long" or "short")
       - entry_price
       - size
       - created_at
       - closed_at (NULL if open)
       - exit_price (NULL if open)
       - realized_pnl (NULL if open)
    2) Stop-loss and take-profit checks for open sub-positions.
    3) Daily drawdown checks, computing realized PnL from closed sub-positions.
    4) Clamping new trades to a max position size.

    Typical workflow:
        - Call initialize() once to ensure the table exists.
        - On a new trade signal: adjust_trade(...), if not forced to HOLD,
          add a sub-position to the DB.
        - Periodically call check_stop_loss_take_profit(...) for each pair
          to close any positions that triggered an exit condition.
        - Summarize daily realized PnL via get_daily_realized_pnl(...).

    NOTE: For concurrency: This design expects a single-thread usage or
    that your code handles DB locks carefully.
    """

    def __init__(
        self,
        db_path: str,
        max_position_size: float,
        stop_loss_pct: float = None,
        take_profit_pct: float = None,
        max_daily_drawdown: float = None
    ):
        """
        :param db_path: Path to your SQLite database file (e.g. "trades.db").
        :param max_position_size: Maximum allowed size for any new trade.
        :param stop_loss_pct: If set, e.g. 0.05 => stop out at 5% loss.
        :param take_profit_pct: If set, e.g. 0.10 => exit at 10% gain.
        :param max_daily_drawdown: If set, e.g. -0.02 => if daily realized
                                   PnL is <= -2%, skip new trades (force HOLD).
        """
        self.db_path = db_path
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_daily_drawdown = max_daily_drawdown

    def initialize(self) -> None:
        """
        Ensures the 'sub_positions' table exists in the DB.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute(f"""
                CREATE TABLE IF NOT EXISTS {SUB_POSITIONS_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT,
                    side TEXT,               -- 'long' or 'short'
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
        open a new sub-position in the DB.

        1) Check daily realized PnL => if daily drawdown is exceeded, force HOLD.
        2) If not exceeded, clamp trade size to max_position_size.
        3) If final_signal is BUY or SELL (not HOLD), insert a new row in
           sub_positions to track it as an open sub-position.

        :param signal: "BUY", "SELL", or "HOLD".
        :param suggested_size: The position size the AI suggests.
        :param pair: The trading pair, e.g. "ETH/USD".
        :param current_price: The current market price for the pair.
        :return: (final_signal, final_size) after risk checks.
                 final_size may be 0.0 if forced to HOLD.
        """
        # 1) daily drawdown check
        daily_pnl = self.get_daily_realized_pnl()
        if self.max_daily_drawdown is not None:
            if daily_pnl <= self.max_daily_drawdown:
                logger.warning(
                    f"Daily drawdown limit reached (PnL={daily_pnl:.4f}). Forcing HOLD."
                )
                return ("HOLD", 0.0)

        # 2) clamp size
        if signal not in ("BUY", "SELL"):
            # If it's already hold, or unknown signal, just return
            return ("HOLD", 0.0)

        final_size = min(suggested_size, self.max_position_size)
        if final_size <= 0.0:
            return ("HOLD", 0.0)

        # 3) Insert a new sub-position
        side_str = "long" if signal == "BUY" else "short"
        self._insert_sub_position(pair, side_str, current_price, final_size)
        logger.info(
            f"Opened new sub-position: pair={pair}, side={side_str}, "
            f"entry_price={current_price}, size={final_size}"
        )
        return (signal, final_size)

    def check_stop_loss_take_profit(self, pair: str, current_price: float) -> None:
        """
        Checks all open sub-positions for the given pair. If an open sub-position
        has unrealized_pct <= -stop_loss_pct or >= take_profit_pct, we close it
        immediately by computing realized PnL and updating the DB row.

        :param pair: e.g. "ETH/USD".
        :param current_price: The latest price of the pair.
        """
        open_positions = self._load_open_positions_for_pair(pair)
        if not open_positions:
            return

        for pos in open_positions:
            pos_id = pos["id"]
            side = pos["side"]
            entry_price = pos["entry_price"]
            size = pos["size"]

            if size <= 0:
                continue

            # Compute unrealized PnL% = (current - entry)/entry for long,
            # or (entry - current)/entry for short.
            unrealized_pct = self._compute_unrealized_pct(
                side, entry_price, current_price
            )

            triggered_close = False
            reason = ""
            # Stop-loss check
            if self.stop_loss_pct is not None and self.stop_loss_pct > 0:
                if unrealized_pct <= -abs(self.stop_loss_pct):
                    triggered_close = True
                    reason = f"Stop-loss triggered at {unrealized_pct:.2%}"

            # Take-profit check
            if not triggered_close and self.take_profit_pct is not None and self.take_profit_pct > 0:
                if unrealized_pct >= self.take_profit_pct:
                    triggered_close = True
                    reason = f"Take-profit triggered at {unrealized_pct:.2%}"

            if triggered_close:
                realized = self._compute_realized_pnl(side, entry_price, current_price, size)
                self._close_sub_position(pos_id, current_price, realized)
                logger.info(
                    f"Closed sub-position id={pos_id} for {pair}, reason={reason}, "
                    f"realized_pnl={realized:.4f}"
                )

    def get_daily_realized_pnl(self, date_str: str = None) -> float:
        """
        Computes the total realized PnL for all positions closed TODAY (by default)
        or for a specific date_str if given.

        :param date_str: If None, uses today's UTC date (YYYY-MM-DD).
                        If provided, use that date to filter.
        :return: The sum of realized_pnl for all sub-positions closed on that date.
        """
        if date_str is None:
            # default: today's date in UTC
            today_utc = datetime.datetime.utcnow().strftime("%Y-%m-%d")
            date_str = today_utc

        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            # We'll interpret closed_at as a UNIX timestamp. We'll convert it to a date string:
            #   DATE(closed_at, 'unixepoch') = date_str
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
    # DB HELPER METHODS
    # --------------------------------------------------------------------------
    def _insert_sub_position(self, pair: str, side: str, entry_price: float, size: float):
        """
        Insert a new sub-position row in the DB with current timestamp as created_at.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            ts = int(time.time())
            c.execute(f"""
                INSERT INTO {SUB_POSITIONS_TABLE} 
                    (pair, side, entry_price, size, created_at, closed_at, exit_price, realized_pnl)
                VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL)
            """, (pair, side, entry_price, size, ts))
            conn.commit()
        except Exception as e:
            logger.exception(f"Error inserting sub-position: {e}")
        finally:
            conn.close()

    def _load_open_positions_for_pair(self, pair: str) -> list:
        """
        Loads all open sub-positions (closed_at IS NULL) for the given pair.
        Returns a list of dicts: { id, side, entry_price, size, created_at }
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            c = conn.cursor()
            c.execute(f"""
                SELECT id, side, entry_price, size, created_at
                FROM {SUB_POSITIONS_TABLE}
                WHERE pair=?
                  AND closed_at IS NULL
            """, (pair,))
            rows = c.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.exception(f"Error loading open sub-positions for {pair}: {e}")
            return []
        finally:
            conn.close()

    def _close_sub_position(self, pos_id: int, exit_price: float, realized_pnl: float):
        """
        Marks a sub-position as closed by updating:
            closed_at, exit_price, realized_pnl
        """
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            ts = int(time.time())
            c.execute(f"""
                UPDATE {SUB_POSITIONS_TABLE}
                SET closed_at=?, exit_price=?, realized_pnl=?
                WHERE id=?
            """, (ts, exit_price, realized_pnl, pos_id))
            conn.commit()
        except Exception as e:
            logger.exception(f"Error closing sub-position id={pos_id}: {e}")
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # PNL & UTILITY METHODS
    # --------------------------------------------------------------------------
    def _compute_unrealized_pct(self, side: str, entry_price: float, current_price: float) -> float:
        """
        For a single position, compute (unrealized_PnL / entry_price).
        If side=long, raw_return = (current - entry).
        If side=short, raw_return = (entry - current).

        :return: fraction in range [-1..+∞), e.g. 0.05 => +5% gain
        """
        if side.lower() == "long":
            return (current_price - entry_price) / (entry_price + 1e-9)
        else:
            return (entry_price - current_price) / (entry_price + 1e-9)

    def _compute_realized_pnl(
        self,
        side: str,
        entry_price: float,
        exit_price: float,
        size: float
    ) -> float:
        """
        Realized PnL in *base currency* terms if you interpret 'size' as the quantity of the base asset.
        For a long:
           realized_pnl = (exit_price - entry_price) * size
        For a short:
           realized_pnl = (entry_price - exit_price) * size
        """
        if side.lower() == "long":
            return (exit_price - entry_price) * size
        else:
            return (entry_price - exit_price) * size


# ------------------------------------------------------------------------------
# TEST CODE / EXAMPLE USAGE
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # You can run "python risk_manager.py" to test in isolation.
    TEST_DB_PATH = "test_trades.db"
    # Remove the test DB if you want a fresh run each time:
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

    # Create an instance and ensure the table:
    rm = RiskManagerDB(
        db_path=TEST_DB_PATH,
        max_position_size=0.002,
        stop_loss_pct=0.05,   # 5% stop
        take_profit_pct=0.10, # 10% profit
        max_daily_drawdown=-0.02  # if daily realized PnL < -0.02, no new trades
    )
    rm.initialize()

    # 1) Open a new sub-position with a big suggested size => should clamp to 0.002
    signal, size = rm.adjust_trade(
        signal="BUY",
        suggested_size=0.01,
        pair="ETH/USD",
        current_price=1500.0
    )
    print(f"Trade opened => signal={signal}, size={size}")

    # 2) Another sub-position, SELL side:
    signal2, size2 = rm.adjust_trade(
        signal="SELL",
        suggested_size=0.0015,
        pair="ETH/USD",
        current_price=1600.0
    )
    print(f"Trade opened => signal={signal2}, size={size2}")

    # 3) Check if either hits stop-loss / take-profit at current price=1400 => might trigger stop
    rm.check_stop_loss_take_profit("ETH/USD", current_price=1400.0)

    # 4) Forcefully close a sub-position by faking a big jump => triggers take-profit
    rm.check_stop_loss_take_profit("ETH/USD", current_price=2000.0)

    # 5) Show daily realized PnL
    daily_pnl = rm.get_daily_realized_pnl()
    print(f"Today's realized PnL from closed sub-positions => {daily_pnl:.4f}")

    # 6) Attempt a new trade AFTER daily drawdown check
    # If your daily pnl is below -2%, no new trades are allowed => but hopefully we have gains
    signal3, size3 = rm.adjust_trade(
        signal="BUY",
        suggested_size=0.0005,
        pair="ETH/USD",
        current_price=2100.0
    )
    print(f"New trade after daily check => signal={signal3}, size={size3}")
