import sqlite3
import time
import logging
import datetime
from typing import Dict, List, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

DB_FILE = "trades.db"


class ReportingAndAnalytics:
    """
    ReportingAndAnalytics is responsible for computing short-term performance metrics,
    such as:
      - 24-hour realized PnL (in USD)
      - total fees
      - trade volumes
      - trade counts
      - trade 'effectiveness' or average % gain
      - open positions / cost basis

    This class uses ledger_entries (type='trade') to reconstruct all buys/sells by coin,
    applying FIFO or average cost for PnL. It also can read from trades table if needed
    (for additional references or matching 'order_id', 'kraken_trade_id' etc.).

    Basic usage:
      1) r = ReportingAndAnalytics(db_path="trades.db")
      2) r.load_ledger_data()
      3) r.build_fifo_positions()
      4) daily_pnl = r.get_24h_realized_pnl()
      5) effectiveness = r.get_trade_effectiveness()
      6) or r.generate_report()
    """

    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path
        self.ledger_df: pd.DataFrame = pd.DataFrame()
        self.fifo_trades: List[dict] = []  # Will store each buy/sell match with realized PnL
        self.open_positions: Dict[str, List[Tuple[float, float]]] = {}
        # Example of open_positions[symbol] = list of (quantity, cost_basis_per_unit)
        # if using FIFO-lots approach.

    def load_ledger_data(self) -> None:
        """
        Loads ledger_entries from the DB into a pandas DataFrame for easier manipulation.
        Filters only rows with type='trade' because those are actual buy/sell trades.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query("""
                SELECT 
                    ledger_id,
                    refid,
                    time,
                    type,
                    subtype,
                    asset,
                    amount,
                    fee,
                    balance
                FROM ledger_entries
                WHERE type='trade'
                ORDER BY time ASC, ledger_id ASC
            """, conn)
            self.ledger_df = df
            logger.info(f"[Reporting] Loaded {len(df)} ledger trade-rows from DB.")
        except Exception as e:
            logger.exception(f"[Reporting] Error loading ledger data => {e}")
        finally:
            conn.close()

    def build_fifo_positions(self) -> None:
        """
        Constructs a FIFO-based PnL framework from the ledger data. For each row
        in ascending time:
          - If asset == 'ZUSD' and amount < 0 => this is a buy (USD outflow).
            We'll look up the matching row(s) with the same refid for the crypto asset(s).
          - If asset == 'ZUSD' and amount > 0 => this is a sell (USD inflow).
            We'll also look up the matching row(s) for the crypto asset(s) and realize PnL.

        This method populates:
           - self.fifo_trades: a list of dict with {time, symbol, side, qty, cost_basis, proceeds, realized_pnl, fee}
           - self.open_positions: a dictionary of lots for each symbol (if net is still open).

        Implementation detail:
        - We rely on the fact that each trade has exactly 2 ledger rows with the same refid:
            1) ZUSD row with + or - amount
            2) Crypto row with + or - amount
          If you have combos or partial fills of multiple cryptos, adapt accordingly.
        """
        if self.ledger_df.empty:
            logger.warning("[Reporting] No ledger data loaded; call load_ledger_data() first.")
            return

        # Because each 'trade' typically has 2 lines with the same refid,
        # we can group by refid. For each group, we see which asset is ZUSD vs the crypto asset.
        grouped = self.ledger_df.groupby("refid", sort=False)

        # We'll keep a dict: open_positions[symbol] = list of (qty, cost_per_unit)
        self.open_positions = {}
        self.fifo_trades = []

        for refid, group_df in grouped:
            if len(group_df) < 2:
                # Possibly an edge case if there's a partial fill or something
                continue

            # Put them in ascending order of time just in case
            # Usually they share the same time, but we do it anyway
            group_df = group_df.sort_values("time")
            # We'll separate the ZUSD row from the crypto row
            row_usd = group_df[group_df["asset"] == "ZUSD"]
            row_crypto = group_df[group_df["asset"] != "ZUSD"]

            if row_usd.empty or row_crypto.empty:
                # might happen if e.g. there's a stake reward or margin
                continue

            # We assume there's only 1 row for ZUSD in normal spot trade
            # and only 1 row for the crypto asset
            usd_amt = float(row_usd["amount"].sum())  # Should be 1 row
            usd_fee = float(row_usd["fee"].sum())
            time_val = float(row_usd["time"].iloc[0])
            symbol = row_crypto["asset"].iloc[0]
            crypto_amt = float(row_crypto["amount"].sum())  # can be positive or negative
            crypto_fee = float(row_crypto["fee"].sum())

            # net fee in USD, but also might consider crypto fees if that occurs
            total_fee = usd_fee + crypto_fee

            # If usd_amt < 0 => it's a BUY (ZUSD outflow)
            #    crypto_amt > 0 => we gained crypto
            # If usd_amt > 0 => it's a SELL (ZUSD inflow)
            #    crypto_amt < 0 => we lost crypto
            side = "BUY" if usd_amt < 0 else "SELL"
            abs_usd = abs(usd_amt)
            abs_crypto = abs(crypto_amt)

            if side == "BUY":
                # cost_per_unit => abs_usd / abs_crypto
                cost_per_unit = abs_usd / abs_crypto if abs_crypto else 0
                self._add_lot(symbol, abs_crypto, cost_per_unit)

                trade_info = {
                    "time": time_val,
                    "symbol": symbol,
                    "side": side,
                    "quantity": abs_crypto,
                    "cost_basis": abs_usd,  # total cost in USD
                    "proceeds": 0.0,  # none for a buy
                    "realized_pnl": 0.0,
                    "fee": total_fee,
                    "refid": refid
                }
                self.fifo_trades.append(trade_info)

            else:  # SELL
                # We have abs_crypto coins being sold. We need to remove that from open_positions
                # to figure out cost basis for the portion sold.
                cost_for_this_sell, realized_pnl_for_this_sell = self._remove_lots_for_sale(
                    symbol, abs_crypto, abs_usd, total_fee
                )

                trade_info = {
                    "time": time_val,
                    "symbol": symbol,
                    "side": side,
                    "quantity": abs_crypto,
                    "cost_basis": cost_for_this_sell,
                    "proceeds": abs_usd,
                    "realized_pnl": realized_pnl_for_this_sell,
                    "fee": total_fee,
                    "refid": refid
                }
                self.fifo_trades.append(trade_info)

        # Sort self.fifo_trades by time to be sure
        self.fifo_trades = sorted(self.fifo_trades, key=lambda x: x["time"])

    def _add_lot(self, symbol: str, qty: float, cost_per_unit: float):
        """
        For a BUY: add a lot to open_positions for the symbol.
        FIFO approach => just append to the list.
        """
        if symbol not in self.open_positions:
            self.open_positions[symbol] = []
        self.open_positions[symbol].append((qty, cost_per_unit))

    def _remove_lots_for_sale(
            self,
            symbol: str,
            qty_to_sell: float,
            usd_proceeds: float,
            total_fee: float
    ) -> Tuple[float, float]:
        """
        For a SELL: remove `qty_to_sell` from the open_positions for `symbol`,
        using FIFO. Compute cost_of_this_sale, and thus realized_pnl for that portion.

        Return:
          (total_cost, realized_pnl)

        realized_pnl = (proceeds - cost_of_this_sale - total_fee?),
        or you can consider fees separately.
        Typically you'd subtract the fee from the net proceeds.
        It's up to you how to represent it. This example subtracts the fee from the proceeds.
        """
        if symbol not in self.open_positions or not self.open_positions[symbol]:
            # no open position => possibly short or zero => just treat cost=0
            logger.warning(f"[Reporting] SELL but no open lot for symbol={symbol}. qty={qty_to_sell}")
            return (0.0, 0.0)

        lots = self.open_positions[symbol]
        total_cost = 0.0
        qty_needed = qty_to_sell

        # We iterate over FIFO lots
        i = 0
        while i < len(lots) and qty_needed > 1e-12:
            lot_qty, lot_cost = lots[i]
            if lot_qty <= qty_needed + 1e-12:
                # We use up this entire lot
                total_cost += lot_qty * lot_cost
                qty_needed -= lot_qty
                # remove this lot from the list
                lots.pop(i)
            else:
                # We only use part of this lot
                total_cost += qty_needed * lot_cost
                new_qty = lot_qty - qty_needed
                lots[i] = (new_qty, lot_cost)  # update this lot with remainder
                qty_needed = 0.0
                i += 1

        # net_proceeds = usd_proceeds - total_fee (treating entire fee as from proceeds)
        # But you can also do partial if your exchange reports separate lines, or
        # keep the fee as a separate line item. We'll do the simpler approach here.
        net_proceeds = usd_proceeds - total_fee
        realized_pnl = net_proceeds - total_cost
        return (total_cost, realized_pnl)

    def get_24h_realized_pnl(self) -> float:
        """
        Returns the sum of realized PnL for all SELL trades that occurred
        in the last 24 hours. Also subtracts any fees if not already subtracted
        in your calculation logic. (Here we've already subtracted fees in the
        `_remove_lots_for_sale` method.)
        """
        if not self.fifo_trades:
            return 0.0

        now_ts = time.time()
        one_day_ago = now_ts - 86400

        pnl = 0.0
        for t in self.fifo_trades:
            if t["side"] == "SELL" and t["time"] >= one_day_ago:
                pnl += t["realized_pnl"]
        return pnl

    def get_daily_trade_volume(self) -> float:
        """
        Returns total notional trade volume (USD) in last 24h.
        For BUY trades, that is cost_basis. For SELL trades, that is proceeds.
        """
        if not self.fifo_trades:
            return 0.0

        now_ts = time.time()
        one_day_ago = now_ts - 86400

        volume = 0.0
        for t in self.fifo_trades:
            if t["time"] >= one_day_ago:
                if t["side"] == "BUY":
                    volume += t["cost_basis"]  # how much USD was spent
                else:  # SELL
                    volume += t["proceeds"]  # how much USD was received
        return volume

    def get_number_of_trades_24h(self) -> int:
        """
        Count how many trades (refid) in the last 24 hours.
        We can just count the unique SELL or BUY events if you prefer.
        """
        if not self.fifo_trades:
            return 0

        now_ts = time.time()
        one_day_ago = now_ts - 86400

        # We can just count each 'side' event as a "trade", or group by refid if you want
        count = sum(1 for t in self.fifo_trades if t["time"] >= one_day_ago)
        return count

    def get_trade_effectiveness(self) -> float:
        """
        Example metric: realized_pnl / total_cost for SELL trades in the last 24h,
        i.e. average % ROI of sells that closed in the last 24h.
        This is just a rough definition—adapt as needed.
        """
        now_ts = time.time()
        one_day_ago = now_ts - 86400

        total_cost_sold = 0.0
        total_pnl_sold = 0.0

        for t in self.fifo_trades:
            if t["side"] == "SELL" and t["time"] >= one_day_ago:
                total_cost_sold += t["cost_basis"]
                total_pnl_sold += t["realized_pnl"]

        if total_cost_sold > 0:
            return total_pnl_sold / total_cost_sold
        return 0.0

    def generate_report(self) -> Dict[str, float]:
        """
        Returns a dictionary of key metrics for the last 24 hours.
        You can expand these fields as you like.
        """
        daily_pnl = self.get_24h_realized_pnl()
        daily_vol = self.get_daily_trade_volume()
        trade_count = self.get_number_of_trades_24h()
        effectiveness = self.get_trade_effectiveness()  # fraction or ratio

        return {
            "24h_realized_pnl_usd": daily_pnl,
            "24h_trade_volume_usd": daily_vol,
            "24h_trade_count": trade_count,
            "24h_trade_effectiveness_ratio": effectiveness,
            "open_positions": self._get_open_positions_summary()
        }

    def _get_open_positions_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Summarizes self.open_positions for each symbol:
          - total quantity
          - average cost basis
        """
        summary = {}
        for sym, lots in self.open_positions.items():
            if not lots:
                continue
            total_qty = sum(x[0] for x in lots)
            if total_qty <= 0:
                continue
            # Weighted average cost
            weighted_cost = 0.0
            for (q, c) in lots:
                weighted_cost += q * c
            avg_cost = weighted_cost / total_qty if total_qty else 0.0
            summary[sym] = {
                "quantity": total_qty,
                "avg_cost_basis": avg_cost
            }
        return summary


def main():
    r = ReportingAndAnalytics(db_path=DB_FILE)
    r.load_ledger_data()
    r.build_fifo_positions()

    daily_report = r.generate_report()
    print("=== 24h Report ===")
    for k, v in daily_report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
