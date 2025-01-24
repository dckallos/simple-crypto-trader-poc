#!/usr/bin/env python3
# =============================================================================
# FILE: backtest.py
# =============================================================================
"""
backtest.py

A robust, flexible backtesting module with a "Backtester" class
that can handle multi-asset or single-asset data. It integrates:

1) Single or multi-asset approach:
   - Provide one or more DataFrames with time-series data.

2) Slippage & fees:
   - You can specify slippage rate (fraction of price) on entry & exit
   - Per-trade or per-share fees

3) Multiple exit conditions:
   - Stop-loss (static or trailing)
   - Take-profit
   - Time-based / bar-based exit
   - Partial exit logic (50% exit at half of your profit target, for instance)

4) Large variety of metrics:
   - Per-trade PnL
   - Cumulative PnL
   - Max drawdown
   - Sharpe ratio
   - Sortino ratio
   - [Optional] daily returns

5) Extensible "prediction" approach:
   - Accept a scikit-learn model that implements .predict_proba() or .predict()
   - Or accept a user function that returns buy/sell signals or probabilities
   - For multi-asset, you can feed a model or function per symbol or a single
     universal approach

Usage (minimal example):
    python backtest.py

Or within another Python script:
    from backtest import Backtester

    # Suppose we have a data DataFrame with columns:
    #  ["timestamp", "feature_price", ...] sorted ascending
    # We'll define "predict_func" or use a scikit model with .predict_proba

    def my_predict_func(df_row):
        # return a probability that the price will go up
        return 0.7  # naive

    bt = Backtester(
       data=df,  # single asset
       predict_function=my_predict_func,
       buy_threshold=0.6,
       sell_threshold=0.4
    )
    trades, stats = bt.run_backtest()
    print("Backtest results:\n", stats)

Disclaimer: This is a general blueprint. Real usage may require
you to adapt file inputs, DB queries, or advanced logging.
"""

import logging
import math
import numpy as np
import pandas as pd
from typing import Callable, Optional, Dict, Any, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    """
    A powerful class for running backtests on a single or multi-asset dataset.
    By default, it maintains at most ONE position at a time (long or short),
    but you can adapt it to multi-position if desired.

    Key features:
      - Slippage & fees on entry & exit
      - Stop-loss & take-profit
      - Partial exit logic (50% exit at half TP)
      - Time-based exit (max hold bars)
      - Trailing stop if desired
      - Probabilistic approach (predict_proba) or discrete (predict=0/1)
      - Customizable predictions (pass a scikit model or a function)
      - Detailed trade logs
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        predict_function: Optional[Callable[[pd.DataFrame], float]] = None,
        model=None,
        buy_threshold: float = 0.6,
        sell_threshold: float = 0.4,
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.05,
        max_hold_bars: int = 50,
        trailing_stop: bool = False,
        partial_take_pct: float = 0.5,
        partial_exit_trigger: float = 0.5,
        slippage_rate: float = 0.0005,
        fee_rate: float = 0.001,
        track_daily_returns: bool = True,
    ):
        """
        :param data:
          Single asset => data is a DataFrame sorted ascending by time
            with a 'feature_price' column for entry/exit reference.
          Multi-asset => data is a dict: { "ETH": df_eth, "XBT": df_xbt, ... }
            each DataFrame must have "feature_price" sorted ascending.

        :param predict_function:
          A function that accepts a single-row DF or a single-row series
          and returns the probability of "up" or a discrete 0/1 signal.
          If None, we rely on `model`.

        :param model:
          A scikit-learn model with .predict_proba or .predict. If present,
          we override `predict_function`. If a row is passed in, we do model.predict_proba(row).

        :param buy_threshold:
          If probability of up > this => open long if flat.
        :param sell_threshold:
          If probability of up < this => open short if flat.
        :param stop_loss_pct:
          E.g. 0.03 => close if -3% from entry. Could also be used for trailing stops if trailing_stop=True.
        :param take_profit_pct:
          E.g. 0.05 => close if +5%. Also partial exit if partial_exit_trigger set.
        :param max_hold_bars:
          E.g. 50 => forcibly close after 50 bars if still open.
        :param trailing_stop:
          If True, the stop price "trail" as the position becomes profitable.
          For a long, if the current price is > entry_price, we raise the stop accordingly
          so that the distance remains the same as initial if we want static. Or you can adapt logic below.
        :param partial_take_pct:
          Fraction of position to exit on partial exit, e.g. 0.5 => 50% exit.
        :param partial_exit_trigger:
          E.g. 0.5 => if we reach half the take_profit => partial exit.
        :param slippage_rate:
          E.g. 0.0005 => 0.05% applied on buy & sell, so total cost is ~ 0.1% sometimes.
        :param fee_rate:
          E.g. 0.001 => 0.1% fee each trade.
        :param track_daily_returns:
          If True, we compute a daily time series of returns for advanced stats (Sortino, etc.).

        Example usage:
          data=some DF with "feature_price"
          predict_function=some function returning prob up
        """
        self.data = data
        self.predict_function = predict_function
        self.model = model

        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_hold_bars = max_hold_bars
        self.trailing_stop = trailing_stop
        self.partial_take_pct = partial_take_pct
        self.partial_exit_trigger = partial_exit_trigger
        self.slippage_rate = slippage_rate
        self.fee_rate = fee_rate
        self.track_daily_returns = track_daily_returns

        # We'll store trades in a big DataFrame eventually
        self.trades = pd.DataFrame()

    def run_backtest(self) -> (pd.DataFrame, Dict[str, Any]):
        """
        Runs the backtest. If multi-asset => we do them sequentially, or you
        can adapt to do parallel. For each asset we do a single-position approach.

        :return: (df_trades, stats)
        """
        if isinstance(self.data, dict):
            # multi-asset approach
            all_trades = []
            for symbol, df in self.data.items():
                logger.info(f"Starting backtest for {symbol} with {len(df)} rows.")
                df_trades_symbol = self._run_single_asset(df, symbol)
                all_trades.append(df_trades_symbol)

            self.trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        else:
            # single-asset approach
            logger.info(f"Starting backtest for single asset with {len(self.data)} rows.")
            self.trades = self._run_single_asset(self.data, "Asset")

        # Summarize
        if self.trades.empty:
            logger.info("No trades placed.")
            return self.trades, {}

        stats = self._compute_advanced_stats(self.trades)
        return self.trades, stats

    def _run_single_asset(self, df: pd.DataFrame, symbol: str = "Asset") -> pd.DataFrame:
        """
        Runs the single-asset backtest logic. Returns trades DataFrame for that asset.
        """
        # Confirm sorted ascending
        df = df.reset_index(drop=True)
        if not df.empty and df.index.is_monotonic_increasing:
            pass
        else:
            logger.debug(f"Sorting ascending by index for {symbol}.")
            df = df.sort_index()

        # We'll assume the df is sequential in index. Each row => next bar
        # We'll track position=0 (flat), +1 (long), -1 (short)
        # trailing_stop_price if trailing_stop is True
        position = 0
        entry_price = 0.0
        trailing_stop_price = None
        partial_exit_done = False

        trades = []
        i = 0
        while i < len(df):
            row = df.iloc[i]
            price = row["feature_price"]

            # Probability (or discrete label) of up
            p_up = self._predict_prob(row, df.iloc[[i]])  # pass single row

            if position == 0:
                # FLAT => see if we open a position
                if p_up > self.buy_threshold:
                    # BUY
                    open_price = price * (1 + self.slippage_rate)
                    fee_cost = open_price * self.fee_rate
                    position = 1
                    entry_price = open_price
                    trailing_stop_price = (entry_price * (1 - self.stop_loss_pct)
                                          if self.trailing_stop else None)
                    partial_exit_done = False

                    trades.append({
                        "symbol": symbol,
                        "action": "BUY_OPEN",
                        "index": i,
                        "timestamp": row.get("timestamp", i),
                        "price": open_price,
                        "fee": fee_cost,
                        "p_up": p_up,
                    })
                    i += 1
                elif p_up < self.sell_threshold:
                    # SELL SHORT
                    open_price = price * (1 - self.slippage_rate)
                    fee_cost = open_price * self.fee_rate
                    position = -1
                    entry_price = open_price
                    trailing_stop_price = (entry_price * (1 + self.stop_loss_pct)
                                          if self.trailing_stop else None)
                    partial_exit_done = False

                    trades.append({
                        "symbol": symbol,
                        "action": "SELL_OPEN",
                        "index": i,
                        "timestamp": row.get("timestamp", i),
                        "price": open_price,
                        "fee": fee_cost,
                        "p_up": p_up,
                    })
                    i += 1
                else:
                    # do nothing => next bar
                    i += 1
            else:
                # we have a position => track bar held
                bar_held = 0
                exit_trade = False

                while i < len(df) and not exit_trade:
                    row = df.iloc[i]
                    price = row["feature_price"]
                    p_up = self._predict_prob(row, df.iloc[[i]])

                    # compute returns
                    if position > 0:
                        raw_return = (price - entry_price) / entry_price
                    else:
                        raw_return = (entry_price - price) / entry_price

                    # trailing stop update
                    if self.trailing_stop and trailing_stop_price is not None:
                        if position > 0:
                            # if price has gone up enough, raise trailing stop
                            new_stop = price * (1 - self.stop_loss_pct)
                            if new_stop > trailing_stop_price:
                                trailing_stop_price = new_stop
                        else:
                            # short => if price has dropped, lower trailing stop
                            new_stop = price * (1 + self.stop_loss_pct)
                            if new_stop < trailing_stop_price:
                                trailing_stop_price = new_stop

                    # partial exit logic => if raw_return >= partial_exit_trigger * take_profit_pct
                    # e.g. if partial_exit_trigger=0.5 => half of the total take_profit => 0.025 if tp=0.05
                    if not partial_exit_done and raw_return >= (self.take_profit_pct * self.partial_exit_trigger) and raw_return > 0:
                        exit_price = (price * (1 - self.slippage_rate)
                                      if position > 0
                                      else price * (1 + self.slippage_rate))
                        fee_cost = exit_price * self.fee_rate * self.partial_take_pct
                        realized_pnl = (exit_price - entry_price) * (self.partial_take_pct) if position > 0 \
                            else (entry_price - exit_price) * (self.partial_take_pct)
                        trades.append({
                            "symbol": symbol,
                            "action": "PARTIAL_EXIT",
                            "index": i,
                            "timestamp": row.get("timestamp", i),
                            "price": exit_price,
                            "fee": fee_cost,
                            "pnl": realized_pnl,
                        })
                        partial_exit_done = True
                        i += 1
                        bar_held += 1
                        continue

                    # STOP-LOSS check
                    if self.trailing_stop and trailing_stop_price is not None:
                        # if price < trailing_stop_price => STOP
                        # for long
                        if position > 0 and price <= trailing_stop_price:
                            exit_price = price * (1 - self.slippage_rate)
                            fee_cost = exit_price * self.fee_rate
                            realized_pnl = (exit_price - entry_price)
                            trades.append({
                                "symbol": symbol,
                                "action": "TRAIL_STOP_OUT",
                                "index": i,
                                "timestamp": row.get("timestamp", i),
                                "price": exit_price,
                                "fee": fee_cost,
                                "pnl": realized_pnl
                            })
                            position = 0
                            exit_trade = True
                            i += 1
                            break
                        if position < 0 and price >= trailing_stop_price:
                            exit_price = price * (1 + self.slippage_rate)
                            fee_cost = exit_price * self.fee_rate
                            realized_pnl = (entry_price - exit_price)
                            trades.append({
                                "symbol": symbol,
                                "action": "TRAIL_STOP_OUT",
                                "index": i,
                                "timestamp": row.get("timestamp", i),
                                "price": exit_price,
                                "fee": fee_cost,
                                "pnl": realized_pnl
                            })
                            position = 0
                            exit_trade = True
                            i += 1
                            break
                    else:
                        # normal static stop-loss
                        if raw_return <= -abs(self.stop_loss_pct):
                            exit_price = (price * (1 - self.slippage_rate)
                                          if position > 0
                                          else price * (1 + self.slippage_rate))
                            fee_cost = exit_price * self.fee_rate
                            realized_pnl = (exit_price - entry_price) if position > 0 \
                                else (entry_price - exit_price)
                            trades.append({
                                "symbol": symbol,
                                "action": "STOP_OUT",
                                "index": i,
                                "timestamp": row.get("timestamp", i),
                                "price": exit_price,
                                "fee": fee_cost,
                                "pnl": realized_pnl
                            })
                            position = 0
                            exit_trade = True
                            i += 1
                            break

                    # TAKE-PROFIT check
                    if raw_return >= self.take_profit_pct:
                        exit_price = (price * (1 - self.slippage_rate)
                                      if position > 0
                                      else price * (1 + self.slippage_rate))
                        fee_cost = exit_price * self.fee_rate
                        realized_pnl = (exit_price - entry_price) if position > 0 \
                            else (entry_price - exit_price)
                        trades.append({
                            "symbol": symbol,
                            "action": "TAKE_PROFIT",
                            "index": i,
                            "timestamp": row.get("timestamp", i),
                            "price": exit_price,
                            "fee": fee_cost,
                            "pnl": realized_pnl
                        })
                        position = 0
                        exit_trade = True
                        i += 1
                        break

                    # max_hold_bars
                    if bar_held >= self.max_hold_bars:
                        exit_price = (price * (1 - self.slippage_rate)
                                      if position > 0
                                      else price * (1 + self.slippage_rate))
                        fee_cost = exit_price * self.fee_rate
                        realized_pnl = (exit_price - entry_price) if position > 0 \
                            else (entry_price - exit_price)
                        trades.append({
                            "symbol": symbol,
                            "action": "TIME_EXIT",
                            "index": i,
                            "timestamp": row.get("timestamp", i),
                            "price": exit_price,
                            "fee": fee_cost,
                            "pnl": realized_pnl
                        })
                        position = 0
                        exit_trade = True
                        i += 1
                        break

                    # otherwise => hold
                    bar_held += 1
                    i += 1

        return pd.DataFrame(trades)

    def _predict_prob(self, row, row_df: pd.DataFrame) -> float:
        """
        If we have self.model with predict_proba => we do that,
        else if predict_function => call it,
        else => 0.5
        """
        if self.model is not None:
            # model approach
            if hasattr(self.model, "predict_proba"):
                # row_df must match model features. If row_df is a single row,
                # shape => (1, N). We'll do the best we can. We'll assume
                # columns are aligned or you'll adapt. We do row_df => predict_proba
                proba = self.model.predict_proba(row_df)[0]
                if len(proba) > 1:
                    return proba[1]  # probability of class=1
                else:
                    # maybe it's binary or single
                    return float(self.model.classes_[0] == 1)
            else:
                # fallback => .predict => 0 or 1
                label = self.model.predict(row_df)[0]
                return 1.0 if label == 1 else 0.0

        elif self.predict_function is not None:
            return float(self.predict_function(row))
        else:
            return 0.5

    def _compute_advanced_stats(self, df_trades: pd.DataFrame) -> Dict[str, Any]:
        """
        Summarizes the trades into a dictionary of advanced stats:
         - total_pnl
         - max_drawdown
         - sharpe
         - sortino
         - daily returns approach if track_daily_returns
        """
        if df_trades.empty:
            return {}

        # We'll unify the 'pnl' field for partial & full. partial => 'pnl'
        df_trades["pnl"] = df_trades["pnl"].fillna(0)
        # For open actions => "pnl" won't exist => fill 0
        df_trades["pnl_cum"] = df_trades["pnl"].cumsum()

        # track max drawdown
        peak = -999999
        max_dd = 0.0
        for val in df_trades["pnl_cum"]:
            if val > peak:
                peak = val
            dd = peak - val
            if dd > max_dd:
                max_dd = dd

        final_pnl = df_trades["pnl"].sum()
        # We'll treat "pnl" as absolute, not percentage. If you want percentage, adapt.

        # approximate returns for sharpe
        returns = df_trades["pnl"].copy()
        if returns.std() == 0:
            sharpe = 0.0
        else:
            sharpe = returns.mean() / returns.std()

        # sortino => divide average positive returns by std of negative returns
        negative_returns = returns[returns < 0]
        if negative_returns.std() == 0:
            sortino = float("inf") if returns.mean() > 0 else 0.0
        else:
            sortino = returns.mean() / negative_returns.std()

        stats = {
            "total_pnl": final_pnl,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "sortino": sortino,
            "num_trades": len(df_trades[df_trades["action"].str.contains("OPEN")])
        }

        # Optionally track daily returns if we have timestamp in seconds or daily
        if self.track_daily_returns and "timestamp" in df_trades.columns:
            # Build daily
            # We'll create a "date" column from "timestamp"
            # Then sum "pnl" per date
            df_trades["dt_date"] = pd.to_datetime(df_trades["timestamp"], unit="s").dt.date
            daily_pnl = df_trades.groupby("dt_date")["pnl"].sum()
            daily_returns = daily_pnl.cumsum()  # cumulative
            # we can store them in stats if desired
            stats["daily_returns"] = daily_returns

        logger.info(
            f"Backtest Stats => total_pnl={final_pnl:.2f}, max_dd={max_dd:.2f}, "
            f"sharpe={sharpe:.2f}, sortino={sortino:.2f}, trades={stats['num_trades']}"
        )
        return stats


# ---------------------------------------------------------------------
# Example main
# ---------------------------------------------------------------------
def main():
    # Minimal usage example
    # We'll create a dummy dataset with random walk
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    N = 1000
    price = np.cumprod(1 + 0.001 * np.random.randn(N)) * 100
    df = pd.DataFrame({"feature_price": price})
    df["timestamp"] = np.arange(N) * 60  # pretend each bar is 1 minute
    df["feature_price"] = df["feature_price"].abs()  # ensure positive

    # Example predict function => random
    def random_predict(row):
        return np.random.rand()  # uniform 0..1 => about 50% up

    bt = Backtester(
        data=df,
        predict_function=random_predict,
        buy_threshold=0.55,
        sell_threshold=0.45,
        stop_loss_pct=0.03,
        take_profit_pct=0.05,
        max_hold_bars=30,
        trailing_stop=True,
        partial_take_pct=0.5,
        partial_exit_trigger=0.5,
        slippage_rate=0.0005,
        fee_rate=0.001,
        track_daily_returns=True
    )

    trades, stats = bt.run_backtest()
    print("Final stats =>", stats)
    if not trades.empty:
        print(trades.tail(10))


if __name__ == "__main__":
    main()
