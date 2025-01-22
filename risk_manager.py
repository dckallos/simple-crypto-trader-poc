# =============================
# risk_manager.py
# =============================
"""
risk_manager.py

Encapsulates risk management logic, including position sizing, take-profit
rules, or other constraints.

ENHANCEMENTS ADDED:
1) Optional Stop-Loss and Take-Profit thresholds in constructor.
2) Additional logic in `adjust_trade` to handle open position size and daily drawdown.
3) Example volatility-based size clamp method (optional).
4) Example 'check_take_profit' implementation that closes positions if a threshold is exceeded.

Note:
- If you store or track open positions (entry_price, size, etc.) externally,
  you can pass them into these methods. This file can remain stateless or you
  can store state in here as well—depends on your architecture.
"""

import logging

logger = logging.getLogger(__name__)

class RiskManager:
    """
    RiskManager helps ensure we don't exceed certain constraints,
    and can implement take-profit or stop-loss logic.
    """

    def __init__(
        self,
        max_position_size: float,
        stop_loss_pct: float = None,
        take_profit_pct: float = None,
        max_daily_drawdown: float = None
    ):
        """
        :param max_position_size: The maximum allowed position size per trade.
        :param stop_loss_pct: Optional stop-loss percent (e.g. 0.05 for 5%).
        :param take_profit_pct: Optional take-profit percent (e.g. 0.10 for 10%).
        :param max_daily_drawdown: Optional daily PnL-based stop. If your net PnL
                                   for the day is below e.g. -0.02 ( -2% ),
                                   you might stop trading or reduce risk.
        """
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_daily_drawdown = max_daily_drawdown

        # Example placeholders if you want to track daily PnL or open positions here.
        # This can also be done in the AIStrategy or DB logic.
        self.daily_realized_pnl = 0.0
        self.current_day = None  # track date if you want daily resets

    def adjust_trade(self, signal: str, suggested_size: float):
        """
        Adjust the AI's suggested trade for risk constraints:
          1) clamp trade size to max_position_size
          2) if daily drawdown limit is hit, forcibly HOLD
          3) optionally transform a BUY or SELL if conditions are too risky
        :param signal: "BUY", "SELL", or "HOLD"
        :param suggested_size: float, how many units the AI suggests
        :return: (final_signal, final_size)
        """
        # --------------------------------------------
        # 1) daily drawdown check
        #    If your daily PnL is too negative, skip new trades:
        # --------------------------------------------
        if self.max_daily_drawdown is not None:
            if self.daily_realized_pnl <= self.max_daily_drawdown:
                # Force a HOLD if daily losses are too big
                logger.warning(
                    f"Daily drawdown limit reached ({self.daily_realized_pnl:.2f}), forcing HOLD."
                )
                return ("HOLD", 0.0)

        # --------------------------------------------
        # 2) clamp trade size to max_position_size
        # --------------------------------------------
        if signal in ("BUY", "SELL"):
            final_size = min(suggested_size, self.max_position_size)
            return (signal, final_size)
        else:
            return ("HOLD", 0.0)

    def check_take_profit(
        self,
        current_price: float,
        entry_price: float,
        current_position: float
    ):
        """
        Checks if we should close a profitable (or losing) position
        based on optional thresholds. This is an example approach:
          - if take_profit_pct is set and gain >= that threshold -> SELL
          - if stop_loss_pct is set and loss <= that threshold -> SELL
        This returns a tuple (should_sell: bool, reason: str).

        :param current_price: float, the latest market price
        :param entry_price: float, the price at which the position was opened
        :param current_position: how many units we hold (>0 = long, <0 = short)
        :return: (should_sell, reason)
        """
        if current_position == 0:
            return (False, "No position to exit.")

        # Gains or losses are computed differently for long vs short
        unrealized_pct = 0.0
        if current_position > 0:  # long
            # e.g. (current_price - entry_price) / entry_price
            unrealized_pct = (current_price - entry_price) / (entry_price + 1e-9)
        else:  # short
            # e.g. (entry_price - current_price) / entry_price
            unrealized_pct = (entry_price - current_price) / (entry_price + 1e-9)

        # Check take_profit
        if self.take_profit_pct is not None:
            if unrealized_pct >= self.take_profit_pct:
                return (True, f"Take-profit triggered at {unrealized_pct:.2%}")

        # Check stop_loss
        if self.stop_loss_pct is not None:
            # for a long, if unrealized_pct <= -stop_loss_pct => big negative
            # for a short, if unrealized_pct <= -stop_loss_pct => also big negative
            if unrealized_pct <= -abs(self.stop_loss_pct):
                return (True, f"Stop-loss triggered at {unrealized_pct:.2%}")

        return (False, f"Unrealized Pct={unrealized_pct:.2%}")

    # --------------------------------------------------------------------------
    # OPTIONAL EXAMPLE: volatility-based sizing
    # --------------------------------------------------------------------------
    def adjust_for_volatility(self, pair_volatility: float, base_size: float):
        """
        Suppose you want smaller positions in high-volatility environments:
          size = base_size * (some function of volatility).
        Example: if pair_volatility is standard deviation or ATR,
        you might do 1 / volatility scaling.

        :param pair_volatility: e.g. a daily vol or standard dev from price data
        :param base_size: your normal trade size
        :return: volatility-adjusted size
        """
        # This is just a naive example
        if pair_volatility <= 0:
            return base_size  # can't compute ratio

        # e.g. scale by 1 / vol, then clamp
        scaled_size = base_size / pair_volatility

        # clamp final
        final_size = min(scaled_size, self.max_position_size)
        return final_size

    # --------------------------------------------------------------------------
    # OPTIONAL: track daily PnL or number of trades
    # --------------------------------------------------------------------------
    def record_trade_pnl(self, pnl_amount: float):
        """
        Adds realized PnL to the daily tally. If you want a daily reset,
        track the date and reset each morning.
        """
        self.daily_realized_pnl += pnl_amount
        logger.info(f"Recorded PnL: {pnl_amount:.2f}. Daily total: {self.daily_realized_pnl:.2f}")

    def reset_daily_pnl(self, new_day_str: str):
        """
        If you want to reset daily tracking at midnight or next day.
        """
        if self.current_day != new_day_str:
            logger.info(f"Resetting daily PnL. Previous day = {self.current_day}, new day = {new_day_str}")
            self.current_day = new_day_str
            self.daily_realized_pnl = 0.0
