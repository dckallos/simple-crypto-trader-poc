# ==============================================================================
# FILE: ai_strategy.py
# ==============================================================================
"""
ai_strategy.py

A production-ready AIStrategy that:
1. Loads a trained model if model_path is provided.
2. Replicates advanced indicators for real-time inference (RSI, MACD, Bollinger, etc.).
3. Optionally merges data from:
   - CryptoPanic (db table 'cryptopanic_news' or aggregated daily sentiment).
   - LunarCrush (db table 'lunarcrush_data' with galaxy_score, alt_rank, etc.).
4. Falls back to dummy logic or manual normalization if needed.

NEW/HYBRID UPDATES:
- We assume that 'market_data' might now contain an aggregator summary for the
  last 5 or 10 minutes (e.g. avg_price, total_volume, etc.), and we can incorporate
  that into GPT prompting or advanced scikit inference.
- GPT prompts can reference "aggregated data" instead of a single snapshot price.
- We maintain references to a RiskManager for forced exits (stop-loss, take-profit),
  but the main AI logic decides strategic BUY/SELL/HOLD on aggregator intervals.

All existing code is retained, only appended or revised lightly for aggregator usage.
"""

import logging
import sqlite3
import numpy as np
import os
import joblib
import pandas as pd
import time
from dotenv import load_dotenv
import json

# The official new OpenAI Python library client
from openai import OpenAI

from db import store_price_history, DB_FILE
from risk_manager import RiskManager  # Import our enhanced risk manager

logger = logging.getLogger(__name__)

BTC_PAIR = "XBT/USD"

# The exact columns we used in train_model.py, if you want to replicate them:
TRAIN_FEATURE_COLS = [
    "feature_price", "feature_ma_3", "feature_spread",
    "vol_change", "rsi", "macd_line", "macd_signal",
    "boll_upper", "boll_lower",
    "corr_with_btc",  # only if you'd used correlation with BTC
    "avg_sentiment"   # from CryptoPanic aggregator if you used it in training
]


class AIStrategy:
    """
    AIStrategy class that decides whether to BUY, SELL, or HOLD based on
    AI-driven signals. It can also feed CryptoPanic or LunarCrush data
    into the real-time inference logic or GPT-based logic, if available.

    We now assume 'market_data' might contain aggregator fields like:
    {
      "avg_price": 1234.56,
      "min_price": 1200.00,
      "max_price": 1250.00,
      "total_volume": 4567.89,
      "pair": "ETH/USD",
      "timestamp": 1681234567.0,
      ...
    }
    plus any single-snapshot fields if needed.
    """

    def __init__(self, pairs=None, model_path=None, use_openai=False):
        """
        :param pairs: List of trading pairs, e.g. ["XBT/USD", "ETH/USD"].
        :param model_path: path to a .pkl file with a trained model, if any.
        :param use_openai: bool flag. If True, we'll attempt an OpenAI-based inference.
        """
        self.pairs = pairs if pairs else [BTC_PAIR]
        self.model = None
        self.use_openai = use_openai

        # Attempt to load a scikit model
        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                logger.info(f"AIStrategy: Model loaded from '{model_path}'")
            except Exception as e:
                logger.exception(f"AIStrategy: Error loading model: {e}")
        else:
            if model_path:
                logger.warning(f"AIStrategy: No model file found at '{model_path}'. Using fallback.")
            else:
                logger.info("AIStrategy: No model_path provided. Using fallback logic.")

        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY", "FAKE_OPENAI_KEY")
        self.client = OpenAI(api_key=openai_api_key)
        logger.info("OpenAI client instantiated in AIStrategy.")

        # Simple in-memory position tracking: { "ETH/USD": 0.01, ... }
        self.current_positions = {}

        # Track the "entry price" for each pair if we hold a position
        # so we can compute unrealized gains/losses.
        self.entry_prices = {}  # e.g. { "ETH/USD": 1500.0, ... }

        # NEW: Integrate an enhanced RiskManager with optional stop-loss, take-profit, etc.
        # Set parameters as you see fit. Example:
        self.risk_manager = RiskManager(
            max_position_size=0.001,
            stop_loss_pct=0.05,      # 5% stop-loss
            take_profit_pct=0.10,    # 10% take-profit
            max_daily_drawdown=-0.02 # -2% daily drawdown limit
        )

    # --------------------------------------------------------------------------
    # Main predict() entry point
    # --------------------------------------------------------------------------
    def predict(self, market_data: dict):
        """
        Decide BUY, SELL, or HOLD. If model is loaded & data is complete, we
        do scikit-based inference with advanced indicators + optional data from
        CryptoPanic or LunarCrush. If 'use_openai' is True, try GPT function-calling.

        :param market_data: e.g. {
            "avg_price": float,
            "min_price": float,
            "max_price": float,
            "total_volume": float,
            "pair": "ETH/USD",
            "timestamp": float,
            "current_position": float,  # optional
            ...
          }
        :return: (signal, size)
        """
        pair = market_data.get("pair", self.pairs[-1])

        # 1) GPT approach if user wants
        if self.use_openai:
            try:
                return self._openai_inference(market_data)
            except Exception as e:
                logger.exception(f"OpenAI inference failed: {e}")

        # 2) scikit-based inference if we have a model
        if self.model:
            try:
                return self._model_inference_realtime(market_data)
            except Exception as e:
                logger.exception(f"Error in scikit inference: {e}")

        # 3) fallback
        return self._dummy_logic(market_data)

    # --------------------------------------------------------------------------
    # GPT-based function-calling approach
    # --------------------------------------------------------------------------
    def _openai_inference(self, market_data: dict):
        """
        Example GPT usage with function calling. We incorporate aggregator data if present.
        We combine:
          - short summary of recent trades
          - aggregator data (avg/min/max price, volume) for last X minutes
          - CryptoPanic daily sentiment
          - LunarCrush galaxy_score or alt_rank if you wish

        Then we let GPT produce a final structured "action" + "size."
        """

        pair = market_data.get("pair", self.pairs[-1])
        # aggregator fields might be "avg_price", "total_volume", etc.
        avg_price = market_data.get("avg_price", 0.0)
        min_price = market_data.get("min_price", 0.0)
        max_price = market_data.get("max_price", 0.0)
        total_volume = market_data.get("total_volume", 0.0)

        current_position = market_data.get("current_position", 0.0)
        # if you also pass a single snapshot "price", you can do:
        snapshot_price = market_data.get("price", avg_price)

        # Summaries from DB
        trade_summary = self._summarize_recent_trades(pair, limit=3)
        cryptopanic_sent = self._fetch_cryptopanic_sentiment_today()
        symbol = pair.split("/")[0].upper()
        lunarcrush_metrics = self._fetch_lunarcrush_metrics(symbol)

        # Build function signature
        functions = [
            {
                "name": "trade_decision",
                "description": "Return a trade decision in structured JSON",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["BUY", "SELL", "HOLD"]
                        },
                        "size": {
                            "type": "number",
                            "description": "Amount of the asset to trade"
                        }
                    },
                    "required": ["action", "size"]
                }
            }
        ]

        system_message = {
            "role": "system",
            "content": (
                "You are an advanced crypto trading assistant. The user provides aggregator data "
                "from the last 5-10 minutes plus recent trade info. Use it to decide BUY, SELL, "
                "or HOLD, returning a structured JSON with action + size."
            )
        }

        # aggregator mention
        aggregator_text = (
            f"Aggregator data for {pair} over last period:\n"
            f"  avg_price={avg_price}, min_price={min_price}, max_price={max_price}, volume={total_volume}\n"
        )

        user_prompt = (
            f"{aggregator_text}\n"
            f"Current open position: {current_position}\n"
            f"Recent trades for {pair}:\n{trade_summary}\n\n"
            f"CryptoPanic daily sentiment: {cryptopanic_sent:.2f}\n"
        )
        if lunarcrush_metrics:
            user_prompt += (
                f"LunarCrush data for {symbol}:\n"
                f"  Galaxy Score: {lunarcrush_metrics.get('galaxy_score')}\n"
                f"  AltRank: {lunarcrush_metrics.get('alt_rank')}\n"
                f"  24h Volume: {lunarcrush_metrics.get('volume_24h')}\n"
                f"  MarketCap: {lunarcrush_metrics.get('market_cap')}\n"
                f"  24h Social Volume: {lunarcrush_metrics.get('social_volume_24h')}\n\n"
            )
        user_prompt += (
            "Please suggest a trade decision (BUY, SELL, or HOLD) plus size in a function call.\n"
        )

        user_message = {"role": "user", "content": user_prompt}

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[system_message, user_message],
            functions=functions,
            function_call="auto",
            temperature=0.0,
            max_tokens=500
        )
        import json
        print(f'response: {response}')

        choice = response.choices[0]
        finish_reason = choice.finish_reason

        if finish_reason == "function_call":
            fn_call = choice.message.function_call
            fn_name = fn_call.name
            if fn_name == "trade_decision":
                args = json.loads(fn_call.arguments)
                action = args.get("action", "HOLD")
                size = args.get("size", 0.0)
                # Risk management clamp
                action, size = self.risk_manager.adjust_trade(action, size)
                # Update position tracking if actually buying or selling
                self._update_position(pair, action, size, snapshot_price)
                return (action, size)
            else:
                logger.warning(f"Unknown function call: {fn_name}")
                return self._dummy_logic(market_data)
        else:
            # fallback if GPT didn't do function call
            raw_text = choice.message.content.upper() if choice.message.content else ""
            if "BUY" in raw_text:
                action, size = self.risk_manager.adjust_trade("BUY", 0.0005)
                self._update_position(pair, action, size, snapshot_price)
                return (action, size)
            elif "SELL" in raw_text:
                action, size = self.risk_manager.adjust_trade("SELL", 0.0005)
                self._update_position(pair, action, size, snapshot_price)
                return (action, size)
            else:
                return ("HOLD", 0.0)

    # --------------------------------------------------------------------------
    # Real-time scikit inference with advanced indicators + optional data
    # --------------------------------------------------------------------------
    def _model_inference_realtime(self, market_data: dict):
        """
        1. Load the last ~50 rows from DB to compute rolling RSI, MACD, Bollinger, etc.
        2. Merge correlation with BTC if pair != BTC itself.
        3. Merge CryptoPanic aggregator daily sentiment if needed.
        4. Merge LunarCrush metrics if needed.
        5. Probability-based approach: if prob_up > 0.6 => BUY, if prob_up < 0.4 => SELL, else HOLD.
        6. If we have a position => check risk manager or forcibly exit if needed.
        7. Return final (action, size).
        """
        pair = market_data.get("pair", self.pairs[-1])
        latest_price = market_data.get("price", 0.0)  # aggregator might have only avg_price
        if latest_price == 0.0:
            # fall back to aggregator's avg_price if single snapshot is missing
            latest_price = market_data.get("avg_price", 0.0)

        # 1) fetch recent price data for the primary coin
        recent_df = self._fetch_recent_price_data(pair, limit=50)
        if recent_df.empty:
            logger.warning(f"No recent data for {pair}; fallback to dummy.")
            return self._dummy_logic(market_data)

        # 2) fetch BTC for correlation if needed
        if pair != BTC_PAIR:
            btc_df = self._fetch_recent_price_data(BTC_PAIR, limit=50)
        else:
            btc_df = None

        # 3) compute indicators
        df_with_ind = self._compute_indicators(recent_df, btc_df)

        # 4) Merge aggregator daily sentiment from CryptoPanic if used
        daily_sent = self._fetch_cryptopanic_sentiment_today()
        df_with_ind["avg_sentiment"] = daily_sent

        # We'll take the last row
        latest_row = df_with_ind.iloc[[-1]].copy(deep=True)

        # Ensure all columns from TRAIN_FEATURE_COLS exist
        for col in TRAIN_FEATURE_COLS:
            if col not in latest_row.columns:
                latest_row[col] = 0.0

        X_input = latest_row[TRAIN_FEATURE_COLS]

        # predict_proba => probability of "up"
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_input)[0]
            prob_up = probs[1]
        else:
            pred_label = self.model.predict(X_input)[0]
            prob_up = 1.0 if pred_label == 1 else 0.0

        current_pos = self.current_positions.get(pair, 0.0)
        if current_pos != 0.0:
            entry_price = self.entry_prices.get(pair, latest_price)
            # Check forced exit
            should_close, reason = self.risk_manager.check_take_profit(
                current_price=latest_price,
                entry_price=entry_price,
                current_position=current_pos
            )
            if should_close:
                final_action, final_size = self.risk_manager.adjust_trade("SELL", abs(current_pos))
                self._update_position(pair, final_action, final_size, latest_price)
                logger.info(f"Closing position for {pair} due to risk manager => {reason}")
                return (final_action, final_size)

        action = "HOLD"
        size_suggested = 0.0

        # Simple threshold logic
        if current_pos > 0:
            # If we already hold a long
            if prob_up < 0.4:
                action = "SELL"
                size_suggested = current_pos
        else:
            # no position
            if prob_up > 0.6:
                action = "BUY"
                size_suggested = 0.0005
            elif prob_up < 0.4:
                # optional short
                action = "SELL"
                size_suggested = 0.0005
            else:
                action = "HOLD"

        final_action, final_size = self.risk_manager.adjust_trade(action, size_suggested)
        self._update_position(pair, final_action, final_size, latest_price)
        return (final_action, final_size)

    # --------------------------------------------------------------------------
    # Helper: compute RSI, MACD, Bollinger, correlation from a DF
    # (unchanged from your snippet)
    # --------------------------------------------------------------------------
    def _compute_indicators(self, df: pd.DataFrame, df_btc: pd.DataFrame = None):
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Basic features
        df["feature_price"] = df["last_price"]
        df["feature_ma_3"] = df["last_price"].rolling(3).mean()
        df["feature_spread"] = df["ask_price"] - df["bid_price"]
        df["vol_change"] = df["volume"].pct_change().fillna(0)

        # RSI
        window_length = 14
        close_delta = df["last_price"].diff()
        gain = close_delta.clip(lower=0)
        loss = (-1 * close_delta.clip(upper=0))
        avg_gain = gain.rolling(window=window_length, min_periods=window_length).mean()
        avg_loss = loss.rolling(window=window_length, min_periods=window_length).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df["last_price"].ewm(span=12).mean()
        ema26 = df["last_price"].ewm(span=26).mean()
        df["macd_line"] = ema12 - ema26
        df["macd_signal"] = df["macd_line"].ewm(span=9).mean()

        # Bollinger
        sma20 = df["last_price"].rolling(window=20).mean()
        std20 = df["last_price"].rolling(window=20).std()
        df["boll_upper"] = sma20 + (2 * std20)
        df["boll_lower"] = sma20 - (2 * std20)

        if df_btc is not None and not df_btc.empty:
            df_btc = df_btc.sort_values("timestamp").reset_index(drop=True)
            df_btc_ren = df_btc[["timestamp", "last_price"]].rename(columns={"last_price": "btc_price"})
            merged = pd.merge_asof(
                df, df_btc_ren, on="timestamp", direction="nearest", tolerance=30
            )
            merged["corr_with_btc"] = merged["last_price"].rolling(30).corr(merged["btc_price"])
            df = merged

        df.ffill(inplace=True)
        df.bfill(inplace=True)
        return df

    # --------------------------------------------------------------------------
    # Summarize recent trades for GPT
    # (unchanged from your snippet)
    # --------------------------------------------------------------------------
    def _summarize_recent_trades(self, pair: str, limit=3) -> str:
        conn = sqlite3.connect(DB_FILE)
        summary_lines = []
        try:
            c = conn.cursor()
            c.execute("""
                SELECT timestamp, side, quantity, price
                FROM trades
                WHERE pair=?
                ORDER BY id DESC
                LIMIT ?
            """, (pair, limit))
            rows = c.fetchall()
            if not rows:
                return "No trades found."

            for i, row in enumerate(rows[::-1], start=1):
                tstamp, side, qty, price = row
                import datetime
                dt_str = datetime.datetime.fromtimestamp(tstamp).strftime("%Y-%m-%d %H:%M")
                summary_lines.append(f"{i}) {dt_str} {side} {qty} @ ${price}")
        except Exception as e:
            logger.exception(f"Error summarizing recent trades: {e}")
            return "Error retrieving trades."
        finally:
            conn.close()

        return "\n".join(summary_lines)

    # --------------------------------------------------------------------------
    # CryptoPanic aggregator sentiment
    # (unchanged from your snippet)
    # --------------------------------------------------------------------------
    def _fetch_cryptopanic_sentiment_today(self) -> float:
        import datetime
        today_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        conn = sqlite3.connect(DB_FILE)
        try:
            c = conn.cursor()
            query = f"""
                SELECT avg(sentiment_score)
                FROM cryptopanic_news
                WHERE DATE(timestamp, 'unixepoch') = '{today_str}'
            """
            row = c.execute(query).fetchone()
            if row and row[0] is not None:
                return float(row[0])
            return 0.0
        except Exception as e:
            logger.exception(f"Error fetching today's CryptoPanic sentiment: {e}")
            return 0.0
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # LunarCrush aggregator
    # (unchanged from your snippet)
    # --------------------------------------------------------------------------
    def _fetch_lunarcrush_metrics(self, symbol: str) -> dict:
        conn = sqlite3.connect(DB_FILE)
        try:
            c = conn.cursor()
            c.execute("""
                SELECT price,
                       market_cap,
                       volume_24h,
                       volatility,
                       percent_change_1h,
                       percent_change_24h,
                       percent_change_7d,
                       percent_change_30d,
                       social_volume_24h,
                       interactions_24h,
                       social_dominance,
                       galaxy_score,
                       alt_rank,
                       sentiment,
                       categories,
                       topic,
                       logo
                FROM lunarcrush_data
                WHERE symbol=?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (symbol,))
            row = c.fetchone()
            if not row:
                return {}
            fields = [
                "price", "market_cap", "volume_24h", "volatility",
                "percent_change_1h", "percent_change_24h", "percent_change_7d", "percent_change_30d",
                "social_volume_24h", "interactions_24h", "social_dominance",
                "galaxy_score", "alt_rank", "sentiment", "categories", "topic", "logo"
            ]
            data_dict = {k: v for k, v in zip(fields, row)}
            return data_dict
        except Exception as e:
            logger.exception(f"Error fetching LunarCrush data for symbol={symbol}: {e}")
            return {}
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # Helper: fetch recent price data from 'price_history'
    # (unchanged from your snippet)
    # --------------------------------------------------------------------------
    def _fetch_recent_price_data(self, pair: str, limit=50):
        conn = sqlite3.connect(DB_FILE)
        try:
            query = f"""
                SELECT timestamp, pair, bid_price, ask_price, last_price, volume
                FROM price_history
                WHERE pair='{pair}'
                ORDER BY id DESC
                LIMIT {limit}
            """
            df = pd.read_sql_query(query, conn)
            return df[::-1].reset_index(drop=True)
        except Exception as e:
            logger.exception(f"Error loading recent data for {pair}: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # Dummy fallback
    # --------------------------------------------------------------------------
    def _dummy_logic(self, market_data: dict):
        """
        A fallback strategy that just checks if price < 20000 => BUY else HOLD,
        ignoring advanced indicators.
        """
        price = market_data.get("price", 0.0)
        # fallback if aggregator only has avg_price
        if price == 0.0:
            price = market_data.get("avg_price", 0.0)
        pair = market_data.get("pair", self.pairs[-1])
        if price < 20000:
            action, size = self.risk_manager.adjust_trade("BUY", 0.0005)
            self._update_position(pair, action, size, price)
            return (action, size)
        else:
            return ("HOLD", 0.0)

    # --------------------------------------------------------------------------
    # Updated: Position Update Helper
    # --------------------------------------------------------------------------
    def _update_position(self, pair: str, action: str, trade_size: float, trade_price: float):
        """
        Updates the in-memory position tracking based on the action and trade size.
        If 'BUY', we increase the position. If 'SELL', we decrease or close it.
        Also tracks entry price for new positions, and logs realized PnL on closes.
        """
        if trade_size <= 0:
            return

        current_pos = self.current_positions.get(pair, 0.0)
        old_pos = current_pos

        if action == "BUY":
            new_pos = current_pos + trade_size

            # If we were previously flat (0) and are now opening a new long,
            # set entry_price to the trade_price.
            if old_pos == 0:
                self.entry_prices[pair] = trade_price
            # If we were partially in a position, you might want a more
            # sophisticated approach to recalc average cost basis.
            # For simplicity, we won't do that here.

            self.current_positions[pair] = new_pos
            logger.debug(f"Updated position for {pair}: was {old_pos}, now {new_pos} (BUY {trade_size} @ {trade_price})")

        elif action == "SELL":
            # If we had a long, we reduce it:
            new_pos = current_pos - trade_size
            self.current_positions[pair] = new_pos
            logger.debug(f"Updated position for {pair}: was {old_pos}, now {new_pos} (SELL {trade_size} @ {trade_price})")

            # If we've gone from a positive position to zero or negative, that means
            # we closed or partially closed a long. Let's assume a full close if new_pos <= 0.
            if old_pos > 0 and new_pos <= 0:
                # Realized PnL:
                entry_price = self.entry_prices.get(pair, trade_price)
                # for a long, realized is (exit - entry) * size. We'll do partial logic for entire position.
                # If we are fully closing the position:
                closed_size = old_pos if new_pos <= 0 else (old_pos - new_pos)
                realized_pnl = (trade_price - entry_price) * closed_size
                logger.info(f"Closed LONG for {pair} with realized PnL: {realized_pnl:.4f}")
                # We can record it in the risk_manager if we want
                self.risk_manager.record_trade_pnl(realized_pnl)

                # Reset entry price if fully closed
                if new_pos <= 0:
                    self.entry_prices[pair] = 0.0

            # If shorting, you'd want a new 'entry_price' as well.
            # We'll keep it simple for now.

        else:
            # HOLD or anything else doesn't change the position
            pass
