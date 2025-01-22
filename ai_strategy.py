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

UPDATED CHANGES FOR RISKMANAGER:
- Initialize RiskManager with optional stop_loss_pct, take_profit_pct, and daily drawdown limit.
- Track entry prices in a dict so we can do check_take_profit in real-time.
- If a take-profit or stop-loss triggers, we SELL the entire position.
- Example daily PnL updates on closes.

NOTE: In real usage, you'd refine how PnL is calculated and handle partial closes or shorting logic carefully.
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

        :param market_data: e.g. {"price": float, "timestamp": float, "pair": "ETH/USD", ...}
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
        Example GPT usage with function calling. We combine:
          - short summary of recent trades
          - CryptoPanic daily sentiment
          - LunarCrush galaxy_score or alt_rank if you wish

        Then we let GPT produce a final structured "action" + "size."
        """

        pair = market_data.get("pair", self.pairs[-1])
        current_price = market_data.get("price", 0.0)

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
            "content": "You are an advanced crypto trading assistant, using the user's data for decisions."
        }

        user_prompt = (
            f"Recent trades for {pair}:\n{trade_summary}\n\n"
            f"Current price: {current_price}\n"
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
            "Please suggest a trade decision (BUY, SELL, or HOLD) plus size in a function call."
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
                import json
                args = json.loads(fn_call.arguments)
                action = args.get("action", "HOLD")
                size = args.get("size", 0.0)
                # Risk management clamp
                action, size = self.risk_manager.adjust_trade(action, size)
                # Update position tracking if actually buying or selling
                self._update_position(pair, action, size, current_price)
                return (action, size)
            else:
                logger.warning(f"Unknown function call: {fn_name}")
                return self._dummy_logic(market_data)
        else:
            # fallback if GPT didn't do function call
            raw_text = choice.message.content.upper() if choice.message.content else ""
            if "BUY" in raw_text:
                action, size = self.risk_manager.adjust_trade("BUY", 0.0005)
                self._update_position(pair, action, size, current_price)
                return (action, size)
            elif "SELL" in raw_text:
                action, size = self.risk_manager.adjust_trade("SELL", 0.0005)
                self._update_position(pair, action, size, current_price)
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
        6. Check if we have a position => might apply stop-loss/take-profit logic from RiskManager.
        """
        pair = market_data.get("pair", self.pairs[-1])
        latest_price = market_data.get("price", 0.0)  # we'll pass to risk manager checks

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

        # 4) Merge aggregator daily sentiment from CryptoPanic if you used "avg_sentiment" in your model
        daily_sent = self._fetch_cryptopanic_sentiment_today()
        df_with_ind["avg_sentiment"] = daily_sent

        # We'll take the last row
        latest_row = df_with_ind.iloc[[-1]].copy(deep=True)

        # Ensure all columns from TRAIN_FEATURE_COLS exist
        for col in TRAIN_FEATURE_COLS:
            if col not in latest_row.columns:
                latest_row[col] = 0.0  # fallback for missing data

        X_input = latest_row[TRAIN_FEATURE_COLS]

        # ---- Use predict_proba for probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_input)[0]
            prob_up = probs[1]
        else:
            # Fallback if model doesn't support predict_proba
            pred_label = self.model.predict(X_input)[0]
            prob_up = 1.0 if pred_label == 1 else 0.0

        # Before we decide new trades, let's see if we have a position and need to forcibly close
        current_pos = self.current_positions.get(pair, 0.0)
        if current_pos != 0.0:
            entry_price = self.entry_prices.get(pair, latest_price)
            # Check take-profit / stop-loss, etc.
            should_close, reason = self.risk_manager.check_take_profit(
                current_price=latest_price,
                entry_price=entry_price,
                current_position=current_pos
            )
            if should_close:
                # Force a SELL of the entire position
                final_action, final_size = self.risk_manager.adjust_trade("SELL", abs(current_pos))
                self._update_position(pair, final_action, final_size, latest_price)
                logger.info(f"Closing position for {pair} due to risk manager => {reason}")
                return (final_action, final_size)

        # Now we do normal logic
        action = "HOLD"
        size_suggested = 0.0

        if current_pos > 0:
            # Already in a long, check if we want to hold or close
            if prob_up < 0.4:
                action = "SELL"
                size_suggested = current_pos  # close
            else:
                action = "HOLD"
                size_suggested = 0.0
        else:
            # We have no position => maybe open a new long or short
            if prob_up > 0.6:
                action = "BUY"
                size_suggested = 0.0005
            elif prob_up < 0.4:
                # If you want to short, uncomment:
                action = "SELL"
                size_suggested = 0.0005
            else:
                action = "HOLD"
                size_suggested = 0.0

        # Apply risk manager clamp
        final_action, final_size = self.risk_manager.adjust_trade(action, size_suggested)

        # Update our position tracking
        self._update_position(pair, final_action, final_size, latest_price)

        return (final_action, final_size)

    # --------------------------------------------------------------------------
    # Helper: compute RSI, MACD, Bollinger, correlation from a DF (unchanged)
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
    # Summarize recent trades for GPT (unchanged)
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
    # CryptoPanic aggregator sentiment (unchanged)
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
    # LunarCrush aggregator (unchanged)
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
    # Helper: fetch recent price data from 'price_history' (unchanged)
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
                # for a long, realized is (exit - entry) * size. We'll do partial logic for the entire position.
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
