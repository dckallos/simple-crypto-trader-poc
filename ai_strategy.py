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

FIXES/UPDATES:
- Increase max_tokens in GPT calls to avoid "max_tokens reached" error.
- Use the same TRAIN_FEATURE_COLS as training to avoid feature mismatch.
- Add synergy with CryptoPanic and LunarCrush data for advanced AI decisions.
"""

import logging
import sqlite3
import numpy as np
import os
import joblib
import pandas as pd
import time
from dotenv import load_dotenv

# The official new OpenAI Python library client
from openai import OpenAI

logger = logging.getLogger(__name__)

DB_FILE = "trades.db"
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
        cryptopanic_sent = self._fetch_cryptopanic_sentiment_today()  # average from aggregator
        # If your pair is e.g. "ETH/USD", symbol might be "ETH"
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
            model="gpt-4o",
            messages=[system_message, user_message],
            functions=functions,
            function_call="auto",
            temperature=0.0,
            max_tokens=500  # to avoid 'max tokens' error
        )

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
                return (action, size)
            else:
                logger.warning(f"Unknown function call: {fn_name}")
                return self._dummy_logic(market_data)
        else:
            # fallback if GPT didn't do function call
            raw_text = choice.message.content.upper() if choice.message.content else ""
            if "BUY" in raw_text:
                return ("BUY", 0.0005)
            elif "SELL" in raw_text:
                return ("SELL", 0.0005)
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
        5. Produce final single row with exactly TRAIN_FEATURE_COLS => predict => 0 => HOLD, 1 => BUY.
        """
        pair = market_data.get("pair", self.pairs[-1])
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
        daily_sent = self._fetch_cryptopanic_sentiment_today()  # for demonstration
        # We'll just set that to the last row's "avg_sentiment"
        df_with_ind["avg_sentiment"] = daily_sent

        # 5) If you'd like to incorporate LunarCrush data, e.g. galaxy_score => can do it.
        symbol = pair.split("/")[0].upper()
        lunar_metrics = self._fetch_lunarcrush_metrics(symbol)
        # For demonstration, let's place the 'sentiment' from LC to "avg_sentiment" or a new column.
        # Or if you had "corr_with_btc" or "avg_sentiment" in your TRAIN_FEATURE_COLS,
        # you might create a new column "lunar_sent" or something.
        # We'll skip advanced merges for brevity, but you can do it if your model used them.

        # We'll take the last row
        latest_row = df_with_ind.iloc[[-1]].copy(deep=True)

        # Ensure all columns from TRAIN_FEATURE_COLS exist
        for col in TRAIN_FEATURE_COLS:
            if col not in latest_row.columns:
                latest_row[col] = 0.0  # fallback for missing data

        # restrict to those columns
        X_input = latest_row[TRAIN_FEATURE_COLS]
        pred_label = self.model.predict(X_input)[0]
        if pred_label == 1:
            return ("BUY", 0.0005)
        else:
            return ("HOLD", 0.0)

    # --------------------------------------------------------------------------
    # Helper: compute RSI, MACD, Bollinger, correlation from a DF
    # --------------------------------------------------------------------------
    def _compute_indicators(self, df: pd.DataFrame, df_btc: pd.DataFrame = None):
        """
        Replicates the logic from train_model for RSI, MACD, Bollinger, correlation.
        """
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
    # CryptoPanic aggregator sentiment (simple daily approach)
    # --------------------------------------------------------------------------
    def _fetch_cryptopanic_sentiment_today(self) -> float:
        """
        Example: we have a daily aggregator that groups by date,
        computing an average sentiment_score. We'll return today's average or 0.0
        """
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
    # --------------------------------------------------------------------------
    def _fetch_lunarcrush_metrics(self, symbol: str) -> dict:
        """
        We'll look up the LATEST entry in 'lunarcrush_data' for the given symbol,
        sorted by timestamp desc. Return it as a dict or empty if none found.
        """
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
            # Let's map them to keys
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
        price = market_data.get("price", 0.0)
        if price < 20000:
            return ("BUY", 0.0005)
        else:
            return ("HOLD", 0.0)
