# ==============================================================================
# FILE: ai_strategy.py
# ==============================================================================
"""
ai_strategy.py

A production-ready AIStrategy that:
1. Loads a trained model if model_path is provided.
2. (NEW) Replicates the same indicator logic (RSI, MACD, Bollinger, volume change, etc.)
   for real-time inference.
3. Incorporates correlation with BTC by loading recent BTC data from DB if needed.
4. Incorporates CryptoPanic sentiment in GPT prompts or as a numeric feature.
5. Falls back to manual normalization or dummy logic if needed.

All existing code is retained. We only add new methods to generate these indicators
in real-time, plus optional GPT usage for sentiment data.
"""

import logging
import sqlite3
import numpy as np
import os
import joblib

# The new official library client approach
from openai import OpenAI

logger = logging.getLogger(__name__)

DB_FILE = "trades.db"

BTC_PAIR = "XBT/USD"  # for correlation reference

class AIStrategy:
    """
    AIStrategy class that decides whether to BUY, SELL, or HOLD based on
    a combination of scikit-learn model inference, GPT signals, or fallback logic.
    Now includes real-time RSI, MACD, Bollinger, correlation, CryptoPanic data, etc.
    """

    def __init__(self, pairs=None, model_path=None, use_openai=False):
        self.pairs = pairs if pairs else ["XBT/USD"]
        self.model = None
        self.use_openai = use_openai

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

        # Instantiate the new OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY", "FAKE_OPENAI_KEY")
        self.client = OpenAI(api_key=openai_api_key)
        logger.info("OpenAI client instantiated in AIStrategy.")

    def predict(self, market_data: dict):
        """
        Decide BUY, SELL, or HOLD for the given market_data, e.g. {"pair":..., "price":..., ...}.

        If use_openai=True, we attempt GPT function-calling with historical trades.
        Otherwise we do scikit model inference or fallback.
        """
        pair = market_data.get("pair", self.pairs[-1])

        # (1) Attempt GPT inference if configured
        if self.use_openai:
            try:
                return self._openai_inference(market_data)
            except Exception as e:
                logger.exception(f"OpenAI inference failed: {e}")

        # (2) If we have a scikit model, do the real-time indicator logic
        if self.model:
            try:
                return self._model_inference_realtime(market_data)
            except Exception as e:
                logger.exception(f"Error in scikit inference: {e}")

        # (3) Otherwise fallback
        return self._dummy_logic(market_data)

    # --------------------------------------------------------------------------
    # GPT Logic with CryptoPanic Summaries + Function Calling
    # --------------------------------------------------------------------------
    def _openai_inference(self, market_data: dict):
        pair = market_data.get("pair", self.pairs[-1])
        current_price = market_data.get("price", 0.0)

        # Summarize last 3 trades
        trade_summary = self._summarize_recent_trades(pair, limit=3)

        # Also fetch aggregated CryptoPanic sentiment (e.g. today's avg)
        sentiment_today = self._fetch_today_sentiment()

        # Provide function signature
        functions = [
            {
                "name": "trade_decision",
                "description": "Return a trade decision in structured JSON",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                        "size": {"type": "number"}
                    },
                    "required": ["action","size"]
                }
            }
        ]

        system_message = {
            "role": "system",
            "content": (
                "You are an advanced crypto trading assistant. "
                "Use the user's historical trades and sentiment data to decide. "
                "Return your final answer by calling the function 'trade_decision'."
            )
        }

        user_message = {
            "role": "user",
            "content": (
                f"Recent trades for {pair}:\n{trade_summary}\n\n"
                f"Current price: {current_price}\n"
                f"Today's average sentiment: {sentiment_today:.2f}\n"
                "Please suggest a trade decision (BUY, SELL, or HOLD) and size. "
                "You must call the function 'trade_decision' with JSON."
            )
        }

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[system_message, user_message],
            functions=functions,
            function_call="auto",
            temperature=0.0,
            max_tokens=150
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
                logger.warning(f"Unknown function call: {fn_name}. Fallback to dummy.")
                return self._dummy_logic(market_data)
        else:
            # Fallback parse
            raw = choice.message.content.upper() if choice.message.content else ""
            if "BUY" in raw:
                return ("BUY", 0.0005)
            elif "SELL" in raw:
                return ("SELL", 0.0005)
            else:
                return ("HOLD", 0.0)

    # --------------------------------------------------------------------------
    # Real-Time Scikit Inference with Indicators
    # --------------------------------------------------------------------------
    def _model_inference_realtime(self, market_data: dict):
        """
        Recompute indicators for the last ~30 rows in DB, then produce scikit prediction.
        """
        pair = market_data.get("pair", self.pairs[-1])

        # 1) load recent data for this pair
        recent_df = self._fetch_recent_price_data(pair, 50)  # 50 bars
        if recent_df.empty:
            logger.warning(f"No recent data to do real-time scikit inference. Fallback.")
            return self._dummy_logic(market_data)

        # 2) possibly load BTC data for correlation
        if pair != BTC_PAIR:
            btc_df = self._fetch_recent_price_data(BTC_PAIR, 50)
        else:
            btc_df = None

        # 3) compute indicators
        df_with_ind = self._compute_indicators(recent_df, btc_df)

        # 4) generate row for the "latest" bar
        latest_row = df_with_ind.iloc[[-1]]  # shape (1, n_features)
        # drop columns not in the model
        needed_cols = [
            col for col in latest_row.columns
            if col not in ["timestamp","pair","trade_date","future_price","label_up"]
        ]
        X_input = latest_row[needed_cols]

        # 5) predict
        pred_label = self.model.predict(X_input)[0]  # 0 or 1
        if pred_label == 1:
            # e.g. size 0.0005
            return ("BUY", 0.0005)
        else:
            return ("HOLD", 0.0)

    def _compute_indicators(self, df: np.ndarray, df_btc: np.ndarray=None):
        """
        Similar logic to build_features_and_labels but for real-time.
        We won't produce a label. We'll just create features in the DataFrame.
        """
        import pandas as pd
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Basic
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

        # Correlation if we have BTC
        if df_btc is not None and not df_btc.empty:
            df_btc = df_btc.sort_values("timestamp").reset_index(drop=True)
            df_btc_ren = df_btc[["timestamp","last_price"]].rename(columns={"last_price":"btc_price"})
            df_merged = pd.merge_asof(df, df_btc_ren, on="timestamp", direction="nearest", tolerance=30)
            df = df_merged
            df["corr_with_btc"] = df["last_price"].rolling(30).corr(df["btc_price"])

        # Fill
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        return df

    def _fetch_recent_price_data(self, pair: str, limit=50):
        """
        Load last 'limit' rows from price_history for the pair, return as DataFrame.
        """
        conn = sqlite3.connect(DB_FILE)
        try:
            import pandas as pd
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

    def _fetch_today_sentiment(self):
        """
        Example: fetch today's average sentiment from cryptopanic_news table.
        Returns a float, or 0.0 if not found.
        """
        import datetime
        today_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        conn = sqlite3.connect(DB_FILE)
        try:
            c = conn.cursor()
            query = f"""
                SELECT AVG(sentiment_score)
                FROM cryptopanic_news
                WHERE DATE(timestamp, 'unixepoch') = '{today_str}'
            """
            row = c.execute(query).fetchone()
            if row and row[0] is not None:
                return float(row[0])
            return 0.0
        except Exception as e:
            logger.exception(f"Error fetching today's sentiment: {e}")
            return 0.0
        finally:
            conn.close()

    def _summarize_recent_trades(self, pair: str, limit=3):
        """
        Similar to your existing summarizing approach, returning short text.
        """
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

    def _dummy_logic(self, market_data: dict):
        price = market_data.get("price", 0.0)
        if price < 20000:
            return ("BUY", 0.0005)
        else:
            return ("HOLD", 0.0)

    def _fetch_latest_prices_for_pairs(self):
        """
        Existing code to gather last prices for all pairs.
        """
        multi_prices = {}
        conn = sqlite3.connect(DB_FILE)
        try:
            c = conn.cursor()
            for pair in self.pairs:
                c.execute("""
                    SELECT last_price
                    FROM price_history
                    WHERE pair=?
                    ORDER BY id DESC
                    LIMIT 1
                """, (pair,))
                row = c.fetchone()
                if row:
                    multi_prices[pair] = row[0]
        except Exception as e:
            logger.exception(f"Error fetching multi-pair data: {e}")
        finally:
            conn.close()
        return multi_prices
