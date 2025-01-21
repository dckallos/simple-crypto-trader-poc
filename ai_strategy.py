# ==============================================================================
# FILE: ai_strategy.py
# ==============================================================================
"""
ai_strategy.py

A production-ready AIStrategy that:
1. Loads a trained model if model_path is provided.
2. Replicates the 3-feature logic used in `train_model.py`:
   - feature_price  = last_price
   - feature_ma_3   = rolling mean over the last 3 rows
   - feature_spread = ask_price - bid_price
3. If data is incomplete, we fallback to dummy logic or manual normalization.

ADDITION: We now integrate the new OpenAI Python library usage (v1), which uses
the 'OpenAI' client class. We also:
- Provide a short summary of historical data to the GPT model (so it has more context).
- Use function calling to parse structured output.

Everything else remains from earlier versions, ensuring no code is lost.
"""

import logging
import sqlite3
import numpy as np
import os
import joblib

# Use the new official OpenAI library client approach
from openai import OpenAI

logger = logging.getLogger(__name__)

DB_FILE = "trades.db"

class AIStrategy:
    """
    AIStrategy class that decides whether to BUY, SELL, or HOLD based on
    AI-driven signals. We can also feed a short summary of historical data to GPT
    for better context. The model can then produce structured output
    (action + size) via function calling. If GPT fails or isn't used, we fallback
    to scikit or manual logic.
    """

    def __init__(self, pairs=None, model_path=None, use_openai=False):
        """
        :param pairs: List of trading pairs, e.g. ["XBT/USD", "ETH/USD"].
        :param model_path: path to a .pkl file with a trained model, if any.
        :param use_openai: bool flag. If True, we'll attempt an OpenAI-based inference.
        """
        self.pairs = pairs if pairs else ["XBT/USD"]
        self.model = None
        self.use_openai = use_openai

        # Attempt to load scikit model
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
        Decide BUY, SELL, or HOLD. If model is loaded & data is complete, we
        do the scikit-based 3-feature inference. Otherwise fallback to manual.

        If 'use_openai' is True, we attempt a function-calling approach in
        _openai_inference(...), providing a short summary of historical data as context.

        :param market_data: A dict with {"price": float, "timestamp": float, "pair": str}, etc.
        :return: (signal, size)
        """
        # Check the DB for multi-pair data
        multi_data = self._fetch_latest_prices_for_pairs()
        if len(multi_data) < len(self.pairs):
            logger.warning("Not all pairs had data. Running fallback logic.")
            return self._dummy_logic(market_data)

        # If user wants GPT, try that first
        if self.use_openai:
            try:
                return self._openai_inference(market_data)
            except Exception as e:
                logger.exception(f"OpenAI inference failed: {e}. Fallback logic.")
                # then proceed to scikit or fallback

        # If we have a scikit model, do 3-feature inference
        if self.model:
            try:
                return self._model_inference(market_data)
            except Exception as e:
                logger.exception(f"Error in model inference: {e}. Using fallback logic.")
                return self._dummy_logic(market_data)
        else:
            # No model => fallback to manual normalization
            return self._manual_normalization_logic(market_data, multi_data)

    # --------------------------------------------------------------------------
    # NEW: Provide a short historical summary to GPT
    # --------------------------------------------------------------------------
    def _openai_inference(self, market_data: dict):
        """
        Attempts a function-calling approach to obtain structured JSON:
        e.g., { "action": "BUY", "size": 0.0005 }.
        We also gather a short summary of historical data (like last 3 trades)
        to give GPT context. GPT can then see that context in the prompt.
        """
        pair = market_data.get("pair", self.pairs[-1])
        current_price = market_data.get("price", 0.0)

        # Summarize last 3 trades from the DB for this pair
        trade_summary = self._summarize_recent_trades(pair, limit=3)  # <-- added

        # Provide the function signature
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
                    "required": ["action","size"]
                }
            }
        ]

        # System message
        system_message = {
            "role": "system",
            "content": (
                "You are a crypto trading assistant. "
                "Decide whether to BUY, SELL, or HOLD for each query. "
                "Return your final answer by calling the function 'trade_decision' with structured JSON."
            )
        }

        # Combine user message with historical context
        user_message_content = (
            f"Recent trades for {pair}:\n{trade_summary}\n"
            f"Current price is {current_price}. "
            "Suggest a trade decision. You must call the function with your final decision."
        )
        user_message = {
            "role": "user",
            "content": user_message_content
        }

        # Call GPT with function-calling approach
        response = self.client.chat.completions.create(
            model="gpt-4o",  # e.g. "gpt-4o" from the readme, or "gpt-3.5-turbo"
            messages=[system_message, user_message],
            functions=functions,
            function_call="auto",
            temperature=0.0,
            max_tokens=150
        )

        choice = response.choices[0]
        finish_reason = choice.finish_reason

        if finish_reason == "function_call":
            function_call = choice.message.function_call
            function_name = function_call.name
            if function_name == "trade_decision":
                import json
                args = json.loads(function_call.arguments)
                action = args.get("action", "HOLD")
                size = args.get("size", 0.0)
                return (action, size)
            else:
                logger.warning(f"Unknown function call: {function_name}. Falling back.")
                raise ValueError("Unknown function in GPT result")
        else:
            raw_reply = choice.message.content.strip().upper() if choice.message.content else ""
            logger.info(f"GPT raw text = {raw_reply}")
            if "BUY" in raw_reply:
                return ("BUY", 0.0005)
            elif "SELL" in raw_reply:
                return ("SELL", 0.0005)
            else:
                return ("HOLD", 0.0)

    # --------------------------------------------------------------------------
    # Model Inference with 3 Features (unchanged)
    # --------------------------------------------------------------------------
    def _model_inference(self, market_data: dict):
        """
        1. Gather last 3 DB rows for the coin to compute feature_ma_3.
        2. Use the current row's bid, ask, last_price for feature_spread & feature_price.
        3. Construct [feature_price, feature_ma_3, feature_spread] => shape=(1,3).
        4. self.model.predict(...) => 0 => HOLD, 1 => BUY, or 2 => SELL if you want more classes.

        For now, we'll assume it's binary (0 => hold, 1 => buy).
        """
        pair = market_data.get("pair", self.pairs[-1])

        feats = self._build_3features_for_inference(pair)
        if feats is None:
            logger.warning(f"Could not build 3 features for {pair}. Fallback.")
            return self._dummy_logic(market_data)

        feats_2d = feats.reshape(1, -1)
        pred_label = self.model.predict(feats_2d)[0]
        if pred_label == 1:
            return ("BUY", 0.0005)
        else:
            return ("HOLD", 0.0)

    # --------------------------------------------------------------------------
    # Build the 3 training features from DB (unchanged)
    # --------------------------------------------------------------------------
    def _build_3features_for_inference(self, pair: str):
        rows = self._fetch_recent_rows(pair, num_rows=3)
        if len(rows) < 3:
            return None

        # each row => (timestamp, bid, ask, last, volume)
        last_prices = [r[3] for r in rows]
        feature_price = rows[-1][3]
        feature_ma_3 = np.mean(last_prices)
        last_bid = rows[-1][1]
        last_ask = rows[-1][2]
        feature_spread = last_ask - last_bid

        return np.array([feature_price, feature_ma_3, feature_spread], dtype=float)

    def _fetch_recent_rows(self, pair: str, num_rows=3):
        conn = sqlite3.connect(DB_FILE)
        try:
            c = conn.cursor()
            c.execute(f"""
                SELECT timestamp, bid_price, ask_price, last_price, volume
                FROM price_history
                WHERE pair=?
                ORDER BY id DESC
                LIMIT ?
            """, (pair, num_rows))
            rows_desc = c.fetchall()
            return rows_desc[::-1]  # Reverse to ascending
        except Exception as e:
            logger.exception(f"Error fetching recent rows for pair={pair}: {e}")
            return []
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # Summarize recent trades for GPT
    # --------------------------------------------------------------------------
    def _summarize_recent_trades(self, pair: str, limit=3) -> str:
        """
        Fetch last 'limit' trades from the 'trades' table for this pair.
        Return a short text summary that GPT can read in the prompt.

        e.g. "1) 2023-11-10 BUY 0.001 BTC @ $24,500\n 2) 2023-11-11 SELL..."

        If none found, returns "No trades found."
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

            # Build summary
            for i, row in enumerate(rows[::-1], start=1):
                tstamp, side, qty, price = row
                # convert timestamp to readable form if desired
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
    # Manual Normalization (No Model) (unchanged)
    # --------------------------------------------------------------------------
    def _manual_normalization_logic(self, market_data: dict, multi_prices: dict):
        prices_array = np.array(list(multi_prices.values()), dtype=float)
        min_p = prices_array.min()
        max_p = prices_array.max()
        if max_p == min_p:
            norm_prices = np.ones_like(prices_array)
        else:
            norm_prices = (prices_array - min_p) / (max_p - min_p)

        my_pair = market_data.get("pair", self.pairs[-1])
        my_price_index = self.pairs.index(my_pair)
        my_norm = norm_prices[my_price_index]
        logger.info(f"Multi-pair normalization: {multi_prices}, "
                    f"normalized={norm_prices}, my_norm={my_norm:.4f}")

        if my_norm < 0.5:
            return ("BUY", 0.0005)
        else:
            return ("HOLD", 0.0)

    # --------------------------------------------------------------------------
    # Fallback Logic (unchanged)
    # --------------------------------------------------------------------------
    def _dummy_logic(self, market_data: dict):
        price = market_data.get("price", 0.0)
        if price < 20000:
            return ("BUY", 0.0005)
        else:
            return ("HOLD", 0.0)

    # --------------------------------------------------------------------------
    # Fetch Latest 'last_price' for Each Pair (unchanged)
    # --------------------------------------------------------------------------
    def _fetch_latest_prices_for_pairs(self):
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
