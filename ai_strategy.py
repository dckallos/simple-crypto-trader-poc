# ==============================================================================
# FILE: ai_strategy.py
# ==============================================================================
"""
ai_strategy.py

A production-ready AIStrategy that:
1) Persists sub-positions in the DB via RiskManagerDB (risk_manager.py).
2) Loads/stores GPT conversation context in DB (table 'ai_context') if desired,
   so we can keep memory beyond single calls.
3) Incorporates aggregator data (CryptoPanic, LunarCrush, etc.) in:
   - scikit model features
   - GPT prompts (via new OpenAI "tools" approach)
4) Supports partial entries/exits, cost-basis recalculation, flipping from
   long to short, all at the sub-position level stored in the DB.

Requires:
 - risk_manager.py: containing RiskManagerDB for sub-positions
 - db.py: for GPT context, trades table, aggregator queries
 - An environment file (.env) with OPENAI_API_KEY, etc.

Upgraded to use the new OpenAI "tools" approach instead of the older "functions"
parameter. See the relevant docs for details.

Usage:
    from ai_strategy import AIStrategy

    strategy = AIStrategy(pairs=["ETH/USD", "XBT/USD"], use_openai=True)
    decision = strategy.predict({
        "pair": "ETH/USD",
        "price": 1500.0,
        "cryptopanic_sentiment": 0.2,
        "galaxy_score": 50.0
    })
    print(decision)   # e.g. ("BUY", 0.001)
"""

import logging
import os
import time
import json
import sqlite3
import numpy as np
import pandas as pd
import joblib
from typing import Optional

from dotenv import load_dotenv
from db import (
    DB_FILE,
    load_gpt_context_from_db,
    save_gpt_context_to_db,
    record_trade_in_db
)
# The new approach for sub-positions
from risk_manager import RiskManagerDB

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# If your scikit model was trained on certain features:
TRAIN_FEATURE_COLS = [
    "feature_price", "feature_ma_3", "feature_spread",
    "vol_change", "rsi", "macd_line", "macd_signal",
    "boll_upper", "boll_lower",
    "corr_with_btc",  # optional
    "avg_sentiment"   # aggregator data
]

class AIStrategy:
    """
    AIStrategy class that decides whether to BUY, SELL, or HOLD using
    either:
      - OpenAI GPT with "tools" approach
      - A trained scikit model (RandomForest or similar)
      - A fallback/dummy logic

    Sub-positions are managed in the DB via RiskManagerDB. We no longer
    store positions in memory.

    GPT context can be persisted in 'ai_context' for memory across calls.
    """

    def __init__(
        self,
        pairs=None,
        model_path: Optional[str] = None,
        use_openai: bool = False,
        max_position_size: float = 0.001,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10,
        max_daily_drawdown: float = -0.02
    ):
        """
        :param pairs: List of trading pairs, e.g. ["XBT/USD", "ETH/USD"].
        :param model_path: Path to a .pkl file with a trained scikit model, if any.
        :param use_openai: If True, we'll attempt an OpenAI-based inference.
        :param max_position_size: Passed to RiskManagerDB to limit trade sizes.
        :param stop_loss_pct: e.g. 5% => auto-close if we drop that far.
        :param take_profit_pct: e.g. 10% => auto-close if we gain that much.
        :param max_daily_drawdown: e.g. -2% => if daily PnL < -2%, no new trades allowed.
        """
        self.pairs = pairs if pairs else ["XBT/USD"]
        self.use_openai = use_openai
        self.model = None

        # Attempt to load scikit model
        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                logger.info(f"AIStrategy: Model loaded from '{model_path}'")
            except Exception as e:
                logger.exception(f"Error loading model: {e}")
        else:
            if model_path:
                logger.warning(f"No model file at '{model_path}'. Using fallback.")
            else:
                logger.info("No model_path provided => fallback logic if use_openai is False.")

        # GPT setup
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY", "FAKE_OPENAI_KEY")
        from openai import OpenAI
        self.client = OpenAI(api_key=openai_api_key)
        logger.info("OpenAI client instantiated in AIStrategy.")

        # Database-based risk manager for sub-positions
        from risk_manager import RiskManagerDB
        self.risk_manager_db = RiskManagerDB(
            db_path=DB_FILE,
            max_position_size=max_position_size,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_daily_drawdown=max_daily_drawdown
        )
        # Ensure sub_positions table exists
        self.risk_manager_db.initialize()

        # GPT conversation context
        self.gpt_context = self._init_gpt_context()

    def _init_gpt_context(self) -> str:
        """
        If you want GPT memory across calls, load from DB.
        Returns the string or empty if none found.
        """
        context_data = load_gpt_context_from_db()
        if context_data:
            logger.info("Loaded GPT context from DB.")
            return context_data
        else:
            return ""

    def _save_gpt_context(self, new_context_str: str) -> None:
        """
        Persists new GPT conversation memory to DB, so we can resume after restarts.
        """
        save_gpt_context_to_db(new_context_str)

    def _append_gpt_context(self, new_text: str) -> None:
        """
        Append text to our GPT memory, then save to DB.
        """
        self.gpt_context += "\n" + new_text
        self._save_gpt_context(self.gpt_context)

    # --------------------------------------------------------------------------
    # Main predict() entry point
    # --------------------------------------------------------------------------
    def predict(self, market_data: dict):
        """
        Decide BUY, SELL, or HOLD given new aggregator data or price snapshot.

        1) Check for forced closures (stop-loss/take-profit) on all sub-positions
           for the given pair (since we got new data).
        2) Attempt GPT or scikit-based inference.
        3) If no model or GPT, fallback to dummy logic.
        4) Return the final decision: ("BUY"/"SELL"/"HOLD", size).

        :param market_data: e.g. {
            "pair": "ETH/USD",
            "price": 1500.0,
            "cryptopanic_sentiment": 0.2,
            "galaxy_score": 50.0,
            ...
        }
        :return: A tuple (action, size).
        """
        pair = market_data.get("pair", self.pairs[-1])
        current_price = market_data.get("price", 0.0)
        if current_price <= 0:
            logger.warning(f"No valid price in market_data for {pair}. Using fallback=HOLD.")
            return ("HOLD", 0.0)

        # 1) Check stop-loss/take-profit on open sub-positions for this pair
        self.risk_manager_db.check_stop_loss_take_profit(pair, current_price)

        # 2) Decide
        if self.use_openai:
            try:
                return self._openai_inference(market_data)
            except Exception as e:
                logger.exception(f"OpenAI inference failed: {e}")
                return self._dummy_logic(market_data)
        elif self.model:
            try:
                return self._model_inference_realtime(market_data)
            except Exception as e:
                logger.exception(f"Error in scikit inference: {e}")
                return self._dummy_logic(market_data)
        else:
            return self._dummy_logic(market_data)

    # --------------------------------------------------------------------------
    # GPT-based approach with aggregator fields
    # Using the new "tools" approach
    # --------------------------------------------------------------------------
    def _openai_inference(self, market_data: dict):
        """
        Calls the OpenAI chat completions with "tools" approach. We supply a single
        tool called "trade_decision" that can be used by GPT to return a trade action.

        The model can pick to either produce normal text or call the tool. If it
        calls the tool, we parse the arguments to get (action, size). We then
        pass that through the RiskManagerDB to open sub-positions.
        """
        import datetime

        pair = market_data.get("pair", "UNKNOWN")
        avg_price = market_data.get("price", 0.0)
        cryptopanic_sent = market_data.get("cryptopanic_sentiment", 0.0)
        galaxy = market_data.get("galaxy_score", 0.0)
        alt_rank = market_data.get("alt_rank", 0)
        # You can also do self._summarize_recent_trades(...) if you'd like
        # or summarize from sub_positions

        # Summaries from 'trades' table to feed GPT
        trade_summary = self._summarize_recent_trades(pair, limit=3)

        # system message
        system_message = {
            "role": "system",
            "content": (
                "You are an advanced crypto trading assistant. "
                "The user provides aggregator data. You may call the tool "
                "'trade_decision' to propose a JSON structured decision."
            )
        }

        user_prompt = (
            f"Context so far:\n{self.gpt_context}\n\n"
            f"Aggregator data for {pair}:\n"
            f"  price={avg_price}, cryptopanic_sentiment={cryptopanic_sent}, "
            f"  galaxy_score={galaxy}, alt_rank={alt_rank}\n"
            f"Recent trades (from 'trades' table):\n{trade_summary}\n\n"
            "Decide: BUY, SELL, or HOLD, returning a tool call if you want."
        )
        user_message = {
            "role": "user",
            "content": user_prompt
        }

        # Tools definition
        tools = [
            {
                "type": "function",
                "function": {
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
                                "description": "Amount to trade"
                            }
                        },
                        "required": ["action", "size"]
                    }
                }
            }
        ]

        # We'll let the model pick whether to call the tool or not
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[system_message, user_message],
            tools=tools,
            tool_choice="auto",   # allows text or tool call
            temperature=0.0,
            max_tokens=500
        )

        logger.debug(f"GPT response: {response}")

        # The response might have finish_reason = "tool_calls" or "stop"
        # We'll handle each choice
        if not response.choices:
            logger.warning("No choices from GPT => fallback HOLD.")
            return ("HOLD", 0.0)

        choice = response.choices[0]
        finish_reason = choice.finish_reason

        if finish_reason == "tool_calls":
            # The model is calling at least one tool
            # We'll parse them from choice.message.tool_calls if present
            if not hasattr(choice.message, "tool_calls"):
                logger.warning("finish_reason='tool_calls' but no tool_calls attribute => fallback HOLD.")
                return ("HOLD", 0.0)

            tool_calls = choice.message.tool_calls
            # Typically it's a list of calls. We'll just handle the first one for simplicity
            if not tool_calls:
                logger.warning("No actual calls in 'tool_calls'. fallback => HOLD.")
                return ("HOLD", 0.0)
            first_call = tool_calls[0]
            tool_name = first_call.name
            tool_args_str = first_call.arguments  # JSON string
            if tool_name == "trade_decision":
                args = json.loads(tool_args_str)
                action = args.get("action", "HOLD")
                size = args.get("size", 0.0)

                # Pass it through risk manager
                final_signal, final_size = self.risk_manager_db.adjust_trade(
                    action,
                    size,
                    pair,
                    avg_price
                )
                # Optionally record in 'trades' if you want
                if final_signal in ("BUY", "SELL") and final_size > 0:
                    record_trade_in_db(final_signal, final_size, avg_price, "GPT_DECISION", pair)
                # Update GPT memory
                self._append_gpt_context(f"GPT function call => action={final_signal}, size={final_size}")
                return (final_signal, final_size)
            else:
                logger.warning(f"GPT called unknown tool={tool_name}. fallback => HOLD.")
                return ("HOLD", 0.0)
        else:
            # The model produced normal text
            # We'll parse if it says "BUY" or "SELL" or "HOLD"
            content_text = choice.message.content or ""
            action = "HOLD"
            if "BUY" in content_text.upper():
                action = "BUY"
            elif "SELL" in content_text.upper():
                action = "SELL"
            size_suggested = 0.0005  # a default

            final_signal, final_size = self.risk_manager_db.adjust_trade(
                action,
                size_suggested,
                pair,
                avg_price
            )
            if final_signal in ("BUY", "SELL") and final_size > 0:
                record_trade_in_db(final_signal, final_size, avg_price, "GPT_DECISION_TEXT", pair)
            self._append_gpt_context(f"GPT text => action={final_signal}, size={final_size}")
            return (final_signal, final_size)

    # --------------------------------------------------------------------------
    # scikit model approach
    # --------------------------------------------------------------------------
    def _model_inference_realtime(self, market_data: dict):
        """
        Uses the scikit model to produce a probability (p_up).
        Then we interpret p_up > 0.6 => BUY, p_up < 0.4 => SELL, else HOLD.
        We open new sub-positions via RiskManagerDB.

        :param market_data: includes "pair", "price", aggregator fields, etc.
        :return: (action, size)
        """
        pair = market_data.get("pair", self.pairs[-1])
        snap_price = market_data.get("price", 0.0)
        if snap_price <= 0.0:
            logger.warning(f"No valid price => fallback hold for {pair}.")
            return ("HOLD", 0.0)

        # fetch last 50 bars
        df_recent = self._fetch_recent_price_data(pair, limit=50)
        if df_recent.empty:
            logger.warning(f"No recent data for {pair}, fallback hold.")
            return ("HOLD", 0.0)

        # (Optional) fetch BTC for correlation if you want that feature
        # skip if pair == "XBT/USD"
        btc_df = None
        if pair != "XBT/USD":
            btc_df = self._fetch_recent_price_data("XBT/USD", limit=50)
        df_ind = self._compute_indicators(df_recent, btc_df)

        # aggregator data => e.g. cryptopanic_sentiment, galaxy_score => if used in model
        cpanic_sent = market_data.get("cryptopanic_sentiment", 0.0)
        df_ind["avg_sentiment"] = cpanic_sent

        # etc. for galaxy_score
        # df_ind["galaxy_score"] = market_data.get("galaxy_score", 0.0)
        # df_ind["alt_rank"] = market_data.get("alt_rank", 0)

        latest_row = df_ind.iloc[[-1]].copy(deep=True)
        for col in TRAIN_FEATURE_COLS:
            if col not in latest_row.columns:
                latest_row[col] = 0.0

        X_input = latest_row[TRAIN_FEATURE_COLS]

        prob_up = 0.5
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_input)[0]
            # class 1 => up
            if len(probs) > 1:
                prob_up = probs[1]
            else:
                # If single class, fallback
                prob_up = 1.0 if self.model.classes_[0] == 1 else 0.0
        else:
            pred_label = self.model.predict(X_input)[0]
            prob_up = 1.0 if pred_label == 1 else 0.0

        action = "HOLD"
        size_suggested = 0.0

        # Basic threshold approach
        if prob_up > 0.6:
            action = "BUY"
            size_suggested = 0.0005
        elif prob_up < 0.4:
            action = "SELL"
            size_suggested = 0.0005

        final_signal, final_size = self.risk_manager_db.adjust_trade(
            action,
            size_suggested,
            pair,
            snap_price
        )
        if final_signal in ("BUY", "SELL") and final_size > 0:
            record_trade_in_db(final_signal, final_size, snap_price, "MODEL_DECISION", pair)

        return (final_signal, final_size)

    # --------------------------------------------------------------------------
    # A fallback logic (dummy) if GPT or scikit fails
    # --------------------------------------------------------------------------
    def _dummy_logic(self, market_data: dict):
        """
        Very naive approach:
        If price < 20k => BUY, else HOLD
        """
        price = market_data.get("price", 0.0)
        pair = market_data.get("pair", self.pairs[-1])
        if price < 20000:
            final_signal, final_size = self.risk_manager_db.adjust_trade(
                "BUY", 0.0005, pair, price
            )
            if final_signal == "BUY" and final_size > 0:
                record_trade_in_db(final_signal, final_size, price, "DUMMY_DECISION", pair)
            return (final_signal, final_size)
        else:
            return ("HOLD", 0.0)

    # --------------------------------------------------------------------------
    # Data / Indicators
    # --------------------------------------------------------------------------
    def _fetch_recent_price_data(self, pair: str, limit=50) -> pd.DataFrame:
        """
        Loads the last 'limit' rows from price_history for the given pair,
        ordered descending by id, then re-ordered ascending by time.

        :param pair: e.g. "ETH/USD"
        :param limit: number of rows to load
        :return: DataFrame with columns: [timestamp, pair, bid_price, ask_price, last_price, volume].
        """
        import sqlite3
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
            # reverse so earliest row is first
            df = df.iloc[::-1].reset_index(drop=True)
            return df
        except Exception as e:
            logger.exception(f"Error loading recent data for {pair}: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def _compute_indicators(self, df: pd.DataFrame, df_btc: pd.DataFrame = None) -> pd.DataFrame:
        """
        Basic technical indicators for the last N rows of price_history data.
        Optionally merges a BTC correlation if df_btc is provided.

        :param df: price_history for a single pair, ascending by timestamp
        :param df_btc: price_history for XBT/USD if you want correlation
        :return: df with additional columns for RSI, MACD, Bollinger, etc.
        """
        if df.empty or "last_price" not in df.columns:
            return pd.DataFrame()

        df = df.sort_values("timestamp").reset_index(drop=True)
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
        sma20 = df["last_price"].rolling(20).mean()
        std20 = df["last_price"].rolling(20).std()
        df["boll_upper"] = sma20 + (2 * std20)
        df["boll_lower"] = sma20 - (2 * std20)

        # Merge correlation with BTC
        if df_btc is not None and not df_btc.empty:
            df_btc = df_btc.sort_values("timestamp").reset_index(drop=True)
            df_btc_ren = df_btc[["timestamp", "last_price"]].rename(columns={"last_price": "btc_price"})
            merged = pd.merge_asof(
                df, df_btc_ren,
                on="timestamp",
                direction="nearest",
                tolerance=30
            )
            merged["corr_with_btc"] = merged["last_price"].rolling(30).corr(merged["btc_price"])
            df = merged

        df.ffill(inplace=True)
        df.bfill(inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method="ffill", inplace=True)
        df.fillna(method="bfill", inplace=True)
        return df

    # --------------------------------------------------------------------------
    # Summarize recent trades from 'trades' table
    # --------------------------------------------------------------------------
    def _summarize_recent_trades(self, pair: str, limit=3) -> str:
        """
        Fetches the last `limit` trades from the 'trades' table for the given pair,
        returning a short textual summary for GPT prompts.

        :return: A string of lines, or "No trades found." if none.
        """
        lines = []
        conn = sqlite3.connect(DB_FILE)
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
                tstamp, side, qty, px = row
                import datetime
                dt_str = datetime.datetime.fromtimestamp(tstamp).strftime("%Y-%m-%d %H:%M")
                lines.append(f"{i}) {dt_str} {side} {qty} @ ${px}")
        except Exception as e:
            logger.exception(f"Error summarizing recent trades for {pair}: {e}")
            return "Error retrieving trades."
        finally:
            conn.close()

        return "\n".join(lines)

