# ==============================================================================
# FILE: ai_strategy.py
# ==============================================================================
"""
ai_strategy.py

A production-ready AIStrategy that:
1) Persists positions in the DB (table 'open_positions') for crash/restart recovery.
2) Loads/stores GPT conversation context in DB (table 'ai_context') if desired,
   so we can keep a memory beyond single calls.
3) Incorporates aggregator data (CryptoPanic daily sentiment, LunarCrush metrics, etc.) in:
   - scikit model features
   - GPT prompts with function calling
4) Supports partial entries/exits, cost-basis recalculation, flipping from long to short.

Requires expansions to db.py (open_positions, ai_context) shown below.

IMPORTANT:
- On startup, you'll call `load_positions_from_db()` to fill `self.current_positions`.
- On each `_update_position`, we call `_save_position_to_db(...)`.
- For GPT context, you can store conversation logs or relevant “memory” in `ai_context`
  so the next call can retrieve them.

Everything is shown in full; you can copy/paste to replace your entire ai_strategy.py.
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

from db import (
    store_price_history,
    DB_FILE,
    load_positions_from_db,
    save_position_to_db,
    load_gpt_context_from_db,
    save_gpt_context_to_db
)
from risk_manager import RiskManager

logger = logging.getLogger(__name__)

BTC_PAIR = "XBT/USD"

# If your scikit model was trained with these columns:
TRAIN_FEATURE_COLS = [
    "feature_price", "feature_ma_3", "feature_spread",
    "vol_change", "rsi", "macd_line", "macd_signal",
    "boll_upper", "boll_lower",
    "corr_with_btc",
    "avg_sentiment"
]


class AIStrategy:
    """
    AIStrategy class that decides whether to BUY, SELL, or HOLD.
    - Positions are persisted in 'open_positions' table so we can recover after a crash.
    - GPT context can be persisted in 'ai_context' if you want memory across calls.
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

        # Attempt to load scikit model
        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                logger.info(f"AIStrategy: Model loaded from '{model_path}'")
            except Exception as e:
                logger.exception(f"AIStrategy: Error loading model: {e}")
        else:
            if model_path:
                logger.warning(f"AIStrategy: No model file at '{model_path}'. Using fallback.")
            else:
                logger.info("AIStrategy: No model_path provided. Using fallback logic.")

        # GPT client
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY", "FAKE_OPENAI_KEY")
        self.client = OpenAI(api_key=openai_api_key)
        logger.info("OpenAI client instantiated in AIStrategy.")

        # 1) Load previously open positions from DB, so we can recover after restarts
        self.current_positions, self.entry_prices = self._init_positions()

        # 2) RiskManager with optional stop-loss, etc.
        self.risk_manager = RiskManager(
            max_position_size=0.001,
            stop_loss_pct=0.05,    # e.g. 5% stop-loss
            take_profit_pct=0.10,  # e.g. 10% take-profit
            max_daily_drawdown=-0.02
        )

        # GPT conversation context (if you want memory across calls)
        # Optional: load from DB if you have a single row or store multiple rows
        self.gpt_context = self._init_gpt_context()

    def _init_positions(self):
        """
        Loads existing positions from 'open_positions' table, returning
        {pair: position_size} and {pair: entry_price} for each open position.
        If the table is empty or pairs not found, returns 0 for those pairs.
        """
        db_positions = load_positions_from_db()
        current_positions = {}
        entry_prices = {}
        for p in self.pairs:
            if p in db_positions:
                data = db_positions[p]
                current_positions[p] = data["size"]
                entry_prices[p] = data["basis"]
            else:
                current_positions[p] = 0.0
                entry_prices[p] = 0.0

        logger.info(f"Loaded positions from DB: {current_positions}")
        return current_positions, entry_prices

    def _init_gpt_context(self):
        """
        If you want GPT memory across calls, load from DB.
        For demonstration, we assume 1 row in 'ai_context' per system.
        """
        context_data = load_gpt_context_from_db()  # might return a str or JSON
        if context_data:
            logger.info("Loaded GPT context from DB.")
            return context_data
        else:
            return ""

    def _fetch_recent_price_data(self, pair: str, limit=50):
        """
        Loads the last 'limit' rows from price_history for the given pair,
        ordered ascending by timestamp.
        """
        import sqlite3
        import pandas as pd

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
            return df[::-1].reset_index(drop=True)
        except Exception as e:
            logger.exception(f"Error loading recent data for {pair}: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # GPT Memory or Conversation Functions
    # --------------------------------------------------------------------------
    def _save_gpt_context(self, new_context_str: str):
        """
        Persists new GPT conversation memory to DB, so we can resume after restarts.
        """
        save_gpt_context_to_db(new_context_str)

    def _append_gpt_context(self, new_text: str):
        """
        If you want a simple approach to append text, then save.
        """
        self.gpt_context += "\n" + new_text
        self._save_gpt_context(self.gpt_context)

    # --------------------------------------------------------------------------
    # Main predict() entry point
    # --------------------------------------------------------------------------
    def predict(self, market_data: dict):
        """
        Decide BUY, SELL, or HOLD.
        aggregator data might have "cryptopanic_sentiment", "galaxy_score", etc.
        """
        pair = market_data.get("pair", self.pairs[-1])
        if self.use_openai:
            try:
                return self._openai_inference(market_data)
            except Exception as e:
                logger.exception(f"OpenAI inference failed: {e}")

        # or scikit
        if self.model:
            try:
                return self._model_inference_realtime(market_data)
            except Exception as e:
                logger.exception(f"Error in scikit inference: {e}")

        return self._dummy_logic(market_data)

    # --------------------------------------------------------------------------
    # GPT-based approach with aggregator data + context
    # --------------------------------------------------------------------------
    def _openai_inference(self, market_data: dict):
        """
        Incorporates aggregator fields. Also includes 'self.gpt_context'
        if you want to provide prior conversation or memory.
        """
        pair = market_data.get("pair", "UNKNOWN")
        avg_price = market_data.get("avg_price", 0.0)
        cryptopanic_sent = market_data.get("cryptopanic_sentiment", 0.0)
        galaxy = market_data.get("galaxy_score", 0.0)
        alt_rank = market_data.get("alt_rank", 0)
        current_pos = self.current_positions.get(pair, 0.0)

        # Summaries from DB
        trade_summary = self._summarize_recent_trades(pair, limit=3)

        system_message = {
            "role": "system",
            "content": (
                "You are an advanced crypto trading assistant with memory. "
                "The user provides aggregator data plus prior context."
            )
        }

        # We include self.gpt_context if you want to feed prior conversation or memory
        user_prompt = (
            f"Previous GPT Memory:\n{self.gpt_context}\n\n"
            f"Aggregator data for {pair}:\n"
            f"  avg_price={avg_price}, cryptopanic={cryptopanic_sent}, galaxy={galaxy}, alt_rank={alt_rank}\n"
            f"Current open position: {current_pos}\n"
            f"Recent trades for {pair}:\n{trade_summary}\n\n"
            "Decide: BUY, SELL, or HOLD, returning a function call."
        )

        user_message = {"role": "user", "content": user_prompt}

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
                            "description": "Amount to trade"
                        }
                    },
                    "required": ["action", "size"]
                }
            }
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[system_message, user_message],
            functions=functions,
            function_call="auto",
            temperature=0.0,
            max_tokens=500
        )
        logger.debug(f'GPT response: {response}')
        choice = response.choices[0]
        finish_reason = choice.finish_reason

        if finish_reason == "function_call":
            fn_call = choice.message.function_call
            if fn_call.name == "trade_decision":
                args = json.loads(fn_call.arguments)
                action = args.get("action", "HOLD")
                size = args.get("size", 0.0)
                action, size = self.risk_manager.adjust_trade(action, size)
                self._update_position(pair, action, size, avg_price)
                # If you want to keep GPT memory:
                self._append_gpt_context("User called function with action=" + action)
                return (action, size)
            else:
                logger.warning(f"Unknown function call: {fn_call.name}")
                return self._dummy_logic(market_data)
        else:
            # fallback text approach
            content_text = choice.message.content or ""
            if "BUY" in content_text.upper():
                action, size = self.risk_manager.adjust_trade("BUY", 0.0005)
                self._update_position(pair, action, size, avg_price)
                self._append_gpt_context("User decided buy from fallback text")
                return (action, size)
            elif "SELL" in content_text.upper():
                action, size = self.risk_manager.adjust_trade("SELL", 0.0005)
                self._update_position(pair, action, size, avg_price)
                self._append_gpt_context("User decided sell from fallback text")
                return (action, size)
            else:
                self._append_gpt_context("No trade in fallback text => HOLD")
                return ("HOLD", 0.0)

    # --------------------------------------------------------------------------
    # scikit model approach with aggregator fields
    # --------------------------------------------------------------------------
    def _model_inference_realtime(self, market_data: dict):
        pair = market_data.get("pair", self.pairs[-1])
        snap_price = market_data.get("price", 0.0)
        if snap_price == 0.0:
            snap_price = market_data.get("avg_price", 0.0)

        # fetch last 50 bars
        df_recent = self._fetch_recent_price_data(pair, limit=50)
        if df_recent.empty:
            logger.warning(f"No recent data for {pair}, fallback to dummy.")
            return self._dummy_logic(market_data)

        # fetch BTC for correlation if needed
        if pair != BTC_PAIR:
            btc_df = self._fetch_recent_price_data(BTC_PAIR, limit=50)
        else:
            btc_df = None

        df_ind = self._compute_indicators(df_recent, btc_df)

        # aggregator data => e.g. cryptopanic_sentiment, galaxy_score => if used in model
        cpanic_sent = market_data.get("cryptopanic_sentiment", 0.0)
        df_ind["avg_sentiment"] = cpanic_sent

        # e.g. if model used "galaxy_score", "alt_rank", you do:
        # df_ind["galaxy_score"] = market_data.get("galaxy_score", 0.0)
        # df_ind["alt_rank"] = market_data.get("alt_rank", 0)

        latest_row = df_ind.iloc[[-1]].copy(deep=True)
        for col in TRAIN_FEATURE_COLS:
            if col not in latest_row.columns:
                latest_row[col] = 0.0

        X_input = latest_row[TRAIN_FEATURE_COLS]

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_input)[0]
            prob_up = probs[1]
        else:
            pred_label = self.model.predict(X_input)[0]
            prob_up = 1.0 if pred_label == 1 else 0.0

        # check forced exit if we have a position
        current_pos = self.current_positions.get(pair, 0.0)
        if current_pos != 0.0:
            entry_price = self.entry_prices.get(pair, snap_price)
            should_close, reason = self.risk_manager.check_take_profit(
                current_price=snap_price,
                entry_price=entry_price,
                current_position=current_pos
            )
            if should_close:
                final_action, final_size = self.risk_manager.adjust_trade("SELL", abs(current_pos))
                self._update_position(pair, final_action, final_size, snap_price)
                logger.info(f"Exiting due to forced exit => {reason}")
                return (final_action, final_size)

        action = "HOLD"
        size_suggested = 0.0
        if current_pos > 0:
            if prob_up < 0.4:
                action = "SELL"
                size_suggested = current_pos
        else:
            if prob_up > 0.6:
                action = "BUY"
                size_suggested = 0.0005
            elif prob_up < 0.4:
                action = "SELL"
                size_suggested = 0.0005
            else:
                action = "HOLD"

        final_action, final_size = self.risk_manager.adjust_trade(action, size_suggested)
        self._update_position(pair, final_action, final_size, snap_price)
        return (final_action, final_size)

    # --------------------------------------------------------------------------
    # compute_indicators
    # --------------------------------------------------------------------------
    def _compute_indicators(self, df: pd.DataFrame, df_btc: pd.DataFrame = None):
        df = df.sort_values("timestamp").reset_index(drop=True)

        df["feature_price"] = df["last_price"]
        df["feature_ma_3"] = df["last_price"].rolling(3).mean()
        df["feature_spread"] = df["ask_price"] - df["bid_price"]
        df["vol_change"] = df["volume"].pct_change().fillna(0)

        window_length = 14
        close_delta = df["last_price"].diff()
        gain = close_delta.clip(lower=0)
        loss = (-1 * close_delta.clip(upper=0))
        avg_gain = gain.rolling(window=window_length, min_periods=window_length).mean()
        avg_loss = loss.rolling(window=window_length, min_periods=window_length).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df["rsi"] = 100 - (100 / (1 + rs))

        ema12 = df["last_price"].ewm(span=12).mean()
        ema26 = df["last_price"].ewm(span=26).mean()
        df["macd_line"] = ema12 - ema26
        df["macd_signal"] = df["macd_line"].ewm(span=9).mean()

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
    # Summarize recent trades
    # --------------------------------------------------------------------------
    def _summarize_recent_trades(self, pair: str, limit=3) -> str:
        conn = sqlite3.connect(DB_FILE)
        lines = []
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
                lines.append(f"{i}) {dt_str} {side} {qty} @ ${price}")
        except Exception as e:
            logger.exception(f"Error summarizing recent trades for {pair}: {e}")
            return "Error retrieving trades."
        finally:
            conn.close()
        return "\n".join(lines)

    # --------------------------------------------------------------------------
    # A fallback logic
    # --------------------------------------------------------------------------
    def _dummy_logic(self, market_data: dict):
        price = market_data.get("price", 0.0)
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
    # Position Update with DB persistence
    # --------------------------------------------------------------------------
    def _update_position(self, pair: str, action: str, trade_size: float, trade_price: float):
        """
        Extended partial entry/exit logic that also saves positions to DB for crash recovery.
        """
        if trade_size <= 0:
            return

        current_pos = self.current_positions.get(pair, 0.0)
        old_pos = current_pos
        entry_price = self.entry_prices.get(pair, 0.0)

        def realize_pnl(closed_qty, exit_price, basis, was_long):
            if was_long:
                return (exit_price - basis) * closed_qty
            else:
                return (basis - exit_price) * closed_qty

        if action == "BUY":
            if old_pos >= 0:
                new_pos = old_pos + trade_size
                if old_pos == 0:
                    self.entry_prices[pair] = trade_price
                else:
                    old_basis = entry_price
                    new_basis = ((old_pos * old_basis) + (trade_size * trade_price)) / (old_pos + trade_size)
                    self.entry_prices[pair] = new_basis
                self.current_positions[pair] = new_pos
            else:
                # short => reduce or flip
                was_short = abs(old_pos)
                if trade_size <= was_short:
                    # partial cover
                    closed_qty = trade_size
                    realized = realize_pnl(closed_qty, trade_price, entry_price, was_long=False)
                    logger.info(f"Covered short for {pair}, realized PnL: {realized:.4f}")
                    self.risk_manager.record_trade_pnl(realized)
                    new_short = was_short - closed_qty
                    self.current_positions[pair] = -new_short
                    if new_short == 0:
                        self.entry_prices[pair] = 0.0
                else:
                    # flip to net long
                    cover_qty = was_short
                    realized = realize_pnl(cover_qty, trade_price, entry_price, was_long=False)
                    logger.info(f"Closed entire short for {pair}, realized PnL: {realized:.4f}")
                    self.risk_manager.record_trade_pnl(realized)
                    leftover_buy = trade_size - was_short
                    self.current_positions[pair] = leftover_buy
                    self.entry_prices[pair] = trade_price

        elif action == "SELL":
            if old_pos > 0:
                if trade_size <= old_pos:
                    # partial or full close
                    realized = realize_pnl(trade_size, trade_price, entry_price, was_long=True)
                    logger.info(f"Sold {trade_size} of long for {pair}, PnL={realized:.4f}")
                    self.risk_manager.record_trade_pnl(realized)
                    new_pos = old_pos - trade_size
                    self.current_positions[pair] = new_pos
                    if new_pos <= 0:
                        if new_pos < 0:
                            leftover_short = abs(new_pos)
                            self.entry_prices[pair] = trade_price
                        else:
                            self.entry_prices[pair] = 0.0
                else:
                    # flipping
                    realized = realize_pnl(old_pos, trade_price, entry_price, was_long=True)
                    logger.info(f"Closed entire long for {pair}, PnL={realized:.4f}")
                    self.risk_manager.record_trade_pnl(realized)
                    leftover_short = trade_size - old_pos
                    self.current_positions[pair] = -leftover_short
                    self.entry_prices[pair] = trade_price
            elif old_pos < 0:
                # add more short
                old_short = abs(old_pos)
                new_short = old_short + trade_size
                if old_pos == 0:
                    self.entry_prices[pair] = trade_price
                    self.current_positions[pair] = -new_short
                else:
                    old_basis = entry_price
                    new_basis = ((old_short * old_basis) + (trade_size * trade_price)) / (old_short + trade_size)
                    self.entry_prices[pair] = new_basis
                    self.current_positions[pair] = -new_short
            else:
                # from flat to short
                self.entry_prices[pair] = trade_price
                self.current_positions[pair] = -trade_size

        else:
            # HOLD => no change
            pass

        # after adjusting position, persist to DB
        final_pos = self.current_positions[pair]
        final_basis = self.entry_prices[pair]
        save_position_to_db(pair, final_pos, final_basis)
