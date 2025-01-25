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
   - GPT prompts (full GPT-based logic)
   - A fallback/dummy logic if GPT fails or is disabled
4) Supports partial entries/exits, cost-basis recalculation, flipping from
   long to short, all at the sub-position level stored in the DB.
5) Records a short rationale in 'ai_decisions' each time it decides
   (BUY, SELL, HOLD).

Important:
- We have removed the scikit-based approach from `_model_inference_realtime`
  in favor of purely GPT-based logic. If you wish to reintroduce a scikit
  path, you can do so, but here we focus on a fully GPT-driven strategy.
- The `_openai_inference(...)` method is our main path for deciding both
  the direction (BUY/SELL/HOLD) and the position size, delegating the logic
  to GPT.
- We still do fallback logic if GPT fails or returns nonsense.

We also rely on:
- `risk_controls` in a dictionary for min buy amounts, max position cost, etc.
- `risk_manager.py` for sub-position DB logic (stop-loss, take-profit).

Usage:
    In `main.py`, create AIStrategy(use_openai=True, risk_controls={...}),
    pass aggregator data as a dict with "price", "cryptopanic_sentiment", etc.
    Then call `predict(market_data)`. The result is (action, size).
"""

import logging
import os
import time
import json
import sqlite3
import numpy as np
import pandas as pd
import joblib
from typing import Optional, Dict, Any

from dotenv import load_dotenv

# Local imports
from db import (
    DB_FILE,
    load_gpt_context_from_db,
    save_gpt_context_to_db,
    record_trade_in_db
)
from risk_manager import RiskManagerDB

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

TRAIN_FEATURE_COLS = [
    # Kept only if you want to compute advanced indicators in the future
    # For a purely GPT approach, these might not be used at all.
    "feature_price",
    "feature_ma_3",
    "feature_spread",
    "vol_change",
    "rsi",
    "macd_line",
    "macd_signal",
    "boll_upper",
    "boll_lower",
    "corr_with_btc",
    "avg_sentiment"
]


class AIStrategy:
    """
    AIStrategy class that decides whether to BUY, SELL, or HOLD using
    GPT-based logic, with a fallback/dummy approach if GPT is unavailable.

    Sub-positions are managed in the DB via RiskManagerDB. We do not
    store position data in memory here.

    GPT context can be persisted in 'ai_context' for memory across calls.

    We also store each final decision in the 'ai_decisions' table with
    a short rationale (message).
    """

    def __init__(
        self,
        pairs=None,
        model_path: Optional[str] = None,  # no longer used, but we keep the param for backward compatibility
        use_openai: bool = False,
        # risk_manager settings => pass to RiskManagerDB:
        max_position_size: float = 0.001,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10,
        max_daily_drawdown: float = -0.02,
        # optional dict of user-defined risk controls for GPT prompt & post-validate
        risk_controls: Optional[Dict[str, Any]] = None
    ):
        """
        :param pairs: e.g. ["XBT/USD", "ETH/USD"].
        :param model_path: Previously used for scikit. Retained for backward compat, not used now.
        :param use_openai: If True, attempt GPT-based inference with function-calling or text parsing.
        :param max_position_size: For RiskManagerDB => clamp trade sizes in quantity terms if you wish.
        :param stop_loss_pct: e.g. 5% => auto-close if we drop that far.
        :param take_profit_pct: e.g. 10% => auto-close if we gain that much.
        :param max_daily_drawdown: e.g. -2% => skip new trades below that PnL.
        :param risk_controls: optional dict with user constraints, e.g.:
              {
                "initial_spending_account": 50.0,
                "purchase_upper_limit_percent": 1.0,
                "minimum_buy_amount": 10.0,
                "max_position_value": 50.0
              }
        """
        self.pairs = pairs if pairs else ["XBT/USD"]
        self.use_openai = use_openai
        self.model = None  # scikit model is no longer used
        self.risk_controls = risk_controls or {}

        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        from openai import OpenAI
        self.client = OpenAI(api_key=openai_api_key)
        logger.info("OpenAI client instantiated in AIStrategy.")

        self.risk_manager_db = RiskManagerDB(
            db_path=DB_FILE,
            max_position_size=max_position_size,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_daily_drawdown=max_daily_drawdown
        )
        self.risk_manager_db.initialize()

        # GPT conversation context from DB
        self.gpt_context = self._init_gpt_context()

        # Create table 'ai_decisions' if missing
        self._create_ai_decisions_table()

    # --------------------------------------------------------------------------
    # GPT context
    # --------------------------------------------------------------------------
    def _init_gpt_context(self) -> str:
        data = load_gpt_context_from_db()
        if data:
            logger.info("Loaded GPT context from DB.")
            return data
        return ""

    def _save_gpt_context(self, new_data: str) -> None:
        save_gpt_context_to_db(new_data)

    def _append_gpt_context(self, new_text: str) -> None:
        """
        Simple approach: just keep appending. A more advanced approach would
        periodically summarize older messages to prevent context from ballooning.
        """
        self.gpt_context += "\n" + new_text
        self._save_gpt_context(self.gpt_context)

    # --------------------------------------------------------------------------
    # 'ai_decisions' table
    # --------------------------------------------------------------------------
    def _create_ai_decisions_table(self):
        conn = sqlite3.connect(DB_FILE)
        try:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS ai_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    pair TEXT,
                    action TEXT,
                    size REAL,
                    rationale TEXT
                )
            """)
            conn.commit()
        except Exception as e:
            logger.exception(f"Error creating ai_decisions table: {e}")
        finally:
            conn.close()

    def _store_decision(self, pair: str, action: str, size: float, rationale: str):
        logger.debug(f"Storing AI decision => {pair}, {action}, {size}, reason={rationale}")
        conn = sqlite3.connect(DB_FILE)
        try:
            c = conn.cursor()
            c.execute("""
                INSERT INTO ai_decisions (timestamp, pair, action, size, rationale)
                VALUES (?, ?, ?, ?, ?)
            """, (int(time.time()), pair, action, size, rationale))
            conn.commit()
        except Exception as e:
            logger.exception(f"Error storing AI decision: {e}")
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # MAIN predict entry
    # --------------------------------------------------------------------------
    def predict(self, market_data: dict):
        """
        Top-level method to produce a final (action, size).
        Also forcibly calls check_stop_loss_take_profit for the given pair
        before or after the GPT logic. Then tries GPT or fallback logic.
        """
        pair = market_data.get("pair", self.pairs[-1])
        current_price = market_data.get("price", 0.0)
        if current_price <= 0:
            rationale = f"No valid price => HOLD {pair}"
            self._store_decision(pair, "HOLD", 0.0, rationale)
            logger.warning(rationale)
            return ("HOLD", 0.0)

        # forced closures first:
        self.risk_manager_db.check_stop_loss_take_profit(pair, current_price)

        # GPT-based approach
        if self.use_openai:
            try:
                return self._full_gpt_inference(market_data)
            except Exception as e:
                logger.exception(f"OpenAI inference failed: {e}")
                return self._dummy_logic(market_data)
        # fallback if use_openai=False
        else:
            return self._dummy_logic(market_data)

    # --------------------------------------------------------------------------
    # Full GPT approach
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # Full GPT approach
    # --------------------------------------------------------------------------
    def _full_gpt_inference(self, market_data: dict):
        """
        We rely entirely on GPT to decide both direction and size.

        We build a prompt that includes aggregator data, recent trades summary,
        plus any risk controls. GPT returns a JSON with {action, size} or text
        that we parse. We specifically use a 'developer' role for top-level instructions
        (priority instructions about how to respond with JSON), and a 'user' role
        for aggregator data and risk constraints. This aligns with OpenAI's recommended
        approach of leveraging message roles for clarity.
        """
        pair = market_data.get("pair", "UNKNOWN")
        price = market_data.get("price", 0.0)
        cpanic = market_data.get("cryptopanic_sentiment", 0.0)
        galaxy = market_data.get("galaxy_score", 0.0)
        alt_rank = market_data.get("alt_rank", 0)
        print(market_data)
        base_rationale = (
            f"Full GPT => pair={pair}, price={price}, "
            f"cryptopanic={cpanic}, galaxy={galaxy}, alt_rank={alt_rank}"
        )

        # Summarize last trades
        summary = self._summarize_recent_trades(pair, limit=3)

        # risk controls if present
        rc = self.risk_controls

        # We'll use a "developer" role for high-level instructions about the output format.
        developer_msg = {
            "role": "assistant",
            "content": (
                "You are an advanced crypto trading assistant. You must produce a single "
                "final action among [BUY, SELL, HOLD] plus a 'size'. If HOLD => size=0. "
                "Return your response as valid JSON only, without any markdown or code block formatting. "
                "The JSON should have the following structure:\n"
                "{\"action\":\"BUY|SELL|HOLD\",\"size\":float}\n\n"
                "Obey these constraints:\n"
                "- If the cost (size * price) < minimum_buy_amount, set action to HOLD and size to 0.\n"
                "- If the cost > purchase_upper_limit, clamp the size so that cost equals purchase_upper_limit.\n"
                "- Do not include any additional text, explanations, or formatting."
            )
        }

        # We'll place aggregator data, GPT context, and constraints into a "user" role.
        user_text = (
            f"GPT context:\n{self.gpt_context}\n\n"
            f"Aggregator:\n  pair={pair}, price={price}, "
            f"cryptopanic={cpanic}, galaxy={galaxy}, alt_rank={alt_rank}\n"
            f"Recent trades:\n{summary}\n"
            "\nConstraints:\n"
            f"  initial_spending_account={rc.get('initial_spending_account', 0.0)}\n"
            f"  purchase_upper_limit_percent={rc.get('purchase_upper_limit_percent', 1.0)}\n"
            f"  minimum_buy_amount={rc.get('minimum_buy_amount', 0.0)}\n"
            f"  max_position_value={rc.get('max_position_value', 999999.0)}\n"
            "\nPlease respond with ONLY the JSON object as specified above."
        )

        user_msg = {"role": "user", "content": user_text}

        # We'll do a simpler approach: pass no function calls, just parse JSON from the assistant text
        print(f'GPT prompt: \ndeveloper message: {developer_msg}, \nuser message: {user_msg}')
        response = self.client.chat.completions.create(
            model="o1-mini",
            messages=[developer_msg, user_msg],
            max_completion_tokens=5000
        )
        print(response)
        logger.debug(f"GPT response: {response}")

        if not response.choices:
            # fallback => hold
            rationale = f"No GPT choices => fallback. {base_rationale}"
            self._store_decision(pair, "HOLD", 0.0, rationale)
            return ("HOLD", 0.0)

        choice = response.choices[0]
        finish_reason = choice.finish_reason
        msg_content = choice.message.content or ""
        logger.info(f"GPT finish_reason={finish_reason}")

        # Remove any potential surrounding whitespace
        msg_content = msg_content.strip()

        # Ensure the response does not contain markdown or code blocks
        if msg_content.startswith("```") and msg_content.endswith("```"):
            # Attempt to extract JSON from within the code block
            try:
                # Find the first newline after the opening ```
                first_newline = msg_content.find("\n")
                if first_newline != -1:
                    # Extract the content between the first newline and the closing ```
                    json_str = msg_content[first_newline + 1:-3].strip()
                else:
                    # If no newline, assume everything between ``` and ``` is JSON
                    json_str = msg_content[3:-3].strip()
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning("Could not parse GPT content as JSON within code blocks => fallback hold.")
                action = "HOLD"
                size_suggested = 0.0
        else:
            # Direct JSON response
            try:
                parsed = json.loads(msg_content)
            except json.JSONDecodeError:
                logger.warning("Could not parse GPT content as JSON => fallback hold.")
                parsed = {}

        # Extract action and size from parsed JSON
        action = parsed.get("action", "HOLD").upper()
        size_suggested = float(parsed.get("size", 0.0)) if "size" in parsed else 0.0

        # post-validate
        final_signal, final_size = self._post_validate_and_adjust(action, size_suggested, price)
        final_signal, final_size = self.risk_manager_db.adjust_trade(
            final_signal, final_size, pair, price
        )

        if final_signal in ("BUY", "SELL") and final_size > 0:
            record_trade_in_db(final_signal, final_size, price, "GPT_DECISION", pair)
        rationale = f"GPT => final={final_signal}, size={final_size}. {base_rationale}"
        self._store_decision(pair, final_signal, final_size, rationale)
        self._append_gpt_context(f"GPT => action={final_signal}, size={final_size}")
        return (final_signal, final_size)

    # --------------------------------------------------------------------------
    # fallback logic
    # --------------------------------------------------------------------------
    def _dummy_logic(self, market_data: dict):
        """
        If price < 20000 => BUY 0.0005
        else => HOLD
        """
        pair = market_data.get("pair", self.pairs[-1])
        px = market_data.get("price", 0.0)
        if px < 20000:
            # We do a final clamp
            final_signal, final_size = self._post_validate_and_adjust("BUY", 0.0005, px)
            final_signal, final_size = self.risk_manager_db.adjust_trade(
                final_signal, final_size, pair, px
            )
            rationale = f"Dummy => price < 20000 => {final_signal}, size={final_size}"
            if final_signal == "BUY" and final_size > 0:
                record_trade_in_db(final_signal, final_size, px, "DUMMY_DECISION", pair)
            self._store_decision(pair, final_signal, final_size, rationale)
            return (final_signal, final_size)
        else:
            rationale = "Dummy => HOLD for price >= 20000"
            self._store_decision(pair, "HOLD", 0.0, rationale)
            return ("HOLD", 0.0)

    # --------------------------------------------------------------------------
    # validate & adjust
    # --------------------------------------------------------------------------
    def _post_validate_and_adjust(self, action: str, size_suggested: float, current_price: float):
        """
        Ensures GPT suggestion or fallback suggestion doesn't exceed or drop
        below constraints from self.risk_controls. For example, do not allow
        trade cost < minimum_buy_amount or > purchase_upper_limit. Return final.

        If action is not in ("BUY", "SELL"), we do ("HOLD", 0.0).
        """
        if action not in ("BUY", "SELL"):
            return ("HOLD", 0.0)

        cost = size_suggested * current_price
        rc = self.risk_controls

        if not rc:  # no constraints => pass through
            return (action, size_suggested)

        min_buy = rc.get("minimum_buy_amount", 0.0)
        upper_percent = rc.get("purchase_upper_limit_percent", 1.0)
        account_total = rc.get("initial_spending_account", 9999999.0)
        purchase_upper = account_total * upper_percent

        # If BUY => check min
        if action == "BUY":
            if cost < min_buy:
                logger.info(f"GPT suggests BUY => cost={cost} < min_buy={min_buy}, forcing HOLD.")
                return ("HOLD", 0.0)
            if cost > purchase_upper:
                new_size = purchase_upper / max(current_price, 1e-9)
                logger.info(
                    f"GPT suggests BUY => cost={cost} > purchase_upper={purchase_upper}, clamping size={new_size}"
                )
                return ("BUY", new_size)

        # For SELL, or if no violation => pass
        return (action, size_suggested)

    # --------------------------------------------------------------------------
    # Data / Indicators (optional if you want advanced aggregator or indicator usage)
    # --------------------------------------------------------------------------
    def _fetch_recent_price_data(self, pair: str, limit: int = 50) -> pd.DataFrame:
        """
        Fetch up to `limit` rows of the most recent price data for `pair` from the
        'price_history' table, then return them in ascending order by timestamp.

        Using named placeholders (:pair, :limit) can address certain IDE warnings
        about param typing in pd.read_sql_query.
        """
        conn = sqlite3.connect(DB_FILE)
        try:
            query = """
                SELECT
                  timestamp,
                  pair,
                  bid_price,
                  ask_price,
                  last_price,
                  volume
                FROM price_history
                WHERE pair = :pair
                ORDER BY id DESC
                LIMIT :limit
            """
            # Using a dictionary for params can help IDEs understand the param types
            df = pd.read_sql_query(query, conn, params={"pair": pair, "limit": limit})
            df = df.iloc[::-1].reset_index(drop=True)
            return df
        except Exception as e:
            logger.exception(f"Error loading price_history for {pair}: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def _compute_indicators(self, df: pd.DataFrame, df_btc: pd.DataFrame = None) -> pd.DataFrame:
        """
        We keep this function if you want advanced indicators to pass to GPT in aggregator data,
        or to store them for debugging. But the current code doesn't necessarily use them for logic.
        """
        if df.empty or "last_price" not in df.columns:
            return pd.DataFrame()

        df = df.sort_values("timestamp").reset_index(drop=True)
        df["feature_price"] = df["last_price"]
        df["feature_ma_3"] = df["last_price"].rolling(3).mean()
        df["feature_spread"] = df["ask_price"] - df["bid_price"]
        df["vol_change"] = df["volume"].pct_change().fillna(0)

        # RSI
        window_len = 14
        delta = df["last_price"].diff()
        gain = delta.clip(lower=0)
        loss = (-1 * delta.clip(upper=0))
        avg_gain = gain.rolling(window=window_len, min_periods=window_len).mean()
        avg_loss = loss.rolling(window=window_len, min_periods=window_len).mean()
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
        df["boll_upper"] = sma20 + 2.0 * std20
        df["boll_lower"] = sma20 - 2.0 * std20

        # correlation with BTC
        if df_btc is not None and not df_btc.empty:
            df_btc = df_btc.sort_values("timestamp").reset_index(drop=True)
            rename_btc = df_btc[["timestamp", "last_price"]].rename(columns={"last_price": "btc_price"})
            merged = pd.merge_asof(
                df, rename_btc,
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
    # Summarize trades
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
                tstamp, side, qty, px = row
                import datetime
                dt_str = datetime.datetime.fromtimestamp(tstamp).strftime("%Y-%m-%d %H:%M")
                lines.append(f"{i}) {dt_str} {side} {qty} @ ${px}")
        except Exception as e:
            logger.exception(f"Error summarizing trades for {pair}: {e}")
            return "Error retrieving trades."
        finally:
            conn.close()

        return "\n".join(lines)
