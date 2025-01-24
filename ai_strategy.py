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
5) Stores a short message or rationale in a new 'ai_decisions' table
   each time it makes a final decision (BUY, SELL, HOLD).

Requires:
 - risk_manager.py: containing RiskManagerDB for sub-positions
 - db.py for GPT context, aggregator queries, or just minimal usage
 - A valid environment file (.env) with OPENAI_API_KEY, etc.

Usage:
    from ai_strategy import AIStrategy

    strategy = AIStrategy(pairs=["ETH/USD", "XBT/USD"], use_openai=True, model_path="trained_model.pkl")
    market_data = {
        "pair": "ETH/USD",
        "price": 1500.0,
        "cryptopanic_sentiment": 0.2,
        "galaxy_score": 50.0
    }
    decision = strategy.predict(market_data)
    print(decision)   # e.g. ("BUY", 0.001)

    # Meanwhile, each final decision is recorded in 'ai_decisions' with a rationale.
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
from risk_manager import RiskManagerDB

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# If your scikit model was trained on certain features:
TRAIN_FEATURE_COLS = [
    "feature_price", "feature_ma_3", "feature_spread",
    "vol_change", "rsi", "macd_line", "macd_signal",
    "boll_upper", "boll_lower",
    "corr_with_btc",  # optional if you trained with it
    "avg_sentiment"   # aggregator data
]


class AIStrategy:
    """
    AIStrategy class that decides whether to BUY, SELL, or HOLD using
    either:
      - GPT-based logic (new "tools" approach)
      - A trained scikit model
      - A fallback/dummy approach (price < 20k => buy, else hold)

    Sub-positions are stored in 'sub_positions' via RiskManagerDB.
    Each final decision is also recorded in 'ai_decisions' with a short rationale.
    GPT context can be stored in 'ai_context' for memory across calls.
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
        :param pairs: e.g. ["XBT/USD", "ETH/USD"].
        :param model_path: path to a .pkl scikit model if any.
        :param use_openai: If True, do GPT-based inference.
        :param max_position_size: for RiskManagerDB, clamp trade sizes.
        :param stop_loss_pct: e.g. 0.05 => auto-close if -5% losing.
        :param take_profit_pct: e.g. 0.10 => auto-close if +10%.
        :param max_daily_drawdown: e.g. -0.02 => if daily realized PnL < -2%, skip new trades => HOLD
        """
        self.pairs = pairs if pairs else ["XBT/USD"]
        self.use_openai = use_openai
        self.model = None

        # Try to load scikit model
        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                logger.info(f"AIStrategy: Model loaded from '{model_path}'")
            except Exception as e:
                logger.exception(f"AIStrategy: Error loading model: {e}")
        else:
            if model_path:
                logger.warning(f"AIStrategy: No model file at '{model_path}'. Using dummy fallback.")
            else:
                logger.info("AIStrategy: No model_path => dummy fallback or GPT if enabled.")

        # GPT
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY", "FAKE_OPENAI_KEY")
        from openai import OpenAI
        self.client = OpenAI(api_key=openai_api_key)
        logger.info("OpenAI client instantiated in AIStrategy.")

        # DB-based risk manager for sub-positions
        self.risk_manager_db = RiskManagerDB(
            db_path=DB_FILE,
            max_position_size=max_position_size,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_daily_drawdown=max_daily_drawdown
        )
        self.risk_manager_db.initialize()

        # GPT context
        self.gpt_context = self._init_gpt_context()

        # Create 'ai_decisions' table
        self._create_ai_decisions_table()

    def _init_gpt_context(self) -> str:
        ctx_data = load_gpt_context_from_db()
        if ctx_data:
            logger.info("Loaded GPT context from DB.")
            return ctx_data
        return ""

    def _save_gpt_context(self, new_context: str):
        save_gpt_context_to_db(new_context)

    def _append_gpt_context(self, new_text: str):
        self.gpt_context += "\n" + new_text
        self._save_gpt_context(self.gpt_context)

    # --------------------------------------------------------------------------
    # Decision Logging: 'ai_decisions'
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
        logger.debug(f"Storing decision => action={action}, rationale={rationale}")
        conn = sqlite3.connect(DB_FILE)
        try:
            c = conn.cursor()
            ts = int(time.time())
            c.execute("""
                INSERT INTO ai_decisions (timestamp, pair, action, size, rationale)
                VALUES (?, ?, ?, ?, ?)
            """, (ts, pair, action, size, rationale))
            conn.commit()
        except Exception as e:
            logger.exception(f"Error storing AI decision: {e}")
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # predict(market_data) => final (action, size)
    # --------------------------------------------------------------------------
    def predict(self, market_data: dict):
        """
        The main decision point. We pass aggregator data, see if we want GPT, scikit, or fallback.

        Steps:
          1) Possibly call GPT if use_openai is True.
          2) Else check if we have a scikit model => scikit approach.
          3) Else dummy approach => price < 20k => buy.

        The final action/size is also stored in 'sub_positions' by risk_manager_db,
        and we store a short rationale in 'ai_decisions'.
        """
        pair = market_data.get("pair", self.pairs[-1])
        # Check forced closure for sub-positions if you want. We'll do it in risk_manager_db or aggregator cycle.

        if self.use_openai:
            try:
                return self._openai_inference(market_data)
            except Exception as e:
                logger.exception(f"OpenAI failed => {e}")
                return self._dummy_logic(market_data)
        elif self.model:
            try:
                return self._model_inference(market_data)
            except Exception as e:
                logger.exception(f"Scikit inference error => {e}")
                return self._dummy_logic(market_data)
        else:
            return self._dummy_logic(market_data)

    # --------------------------------------------------------------------------
    # GPT logic
    # --------------------------------------------------------------------------
    def _openai_inference(self, market_data: dict):
        """
        Creates GPT system/user messages with aggregator fields + self.gpt_context.
        Then uses the 'tool' approach to call 'trade_decision'.
        """
        pair = market_data.get("pair", "UNKNOWN")
        snap_price = market_data.get("price", 0.0)
        cpanic_sent = market_data.get("cryptopanic_sentiment", 0.0)
        galaxy = market_data.get("galaxy_score", 0.0)
        alt_rank = market_data.get("alt_rank", 0)
        rationale_prefix = f"GPT-based => pair={pair}, price={snap_price:.2f}, cpanic={cpanic_sent}, galaxy={galaxy}, alt_rank={alt_rank}"

        system_msg = {
            "role": "system",
            "content": "You are an advanced trading assistant with aggregator data."
        }

        user_msg = {
            "role": "user",
            "content": (
                f"Previous GPT context:\n{self.gpt_context}\n\n"
                f"Aggregator => price={snap_price}, cryptopanic={cpanic_sent}, galaxy={galaxy}, alt_rank={alt_rank}\n"
                "Decide: BUY, SELL, or HOLD. Return a JSON with 'action' & 'size'."
            )
        }

        functions = [
            {
                "type": "function",
                "function": {
                    "name": "trade_decision",
                    "description": "Return a trade decision in structured JSON",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                            "size": {"type": "number"}
                        },
                        "required": ["action", "size"]
                    }
                }
            }
        ]

        from openai import APIError
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[system_msg, user_msg],
                tools=functions,
                tool_choice="auto",
                temperature=0.0
            )
        except APIError as e:
            rationale = f"{rationale_prefix} => GPT call error => {e}"
            logger.exception(rationale)
            self._store_decision(pair, "HOLD", 0.0, rationale)
            return ("HOLD", 0.0)

        logger.debug(f"GPT response => {resp}")
        if not resp.choices:
            rationale = f"No GPT choices => fallback. {rationale_prefix}"
            self._store_decision(pair, "HOLD", 0.0, rationale)
            return ("HOLD", 0.0)

        choice = resp.choices[0]
        finish_reason = choice.finish_reason
        # If the model calls the 'trade_decision' tool
        if finish_reason == "tool_calls":
            if not hasattr(choice.message, "tool_calls"):
                rationale = f"No actual tool_calls => fallback. {rationale_prefix}"
                self._store_decision(pair, "HOLD", 0.0, rationale)
                return ("HOLD", 0.0)

            calls = choice.message.tool_calls
            if not calls:
                rationale = f"No tool call => fallback. {rationale_prefix}"
                self._store_decision(pair, "HOLD", 0.0, rationale)
                return ("HOLD", 0.0)

            first_call = calls[0]
            try:
                args = json.loads(first_call.arguments)
            except json.JSONDecodeError:
                rationale = f"Tool call parse error => fallback. {rationale_prefix}"
                self._store_decision(pair, "HOLD", 0.0, rationale)
                return ("HOLD", 0.0)

            action = args.get("action", "HOLD")
            size = args.get("size", 0.0)
            rationale = f"{rationale_prefix} => tool_calls => {action}, size={size}"
            final_action, final_size = self.risk_manager_db.adjust_trade(action, size, pair, snap_price)
            if final_action in ("BUY", "SELL") and final_size > 0:
                record_trade_in_db(final_action, final_size, snap_price, "GPT_DECISION", pair)
            self._store_decision(pair, final_action, final_size, rationale)
            self._append_gpt_context(rationale)
            return (final_action, final_size)

        else:
            # fallback text approach
            text_out = choice.message.content.lower()
            rationale = f"{rationale_prefix} => fallback text parse => {text_out}"
            if "buy" in text_out:
                action = "BUY"
                size = 0.0005
            elif "sell" in text_out:
                action = "SELL"
                size = 0.0005
            else:
                action = "HOLD"
                size = 0.0

            final_action, final_size = self.risk_manager_db.adjust_trade(action, size, pair, snap_price)
            if final_action in ("BUY", "SELL") and final_size > 0:
                record_trade_in_db(final_action, final_size, snap_price, "GPT_DECISION_TEXT", pair)
            rationale += f". final => {final_action}, size={final_size}"
            self._store_decision(pair, final_action, final_size, rationale)
            self._append_gpt_context(rationale)
            return (final_action, final_size)

    # --------------------------------------------------------------------------
    # scikit approach
    # --------------------------------------------------------------------------
    def _model_inference(self, market_data: dict):
        pair = market_data.get("pair", self.pairs[-1])
        snap_price = market_data.get("price", 0.0)
        if snap_price <= 0.0:
            # fallback
            rationale = f"scikit => no valid price => fallback dummy"
            self._store_decision(pair, "HOLD", 0.0, rationale)
            return self._dummy_logic(market_data)

        # retrieve data from price_history
        df_recent = self._fetch_recent_price_data(pair, 50)
        if df_recent.empty:
            rationale = f"No recent data => fallback dummy for {pair}."
            self._store_decision(pair, "HOLD", 0.0, rationale)
            return self._dummy_logic(market_data)

        # aggregator data => e.g. cryptopanic, galaxy
        aggregator_sent = market_data.get("cryptopanic_sentiment", 0.0)
        aggregator_galaxy = market_data.get("galaxy_score", 0.0)
        aggregator_alt = market_data.get("alt_rank", 0)

        # if pair != "XBT/USD", we can load BTC for correlation
        # For demonstration, skip or do partial
        # build indicators
        df_ind = self._compute_indicators(df_recent)

        df_ind["avg_sentiment"] = aggregator_sent
        # optionally df_ind["galaxy_score"] = aggregator_galaxy
        # optionally df_ind["alt_rank"] = aggregator_alt

        latest_row = df_ind.iloc[[-1]].copy()
        for col in TRAIN_FEATURE_COLS:
            if col not in latest_row.columns:
                latest_row[col] = 0.0

        X_input = latest_row[TRAIN_FEATURE_COLS]

        # Probability => p_up
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_input)[0]
            p_up = probs[1]
        else:
            pred_label = self.model.predict(X_input)[0]
            p_up = 1.0 if pred_label == 1 else 0.0

        # example: p_up>0.6 => BUY, p_up<0.4 => SELL
        rationale = f"scikit => p_up={p_up:.2f}"
        action = "HOLD"
        size = 0.0
        if p_up > 0.6:
            action = "BUY"
            size = 0.0005
        elif p_up < 0.4:
            action = "SELL"
            size = 0.0005

        final_action, final_size = self.risk_manager_db.adjust_trade(action, size, pair, snap_price)
        if final_action in ("BUY", "SELL") and final_size > 0:
            record_trade_in_db(final_action, final_size, snap_price, "MODEL_DECISION", pair)
        rationale += f". final => {final_action}, size={final_size}"
        self._store_decision(pair, final_action, final_size, rationale)
        return (final_action, final_size)

    # --------------------------------------------------------------------------
    # Fallback Dummy
    # --------------------------------------------------------------------------
    def _dummy_logic(self, market_data: dict):
        """
        If price < 20000 => BUY, else HOLD. We also log a rationale in 'ai_decisions'.
        """
        pair = market_data.get("pair", self.pairs[-1])
        snap_price = market_data.get("price", 0.0)
        if snap_price < 20000:
            final_action, final_size = self.risk_manager_db.adjust_trade("BUY", 0.0005, pair, snap_price)
            rationale = f"Dummy => price<20000 => BUY => final_size={final_size}"
            self._store_decision(pair, final_action, final_size, rationale)
            if final_action == "BUY" and final_size > 0:
                record_trade_in_db(final_action, final_size, snap_price, "DUMMY_DECISION", pair)
            return (final_action, final_size)
        else:
            rationale = f"Dummy => price>=20000 => HOLD"
            self._store_decision(pair, "HOLD", 0.0, rationale)
            return ("HOLD", 0.0)

    # --------------------------------------------------------------------------
    # Data / Indicators
    # --------------------------------------------------------------------------
    def _fetch_recent_price_data(self, pair: str, limit=50) -> pd.DataFrame:
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
            df = df.iloc[::-1].reset_index(drop=True)
            return df
        except Exception as e:
            logger.exception(f"Error loading recent data for {pair}: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
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

        df.ffill(inplace=True)
        df.bfill(inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method="ffill", inplace=True)
        df.fillna(method="bfill", inplace=True)

        return df

