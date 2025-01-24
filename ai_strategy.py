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
   - GPT prompts with optional function calls
4) Supports partial entries/exits, cost-basis recalculation, flipping from
   long to short, all at the sub-position level stored in the DB.
5) Records a short rationale in 'ai_decisions' each time it decides
   (BUY, SELL, HOLD).

Important:
- The new openai Python library v1 often sets `choice.finish_reason` to
  'stop', 'tool_calls', 'content_filter', etc. and places function call
  data in `choice.message.function_call` if GPT calls a function.
- We handle both function_call logic (post-validating GPT's size) and
  fallback text logic.
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

# If your scikit model was trained on certain features:
TRAIN_FEATURE_COLS = [
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
    either:
      - GPT-based inference (with optional function_call usage)
      - A trained scikit model
      - A fallback/dummy logic

    Sub-positions are managed in the DB via RiskManagerDB. We do not
    store position data in memory here.

    GPT context can be persisted in 'ai_context' for memory across calls.

    We also store each final decision in the 'ai_decisions' table with
    a short rationale (message).
    """

    def __init__(
        self,
        pairs=None,
        model_path: Optional[str] = None,
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
        :param model_path: Path to a .pkl scikit model if any.
        :param use_openai: If True, attempt GPT-based inference.
        :param max_position_size: For RiskManagerDB => clamp trade sizes.
        :param stop_loss_pct: e.g. 5% => auto-close if we drop that far.
        :param take_profit_pct: e.g. 10% => auto-close if we gain that much.
        :param max_daily_drawdown: e.g. -2% => skip new trades below that PnL.
        :param risk_controls: optional dict with user constraints, e.g.:
              {
                "initial_spending_account": 10000.0,
                "purchase_upper_limit_percent": 0.01,
                "minimum_buy_amount": 20.0,
                "max_position_value": 5000.0
              }
        """
        self.pairs = pairs if pairs else ["XBT/USD"]
        self.use_openai = use_openai
        self.model = None
        self.risk_controls = risk_controls or {}

        # Attempt to load scikit model if provided
        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                logger.info(f"AIStrategy: Model loaded from '{model_path}'")
            except Exception as e:
                logger.exception(f"Error loading scikit model: {e}")
        else:
            if model_path:
                logger.warning(f"No file at '{model_path}'. Using fallback logic.")
            else:
                logger.info("No model_path => fallback logic if GPT not used.")

        # GPT setup
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        from openai import OpenAI
        self.client = OpenAI(api_key=openai_api_key)
        logger.info("OpenAI client instantiated in AIStrategy.")

        # RiskManager for DB-based sub-positions
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
        Also forcibly calls check_stop_loss_take_profit for the given pair.
        Then tries GPT, scikit, or fallback logic.
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

        # GPT approach:
        if self.use_openai:
            try:
                return self._openai_inference(market_data)
            except Exception as e:
                logger.exception(f"OpenAI inference failed: {e}")
                return self._dummy_logic(market_data)
        # scikit approach:
        elif self.model:
            try:
                return self._model_inference_realtime(market_data)
            except Exception as e:
                logger.exception(f"Scikit inference error: {e}")
                return self._dummy_logic(market_data)
        # fallback:
        else:
            return self._dummy_logic(market_data)

    # --------------------------------------------------------------------------
    # GPT-based approach
    # --------------------------------------------------------------------------
    def _openai_inference(self, market_data: dict):
        pair = market_data.get("pair", "UNKNOWN")
        price = market_data.get("price", 0.0)
        cpanic = market_data.get("cryptopanic_sentiment", 0.0)
        galaxy = market_data.get("galaxy_score", 0.0)
        alt_rank = market_data.get("alt_rank", 0)
        base_rationale = f"GPT-based => {pair}, price={price}, cpanic={cpanic}, galaxy={galaxy}, alt={alt_rank}"

        # Summarize last trades
        summary = self._summarize_recent_trades(pair, limit=3)

        # Build prompt with optional risk_controls from config
        system_msg = {
            "role": "system",
            "content": (
                "You are an advanced crypto trading assistant. The user provides aggregator data. "
                "You may produce function calls or normal text to convey a trade decision. "
                "Respect the user's numeric constraints for position sizing."
            )
        }

        # We'll inject risk controls if present:
        rc = self.risk_controls
        user_text = (
            f"GPT context:\n{self.gpt_context}\n\n"
            f"Aggregator:\n  pair={pair}, price={price}, cryptopanic={cpanic}, galaxy={galaxy}, alt_rank={alt_rank}\n"
            f"Trades:\n{summary}\n"
        )
        if rc:
            user_text += (
                "\nConstraints:\n"
                f"  initial_spending_account={rc.get('initial_spending_account',0.0)}\n"
                f"  purchase_upper_limit_percent={rc.get('purchase_upper_limit_percent',0.0)}\n"
                f"  minimum_buy_amount={rc.get('minimum_buy_amount',0.0)}\n"
                f"  max_position_value={rc.get('max_position_value',0.0)}\n"
                "In other words, do not produce a BUY that costs more than (purchase_upper_limit_percent * initial_spending_account), "
                "nor less than minimum_buy_amount, etc.\n"
            )
        user_text += (
            "\nDecide on a single action: BUY, SELL, or HOLD. If BUY => show 'size'. If SELL => show 'size'. "
            "If HOLD => size=0.\n"
        )

        user_msg = {"role": "user", "content": user_text}

        # Call openai
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[system_msg, user_msg],
            temperature=0.0,
            max_tokens=400
        )
        logger.debug(f"GPT response: {response}")

        if not response.choices:
            rationale = f"No GPT choices => fallback. {base_rationale}"
            self._store_decision(pair, "HOLD", 0.0, rationale)
            return ("HOLD", 0.0)

        choice = response.choices[0]
        finish_reason = choice.finish_reason
        msg = choice.message  # a ChatCompletionMessage

        # If GPT tries function calls:
        if finish_reason in ("tool_calls", "function_call"):
            fn_call = getattr(msg, "function_call", None)
            if not fn_call:
                # no actual call => fallback
                rationale = f"No function_call => fallback. {base_rationale}"
                self._store_decision(pair, "HOLD", 0.0, rationale)
                return ("HOLD", 0.0)

            fn_args_str = fn_call.arguments
            try:
                parsed_args = json.loads(fn_args_str)
            except json.JSONDecodeError:
                rationale = f"Cannot parse function args => fallback. {base_rationale}"
                self._store_decision(pair, "HOLD", 0.0, rationale)
                return ("HOLD", 0.0)

            action = parsed_args.get("action", "HOLD").upper()
            size_suggested = float(parsed_args.get("size", 0.0))

            # post-validate
            final_signal, final_size = self._post_validate_and_adjust(action, size_suggested, price)

            # Then call DB
            final_signal, final_size = self.risk_manager_db.adjust_trade(
                final_signal, final_size, pair, price
            )

            # record final
            full_rationale = f"GPT function_call => {fn_call.name}, {base_rationale}"
            if final_signal in ("BUY", "SELL") and final_size > 0:
                record_trade_in_db(final_signal, final_size, price, "GPT_DECISION", pair)
            self._store_decision(pair, final_signal, final_size, full_rationale)
            self._append_gpt_context(f"fn_call => {fn_call.name}, action={final_signal}, size={final_size}")
            return (final_signal, final_size)

        else:
            # normal text
            content_text = msg.content or ""
            if "BUY" in content_text.upper():
                action = "BUY"
                size_suggested = 0.0005
            elif "SELL" in content_text.upper():
                action = "SELL"
                size_suggested = 0.0005
            else:
                action = "HOLD"
                size_suggested = 0.0

            # post-validate
            final_signal, final_size = self._post_validate_and_adjust(action, size_suggested, price)

            # risk manager
            final_signal, final_size = self.risk_manager_db.adjust_trade(
                final_signal, final_size, pair, price
            )

            rationale = f"GPT text => action={final_signal}, size={final_size}. {base_rationale}"
            if final_signal in ("BUY", "SELL") and final_size > 0:
                record_trade_in_db(final_signal, final_size, price, "GPT_DECISION_TEXT", pair)
            self._store_decision(pair, final_signal, final_size, rationale)
            self._append_gpt_context(f"GPT text => action={final_signal}, size={final_size}")
            return (final_signal, final_size)

    def _post_validate_and_adjust(self, action: str, size_suggested: float, current_price: float):
        """
        Ensures GPT suggestion doesn't exceed or drop below constraints
        from self.risk_controls. For example, do not allow trade cost < minimum_buy_amount
        or > purchase_upper_limit. Return the final (action, size).
        """
        if action not in ("BUY", "SELL"):
            # For HOLD or unknown => size=0
            return ("HOLD", 0.0)

        cost = size_suggested * current_price
        rc = self.risk_controls

        # If risk_controls is empty, skip
        if not rc:
            # no constraints => pass through
            return (action, size_suggested)

        min_buy = rc.get("minimum_buy_amount", 0.0)
        upper_percent = rc.get("purchase_upper_limit_percent", 1.0)
        account_total = rc.get("initial_spending_account", 9999999.0)
        purchase_upper = account_total * upper_percent

        if action == "BUY":
            # check min
            if cost < min_buy:
                logger.info(f"GPT suggests BUY => cost={cost} < {min_buy}, forcing HOLD instead.")
                return ("HOLD", 0.0)
            # check max
            if cost > purchase_upper:
                # clamp or hold
                new_size = purchase_upper / max(current_price, 1e-9)
                logger.info(
                    f"GPT suggests BUY => cost={cost} > {purchase_upper}, clamping size to {new_size}"
                )
                return ("BUY", new_size)

        # For SELL, we might do a partial exit logic or let it pass
        return (action, size_suggested)

    # --------------------------------------------------------------------------
    # scikit approach
    # --------------------------------------------------------------------------
    def _model_inference_realtime(self, market_data: dict):
        pair = market_data.get("pair", self.pairs[-1])
        snap_price = market_data.get("price", 0.0)
        if snap_price <= 0.0:
            rationale = f"No valid price => scikit fallback => HOLD for {pair}"
            self._store_decision(pair, "HOLD", 0.0, rationale)
            return ("HOLD", 0.0)

        df_recent = self._fetch_recent_price_data(pair, limit=50)
        if df_recent.empty:
            rationale = f"No data => scikit => HOLD for {pair}"
            self._store_decision(pair, "HOLD", 0.0, rationale)
            return ("HOLD", 0.0)

        # Optionally load BTC for correlation
        btc_df = None
        if pair != "XBT/USD":
            btc_df = self._fetch_recent_price_data("XBT/USD", limit=50)

        df_ind = self._compute_indicators(df_recent, btc_df)
        df_ind["avg_sentiment"] = market_data.get("cryptopanic_sentiment", 0.0)

        latest = df_ind.iloc[[-1]].copy()
        for col in TRAIN_FEATURE_COLS:
            if col not in latest.columns:
                latest[col] = 0.0

        X_in = latest[TRAIN_FEATURE_COLS]
        prob_up = 0.5
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(X_in)[0]
            if len(p) > 1:
                prob_up = p[1]
        else:
            label = self.model.predict(X_in)[0]
            prob_up = 1.0 if label == 1 else 0.0

        rationale = f"scikit => prob_up={prob_up:.2f} for {pair}"
        action = "HOLD"
        size_suggested = 0.0
        if prob_up > 0.6:
            action = "BUY"
            size_suggested = 0.0005
        elif prob_up < 0.4:
            action = "SELL"
            size_suggested = 0.0005

        # post-validate
        final_signal, final_size = self._post_validate_and_adjust(action, size_suggested, snap_price)
        final_signal, final_size = self.risk_manager_db.adjust_trade(
            final_signal, final_size, pair, snap_price
        )

        if final_signal in ("BUY", "SELL") and final_size > 0:
            record_trade_in_db(final_signal, final_size, snap_price, "MODEL_DECISION", pair)
        rationale += f" => final={final_signal}, size={final_size}"
        self._store_decision(pair, final_signal, final_size, rationale)
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
            final_signal, final_size = self.risk_manager_db.adjust_trade("BUY", 0.0005, pair, px)
            rationale = f"Dummy => BUY for price < 20000 => {final_size}"
            if final_signal == "BUY" and final_size > 0:
                record_trade_in_db(final_signal, final_size, px, "DUMMY_DECISION", pair)
            self._store_decision(pair, final_signal, final_size, rationale)
            return (final_signal, final_size)
        else:
            rationale = "Dummy => HOLD for price >= 20000"
            self._store_decision(pair, "HOLD", 0.0, rationale)
            return ("HOLD", 0.0)

    # --------------------------------------------------------------------------
    # Data / Indicators
    # --------------------------------------------------------------------------
    def _fetch_recent_price_data(self, pair: str, limit=50) -> pd.DataFrame:
        conn = sqlite3.connect(DB_FILE)
        try:
            q = f"""
                SELECT
                  timestamp, pair, bid_price, ask_price, last_price, volume
                FROM price_history
                WHERE pair=?
                ORDER BY id DESC
                LIMIT ?
            """
            df = pd.read_sql_query(q, conn, params=(pair, limit))
            # reverse so earliest is first
            df = df.iloc[::-1].reset_index(drop=True)
            return df
        except Exception as e:
            logger.exception(f"Error loading price_history for {pair}: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def _compute_indicators(self, df: pd.DataFrame, df_btc: pd.DataFrame = None) -> pd.DataFrame:
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

        # fill
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
