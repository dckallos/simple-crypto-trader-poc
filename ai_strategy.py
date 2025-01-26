# ==============================================================================
# FILE: ai_strategy.py
# ==============================================================================
"""
ai_strategy.py

A production-ready AIStrategy that:
1) Persists sub-positions in the DB via RiskManagerDB (risk_manager.py).
2) Loads/stores GPT conversation context in DB (table 'ai_context').
3) Incorporates aggregator data (CryptoPanic, LunarCrush, advanced price features, etc.)
   in GPT-based logic, passing them to the prompt for reasoning.
4) Supports partial entries/exits, cost-basis recalculation, flipping from long to short,
   all at the sub-position level (stored in DB).
5) Logs each final decision in 'ai_decisions'.

We now accept a more data-rich approach, e.g., passing advanced metrics from your
time-series (Bollinger, RSI, sentiment rolls, galaxy_score, alt_rank, etc.) directly
in `market_data`, so GPT can see them.

A local test approach is also shown at the bottom, where you can feed a “dummy” aggregator
dict and see the resulting action/size from `_full_gpt_inference` or fallback logic.

Usage:
    from ai_strategy import AIStrategy

    strat = AIStrategy([...], use_openai=True, risk_controls=...)
    result = strat.predict({
       "pair":"ETH/USD",
       "price":1870.0,
       "boll_upper":1900.2,
       "rsi":45.8,
       "galaxy_score":68,
       "avg_sentiment":0.2,
       ...
    })
    print(result)
"""

import logging
import os
import time
import json
import sqlite3
import numpy as np
import pandas as pd
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

class AIStrategy:
    """
    AIStrategy class that decides whether to BUY, SELL, or HOLD using GPT-based logic
    (with aggregator data) or a simple fallback. Also calls stop-loss checks.

    We store sub-positions in DB (via risk_manager_db). GPT conversation context
    is also loaded/persisted in the 'ai_context' table.
    """

    def __init__(
        self,
        pairs=None,
        # model_path was for scikit, can keep for backward compat or remove:
        model_path: Optional[str] = None,
        use_openai: bool = False,
        max_position_size: float = 0.001,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.01,
        max_daily_drawdown: float = -0.02,
        risk_controls: Optional[Dict[str, Any]] = None
    ):
        """
        :param pairs: e.g. ["ETH/USD","XBT/USD"] => list of pairs you trade
        :param model_path: scikit fallback, unused if we do GPT-based logic
        :param use_openai: bool => if True, use GPT approach
        :param max_position_size: clamp size for each trade
        :param stop_loss_pct: e.g. 5% => auto-close sub-position if -5%
        :param take_profit_pct: e.g. 1% => auto-close if +1%
        :param max_daily_drawdown: daily realized pnl limit
        :param risk_controls: dict => e.g. { "minimum_buy_amount":10.0, "max_position_value":100.0, ... }
        """
        self.pairs = pairs if pairs else ["XBT/USD"]
        self.use_openai = use_openai
        self.risk_controls = risk_controls or {}

        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY","")
        # if no openai_api_key => possibly fallback or raise
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

        # GPT context
        self.gpt_context = self._init_gpt_context()

        # ensure ai_decisions table
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
        Appends new text => merges into the single stored context.
        Possibly summarize older context if needed in future.
        """
        self.gpt_context += "\n" + new_text
        self._save_gpt_context(self.gpt_context)

    # --------------------------------------------------------------------------
    # 'ai_decisions' table
    # --------------------------------------------------------------------------
    def _create_ai_decisions_table(self):
        conn=sqlite3.connect(DB_FILE)
        try:
            c=conn.cursor()
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
            logger.exception(f"Error creating ai_decisions => {e}")
        finally:
            conn.close()

    def _store_decision(self, pair: str, action: str, size: float, rationale: str):
        logger.debug(f"Storing AI => {pair},{action},{size}, reason={rationale}")
        conn=sqlite3.connect(DB_FILE)
        try:
            c=conn.cursor()
            c.execute("""
            INSERT INTO ai_decisions (timestamp, pair, action, size, rationale)
            VALUES (?, ?, ?, ?, ?)
            """,(int(time.time()), pair, action, size, rationale))
            conn.commit()
        except Exception as e:
            logger.exception(f"Error storing AI decision => {e}")
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # MAIN predict
    # --------------------------------------------------------------------------
    def predict(self, market_data: dict):
        """
        The main interface. Takes a dictionary like:
           {
             "pair":"ETH/USD",
             "price":1870.5,
             "rsi":50.3,
             "boll_lower":1850.1,
             "boll_upper":1902.3,
             "galaxy_score":70.0,
             "alt_rank": 300,
             "avg_sentiment": 0.1,
             ...
           }
        Then calls check_stop_loss_take_profit, then GPT or fallback logic.

        Return => (action, size).
        """
        pair = market_data.get("pair", self.pairs[-1])
        current_price = market_data.get("price", 0.0)
        if current_price<=0:
            rationale=f"No valid price => HOLD {pair}"
            self._store_decision(pair,"HOLD",0.0,rationale)
            logger.warning(rationale)
            return ("HOLD",0.0)

        # forcibly check for sub-position stop/take
        self.risk_manager_db.check_stop_loss_take_profit(pair, current_price)

        if self.use_openai:
            try:
                return self._full_gpt_inference(market_data)
            except Exception as e:
                logger.exception(f"[AIStrategy] GPT inference failed => {e}")
                return self._dummy_logic(market_data)
        else:
            return self._dummy_logic(market_data)

    def _full_gpt_inference(self, market_data: dict):
        """
        GPT approach => pass aggregator data & constraints => parse JSON => post-validate
        This updated version clarifies that we only want raw JSON, with no code blocks.
        """

        pair = market_data.get("pair", "UNK")
        px = market_data.get("price", 0.0)

        # We'll build a text summary for aggregator fields:
        aggregator_text = ""
        for k, v in market_data.items():
            if k not in ("pair", "price"):
                aggregator_text += f"{k}={v}, "
        aggregator_text += f"price={px}"

        # Summarize trades:
        trade_summary = self._summarize_recent_trades(pair)

        # risk controls:
        rc = self.risk_controls

        # System or "developer" prompt clarifying that *no* triple backticks or code blocks are allowed:
        dev_msg = {
            "role": "assistant",
            "content": (
                "You are an advanced crypto trading assistant. "
                "You MUST output valid JSON with the form {\"action\":\"BUY|SELL|HOLD\",\"size\":float} "
                "and NOTHING else. No triple backtick code fences. No markdown. "
                "Example of correct output: {\"action\":\"BUY\",\"size\":0.001}."
            )
        }

        # The user prompt includes aggregator data, recent trades, risk constraints:
        user_text = (
            f"GPT context so far:\n{self.gpt_context}\n\n"
            f"Aggregators => {aggregator_text}\n"
            f"Recent trades => {trade_summary}\n"
            f"Constraints => {rc}\n"
            "Respond ONLY with a raw JSON object: {\"action\":\"BUY|SELL|HOLD\",\"size\":float}."
        )
        user_msg = {"role": "user", "content": user_text}

        logger.debug(f"[GPT] user_msg => {user_text}")

        # Example OpenAI client usage:
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",  # or your preferred model
            messages=[dev_msg, user_msg],
            temperature=1.0,
            max_completion_tokens=2000
        )

        logger.debug(f"[GPT] raw response => {resp}")
        if not resp.choices:
            rationale = "No GPT choices => fallback => HOLD"
            self._store_decision(pair, "HOLD", 0.0, rationale)
            return ("HOLD", 0.0)

        choice = resp.choices[0]
        msg_content = choice.message.content.strip()
        logger.info(f"[GPT] finish_reason={choice.finish_reason}")

        # Parse JSON from the raw content, ensuring we handle it if the user tries to slip in a code block:
        try:
            # If there's a chance the model might still include backticks, strip them out:
            cleaned = msg_content.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("[GPT] parse error => fallback => HOLD")
            parsed = {}

        action = parsed.get("action", "HOLD").upper()
        suggested_size = float(parsed.get("size", 0.0))

        # post-validate, risk manage:
        final_signal, final_size = self._post_validate(action, suggested_size, px)
        final_signal, final_size = self.risk_manager_db.adjust_trade(final_signal, final_size, pair, px)

        # store final decision:
        rationale = (
            f"GPT => final={final_signal}, size={final_size}. aggregator={aggregator_text}"
        )
        if final_signal in ("BUY", "SELL") and final_size > 0:
            record_trade_in_db(final_signal, final_size, px, "GPT_DECISION", pair)

        self._store_decision(pair, final_signal, final_size, rationale)
        self._append_gpt_context(rationale)
        return (final_signal, final_size)

    def _dummy_logic(self, market_data: dict):
        """
        If price < 20000 => BUY 0.0005, else HOLD
        """
        pair = market_data.get("pair","XBT/USD")
        px   = market_data.get("price",0.0)
        if px<20000:
            sig, sz = self._post_validate("BUY",0.0005,px)
            sig, sz = self.risk_manager_db.adjust_trade(sig,sz,pair,px)
            rationale=f"[DUMMY] => px={px}<20000 => {sig} {sz}"
            if sig=="BUY" and sz>0:
                record_trade_in_db(sig,sz,px,"DUMMY_DECISION",pair)
            self._store_decision(pair, sig, sz, rationale)
            return (sig,sz)
        else:
            rationale=f"[DUMMY] => px={px}>=20000 => HOLD"
            self._store_decision(pair,"HOLD",0.0,rationale)
            return ("HOLD",0.0)

    def _post_validate(self, action: str, size_suggested: float, current_price: float):
        """
        Check risk controls => skip or clamp if needed.
        If not BUY/SELL => => HOLD,0
        """
        if action not in ("BUY","SELL"):
            return ("HOLD",0.0)

        cost = size_suggested*current_price
        rc = self.risk_controls
        if not rc:
            return (action, size_suggested)

        min_buy = rc.get("minimum_buy_amount",6.0)
        upper_pct = rc.get("purchase_upper_limit_percent",25.0)
        init_acct = rc.get("initial_spending_account",50.0)
        purchase_upper = init_acct*upper_pct

        if action=="BUY":
            if cost<min_buy:
                logger.info(f"[post_validate] cost={cost}<min_buy={min_buy}, => hold")
                return ("HOLD",0.0)
            if cost>purchase_upper:
                new_size = purchase_upper/max(1e-9, current_price)
                logger.info(f"[post_validate] cost={cost}>purchase_upper={purchase_upper}, => clamp size={new_size}")
                return ("BUY", new_size)

        return (action, size_suggested)

    def _summarize_recent_trades(self, pair: str, limit=3) -> str:
        """
        Return a short text summarizing last N trades from 'trades' table for the pair.
        """
        conn=sqlite3.connect(DB_FILE)
        out=[]
        try:
            c=conn.cursor()
            c.execute("""
            SELECT timestamp, side, quantity, price
            FROM trades
            WHERE pair=?
            ORDER BY id DESC
            LIMIT ?
            """,(pair,limit))
            rows=c.fetchall()
            if not rows:
                return "No trades found."
            for i,row in enumerate(rows[::-1],start=1):
                t,sd,qty,px=row
                import datetime
                dt_s=datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")
                out.append(f"{i}) {dt_s} {sd} {qty}@{px}")
        except Exception as e:
            logger.exception(f"Error summarizing trades => {e}")
            return "Error retrieving trades."
        finally:
            conn.close()
        return "\n".join(out)


# ------------------------------------------------------------------------------
# Example local test usage
# ------------------------------------------------------------------------------
if __name__=="__main__":
    """
    We'll show how you might run a quick local test with a dummy aggregator dictionary.
    This won't call real GPT unless you have API keys set. It's just an example of usage.
    """
    # 1) Construct an AIStrategy instance
    strat = AIStrategy(
        pairs=["ETH/USD"],
        use_openai=True,  # or True if you want to test GPT
        risk_controls={
            "initial_spending_account": 100.0,
            "purchase_upper_limit_percent": 0.1,
            "minimum_buy_amount": 5.0,
            "max_position_value": 50.0
        }
    )

    # 2) Create a mock aggregator dict that might represent the latest known data
    mock_data={
      "pair":"ETH/USD",
      "price":1870.0,
      "rsi":49.3,
      "boll_lower":1850.2,
      "boll_upper":1920.1,
      "galaxy_score":70.2,
      "alt_rank": 350,
      "avg_sentiment": 0.15
    }

    action, size = strat.predict(mock_data)
    print(f"Local test => action={action}, size={size}")
