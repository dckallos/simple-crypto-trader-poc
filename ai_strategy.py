"""
ai_strategy.py

An enhanced, production-ready AIStrategy module that can:
1) Use GPT logic (via GPTManager) or fallback logic for a single trading pair.
2) Optionally handle multiple trading pairs concurrently (multi-thread or async approach),
   fetching trade histories and aggregator data in parallel and then making
   a single or multiple GPT calls to generate buy/sell/hold decisions at once.
3) Store sub-positions in a DB-based RiskManager, enforcing daily drawdown limits,
   position sizing, cost constraints, etc.
4) Log each final decision in 'ai_decisions' for auditing.

Requires:
    - db.py (for DB_FILE, init_db, store_decisions, etc.)
    - risk_manager.py (for RiskManagerDB)
    - gpt_manager.py (for GPTManager)
    - openai >= v1 library

High-Level Flow:
    AIStrategy can call:
        - predict(market_data)      -> single pair
        - predict_multi(pairs_data) -> multiple pairs at once
          - optionally gather aggregator info/trade summaries in parallel
          - pass all to GPT or one-by-one
          - store final decisions
"""

import logging
import os
import time
import json
import sqlite3
from typing import Optional, Dict, Any, List, Tuple, Union

# We assume these modules exist in your codebase:
from db import (
    DB_FILE,
    load_gpt_context_from_db,
    save_gpt_context_to_db,
    record_trade_in_db
)
from risk_manager import RiskManagerDB
from gpt_manager import GPTManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class AIStrategy:
    """
    AIStrategy: A flexible class that decides trades for one or multiple pairs,
    using GPT or fallback logic. Integrates with a RiskManager for sub-positions
    and daily drawdown constraints.
    """

    def __init__(
        self,
        pairs: List[str] = None,
        use_openai: bool = False,
        max_position_size: float = 0.001,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.01,
        max_daily_drawdown: float = -0.02,
        risk_controls: Optional[Dict[str, Any]] = None,
        gpt_model: str = "gpt-4o",
        gpt_temperature: float = 0.7,
        gpt_max_tokens: int = 500,
        **gpt_client_options
    ):
        """
        :param pairs: A list of trading pairs, e.g. ["ETH/USD","XBT/USD"], if you want a default set.
        :param use_openai: If True, uses GPT logic. Otherwise fallback/dummy logic.
        :param max_position_size: A clamp on per-trade size in base currency.
        :param stop_loss_pct: e.g. 0.05 => auto-close if -5%.
        :param take_profit_pct: e.g. 0.01 => auto-close if +1%.
        :param max_daily_drawdown: e.g. -0.02 => skip new trades if daily < -2%.
        :param risk_controls: Additional constraints (e.g. min buy, purchase_upper, etc.).
        :param gpt_model: GPT model name if use_openai is True.
        :param gpt_temperature: GPT temperature for creativity.
        :param gpt_max_tokens: Max tokens for GPT responses.
        :param gpt_client_options: Additional kwargs for the GPTManager's OpenAI client
                                   (e.g. max_retries, timeout, proxies).
        """
        self.pairs = pairs if pairs else []
        self.use_openai = use_openai
        self.risk_controls = risk_controls or {}

        # Create a GPTManager if needed
        self.gpt_manager = None
        if self.use_openai:
            self.gpt_manager = GPTManager(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                model=gpt_model,
                temperature=gpt_temperature,
                max_tokens=gpt_max_tokens,
                **gpt_client_options
            )

        # RiskManager for sub-positions
        self.risk_manager_db = RiskManagerDB(
            db_path=DB_FILE,
            max_position_size=max_position_size,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_daily_drawdown=max_daily_drawdown
        )
        self.risk_manager_db.initialize()

        # GPT conversation context (if we want persistent memory)
        self.gpt_context = self._init_gpt_context()

        # create ai_decisions table if not exist
        self._create_ai_decisions_table()

    # --------------------------------------------------------------------------
    # SINGLE-PAIR PREDICT
    # --------------------------------------------------------------------------
    def predict(self, market_data: Dict[str, Any]) -> Tuple[str, float]:
        """
        Main interface for a single trading pair. Return => (action, size).

        Example market_data:
            {
              "pair": "ETH/USD",
              "price": 1870.0,
              "rsi": 44.2,
              "boll_upper": 1900.2,
              "boll_lower": 1850.1,
              ...
            }
        Steps:
            1) risk_manager_db.check_stop_loss_take_profit()
            2) if use_openai => _gpt_flow(...) else => _fallback_logic(...)
            3) store decision in ai_decisions table
        """
        pair = market_data.get("pair", "UNK")
        current_price = market_data.get("price", 0.0)
        if current_price <= 0:
            rationale = f"No valid price => HOLD {pair}"
            self._store_decision(pair, "HOLD", 0.0, rationale)
            return ("HOLD", 0.0)

        # forcibly check for sub-position stop/take
        self.risk_manager_db.check_stop_loss_take_profit(pair, current_price)

        if self.use_openai and self.gpt_manager:
            try:
                return self._gpt_flow_single(pair, market_data)
            except Exception as e:
                logger.exception(f"[AIStrategy] GPT inference failed => fallback => {e}")
                return self._fallback_logic(pair, market_data)
        else:
            return self._fallback_logic(pair, market_data)

    def _gpt_flow_single(self, pair: str, market_data: Dict[str, Any]) -> Tuple[str, float]:
        """
        Single-pair GPT logic. Builds aggregator text + recent trade summary,
        calls GPT, runs post-validation + risk checks, logs final.
        """
        px = market_data["price"]

        aggregator_text = self._build_aggregator_text(market_data, exclude=["pair", "price"])
        # Summarize last N trades from DB
        trade_summary = self._summarize_recent_trades(pair, limit=3)

        # We'll feed the entire conversation context
        # and the aggregator/trade summary/risk.
        # For single-pair usage, we treat trade_history as just lines of text:
        trade_history_list = trade_summary.split("\n") if trade_summary else []
        # For demonstration, we pass max_trades=3
        result = self.gpt_manager.generate_trade_decision(
            conversation_context=self.gpt_context,
            aggregator_text=aggregator_text,
            trade_history=trade_history_list,
            max_trades=3,
            risk_controls=self.risk_controls
        )
        action = result.get("action", "HOLD").upper()
        suggested_size = float(result.get("size", 0.0))

        final_signal, final_size = self._post_validate(action, suggested_size, px)
        final_signal, final_size = self.risk_manager_db.adjust_trade(
            final_signal, final_size, pair, px
        )

        rationale = (
            f"GPT => final={final_signal}, size={final_size}, aggregator={aggregator_text}"
        )
        if final_signal in ("BUY", "SELL") and final_size > 0:
            # record in trades table
            record_trade_in_db(final_signal, final_size, px, "GPT_DECISION", pair)

        # store in ai_decisions
        self._store_decision(pair, final_signal, final_size, rationale)

        # update GPT context
        self._append_gpt_context(rationale)

        return (final_signal, final_size)

    def _fallback_logic(self, pair: str, market_data: Dict[str, Any]) -> Tuple[str, float]:
        """
        Simple fallback if GPT is disabled or fails. Example:
          If price < 20000 => BUY 0.0005 else HOLD
        """
        px = market_data.get("price", 0.0)
        if px < 20000:
            sig, sz = self._post_validate("BUY", 0.0005, px)
            sig, sz = self.risk_manager_db.adjust_trade(sig, sz, pair, px)
            rationale = f"[DUMMY] => px={px} < 20000 => {sig} {sz}"
            if sig == "BUY" and sz > 0:
                record_trade_in_db(sig, sz, px, "DUMMY_DECISION", pair)
            self._store_decision(pair, sig, sz, rationale)
            return (sig, sz)
        else:
            rationale = f"[DUMMY] => px={px} >= 20000 => HOLD"
            self._store_decision(pair, "HOLD", 0.0, rationale)
            return ("HOLD", 0.0)

    # --------------------------------------------------------------------------
    # MULTI-PAIR WORKFLOW (CONCURRENT or ASYNC)
    # --------------------------------------------------------------------------
    def predict_multi(self, pairs_data: List[Dict[str, Any]], concurrency="thread") -> Dict[str, Tuple[str, float]]:
        """
        A method to handle multiple pairs at once, possibly in parallel.

        :param pairs_data: A list of dicts, each containing aggregator data like:
            [
              {"pair":"ETH/USD","price":1870,"rsi":44,"boll_upper":1900,...},
              {"pair":"XBT/USD","price":28000,"volume":1000,...},
              ...
            ]
        :param concurrency: "thread", "asyncio", or "none"
                           (example placeholder for your concurrency approach)

        Return: A dict mapping pair => (action, size)

        Steps (schematic):
          1) Possibly fetch aggregator data/trade history concurrently for each pair.
          2) Either call GPT once with the combined data or call GPT separately for each pair.
          3) Return final decisions in a dict.
        """
        if not pairs_data:
            logger.warning("No pairs_data provided => nothing to do => return empty.")
            return {}

        # 1) Check concurrency approach. We'll demonstrate a simplified approach here,
        #    but in real usage, you'd use concurrent.futures or asyncio.gather.
        if concurrency == "thread":
            # We'll do a naive synchronous approach for demonstration.
            # You can adapt to concurrent.futures.ThreadPoolExecutor if you want real parallel.
            logger.info("Predicting multiple pairs in naive synchronous approach (concurrency=thread).")

        elif concurrency == "asyncio":
            # This is where you might do: await gather(*[self._async_predict(p) for p in pairs_data])
            logger.info("Predicting multiple pairs in an async approach (placeholder).")
            # For actual usage, you'd define an async method _async_predict(...) and gather them.
            pass

        decisions = {}
        for pd in pairs_data:
            pair = pd.get("pair","UNK")
            # forcibly check stops
            px = pd.get("price",0.0)
            if px>0:
                self.risk_manager_db.check_stop_loss_take_profit(pair, px)

            # now do GPT or fallback
            if self.use_openai and self.gpt_manager:
                # We'll do a single call for each pair or optionally do a single GPT call for all pairs
                # But let's keep it simple => one GPT call per pair.
                action, size = self._gpt_flow_single(pair, pd)
            else:
                action, size = self._fallback_logic(pair, pd)

            decisions[pair] = (action, size)
        return decisions

    # Alternatively, if you want a single GPT call for all pairs at once:
    #   you might gather aggregator data for each pair => build a large prompt => call GPT => parse multiple decisions.
    # That approach is more advanced and requires carefully instructing GPT to output structured JSON
    # for each pair.

    # --------------------------------------------------------------------------
    # GPT OR FALLBACK HELPER LOGIC
    # --------------------------------------------------------------------------
    def _build_aggregator_text(self, market_data: Dict[str, Any], exclude: List[str] = None) -> str:
        """
        Build aggregator text from market_data, skipping keys in exclude.
        Example: rsi=44.2, boll_upper=1900.2, ...
        """
        if exclude is None:
            exclude = []
        parts = []
        for k, v in market_data.items():
            if k not in exclude:
                parts.append(f"{k}={v}")
        return ", ".join(parts)

    def _post_validate(self, action: str, size_suggested: float, current_price: float) -> Tuple[str, float]:
        """
        Check local constraints. For example, if cost < minimal, we do HOLD.
        """
        if action not in ("BUY", "SELL"):
            return ("HOLD", 0.0)

        cost = size_suggested * current_price
        rc = self.risk_controls
        if not rc:
            return (action, size_suggested)

        min_buy = rc.get("minimum_buy_amount", 6.0)
        if action == "BUY" and cost < min_buy:
            logger.info(f"[post_validate] cost={cost:.2f} < min_buy={min_buy:.2f} => hold")
            return ("HOLD", 0.0)

        return (action, size_suggested)

    # --------------------------------------------------------------------------
    # GPT CONTEXT PERSISTENCE
    # --------------------------------------------------------------------------
    def _init_gpt_context(self) -> str:
        """
        Load GPT context from DB if any. Return the string or empty.
        """
        data = load_gpt_context_from_db()
        if data:
            logger.info("Loaded GPT context from DB.")
            return data
        return ""

    def _append_gpt_context(self, new_text: str) -> None:
        """
        Append a new line to the GPT context, then save.
        You might want to limit size eventually for cost reasons.
        """
        self.gpt_context += "\n" + new_text
        save_gpt_context_to_db(self.gpt_context)

    # --------------------------------------------------------------------------
    # AI_DECISIONS TABLE
    # --------------------------------------------------------------------------
    def _create_ai_decisions_table(self):
        """
        Ensures we have a local table for storing decisions if not exist.
        """
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
            logger.exception(f"Error creating ai_decisions => {e}")
        finally:
            conn.close()

    def _store_decision(self, pair: str, action: str, size: float, rationale: str):
        """
        Store the final decision in the 'ai_decisions' table for auditing.
        """
        logger.debug(f"Storing AI => {pair},{action},{size}, reason={rationale}")
        conn = sqlite3.connect(DB_FILE)
        try:
            c = conn.cursor()
            c.execute("""
                INSERT INTO ai_decisions (timestamp, pair, action, size, rationale)
                VALUES (?, ?, ?, ?, ?)
            """, (int(time.time()), pair, action, size, rationale))
            conn.commit()
        except Exception as e:
            logger.exception(f"Error storing AI decision => {e}")
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # TRADE HISTORY SUMMARIES
    # --------------------------------------------------------------------------
    def _summarize_recent_trades(self, pair: str, limit: int = 3) -> str:
        """
        Return a string summarizing the last N trades from the 'trades' table for this pair.
        E.g.:
            1) 2025-01-20 10:15 BUY 0.001@25000
            2) 2025-01-21 11:20 SELL 0.0005@25500
            ...
        """
        conn = sqlite3.connect(DB_FILE)
        out = []
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
            # Reverse to chronological
            rows = rows[::-1]
            for i, row in enumerate(rows, start=1):
                t, sd, qty, px = row
                import datetime
                dt_s = datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")
                out.append(f"{i}) {dt_s} {sd} {qty}@{px}")
        except Exception as e:
            logger.exception(f"Error summarizing trades => {e}")
            return "Error retrieving trades."
        finally:
            conn.close()

        return "\n".join(out)


# ------------------------------------------------------------------------------
# Example usage snippet (not part of the class):
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    For demonstration or quick testing:
      python ai_strategy.py
    Make sure your environment has the DB and risk_manager set up.
    """
    logging.basicConfig(level=logging.INFO)

    # Suppose we initialize a strategy that can handle multiple pairs
    strategy = AIStrategy(
        pairs=["ETH/USD", "XBT/USD"],
        use_openai=True,
        risk_controls={
            "initial_spending_account": 100.0,
            "minimum_buy_amount": 5.0,
            "purchase_upper_limit_percent": 0.1,
        },
        gpt_model="gpt-4o-mini",
        gpt_temperature=0.8,
        gpt_max_tokens=500
    )

    # Single pair usage:
    single_data = {
        "pair": "ETH/USD",
        "price": 1850.0,
        "rsi": 45.0,
        "volume": 1000.0,
    }
    action, size = strategy.predict(single_data)
    print(f"[Single-Pair] => final action={action}, size={size}")

    # Multi pair usage:
    multi_data = [
        {"pair": "ETH/USD", "price": 1800.0, "rsi": 46.0, "volume": 1100.0},
        {"pair": "XBT/USD", "price": 28000.0, "rsi": 50.0, "volume": 500.0},
    ]
    results = strategy.predict_multi(multi_data, concurrency="thread")
    print("Multi-Pair decisions =>", results)
