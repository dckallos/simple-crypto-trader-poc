"""
ai_strategy.py

An enhanced, production-ready AIStrategy that can:
1) Use GPT logic (via GPTManager) or fallback logic for single or multiple coins in a single GPT call.
2) Optionally handle concurrency (multi-thread or async) to gather aggregator data for each pair,
   but crucially, it can pass **all** aggregator data to GPT in a single request to find the best
   decisions across multiple coins.
3) Store sub-positions in DB (RiskManagerDB), enforcing daily drawdown, position sizing, cost constraints, etc.
4) Log each final decision in 'ai_decisions' for auditing.
5) If GPT decides to SELL or SWAP a position, we handle that by calling the risk manager to close or reduce
   any existing sub-position for that pair.

Requirements:
    - db.py (DB_FILE, load_gpt_context_from_db, save_gpt_context_to_db, record_trade_in_db)
    - risk_manager.py (RiskManagerDB)
    - gpt_manager.py => now with generate_multi_trade_decision(...) for advanced multi-coin logic
    - openai >= v1 library

Basic Flow:
    - Single pair => call predict(market_data)
    - Multi pairs => call predict_multi(pairs_data)
      * If you want advanced one-shot GPT for all pairs, see the new method: predict_multi_coins(...)
"""

import logging
import os
import time
import json
import sqlite3
from typing import Optional, Dict, Any, List, Tuple, Union

from dotenv import load_dotenv

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

    Key Updates:
      - We can now gather aggregator data for all coins in a single cycle,
        call GPT once with generate_multi_trade_decision(...) from gpt_manager.py,
        and parse a multi-coin decision array.
      - If GPT decides to SELL or reduce a position, we attempt to close or reduce
        the corresponding sub-position in risk_manager_db.
    """

    def __init__(
        self,
        pairs: List[str] = None,
        use_openai: bool = False,
        max_position_size: float = 3,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.01,
        max_daily_drawdown: float = -0.02,
        risk_controls: Optional[Dict[str, Any]] = None,
        gpt_model: str = "gpt-4o",
        gpt_temperature: float = 1.0,
        gpt_max_tokens: int = 2000,
        **gpt_client_options
    ):
        """
        :param pairs: E.g. ["ETH/USD","XBT/USD"], if you want a default set
        :param use_openai: If True => GPT logic; else fallback logic
        :param max_position_size: clamp on per-trade size
        :param stop_loss_pct: e.g. 0.05 => auto-close if -5%
        :param take_profit_pct: e.g. 0.01 => auto-close if +1%
        :param max_daily_drawdown: e.g. -0.02 => skip new trades if daily < -2%
        :param risk_controls: dict of risk constraints (e.g. min buy, etc.)
        :param gpt_model: GPT model name
        :param gpt_temperature: GPT creativity
        :param gpt_max_tokens: max tokens in GPT responses
        :param gpt_client_options: Additional arguments for GPTManager's client,
                                   e.g. timeout, max_retries, proxies, etc.
        """
        self.pairs = pairs if pairs else []
        self.use_openai = use_openai
        self.risk_controls = risk_controls or {}

        load_dotenv()

        self.gpt_manager = None
        if self.use_openai:
            # Create GPTManager with multi-trade features
            self.gpt_manager = GPTManager(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                model=gpt_model,
                temperature=gpt_temperature,
                max_tokens=gpt_max_tokens,
                **gpt_client_options
            )

        # Risk manager for sub-positions
        self.risk_manager_db = RiskManagerDB(
            db_path=DB_FILE,
            max_position_size=max_position_size,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_daily_drawdown=max_daily_drawdown
        )
        self.risk_manager_db.initialize()

        # GPT context from DB (if using conversation memory)
        self.gpt_context = self._init_gpt_context()

        # ensure ai_decisions table
        self._create_ai_decisions_table()

    # --------------------------------------------------------------------------
    # SINGLE-PAIR PREDICT
    # --------------------------------------------------------------------------
    def predict(self, market_data: Dict[str, Any]) -> Tuple[str, float]:
        """
        Predict for a single pair => calls GPT or fallback. Return => (action, size).
        Steps:
          1) check stops
          2) GPT or fallback
          3) risk manager => store decision
        """
        pair = market_data.get("pair", "UNK")
        current_price = market_data.get("price", 0.0)
        if current_price <= 0:
            rationale = f"No valid price => HOLD {pair}"
            self._store_decision(pair, "HOLD", 0.0, rationale)
            return ("HOLD", 0.0)

        # forcibly check sub-position stops
        self.risk_manager_db.check_stop_loss_take_profit(pair, current_price)

        if self.use_openai and self.gpt_manager:
            try:
                return self._gpt_flow_single(pair, market_data)
            except Exception as e:
                logger.exception(f"[AIStrategy] GPT single inference failed => fallback => {e}")
                return self._fallback_logic(pair, market_data)
        else:
            return self._fallback_logic(pair, market_data)

    def _gpt_flow_single(self, pair: str, market_data: Dict[str, Any]) -> Tuple[str, float]:
        """
        Single-pair GPT logic. We build aggregator text, get 3 trades summary,
        pass them to generate_trade_decision(...) in GPT manager, parse result,
        risk-check, store decision, etc.
        """
        px = market_data["price"]
        aggregator_text = self._build_aggregator_text(market_data, exclude=["pair", "price"])
        trade_summary = self._summarize_recent_trades(pair, limit=3)
        trade_history_list = trade_summary.split("\n") if trade_summary else []

        # For single approach, we call the older "generate_trade_decision" from GPT manager
        result = self.gpt_manager.generate_trade_decision(
            conversation_context=self.gpt_context,
            aggregator_text=aggregator_text,
            trade_history=trade_history_list,
            max_trades=10,
            risk_controls=self.risk_controls,
            open_positions=None  # single approach => ignore for now
        )
        action = result.get("action", "HOLD").upper()
        suggested_size = float(result.get("size", 0.0))

        final_signal, final_size = self._post_validate(action, suggested_size, px)
        final_signal, final_size = self.risk_manager_db.adjust_trade(final_signal, final_size, pair, px)

        rationale = f"GPT single => final={final_signal}, size={final_size}, aggregator={aggregator_text}"
        if final_signal in ("BUY","SELL") and final_size > 0:
            record_trade_in_db(final_signal, final_size, px, "GPT_SINGLE_DECISION", pair)
        self._store_decision(pair, final_signal, final_size, rationale)
        self._append_gpt_context(rationale)

        return (final_signal, final_size)

    def _fallback_logic(self, pair: str, market_data: Dict[str, Any]) -> Tuple[str, float]:
        """
        Dummy fallback if GPT is off or fails. E.g. if price<20000 => BUY => 0.0005, else HOLD
        """
        px = market_data.get("price", 0.0)
        if px < 20000:
            sig, sz = self._post_validate("BUY", 0.0005, px)
            sig, sz = self.risk_manager_db.adjust_trade(sig, sz, pair, px)
            rationale = f"[DUMMY] => px={px} <20000 => {sig} {sz}"
            if sig=="BUY" and sz>0:
                record_trade_in_db(sig, sz, px, "DUMMY_SINGLE", pair)
            self._store_decision(pair, sig, sz, rationale)
            return (sig, sz)
        else:
            rationale = f"[DUMMY] => px={px} >=20000 => HOLD"
            self._store_decision(pair,"HOLD",0.0,rationale)
            return ("HOLD",0.0)

    # --------------------------------------------------------------------------
    # MULTI-PAIR - LEGACY
    # --------------------------------------------------------------------------
    def predict_multi(self, pairs_data: List[Dict[str, Any]], concurrency="thread") -> Dict[str, Tuple[str, float]]:
        """
        A legacy multi approach that calls GPT or fallback on each pair individually.
        If you want the advanced single-call approach, see `predict_multi_coins`.
        """
        if not pairs_data:
            logger.warning("No pairs_data provided => returning empty decisions.")
            return {}

        if concurrency=="thread":
            logger.info("[AIStrategy] multi => naive synchronous approach.")
        elif concurrency=="asyncio":
            logger.info("[AIStrategy] multi => placeholder for async approach.")
            # In real usage, you'd do concurrency with an event loop, gather calls, etc.

        decisions={}
        for pd in pairs_data:
            pair = pd.get("pair","UNK")
            px = pd.get("price",0.0)
            if px>0:
                self.risk_manager_db.check_stop_loss_take_profit(pair, px)

            if self.use_openai and self.gpt_manager:
                try:
                    action, size = self._gpt_flow_single(pair, pd)
                except Exception as e:
                    logger.exception(f"[AIStrategy] GPT single inference fail => fallback => {e}")
                    action, size = self._fallback_logic(pair, pd)
            else:
                action, size = self._fallback_logic(pair, pd)

            decisions[pair] = (action, size)
        return decisions

    # --------------------------------------------------------------------------
    # MULTI-PAIR - ADVANCED (All aggregator data => single GPT call)
    # --------------------------------------------------------------------------
    def predict_multi_coins(
        self,
        aggregator_list: List[Dict[str, Any]],
        trade_history: List[str],
        max_trades: int,
        open_positions: List[str]
    ) -> Dict[str, Tuple[str, float]]:
        """
        The new approach: gather aggregator data for all coins in aggregator_list,
        pass them to GPT in a single call => GPT sees entire market and open positions,
        and returns decisions for each coin in a single JSON.

        aggregator_list => e.g.
          [
            {"pair":"ETH/USD","aggregator_data":"rsi=44,price=1850,sentiment=0.4,..."},
            {"pair":"XBT/USD","aggregator_data":"rsi=50,price=28000,sentiment=0.3,..."}
          ]

        open_positions => list of strings describing all active positions:
          ["ETH/USD LONG 0.003@entry=1860.0", "XBT/USD SHORT 0.001@entry=29500.0"]

        We return a dict => { "ETH/USD":("BUY",0.001), "XBT/USD":("SELL",0.001) } etc.

        Steps:
          1) check stops for each aggregator => forcibly close if needed
          2) call generate_multi_trade_decision(...) once
          3) parse the array => for each pair => run post-validate + risk manager => store
          4) if GPT says SELL => we can interpret that as closing or reducing existing sub-position
        """
        results = {}
        if not aggregator_list:
            logger.warning("No aggregator_list => returning empty.")
            return {}

        # forcibly check sub-position stops for each coin
        for item in aggregator_list:
            pair = item.get("pair","UNK")
            # parse price from aggregator_data or from a separate field if you store it
            # or if aggregator_data is just a string, you might parse out price manually
            # for demonstration, let's do a naive approach:
            # if aggregator_data includes 'price=', we parse it (omitted for brevity).
            # We'll do an optional 'price' key
            # If you only store aggregator_data as text, you'd need a parse or store price separately
            maybe_price = item.get("price", 0.0)
            if maybe_price>0:
                self.risk_manager_db.check_stop_loss_take_profit(pair, maybe_price)

        if not self.use_openai or not self.gpt_manager:
            logger.info("[AIStrategy] no GPT => fallback => all hold")
            for it in aggregator_list:
                results[it["pair"]] = ("HOLD",0.0)
            return results

        # 2) single GPT call
        try:
            # We call new generate_multi_trade_decision
            # aggregator_list items must each have => "pair":"X/Y","aggregator_data":"some text"
            # we'll convert that text into e.g. "rsi=..., price=..., galaxy_score=..." as a single string
            # or do a read from item["aggregator_data"]. We assume the user code has built that string.

            # pass entire self.gpt_context as conversation_context
            # pass the open_positions
            # pass the user trade_history
            # pass risk_controls => self.risk_controls
            multi_resp = self.gpt_manager.generate_multi_trade_decision(
                conversation_context=self.gpt_context,
                aggregator_list=aggregator_list,
                open_positions=open_positions,
                trade_history=trade_history,
                max_trades=max_trades,
                risk_controls=self.risk_controls
            )
        except Exception as e:
            logger.exception("[AIStrategy] GPT multi inference fail => fallback => all hold")
            for it in aggregator_list:
                results[it["pair"]] = ("HOLD",0.0)
            return results

        # 3) parse the result => a dict with "decisions":[{pair,action,size}...]
        decisions_array = multi_resp.get("decisions", [])
        # e.g. decisions_array => [ {"pair":"ETH/USD","action":"BUY","size":0.001}, ...]

        # for each item => do post validation => risk => store
        for dec in decisions_array:
            pair = dec.get("pair","UNK")
            action_raw = dec.get("action","HOLD").upper()
            size_suggested = float(dec.get("size",0.0))

            # if aggregator_list includes a 'price' field for that pair:
            # We find it:
            found_item = next((x for x in aggregator_list if x.get("pair")==pair), None)
            if found_item and "price" in found_item:
                current_price = float(found_item["price"])
            else:
                # fallback => can't do a stop check => hold
                current_price = 0.0

            if current_price>0:
                # post validate
                final_signal, final_size = self._post_validate(action_raw, size_suggested, current_price)
                final_signal, final_size = self.risk_manager_db.adjust_trade(final_signal, final_size, pair, current_price)

                # store
                rationale = f"GPT multi => final={final_signal}, size={final_size}, aggregator_data for {pair}"
                if final_signal in ("BUY","SELL") and final_size>0:
                    record_trade_in_db(final_signal, final_size, current_price, "GPT_MULTI_DECISION", pair)
                self._store_decision(pair, final_signal, final_size, rationale)
                self._append_gpt_context(rationale)
                results[pair] = (final_signal, final_size)
            else:
                # no price => can't trade
                logger.warning(f"[AIStrategy] No price for pair={pair}, skipping => HOLD.")
                self._store_decision(pair,"HOLD",0.0,"No price => hold")
                results[pair] = ("HOLD",0.0)

        return results

    # --------------------------------------------------------------------------
    # HELPER LOGIC
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
        Check local constraints => e.g. if cost < min_buy => hold
        """
        if action not in ("BUY","SELL"):
            return ("HOLD",0.0)

        cost = size_suggested*current_price
        rc = self.risk_controls
        if not rc:
            return (action, size_suggested)

        min_buy = rc.get("minimum_buy_amount",6.0)
        if action=="BUY" and cost<min_buy:
            logger.info(f"[post_validate] cost={cost:.2f} < min_buy={min_buy:.2f} => hold")
            return ("HOLD",0.0)

        return (action, size_suggested)

    # --------------------------------------------------------------------------
    # GPT CONTEXT
    # --------------------------------------------------------------------------
    def _init_gpt_context(self) -> str:
        data = load_gpt_context_from_db()
        if data:
            logger.info("Loaded GPT context from DB.")
            return data
        return ""

    def _append_gpt_context(self, new_text: str) -> None:
        self.gpt_context += "\n" + new_text
        save_gpt_context_to_db(self.gpt_context)

    # --------------------------------------------------------------------------
    # ai_decisions Table
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
    # Summarize recent trades
    # --------------------------------------------------------------------------
    def _summarize_recent_trades(self, pair: str, limit: int=3) -> str:
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
            rows=rows[::-1] # chronological
            for i,row in enumerate(rows,start=1):
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
# Example usage snippet
# ------------------------------------------------------------------------------
if __name__=="__main__":
    """
    For demonstration or local testing:
      python ai_strategy.py
    """
    logging.basicConfig(level=logging.INFO)

    strategy = AIStrategy(
        pairs=["ETH/USD","XBT/USD"],
        use_openai=True,
        max_position_size=1.0,
        stop_loss_pct=0.05,
        take_profit_pct=0.01,
        max_daily_drawdown=-0.02,
        risk_controls={
            "initial_spending_account":100.0,
            "minimum_buy_amount":10.0,
            "purchase_upper_limit_percent":0.1,
        },
        gpt_model="gpt-4o-mini",
        gpt_temperature=0.8,
        gpt_max_tokens=1000
    )

    # Single pair usage
    single_data = {
        "pair":"ETH/USD",
        "price":1850.0,
        "rsi":45.0,
        "volume":1000.0,
    }
    a, s = strategy.predict(single_data)
    print(f"[SinglePair] => final action={a}, size={s}")

    # Multi pair usage (legacy approach => calls GPT or fallback per pair)
    multi_data = [
        {"pair":"ETH/USD","price":1800.0,"rsi":46.0,"volume":1100.0},
        {"pair":"XBT/USD","price":28000.0,"rsi":50.0,"volume":500.0},
    ]
    decisions = strategy.predict_multi(multi_data)
    print("Multi-pair (legacy) =>", decisions)

    # Multi coins advanced approach => single GPT call:
    aggregator_list = [
        {
            "pair": "ETH/USD",
            "price": 1800.0,
            "aggregator_data": "rsi=46, price=1800, galaxy_score=62, alt_rank=170, sentiment=0.35"
        },
        {
            "pair": "XBT/USD",
            "price": 28000.0,
            "aggregator_data": "rsi=50, price=28000, galaxy_score=65, alt_rank=150, sentiment=0.30"
        }
    ]
    trade_hist = [
        "2025-01-20 10:15 BUY ETH/USD 0.001@25000",
        "2025-01-21 11:20 SELL ETH/USD 0.001@25500"
    ]
    open_positions = [
        "ETH/USD LONG 0.002, entry=1860.0",
        "XBT/USD SHORT 0.001, entry=29500.0"
    ]
    multi_decisions = strategy.predict_multi_coins(
        aggregator_list=aggregator_list,
        trade_history=trade_hist,
        max_trades=5,
        open_positions=open_positions
    )
    print("Multi coins (advanced single GPT call) =>", multi_decisions)
