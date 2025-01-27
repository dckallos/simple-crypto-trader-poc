"""
ai_strategy.py

An enhanced AIStrategy that:
1) Uses GPT logic (via GPTManager) or fallback logic for single or multi-coin GPT calls.
2) Integrates daily drawdown, risk constraints, sub-position logic via RiskManagerDB.
3) **Now** loads aggregator data from aggregator_summaries, aggregator_classifier_probs, aggregator_embeddings,
   then uses build_aggregator_prompt(...) to produce a short aggregator text.

Basic Flow:
    - Single pair => call predict(market_data)
    - Multi pairs => call predict_multi(pairs_data)
    - For advanced one-shot multi-coin => predict_multi_coins(...)
"""

import logging
import os
import time
import json
import sqlite3
from typing import Optional, Dict, Any, List, Tuple

from dotenv import load_dotenv

from db import (
    DB_FILE,
    load_gpt_context_from_db,
    save_gpt_context_to_db,
    record_trade_in_db
)
from risk_manager import RiskManagerDB
from gpt_manager import GPTManager, build_aggregator_prompt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class AIStrategy:
    """
    AIStrategy: A flexible class that decides trades for one or multiple pairs,
    using GPT or fallback logic. Now it loads aggregator data from aggregator_summaries,
    aggregator_classifier_probs, aggregator_embeddings to build a short aggregator prompt
    for GPT.
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

        # GPT
        self.gpt_manager = None
        if self.use_openai:
            self.gpt_manager = GPTManager(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                model=gpt_model,
                temperature=gpt_temperature,
                max_tokens=gpt_max_tokens,
                **gpt_client_options
            )

        # Risk manager
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
        Single-pair GPT logic. We gather aggregator data from aggregator_summaries, aggregator_classifier_probs,
        aggregator_embeddings, build a short aggregator text, pass it to GPT, parse result,
        apply risk manager, store decision, etc.
        """
        px = market_data["price"]

        # (A) Build aggregator text from DB
        aggregator_text = self._build_full_aggregator_text(pair)

        # (B) Summarize recent trades
        trade_summary = self._summarize_recent_trades(pair, limit=3)
        trade_history_list = trade_summary.split("\n") if trade_summary else []

        # (C) GPT => single-coin approach
        result = self.gpt_manager.generate_trade_decision(
            conversation_context=self.gpt_context,
            aggregator_text=aggregator_text,
            trade_history=trade_history_list,
            max_trades=10,
            risk_controls=self.risk_controls,
            open_positions=None  # single approach => ignore
        )
        action = result.get("action", "HOLD").upper()
        suggested_size = float(result.get("size", 0.0))

        # (D) final risk checks
        final_signal, final_size = self._post_validate(action, suggested_size, px)
        final_signal, final_size = self.risk_manager_db.adjust_trade(final_signal, final_size, pair, px)

        # (E) store trade & decision
        rationale = f"GPT single => final={final_signal}, size={final_size}, aggregator=({aggregator_text})"
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
    # MULTI-PAIR (LEGACY)
    # --------------------------------------------------------------------------
    def predict_multi(self, pairs_data: List[Dict[str, Any]], concurrency="thread") -> Dict[str, Tuple[str, float]]:
        """
        A naive multi approach => calls predict(...) on each pair. This references the same code path
        for single predictions, so aggregator data is loaded individually.
        If you want one-shot multi-coin GPT => see predict_multi_coins.
        """
        if not pairs_data:
            logger.warning("No pairs_data => empty decisions.")
            return {}

        if concurrency=="thread":
            logger.info("[AIStrategy] multi => naive synchronous approach.")
        elif concurrency=="asyncio":
            logger.info("[AIStrategy] multi => placeholder for async approach.")
            # In real usage, you'd do concurrency with an event loop, gather, etc.

        decisions={}
        for pd in pairs_data:
            pair = pd.get("pair","UNK")
            px = pd.get("price",0.0)
            # forcibly check stops
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
    # MULTI-PAIR (ADVANCED) => Single GPT call
    # --------------------------------------------------------------------------
    def predict_multi_coins(
        self,
        aggregator_list: List[Dict[str, Any]],
        trade_history: List[str],
        max_trades: int,
        open_positions: List[str]
    ) -> Dict[str, Tuple[str, float]]:
        """
        The advanced approach: aggregator_list => pass them all at once to GPT => single JSON response.
        aggregator_list => each item => {"pair":"ETH/USD","price":1850,"aggregator_data":"summary:..., prob=0.72, ..."}
        We'll parse GPT decisions => apply risk => store.
        """
        results = {}
        if not aggregator_list:
            logger.warning("No aggregator_list => empty decisions.")
            return {}

        # forcibly check stops for each aggregator
        for item in aggregator_list:
            pair = item.get("pair","UNK")
            maybe_price = item.get("price", 0.0)
            if maybe_price>0:
                self.risk_manager_db.check_stop_loss_take_profit(pair, maybe_price)

        if not self.use_openai or not self.gpt_manager:
            logger.info("[AIStrategy] no GPT => fallback => hold all")
            for it in aggregator_list:
                results[it["pair"]] = ("HOLD",0.0)
            return results

        # single GPT call => parse => post validate => store
        try:
            multi_resp = self.gpt_manager.generate_multi_trade_decision(
                conversation_context=self.gpt_context,
                aggregator_list=aggregator_list,
                open_positions=open_positions,
                trade_history=trade_history,
                max_trades=max_trades,
                risk_controls=self.risk_controls
            )
        except Exception as e:
            logger.exception("[AIStrategy] GPT multi fail => fallback => hold all.")
            for it in aggregator_list:
                results[it["pair"]] = ("HOLD",0.0)
            return results

        decisions_array = multi_resp.get("decisions", [])
        if not decisions_array:
            logger.warning("[AIStrategy] No GPT decisions => hold all")
            for it in aggregator_list:
                results[it["pair"]] = ("HOLD",0.0)
            return results

        for dec in decisions_array:
            pair = dec.get("pair","UNK")
            action_raw = dec.get("action","HOLD").upper()
            size_suggested = float(dec.get("size",0.0))

            found_item = next((x for x in aggregator_list if x.get("pair")==pair), None)
            current_price = float(found_item["price"]) if (found_item and "price" in found_item) else 0.0

            if current_price>0:
                final_signal, final_size = self._post_validate(action_raw, size_suggested, current_price)
                final_signal, final_size = self.risk_manager_db.adjust_trade(final_signal, final_size, pair, current_price)
                rationale = f"GPT multi => {pair} => final={final_signal}, size={final_size}"
                if final_signal in ("BUY","SELL") and final_size>0:
                    record_trade_in_db(final_signal, final_size, current_price, "GPT_MULTI_DECISION", pair)
                self._store_decision(pair, final_signal, final_size, rationale)
                self._append_gpt_context(rationale)
                results[pair] = (final_signal, final_size)
            else:
                logger.warning(f"[AIStrategy] No price for {pair} => hold.")
                self._store_decision(pair,"HOLD",0.0,"No price => hold")
                results[pair] = ("HOLD",0.0)

        return results

    # --------------------------------------------------------------------------
    # HELPER LOGIC
    # --------------------------------------------------------------------------
    def _build_full_aggregator_text(self, pair: str) -> str:
        """
        A specialized aggregator retrieval for single-pair usage.
        1) Parse symbol from pair => e.g. "ETH/USD" => "ETH"
        2) Look up aggregator_summaries => gather textual summary
        3) aggregator_classifier_probs => prob_up
        4) aggregator_embeddings => small vector [comp1, comp2, comp3]
        5) Pass them to build_aggregator_prompt(...) => returns a single short string

        Return that aggregator string for GPT usage.
        """
        symbol = pair.split("/")[0].upper()
        # We'll do naive approach: pick the LATEST row for that symbol from each aggregator table.
        summary_str = None
        prob_up = None
        embedding_vec = None

        # aggregator_summaries => get price_bucket, sentiment_label, galaxy_score, alt_rank
        conn = sqlite3.connect(DB_FILE)
        try:
            c = conn.cursor()

            # Summaries
            c.execute("""
            SELECT price_bucket, galaxy_score, alt_rank, sentiment_label
            FROM aggregator_summaries
            WHERE UPPER(symbol)=?
            ORDER BY timestamp DESC
            LIMIT 1
            """, (symbol,))
            row_summ = c.fetchone()
            if row_summ:
                pb, gs, ar, sl = row_summ
                # e.g. "price_bucket=high, galaxy_score=64, alt_rank=150, sentiment_label=slightly_positive"
                summary_list = []
                if pb: summary_list.append(f"price_bucket={pb}")
                if gs is not None: summary_list.append(f"galaxy_score={gs}")
                if ar is not None: summary_list.append(f"alt_rank={ar}")
                if sl: summary_list.append(f"sentiment_label={sl}")
                summary_str = ", ".join(summary_list)

            # Classifier => prob_up
            c.execute("""
            SELECT prob_up
            FROM aggregator_classifier_probs
            WHERE UPPER(symbol)=?
            ORDER BY timestamp DESC
            LIMIT 1
            """, (symbol,))
            row_prob = c.fetchone()
            if row_prob:
                prob_up = float(row_prob[0])

            # Embeddings => comp1, comp2, comp3
            c.execute("""
            SELECT comp1, comp2, comp3
            FROM aggregator_embeddings
            WHERE UPPER(symbol)=?
            ORDER BY timestamp DESC
            LIMIT 1
            """, (symbol,))
            row_emb = c.fetchone()
            if row_emb:
                comp1, comp2, comp3 = row_emb
                embedding_vec = [float(comp1), float(comp2), float(comp3)]

        except Exception as e:
            logger.exception(f"Error retrieving aggregator data => {e}")
        finally:
            conn.close()

        # Now build aggregator_text
        aggregator_text = build_aggregator_prompt(
            summary=summary_str,
            probability=prob_up,
            embedding=embedding_vec
        )
        if not aggregator_text:
            aggregator_text = "No aggregator data found."
        return aggregator_text

    def _post_validate(self, action: str, size_suggested: float, current_price: float) -> Tuple[str, float]:
        """
        Local check => e.g. cost < min_buy => hold
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

    # Single pair usage => aggregator text is built from aggregator_* tables
    single_data = {
        "pair":"ETH/USD",
        "price":1850.0,
    }
    action, size = strategy.predict(single_data)
    print(f"[SinglePair] => final action={action}, size={size}")

    # Multi pair usage => legacy => calls predict(...) individually
    multi_data = [
        {"pair":"ETH/USD","price":1800.0},
        {"pair":"XBT/USD","price":28000.0},
    ]
    results = strategy.predict_multi(multi_data)
    print("Multi-pair =>", results)

    # Advanced multi => aggregator_list has aggregator_data => if you want a single GPT call
    aggregator_list = [
        {
            "pair": "ETH/USD",
            "price": 1800.0,
            "aggregator_data": "summary: price_bucket=high, galaxy_score=64, sentiment_label=slightly_positive; probability=0.72; embedding_vector=[0.25,-0.1,0.36]"
        },
        {
            "pair": "XBT/USD",
            "price": 28000.0,
            "aggregator_data": "summary: price_bucket=medium, galaxy_score=65, sentiment_label=neutral; probability=0.55; embedding_vector=[0.19,0.02,-0.15]"
        }
    ]
    trade_hist = ["2025-01-20 10:15 BUY ETH/USD 0.001@25000","2025-01-21 SELL ETH/USD 0.001@25500"]
    open_positions = ["ETH/USD LONG 0.002, entry=1860.0","XBT/USD SHORT 0.001, entry=29500.0"]
    final = strategy.predict_multi_coins(aggregator_list, trade_hist, max_trades=5, open_positions=open_positions)
    print("Multi coins =>", final)
