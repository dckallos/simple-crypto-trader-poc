#!/usr/bin/env python3
# =============================================================================
# FILE: ai_strategy.py
# =============================================================================
"""
ai_strategy.py

Enhanced AIStrategy with:
1) GPT logic (via GPTManager) or fallback logic for single or multi-coin GPT calls.
2) Daily drawdown, risk constraints, sub-position logic via RiskManagerDB.
3) Loads aggregator data from aggregator_summaries (and possibly other tables) to produce
   aggregator text **including percentage/absolute changes** from an older timestamp (e.g. 1 hour ago).
4) On GPT decisions, we record them in 'ai_decisions' and create a row in the new
   'pending_trades' table if the action is BUY or SELL. The actual fill is handled by
   the private WebSocket feed or your own logic.

NEW INTEGRATION:
   - We optionally send live orders to Kraken through 'private_ws_client.send_order()'.
   - This is controlled by 'self.place_live_orders' and requires passing a private_ws_client
     in AIStrategy's constructor.

Basic Flow:
    - Single pair => call predict(market_data)
    - Multi pairs => call predict_multi(pairs_data)
    - Advanced => predict_multi_coins(...) for one-shot multi-coin GPT

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
    create_pending_trade
)
from risk_manager import RiskManagerDB
from gpt_manager import GPTManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class AIStrategy:
    """
    AIStrategy: A flexible class that decides trades for one or multiple pairs,
    using GPT or fallback logic. We integrate RiskManagerDB for sub-position logic,
    daily drawdown checks, and store final decisions in 'ai_decisions'. GPT-suggested
    trades are only placed as 'pending' in 'pending_trades' until fills are confirmed
    by Kraken.

    If you want to actually place the orders on Kraken, set:
       place_live_orders=True
    and pass private_ws_client=<KrakenPrivateWSClient> in the constructor.
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
        gpt_model: str = "o1-mini",
        gpt_temperature: float = 1.0,
        gpt_max_tokens: int = 2000,
        private_ws_client=None,
        place_live_orders: bool = False
    ):
        """
        :param pairs: e.g. ["ETH/USD","XBT/USD"] if you want a default set
        :param use_openai: True => GPT logic is used; else fallback logic
        :param max_position_size: clamp on per-trade size
        :param stop_loss_pct: e.g. 0.05 => auto-close if -5%
        :param take_profit_pct: e.g. 0.01 => auto-close if +1%
        :param max_daily_drawdown: e.g. -0.02 => skip new trades if daily < -2%
        :param risk_controls: e.g. {"minimum_buy_amount":10.0}
        :param gpt_model: override GPT model name after GPTManager creation
        :param gpt_temperature: sampling temperature
        :param gpt_max_tokens: max GPT tokens
        :param private_ws_client: If provided, an instance of KrakenPrivateWSClient
        :param place_live_orders: If True, calls private_ws_client.send_order(...) after creating pending trades
        """
        self.pairs = pairs if pairs else []
        self.use_openai = use_openai
        self.risk_controls = risk_controls or {}

        load_dotenv()

        # GPT
        self.gpt_manager = None
        if self.use_openai:
            self.gpt_manager = GPTManager(
                config_file="config.yaml",
                temperature=gpt_temperature,
                max_tokens=gpt_max_tokens,
                log_gpt_calls=True
            )
            if gpt_model:
                self.gpt_manager.model = gpt_model

        # Risk manager
        self.risk_manager_db = RiskManagerDB(
            db_path=DB_FILE,
            max_position_size=max_position_size,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_daily_drawdown=max_daily_drawdown
        )
        self.risk_manager_db.initialize()

        # NEW: store references to private WS and a toggle for placing real orders
        self.private_ws_client = private_ws_client
        self.place_live_orders = place_live_orders

        # ensure ai_decisions table
        self._create_ai_decisions_table()

    # --------------------------------------------------------------------------
    # SINGLE-PAIR PREDICT
    # --------------------------------------------------------------------------
    def predict(self, market_data: Dict[str, Any]) -> Tuple[str, float]:
        """
        Predict for a single pair => calls GPT or fallback => returns (action,size).

        :param market_data: e.g. {"pair":"ETH/USD", "price":1850.0, ...}
        :return: (action, size)
        """
        pair = market_data.get("pair", "UNK")
        current_price = market_data.get("price", 0.0)
        if current_price <= 0:
            rationale = f"No valid price => HOLD {pair}"
            self._store_decision(pair, "HOLD", 0.0, rationale)
            return ("HOLD", 0.0)

        # forcibly check sub-position stops (daily drawdown, etc.)
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
        Single-pair GPT logic. We'll gather aggregator data (with changes),
        pass it to GPT, parse result, apply risk manager, store final decision.
        """
        current_px = market_data["price"]
        aggregator_text = self._build_aggregator_text_changes(pair)
        logger.info(f"[AIStrategy-single] aggregator_text => {aggregator_text}")

        trade_summary = self._summarize_recent_trades(pair, limit=3)
        trade_history_list = trade_summary.split("\n") if trade_summary else ["No trades found"]

        result = self.gpt_manager.generate_trade_decision(
            aggregator_text=aggregator_text,
            trade_history=trade_history_list,
            max_trades=10,
            risk_controls=self.risk_controls,
            open_positions=None,
            reflection_enabled=False
        )
        action = result.get("action", "HOLD").upper()
        suggested_size = float(result.get("size", 0.0))

        final_signal, final_size = self._post_validate(action, suggested_size, current_px)
        final_signal, final_size = self.risk_manager_db.adjust_trade(final_signal, final_size, pair, current_px)

        rationale = f"GPT single => final={final_signal}, size={final_size}, aggregator=({aggregator_text})"
        if final_signal in ("BUY", "SELL") and final_size > 0:
            # CHANGED: we create a pending trade AND optionally place a real order
            pending_id = create_pending_trade(
                side=final_signal,
                requested_qty=final_size,
                pair=pair,
                reason="GPT_SINGLE_DECISION"
            )
            # If place_live_orders is True, call private_ws_client
            self._maybe_place_kraken_order(pair, final_signal, final_size, pending_id)

        self._store_decision(pair, final_signal, final_size, rationale)
        return (final_signal, final_size)

    def _fallback_logic(self, pair: str, market_data: Dict[str, Any]) -> Tuple[str, float]:
        """
        Dummy fallback logic if GPT is off or fails:
        If price<20k => BUY => 0.0005, else HOLD.
        """
        px = market_data.get("price", 0.0)
        if px < 20000:
            sig, sz = self._post_validate("BUY", 0.0005, px)
            sig, sz = self.risk_manager_db.adjust_trade(sig, sz, pair, px)
            rationale = f"[FALLBACK] => px={px} <20000 => {sig} {sz}"
            if sig == "BUY" and sz > 0:
                pending_id = create_pending_trade(
                    side=sig,
                    requested_qty=sz,
                    pair=pair,
                    reason="FALLBACK_SINGLE"
                )
                self._maybe_place_kraken_order(pair, sig, sz, pending_id)
            self._store_decision(pair, sig, sz, rationale)
            return (sig, sz)
        else:
            rationale = f"[FALLBACK] => px={px} >=20000 => HOLD"
            self._store_decision(pair, "HOLD", 0.0, rationale)
            return ("HOLD", 0.0)

    # --------------------------------------------------------------------------
    # MULTI-PAIR (LEGACY, calling single predict on each)
    # --------------------------------------------------------------------------
    def predict_multi(self, pairs_data: List[Dict[str, Any]], concurrency="thread") -> Dict[str, Tuple[str, float]]:
        """
        A naive multi approach => calls self.predict(...) on each pair.
        If you want a single GPT call for multi-coin, see predict_multi_coins.
        """
        if not pairs_data:
            logger.warning("No pairs_data => empty decisions.")
            return {}

        if concurrency == "thread":
            logger.info("[AIStrategy] multi => naive synchronous approach.")
        elif concurrency == "asyncio":
            logger.info("[AIStrategy] multi => placeholder for async approach.")

        decisions = {}
        for pd in pairs_data:
            pair = pd.get("pair", "UNK")
            px = pd.get("price", 0.0)
            if px > 0:
                self.risk_manager_db.check_stop_loss_take_profit(pair, px)

            if self.use_openai and self.gpt_manager:
                try:
                    output_action, output_size = self._gpt_flow_single(pair, pd)
                except Exception as e:
                    logger.exception(f"[AIStrategy] GPT single inference fail => fallback => {e}")
                    output_action, output_size = self._fallback_logic(pair, pd)
            else:
                output_action, output_size = self._fallback_logic(pair, pd)

            decisions[pair] = (output_action, output_size)
        return decisions

    # --------------------------------------------------------------------------
    # MULTI-PAIR (ADVANCED, single GPT call)
    # --------------------------------------------------------------------------
    def predict_multi_coins(
        self,
        input_aggregator_list: List[Dict[str, Any]],
        trade_history: List[str],
        max_trades: int,
        input_open_positions: List[str],
        current_balance: float = 0.0,
        current_trade_balance: dict[str, float] = None
    ) -> Dict[str, Tuple[str, float]]:
        """
        The advanced approach => single GPT call for multiple coins => returns
        final decisions => store them in 'ai_decisions' and optionally create
        pending trades if action=BUY/SELL + place real orders if configured.

        aggregator_list => each item => {
           "pair":"ETH/USD","price":1850.0,
           "aggregator_data":"some aggregator snippet with changes"
        }
        """
        results_set = {}
        if not input_aggregator_list:
            logger.warning("No aggregator_list => empty multi-coin decisions.")
            return results_set

        # forcibly check sub-position stops
        for item in input_aggregator_list:
            pair = item.get("pair", "UNK")
            px = item.get("price", 0.0)
            if px > 0:
                self.risk_manager_db.check_stop_loss_take_profit(pair, px)

        if not self.use_openai or not self.gpt_manager:
            logger.info("[AIStrategy] no GPT => fallback => hold all coins")
            for it in input_aggregator_list:
                results_set[it["pair"]] = ("HOLD", 0.0)
            return results_set

        try:
            multi_resp = self.gpt_manager.generate_multi_trade_decision(
                aggregator_lines=input_aggregator_list,
                open_positions=input_open_positions,
                trade_history=trade_history,
                max_trades=max_trades,
                risk_controls=self.risk_controls,
                reflection_enabled=True,
                current_balance=current_balance,
                current_trade_balance=current_trade_balance
            )
        except Exception as e:
            logger.exception("[AIStrategy] GPT multi call fail => fallback => hold.")
            for it in input_aggregator_list:
                results_set[it["pair"]] = ("HOLD", 0.0)
            return results_set

        decisions_array = multi_resp.get("decisions", [])
        if not decisions_array:
            logger.warning("[AIStrategy] GPT => no decisions => hold all.")
            for it in input_aggregator_list:
                pair = it.get("pair", "UNK")
                results_set[pair] = ("HOLD", 0.0)
            return results_set

        for dec in decisions_array:
            pair = dec.get("pair", "UNK")
            action_raw = dec.get("action", "HOLD").upper()
            size_suggested = float(dec.get("size", 0.0))

            found_item = next((x for x in input_aggregator_list if x.get("pair") == pair), None)
            px = found_item.get("price", 0.0) if found_item else 0.0

            if px <= 0:
                logger.warning(f"[AIStrategy] no valid price for {pair} => HOLD")
                self._store_decision(pair, "HOLD", 0.0, "No price => hold")
                results_set[pair] = ("HOLD", 0.0)
                continue

            final_signal, final_size = self._post_validate(action_raw, size_suggested, px)
            final_signal, final_size = self.risk_manager_db.adjust_trade(final_signal, final_size, pair, px)
            rationale = f"GPT multi => {pair} => final={final_signal}, size={final_size}"

            if final_signal in ("BUY", "SELL") and final_size > 0:
                pending_id = create_pending_trade(
                    side=final_signal,
                    requested_qty=final_size,
                    pair=pair,
                    reason="GPT_MULTI_DECISION"
                )
                # If live orders are enabled, send to Kraken
                self._maybe_place_kraken_order(pair, final_signal, final_size, pending_id)

            self._store_decision(pair, final_signal, final_size, rationale)
            results_set[pair] = (final_signal, final_size)

        return results_set

    # --------------------------------------------------------------------------
    # HELPER: aggregator_text with changes
    # --------------------------------------------------------------------------
    def _build_aggregator_text_changes(self, pair: str) -> str:
        """
        Example function to return aggregator data with changes since the last hour.
        This is a skeleton for aggregator_summaries. You can expand for numeric fields.
        """
        import time
        import sqlite3
        now_ts = int(time.time())
        one_hour_ago = now_ts - 3600

        import db_lookup
        symbol = db_lookup.get_base_asset(pair)
        current_data = {"symbol": symbol, "galaxy_score": None, "alt_rank": None, "sentiment": None}
        old_data = {"symbol": symbol, "galaxy_score": None, "alt_rank": None, "sentiment": None}

        conn = sqlite3.connect(DB_FILE)
        try:
            c = conn.cursor()
            c.execute("""
            SELECT timestamp, galaxy_score, alt_rank, sentiment_label
            FROM aggregator_summaries
            WHERE UPPER(symbol)=?
            ORDER BY timestamp DESC
            LIMIT 1
            """, (symbol,))
            row_now = c.fetchone()
            if row_now:
                _, gs, ar, slabel = row_now
                current_data["galaxy_score"] = float(gs) if gs else None
                current_data["alt_rank"] = float(ar) if ar else None
                current_data["sentiment"] = slabel or "neutral"

            c.execute("""
            SELECT timestamp, galaxy_score, alt_rank, sentiment_label
            FROM aggregator_summaries
            WHERE UPPER(symbol)=?
              AND timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT 1
            """, (symbol, one_hour_ago))
            row_old = c.fetchone()
            if row_old:
                _, gs2, ar2, slabel2 = row_old
                old_data["galaxy_score"] = float(gs2) if gs2 else None
                old_data["alt_rank"] = float(ar2) if ar2 else None
                old_data["sentiment"] = slabel2 or "neutral"

        except Exception as e:
            logger.exception(f"[AIStrategy] aggregator changes => {e}")
        finally:
            conn.close()

        def format_change(current_val, old_val, label):
            if current_val is None or old_val is None:
                return f"{label}={current_val} (NA)"
            try:
                diff = current_val - old_val
                sign = f"{diff:+.2f}"
                return f"{label}={current_val} ({sign})"
            except:
                return f"{label}={current_val} (NA)"

        gs_str = format_change(current_data["galaxy_score"], old_data["galaxy_score"], "galaxy_score")
        ar_str = format_change(current_data["alt_rank"], old_data["alt_rank"], "alt_rank")
        sent_str = f"sentiment={current_data['sentiment']} (old={old_data['sentiment']})"

        aggregator_text = (
            f"Pair={pair} => {gs_str}, {ar_str}, {sent_str}\n"
            "(Note: Example aggregator changes. "
            "You can expand to other numeric fields if stored.)"
        )
        return aggregator_text

    # --------------------------------------------------------------------------
    # POST-VALIDATION
    # --------------------------------------------------------------------------
    def _post_validate(self, action: str, size_suggested: float, current_price: float) -> Tuple[str, float]:
        """
        e.g. clamp or hold if cost < min buy.
        Risk controls can be extended for more advanced checks.
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
    # ai_decisions table
    # --------------------------------------------------------------------------
    def _create_ai_decisions_table(self):
        """Ensure 'ai_decisions' table for storing final GPT signals."""
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
        """Insert a row into 'ai_decisions' with the final GPT or fallback decision."""
        logger.debug(f"Storing AI => pair={pair}, action={action}, size={size}, reason={rationale}")
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
    # Summarize recent trades
    # --------------------------------------------------------------------------
    def _summarize_recent_trades(self, pair: str, limit: int = 3) -> str:
        """
        Returns e.g. "1) 2025-01-01 10:00 BUY 0.001@2000\n2) ..."
        referencing the final 'trades' table for completed trades only.
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
            rows = rows[::-1]
            for i, row in enumerate(rows, start=1):
                t, sd, qty, px = row
                import datetime
                dt_s = datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")
                out.append(f"{i}) {dt_s} {sd.upper()} {qty}@{px}")
        except Exception as e:
            logger.exception(f"Error summarizing trades => {e}")
            return "Error retrieving trades."
        finally:
            conn.close()

        return "\n".join(out)

    # --------------------------------------------------------------------------
    # NEW: maybe_place_kraken_order
    # --------------------------------------------------------------------------
    def _maybe_place_kraken_order(self, pair: str, action: str, volume: float, pending_id: int = None):
        """
        If self.place_live_orders is True and self.private_ws_client is set,
        call the client's send_order(...) method to place the order on Kraken.
        We'll do a basic market order. The 'pending_id' is used as 'userref'
        so we can match the final 'txid' to our local DB row.
        """
        if not self.place_live_orders:
            return

        if not self.private_ws_client:
            logger.warning(
                f"[AIStrategy] place_live_orders=True but no private_ws_client provided. "
                f"Cannot place real order for pair={pair}."
            )
            return

        side_for_kraken = "buy" if action.upper() == "BUY" else "sell"
        logger.info(
            f"[AIStrategy] Sending real order => pair={pair}, side={side_for_kraken}, volume={volume}, pending_id={pending_id}"
        )

        # userref = pending_id so we can match it later
        self.private_ws_client.send_order(
            pair=pair,
            side=side_for_kraken,
            ordertype="market",
            volume=volume,
            userref=str(pending_id) if pending_id else None
        )

