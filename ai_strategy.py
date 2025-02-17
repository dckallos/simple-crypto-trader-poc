#!/usr/bin/env python3
# =============================================================================
# FILE: ai_strategy.py
# =============================================================================
"""
ai_strategy.py (Production-Ready)

AIStrategy is responsible for:
1) Generating BUY/SELL/HOLD signals for one or multiple pairs (either via GPT or fallback logic).
2) Relying on RiskManagerDB to enforce daily drawdown checks and real-time capital usage.
3) Creating rows in 'pending_trades' when a new trade is signaled. The private feed finalizes these,
   and inserts actual fills into 'trades'.

Stop-loss and take-profit are not handled directly here. That logic lives in the
lots-based RiskManager (risk_manager.py) or can be called from ws_data_feed.py via
on_price_update(...).

If you want to place real Kraken orders automatically:
  - set place_live_orders=True
  - pass private_ws_client=<KrakenPrivateWSClient> in the constructor
    so we can call private_ws_client.send_order(...) upon creating a pending trade.
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
from kraken_rest_manager import KrakenRestManager

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Used for checking real-time balances whenever we call adjust_trade.
rest_manager = KrakenRestManager(
    os.getenv("KRAKEN_API_KEY"),
    os.getenv("KRAKEN_SECRET_API_KEY")
)


class AIStrategy:
    """
    AIStrategy: Decides trades for one or multiple pairs,
    using GPT or fallback logic. Key points:

      - RiskManagerDB performs daily drawdown checks & real-time usage checks
      - 'pending_trades' holds ephemeral states
      - 'trades' table receives final fills (inserted by the private feed)
      - Stop-loss & take-profit: handled outside (risk_manager.py or ws_data_feed)

    Constructor:
      - place_live_orders => if True, calls private_ws_client.send_order(...)
      - risk_controls => can store custom user constraints if needed
    """

    def __init__(
        self,
        pairs: List[str] = None,
        use_openai: bool = False,
        max_position_size: float = 3.0,
        max_daily_drawdown: float = -0.02,
        risk_controls: Optional[Dict[str, Any]] = None,
        risk_manager: RiskManagerDB = None,
        gpt_model: str = "o1-mini",
        gpt_temperature: float = 1.0,
        gpt_max_tokens: int = 5000,
        private_ws_client=None,
        place_live_orders: bool = False
    ):
        """
        :param pairs: e.g. ["ETH/USD", "XBT/USD"]
        :param use_openai: True => GPT logic is used; else fallback logic
        :param max_position_size: clamp on per-trade size in RiskManager
        :param max_daily_drawdown: e.g. -0.02 => skip new trades if daily PnL < -2%
        :param risk_controls: optional dict with custom fields (like "initial_spending_account")
        :param gpt_model: GPT model name override
        :param gpt_temperature: GPT model temperature
        :param gpt_max_tokens: GPT model maximum tokens
        :param private_ws_client: an instance of KrakenPrivateWSClient for live orders (optional)
        :param place_live_orders: if True => we call private_ws_client.send_order(...) after each new trade
        """
        self.pairs = pairs if pairs else []
        self.use_openai = use_openai
        self.risk_controls = risk_controls or {}

        # GPT Initialization
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

        # Whether to place real orders via KrakenPrivateWSClient
        self.private_ws_client = private_ws_client
        self.place_live_orders = place_live_orders

        # Ensure ai_decisions table exists
        self._create_ai_decisions_table()

        self.risk_manager = risk_manager

    # --------------------------------------------------------------------------
    # 1) SINGLE-PAIR PREDICT
    # --------------------------------------------------------------------------
    def predict(self, market_data: Dict[str, Any]) -> Tuple[str, float]:
        """
        Predict for a single pair => returns (action, size).
        1) Use GPT if enabled, else fallback logic
        2) Call risk_manager_db.adjust_trade(...) for daily drawdown checks & size clamps
        3) If final action=BUY or SELL => create a pending_trades row (+ optionally place a real order)

        :param market_data: e.g. {"pair":"ETH/USD", "price":1850.0, ...}
        :return: (final_action, final_size)
        """
        pair = market_data.get("pair", "UNK")
        current_price = market_data.get("price", 0.0)
        if current_price <= 0:
            rationale = f"No valid price => HOLD {pair}"
            self._store_decision(pair, "HOLD", 0.0, rationale)
            return ("HOLD", 0.0)

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
        Single-pair GPT logic:
        1) Build aggregator text
        2) Query GPT
        3) risk_manager_db.adjust_trade(...) for daily drawdown & real-time usage checks
        4) If action=BUY => also call risk_manager_db.add_or_combine_lot(...) after creating a pending trade
        """
        current_px = market_data["price"]
        aggregator_text = self._build_aggregator_text_changes(pair)
        logger.info(f"[AIStrategy] aggregator_text => {aggregator_text}")

        trade_summary = self._summarize_recent_trades(pair, limit=3)
        trade_history_list = trade_summary.split("\n") if trade_summary else ["No trades found"]

        # 1) GPT => Single-Pair Decision
        result = self.gpt_manager.generate_trade_decision(
            aggregator_text=aggregator_text,
            trade_history=trade_history_list,
            max_trades=10,
            risk_controls=self.risk_controls,
            open_positions=None,
            reflection_enabled=False
        )
        action_raw = result.get("action", "HOLD").upper()
        suggested_size = float(result.get("size", 0.0))

        # 2) local clamp => post_validate
        final_action, final_size = self._post_validate(action_raw, suggested_size, current_px)
        # 3) daily drawdown check => real-time usage check
        final_action, final_size = self.risk_manager.adjust_trade(
            final_action, final_size, pair, current_px, rest_manager.fetch_balance()
        )

        # 4) Create pending trade if BUY/SELL => optionally place a real order
        rationale = (
            f"GPT single => final={final_action}, size={final_size}, aggregator=({aggregator_text})"
        )
        if final_action in ("BUY", "SELL") and final_size > 0:
            pending_id = create_pending_trade(
                side=final_action,
                requested_qty=final_size,
                pair=pair,
                reason="GPT_SINGLE_DECISION"
            )
            # Optionally place a real order on Kraken
            self._maybe_place_kraken_order(pair, final_action, final_size, pending_id)

            # If it is a BUY => combine or create a new lot in RiskManager
            if final_action == "BUY":
                self.risk_manager.add_or_combine_lot(
                    pair=pair,
                    buy_quantity=final_size,
                    buy_price=current_px
                )

        self._store_decision(pair, final_action, final_size, rationale)
        return (final_action, final_size)

    def _fallback_logic(self, pair: str, market_data: Dict[str, Any]) -> Tuple[str, float]:
        """
        Minimal fallback logic if GPT is disabled or errors out.
        Example: if price < 20k => buy a small amount, otherwise hold.
        After the local check, we pass the action to risk_manager_db.adjust_trade(...).
        """
        px = market_data.get("price", 0.0)
        if px < 20000:
            action, size = self._post_validate("BUY", 0.0005, px)
            action, size = self.risk_manager.adjust_trade(
                action, size, pair, px, rest_manager.fetch_balance()
            )
            rationale = f"[Fallback] => px={px:.2f} <20k => {action} {size}"
            if action == "BUY" and size > 0:
                # Create pending trade => possibly place real order
                pending_id = create_pending_trade(
                    side=action,
                    requested_qty=size,
                    pair=pair,
                    reason="FALLBACK_SINGLE",
                    source="fallback",
                    rationale="Fallback logic from an AI decision."
                )
                self._maybe_place_kraken_order(pair, action, size, pending_id)

                # add or combine lot
                self.risk_manager.add_or_combine_lot(
                    pair=pair,
                    buy_quantity=size,
                    buy_price=px
                )
            self._store_decision(pair, action, size, rationale, group_rationale="Fallback logic from an AI decision.")
            return (action, size)

        else:
            rationale = f"[Fallback] => px={px:.2f} >=20k => HOLD"
            self._store_decision(
                pair,
                "HOLD",
                0.0,
                rationale,
                group_rationale="Fallback logic from an AI decision."
            )
            return ("HOLD", 0.0)

    # --------------------------------------------------------------------------
    # 2) MULTI-PAIR (Naive approach => calls single predict() for each pair)
    # --------------------------------------------------------------------------
    def predict_multi(self, pairs_data: List[Dict[str, Any]], concurrency="thread") -> Dict[str, Tuple[str, float]]:
        """
        A simple loop that calls self.predict(...) for each pair.
        This is an older approach, replaced by the single GPT call in predict_multi_coins(...).
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
            if px <= 0:
                rationale = f"No valid price => HOLD {pair}"
                self._store_decision(pair, "HOLD", 0.0, rationale)
                decisions[pair] = ("HOLD", 0.0)
                continue

            # Attempt GPT or fallback
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
    # 3) MULTI-PAIR (Advanced => Single GPT Call)
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
        The advanced approach => single GPT call for multiple coins => returns final decisions.
        1) We build aggregator_text for each coin
        2) GPT returns an array of decisions
        3) We finalize them with risk_manager_db.adjust_trade(...)
        4) For BUY => create pending trade => optionally place real order => call add_or_combine_lot(...)

        :param input_aggregator_list: Each dict => {"pair":"ETH/USD","price":1850.0,"aggregator_data":...}
        :param trade_history: lines describing recent trades
        :param max_trades: max lines from trade_history to feed GPT
        :param input_open_positions: not used in detail here, but GPT can see them if needed
        :param current_balance: user’s USD (or base currency) if needed for GPT
        :param current_trade_balance: dict of {symbol: balance} for GPT references
        :return: decisions => { pair => (action, size) }
        """
        results_set = {}
        if not input_aggregator_list:
            logger.warning("[AIStrategy-multi_coins] aggregator_list empty => hold all.")
            return results_set

        if not self.use_openai or not self.gpt_manager:
            logger.info("[AIStrategy-multi_coins] GPT disabled => fallback => hold all.")
            for it in input_aggregator_list:
                results_set[it["pair"]] = ("HOLD", 0.0)
            return results_set

        # 1) Attempt a single GPT call
        try:
            multi_resp = self.gpt_manager.generate_multi_trade_decision(
                aggregator_lines=input_aggregator_list,
                open_positions=input_open_positions,
                trade_history=trade_history[-max_trades:],  # trim if needed
                max_trades=max_trades,
                risk_controls=self.risk_controls,
                reflection_enabled=True,
                current_balance=current_balance,
                current_trade_balance=current_trade_balance
            )
        except Exception as e:
            logger.exception("[AIStrategy] GPT multi call fail => fallback => hold")
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
        ai_rationale = multi_resp.get("rationale", "No GPT rationale provided")

        # 2) For each GPT decision => do local post-validation => risk_manager_db => create pending trades
        for dec in decisions_array:
            pair = dec.get("pair", "UNK")
            action_raw = dec.get("action", "HOLD").upper()
            size_suggested = float(dec.get("size", 0.0))

            # find the aggregator item => get its price
            found_item = next((x for x in input_aggregator_list if x.get("pair") == pair), None)
            px = found_item.get("price", 0.0) if found_item else 0.0
            if px <= 0:
                logger.warning(f"[AIStrategy-multi_coins] No valid px for {pair} => hold")
                self._store_decision(pair, "HOLD", 0.0, "No price => hold")
                results_set[pair] = ("HOLD", 0.0)
                continue

            # local clamp => daily drawdown
            final_action, final_size = self._post_validate(action_raw, size_suggested, px)
            final_action, final_size = self.risk_manager.adjust_trade(
                final_action, final_size, pair, px, rest_manager.fetch_balance()
            )
            rationale = f"GPT multi => {pair} => final={final_action}, size={final_size}"

            # If it’s a valid trade => create pending + optionally place an order + update lots if BUY
            if final_action in ("BUY", "SELL") and final_size > 0:
                pending_id = create_pending_trade(
                    side=final_action,
                    requested_qty=final_size,
                    pair=pair,
                    reason="GPT_MULTI_DECISION",
                    source="ai_strategy",
                    rationale=ai_rationale
                )
                self._maybe_place_kraken_order(pair, final_action, final_size, pending_id)

                # For a BUY => add or combine lot
                if final_action == "BUY":
                    self.risk_manager.add_or_combine_lot(
                        pair=pair,
                        buy_quantity=final_size,
                        buy_price=px
                    )

            self._store_decision(pair, final_action, final_size, rationale, group_rationale=ai_rationale)
            results_set[pair] = (final_action, final_size)

        return results_set

    def predict_multi_coins_simple(
            self,
            aggregator_list_simple: List[Dict[str, Any]],
            trade_history: List[str],
            max_trades: int,
            input_open_positions: List[str],
            current_balance: float,
            current_trade_balance: Dict[str, float]
    ) -> Dict[str, Tuple[str, float]]:
        """
        A streamlined, 'simple prompt' version of the multi-coin trading logic. This method
        calls generate_multi_trade_decision_simple_prompt(...) to retrieve BUY/SELL/HOLD
        instructions, then follows a nearly identical workflow to predict_multi_coins(...):

          1) If no aggregator data is provided or GPT usage is disabled, returns an empty
             dict or HOLD decisions.
          2) Calls GPT with a simplified prompt, capturing a JSON output of recommended trades.
          3) For each pair in the response:
             - Retrieve a valid 'price' from aggregator_list_simple or set to 0 if missing.
             - If the price <= 0, we set action=HOLD and skip further steps.
             - Otherwise, we run local post-validation to clamp the action and size
               (via self._post_validate(...)).
             - We pass the action and size to self.risk_manager.adjust_trade(...) to apply
               risk checks, daily drawdown logic, and real-time balance checks.
             - If the final action is BUY or SELL with size > 0, create a new row in
               pending_trades and optionally place a live Kraken order through
               self._maybe_place_kraken_order(...).
             - For BUY actions, call self.risk_manager.add_or_combine_lot(...) to maintain
               the open-lot tracking.
             - Store the final action, size, and rationale in ai_decisions via
               self._store_decision(...).

        :param aggregator_list_simple: A list of dictionaries where each entry includes at
               least {"pair": "...", "price": float, "prompt_text": "..."} used in GPT's
               simple prompt. The "price" is used to calculate trade cost or clamp sizes.
        :param trade_history: A list of strings summarizing recent trades; typically
               truncated to max_trades items before use in GPT.
        :param max_trades: The max number of trade-history lines to provide to GPT.
        :param input_open_positions: A list of strings describing currently open positions,
               if you want GPT to see them. (Optional usage in the prompt.)
        :param current_balance: The user’s USD (or main currency) available.
        :param current_trade_balance: A dictionary like {"ETH/USD": 0.5, "XBT/USD": 0.01}
               representing coin holdings, used by GPT for context.
        :return: A dictionary mapping { "PAIR": ("ACTION", final_size) } reflecting the
                 final decisions after risk checks and pending-trade creation.
        """
        results_set: Dict[str, Tuple[str, float]] = {}

        # 1) Early exit if GPT is disabled or aggregator data is empty
        if not self.use_openai or not self.gpt_manager or not aggregator_list_simple:
            logger.info("[AIStrategy-simple] GPT disabled or aggregator_list_simple empty => HOLD all.")
            for item in aggregator_list_simple:
                pair_name = item.get("pair", "UNK")
                results_set[pair_name] = ("HOLD", 0.0)
            return results_set

        # 2) Invoke GPT with a simple prompt approach
        try:
            gpt_result = self.gpt_manager.generate_multi_trade_decision_simple_prompt(
                aggregator_list_simple=aggregator_list_simple,
                reflection_enabled=False,
                current_balance=current_balance,
                current_trade_balance=current_trade_balance
            )
        except Exception as e:
            logger.exception("[AIStrategy-simple] GPT call failed => fallback => HOLD.")
            for item in aggregator_list_simple:
                pair_name = item.get("pair", "UNK")
                results_set[pair_name] = ("HOLD", 0.0)
            return results_set

        decisions_array = gpt_result.get("decisions", [])
        if not decisions_array:
            logger.warning("[AIStrategy-simple] GPT returned no decisions => hold all.")
            for item in aggregator_list_simple:
                pair_name = item.get("pair", "UNK")
                results_set[pair_name] = ("HOLD", 0.0)
            return results_set

        # Extract a shared rationale if GPT provided one
        ai_rationale = gpt_result.get("rationale", "No GPT rationale provided")

        # 3) Process each GPT decision => clamp, pass to risk_manager, create pending trades
        for decision in decisions_array:
            pair_name = decision.get("pair", "UNK")
            action_raw = decision.get("action", "HOLD").upper()
            suggested_size = float(decision.get("size", 0.0))

            # Find the aggregator entry for a valid last price
            found_item = next((x for x in aggregator_list_simple if x.get("pair") == pair_name), None)
            current_price = 0.0
            if found_item is not None:
                current_price = float(found_item.get("price", 0.0))

            if current_price <= 0:
                logger.warning(f"[AIStrategy-simple] No valid price for {pair_name} => forcing HOLD.")
                self._store_decision(pair_name, "HOLD", 0.0, "No valid price => hold", group_rationale=ai_rationale)
                results_set[pair_name] = ("HOLD", 0.0)
                continue

            # Post-validate (e.g. ensure size or cost is not nonsensical)
            final_action, final_size = self._post_validate(action_raw, suggested_size, current_price)

            # daily drawdown / capacity checks via risk_manager
            final_action, final_size = self.risk_manager.adjust_trade(
                final_action,
                final_size,
                pair_name,
                current_price,
                rest_manager.fetch_balance()
            )

            # If after all checks we still have a trade, create a pending_trades row
            rationale_str = f"GPT simple => {pair_name} => final={final_action}, size={final_size}"
            if final_action.upper() in ("BUY", "SELL") and final_size > 0:
                pending_id = create_pending_trade(
                    side=final_action,
                    requested_qty=final_size,
                    pair=pair_name,
                    reason="GPT_MULTI_DECISION_SIMPLE",
                    source="ai_strategy_simple",
                    rationale=ai_rationale
                )
                if pending_id:
                    logger.info(f"[AIStrategy] Successfully created pending trade with ID {pending_id}")
                else:
                    logger.error(f"[AIStrategy] Failed to create pending trade for {pair}")
                self._maybe_place_kraken_order(
                    pair=pair_name,
                    action=final_action,
                    volume=final_size,
                    pending_id=pending_id
                )
                if final_action.upper() == "BUY":
                    # Merge or add new lot in holding_lots
                    self.risk_manager.add_or_combine_lot(
                        pair=pair_name,
                        buy_quantity=final_size,
                        buy_price=current_price
                    )

            # Record the final result in ai_decisions and in the returned dict
            self._store_decision(pair_name, final_action, final_size, rationale_str, group_rationale=ai_rationale)
            results_set[pair_name] = (final_action, final_size)

        return results_set

    # --------------------------------------------------------------------------
    # HELPER: aggregator_text builder (example code)
    # --------------------------------------------------------------------------
    def _build_aggregator_text_changes(self, pair: str) -> str:
        """
        Illustrative aggregator text builder referencing aggregator_summaries for 1-hour-ago changes.
        You can adapt or remove if not used.
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

        def format_change(curr, old, label):
            if curr is None or old is None:
                return f"{label}={curr} (NA)"
            try:
                diff = curr - old
                sign = f"{diff:+.2f}"
                return f"{label}={curr} ({sign})"
            except:
                return f"{label}={curr} (NA)"

        gs_str = format_change(current_data["galaxy_score"], old_data["galaxy_score"], "galaxy_score")
        ar_str = format_change(current_data["alt_rank"], old_data["alt_rank"], "alt_rank")
        sent_str = f"sentiment={current_data['sentiment']} (old={old_data['sentiment']})"

        aggregator_text = (
            f"Pair={pair} => {gs_str}, {ar_str}, {sent_str}\n"
            "(Note: aggregator changes. Expand if more fields are stored.)"
        )
        return aggregator_text

    # --------------------------------------------------------------------------
    # HELPER: post-validate (e.g. min buy check)
    # --------------------------------------------------------------------------
    def _post_validate(self, action: str, size_suggested: float, current_price: float) -> Tuple[str, float]:
        """
        A quick local clamp. E.g. if there's a minimum buy cost or size.
        We rely on risk_manager_db for daily drawdown, so we keep this minimal.
        """
        if action not in ("BUY", "SELL"):
            return ("HOLD", 0.0)

        # You can do cost or size checks here
        cost = size_suggested * current_price
        rc = self.risk_controls
        if not rc:
            return (action, size_suggested)

        # Example usage if you want a minimum buy cost:
        # min_buy = rc.get("minimum_buy_amount", 5.0)
        # if action == "BUY" and cost < min_buy:
        #     logger.info(f"[post_validate] cost={cost:.2f} < min_buy={min_buy:.2f} => hold")
        #     return ("HOLD", 0.0)

        return (action, size_suggested)

    # --------------------------------------------------------------------------
    # DB Setup => ai_decisions
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
                rationale TEXT,
                group_rationale TEXT
            )
            """)
            conn.commit()
        except Exception as e:
            logger.exception(f"[AIStrategy] Error creating ai_decisions => {e}")
        finally:
            conn.close()

    def _store_decision(self, pair: str, action: str, size: float, rationale: str, group_rationale: str = None):
        """
        Insert a record into 'ai_decisions' reflecting the final GPT or fallback decision.
        """
        logger.debug(f"[AIStrategy] Storing decision => pair={pair}, action={action}, size={size}, reason={rationale}")
        conn = sqlite3.connect(DB_FILE)
        try:
            c = conn.cursor()
            c.execute("""
            INSERT INTO ai_decisions (timestamp, pair, action, size, rationale, group_rationale)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (int(time.time()), pair, action, size, rationale, group_rationale))
            conn.commit()
        except Exception as e:
            logger.exception(f"[AIStrategy] Error storing AI decision => {e}")
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # HELPER: Summarize recent trades
    # --------------------------------------------------------------------------
    def _summarize_recent_trades(self, pair: str, limit: int = 3) -> str:
        """
        Returns a short summary of the last 'limit' trades from 'trades' for the given pair.
        Example:
            "1) 2025-01-01 10:00 BUY 0.001@2000\n2) 2025-01-02 11:00 SELL 0.001@2100"
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
            # Reverse them so oldest is first
            rows = rows[::-1]
            for i, row in enumerate(rows, start=1):
                t, sd, qty, px = row
                import datetime
                dt_s = datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")
                out.append(f"{i}) {dt_s} {sd.upper()} {qty}@{px}")
        except Exception as e:
            logger.exception(f"[AIStrategy] Error summarizing trades => {e}")
            return "Error retrieving trades."
        finally:
            conn.close()

        return "\n".join(out)

    # --------------------------------------------------------------------------
    # HELPER: possibly place real order
    # --------------------------------------------------------------------------
    def _maybe_place_kraken_order(self, pair: str, action: str, volume: float, pending_id: int = None):
        """
        If place_live_orders=True and private_ws_client is available, we send a simple
        market order. 'pending_id' is used as userref to match pending_trades.
        """
        if not self.place_live_orders:
            return
        if not self.private_ws_client:
            logger.warning(
                f"[AIStrategy] place_live_orders=True but private_ws_client is None => cannot place real order."
            )
            return

        side_for_kraken = "buy" if action.upper() == "BUY" else "sell"
        logger.info(
            f"[AIStrategy] Sending real order => pair={pair}, side={side_for_kraken}, "
            f"volume={volume}, pending_id={pending_id}"
        )
        self.private_ws_client.send_order(
            pair=pair,
            side=side_for_kraken,
            ordertype="market",
            volume=volume,
            userref=str(pending_id) if pending_id else None
        )
