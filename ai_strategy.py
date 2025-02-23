#!/usr/bin/env python3
# =============================================================================
# FILE: ai_strategy.py
# =============================================================================
"""
ai_strategy.py (Refactored, Production-Ready)

AIStrategy is now streamlined for a *single aggregator process* using a Mustache-
rendered prompt. We no longer distinguish between “multi_coins” vs. “simple_prompt”
flows. Instead, we call `gpt_manager.generate_decisions_from_prompt(...)` with
a single final prompt string, parse the JSON decisions, and finalize trades.

High-Level Workflow:
--------------------
1) The aggregator (in main.py or HybridApp) gathers data (e.g. ledger, price,
   trade history).
2) The aggregator **renders** a Mustache template. That template includes all
   disclaimers, instructions, aggregator data, etc.
3) AIStrategy => `predict_from_prompt(final_prompt_text=...)` is called with
   the final, pre-rendered prompt string.
4) AIStrategy calls GPTManager => parse final JSON => {decisions,rationale}.
5) AIStrategy enforces daily drawdown, position sizing, etc. via RiskManagerDB.
6) We create new pending trades in `pending_trades` if GPT says BUY/SELL.
7) If `place_live_orders=True` and a private_ws_client is available, we place
   actual orders on Kraken.

Stop-loss & take-profit remain outside this logic (risk_manager or ws_data_feed).

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
from config_loader import ConfigLoader

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


# A global rest_manager instance if you want to query balances from Kraken
rest_manager = KrakenRestManager(
    os.getenv("KRAKEN_API_KEY"),
    os.getenv("KRAKEN_SECRET_API_KEY")
)


class AIStrategy:
    """
    AIStrategy: Decides trades from GPT or fallback, using a single aggregator cycle approach.
    Pseudocode:

    aggregator =>
        1) gather data from ledger, price, trade history
        2) render Mustache => final_prompt_text
        3) decisions = ais.strategy.predict_from_prompt(final_prompt_text, trade_balances, etc.)
        4) aggregator logs decisions or creates pending trades

    This class:
    - Takes the final prompt text
    - Calls GPT
    - Risk checks each decision (daily drawdown, position sizing)
    - Creates pending trades
    - Possibly calls private_ws_client.send_order(...) if enabled
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
        private_ws_client=None
    ):
        """
        :param pairs: e.g. ["ETH/USD", "XBT/USD"]
        :param use_openai: True => GPT logic is used; else fallback logic
        :param max_position_size: For local clamp usage
        :param max_daily_drawdown: e.g. -0.02 => skip new trades if daily PnL < -2%
        :param risk_controls: optional dict with custom fields
        :param risk_manager: a RiskManagerDB instance
        :param gpt_model: GPT model name override
        :param gpt_temperature: GPT model temperature
        :param gpt_max_tokens: GPT model max tokens
        :param private_ws_client: KrakenPrivateWSClient for live orders (optional)
        """
        self.pairs = pairs if pairs else []
        self.use_openai = use_openai
        self.risk_manager = risk_manager
        self.risk_controls = risk_controls or {}
        self.private_ws_client = private_ws_client

        # GPT Initialization
        self.gpt_manager = None
        if self.use_openai:
            self.gpt_manager = GPTManager(
                config_file="config.yaml",
                temperature=gpt_temperature,
                max_tokens=gpt_max_tokens,
                log_gpt_calls=True
            )

        # Ensure ai_decisions table exists
        self._create_ai_decisions_table()

        # For local clamp usage
        self.max_position_size = max_position_size
        self.max_daily_drawdown = max_daily_drawdown

    def predict_from_prompt(
        self,
        final_prompt_text: str,
        current_trade_balance: Dict[str, float],
        current_balance: float = 0.0
    ) -> Tuple[Dict[str, Tuple[str, float]], Dict[str, Any]]:
        """
        The unified aggregator approach:
          1) If GPT is disabled, do a fallback => hold all
          2) Otherwise, call GPT => parse => { "decisions":[...], "rationale":"..." }
          3) For each decision => check daily drawdown, size clamp => create pending trade => possibly place real order
          4) Return decisions and the raw GPT output (for confidence and rationale).

        :param final_prompt_text: The entire Mustache-rendered prompt that includes instructions + aggregator data
        :param current_trade_balance: e.g. {"ETH/USD": 0.5, "XBT/USD": 0.01} user holdings
        :param current_balance: USD (or similar) currently available
        :return: A tuple containing decisions { "PAIR": ("ACTION","SIZE") } and the raw GPT output
        """
        # If GPT is disabled, skip
        if not self.use_openai or not self.gpt_manager:
            return {p: ("HOLD", 0.0) for p in self.pairs}, {"decisions": [], "rationale": "GPT disabled", "confidence": 0.0}

        # Retrieve the actual GPT model name from config
        self.gpt_manager.model = ConfigLoader.get_value("openai_model", "o1-mini")
        ai_model_name = self.gpt_manager.model

        results_set = {}

        logger.info("[AIStrategy] Submitting final prompt to GPT => generate_decisions_from_prompt.")
        gpt_out = self.gpt_manager.generate_decisions_from_prompt(final_prompt_text)
        decisions_list = gpt_out.get("decisions", [])

        if not decisions_list:
            logger.warning("[AIStrategy] GPT returned no decisions => hold all.")
            return {p: ("HOLD", 0.0) for p in self.pairs}, gpt_out

        # For each decision => local checks => create pending trades => possibly place order
        for dec in decisions_list:
            pair = dec.get("pair", "UNK")
            action_raw = dec.get("action", "HOLD").upper()
            size_requested = float(dec.get("size", 0.0))

            # If GPT lists a pair we aren't tracking, skip
            if pair not in self.pairs:
                logger.warning(f"[AIStrategy] GPT returned {pair}, but not in self.pairs => skipping.")
                results_set[pair] = ("HOLD", 0.0)
                continue

            # Quick local clamp => check daily drawdown => skip if needed
            final_action, final_size = self._post_validate(action_raw, size_requested, pair, current_balance)

            # Then pass to risk_manager => ensures min cost, enough balance, etc.
            final_action, final_size = self.risk_manager.adjust_trade(
                final_action,
                final_size,
                pair,
                self._get_price_estimate(pair),
                rest_manager.fetch_balance()
            )

            # If final is still BUY or SELL => create a pending trade => maybe place real order
            if final_action in ("BUY", "SELL") and final_size > 0:
                # We'll embed the GPT rationale + model name in the final trade record
                combined_rationale = f"{gpt_out.get('rationale', 'No rationale')}\n[ai_model={ai_model_name}]"

                pending_id = create_pending_trade(
                    side=final_action,
                    requested_qty=final_size,
                    pair=pair,
                    reason="GPT_AGGREGATION",
                    source="ai_strategy",
                    rationale=combined_rationale
                )
                if pending_id:
                    logger.info(f"[AIStrategy] Created pending trade ID {pending_id} => {final_action} {final_size} {pair}")

                    self._maybe_place_kraken_order(pair, final_action, final_size, pending_id)

                    # If BUY => incorporate into or create a new lot
                    if final_action == "BUY":
                        self.risk_manager.add_or_combine_lot(
                            pair=pair,
                            buy_quantity=final_size,
                            buy_price=self._get_price_estimate(pair),
                            origin_source="ai_strategy",
                            # Optionally store the same rationale in the lot
                            risk_params_json=json.dumps(self.risk_controls)
                        )
                else:
                    logger.error(f"[AIStrategy] Failed creating pending trade for {pair}; skipping order.")
                results_set[pair] = (final_action, final_size)
            else:
                # GPT said HOLD or risk checks forced us to skip
                results_set[pair] = ("HOLD", 0.0)

        return results_set, gpt_out

    # --------------------------------------------------------------------------
    # Local clamp => daily drawdown
    # --------------------------------------------------------------------------
    def _post_validate(self, action: str, suggested_size: float, pair: str, current_balance: float) -> Tuple[str, float]:
        """
        Minimal local clamp:
          1) If daily drawdown < max => skip buy
          2) If suggested_size > max_position_size => clamp
        """
        if action not in ("BUY", "SELL"):
            return ("HOLD", 0.0)

        # If daily drawdown is below threshold => skip buys
        daily_pnl = self.risk_manager.get_daily_realized_pnl()
        if daily_pnl <= self.max_daily_drawdown and action == "BUY":
            logger.warning(f"[AIStrategy] dailyPnL={daily_pnl:.4f} < {self.max_daily_drawdown} => skip buy => HOLD")
            return ("HOLD", 0.0)

        # clamp
        final_size = min(suggested_size, self.max_position_size)
        if final_size <= 0:
            return ("HOLD", 0.0)

        return (action, final_size)

    def _get_price_estimate(self, pair: str) -> float:
        """
        Returns the last known 'last_price' from price_history or 0 if none.
        """
        from db import fetch_price_history_desc
        rows = fetch_price_history_desc(pair, limit=1)
        if rows:
            return float(rows[0]["last_price"] or 0.0)
        return 0.0

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

    def _store_decision(
        self,
        pair: str,
        action: str,
        size: float,
        rationale: str,
        group_rationale: str = None
    ):
        """
        Insert a record into 'ai_decisions' capturing the final action, size, and rationale.
        This is purely for logging or analysis. The real trades are in 'pending_trades' + 'trades'.
        """
        logger.debug(f"[AIStrategy] Storing decision => pair={pair}, action={action}, size={size}, reason={rationale}")
        conn = sqlite3.connect(DB_FILE)
        try:
            c = conn.cursor()
            c.execute("""
                INSERT INTO ai_decisions (timestamp, pair, action, size, rationale, group_rationale)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                int(time.time()),
                pair,
                action,
                size,
                rationale,
                group_rationale
            ))
            conn.commit()
        except Exception as e:
            logger.exception(f"[AIStrategy] Error storing AI decision => {e}")
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # Possibly place real order
    # --------------------------------------------------------------------------
    def _maybe_place_kraken_order(self, pair: str, action: str, volume: float, pending_id: int = None):
        """
        If place_live_orders=True, tries to submit a market order via private_ws_client.
        'pending_id' is used as userref in Kraken's system for tracking.
        """
        place_live = ConfigLoader.get_value("place_live_orders", False)
        if not place_live:
            logger.info("[AIStrategy] place_live_orders=False => skipping real order.")
            return

        if not self.private_ws_client:
            logger.warning("[AIStrategy] No private_ws_client => cannot place live order.")
            return

        side_for_kraken = "buy" if action.upper() == "BUY" else "sell"
        logger.info(
            f"[AIStrategy] Placing real order => {pair}, side={side_for_kraken}, volume={volume}, userref={pending_id}"
        )
        self.private_ws_client.send_order(
            pair=pair,
            side=side_for_kraken,
            ordertype="market",
            volume=volume,
            userref=str(pending_id) if pending_id else None
        )
