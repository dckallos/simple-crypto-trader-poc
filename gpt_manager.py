"""
gpt_manager.py

A production-ready module to handle GPT-based inference for **multi-coin** and single-coin
trading decisions. This version offers two main methods:

    (1) generate_trade_decision(...)      -- single-coin approach
    (2) generate_multi_trade_decision(...)-- multi-coin approach

Usage Example:

    from gpt_manager import GPTManager

    gpt = GPTManager(
        api_key="sk-...",
        model="gpt-4o",
        temperature=0.8,
        max_tokens=1000
    )

    # Single-coin usage:
    single_result = gpt.generate_trade_decision(
        conversation_context="Summarized conversation so far.",
        aggregator_text="rsi=44, price=1850, galaxy_score=60 ...",
        trade_history=["2025-01-20 10:15 BUY ETH 0.001@25000", "..."],
        max_trades=5,
        risk_controls={"initial_spending_account":100.0},
        open_positions=None
    )
    # single_result => {"action":"BUY","size":0.001}

    # Multi-coin usage:
    aggregator_list = [
        {"pair":"ETH/USD","aggregator_data":"rsi=44, price=1850, galaxy_score=60..."},
        {"pair":"XBT/USD","aggregator_data":"rsi=50, price=28000, galaxy_score=65..."}
    ]
    multi_result = gpt.generate_multi_trade_decision(
        conversation_context="Summarized conversation so far",
        aggregator_list=aggregator_list,
        open_positions=["ETH/USD LONG 0.002@1860.0","XBT/USD SHORT 0.001@29500.0"],
        trade_history=["2025-01-20 BUY ETH/USD 0.001@25000","2025-01-21 SELL ETH/USD ..."],
        max_trades=5,
        risk_controls={"initial_spending_account":100.0},
    )
    # e.g. => {
    #   "decisions": [
    #       {"pair":"ETH/USD","action":"BUY","size":0.0005},
    #       {"pair":"XBT/USD","action":"HOLD","size":0.0},
    #   ]
    # }

"""


import os
import json
import logging
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError(
        "Missing the new 'openai' Python package (≥ v1). "
        "Install via: pip install --upgrade openai"
    ) from e

logger = logging.getLogger(__name__)


class GPTManager:
    """
    GPTManager orchestrates GPT logic for advanced single-coin and multi-coin
    trading decisions.

    Key Public Methods:
        generate_trade_decision(...)        => Single-coin decision
        generate_multi_trade_decision(...)  => Multi-coin holistic approach
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 500,
        **client_options
    ):
        """
        :param api_key:        The API key to use; can also come from OPENAI_API_KEY env var.
        :param model:          GPT model name, e.g. "gpt-4o".
        :param temperature:    Sampling temperature for GPT calls.
        :param max_tokens:     The max tokens in GPT responses.
        :param client_options: Additional arguments for the openai.OpenAI constructor
                               (e.g. timeout, max_retries, proxies).
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            logger.warning(
                "No OPENAI_API_KEY found. GPT calls may fail unless an environment "
                "variable or explicit api_key is set."
            )

        self.client = OpenAI(api_key=self.api_key, **client_options)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_trade_decision(
        self,
        conversation_context: str,
        aggregator_text: str,
        trade_history: List[str],
        max_trades: int,
        risk_controls: Dict[str, Any],
        open_positions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generates a single-coin trading decision (action & size) by calling GPT.

        Expects GPT to return JSON of the form:
            {
                "action": "BUY|SELL|HOLD",
                "size": 0.001
            }

        Args:
            conversation_context: Summarized or partial conversation memory from prior usage.
            aggregator_text: The aggregator data string, e.g. "rsi=44, price=1850, sentiment=0.4".
            trade_history: Last few trades in string form. We'll use up to max_trades lines.
            max_trades: How many lines from trade_history to include in the prompt.
            risk_controls: Dict with relevant constraints, e.g. minimum_buy_amount, etc.
            open_positions: Optional list describing open sub-positions.

        Returns:
            A dictionary with "action" and "size", or fallback => {"action":"HOLD","size":0.0}
        """
        # 1) Summarize trade history (take last max_trades)
        truncated_history = trade_history[-max_trades:] if trade_history else []
        trade_summary = "\n".join(truncated_history) if truncated_history else "No trade history."

        # 2) Summarize open positions (optional)
        if open_positions:
            open_pos_summary = "\n".join(open_positions)
        else:
            open_pos_summary = "No open positions."

        # Build user prompt (for single pair)
        user_prompt = (
            f"Conversation context:\n{conversation_context}\n\n"
            f"Aggregator data:\n{aggregator_text}\n\n"
            f"Recent trades (up to {max_trades}):\n{trade_summary}\n\n"
            f"Open positions:\n{open_pos_summary}\n\n"
            f"Risk controls => {risk_controls}\n\n"
            "Return strictly JSON => {\"action\":\"BUY|SELL|HOLD\",\"size\":float}\n"
        )
        # We'll create a system message that clarifies the role
        system_msg = {
            "role": "assistant",
            "content": (
                "You are a specialized single-coin trading assistant. "
                "You must respond with a single JSON object of the form:\n"
                "{\"action\":\"BUY|SELL|HOLD\",\"size\":0.001}\n"
                "No extra text, code blocks, or explanations. "
            )
        }
        user_msg = {"role": "user", "content": user_prompt}

        # 3) GPT call
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[system_msg, user_msg],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        except Exception as e:
            logger.exception("Error contacting GPT => fallback => empty.")
            return {"action": "HOLD", "size": 0.0}

        if not response.choices:
            logger.warning("No GPT choices => fallback => empty decisions.")
            return {"action": "HOLD", "size": 0.0}

        choice = response.choices[0]
        raw_text = choice.message.content or ""
        finish_reason = choice.finish_reason
        logger.debug(f"[GPT-Single] raw={raw_text}, finish_reason={finish_reason}")

        # Remove code fences if any
        cleaned = raw_text.replace("```json", "").replace("```", "").strip()

        # parse JSON
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Could not parse single-coin GPT response => fallback => hold")
            return {"action": "HOLD", "size": 0.0}

        if not isinstance(parsed, dict):
            logger.warning("Parsed JSON not a dict => fallback => hold")
            return {"action": "HOLD", "size": 0.0}

        action = parsed.get("action", "HOLD").upper()
        size = float(parsed.get("size", 0.0))

        # Minimal validation
        if action not in ("BUY","SELL","HOLD"):
            action = "HOLD"
        if size < 0.0:
            size = 0.0

        return {"action": action, "size": size}

    def generate_multi_trade_decision(
        self,
        conversation_context: str,
        aggregator_list: List[Dict[str, Any]],
        open_positions: List[str],
        trade_history: List[str],
        max_trades: int,
        risk_controls: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        In a single GPT call, pass aggregator data for multiple coins so GPT
        can produce a global best decision across them. We expect a JSON result:
            {
              "decisions": [
                {"pair":"ETH/USD","action":"BUY|SELL|HOLD","size":0.001},
                ...
              ]
            }

        Args:
            conversation_context: str => prior GPT conversation or summary.
            aggregator_list: Each item => {"pair":"X/Y","aggregator_data":"..."}
            open_positions: list of strings describing open positions.
            trade_history: list of strings describing recent trades.
            max_trades: how many lines from trade_history to pass.
            risk_controls: dict with constraints, e.g. min buy size, purchase_upper_limit_percent, etc.

        Returns:
            A dictionary => {"decisions": [...]} or fallback => {"decisions":[]}.
        """
        # Summarize aggregator data
        aggregator_text_block = []
        for idx, item in enumerate(aggregator_list, start=1):
            p = item.get("pair","UNK")
            data_str = item.get("aggregator_data","")
            aggregator_text_block.append(f"{idx}) {p} => {data_str}")
        aggregator_text_joined = "\n".join(aggregator_text_block)

        # Summarize open_positions
        if open_positions:
            open_pos_str = "\n".join(open_positions)
        else:
            open_pos_str = "No open positions."

        # Summarize trade_history
        truncated = trade_history[-max_trades:] if trade_history else []
        trade_summary = "\n".join(truncated) if truncated else "No past trades found."

        # Build system & user messages
        system_msg = {
            "role": "assistant",
            "content": (
                "You are a specialized multi-coin crypto trading assistant. "
                "You see aggregator data for multiple coins, plus open positions. "
                "You MUST return a single JSON object of the form:\n"
                "{ \"decisions\": [ {\"pair\":\"COIN_PAIR\",\"action\":\"BUY|SELL|HOLD\",\"size\":0.001}, ... ] }\n"
                "No extra text or code blocks. If you want to close a position, use 'SELL' with an appropriate 'size'."
            )
        }
        user_prompt = (
            f"Conversation context:\n{conversation_context}\n\n"
            f"Aggregators for all coins =>\n{aggregator_text_joined}\n\n"
            f"Open positions =>\n{open_pos_str}\n\n"
            f"Recent trades =>\n{trade_summary}\n\n"
            f"Risk controls => {risk_controls}\n\n"
            "Return strictly JSON => {\"decisions\":[...]}\n"
        )
        user_msg = {"role": "user", "content": user_prompt}

        # GPT call
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[system_msg, user_msg],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        except Exception as e:
            logger.exception("[GPT-Multi] Error => fallback => empty decisions")
            return {"decisions": []}

        if not response.choices:
            logger.warning("[GPT-Multi] No GPT choices => fallback => empty decisions")
            return {"decisions": []}

        choice = response.choices[0]
        raw_text = choice.message.content or ""
        finish_reason = choice.finish_reason
        logger.debug(f"[GPT-Multi] raw={raw_text}, finish_reason={finish_reason}")

        cleaned = raw_text.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("[GPT-Multi] JSON parse error => fallback => empty decisions")
            return {"decisions": []}

        if not isinstance(parsed, dict):
            logger.warning("[GPT-Multi] JSON was not a dict => fallback => empty decisions")
            return {"decisions": []}

        decisions_list = parsed.get("decisions", [])
        if not isinstance(decisions_list, list):
            logger.warning("[GPT-Multi] 'decisions' not a list => fallback => []")
            return {"decisions": []}

        final_out = []
        for d in decisions_list:
            if not isinstance(d, dict):
                logger.warning(f"[GPT-Multi] invalid item => {d}")
                continue
            pair_name = str(d.get("pair","UNK"))
            action = str(d.get("action","HOLD")).upper()
            size = float(d.get("size",0.0))
            final_out.append({"pair": pair_name, "action": action, "size": size})

        return {"decisions": final_out}
