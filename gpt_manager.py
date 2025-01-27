"""
gpt_manager.py

A production-ready module to handle GPT-based inference for single-coin and multi-coin
trading decisions, now with a modular approach to aggregator data.

Changes:
    - We introduce build_aggregator_prompt(...) to unify aggregator summary, local classifier
      probability, and embedding vector into a single short text or JSON snippet for GPT.
    - generate_trade_decision(...) and generate_multi_trade_decision(...) can be fed
      aggregator text built by that helper. That drastically reduces the amount of raw data
      we pass to GPT, because we only pass "prob_up=0.72" or "embedding_vector=[0.12,-0.09,0.31]"
      plus a short summary.

Usage Example:
    from gpt_manager import GPTManager, build_aggregator_prompt

    # Suppose aggregator_summaries row => "price_bucket=high, sentiment_label=slightly_positive"
    # local classifier => prob_up=0.72
    # embeddings => [0.25, -0.1, 0.36]
    aggregator_text = build_aggregator_prompt(
        summary="price_bucket=high, sentiment_label=slightly_positive, galaxy_score=64",
        probability=0.72,
        embedding=[0.25, -0.1, 0.36],
    )
    # aggregator_text => "summary: price_bucket=high, sentiment_label=slightly_positive, galaxy_score=64; probability=0.72; embedding=[0.25,-0.1,0.36]"

    gpt = GPTManager(api_key="sk-...", model="gpt-4o")

    single_result = gpt.generate_trade_decision(
        conversation_context="My prior conversation context here.",
        aggregator_text=aggregator_text,
        trade_history=["2025-01-21 10:00 SELL BTC 0.001@21000", "..."],
        max_trades=5,
        risk_controls={"initial_spending_account":100.0},
        open_positions=["BTC/USD LONG 0.002, entry=22000.0"]
    )
    # => e.g. {"action":"BUY", "size":0.0005}
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
logging.basicConfig(level=logging.INFO)


def build_aggregator_prompt(
    summary: Optional[str] = None,
    probability: Optional[float] = None,
    embedding: Optional[List[float]] = None,
    additional_fields: Optional[Dict[str, Any]] = None
) -> str:
    """
    A helper function to unify aggregator data (summaries, local classifier probability,
    embedding vectors) into a short text snippet for GPT.

    Args:
        summary: A short textual summary (e.g. "price_bucket=high, sentiment_label=positive").
        probability: The local classifier probability of "UP" or "positive outcome" (float).
        embedding: A small list of floats from PCA or embeddings (e.g. [0.2, -0.1, 0.36]).
        additional_fields: If you want to pass more custom data. We'll convert them to text.

    Returns:
        A single string that concisely describes aggregator data to GPT. For example:
          "summary: price_bucket=high, sentiment_label=slightly_positive; probability=0.72; embedding=[0.25,-0.1,0.36]"
    """
    parts = []
    if summary:
        parts.append(f"summary: {summary}")

    if probability is not None:
        # rounding or direct
        parts.append(f"probability={round(probability, 4)}")

    if embedding:
        # convert to short list string
        emb_str = "[" + ",".join(str(round(v, 3)) for v in embedding) + "]"
        parts.append(f"embedding_vector={emb_str}")

    if additional_fields:
        # e.g. {"volatility":"high"} => "volatility=high"
        add_list = [f"{k}={v}" for k, v in additional_fields.items()]
        joined_add = ", ".join(add_list)
        parts.append(f"additional=({joined_add})")

    # combine everything in a single line
    aggregator_text = "; ".join(parts)
    return aggregator_text


class GPTManager:
    """
    GPTManager orchestrates GPT logic for advanced single-coin and multi-coin
    trading decisions, now with a more modular aggregator approach.

    Key Public Methods:
        generate_trade_decision(...)        => Single-coin decision
        generate_multi_trade_decision(...)  => Multi-coin approach
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
            aggregator_text: The aggregator snippet built via `build_aggregator_prompt` or manually.
            trade_history: Last few trades in string form. We'll use up to max_trades lines.
            max_trades: How many lines from trade_history to include in the prompt.
            risk_controls: Dict with relevant constraints, e.g. minimum_buy_amount, etc.
            open_positions: Optional list describing open sub-positions.

        Returns:
            A dictionary with "action" and "size", or fallback => {"action":"HOLD","size":0.0}
        """
        # Summarize trade history (take last max_trades)
        truncated_history = trade_history[-max_trades:] if trade_history else []
        trade_summary = "\n".join(truncated_history) if truncated_history else "No trade history."

        # Summarize open positions
        if open_positions:
            open_pos_summary = "\n".join(open_positions)
        else:
            open_pos_summary = "No open positions."

        # Build user prompt for single pair
        user_prompt = (
            f"Conversation context:\n{conversation_context}\n\n"
            f"Aggregator data => {aggregator_text}\n\n"
            f"Recent trades (up to {max_trades}):\n{trade_summary}\n\n"
            f"Open positions:\n{open_pos_summary}\n\n"
            f"Risk controls => {risk_controls}\n\n"
            "Return strictly JSON => {\"action\":\"BUY|SELL|HOLD\",\"size\":float}\n"
        )
        system_msg = {
            "role": "assistant",
            "content": (
                "You are a specialized single-coin trading assistant. "
                "You must respond with a single JSON object of the form:\n"
                "{\"action\":\"BUY|SELL|HOLD\",\"size\":0.001}\n"
                "No extra text or code blocks."
            )
        }
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
            logger.exception("Error contacting GPT => fallback => hold.")
            return {"action": "HOLD", "size": 0.0}

        if not response.choices:
            logger.warning("No GPT choices => fallback => hold.")
            return {"action": "HOLD", "size": 0.0}

        choice = response.choices[0]
        raw_text = choice.message.content or ""
        logger.debug(f"[GPT-Single] raw={raw_text}, reason={choice.finish_reason}")

        # Remove possible code fences
        cleaned = raw_text.replace("```json", "").replace("```", "").strip()

        # parse
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Could not parse single-coin GPT response => fallback => hold.")
            return {"action": "HOLD", "size": 0.0}

        if not isinstance(parsed, dict):
            logger.warning("Parsed JSON not a dict => fallback => hold.")
            return {"action": "HOLD", "size": 0.0}

        action = parsed.get("action", "HOLD").upper()
        size = float(parsed.get("size", 0.0))
        if action not in ("BUY","SELL","HOLD"):
            action = "HOLD"
        if size < 0:
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
        can produce a global best decision across them.

        aggregator_list => each item => { "pair":"ETH/USD","aggregator_data":"some short text" }
          e.g. aggregator_data might be "summary: price_bucket=high, sentiment_label=slightly_positive; probability=0.72"

        We expect GPT to return a JSON result:
            {
              "decisions": [
                {"pair":"ETH/USD", "action":"BUY|SELL|HOLD", "size":0.001},
                ...
              ]
            }

        Args:
            conversation_context: str => prior GPT conversation or summary.
            aggregator_list: A list => each item has "pair" + "aggregator_data" (short text).
            open_positions: list of strings describing open positions.
            trade_history: list of strings describing recent trades.
            max_trades: how many lines from trade_history to pass.
            risk_controls: dict with relevant constraints.

        Returns:
            A dict => {"decisions": [...]} or fallback => {"decisions":[]}
        """
        # Summarize aggregator data
        aggregator_text_block = []
        for idx, item in enumerate(aggregator_list, start=1):
            p = item.get("pair","UNK")
            data_str = item.get("aggregator_data","")
            aggregator_text_block.append(f"{idx}) {p} => {data_str}")
        aggregator_text_joined = "\n".join(aggregator_text_block)

        # Summarize open_positions
        open_pos_str = "\n".join(open_positions) if open_positions else "No open positions."

        # Summarize trade_history
        truncated = trade_history[-max_trades:] if trade_history else []
        trade_summary = "\n".join(truncated) if truncated else "No past trades found."

        # Build system & user messages
        system_msg = {
            "role": "assistant",
            "content": (
                "You are a specialized multi-coin crypto trading assistant. "
                "You see aggregator data for multiple coins at once, plus open positions. "
                "You MUST return a single JSON object of the form:\n"
                "{ \"decisions\": [ {\"pair\":\"COIN_PAIR\",\"action\":\"BUY|SELL|HOLD\",\"size\":0.001}, ... ] }\n"
                "No extra text or code blocks. If you want to close or reduce a position, "
                "use 'SELL' with an appropriate 'size'."
            )
        }
        user_prompt = (
            f"Conversation context:\n{conversation_context}\n\n"
            f"Aggregators =>\n{aggregator_text_joined}\n\n"
            f"Open positions =>\n{open_pos_str}\n\n"
            f"Recent trades =>\n{trade_summary}\n\n"
            f"Risk controls => {risk_controls}\n\n"
            "Return strictly JSON => {\"decisions\":[...]}\n"
        )
        user_msg = {"role": "user", "content": user_prompt}

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
            logger.warning("[GPT-Multi] No GPT choices => fallback => []")
            return {"decisions": []}

        choice = response.choices[0]
        raw_text = choice.message.content or ""
        logger.debug(f"[GPT-Multi] raw={raw_text}, reason={choice.finish_reason}")

        cleaned = raw_text.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("[GPT-Multi] JSON parse error => fallback => []")
            return {"decisions": []}

        if not isinstance(parsed, dict):
            logger.warning("[GPT-Multi] JSON not a dict => fallback => []")
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
