"""
gpt_manager.py

A production-ready module to handle GPT-based inference for trading decisions,
now allowing the user to control how much trade history is included (via a numeric parameter).

Key Features:
    1) Centralizes GPT logic in one class, GPTManager.
    2) Provides flexible configuration (model, temperature, etc.).
    3) Parses JSON from GPT responses, with fallback if JSON parse fails.
    4) Strips triple backticks/code blocks to avoid malformed JSON.
    5) Accepts a trade_history list and a max_trades integer to limit how many trades we supply to GPT.

Usage Example:
    from gpt_manager import GPTManager

    # Suppose you store the last N trades in a list of strings:
    # e.g. trade_history = [
    #   "2025-01-20 10:15 BUY 0.001@25000",
    #   "2025-01-21 11:20 SELL 0.001@25500",
    #   ...
    # ]

    gpt = GPTManager(
        api_key="sk-...",       # or rely on OPENAI_API_KEY env var
        model="gpt-4o-mini",    # your chosen model
        temperature=0.8,
        max_tokens=500
    )

    result_dict = gpt.generate_trade_decision(
        conversation_context="Summarized conversation so far.",
        aggregator_text="rsi=44.2, boll_upper=1910.5, price=1870",
        trade_history=[
            "2025-01-20 10:15 BUY 0.001@25000",
            "2025-01-21 11:20 SELL 0.001@25500",
            "2025-01-22 09:00 BUY 0.0005@24500",
            "2025-01-23 12:40 BUY 0.0007@24800"
        ],
        max_trades=3,  # supply only the last 3 trades
        risk_controls={
            "initial_spending_account": 40.0,
            "purchase_upper_limit_percent": 50.0,
            "minimum_buy_amount": 10.0,
            ...
        }
    )
    # result_dict => {"action": "BUY", "size": 0.001}, etc.

Testing:
    - Mock or patch gpt.client.chat.completions.create(...) to simulate GPT responses.
    - Verify the truncated trade history is used in the prompt, and JSON parse fallback logic works.
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
    GPTManager orchestrates:
      - Building system/user prompts for trade decisions.
      - Calling the OpenAI chat completion API.
      - Parsing the JSON outcome {"action":..., "size":...}, with a fallback to HOLD if parse fails.
      - Letting the caller control how many lines of trade history to supply (max_trades).
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
        Constructor for GPTManager.

        :param api_key:        Override for OPENAI_API_KEY env var if desired.
        :param model:          Which GPT model to use, e.g. "gpt-4o-mini".
        :param temperature:    The GPT temperature.
        :param max_tokens:     Max tokens in GPT response.
        :param client_options: Additional kwargs for the openai.OpenAI constructor
                               (e.g., timeout, max_retries, proxies, etc.).
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            logger.warning(
                "No OPENAI_API_KEY found. GPT calls may fail unless an environment "
                "variable or explicit api_key is set."
            )

        # Instantiate the OpenAI client
        self.client = OpenAI(api_key=self.api_key, **client_options)

        # Config
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
    ) -> Dict[str, Any]:
        """
        Generates a trading decision using GPT chat completion.

        :param conversation_context: A string representing prior GPT conversation or summary.
        :param aggregator_text:      A short string summarizing market indicators/aggregators.
        :param trade_history:        A list of trade history strings (e.g. ["2025-01-20 10:15 BUY 0.001@25000", ...]).
        :param max_trades:          How many of the most recent trades to include in the prompt.
        :param risk_controls:        A dict for relevant risk control parameters.

        :return: {"action": <BUY|SELL|HOLD>, "size": <float>}
        """

        # Build a short summary of the last N trades
        if trade_history:
            recent_trades = trade_history[-max_trades:]
            trade_summary = "\n".join(recent_trades)
        else:
            trade_summary = "No trades found."

        # 1) System message => strict JSON
        system_msg = {
            "role": "assistant",
            "content": (
                "You are an advanced crypto trading assistant. "
                "Return ONLY valid JSON in the form: "
                "{\"action\":\"BUY|SELL|HOLD\",\"size\":0.0005}. "
                "No code blocks or markdown."
            )
        }

        # 2) User prompt
        user_prompt = (
            f"Conversation so far:\n{conversation_context}\n\n"
            f"Aggregators => {aggregator_text}\n"
            f"Recent trades =>\n{trade_summary}\n"
            f"Risk controls => {risk_controls}\n"
            "Output strictly JSON => {\"action\":\"BUY|SELL|HOLD\",\"size\":float}\n"
        )
        user_msg = {"role": "user", "content": user_prompt}

        # 3) GPT call
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[system_msg, user_msg],
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens
            )
        except Exception as e:
            logger.exception("Error contacting GPT => fallback => HOLD")
            return {"action": "HOLD", "size": 0.0}

        if not response.choices:
            logger.warning("No GPT choices => fallback => HOLD")
            return {"action": "HOLD", "size": 0.0}

        choice = response.choices[0]
        raw_text = choice.message.content or ""
        finish_reason = choice.finish_reason
        logger.debug(f"[GPT] raw={raw_text}, finish_reason={finish_reason}")

        # 4) Cleanup any code fences
        cleaned = raw_text.replace("```json", "").replace("```", "").strip()

        # 5) JSON parse
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Could not parse GPT response => fallback => HOLD")
            return {"action": "HOLD", "size": 0.0}

        if not isinstance(parsed, dict):
            logger.warning("Parsed GPT response is not a dict => fallback => HOLD")
            return {"action": "HOLD", "size": 0.0}

        # Provide defaults
        action = str(parsed.get("action", "HOLD")).upper()
        size = float(parsed.get("size", 0.0))

        return {"action": action, "size": size}
