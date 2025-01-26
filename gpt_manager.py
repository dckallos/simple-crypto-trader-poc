"""
gpt_manager.py

A production-ready module to handle GPT-based inference for **multi-coin** trading decisions
in a single GPT call. This version offers an advanced approach:
  - Instead of calling GPT for each coin individually, we pass a single combined prompt
    containing aggregator data for all tracked coins at once.
  - GPT can then compare them holistically, picking the best actions from the sum of all choices.
  - This includes references to open positions across multiple coins, letting GPT
    decide whether to hold, buy, sell, or swap positions in multiple coins.

Key Features:
    1) `generate_multi_trade_decision()` method that:
       - Accepts a list of aggregator data dicts for each coin.
       - Accepts a consolidated 'open_positions' referencing positions in multiple coins.
       - Returns a single JSON with decisions for each coin, e.g.:
         {
           "decisions": [
             {"pair":"XBT/USD","action":"BUY","size":0.001},
             {"pair":"ETH/USD","action":"SELL","size":0.002}
           ]
         }
    2) Single GPT call => GPT sees the entire market at once, so it can produce
       a *global* best set of trades from the sum of all choices.
    3) Strict JSON response with an array of decisions. If GPT fails to parse
       or returns invalid JSON, we fallback to an empty or hold-based approach.
    4) The rest of the file remains a "production-ready" GPT manager, with advanced prompt
       engineering, fallback logic, code-fence stripping, etc.

Usage Example:
    from gpt_manager import GPTManager

    gpt = GPTManager(
        api_key="sk-...",
        model="gpt-4o",
        temperature=0.8,
        max_tokens=1000
    )

    # aggregator_list is e.g. a list of:
    # [
    #   {
    #     "pair":"ETH/USD",
    #     "aggregator_data":"rsi=44, price=1850, galaxy_score=60, alt_rank=200, ...
    #   },
    #   {
    #     "pair":"XBT/USD",
    #     "aggregator_data":"rsi=50, price=28000, galaxy_score=65, alt_rank=150, ...
    #   }
    # ]

    # open_positions is a list of strings describing all active positions across coins
    # e.g. ["ETH/USD LONG 0.002, entry=1860", "XBT/USD SHORT 0.001, entry=29500"]

    result_dict = gpt.generate_multi_trade_decision(
        conversation_context="Summarized conversation so far.",
        aggregator_list=aggregator_list,
        open_positions=open_positions,
        trade_history=["2025-01-20 10:15 BUY ETH 0.001@25000", "...", "..."],
        max_trades=5,
        risk_controls={
            "initial_spending_account": 100.0,
            "purchase_upper_limit_percent": 10.0,
            "minimum_buy_amount": 10.0,
            ...
        }
    )
    # example output => {
    #   "decisions": [
    #     {"pair":"ETH/USD","action":"BUY","size":0.001},
    #     {"pair":"XBT/USD","action":"HOLD","size":0.0}
    #   ]
    # }

Testing:
    - In your tests, you can patch `self.client.chat.completions.create(...)` to return
      a mock response with `choices[0].message.content` set to a structured JSON that has
      "decisions" as an array. Then confirm your code decodes it properly.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

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
    GPTManager orchestrates GPT logic for advanced multi-coin trading decisions.
    By calling `generate_multi_trade_decision(...)`, you can pass aggregator data
    for multiple coins in a single GPT call. GPT returns a JSON array of
    {"pair":"...","action":"BUY|SELL|HOLD","size":float} decisions.
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
        :param api_key:        Optionally override the OPENAI_API_KEY env var.
        :param model:          GPT model name, e.g. "gpt-4o-mini".
        :param temperature:    GPT temperature.
        :param max_tokens:     Maximum tokens in GPT response.
        :param client_options: Additional arguments for the openai.OpenAI constructor
                               (e.g. timeout, max_retries, proxies).
        """
        load_dotenv()
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
                {"pair":"ETH/USD", "action":"BUY|SELL|HOLD", "size":0.001},
                {"pair":"XBT/USD", "action":"BUY|SELL|HOLD", "size":0.0005}
                ...
              ]
            }

        :param conversation_context: str => prior GPT conversation or summary.
        :param aggregator_list: A list, each item is e.g.
               {"pair":"ETH/USD","aggregator_data":"rsi=44,price=1850,sentiment=0.4,galaxy=65,..."}
        :param open_positions: list of strings describing all open trades across coins
               e.g. ["ETH/USD LONG 0.002, entry=1860", "XBT/USD SHORT 0.001, entry=29500"]
        :param trade_history: A list of past trade strings; we'll show the last N
        :param max_trades: how many of the last trades from trade_history to pass
        :param risk_controls: dict with relevant constraints

        :return => a dictionary with "decisions" => a list of decision objects,
                   or fallback => {"decisions": []} on parse error
        """
        # Summarize aggregator_data
        # aggregator_list might have e.g. N coins => build a text block for each coin
        aggregator_text_block = []
        for idx, item in enumerate(aggregator_list, start=1):
            pair_name = item.get("pair","UNK")
            data_str = item.get("aggregator_data","")
            aggregator_text_block.append(f"{idx}) {pair_name} => {data_str}")

        aggregator_text_joined = "\n".join(aggregator_text_block)

        # Summarize open_positions
        if open_positions:
            open_positions_str = "\n".join(open_positions)
        else:
            open_positions_str = "No open positions."

        # Summarize trade history (the last max_trades lines)
        if trade_history:
            truncated = trade_history[-max_trades:]
            trade_summary = "\n".join(truncated)
        else:
            trade_summary = "No past trades found."

        # 1) System message => we want an array of decisions in JSON
        system_msg = {
            "role": "assistant",
            "content": (
                "You are a specialized multi-coin crypto trading assistant. "
                "You see aggregator data for multiple coins at once, plus open positions. "
                "You MUST return a single JSON object of the form:\n"
                "{\n"
                "  \"decisions\": [\n"
                "    {\"pair\":\"COIN_PAIR\",\"action\":\"BUY|SELL|HOLD\",\"size\":0.001},\n"
                "    ...\n"
                "  ]\n"
                "}\n\n"
                "No extra text or code blocks. If you want to close or swap an existing position, "
                "use 'SELL' with an appropriate 'size'."
            )
        }

        # 2) Build user prompt
        user_prompt = (
            f"Conversation so far:\n{conversation_context}\n\n"
            f"Aggregators for all coins =>\n{aggregator_text_joined}\n\n"
            f"Open positions =>\n{open_positions_str}\n\n"
            f"Recent trades =>\n{trade_summary}\n\n"
            f"Risk controls => {risk_controls}\n\n"
            "IMPORTANT: Return strictly JSON => {\"decisions\":[...]}\n"
            "Decisions array might have multiple objects, each with pair/action/size."
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
            logger.exception("[GPT-Multi] Error contacting GPT => fallback => empty decisions")
            return {"decisions": []}

        if not response.choices:
            logger.warning("[GPT-Multi] No GPT choices => fallback => empty decisions")
            return {"decisions": []}

        choice = response.choices[0]
        raw_text = choice.message.content or ""
        finish_reason = choice.finish_reason
        logger.debug(f"[GPT-Multi] raw={raw_text}, finish_reason={finish_reason}")

        # 4) strip code fences
        cleaned = raw_text.replace("```json", "").replace("```", "").strip()

        # 5) parse JSON
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("[GPT-Multi] Could not parse GPT response => fallback => empty decisions")
            return {"decisions": []}

        if not isinstance(parsed, dict):
            logger.warning("[GPT-Multi] JSON was not a dict => fallback => empty decisions")
            return {"decisions": []}

        # ensure we have 'decisions' => a list
        decisions_list = parsed.get("decisions", [])
        if not isinstance(decisions_list, list):
            logger.warning("[GPT-Multi] 'decisions' is not a list => fallback => empty decisions")
            return {"decisions": []}

        # We can do some basic validation => each item => must have 'pair','action','size'
        # We'll do a naive approach:
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
