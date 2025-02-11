#!/usr/bin/env python3
# =============================================================================
# FILE: gpt_manager.py
# =============================================================================
"""
gpt_manager.py

An updated GPTManager class that:
1) Automatically logs GPT requests & responses (with usage stats) as JSON files in
   `logs/{timestamp}/` directories:
   - request_body.json: The raw request dictionary passed to OpenAI
   - response_body.json: The raw response dictionary from OpenAI
   - usage_stats.json: The token usage block extracted from the response
   - prompt.json: The final, user-facing prompt that GPT sees (combined system+user messages)
2) Includes extended error handling for network issues vs. JSON parse errors.
3) Skips any function-calling logic.
4) The GPT model is parameterized by your `config.yaml`.
5) The aggregator prompt is expanded to incorporate additional LunarCrush data fields
   (e.g. `price_btc`, `alt_rank`, `galaxy_score`, etc.) so GPT has a sense of each data field's meaning.

Dependencies:
    - openai (≥ v1) python package
    - pyyaml for config reading (optional if you already use it)
    - Python 3.8+
"""

import os
import json
import time
import yaml
import logging
import datetime
from typing import IO, Any, Dict, List, Optional

import openai
from openai import OpenAI
from openai import APIConnectionError, APIStatusError, RateLimitError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def build_aggregator_prompt(
    summary: Optional[str] = None,
    probability: Optional[float] = None,
    embedding: Optional[List[float]] = None,
    additional_fields: Optional[Dict[str, Any]] = None
) -> str:
    """
    A helper function to unify aggregator data into a more detailed snippet,
    referencing new LunarCrush fields introduced by fetch_lunarcrush.py.

    This text is then appended to the final GPT prompt so the model
    understands each data field meaning (e.g. price_btc, alt_rank, galaxy_score, etc.).

    :param summary: A short textual summary (e.g. "price_bucket=high, sentiment_label=positive").
    :type summary: str, optional

    :param probability: The local classifier probability of "UP" or "positive outcome" (float).
    :type probability: float, optional

    :param embedding: A small list of floats from PCA or embeddings.
    :type embedding: list of float, optional

    :param additional_fields: If you want to pass custom data fields (like
        price_btc, alt_rank, etc.). We'll add them to the text snippet.
    :type additional_fields: dict, optional

    :return: A single string that combines aggregator data for GPT usage.
    :rtype: str
    """
    # Prepend a short explanation of each field’s meaning:
    field_meaning_block = (
        "Additional LunarCrush data: \n"
        " - price_btc: The coin's price in BTC units.\n"
        " - alt_rank: A lower rank is better, measures altcoin performance.\n"
        " - galaxy_score: 0..100 measure combining performance + social metrics.\n"
        " - market_dominance: The share of total crypto market cap.\n"
        " - volatility: Standard deviation or a measure of price variance.\n"
        " - sentiment: Sentiment index from 0..100 or -1..+1.\n"
        " - interactions_24h: The count of social interactions in 24h.\n"
        " - social_volume_24h: The count of social mentions in 24h.\n"
        "These fields help indicate short-term interest and potential price movement.\n\n"
    )

    # Start building lines
    parts = [field_meaning_block]

    if summary:
        parts.append(f"(summary) {summary}")

    if probability is not None:
        parts.append(f"(local_classifier_probability) {round(probability, 4)}")

    if embedding:
        emb_str = "[" + ",".join(str(round(v, 3)) for v in embedding) + "]"
        parts.append(f"(embedding_vector) {emb_str}")

    if additional_fields:
        for k, v in additional_fields.items():
            parts.append(f"({k}) {v}")

    aggregator_text = "\n".join(parts)
    return aggregator_text


class GPTManager:
    """
    GPTManager:

    - Orchestrates GPT-based logic for single-coin or multi-coin trading decisions.
    - Logs request/response JSON in logs/{timestamp}/, including usage stats and the raw prompt.
    - Has extended error handling for better resilience.
    - Skips function-calling usage to keep the code simpler.

    Usage Example:
        from gpt_manager import GPTManager

        manager = GPTManager()
        single_result = manager.generate_trade_decision(
            conversation_context="Prior context",
            aggregator_text="some aggregator snippet",
            trade_history=["2025-01-21 SELL BTC/USD 0.001@21000", ...],
            max_trades=5,
            risk_controls={"minimum_buy_amount": 10.0},
            open_positions=["BTC/USD LONG 0.002, entry=20000.0"]
        )
    """

    def __init__(
        self,
        config_file: str = "config.yaml",
        temperature: float = 0.7,
        max_tokens: int = 500,
        log_gpt_calls: bool = True
    ):
        """
        Initialize the GPTManager, reading 'model' from config.yaml unless overridden.

        :param config_file: Path to your config YAML (defaults to 'config.yaml').
            The config file can contain the key 'openai_model' which overrides the default model.
        :type config_file: str

        :param temperature: GPT sampling temperature. Higher values produce more variety in outputs.
        :type temperature: float

        :param max_tokens: The maximum number of tokens for GPT's generated response.
        :type max_tokens: int

        :param log_gpt_calls: If True, logs all GPT requests and responses into logs/{timestamp}/.
        :type log_gpt_calls: bool
        """
        self.log_gpt_calls = log_gpt_calls
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Load config => read 'openai_model' from config_file if present
        self.model = "gpt-4"
        if os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            self.model = cfg.get("openai_model", self.model)

        # Initialize the openai client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

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
        Generate a single-coin trading decision from GPT with an emphasis on
        short-term (within ~24 hours) profit in USD.

        The final output is expected to be JSON with keys:
            {"action":"BUY|SELL|HOLD","size": <float>}

        Steps:
          1) Build user/system prompt to incorporate aggregator_text & recent trades.
          2) Clarify that we measure success in USD, aiming to close positions by
             tomorrow if profitable.
          3) Call GPT => handle exceptions => parse response or fallback => hold.
          4) Return {"action":..., "size":...} with fallback if parsing fails.

        :param conversation_context: Summarized or partial conversation memory from prior usage.
        :type conversation_context: str

        :param aggregator_text: Aggregated data snippet from build_aggregator_prompt().
        :type aggregator_text: str

        :param trade_history: Past trades to show GPT. We'll only pass the last 'max_trades'.
        :type trade_history: list of str

        :param max_trades: How many lines from trade_history to include in the prompt.
        :type max_trades: int

        :param risk_controls: Additional constraints, e.g. {"minimum_buy_amount":10.0}.
        :type risk_controls: dict

        :param open_positions: Lines describing any open sub-positions, if any.
        :type open_positions: list of str, optional

        :return: A dictionary with "action" and "size", or fallback => {"action":"HOLD","size":0.0}.
        :rtype: dict
        """
        truncated_history = trade_history[-max_trades:] if trade_history else []
        trade_summary = "\n".join(truncated_history) if truncated_history else "No trades found."
        open_pos_summary = "\n".join(open_positions) if open_positions else "No open positions."

        # System instructions for a single-coin scenario
        system_instructions = (
            "You are a specialized single-coin trading assistant. We trade with US dollars, "
            "and we aim to realize profit within the next day (short-term). Our performance "
            "is measured in net USD by the end of ~24 hours. If the coin is likely to be profitable, "
            "we prefer to exit positions in USD quickly. You MUST return a single JSON object: "
            "{\"action\":\"BUY|SELL|HOLD\",\"size\":float}, with no additional text. "
            "Action must be one of BUY, SELL, or HOLD, and size is a float representing how many coins to trade."
        )
        logger.info(f"[GPT-Single] System Instructions: \n{aggregator_text}")

        user_prompt = (
            f"SYSTEM INSTRUCTIONS:\n"
            f"{system_instructions}\n\n"
            f"AGGREGATOR DATA:\n{aggregator_text}\n\n"
            f"OPEN POSITIONS:\n{open_pos_summary}\n\n"
            f"RECENT TRADES:\n{trade_summary}\n\n"
            f"RISK CONTROLS:\n{risk_controls}\n\n"
            f"PRIOR GPT CONTEXT:\n{conversation_context}\n\n"
            "OUTPUT FORMAT:\n"
            "Return strictly JSON => {\"action\":\"BUY|SELL|HOLD\",\"size\":float}\n"
        )
        logger.info(f"[GPT-Single] User Prompt: \n{user_prompt}")

        messages = [
            {"role": "assistant", "content": system_instructions},
            {"role": "user", "content": user_prompt},
        ]

        request_dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
        }

        if self.log_gpt_calls:
            self._save_prompt_files(user_prompt, request_dict)

        try:
            response = self.client.chat.completions.create(**request_dict)
            if self.log_gpt_calls:
                self._save_response_files(response)
            raw_text = response.choices[0].message.content if response.choices else ""
            parsed = self._parse_single_action_json(raw_text)
            return parsed

        except (APIConnectionError, RateLimitError) as net_exc:
            logger.exception(f"[GPT-Single] Connection or RateLimit error => {net_exc}")
            return {"action": "HOLD", "size": 0.0}
        except APIStatusError as e:
            logger.warning(f"[GPT-Single] API status code => {e.status_code}, response={e.response}")
            return {"action": "HOLD", "size": 0.0}
        except Exception as e:
            logger.exception(f"[GPT-Single] Unknown error => {e}")
            return {"action": "HOLD", "size": 0.0}

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
        Generate multi-coin trading decisions in a single GPT call, emphasizing
        short-term (~24h) profit in USD. We measure success in net USD by the end
        of the day, aiming to close positions quickly if profitable.

        aggregator_list => each item => {
            "pair": "ETH/USD",
            "price": 1800.0,
            "aggregator_data": "some aggregator snippet referencing new fields"
        }

        We expect GPT to return JSON:
        {
          "decisions": [
             {"pair":"ETH/USD","action":"BUY","size":0.001},
             ...
          ]
        }

        :param conversation_context: Prior GPT conversation or summary context.
        :type conversation_context: str

        :param aggregator_list: Each item has "pair","price","aggregator_data".
        :type aggregator_list: list of dict

        :param open_positions: Lines describing open sub-positions.
        :type open_positions: list of str

        :param trade_history: Lines describing recent trades to show GPT.
        :type trade_history: list of str

        :param max_trades: How many lines of trade_history to show.
        :type max_trades: int

        :param risk_controls: Additional constraints (like min buy amount).
        :type risk_controls: dict

        :return: For example {"decisions":[{"pair":"ETH/USD","action":"BUY","size":0.001},...]}
                 or fallback => {"decisions":[]}.
        :rtype: dict
        """
        truncated_history = trade_history[-max_trades:] if trade_history else []
        trade_summary = "\n".join(truncated_history) if truncated_history else "No trades."
        open_pos_summary = "\n".join(open_positions) if open_positions else "No open positions."

        # Build aggregator section
        aggregator_block = []
        for i, item in enumerate(aggregator_list, start=1):
            aggregator_data_str = item.get('aggregator_data', '')
            aggregator_block.append(f"{i}) {item['pair']} => {aggregator_data_str}")
        aggregator_section = "\n".join(aggregator_block) if aggregator_block else "No aggregator data."

        # System instructions for multi-coin approach
        system_instructions = (
            "You are a specialized multi-coin crypto trading assistant. We trade in US dollars, "
            "and we aim to realize profit within ~24 hours if possible. Our performance is measured "
            "in net USD at the end of the day. We prefer to close profitable positions quickly rather "
            "than holding them beyond one day, unless there's a strong reason to hold. You MUST return "
            "a single JSON object:\n"
            "{\"decisions\":[{\"pair\":\"COIN_PAIR\",\"action\":\"BUY|SELL|HOLD\",\"size\":float},...]}\n"
            "No extra text or code blocks."
        )

        # Structured multi-section prompt
        user_prompt = (
            f"SYSTEM INSTRUCTIONS:\n"
            f"{system_instructions}\n\n"
            f"AGGREGATOR DATA:\n{aggregator_section}\n\n"
            f"OPEN POSITIONS:\n{open_pos_summary}\n\n"
            f"RECENT TRADES:\n{trade_summary}\n\n"
            f"RISK CONTROLS:\n{risk_controls}\n\n"
            f"PRIOR GPT CONTEXT:\n{conversation_context}\n\n"
            "OUTPUT FORMAT:\n"
            "Return strictly JSON => {\"decisions\":[{\"pair\":\"COIN_PAIR\",\"action\":\"BUY|SELL|HOLD\",\"size\":0.001},...]}\n"
        )

        messages = [
            {"role": "assistant", "content": system_instructions},
            {"role": "user", "content": user_prompt},
        ]

        request_dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
        }

        if self.log_gpt_calls:
            self._save_prompt_files(user_prompt, request_dict)

        try:
            response = self.client.chat.completions.create(**request_dict)
            if self.log_gpt_calls:
                self._save_response_files(response)
            raw_text = response.choices[0].message.content if response.choices else ""
            parsed = self._parse_multi_decisions_json(raw_text)
            return parsed

        except (APIConnectionError, RateLimitError) as net_exc:
            logger.exception(f"[GPT-Multi] Connection or RateLimit error => {net_exc}")
            return {"decisions": []}
        except APIStatusError as e:
            logger.warning(f"[GPT-Multi] API status code => {e.status_code}, response={e.response}")
            return {"decisions": []}
        except Exception as e:
            logger.exception(f"[GPT-Multi] Unknown error => {e}")
            return {"decisions": []}

    # --------------------------------------------------------------------------
    # JSON Parsing Helpers
    # --------------------------------------------------------------------------
    def _parse_single_action_json(self, raw_text: str) -> Dict[str, Any]:
        """
        Attempt to parse raw_text => {"action":"BUY|SELL|HOLD","size":float}.
        If parse fails, fallback => hold.

        :param raw_text: The GPT output as a string.
        :type raw_text: str

        :return: e.g. {"action":"BUY","size":0.001} or fallback => {"action":"HOLD","size":0.0}.
        :rtype: dict
        """
        if not raw_text.strip():
            return {"action": "HOLD", "size": 0.0}

        # strip code fences
        cleaned = raw_text.replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                return {"action": "HOLD", "size": 0.0}
            action = parsed.get("action", "HOLD").upper()
            size = float(parsed.get("size", 0.0))
            if action not in ["BUY", "SELL", "HOLD"]:
                action = "HOLD"
            if size < 0:
                size = 0.0
            return {"action": action, "size": size}
        except json.JSONDecodeError:
            logger.warning("[GPT-Single] JSON decode error => fallback hold")
            return {"action": "HOLD", "size": 0.0}

    def _parse_multi_decisions_json(self, raw_text: str) -> Dict[str, Any]:
        """
        Attempt to parse raw_text => { "decisions":[ {"pair":"X","action":"BUY|SELL|HOLD","size":n}, ... ] }
        If parse fails => fallback => empty decisions.

        :param raw_text: The GPT output as a string.
        :type raw_text: str

        :return: e.g. {"decisions":[{"pair":"XBT/USD","action":"BUY","size":0.001}]} or fallback => {"decisions":[]}.
        :rtype: dict
        """
        if not raw_text.strip():
            return {"decisions": []}

        cleaned = raw_text.replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                logger.warning("[GPT-Multi] parse => not dict => fallback []")
                return {"decisions": []}
            decisions_list = parsed.get("decisions", [])
            if not isinstance(decisions_list, list):
                logger.warning("[GPT-Multi] 'decisions' not a list => fallback []")
                return {"decisions": []}

            final_decisions = []
            for d in decisions_list:
                if not isinstance(d, dict):
                    continue
                pair = d.get("pair", "UNK")
                action = str(d.get("action", "HOLD")).upper()
                size = float(d.get("size", 0.0))
                if action not in ("BUY", "SELL", "HOLD"):
                    action = "HOLD"
                if size < 0:
                    size = 0.0
                final_decisions.append({"pair": pair, "action": action, "size": size})

            return {"decisions": final_decisions}

        except json.JSONDecodeError:
            logger.warning("[GPT-Multi] JSON decode error => fallback => []")
            return {"decisions": []}

    # --------------------------------------------------------------------------
    # Logging Helpers
    # --------------------------------------------------------------------------
    def _save_prompt_files(self, user_prompt: str, request_dict: Dict[str, Any]) -> None:
        """
        Save the final prompt and the request dict to logs/{timestamp}/prompt.json and request_body.json.

        :param user_prompt: The final user-facing prompt text that GPT sees.
        :type user_prompt: str

        :param request_dict: The entire dictionary to be passed to the openai client,
            including 'model', 'messages', 'temperature', etc.
        :type request_dict: dict
        """
        timestamp_dir = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        log_dir = os.path.join("logs", timestamp_dir)
        os.makedirs(log_dir, exist_ok=True)

        # Save the user prompt as plain text
        prompt_path = os.path.join(log_dir, "prompt.txt")
        with open(prompt_path, "w", encoding="utf-8") as f_prompt:
            f_prompt.write(user_prompt)

        # Save the entire request_dict as raw text (via str())
        req_path = os.path.join(log_dir, "request_body.txt")
        with open(req_path, "w", encoding="utf-8") as f_req:
            # Convert the dictionary to a string (instead of JSON)
            f_req.write(str(request_dict))

        # prompt_path = os.path.join(log_dir, "prompt.json")
        # with open(prompt_path, "w", encoding="utf-8") as f_prompt:  # type: IO[str]
        #     json.dump({"prompt": user_prompt}, f_prompt, indent=4)
        #
        # req_path = os.path.join(log_dir, "request_body.json")
        # with open(req_path, "w", encoding="utf-8") as f_req:  # type: IO[str]
        #     json.dump(request_dict, f_req, indent=4)

    def _save_response_files(self, response_obj: Any) -> None:
        """
        Save the raw response dict and usage stats to logs/{timestamp}/response_body.json and usage_stats.json.

        :param response_obj: The raw object returned by the openai client,
            typically an openai.types.ChatCompletion or dict-like structure.
        :type response_obj: Any
        """
        timestamp_dir = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        log_dir = os.path.join("logs", timestamp_dir)
        os.makedirs(log_dir, exist_ok=True)

        # If you want to store the entire response as raw text:
        resp_path = os.path.join(log_dir, "response_body.txt")
        with open(resp_path, "w", encoding="utf-8") as f_resp:
            # Just write the raw string representation
            f_resp.write(str(response_obj))

        # usage stats, if you still want them in a small JSON or text file
        usage_path = os.path.join(log_dir, "usage_stats.txt")
        usage_data = {}

        # Some responses have usage field
        try:
            if hasattr(response_obj, "usage"):
                usage_data = dict(response_obj.usage)
            else:
                # maybe it's already a dict
                response_dict = response_obj
                if isinstance(response_dict, dict) and "usage" in response_dict:
                    usage_data = response_dict["usage"]
        except Exception:
            logger.warning("[GPTManager] No usage field found in GPT response")

        # write usage_data as text
        with open(usage_path, "w", encoding="utf-8") as f_usage:
            f_usage.write(str(usage_data))
