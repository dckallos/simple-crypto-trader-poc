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
3) Allows skipping function-calling logic (we do not parse 'function_call').
4) The GPT model is parameterized by your `config.yaml`.
5) The aggregator prompt is expanded to incorporate additional LunarCrush data fields (e.g. `price_btc`,
   `alt_rank`, `galaxy_score`, etc.) so GPT has a sense of each data's meaning.

Dependencies:
    - openai (≥ v1) python package
    - pyyaml for config reading (optional if you already use it)
    - Python 3.8+

Classes:
    GPTManager: The main class for single-coin or multi-coin GPT trade decision logic.

Public Methods:
    - build_aggregator_prompt(...)
    - GPTManager.generate_trade_decision(...)
    - GPTManager.generate_multi_trade_decision(...)
"""

import os
import json
import time
import yaml
import logging
import datetime
from typing import Any, Dict, List, Optional

import openai
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError

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

    Args:
        summary (str, optional): A short textual summary (e.g. "price_bucket=high, sentiment_label=positive").
        probability (float, optional): The local classifier probability of "UP" or "positive outcome" (float).
        embedding (List[float], optional): A small list of floats from PCA or embeddings.
        additional_fields (Dict[str, Any], optional): If you want to pass custom data fields (like
            price_btc, alt_rank, etc.). We'll add them to the text snippet.

    Returns:
        str: A single string that combines aggregator data for GPT usage.
    """
    # Prepend a short explanation of each field's meaning:
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
        # For each key => value
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

    Usage Example:
        from gpt_manager import GPTManager

        manager = GPTManager()
        single_result = manager.generate_trade_decision(
            conversation_context="Prior context or conversation",
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

        Args:
            config_file (str): Path to your config YAML (defaults to 'config.yaml').
            temperature (float): GPT sampling temperature.
            max_tokens (int): The max tokens in GPT responses.
            log_gpt_calls (bool): If True, log request/response to logs/{timestamp}/.
        """
        self.log_gpt_calls = log_gpt_calls
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Load config => read 'openai_model' or fallback
        self.model = "gpt-4o-mini"  # default
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                cfg = yaml.safe_load(f)
            # You can store your model name in config with a key like 'openai_model'
            self.model = cfg.get("openai_model", self.model)

        # Initialize the openai client
        # We rely on OPENAI_API_KEY as env var or you can pass it in somehow
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
        Generate a single-coin trading decision from GPT.

        The final output is expected to be JSON with keys:
            {"action":"BUY|SELL|HOLD","size": <float>}

        Steps:
          1) Build user/system prompt to incorporate aggregator_text & recent trades.
          2) Call GPT => handle exceptions => parse response or fallback => hold
          3) Return {"action":..., "size":...} with fallback if parsing fails.

        Args:
            conversation_context (str): Summarized or partial conversation memory.
            aggregator_text (str): Aggregated data snippet from build_aggregator_prompt().
            trade_history (List[str]): Past trades to show GPT. We'll only pass the last 'max_trades'.
            max_trades (int): How many lines from trade_history to show GPT.
            risk_controls (dict): Additional constraints like min buy amount, etc.
            open_positions (Optional[List[str]]): Lines describing any open sub-positions.

        Returns:
            dict: e.g. {"action":"BUY","size":0.043} or fallback => {"action":"HOLD","size":0.0}
        """
        # (A) Build the prompt
        truncated_history = trade_history[-max_trades:] if trade_history else []
        trade_summary = "\n".join(truncated_history) if truncated_history else "No trades found."
        open_pos_summary = "\n".join(open_positions) if open_positions else "No open positions."

        system_instructions = (
            "You are a specialized single-coin trading assistant. "
            "You must respond with a single JSON object: "
            '{"action":"BUY|SELL|HOLD","size":float}. '
            "No extra text or code blocks."
        )
        user_prompt = (
            f"Conversation context:\n{conversation_context}\n\n"
            f"Aggregator data:\n{aggregator_text}\n\n"
            f"Recent trades:\n{trade_summary}\n\n"
            f"Open positions:\n{open_pos_summary}\n\n"
            f"Risk controls:\n{risk_controls}\n\n"
            "Return strictly JSON => {\"action\":\"BUY|SELL|HOLD\",\"size\":float}\n"
        )

        messages = [
            {"role": "assistant", "content": system_instructions},
            {"role": "user", "content": user_prompt},
        ]

        # (B) Prepare request
        request_dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
        }

        # (C) Save request if enabled
        if self.log_gpt_calls:
            self._save_prompt_files(user_prompt, request_dict)

        # (D) Call GPT
        try:
            response = self.client.chat.completions.create(**request_dict)

            # (E) Save response
            if self.log_gpt_calls:
                self._save_response_files(response)

            raw_text = response.choices[0].message.content if response.choices else ""
            parsed = self._parse_single_action_json(raw_text)
            return parsed

        except (APIConnectionError, RateLimitError) as net_exc:
            # network / rate-limiting => fallback => hold
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
        Generate multi-coin trading decisions in a single GPT call.

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

        Args:
            conversation_context (str): Prior GPT conversation or summary context.
            aggregator_list (List[dict]): Each item has "pair","price","aggregator_data".
            open_positions (List[str]): Lines describing open sub-positions.
            trade_history (List[str]): Lines describing recent trades to show GPT.
            max_trades (int): # lines of trade history to show.
            risk_controls (Dict[str, Any]): Additional constraints.

        Returns:
            dict: e.g. {"decisions":[{"pair":"ETH/USD","action":"HOLD","size":0.0},...]}
        """
        # (A) Build prompt
        truncated_history = trade_history[-max_trades:] if trade_history else []
        trade_summary = "\n".join(truncated_history) if truncated_history else "No trades."
        open_pos_summary = "\n".join(open_positions) if open_positions else "No open positions."

        aggregator_text_block = []
        for idx, item in enumerate(aggregator_list, start=1):
            aggregator_text_block.append(
                f"{idx}) {item.get('pair','?')} => {item.get('aggregator_data','')}"
            )
        aggregator_text = "\n".join(aggregator_text_block)

        system_instructions = (
            "You are a specialized multi-coin crypto trading assistant. "
            "You see aggregator data for multiple coins, plus open positions. "
            "You MUST return a single JSON object of the form:\n"
            '{"decisions":[{"pair":"COIN_PAIR","action":"BUY|SELL|HOLD","size":0.001},...]}'
        )
        user_prompt = (
            f"Conversation context:\n{conversation_context}\n\n"
            f"Aggregators:\n{aggregator_text}\n\n"
            f"Open positions:\n{open_pos_summary}\n\n"
            f"Recent trades:\n{trade_summary}\n\n"
            f"Risk controls:\n{risk_controls}\n\n"
            "Return strictly JSON => {\"decisions\":[...]}\n"
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

        # (B) Possibly save request
        if self.log_gpt_calls:
            self._save_prompt_files(user_prompt, request_dict)

        # (C) Call GPT
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

        Args:
            raw_text (str): The GPT output.

        Returns:
            dict: e.g. {"action":"BUY","size":0.001} or fallback.
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

        Args:
            raw_text (str): GPT output.

        Returns:
            dict: e.g. {"decisions":[{"pair":"XBT/USD","action":"BUY","size":0.001}]} or fallback => {"decisions":[]}.
        """
        if not raw_text.strip():
            return {"decisions": []}

        # remove fences
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
    def _save_prompt_files(self, user_prompt: str, request_dict: Dict[str, Any]):
        """
        Save the final prompt and the request dict to logs/{timestamp}/prompt.json and request_body.json.

        Args:
            user_prompt (str): The final user-facing prompt text
            request_dict (dict): The entire dictionary to be passed to openai client
        """
        timestamp_dir = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        log_dir = os.path.join("logs", timestamp_dir)
        os.makedirs(log_dir, exist_ok=True)

        # Save the user prompt in prompt.json
        prompt_path = os.path.join(log_dir, "prompt.json")
        with open(prompt_path, "w", encoding="utf-8") as f:
            json.dump({"prompt": user_prompt}, f, indent=4)

        # Save the entire request
        req_path = os.path.join(log_dir, "request_body.json")
        with open(req_path, "w", encoding="utf-8") as f:
            json.dump(request_dict, f, indent=2)

    def _save_response_files(self, response_obj: Any):
        """
        Save the raw response dict and usage stats to logs/{timestamp}/response_body.json and usage_stats.json.

        Args:
            response_obj (Any): The raw object returned by openai client
                                Typically an openai.types.ChatCompletion or dictionary
        """
        # We do not have a direct 'timestamp' from the GPT response, so let's re-use the folder
        # or create a new one. We'll re-check last logs or create a new one.

        # We'll do a simplistic approach: always create a new folder for the response
        # or we could store them in the same folder as request. For clarity, let's do the same timestamp
        # if there's only one call. But we can't guarantee order for multiple calls simultaneously.
        # We'll just create a new timestamp dir for each call. It's simpler but you'll get more folders.
        timestamp_dir = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        log_dir = os.path.join("logs", timestamp_dir)
        os.makedirs(log_dir, exist_ok=True)

        # Convert the response to a dict
        # The new openai python lib returns typed objects. We can do .to_dict() if it's a pydantic model
        # or just do dict(response_obj) if it's not. We'll try to call .to_dict() else fallback.
        try:
            response_dict = response_obj.to_dict()
        except AttributeError:
            # maybe it's already a dict or standard object
            response_dict = response_obj

        # Write response_body.json
        resp_path = os.path.join(log_dir, "response_body.json")
        with open(resp_path, "w", encoding="utf-8") as f:
            json.dump(response_dict, f, indent=2)

        # Also usage_stats.json => usage
        usage_data = {}
        try:
            if hasattr(response_obj, "usage"):
                usage_data = dict(response_obj.usage)
            elif isinstance(response_dict, dict) and "usage" in response_dict:
                usage_data = response_dict["usage"]
        except Exception:
            logger.warning("No usage field found in GPT response")

        usage_path = os.path.join(log_dir, "usage_stats.json")
        with open(usage_path, "w", encoding="utf-8") as f:
            json.dump(usage_data, f, indent=2)
