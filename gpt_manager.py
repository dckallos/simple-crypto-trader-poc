#!/usr/bin/env python3
# =============================================================================
# FILE: gpt_manager.py
# =============================================================================
"""
gpt_manager.py (Refactored, Production-Ready)

This GPTManager class focuses on a single, unified approach to retrieving
trading decisions from GPT using a *pre-rendered prompt* from Mustache.
All "system instructions" or "user instructions" must be provided via the
external Mustache template, so there is no inline prompt content here.

In short:
    1) You feed GPTManager a single string (final_prompt_text) that
       already has instructions, disclaimers, aggregator data, etc.
    2) GPTManager calls OpenAI's Chat Completion endpoint with that text
       (i.e., a single user message).
    3) We parse the returned JSON into the format:
         {
           "decisions": [
             {"pair":"ETH/USD","action":"BUY|SELL|HOLD","size":float}, ...
           ],
           "rationale": "string up to 300 chars"
         }
       If parsing fails, we return a fallback structure with empty decisions.

Key notes:
    - This class does NOT embed system or user instructions. The Mustache
      template (aggregator_simple_prompt.mustache, etc.) must produce the
      final prompt content, including disclaimers and instructions.
    - We do not maintain separate "single-coin" or "multi-coin" aggregator
      flows here. There's one method, generate_decisions_from_prompt(...),
      which expects the final JSON to have "decisions" and "rationale".
    - If you need to accommodate single-coin usage, simply have your Mustache
      template produce a single item in "decisions", e.g.:
         { "decisions":[ {"pair":"X","action":"BUY","size":...} ], "rationale":... }

Dependencies:
    - openai (≥ v1)
    - pyyaml
    - Python 3.8+
"""

import os
import json
import re
import yaml
import logging
import datetime
from typing import Dict, Any

from openai import OpenAI
from openai import APIConnectionError, APIStatusError, RateLimitError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GPTManager:
    """
    GPTManager: A single-purpose class that:
        1) Receives a final prompt string, fully composed by your Mustache template.
        2) Sends it to OpenAI's Chat Completion endpoint, using a single user message.
        3) Parses the returned JSON into { "decisions": [...], "rationale": "..." }.

    If any step fails (network issues, parse errors, etc.), we return:
        { "decisions": [], "rationale": "Fallback => error." }

    The actual aggregator logic or multi vs. single coin approach is handled
    *outside* this file (in Mustache templates and aggregator code).
    """

    def __init__(
        self,
        config_file: str = "config.yaml",
        temperature: float = 1.0,
        max_tokens: int = 1000,
        log_gpt_calls: bool = True
    ):
        """
        :param config_file: Path to your YAML config, if it has a `openai_model` key.
        :param temperature: GPT temperature for creative variation.
        :param max_tokens: Max tokens in the assistant’s reply.
        :param log_gpt_calls: If True, saves request/response details in ./logs/
        """
        self.log_gpt_calls = log_gpt_calls
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.model = "o1-mini"
        if os.path.exists(config_file):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                maybe_model = cfg.get("openai_model")
                if maybe_model:
                    self.model = maybe_model
            except Exception as e:
                logger.warning(f"[GPTManager] Error loading config: {e} => using default model {self.model}")

        # Initialize OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_api_key:
            logger.warning("[GPTManager] OPENAI_API_KEY is not set. GPT calls will fail if invoked.")
        self.client = OpenAI(api_key=openai_api_key)

    def generate_decisions_from_prompt(self, final_prompt_text: str) -> Dict[str, Any]:
        """
        Given a fully-formed prompt string (usually from a Mustache template),
        calls OpenAI Chat Completion using a single user message. Then parses
        the returned JSON into:
            {
              "decisions": [
                 {"pair":"ETH/USD","action":"BUY|SELL|HOLD","size":0.01},
                 ...
              ],
              "rationale": "short explanation"
            }

        If parsing fails or an API error occurs, we return:
            {"decisions": [], "rationale": "Fallback => error."}

        :param final_prompt_text: The entire prompt to pass to GPT (including disclaimers, instructions).
        :return: A dict with "decisions" (list) and "rationale" (string).
        """
        if not final_prompt_text.strip():
            logger.warning("[GPTManager] Received empty prompt => returning fallback.")
            return {"decisions": [], "rationale": "Fallback => empty prompt."}

        # Prepare Chat Completion request
        messages = [
            {"role": "user", "content": final_prompt_text}
        ]
        request_dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
        }

        if self.log_gpt_calls:
            self._save_prompt_files(final_prompt_text, request_dict)

        try:
            response = self.client.chat.completions.create(**request_dict)
            if self.log_gpt_calls:
                self._save_response_files(response)

            raw_text = response.choices[0].message.content if response.choices else ""
            return self._parse_decision_json(raw_text)
        except (APIConnectionError, RateLimitError) as net_exc:
            logger.exception("[GPTManager] Network or rate-limit error => %s", net_exc)
            return {"decisions": [], "rationale": "Fallback => network error."}
        except APIStatusError as e:
            logger.warning("[GPTManager] API status => %s, response=%s", e.status_code, e.response)
            return {"decisions": [], "rationale": "Fallback => API error."}
        except Exception as e:
            logger.exception("[GPTManager] Unexpected => %s", e)
            return {"decisions": [], "rationale": "Fallback => unknown error."}

    # --------------------------------------------------------------------------
    # Internal: parse the final JSON string
    # --------------------------------------------------------------------------
    def _parse_decision_json(self, raw_text: str) -> Dict[str, Any]:
        """
        Attempts to locate valid JSON inside a triple-backtick code fence labeled as json.
        For example:

            ```json
            {
              "decisions":[...],
              "rationale":"..."
            }
            ```
            Additional disclaimers might appear after the closing ```.

        1) We first look for any pattern like: ```json ... ```
        2) If found, we parse that substring as JSON.
        3) If not, we fallback to removing ```json and ``` from the entire raw_text (old approach).
        4) If still invalid, return an empty "decisions" with "rationale" fallback.

        :param raw_text: The raw GPT assistant message, which may contain disclaimers,
                         triple backticks, etc.
        :return: dict => { "decisions":[...], "rationale":"..." }
        """
        fallback = {"decisions": [], "rationale": "Fallback => parse error."}
        if not raw_text.strip():
            return fallback

        # 1) Attempt to find the fenced JSON block
        match = re.search(r'```json(.*?)```', raw_text, re.DOTALL | re.IGNORECASE)
        if match:
            # Extract just what’s inside the triple backtick
            candidate = match.group(1).strip()
            logger.debug(f"[GPTManager] Found fenced JSON => {candidate[:100]}...")

            # Try parsing that snippet
            try:
                return self._attempt_json_parse(candidate, fallback)
            except json.JSONDecodeError:
                logger.warning("[GPTManager] code fence JSON decode error => fallback to entire text parse.")

        # 2) If no code fence (or parse error), fallback to old approach of stripping fences
        #    This might still fail if disclaimers appear after the JSON
        cleaned = raw_text.replace("```json", "").replace("```", "").strip()

        # One more parse attempt
        return self._attempt_json_parse(cleaned, fallback)

    # --------------------------------------------------------------------------
    # Logging / debug
    # --------------------------------------------------------------------------
    def _save_prompt_files(self, prompt_text: str, request_dict: Dict[str, Any]) -> None:
        """Saves the user prompt text and request body to logs/{timestamp}/ for debugging."""
        timestamp_dir = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        log_dir = os.path.join("logs", timestamp_dir)
        os.makedirs(log_dir, exist_ok=True)

        prompt_path = os.path.join(log_dir, "user_prompt.txt")
        with open(prompt_path, "w", encoding="utf-8") as f_prompt:
            f_prompt.write(prompt_text)

        req_path = os.path.join(log_dir, "request_body.json")
        with open(req_path, "w", encoding="utf-8") as f_req:
            json_str = json.dumps(request_dict, indent=4)
            f_req.write(json_str)

    def _save_response_files(self, response_obj: Any) -> None:
        """Saves the raw GPT response object and usage to logs/{timestamp}/ for debugging."""
        timestamp_dir = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        log_dir = os.path.join("logs", timestamp_dir)
        os.makedirs(log_dir, exist_ok=True)

        # Full response
        resp_path = os.path.join(log_dir, "response_body.txt")
        with open(resp_path, "w", encoding="utf-8") as f_resp:
            f_resp.write(str(response_obj))

        # Usage stats
        usage_path = os.path.join(log_dir, "usage_stats.txt")
        usage_data = {}
        try:
            # Check if the response object contains usage
            usage_data = dict(response_obj.usage) if hasattr(response_obj, "usage") else {}
        except Exception:
            logger.warning("[GPTManager] No usage field found in GPT response")

        with open(usage_path, "w", encoding="utf-8") as f_usage:
            f_usage.write(str(usage_data))

    def _attempt_json_parse(self, candidate: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempts json.loads on candidate. If success, we do minimal validation
        on 'decisions' & 'rationale'. Otherwise returns fallback.
        """
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            return fallback

        decisions = parsed.get("decisions", [])
        rationale = parsed.get("rationale", "No rationale")
        if not isinstance(decisions, list):
            decisions = []
        if len(rationale) > 300:
            rationale = rationale[:300]

        final_decisions = []
        for d in decisions:
            if not isinstance(d, dict):
                continue
            pair = str(d.get("pair", "UNK"))
            action = str(d.get("action", "HOLD")).upper()
            size = float(d.get("size", 0.0))
            if action not in ("BUY", "SELL", "HOLD"):
                action = "HOLD"
            if size < 0:
                size = 0.0
            final_decisions.append({"pair": pair, "action": action, "size": size})

        return {
            "decisions": final_decisions,
            "rationale": rationale
        }
