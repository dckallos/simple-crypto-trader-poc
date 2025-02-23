#!/usr/bin/env python3
# =============================================================================
# FILE: gpt_manager.py
# =============================================================================
"""
gpt_manager.py (Refactored, Production-Ready)

Enhancements:
-------------
1) We now parse an optional "confidence" float from the GPT JSON if present,
   returning it in our final dict under "confidence".
2) We add an optional mechanism to store aggregator context (price changes,
   last price, LunarCrush data, risk configs, etc.) to the ai_snapshots table
   if desired, for ML usage.

Key points:
-----------
 - The Mustache template should provide "confidence" in the final JSON if you
   want to track it. Example:

    {
      "decisions":[ {...}, ...],
      "rationale":"some text",
      "confidence": 0.85
    }

 - The aggregator or AIStrategy can optionally call `generate_decisions_from_prompt(...,
   aggregator_data=...)` and pass details such as:

     aggregator_data = {
       "pair": "ETH/USD",
       "price_changes": [...],
       "last_price": 123.45,
       "stop_loss_pct": 0.04,
       "take_profit_pct": 0.01,
       "daily_drawdown_limit": -0.02,
       "lunarcrush_data": {...},
       "notes": "any additional info",
       ...
     }

   The GPTManager then automatically calls `store_ai_snapshot(...)` in db.py
   with these aggregator fields plus the GPT "confidence" if present.

 - This design ensures we have a robust record of each AI inference, including
   the numeric context, for future ML analysis.

Dependencies:
-------------
 - openai (≥ v1)
 - pyyaml
 - Python 3.8+
 - The local `db.py` must define `store_ai_snapshot(...)`.
"""

import os
import json
import re

import httpx
import yaml
import logging
import datetime
from typing import Dict, Any, Optional

from openai import OpenAI
from openai import APIConnectionError, APIStatusError, RateLimitError

# Local imports if needed
import db

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GPTManager:
    """
    GPTManager: A single-purpose class that:
        1) Receives a final prompt string, fully composed by your Mustache template.
        2) Sends it to OpenAI's Chat Completion endpoint, using a single user message.
        3) Parses the returned JSON into { "decisions": [...], "rationale": "...", "confidence": float? }.

    If any step fails (network issues, parse errors, etc.), we return:
        { "decisions": [], "rationale": "Fallback => error.", "confidence": 0.0 }

    Optionally, if aggregator_data is provided, we store an AI snapshot in the
    DB with aggregator context plus the parsed confidence. This helps with ML.
    """

    def __init__(
        self,
        config_file: str = "config.yaml",
        temperature: float = 1.0,
        max_tokens: int = 1000,
        log_gpt_calls: bool = True
    ):
        """
        :param config_file: Path to your YAML config, if it has an `openai_model` key.
        :param temperature: GPT temperature for creative variation.
        :param max_tokens: GPT model max tokens in the response.
        :param log_gpt_calls: If True, saves request/response details in ./logs/
        """
        self.log_gpt_calls = log_gpt_calls
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Load default model from config
        self.model = "o1-mini"
        if os.path.exists(config_file):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                maybe_model = cfg.get("openai_model")
                if maybe_model:
                    self.model = maybe_model
            except Exception as e:
                logger.warning(f"[GPTManager] Error loading config: {e} => using default model '{self.model}'")

        # Initialize OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_api_key:
            logger.warning("[GPTManager] OPENAI_API_KEY is not set. GPT calls will fail if invoked.")
        self.client = OpenAI(
            api_key=openai_api_key,
            max_retries=0,
            http_client=httpx.Client(
                ## TODO: Make configurable
                timeout=120.0,
            )
        )

    def generate_decisions_from_prompt(
        self,
        final_prompt_text: str,
        aggregator_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Given a fully-formed prompt string (usually from a Mustache template),
        calls OpenAI Chat Completion with a single user message. Then parses
        the returned JSON into:
            {
              "decisions": [...],
              "rationale": "...",
              "confidence": float (optional)
            }

        If parsing fails or an API error occurs, we return:
            {"decisions": [], "rationale": "Fallback => error.", "confidence": 0.0}

        aggregator_data (optional) can be:
            {
              "pair": "ETH/USD",
              "price_changes": [...],
              "last_price": 123.45,
              "aggregator_interval_secs": 120,
              "stop_loss_pct": 0.04,
              "take_profit_pct": 0.01,
              "daily_drawdown_limit": -0.02,
              "coin_volatility": 0.17,
              "lunarcrush_data": {...},
              "is_market_bullish": "YES"/"NO",
              "risk_estimate": 0.75,
              "notes": "anything else"
            }

        If aggregator_data is provided, we will automatically store a row
        in ai_snapshots to track these numeric indicators plus the GPT "confidence."
        """
        # 1) Quick check for empty prompt
        if not final_prompt_text.strip():
            logger.warning("[GPTManager] Received empty prompt => returning fallback.")
            fallback = {"decisions": [], "rationale": "Fallback => empty prompt.", "confidence": 0.0}
            self._maybe_store_ai_snapshot(fallback, aggregator_data)
            return fallback

        # 2) Build Chat Completion request
        messages = [
            {"role": "user", "content": final_prompt_text}
        ]
        request_dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
        }

        from config_loader import ConfigLoader
        if ConfigLoader.get_value("is_reasoning_model"):
            request_dict["reasoning_effort"] = ConfigLoader.get_value("reasoning_effort")


        # 3) Optionally log request
        if self.log_gpt_calls:
            self._save_prompt_files(final_prompt_text, request_dict)

        # 4) Send to OpenAI
        raw_text = ""
        gpt_parsed = {}
        try:
            response = self.client.chat.completions.create(**request_dict)
            if self.log_gpt_calls:
                self._save_response_files(response)

            raw_text = response.choices[0].message.content if response.choices else ""
            gpt_parsed = self._parse_decision_json(raw_text)

        except (APIConnectionError, RateLimitError) as net_exc:
            logger.exception("[GPTManager] Network or rate-limit error => %s", net_exc)
            gpt_parsed = {"decisions": [], "rationale": "Fallback => network error.", "confidence": 0.0}
        except APIStatusError as e:
            logger.warning("[GPTManager] API status => %s, response=%s", e.status_code, e.response)
            gpt_parsed = {"decisions": [], "rationale": "Fallback => API error.", "confidence": 0.0}
        except Exception as e:
            logger.exception("[GPTManager] Unexpected => %s", e)
            gpt_parsed = {"decisions": [], "rationale": "Fallback => unknown error.", "confidence": 0.0}

        # 5) Possibly store aggregator data + GPT "confidence" in ai_snapshots
        self._maybe_store_ai_snapshot(gpt_parsed, aggregator_data)

        return gpt_parsed

    # --------------------------------------------------------------------------
    # Private methods
    # --------------------------------------------------------------------------

    def _parse_decision_json(self, raw_text: str) -> Dict[str, Any]:
        """
        Attempts to parse GPT's reply for a valid JSON block. We also attempt to
        parse an optional "confidence" field.
        """
        fallback = {"decisions": [], "rationale": "Fallback => parse error.", "confidence": 0.0}
        if not raw_text.strip():
            return fallback

        # 1) Attempt to find a ```json ... ``` fenced block
        match = re.search(r'```json(.*?)```', raw_text, re.DOTALL | re.IGNORECASE)
        candidate = ""
        if match:
            candidate = match.group(1).strip()
        else:
            # fallback: strip code fences in the entire text
            candidate = raw_text.replace("```json", "").replace("```", "").strip()

        # 2) Attempt JSON decode
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            logger.warning("[GPTManager] JSON decode error => fallback parse => raw_text length=%d", len(raw_text))
            return fallback

        if not isinstance(parsed, dict):
            return fallback

        # parse decisions
        decisions = parsed.get("decisions", [])
        if not isinstance(decisions, list):
            decisions = []

        # parse rationale
        rationale = parsed.get("rationale", "No rationale")
        if not isinstance(rationale, str):
            rationale = str(rationale)
        if len(rationale) > 300:
            rationale = rationale[:300]

        # parse confidence
        confidence_val = 0.0
        if "confidence" in parsed:
            try:
                confidence_val = float(parsed["confidence"])
            except:
                pass

        # build final decisions
        final_decisions = []
        for d in decisions:
            if not isinstance(d, dict):
                continue
            pair = str(d.get("pair", "UNK"))
            action = str(d.get("action", "HOLD")).upper()
            size = float(d.get("size", 0.0))
            # clamp action
            if action not in ("BUY", "SELL", "HOLD"):
                action = "HOLD"
            if size < 0:
                size = 0.0
            final_decisions.append({"pair": pair, "action": action, "size": size})

        return {
            "decisions": final_decisions,
            "rationale": rationale,
            "confidence": confidence_val
        }

    def _save_prompt_files(self, prompt_text: str, request_dict: Dict[str, Any]) -> None:
        """Saves the user prompt text and request body for debugging."""
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

        # Usage stats if present
        usage_path = os.path.join(log_dir, "usage_stats.txt")
        usage_data = {}
        try:
            usage_data = dict(response_obj.usage) if hasattr(response_obj, "usage") else {}
        except Exception:
            logger.warning("[GPTManager] No usage field found in GPT response")

        with open(usage_path, "w", encoding="utf-8") as f_usage:
            f_usage.write(str(usage_data))

    def _maybe_store_ai_snapshot(self, gpt_parsed: Dict[str, Any], aggregator_data: Optional[Dict[str, Any]]):
        """
        If aggregator_data is provided, store a row in ai_snapshots with
        relevant numeric context plus the GPT confidence. This is optional.
        """
        if not aggregator_data:
            return  # skip

        # For safety, gather fields from aggregator_data
        pair = aggregator_data.get("pair", "ALL")
        price_changes = aggregator_data.get("price_changes", [])
        last_price = float(aggregator_data.get("last_price", 0.0))
        aggregator_interval_secs = int(aggregator_data.get("aggregator_interval_secs", 0))
        stop_loss_pct = float(aggregator_data.get("stop_loss_pct", 0.0))
        take_profit_pct = float(aggregator_data.get("take_profit_pct", 0.0))
        daily_drawdown_limit = float(aggregator_data.get("daily_drawdown_limit", 0.0))
        coin_volatility = float(aggregator_data.get("coin_volatility", 0.0))
        is_market_bullish = str(aggregator_data.get("is_market_bullish", "UNKNOWN"))
        risk_estimate = float(aggregator_data.get("risk_estimate", 0.0))
        lunarcrush_data = aggregator_data.get("lunarcrush_data", {})
        notes = str(aggregator_data.get("notes", ""))

        # GPT confidence
        gpt_confidence = float(gpt_parsed.get("confidence", 0.0))

        # We can place the confidence or any other AI metric in notes, or store separately
        # For now let's store it in 'risk_estimate' if you prefer, or keep separate. We'll keep it separate in 'notes'.
        if gpt_confidence != 0.0:
            notes = f"{notes}\n[GPT Confidence={gpt_confidence}]".strip()

        # Now call db.store_ai_snapshot
        db.store_ai_snapshot(
            pair=pair,
            price_changes=price_changes,
            last_price=last_price,
            aggregator_interval_secs=aggregator_interval_secs,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            daily_drawdown_limit=daily_drawdown_limit,
            coin_volatility=coin_volatility,
            is_market_bullish=is_market_bullish,
            risk_estimate=risk_estimate,
            lunarcrush_data=lunarcrush_data,
            notes=notes
        )

        logger.info("[GPTManager] Stored AI snapshot w/ aggregator data & GPT confidence=%s", gpt_confidence)
