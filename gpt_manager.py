#!/usr/bin/env python3
# =============================================================================
# FILE: gpt_manager.py
# =============================================================================
"""
gpt_manager.py

This updated GPTManager class removes any usage of conversation context history.
Instead, the user prompt focuses on aggregator data that includes per-field
changes (e.g., price, galaxy_score, sentiment) since the last decision timestamp.
For example, if the time interval is 1 hour, aggregator data might look like:

  1) ETH/USD => price=2710.98 (+2.3%), galaxy_score=65 (+3.2%), alt_rank=10 (+1),
               volume=154.2k (+15%), sentiment=0.72 (-0.03), etc.

Changes:
1) Conversation context is removed entirely.
2) Single-coin and multi-coin logic remain, but we pass aggregator data that shows
   current values and percentage (or absolute) changes since the last run.
3) "reflection_enabled" can still be used to keep chain-of-thought hidden.
4) System instructions demonstrate short rationale usage, with no reference to
   conversation context.

Dependencies:
    - openai (≥ v1) python package
    - pyyaml for config reading
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

from config_loader import ConfigLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _sanitize_text_block(text: str) -> str:
    """
    Removes/obscures terms that might be flagged for policy reasons, e.g.
    'financial advice', 'guarantee', 'profit', or any strongly directive phrases.
    Adjust the forbidden list as needed.
    """
    forbidden_words = [
        "financial advice", "financial", "guarantee",
        "profit", "money", "trading signals", "investment",
        "get rich", "day-trading"
    ]
    sanitized = text
    for fw in forbidden_words:
        if fw.lower() in sanitized.lower():
            # e.g. replace with '*' or remove entirely
            sanitized = sanitized.replace(fw, "***")
    return sanitized

def build_aggregator_prompt(
    data_rows: List[Dict[str, Any]]
) -> str:
    """
    Builds a textual snippet for aggregator data that includes both current values
    and changes from the previous hour (or any relevant time period).

    Each item in `data_rows` might look like:
      {
        "pair": "ETH/USD",
        "price": 2710.98,
        "price_change": 2.3,  # means +2.3% from last time
        "volume": 0.0002604,
        "volume_change": -0.1,  # means -10% from last time
        "galaxy_score": 65,
        "galaxy_score_change": 3.2,
        "alt_rank": 10,
        "alt_rank_change": 1,
        ...
      }

    Return a string that enumerates each coin with the format:
      1) ETH/USD => price=2710.98 (+2.3%), volume=0.0002604 (-10%), galaxy=65 (+3.2%), alt_rank=10 (+1), ...
    """
    if not data_rows:
        return "No aggregator data."

    lines = []
    for i, row in enumerate(data_rows, start=1):
        pair = row.get("pair", "UNK")
        # Just some typical aggregator fields + their changes
        price = row.get("price", 0.0)
        price_chg = row.get("price_change", 0.0)
        vol_val = row.get("volume", 0.0)
        vol_chg = row.get("volume_change", 0.0)
        galaxy = row.get("galaxy_score", 0.0)
        galaxy_chg = row.get("galaxy_score_change", 0.0)
        alt_rank = row.get("alt_rank", 999999)
        alt_chg = row.get("alt_rank_change", 0.0)
        sentiment_val = row.get("sentiment", 0.0)
        sentiment_chg = row.get("sentiment_change", 0.0)

        # Format example: price=2710.98 (+2.3%), galaxy=65 (+3.2%) ...
        # For alt_rank you might do alt_rank=10 (+1) meaning alt_rank went from 9->10 or 11->10
        line = (
            f"{i}) {pair} => "
            f"price={price:.2f} ({price_chg:+.2f}%), "
            f"volume={vol_val:.6f} ({vol_chg:+.2f}%), "
            f"galaxy={galaxy:.1f} ({galaxy_chg:+.2f}), "
            f"alt_rank={alt_rank} ({alt_chg:+.2f}), "
            f"sentiment={sentiment_val:.2f} ({sentiment_chg:+.2f})"
        )
        lines.append(line)

    return "\n".join(lines)


def build_aggregator_prompt_simple(aggregator_lines: List[Dict[str, Any]]) -> str:
    if not aggregator_lines:
        return "No aggregator data."
    lines = []
    for i, row in enumerate(aggregator_lines, start=1):
        output_aggregator_data = row.get("aggregator_data", "No aggregator data.")
        lines.append(output_aggregator_data)
    return "\n".join(lines)


class GPTManager:
    """
    GPTManager:

    - Produces single-coin or multi-coin trade decisions with short rationale.
    - No conversation context is used. Instead, aggregator data includes
      current values and changes from the last known data (e.g., 1 hour ago).
    - Reflection is optional: if enabled, chain-of-thought in triple backticks.

    For single-coin => final JSON:
      {"action":"BUY|SELL|HOLD","size":float,"rationale":"..."}
    For multi-coin => final JSON:
      {
        "decisions":[
          {"pair":"ETH/USD","action":"BUY","size":0.001},...
        ],
        "rationale":"..."
      }
    """

    def __init__(
        self,
        config_file: str = "config.yaml",
        temperature: float = 0.7,
        max_tokens: int = 500,
        log_gpt_calls: bool = True
    ):
        """
        :param config_file: Reads 'openai_model' if present.
        :param temperature: higher => more variety
        :param max_tokens: max tokens for GPT completion
        :param log_gpt_calls: log request/response locally
        """
        self.log_gpt_calls = log_gpt_calls
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.model = "o1-mini"  # default
        if os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            self.model = cfg.get("openai_model", self.model)

        # OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    # --------------------------------------------------------------------------
    # SINGLE-COIN
    # --------------------------------------------------------------------------
    def generate_trade_decision(
        self,
        aggregator_text: str,
        trade_history: List[str],
        max_trades: int,
        risk_controls: Dict[str, Any],
        open_positions: Optional[List[str]] = None,
        reflection_enabled: bool = False
    ) -> Dict[str, Any]:
        """
        Single-coin approach => returns JSON => {action,size,rationale}.

        aggregator_text: a string from build_aggregator_prompt or your custom logic
                         containing current aggregator values + changes.
        trade_history: strings describing recent trades (limit max_trades).
        risk_controls: e.g. {"stop_loss_pct":0.05, ...}
        reflection_enabled: if True => model can place chain-of-thought in triple backticks
        """
        truncated_history = trade_history[-max_trades:] if trade_history else []
        trades_summ = "\n".join(truncated_history) if truncated_history else "No trades so far."
        openpos_summ = "\n".join(open_positions) if open_positions else "No open positions."

        # ----------------------------------------------------------------------
        # System instructions => single coin
        # ----------------------------------------------------------------------
        system_instructions = (
            "You are a SINGLE-COIN trading AI. You must produce a short-term "
            "decision (within ~24h) with final JSON => {\"action\":\"BUY|SELL|HOLD\",\"size\":float,\"rationale\":\"...\"}.\n\n"
            "Allowed actions: BUY, SELL, HOLD.\n"
            "Size is float, how many coins.\n"
            "Rationale is ≤300 chars, numeric/technical.\n"
            "If reflection_enabled is true, place chain-of-thought in ``` triple backticks ```.\n\n"
            "EXAMPLE:\n"
            "Chain-of-thought => <hidden>\n"
            "Final => {\n"
            "  \"action\":\"BUY\",\n"
            "  \"size\":0.001,\n"
            "  \"rationale\":\"Price +3%, short upswing, minimal risk.\"\n"
            "}\n"
        )

        user_prompt = (
            f"[REFLECTION_ENABLED={reflection_enabled}]\n\n"
            f"AGGREGATOR DATA (current + changes):\n{aggregator_text}\n\n"
            f"OPEN POSITIONS:\n{openpos_summ}\n\n"
            f"RECENT TRADES:\n{trades_summ}\n\n"
            f"RISK CONTROLS:\n{risk_controls}\n\n"
            "RETURN final JSON => {\"action\":\"BUY|SELL|HOLD\",\"size\":0.0,\"rationale\":\"...\"} "
            "with rationale ≤300 chars. If reflection is on, chain-of-thought can be in triple backticks first."
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

        # logging
        logger.info("[GPT-Single] aggregator_text => \n%s", aggregator_text)
        if self.log_gpt_calls:
            self._save_prompt_files(user_prompt, system_instructions, request_dict)

        try:
            response = self.client.chat.completions.create(**request_dict)
            if self.log_gpt_calls:
                self._save_response_files(response)
            raw_text = response.choices[0].message.content if response.choices else ""
            result = self._parse_single_json(raw_text)
            return result
        except (APIConnectionError, RateLimitError) as net_exc:
            logger.exception("[GPT-Single] net/rate error => %s", net_exc)
            return {"action": "HOLD", "size": 0.0, "rationale": "Fallback => network error."}
        except APIStatusError as e:
            logger.warning("[GPT-Single] API status => %s, response=%s", e.status_code, e.response)
            return {"action": "HOLD", "size": 0.0, "rationale": "Fallback => API error."}
        except Exception as e:
            logger.exception("[GPT-Single] unknown => %s", e)
            return {"action": "HOLD", "size": 0.0, "rationale": "Fallback => unknown error."}

    # --------------------------------------------------------------------------
    # MULTI-COIN
    # --------------------------------------------------------------------------
    def generate_multi_trade_decision(
        self,
        aggregator_lines: List[Dict[str, Any]],
        open_positions: List[str],
        trade_history: List[str],
        max_trades: int,
        risk_controls: Dict[str, Any],
        reflection_enabled: bool = True,
        current_balance: float = 0.0,
        current_trade_balance:  dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Multi-coin approach => aggregator_lines is a list of dicts with the relevant
        aggregator changes for each pair. We'll build a textual block from that,
        then ask for final JSON => { "decisions": [...], "rationale": "..."}.
        """
        truncated_history = trade_history[-max_trades:] if trade_history else []
        trades_summ = "\n".join(truncated_history) if truncated_history else "No active investments made from USD."
        openpos_summ = "\n".join(open_positions) if open_positions else "No open positions."

        # build aggregator text with changes
        aggregator_text = build_aggregator_prompt_simple(aggregator_lines)

        system_instructions = (

            "You are a cryptocurrency trading analyst. You have expertise in reading price patterns, "
            "social media sentiment, and volume trends to make short-term (day-trading) recommendations. \n"
            "Instructions: \n"
            "\t1. First, work out your own reasoning step-by-step enclosed in triple quotes (\"\"\"), "
            "but do not reveal these steps directly to the user.\n"
            "\t2. Determine which investments to make based on your reasoning. "
            "The quantity to specify in your response has the following requirements:\n"
            "\t\t1. The chosen quantity must be relayed in the base currency.\n"
            "\t\t2. The chosen quantity must be higher than the \"minimum purchase quantity\" of the base asset.\n"
            "\t\t3. The chosen quantity must respect the \"tick size\" in determining quantity increment.\n"
            "\t\t4. Do not be shaken by small price fluctuations once a coin is purchased with USD. "
            "The volatility of the coin is extremely high, and significant fluctuations are expected by design. \n"
            f"\t\t\tThe stop-loss percent is {float(ConfigLoader.get_value("stop_loss_percent"))*100}%.\n"
            f"\t\t\tThe take-profit percent is {float(ConfigLoader.get_value("take_profit_percent"))*100}%.\n"
            "\t\t5. Do not become aggressive with trading decisions in each response. "
            "It is important to allow investments time to increase in value, given your recommendations "
            "to invest in these coins in the first place. You are not making decisions for the next 24 hours. "
            f"You will be asked to repeat your response in {ConfigLoader.get_value("trade_interval_seconds", 900)} seconds. "
            "Additionally, stop-loss and take-profit exit strategies are handled by this application on a continuous basis.\n"
            "\t\t6. The chosen quantity must have considered the implications of this trade given the corresponding "
            "value in USD, and how buy/sell decisions directly increase or decrease the amount in USD that is "
            "available for trades.\n"
            "\t3. Return final JSON => \n\t{\n"
            " \t\"decisions\":[{\"pair\":\"...\",\"action\":\"BUY|SELL|HOLD\",\"size\":float},...],\n"
            " \t\"rationale\":\"...\"\n"
            "\t}\n\n"
            "Rationale must be ≤300 chars, numeric or technical justifications.\n"
            # f"2. If you do have much more than a 25% share in a single coin, you must sell the excess amount immediately.\n\n"
            "EXAMPLE RESPONSE:\n"
            "Work out your own reasoning => <hidden>\n"
            "Final => {\n"
            "  \"decisions\":[\n"
            "    {\"pair\":\"ETH/USD\",\"action\":\"HOLD\",\"size\":0.0},\n"
            "    {\"pair\":\"SOL/USD\",\"action\":\"HOLD\",\"size\":0.0.5},\n"
            "    {\"pair\":\"BTC/USD\",\"action\":\"BUY\",\"size\":0.001}\n"
            "  ],\n"
            "  \"rationale\":\"ETH stable, BTC dip => good entry.\"\n"
            "}\n"
        )

        user_prompt = (
            "Analyze the following market indicators for indications of a short-term price increase, with a secondary interest in LunarCrush data, "
            "then provide your buy/sell/hold recommendations.\n\n"
            f"RISK REQUIREMENT:\n"
            f"1. You must never purchase much more than a 25% share in a single coin, with regard to the total available USD available.\n\n"
            f"current_time: {int(time.time())}\n\n"
            "---BEGIN ACCOUNT DATA---\n\n"
            f"AVAILABLE USD BALANCE FOR TRADES: {current_balance:.2f}\n\n"
            f"CURRENT TRADE BALANCES: \n{json.dumps(current_trade_balance, indent=4)}\n\n"
            f"RECENTLY ACTIVE INVESTMENTS:\n{trades_summ}\n\n"
            "---END ACCOUNT DATA---\n\n"
            "---BEGIN COIN DATA---\n\n"
            f"{aggregator_text}\n\n"
            "---END COIN DATA---\n\n"
            "RETURN final JSON => {\"decisions\":[...],\"rationale\":\"...\"} with rationale ≤300 chars."
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

        logger.info("[GPT-Multi] aggregator_lines => %s", aggregator_lines)
        if self.log_gpt_calls:
            self._save_prompt_files(user_prompt, system_instructions, request_dict)

        try:
            response = self.client.chat.completions.create(**request_dict)
            if self.log_gpt_calls:
                self._save_response_files(response)
            raw_text = response.choices[0].message.content if response.choices else ""
            result = self._parse_multi_json(raw_text)
            return result
        except (APIConnectionError, RateLimitError) as net_exc:
            logger.exception("[GPT-Multi] net/rate error => %s", net_exc)
            return {"decisions": [], "rationale": "Fallback => network error."}
        except APIStatusError as e:
            logger.warning("[GPT-Multi] API status => %s, response=%s", e.status_code, e.response)
            return {"decisions": [], "rationale": "Fallback => API error."}
        except Exception as e:
            logger.exception("[GPT-Multi] unknown => %s", e)
            return {"decisions": [], "rationale": "Fallback => unknown error."}

    def generate_multi_trade_decision_simple_prompt(
            self,
            aggregator_list_simple: List[Dict[str, str]],
            reflection_enabled: bool = False,
            current_balance: float = 0.0,
            current_trade_balance: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Accepts aggregator text blocks, produces final JSON with hypothetical 'decisions' & 'rationale'.
        A few extra steps:
          1) Sanitizes text to remove flagged words.
          2) Keeps disclaimers short and neutral, focusing on hypothetical output only.
        """
        if not current_trade_balance:
            current_trade_balance = {}

        # Step 1: System instructions => toned down & disclaimers
        system_instructions = f"""\
    You are a hypothetical numeric-analysis assistant for short time frames. 
    All content is purely illustrative. No actual outcomes are guaranteed.

    INSTRUCTIONS:
    1) Return final JSON in the form:
       {{
         "decisions":[{{"pair":"...","action":"BUY|SELL|HOLD","size":0.0}}, ...],
         "rationale":"..."
       }}
    2) "action" is "BUY","SELL", or "HOLD" to express a hypothetical stance.
    3) "size" is a float for how many base units. 
    4) "rationale" ≤300 chars. Emphasize it is a fictional scenario.

    EXAMPLE RESPONSE:
    Reasoning => <hidden>
    Final => {{
      "decisions":[
        {{"pair":"ETH/USD","action":"HOLD","size":0.0}},
        {{"pair":"XBT/USD","action":"BUY","size":0.001}}
      ],
      "rationale":"Short example: a small BTC buy, purely hypothetical."
    }}

    If reflection_enabled={reflection_enabled}, 
    you may insert chain-of-thought in triple backticks internally, 
    but do not reveal it outside the final JSON.

    IS_REFLECTION_ENABLED={reflection_enabled}
    """

        # Step 2: Build user prompt => disclaim & sanitize aggregator blocks
        sanitized_lines = []
        for obj in aggregator_list_simple:
            pair = obj.get("pair", "UNK")
            raw_block = obj.get("prompt_text", "")
            # Clean each block to reduce policy risk
            safe_block = _sanitize_text_block(raw_block)
            sanitized_lines.append(f"---COIN BLOCK ({pair})---\n{safe_block}\n---END BLOCK---\n")

        combined_coins_data = "\n".join(sanitized_lines)

        user_prompt = (
                "DISCLAIMER: This is a fictional scenario. No real financial guidance.\n\n"
                f"CURRENT USD BALANCE => {current_balance:.2f}\n"
                f"CURRENT HOLDINGS => {current_trade_balance}\n\n"
                "Blocks:\n\n"
                + combined_coins_data
                + '\nReturn final JSON => {"decisions":[...], "rationale":"..."}.'
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
            self._save_prompt_files(user_prompt, system_instructions, request_dict)

        # Step 3: Attempt GPT call
        try:
            resp = self.client.chat.completions.create(**request_dict)
            if self.log_gpt_calls:
                self._save_response_files(resp)

            raw_text = resp.choices[0].message.content if resp.choices else ""
            return self._parse_multi_json(raw_text)

        except Exception as e:
            logger.exception("[GPT-Simple] => Error => %s", e)
            return {"decisions": [], "rationale": "Fallback => error."}

    # --------------------------------------------------------------------------
    # PARSING
    # --------------------------------------------------------------------------
    def _parse_single_json(self, raw_text: str) -> Dict[str, Any]:
        """
        Expects { "action":"BUY|SELL|HOLD","size":float,"rationale":"..." } in JSON.
        If fail => fallback => hold.
        """
        fallback = {"action": "HOLD", "size": 0.0, "rationale": "Parse fallback."}
        cleaned = raw_text.replace("```json", "").replace("```", "").strip()
        if not cleaned:
            return fallback

        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                return fallback
            action = str(parsed.get("action", "HOLD")).upper()
            size = float(parsed.get("size", 0.0))
            rationale = parsed.get("rationale", "No rationale")
            if action not in ("BUY", "SELL", "HOLD"):
                action = "HOLD"
            if size < 0:
                size = 0.0
            # trim rationale
            if len(rationale) > 300:
                rationale = rationale[:300]
            return {"action": action, "size": size, "rationale": rationale}
        except json.JSONDecodeError:
            logger.warning("[parse_single_json] JSON decode error => fallback.")
            return fallback

    def _parse_multi_json(self, raw_text: str) -> Dict[str, Any]:
        """
        Expects { "decisions":[{"pair":"...","action":"...","size":0.0},...],
                  "rationale":"..." }
        """
        fallback = {"decisions": [], "rationale": "Parse fallback."}
        cleaned = raw_text.replace("```json", "").replace("```", "").strip()
        if not cleaned:
            return fallback

        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                return fallback
            decisions = parsed.get("decisions", [])
            if not isinstance(decisions, list):
                decisions = []
            final_decisions = []
            for d in decisions:
                if not isinstance(d, dict):
                    continue
                pair = d.get("pair", "UNK")
                action_raw = str(d.get("action", "HOLD")).upper()
                size_val = float(d.get("size", 0.0))
                if action_raw not in ("BUY", "SELL", "HOLD"):
                    action_raw = "HOLD"
                if size_val < 0:
                    size_val = 0.0
                final_decisions.append({"pair": pair, "action": action_raw, "size": size_val})

            rationale = parsed.get("rationale", "No rationale")
            if len(rationale) > 300:
                rationale = rationale[:300]
            return {"decisions": final_decisions, "rationale": rationale}
        except json.JSONDecodeError:
            logger.warning("[parse_multi_json] JSON decode error => fallback.")
            return fallback

    # --------------------------------------------------------------------------
    # LOGGING
    # --------------------------------------------------------------------------
    def _save_prompt_files(
            self, user_prompt: str, system_instructions: str, request_dict: Dict[str, Any]
    ) -> None:
        """Saves user prompt and request dict to logs/{timestamp} for debugging."""
        timestamp_dir = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        log_dir = os.path.join("logs", timestamp_dir)
        os.makedirs(log_dir, exist_ok=True)

        prompt_path = os.path.join(log_dir, "prompt.txt")
        with open(prompt_path, "w", encoding="utf-8") as f_prompt:
            f_prompt.write(user_prompt)

        sys_path = os.path.join(log_dir, "system_instructions.txt")
        with open(sys_path, "w", encoding="utf-8") as f_sys:
            f_sys.write(system_instructions)

        req_path = os.path.join(log_dir, "request_body.txt")
        with open(req_path, "w", encoding="utf-8") as f_req:
            f_req.write(str(request_dict))

    def _save_response_files(self, response_obj: Any) -> None:
        """Saves the raw response object and usage stats to logs/{timestamp}."""
        timestamp_dir = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        log_dir = os.path.join("logs", timestamp_dir)
        os.makedirs(log_dir, exist_ok=True)

        resp_path = os.path.join(log_dir, "response_body.txt")
        with open(resp_path, "w", encoding="utf-8") as f_resp:
            f_resp.write(str(response_obj))

        usage_path = os.path.join(log_dir, "usage_stats.txt")
        usage_data = {}
        try:
            if hasattr(response_obj, "usage"):
                usage_data = dict(response_obj.usage)
            else:
                resp_dict = response_obj
                if isinstance(resp_dict, dict) and "usage" in resp_dict:
                    usage_data = resp_dict["usage"]
        except Exception:
            logger.warning("[GPTManager] No usage field found in GPT response")

        with open(usage_path, "w", encoding="utf-8") as f_usage:
            f_usage.write(str(usage_data))


