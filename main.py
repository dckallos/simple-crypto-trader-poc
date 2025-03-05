#!/usr/bin/env python3
# ==============================================================================
# FILE: main.py
# ==============================================================================
"""
main.py

Consolidated single-aggregator design with enhanced ai_snapshots integration:
--------------------------------------------------------------------
This version retains the original workflow while enhancing the 'ai_snapshots' table
with advanced metrics during the aggregator_cycle:

1) Fetches ledger data => updates local DB
2) Fetches current Kraken balances => determines ZUSD
3) Builds aggregator data for each pair => stores enhanced data in ai_snapshots
4) Loads a Mustache template name from config (prompt_template_name or fallback)
5) Renders the Mustache template with aggregator data
6) Calls AIStrategy => predict_from_prompt => GPT => create pending trades => possibly place real orders

Enhancements:
- Volatility: Computes coin_volatility from price_changes or fetches from LunarCrush.
- Market Sentiment: Uses LunarCrush sentiment for is_market_bullish ("YES" > 65, "NO" < 35, "MIXED" otherwise).
- Risk Estimate: Combines GPT confidence (70%) with historical trade success rate (30%).
- All unrelated logic (e.g., WebSocket handling, risk management) remains unchanged.

Graceful exit via KeyboardInterrupt cancels the risk_manager_task and stops WebSockets.
"""

import time
import requests
import hashlib
import hmac
import base64
import asyncio
import logging.config
import os
import json
import warnings
import sqlite3
from typing import Optional, Dict, Any, List
import statistics
from threading import Lock

import pandas as pd
from dotenv import load_dotenv

# Local modules
import db
from db import init_db, DB_FILE, store_ai_snapshot
from config_loader import ConfigLoader
from risk_manager import RiskManagerDB
from kraken_rest_manager import KrakenRestManager
from fetch_lunarcrush import LunarCrushFetcher
from ws_data_feed import KrakenPublicWSClient, KrakenPrivateWSClient
from ai_strategy import AIStrategy
from prompt_builder import PromptBuilder

import db_lookup

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ai_lock = Lock()

warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)


class NoHeartbeatFilter(logging.Filter):
    """
    A custom logging filter that excludes any records containing 'heartbeat'
    in their message (case-insensitive). This helps remove repetitive messages
    from ws_data_feed.py without raising the log level or altering other logs.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage().lower()
        if "heartbeat" in message:
            return False
        if "< TEXT '[" in message:
            return False
        return True


LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s %(levelname)s [%(name)s] %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
    },
    'loggers': {
        '': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': True
        },
        'requests': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True
        },
        'urllib3': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True
        },
    }
}


def compute_minute_percent_changes(price_records: list[tuple[int, float]]) -> list[float]:
    """
    Given a list of (timestamp, price) in ascending order, return an array of % changes
    from each record to the next.

    If price[i] == 0, or there's not enough data, returns 0 for that slot.

    Example:
       price_records = [(t1, p1), (t2, p2), (t3, p3)]
       returns => [%_change_2_1, %_change_3_2]

    :return: e.g. [+0.12, -0.03, +0.25, ...]
    """
    changes = []
    for i in range(1, len(price_records)):
        _, prev_price = price_records[i - 1]
        _, curr_price = price_records[i]
        if prev_price == 0:
            pct_change = 0.0
        else:
            pct_change = (curr_price - prev_price) / prev_price * 100.0
        changes.append(pct_change)
    return changes


def convert_base_asset_keys_to_wsname(trade_balances: Dict[str, float]) -> Dict[str, float]:
    """
    Convert each key in trade_balances from a base-asset name (e.g. "XETH")
    into its corresponding wsname, e.g. "ETH/USD".

    :param trade_balances: something like {"XETH": 1.25, "XXBT": 0.3, ...}
    :return: new dict with e.g. {"ETH/USD": 1.25, "XBT/USD": 0.3, ...}
             if no wsname is found for a base asset, that entry is skipped.
    """
    import db_lookup
    new_dict = {}
    for base_asset, balance in trade_balances.items():
        wsname = db_lookup.get_websocket_name_from_base_asset(base_asset)
        if wsname:
            new_dict[wsname] = balance
        else:
            logger.warning(f"No wsname found for base_asset={base_asset}, skipping.")
            pass
    return new_dict


def setup_logging():
    """
    Sets up Python logging using the existing LOG_CONFIG, then attaches
    a NoHeartbeatFilter to the logger used by ws_data_feed.py. This
    ensures that only 'heartbeat' messages are suppressed, preserving all
    other logs at their current levels.
    """
    import logging.config
    logging.config.dictConfig(LOG_CONFIG)

    ws_logger = logging.getLogger("ws_data_feed")
    wc_logger = logging.getLogger("websockets.client")
    main_logger = logging.getLogger("__main__")

    no_heartbeat_filter = NoHeartbeatFilter()
    ws_logger.addFilter(no_heartbeat_filter)
    wc_logger.addFilter(no_heartbeat_filter)
    main_logger.addFilter(no_heartbeat_filter)

    logger.info("Logging setup complete; heartbeat messages are now suppressed.")


def fetch_kraken_balance(api_key: str, api_secret: str) -> Dict[str, float]:
    """
    Calls Kraken's /0/private/Balance endpoint to retrieve all cash balances
    (e.g. "ZUSD", "ZEUR", "XXBT", etc.). Returns a dict of {asset: float_balance}
    or empty if error.
    """
    url = "https://api.kraken.com/0/private/Balance"
    path = "/0/private/Balance"
    nonce_val = int(time.time() * 1000)

    postdata = f"nonce={nonce_val}"
    sha256_digest = hashlib.sha256((str(nonce_val) + postdata).encode("utf-8")).digest()
    message = path.encode("utf-8") + sha256_digest
    secret = base64.b64decode(api_secret)
    sig = hmac.new(secret, message, hashlib.sha512)
    signature = base64.b64encode(sig.digest())

    headers = {
        'API-Key': api_key,
        'API-Sign': signature.decode(),
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    try:
        resp = requests.post(url, headers=headers, data=postdata, timeout=10)
        resp.raise_for_status()
        j = resp.json()
        if j.get("error"):
            logger.error(f"[Balance] Kraken returned error => {j['error']}")
            return {}
        result = j.get("result", {})
        out = {}
        for k, v in result.items():
            try:
                out[k] = float(v)
            except:
                out[k] = 0.0
        return out
    except requests.exceptions.RequestException as e:
        logger.exception(f"[Balance] Error fetching kraken balance => {e}")
        return {}


def fetch_and_store_kraken_ledger(
        api_key: str,
        api_secret: str,
        asset: str = None,
        ledger_type: str = "all",
        start: int = None,
        end: int = None,
        db_path: str = "trades.db"
):
    """
    Calls /0/private/Ledgers with the specified filters, then upserts each ledger entry
    into 'ledger_entries' table. This helps track deposits, withdrawals, etc.
    """
    import urllib.parse
    url = "https://api.kraken.com/0/private/Ledgers"
    path = "/0/private/Ledgers"

    nonce_val = str(int(time.time() * 1000))
    payload = {"nonce": nonce_val, "type": ledger_type}
    if asset:
        payload["asset"] = asset
    if start:
        payload["start"] = start
    if end:
        payload["end"] = end

    postdata_str = urllib.parse.urlencode(payload)

    # Build signature
    sha256_digest = hashlib.sha256((nonce_val + postdata_str).encode("utf-8")).digest()
    message = path.encode("utf-8") + sha256_digest
    secret = base64.b64decode(api_secret)
    sig = hmac.new(secret, message, hashlib.sha512)
    signature = base64.b64encode(sig.digest())

    headers = {
        "API-Key": api_key,
        "API-Sign": signature.decode(),
        "Content-Type": "application/x-www-form-urlencoded"
    }

    try:
        r = requests.post(url, headers=headers, data=postdata_str, timeout=10)
        r.raise_for_status()
        j = r.json()
        if j.get("error"):
            logger.error(f"[Ledger] Kraken returned error => {j['error']}")
            return
        ledger_dict = j.get("result", {}).get("ledger", {})
        if not ledger_dict:
            logger.info("[Ledger] No ledger entries returned.")
            return

        conn = sqlite3.connect(db_path)
        try:
            c = conn.cursor()
            rows_inserted = 0
            for ledger_id, entry_obj in ledger_dict.items():
                refid = entry_obj.get("refid", "")
                time_val = float(entry_obj.get("time", 0.0))
                ltype = entry_obj.get("type", "")
                subtype = entry_obj.get("subtype", "")
                asset_val = entry_obj.get("asset", "")
                amt = float(entry_obj.get("amount", 0.0))
                fee = float(entry_obj.get("fee", 0.0))
                bal = float(entry_obj.get("balance", 0.0))

                c.execute("""
                    INSERT OR REPLACE INTO ledger_entries (
                        ledger_id, refid, time, type, subtype, asset, amount, fee, balance
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (ledger_id, refid, time_val, ltype, subtype, asset_val, amt, fee, bal))
                rows_inserted += 1
            conn.commit()
            logger.info(f"[Ledger] Upserted {rows_inserted} ledger entries.")
        except Exception as e:
            logger.exception(f"[Ledger] Error storing ledger => {e}")
        finally:
            conn.close()

    except requests.exceptions.RequestException as e:
        logger.exception(f"[Ledger] HTTP error => {e}")


def get_latest_zusd_balance(db_path="trades.db") -> float:
    """
    Reads ledger_entries for 'ZUSD' => returns the last known balance.
    This is a naive approach, your ledger might have multiple changes.
    """
    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        q = """
        SELECT balance
        FROM ledger_entries
        WHERE asset='ZUSD'
        ORDER BY time DESC
        LIMIT 1
        """
        row = c.execute(q).fetchone()
        if row:
            return float(row[0])
        return 0.0
    except Exception as e:
        logger.exception(f"[Ledger] error retrieving ZUSD balance => {e}")
        return 0.0
    finally:
        conn.close()


def fetch_and_store_kraken_public_asset_pairs(
        pair_list: Optional[List[str]] = None,
        info: str = "info",
        country_code: Optional[str] = None
):
    """
    (Truncated doc) => Retrieves and stores kraken asset pairs in 'kraken_asset_pairs'.
    """
    from db import store_kraken_asset_pair_info

    base_url = "https://api.kraken.com/0/public/AssetPairs"
    params = {}
    if pair_list:
        joined_pairs = ",".join(pair_list)
        params["pair"] = joined_pairs
    if info:
        params["info"] = info
    if country_code:
        params["country_code"] = country_code

    logger.info(f"[AssetPairs] GET => {base_url}, params={params}")
    try:
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        j = resp.json()
        err_arr = j.get("error", [])
        if err_arr:
            logger.error(f"[AssetPairs] Returned error => {err_arr}")
            return

        results = j.get("result", {})
        if not results:
            logger.warning("[AssetPairs] result is empty => no asset pairs returned.")
            return

        count = 0
        for pair_name, pair_info in results.items():
            store_kraken_asset_pair_info(pair_name, pair_info)
            count += 1
        logger.info(f"[AssetPairs] Upserted {count} pairs into kraken_asset_pairs.")
    except requests.exceptions.RequestException as e:
        logger.exception(f"[AssetPairs] HTTP request error => {e}")
    except json.JSONDecodeError as e:
        logger.exception(f"[AssetPairs] Non-JSON response => {e}")
    except Exception as e:
        logger.exception(f"[AssetPairs] Unexpected => {e}")


def format_no_sci(value: float, decimal_places: int = 8) -> str:
    """
    Returns a string representation of 'value' without scientific notation,
    truncated or rounded to 'decimal_places' digits after the decimal.

    Examples:
      format_no_sci(5e-05) => "0.00005000" (default 8 decimals)
      format_no_sci(12345.6789, decimal_places=2) => "12345.68"

    :param value: the float to format
    :param decimal_places: how many digits after the decimal
    :return: string e.g. "0.00005"
    """
    format_str = f"{{:.{decimal_places}f}}"
    out = format_str.format(value)
    out = out.rstrip("0").rstrip(".") if "." in out else out
    if out == "":
        out = "0"
    return out


class HybridApp:
    """
    Aggregator with a single aggregator_cycle method, now storing enhanced data in ai_snapshots.
    on_ticker => aggregator_cycle every aggregator_interval seconds.

    aggregator_cycle =>
      1) fetch ledger => read ZUSD
      2) build aggregator data => store in ai_snapshots => Mustache => AIStrategy => GPT
      3) Create pending trades => done
    """

    def __init__(
            self,
            pairs: List[str],
            strategy: AIStrategy,
            aggregator_interval: int,
            private_ws_client: Optional[KrakenPrivateWSClient],
            kraken_api_key: str,
            kraken_api_secret: str
    ):
        self.pairs = pairs
        self.strategy = strategy
        self.aggregator_interval = aggregator_interval
        self.ws_private = private_ws_client
        self.kraken_api_key = kraken_api_key
        self.kraken_api_secret = kraken_api_secret

        # manager is optional for rest calls
        self.manager = None

        # track live prices to skip aggregator if zero
        self.latest_prices: Dict[str, float] = {p: 0.0 for p in pairs}
        self.last_agg_ts = 0

    def on_ticker(self, pair: str, last_price: float):
        """
        Called by public feed on each new ticker => store in memory,
        then check aggregator_interval => aggregator_cycle_all_coins.
        """
        self.latest_prices[pair] = last_price
        now = time.time()
        interval = ConfigLoader.get_value("trade_interval_seconds", self.aggregator_interval)
        if (now - self.last_agg_ts) >= interval:
            self.aggregator_cycle()
            self.last_agg_ts = now

    def _compute_volatility(self, pair: str, price_changes: List[float]) -> float:
        """
        Compute volatility as the standard deviation of price_changes.
        If insufficient data (< 2 points), fetch from lunarcrush_data.
        """
        if len(price_changes) >= 2:
            try:
                return statistics.stdev(price_changes)
            except statistics.StatisticsError:
                logger.warning(f"[Aggregator] Volatility calculation failed for {pair}, falling back to LunarCrush.")

        # Fallback to LunarCrush
        symbol = db_lookup.get_formatted_name_from_pair_name(pair)
        conn = sqlite3.connect(DB_FILE)
        try:
            c = conn.cursor()
            c.execute("""
                SELECT volatility
                FROM lunarcrush_data
                WHERE UPPER(symbol) = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (symbol.upper(),))
            row = c.fetchone()
            return float(row[0]) if row and row[0] is not None else 0.0
        except Exception as e:
            logger.exception(f"[Aggregator] Error fetching LunarCrush volatility for {pair}: {e}")
            return 0.0
        finally:
            conn.close()

    def _should_include_in_prompt(self, pair: str) -> bool:
        import db_lookup
        """
        Determine if the pair should be included in the AI prompt based on:
        1. Having an active position, OR
        2. Having positive market sentiment.
        """
        return db_lookup.has_active_position(pair) or (self._get_market_sentiment(pair) == "YES")

    def _get_market_sentiment(self, pair: str) -> str:
        """
        Fetch the latest sentiment from lunarcrush_data and classify market condition.
        Returns "YES" if sentiment > 65, "NO" if < 35, "MIXED" otherwise.
        """
        symbol = db_lookup.get_formatted_name_from_pair_name(pair)
        conn = sqlite3.connect(DB_FILE)
        try:
            c = conn.cursor()
            c.execute("""
                SELECT sentiment
                FROM lunarcrush_data
                WHERE UPPER(symbol) = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (symbol.upper(),))
            row = c.fetchone()
            sentiment = float(row[0]) if row and row[0] is not None else 50.0  # Default to neutral
            if sentiment > 75:
                return "YES"
            elif sentiment < 35:
                return "NO"
            else:
                return "MIXED"
        except Exception as e:
            logger.exception(f"[Aggregator] Error fetching sentiment for {pair}: {e}")
            return "MIXED"
        finally:
            conn.close()

    def _compute_risk_estimate(self, pair: str, gpt_confidence: float) -> float:
        """
        Compute risk estimate as a weighted average: 70% GPT confidence, 30% historical trade success rate.
        """
        # Fetch historical success rate from trades table
        conn = sqlite3.connect(DB_FILE)
        try:
            c = conn.cursor()
            c.execute("""
                SELECT COUNT(*) as wins, (SELECT COUNT(*) FROM trades WHERE pair = ? AND realized_pnl IS NOT NULL) as total
                FROM trades
                WHERE pair = ? AND realized_pnl > 0
            """, (pair, pair))
            row = c.fetchone()
            wins, total = row if row else (0, 0)
            success_rate = wins / total if total > 0 else 0.5  # Default to 50% if no data
        except Exception as e:
            logger.exception(f"[Aggregator] Error computing success rate for {pair}: {e}")
            success_rate = 0.5
        finally:
            conn.close()

        # Weighted average
        return (0.7 * gpt_confidence) + (0.3 * success_rate)

    def aggregator_cycle(self):
        """
        Single aggregator cycle => fetch ledger => build aggregator data => store in ai_snapshots => Mustache => AIStrategy => GPT => trades.

        Enhanced Steps:
          1) Update the local ledger DB by calling fetch_and_store_kraken_ledger(...).
          2) Fetch the latest trade balances from Kraken => convert base asset keys => determine USD.
          3) Build a raw aggregator list with numeric data => store each in ai_snapshots with volatility, sentiment, and risk.
          4) Render the Mustache template with the aggregator list.
          5) Pass the rendered prompt to AIStrategy => GPT => parse => produce decisions => extract confidence.
          6) If valid trades come back, create pending trades + possibly place real orders.
        """
        # 1) Update LunarCrush data
        self._update_lunarcrush_data()

        logger.info("[Aggregator] Starting aggregator_cycle => checking ledger for ZUSD balance...")

        # 2) Refresh ledger => read ZUSD balance
        fetch_and_store_kraken_ledger(
            api_key=self.kraken_api_key,
            api_secret=self.kraken_api_secret,
            asset="all",
            ledger_type="all",
            db_path=db.DB_FILE
        )
        trade_balances = self.manager.fetch_balance()
        current_usd_balance = get_latest_zusd_balance(db_path=db.DB_FILE)

        # Convert Kraken base-asset keys to wsname
        trade_balances_ws = convert_base_asset_keys_to_wsname(trade_balances)
        trade_balances_ws["USD"] = current_usd_balance

        # 3) Filter coins based on criteria: active position OR positive sentiment
        filtered_pairs = [pair for pair in self.pairs if self._should_include_in_prompt(pair)]

        # 3) Build aggregator items and store in ai_snapshots
        aggregator_list = []
        zero_count = 0
        quantity = ConfigLoader.get_value("quantity_of_price_history", 50)

        # Risk parameters from config
        aggregator_interval_secs = ConfigLoader.get_value("trade_interval_seconds", 3600)
        stop_loss_pct = ConfigLoader.get_value("stop_loss_percent", 0.04)
        take_profit_pct = ConfigLoader.get_value("take_profit_percent", 0.01)
        daily_drawdown_limit = self.strategy.max_daily_drawdown if self.strategy else -0.02

        with ai_lock:
            for pair in filtered_pairs:
                last_price = self.latest_prices.get(pair, 0.0)
                if last_price == 0.0:
                    zero_count += 1

                # Fetch price history
                minute_data = db.fetch_minute_spaced_prices(
                    pair=pair,
                    db_path=db.DB_FILE,
                    num_points=quantity + 1
                )
                from technical_indicators import compute_technical_indicators
                indicators = compute_technical_indicators(minute_data)

                # Compute price changes
                changes = []
                if len(minute_data) >= 2:
                    changes_raw = compute_minute_percent_changes(minute_data)
                    changes = [round(chg, 3) for chg in changes_raw]

                # Retrieve minimum purchase constraints
                min_qty = format_no_sci(db_lookup.get_ordermin(pair))
                min_cost = format_no_sci(db_lookup.get_minimum_cost_in_usd(pair))

                coin_volatility = self._compute_volatility(pair, changes)
                is_market_bullish = self._get_market_sentiment(pair)
                # Add raw data to aggregator_list
                aggregator_item = {
                    "pair": pair,
                    "last_price": last_price,
                    "changes": changes,
                    "min_qty": min_qty,
                    "min_cost": min_cost,
                    "coin_volatility": coin_volatility,
                    "is_market_bullish": is_market_bullish
                }
                aggregator_list.append(aggregator_item)

                # Enhanced metrics
                coin_volatility = self._compute_volatility(pair, changes)
                is_market_bullish = self._get_market_sentiment(pair)
                notes = f"Aggregator cycle data for {pair}"

                # Store in ai_snapshots (risk_estimate will be updated post-GPT)
                store_ai_snapshot(
                    pair=pair,
                    price_changes=changes,
                    last_price=last_price,
                    aggregator_interval_secs=aggregator_interval_secs,
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct,
                    daily_drawdown_limit=daily_drawdown_limit,
                    coin_volatility=coin_volatility,
                    is_market_bullish=is_market_bullish,
                    risk_estimate=0.0,
                    lunarcrush_data=self._fetch_lunarcrush_snapshot(pair),
                    sma_10=indicators["sma_10"],
                    ema_10=indicators["ema_10"],
                    rsi_14=indicators["rsi_14"],
                    macd=indicators["macd"],
                    macd_signal=indicators["macd_signal"],
                    bollinger_upper=indicators["bollinger_upper"],
                    bollinger_lower=indicators["bollinger_lower"],
                    notes=notes,
                    db_path=DB_FILE
                )

            # If all pairs have zero price, skip
            if zero_count == len(self.pairs):
                logger.info("[Aggregator] All pairs last_price=0 => skipping aggregator_cycle.")
                return

            # 4) Prepare Mustache context
            template_name = ConfigLoader.get_value("prompt_template_name", "aggregator_simple_prompt.mustache")
            builder = PromptBuilder(template_dir="templates")

            context = {
                "current_usd_balance": f"{current_usd_balance:.4f}",
                "trade_balances": json.dumps(trade_balances_ws, indent=2, ensure_ascii=False),
                "aggregator_items": aggregator_list
            }

            # Render the Mustache template
            final_prompt_text = builder.render_template(template_name, context)
            logger.info("[Aggregator] Successfully rendered Mustache prompt => sending to AIStrategy/GPT.")

            # 5) Call AIStrategy => get decisions and raw GPT output
            decisions, gpt_out = self.strategy.predict_from_prompt(
                final_prompt_text=final_prompt_text,
                current_trade_balance=trade_balances_ws,
                current_balance=current_usd_balance
            )

            # Extract GPT confidence from the returned gpt_out
            gpt_confidence = gpt_out.get("confidence", 0.5)  # Default to 0.5 if not provided
            logger.info(f"[Aggregator] GPT decisions => {decisions}, confidence => {gpt_confidence}")

            # Update ai_snapshots with risk_estimate
            for pair in self.pairs:
                risk_estimate = self._compute_risk_estimate(pair, gpt_confidence)
                conn = sqlite3.connect(DB_FILE)
                try:
                    c = conn.cursor()
                    c.execute("""
                                UPDATE ai_snapshots
                                SET risk_estimate = ?
                                WHERE pair = ? AND created_at = (SELECT MAX(created_at) FROM ai_snapshots WHERE pair = ?)
                            """, (risk_estimate, pair, pair))
                    conn.commit()
                except Exception as e:
                    logger.exception(f"[Aggregator] Error updating risk_estimate for {pair}: {e}")
                finally:
                    conn.close()

    def _fetch_lunarcrush_snapshot(self, pair: str) -> Dict[str, Any]:
        """
        Fetch the latest LunarCrush snapshot for the given pair.
        """
        symbol = db_lookup.get_formatted_name_from_pair_name(pair)
        conn = sqlite3.connect(DB_FILE)
        try:
            c = conn.cursor()
            c.execute("""
                SELECT *
                FROM lunarcrush_data
                WHERE UPPER(symbol) = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (symbol.upper(),))
            row = c.fetchone()
            if row:
                columns = [desc[0] for desc in c.description]
                return dict(zip(columns, row))
            return {}
        except Exception as e:
            logger.exception(f"[Aggregator] Error fetching LunarCrush data for {pair}: {e}")
            return {}
        finally:
            conn.close()

    def _update_lunarcrush_data(self):
        """
        Fetch and update LunarCrush data if needed.
        """
        try:
            fetcher = LunarCrushFetcher(db_file=DB_FILE)
            fetcher.fetch_snapshot_data_filtered(limit=100)
            logger.info("[Aggregator] Updated lunarcrush_data snapshot.")
        except Exception as e:
            logger.exception(f"[Aggregator] Error updating lunarcrush => {e}")


def main():
    """
    Main entry point:
      1) Setup logging + load config
      2) init DB => optionally fetch asset pairs + lunarcrush data
      3) create risk_manager => AIStrategy => aggregator => public WS => run
      4) On interrupt => graceful shutdown
    """
    setup_logging()
    load_dotenv()
    import db_lookup

    # 1) Load config
    ENABLE_TRAINING = ConfigLoader.get_value("enable_training", True)
    RISK_INTERVAL = ConfigLoader.get_value("risk_manager_interval_seconds", 60)
    ENABLE_GPT = ConfigLoader.get_value("enable_gpt_integration", True)
    OPENAI_MODEL = ConfigLoader.get_value("openai_model", "o1-mini")
    TRADED_PAIRS = ConfigLoader.get_traded_pairs()
    AGG_INTERVAL = ConfigLoader.get_value("trade_interval_seconds", 300)
    PRE_POPULATE_DB = ConfigLoader.get_value("pre_populate_db_tables", True)
    PRE_POPULATE_LUNARCRUSH_DATA = ConfigLoader.get_value("pre_populate_lunarcrush_data", True)
    RISK_CONTROLS = ConfigLoader.get_value("risk_controls", {
        "initial_spending_account": 50.0,
        "purchase_upper_limit_percent": 75.0,
        "max_position_value": 25.0
    })

    PLACE_LIVE_ORDERS = ConfigLoader.get_value("place_live_orders", False)

    KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "")
    KRAKEN_API_SECRET = os.getenv("KRAKEN_SECRET_API_KEY", "")

    # 2) init DB => includes new pending_trades, trades, ledger_entries, etc.
    init_db()

    IS_TIMESERIES_UNDERPOPULATED = db_lookup.is_table_underpopulated(
        table_name="lunarcrush_timeseries",
        db_file="trades.db",
        grouping_vars=["coin_id"],
        min_row_count=100
    )

    IS_ASSETS_UNDERPOPULATED = db_lookup.is_table_underpopulated(
        table_name="kraken_asset_name_lookup",
        db_file="trades.db",
        grouping_vars=[],
        min_row_count=100
    )

    fetch_and_store_kraken_public_asset_pairs(
        info="info",
        country_code="US:MI"
    )

    # 3) Optionally fetch Kraken public asset pairs and LunarCrush data
    logger.info("[Main] Optionally fetching Kraken public asset pairs at startup...")
    rest_manager = KrakenRestManager(api_key=KRAKEN_API_KEY, api_secret=KRAKEN_API_SECRET)
    if PRE_POPULATE_DB | IS_ASSETS_UNDERPOPULATED:
        rest_manager.build_coin_name_lookup_from_db()

    logger.info("[Main] Updating local lunarcrush_data + timeseries for configured symbols...")
    try:
        fetcher = LunarCrushFetcher(db_file=DB_FILE)
        fetcher.fetch_snapshot_data_filtered(limit=100)

        coin_ids = [db_lookup.get_formatted_name_from_pair_name(pair) for pair in TRADED_PAIRS]
        if PRE_POPULATE_LUNARCRUSH_DATA | IS_TIMESERIES_UNDERPOPULATED:
            fetcher.backfill_coins(
                coin_ids=coin_ids,
                months=1,
                bucket="hour",
                interval="1w"
            )
        logger.info("[Main] Successfully updated lunarcrush_data & backfilled timeseries.")
    except Exception as e:
        logger.exception(f"[Main] Error updating lunarcrush => {e}")

    if ENABLE_TRAINING:
        from train_model import run_full_training_pipeline
        run_full_training_pipeline(
            db_path="trades.db",
            symbols_for_timeseries=db_lookup.get_symbols_for_time_series(TRADED_PAIRS)
        )

    rest_manager = KrakenRestManager(KRAKEN_API_KEY, KRAKEN_API_SECRET)

    # 4) Create risk_manager + AIStrategy
    risk_manager_db = RiskManagerDB(
        db_path=DB_FILE,
        max_position_size=20,
        max_daily_drawdown=-0.02,
        initial_spending_account=RISK_CONTROLS.get("initial_spending_account", 0.0),
        private_ws_client=None,
        place_live_orders=PLACE_LIVE_ORDERS,
        ai_lock=ai_lock,
        manager=rest_manager
    )
    risk_manager_db.initialize()
    risk_manager_db.rebuild_lots_from_ledger_entries()
    loop = asyncio.get_event_loop()
    risk_manager_task = loop.create_task(
        risk_manager_db.start_db_price_check_cycle(
            pairs=TRADED_PAIRS
        )
    )

    ai_strategy = AIStrategy(
        pairs=TRADED_PAIRS,
        use_openai=ENABLE_GPT,
        max_position_size=20,
        max_daily_drawdown=-0.02,
        risk_controls=RISK_CONTROLS,
        risk_manager=risk_manager_db,
        gpt_model=OPENAI_MODEL,
        gpt_temperature=1.0,
        gpt_max_tokens=40000,
        private_ws_client=None
    )
    logger.info(
        f"[Main] AIStrategy => multi-coin GPT => pairs={TRADED_PAIRS}, GPT={ENABLE_GPT}, place_live_orders={PLACE_LIVE_ORDERS}")

    # 5) Get private WS token
    def get_ws_token(api_key: str, api_secret: str):
        """
        Example function retrieving WS token from Kraken.
        Possibly move to a manager or keep here.
        """
        import requests
        url = "https://api.kraken.com/0/private/GetWebSocketsToken"
        path = "/0/private/GetWebSocketsToken"
        nonce = str(int(time.time() * 1000))

        data = {"nonce": nonce}
        postdata = f"nonce={nonce}"
        import hashlib
        sha256_digest = hashlib.sha256((nonce + postdata).encode("utf-8")).digest()
        message = path.encode("utf-8") + sha256_digest
        secret = base64.b64decode(api_secret)
        import hmac
        sig = hmac.new(secret, message, hashlib.sha512)
        signature = base64.b64encode(sig.digest())

        headers = {
            "API-Key": api_key,
            "API-Sign": signature.decode()
        }

        try:
            resp = requests.post(url, headers=headers, data=data, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"[WS Token] error => {e}")
            return None

    token_json = get_ws_token(KRAKEN_API_KEY, KRAKEN_API_SECRET)
    token_str = None
    if token_json and token_json.get("result") and token_json["result"].get("token"):
        token_str = token_json["result"]["token"]
        logger.info(f"[Main] Got private WS token => {token_str[:5]}...")

    # 6) Private feed
    private_client = None
    if token_str:
        def on_private_event(evt_dict):
            logger.debug(f"[PrivateWS] Event => {evt_dict}")

        private_client = KrakenPrivateWSClient(
            token=token_str,
            on_private_event_callback=on_private_event,
            risk_manager=risk_manager_db
        )
        if PLACE_LIVE_ORDERS:
            ai_strategy.private_ws_client = private_client
            risk_manager_db.private_ws_client = private_client

    # Aggregator => calls AIStrategy => multi-coin GPT
    class HybridAppAggregator(HybridApp):
        pass

    aggregator_app = HybridAppAggregator(
        pairs=TRADED_PAIRS,
        strategy=ai_strategy,
        aggregator_interval=AGG_INTERVAL,
        private_ws_client=private_client,
        kraken_api_key=KRAKEN_API_KEY,
        kraken_api_secret=KRAKEN_API_SECRET
    )

    aggregator_app.manager = rest_manager

    # 7) Create public WS => aggregator cycles
    pub_client = KrakenPublicWSClient(
        pairs=TRADED_PAIRS,
        feed_type="ticker",
        on_ticker_callback=aggregator_app.on_ticker
    )
    pub_client.risk_manager = risk_manager_db
    pub_client.kraken_balances = rest_manager.fetch_balance()
    pub_client.start()

    if private_client:
        private_client.start()

    logger.info("[Main] Aggregator approach running. Press Ctrl+C to exit.")
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        logger.info("[Main] User exit => stopping.")
        risk_manager_task.cancel()
        loop.run_until_complete(risk_manager_task)
    finally:
        pub_client.stop()
        if private_client:
            private_client.stop()
        logger.info("[Main] Aggregator halted.")


if __name__ == "__main__":
    main()