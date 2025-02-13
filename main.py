#!/usr/bin/env python3
# ==============================================================================
# FILE: main.py
# ==============================================================================
"""
main.py

Demonstration of consuming both the PUBLIC and PRIVATE Kraken websockets feeds,
using the classes defined in ws_data_feed.py:

- KrakenPublicWSClient: for public data like ticker/trade
- KrakenPrivateWSClient: for private data like openOrders, placing/canceling orders

We also have a HybridApp aggregator that calls AIStrategy => predict_multi_coins(...)
periodically. However, in this updated version we explicitly populate the
lunarcrush_data and lunarcrush_timeseries tables at application startup—rather
than waiting for aggregator cycles.

Environment:
    - KRAKEN_API_KEY, KRAKEN_API_SECRET for private feed usage
    - OPENAI_API_KEY if GPT is used in AIStrategy
    - config_loader.py for aggregator settings (pairs, intervals, risk_controls, etc.)
"""

import time
import requests
import hashlib
import hmac
import base64
import logging
import logging.config
import os
import urllib
import urllib.parse
import json
from dotenv import load_dotenv
import warnings
import sqlite3
import pandas as pd
from typing import Optional, Dict, Any, List

from db import init_db, DB_FILE, record_trade_in_db
from ai_strategy import AIStrategy
from ws_data_feed import KrakenPublicWSClient, KrakenPrivateWSClient
from config_loader import ConfigLoader
from fetch_lunarcrush import LunarCrushFetcher

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

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
            'level': 'DEBUG',
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
            'level': 'DEBUG',
            'propagate': True
        },
        'urllib3': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': True
        },
    }
}


def fetch_kraken_balance(api_key: str, api_secret: str) -> Dict[str, float]:
    """
    Calls Kraken's /0/private/Balance endpoint to retrieve all cash balances
    (e.g. "ZUSD", "ZEUR", "XXBT", etc.). Returns a dict of {asset: float_balance}
    or empty if error.

    API Key Permissions Required: Funds permissions => Query

    :param api_key: e.g. "YOUR_KRAKEN_API_KEY"
    :type api_key: str

    :param api_secret: e.g. "YOUR_KRAKEN_SECRET"
    :type api_secret: str

    :return: {"ZUSD":123.45,"XXBT":0.01,...} or {}
    :rtype: dict
    """
    url = "https://api.kraken.com/0/private/Balance"
    path = "/0/private/Balance"
    nonce_val = int(time.time() * 1000)

    postdata = f"nonce={nonce_val}"
    sha256 = hashlib.sha256((str(nonce_val) + postdata).encode("utf-8")).digest()
    message = path.encode("utf-8") + sha256
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


def make_kraken_headers(api_key: str,
                        api_secret: str,
                        url_path: str,
                        data: dict) -> dict:
    """
    Builds 'API-Key' and 'API-Sign' headers for Kraken private endpoints.
    - 'data' must contain all the fields (including 'nonce').
    - We form-encode them for the signature.
    """
    import hashlib
    import hmac
    import base64

    postdata = urllib.parse.urlencode(data)
    nonce_str = data.get("nonce", "")
    message_for_sha256 = (str(nonce_str) + postdata).encode("utf-8")
    sha256_digest = hashlib.sha256(message_for_sha256).digest()
    path_bytes = url_path.encode("utf-8")
    to_sign = path_bytes + sha256_digest

    secret_bytes = base64.b64decode(api_secret)
    hmac_sig = hmac.new(secret_bytes, to_sign, hashlib.sha512)
    signature = base64.b64encode(hmac_sig.digest()).decode()

    return {
        "API-Key": api_key,
        "API-Sign": signature,
        "Content-Type": "application/x-www-form-urlencoded",
    }


def fetch_and_store_kraken_public_asset_pairs(
    pair_list: Optional[List[str]] = None,
    info: str = "info",
    country_code: Optional[str] = None
) -> None:
    """
    Calls the Kraken public endpoint for retrieving asset pairs:
    GET https://api.kraken.com/0/public/AssetPairs

    Parses the JSON response, and for each entry in result, calls
    store_kraken_asset_pair_info(...) to upsert data into the 'kraken_asset_pairs' table
    defined in db.py.

    Note: This is a PUBLIC endpoint, so no API key or signature is required.

    Args:
        pair_list (List[str], optional):
            If provided, a list of pairs to filter results by, e.g. ["BTC/USD","ETH/XBT"].
            They get joined by commas for the 'pair=' query parameter.
            Default=None => no filter => returns all pairs.
        info (str, optional):
            One of ["info","leverage","fees","margin"]. Default="info".
            Controls what info fields are returned by the endpoint.
        country_code (str, optional):
            e.g. "US:TX,GB,CA" => If provided, only pairs available in these regions are returned.

    Returns:
        None
        (But side-effect: upserts into the local 'kraken_asset_pairs' table.)

    Example usage:
    >>> fetch_and_store_kraken_public_asset_pairs(
    ...     pair_list=["XBT/USD","ETH/BTC"],
    ...     info="info"
    ... )

    The function logs any errors or partial warnings,
    and will store the relevant rows in 'kraken_asset_pairs' for each pair returned.
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
    into 'ledger_entries' table. This helps with data on deposits, withdrawals, etc.

    For usage, pass the relevant timeframe or asset if needed.
    """
    url_path = "/0/private/Ledgers"
    url = "https://api.kraken.com" + url_path

    nonce_val = str(int(time.time() * 1000))
    payload = {
        "nonce": nonce_val,
        "type": ledger_type,
    }
    if asset:
        payload["asset"] = asset
    if start:
        payload["start"] = start
    if end:
        payload["end"] = end

    headers = make_kraken_headers(
        api_key=api_key,
        api_secret=api_secret,
        url_path=url_path,
        data=payload
    )
    postdata_str = urllib.parse.urlencode(payload)

    try:
        resp = requests.post(
            url,
            headers=headers,
            data=postdata_str,
            timeout=10
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logger.error(f"[Ledger] HTTP request error => {exc}")
        return

    try:
        j = resp.json()
    except json.JSONDecodeError as e:
        logger.error(f"[Ledger] Non-JSON response => {resp.text}")
        return

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
        logger.exception(f"[Ledger] Error storing ledger entries => {e}")
    finally:
        conn.close()


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
        else:
            return 0.0
    except Exception as e:
        logger.exception(f"Error retrieving latest ZUSD balance => {e}")
        return 0.0
    finally:
        conn.close()


def get_ws_token(api_key: str, api_secret: str) -> Optional[dict]:
    """
    Retrieves a WebSockets auth token from Kraken's REST API => used for private feed.
    """
    url = "https://api.kraken.com/0/private/GetWebSocketsToken"
    path = "/0/private/GetWebSocketsToken"
    nonce = str(int(time.time() * 1000))

    data = {"nonce": nonce}
    postdata = f"nonce={nonce}"
    sha256 = hashlib.sha256((nonce + postdata).encode("utf-8")).digest()
    message = path.encode("utf-8") + sha256
    secret = base64.b64decode(api_secret)
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
    except requests.exceptions.RequestException as e:
        logger.error(f"Error retrieving WS token from Kraken: {e}")
        return None


class HybridApp:
    """
    A combined aggregator that:
      1) Receives live ticker updates from public WS => store in DB => track last known price
      2) At aggregator_interval => aggregator_cycle_all_coins => gather aggregator data for all pairs
         => AIStrategy => single GPT call => produce decisions
      3) Places trades if GPT suggests them => creating pending trades in the
         'pending_trades' table. Then the private feed finalizes them or rejects them.

    We no longer backfill or update lunarcrush_timeseries in aggregator cycles:
    that now happens explicitly at app startup (see main).
    """

    def __init__(
        self,
        pairs: List[str],
        strategy: AIStrategy,
        aggregator_interval: int = 300,
        private_ws_client: Optional[KrakenPrivateWSClient] = None,
        kraken_api_key: str = "",
        kraken_api_secret: str = ""
    ):
        """
        :param pairs: e.g. ["ETH/USD","XBT/USD"] => tracked coins
        :param strategy: AIStrategy instance
        :param aggregator_interval: seconds between aggregator cycles
        :param private_ws_client: optional => private feed for real trades
        :param kraken_api_key, kraken_api_secret: used for ledger or other private calls
        """
        self.pairs = pairs
        self.strategy = strategy
        self.aggregator_interval = aggregator_interval
        self.ws_private = private_ws_client
        self.kraken_api_key = kraken_api_key
        self.kraken_api_secret = kraken_api_secret

        self.latest_prices: Dict[str, float] = {}
        for p in pairs:
            self.latest_prices[p] = 0.0

        self.last_agg_ts = 0

    def on_ticker(self, pair: str, last_price: float):
        """
        Called by public feed on each new ticker => store in memory,
        then check aggregator_interval => aggregator_cycle_all_coins.
        """
        self.latest_prices[pair] = last_price
        now = time.time()
        if (now - self.last_agg_ts) >= self.aggregator_interval:
            self.aggregator_cycle_all_coins()
            self.last_agg_ts = now

    def aggregator_cycle_all_coins(self):
        """
        The aggregator cycle:
         1) fetch ledger => store => read ZUSD balance
         2) build aggregator_list => AIStrategy => multi-coin GPT
         3) GPT => create pending trades if needed
        (We have removed time-series population from here.)
        """
        zero_count = sum(1 for p in self.pairs if self.latest_prices[p] == 0.0)
        if zero_count > 0:
            logger.info(f"Skipping aggregator because {zero_count} pairs have price=0.0.")
            return

        # 1) Update LunarCrush data
        self.update_lunarcrush_data()

        logger.info("[Aggregator] aggregator_cycle_all_coins => Checking ledger for ZUSD balance...")
        fetch_and_store_kraken_ledger(
            api_key=self.kraken_api_key,
            api_secret=self.kraken_api_secret,
            asset="all",
            ledger_type="all",
            db_path=DB_FILE
        )
        trade_balances = fetch_kraken_balance(
            api_key=self.kraken_api_key,
            api_secret=self.kraken_api_secret
        )
        current_usd_balance = get_latest_zusd_balance(db_path=DB_FILE)
        logger.info(f'[Aggregator] Current ZUSD balance: {current_usd_balance}')

        aggregator_list: List[Dict[str, Any]] = []
        for pair in self.pairs:
            aggregator_list.append(self._build_aggregator_for_pair(pair))

        # Example trade_history + open_positions usage for AIStrategy => multi-coin approach
        trade_history = self._build_global_trade_history(limit=10)
        open_positions_txt = self._build_open_positions_list()

        decisions = self.strategy.predict_multi_coins(
            input_aggregator_list=aggregator_list,
            trade_history=trade_history,
            max_trades=5,
            input_open_positions=open_positions_txt,
            current_balance=current_usd_balance,
            current_trade_balance=trade_balances
        )
        logger.info("[Aggregator] GPT decisions => %s", decisions)

    def update_lunarcrush_data(self):
        """
        Calls LunarCrushFetcher to fetch & store snapshot data, or partial time-series, etc.
        Adjust as needed based on your actual fetch logic (snapshot, timeseries, etc.)
        """
        try:
            fetcher = LunarCrushFetcher(db_file=DB_FILE)
            fetcher.fetch_snapshot_data_filtered(limit=100)
            ## snapshot data for top 100 coins or filtered
            logger.info("[Aggregator] Successfully updated lunarcrush_data from LunarCrush API.")
        except Exception as e:
            logger.exception(f"[Aggregator] Error updating lunarcrush_data => {e}")

    def _build_aggregator_for_pair(self, pair: str) -> Dict[str, Any]:
        """
        Loads the latest aggregator_summaries row for the given pair
        and fetches a recent price from lunarcrush_data (instead of timeseries).
        Returns a dict with 'pair', 'price', and 'aggregator_data' describing
        aggregator fields like galaxy_score, alt_rank, sentiment, etc.

        Example aggregator_summaries schema (adapt as needed):
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            timestamp INTEGER,
            price_bucket TEXT,
            galaxy_score REAL,
            galaxy_score_previous REAL,
            alt_rank REAL,
            alt_rank_previous REAL,
            market_dominance REAL,
            dominance_bucket TEXT,
            sentiment_label TEXT

        Example lunarcrush_data schema (for the latest price):
            id INTEGER PRIMARY KEY,
            timestamp INTEGER,
            symbol TEXT,
            price REAL,
            ...
        """
        symbol = pair.split("/")[0].upper()
        last_price = 0.0
        aggregator_text = "No aggregator data found"

        conn = sqlite3.connect(DB_FILE)
        try:
            c = conn.cursor()

            # aggregator_summaries => aggregator fields
            c.execute("""
                SELECT
                    price_bucket,
                    galaxy_score,
                    galaxy_score_previous,
                    alt_rank,
                    alt_rank_previous,
                    market_dominance,
                    dominance_bucket,
                    sentiment_label
                FROM aggregator_summaries
                WHERE UPPER(symbol)=?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (symbol,))
            row_summ = c.fetchone()

            # lunarcrush_data => the latest price
            c.execute("""
                SELECT 
                    price,
                    sentiment,
                    volatility,
                    volume_24h,
                    percent_change_1h,
                    percent_change_24h,
                    percent_change_7d,
                    percent_change_30d
                FROM lunarcrush_data
                WHERE UPPER(symbol)=?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (symbol,))
            row_price = c.fetchone()
            if row_price:
                last_price = float(row_price.get("price", 0.0))
                sentiment = row_price.get("sentiment", "Not Available")
                volatility = row_price.get("volatility", "Not Available")
                volume_24h = row_price.get("volume_24h", "Not Available")
                pct_1h = row_price.get("percent_change_1h", "Not Available")
                pct_24h = row_price.get("percent_change_24h", "Not Available")
                pct_7d = row_price.get("percent_change_7d", "Not Available")
                pct_30d = row_price.get("percent_change_30d", "Not Available")

                if row_summ:
                    (
                        price_bucket,
                        galaxy_score,
                        galaxy_score_previous,
                        alt_rank,
                        alt_rank_previous,
                        market_dominance,
                        dominance_bucket,
                        sentiment_label
                    ) = row_summ

                    aggregator_text = (
                            f"[{symbol}]\n"
                            f"\tprice = {last_price:.2f}\n"
                            f"\tprice_bucket={price_bucket}\n"
                            f"\tvolatility = {volatility}\n"
                            f"\tvolume_24h = {volume_24h:.2%}\n"
                            f"\tpercent_change_1h = {pct_1h:.2%}\n"
                            f"\tpercent_change_24h = {pct_24h:.2%}\n"
                            f"\tpercent_change_7d = {pct_7d:.2%}\n"
                            f"\tpercent_change_30d = {pct_30d:.2%}\n"
                            f"\tgalaxy_score={galaxy_score} (previous_value = {galaxy_score_previous})\n"
                            f"\talt_rank={alt_rank} (previous_value = {alt_rank_previous})\n"
                            f"\tmarket_dominance = {market_dominance:.2%} => {str(dominance_bucket).upper()}\n"
                            f"\tsocial_sentiment = {sentiment}\n"
                            f"\tsentiment_label = {str(sentiment_label).upper()}\n"
                    )
            else:
                aggregator_text = (
                    f"No aggregator_summaries row for symbol={symbol}; last_price={last_price:.2f}"
                )

        except Exception as e:
            logger.exception(f"[HybridApp] Error loading aggregator for {symbol}: {e}")
            aggregator_text = f"Error retrieving aggregator data: {e}"
        finally:
            conn.close()

        return {
            "pair": pair,
            "price": last_price,
            "aggregator_data": aggregator_text
        }

    def _build_global_trade_history(self, limit=10) -> List[str]:
        """
        Gather up to 'limit' trades from 'trades' => reversed => lines
        """
        lines = []
        conn = sqlite3.connect(DB_FILE)
        try:
            c = conn.cursor()
            c.execute("""
            SELECT timestamp, pair, side, quantity, price
            FROM trades
            ORDER BY id DESC
            LIMIT ?
            """, (limit,))
            rows = c.fetchall()
            if not rows:
                return []
            rows = rows[::-1]
            for r in rows:
                t, p, sd, qty, px = r
                import datetime
                dt_s = datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")
                line = f"{dt_s} {sd} {p} {qty}@{px}"
                lines.append(line)
        except Exception as e:
            logger.exception(f"[Aggregator] error building global trade history => {e}")
        finally:
            conn.close()
        return lines

    def _build_open_positions_list(self) -> List[str]:
        """
        Return textual list of open sub_positions.
        This helps AIStrategy see what's currently held.
        """
        lines = []
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        try:
            c = conn.cursor()
            c.execute("""
            SELECT pair, side, entry_price, size
            FROM sub_positions
            WHERE closed_at IS NULL
            """)
            rows = c.fetchall()
            for r in rows:
                p = r["pair"]
                s = r["side"]
                e = r["entry_price"]
                z = r["size"]
                lines.append(f"{p} {s.upper()} {z}, entry={e}")
        except Exception as e:
            logger.exception(f"[Aggregator] error building open_positions => {e}")
        finally:
            conn.close()
        return lines


def main():
    """
    Main entry point:
      1) Reads config from config_loader.py for pairs, intervals, GPT usage, etc.
      2) Creates AIStrategy => multi-coin GPT approach
      3) On startup, explicitly updates DB tables, including:
         - fetching lunarcrush snapshot data for configured symbols
         - backfilling timeseries data for those symbols
         - storing kraken asset pairs for those symbols
      4) Creates a HybridApp aggregator => calls aggregator_cycle_all_coins at intervals
      5) Subscribes to KrakenPublicWSClient => on_ticker => aggregator cycle
      6) If a valid WS token => also subscribe to private feed => place real orders
    """
    logging.config.dictConfig(LOG_CONFIG)

    ENABLE_TRAINING = ConfigLoader.get_value("enable_training", True)
    ENABLE_GPT = ConfigLoader.get_value("enable_gpt_integration", True)
    TRADED_PAIRS = ConfigLoader.get_traded_pairs()
    AGG_INTERVAL = ConfigLoader.get_value("trade_interval_seconds", 300)
    risk_controls = ConfigLoader.get_value("risk_controls", {
        "initial_spending_account": 50.0,
        "purchase_upper_limit_percent": 75.0,
        "minimum_buy_amount": 8.0,
        "max_position_value": 25.0
    })

    load_dotenv()
    KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "FAKE_KEY")
    KRAKEN_API_SECRET = os.getenv("KRAKEN_SECRET_API_KEY", "FAKE_SECRET")

    # 1) init DB => includes new pending_trades, etc.
    init_db()

    # 2) On startup => fetch Kraken public asset pairs for the configured pairs
    logger.info("[Main] Fetching Kraken public asset pairs at startup...")
    fetch_and_store_kraken_public_asset_pairs(
        info="info",
        country_code="US:MI"
    )

    # 3) On startup => update LunarCrush data *and also* backfill timeseries for configured pairs
    logger.info("[Main] Updating local lunarcrush_data + backfilling timeseries for configured symbols...")
    try:
        fetcher = LunarCrushFetcher(db_file=DB_FILE)
        # Snapshot data
        fetcher.fetch_snapshot_data_filtered(limit=100)

        # Backfill timeseries for each symbol in traded_pairs => e.g. 6 months
        coin_ids = [pair.split("/")[0].upper() for pair in TRADED_PAIRS]
        fetcher.backfill_coins(
            coin_ids=coin_ids,
            months=6,          # adjust as needed
            bucket="hour",
            interval="1w"
        )
        logger.info("[Main] Successfully updated lunarcrush_data and backfilled timeseries.")
    except Exception as e:
        logger.exception(f"[Main] Error updating or backfilling lunarcrush => {e}")

    # 4) Optional training
    if ENABLE_TRAINING:
        logger.info("[Main] Potential training step here. Omitted for brevity...")

    # 5) create AIStrategy => multi-coin GPT approach
    ai_strategy = AIStrategy(
        pairs=TRADED_PAIRS,
        use_openai=ENABLE_GPT,
        max_position_size=3,
        stop_loss_pct=0.05,
        take_profit_pct=0.01,
        max_daily_drawdown=-0.02,
        risk_controls=risk_controls,
        gpt_model="o1-mini",
        gpt_temperature=1.0,
        gpt_max_tokens=4000
    )
    logger.info(f"[Main] AIStrategy => multi-coin GPT => pairs={TRADED_PAIRS}, GPT={ENABLE_GPT}")

    # 6) get private token => build private feed if success
    token_json = get_ws_token(KRAKEN_API_KEY, KRAKEN_API_SECRET)
    token_str = None
    if token_json and "result" in token_json and "token" in token_json["result"]:
        token_str = token_json["result"]["token"]
        logger.info(f"[Main] Got private WS token => {token_str[:10]}...")

    # 7) aggregator with multi-coin approach + ledger calls
    aggregator_app = HybridApp(
        pairs=TRADED_PAIRS,
        strategy=ai_strategy,
        aggregator_interval=AGG_INTERVAL,
        private_ws_client=None,
        kraken_api_key=KRAKEN_API_KEY,
        kraken_api_secret=KRAKEN_API_SECRET
    )

    # Build public feed => pass aggregator_app.on_ticker
    pub_client = KrakenPublicWSClient(
        pairs=TRADED_PAIRS,
        feed_type="ticker",
        on_ticker_callback=aggregator_app.on_ticker
    )

    priv_client = None
    if token_str:
        def on_private_event(evt_dict):
            logger.debug(f"[PrivateWS] Event => {evt_dict}")

        priv_client = KrakenPrivateWSClient(
            token=token_str,
            on_private_event_callback=on_private_event
        )
        aggregator_app.ws_private = priv_client

    # start them
    pub_client.start()
    if priv_client:
        priv_client.start()

    logger.info("[Main] Running aggregator approach. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("[Main] user exit => stopping.")
    finally:
        pub_client.stop()
        if priv_client:
            priv_client.stop()
        logger.info("[Main] aggregator halted.")

if __name__ == "__main__":
    main()
