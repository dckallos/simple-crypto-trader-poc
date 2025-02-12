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

At a fixed interval, we gather aggregator data for ALL tracked coin pairs (including
recent price, sentiment, volatility, etc.) in a single pass. Before doing so, we also
call the Kraken "Balance" endpoint to check available funds. Then we pass everything
to AIStrategy => predict_multi_coins(...) for a single GPT invocation. GPT can
output buy/sell/hold decisions for each coin in one shot, including swapping trades
if it sees a better opportunity.

We keep single-coin logic in AIStrategy for optional usage, but the aggregator
calls the multi-coin approach by default.

**UPDATED**:
 - We have removed references to `'pending'` or `'part_filled'` states in the `trades` table.
 - We rely on `init_db()` to create the new `pending_trades` table and the expanded
   `kraken_asset_pairs` table, but we skip scheduling any fetch job for /0/public/AssetPairs.
 - Everything else remains consistent with the older version, aside from those
   references to old 'pending' statuses in `trades`.

Environment:
    - KRAKEN_API_KEY, KRAKEN_API_SECRET for private feed usage
    - OPENAI_API_KEY if GPT is used in AIStrategy
    - config.yaml for aggregator settings (pairs, intervals, risk_controls, etc.)
"""

import time
import requests
import hashlib
import hmac
import base64
import logging
import logging.config
import yaml
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


def make_kraken_headers(api_key: str,
                        api_secret: str,
                        url_path: str,
                        data: dict) -> dict:
    """
    Builds 'API-Key' and 'API-Sign' headers for Kraken private endpoints.
    - 'data' must contain all the fields (including 'nonce').
    - We form-encode them for the signature.
    """
    import hashlib, hmac, base64

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
      3) Place trades if GPT suggests them => But we do so by creating pending trades in the new
         'pending_trades' table. Then the private feed actually finalizes them or rejects them.

    This updated version removes references to 'pending' trades in 'trades'.
    Instead, we store ephemeral states in 'pending_trades' only.
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
        1) fetch ledger => store => read ZUSD balance
        2) gather aggregator data => aggregator_list => pass to AIStrategy => multi-coin
        3) place trades if needed => though AIStrategy already calls create_pending_trade
        """
        zero_count = sum(1 for p in self.pairs if self.latest_prices[p] == 0.0)
        if zero_count > 0:
            logger.info(f"Skipping aggregator because {zero_count} pairs have price=0.0.")
            return

        logger.info("[Aggregator] aggregator_cycle_all_coins => Checking ledger for ZUSD balance...")

        fetch_and_store_kraken_ledger(
            api_key=self.kraken_api_key,
            api_secret=self.kraken_api_secret,
            asset="all",
            ledger_type="all",
            db_path=DB_FILE
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
            current_balance=current_usd_balance
        )
        logger.info("[Aggregator] GPT decisions => %s", decisions)

    def _build_aggregator_for_pair(self, pair: str) -> Dict[str, Any]:
        """
        Return aggregator snippet for 'pair'. We might read aggregator_summaries or
        other data. For demonstration, we store minimal data:
        """
        last_price = self.latest_prices.get(pair, 0.0)
        # Potentially do more advanced aggregator logic here
        aggregator_str = f"price={last_price}, somePlaceholder=..."
        return {
            "pair": pair,
            "price": last_price,
            "aggregator_data": aggregator_str
        }

    def _build_global_trade_history(self, limit=10) -> List[str]:
        """
        Gather up to 'limit' trades from 'trades' table => reversed => lines
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
        This can help AIStrategy see what's currently held.
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
      1) Reads config.yaml for pairs, intervals, GPT usage, etc.
      2) Creates AIStrategy => multi-coin GPT approach
      3) Creates a HybridApp aggregator => calls aggregator_cycle_all_coins at intervals
      4) Subscribes to KrakenPublicWSClient => on_ticker => aggregator cycle
      5) If a valid WS token => also subscribe to private feed => place real orders
    """
    logging.config.dictConfig(LOG_CONFIG)

    CONFIG_FILE = "config.yaml"
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ENABLE_TRAINING = config.get("enable_training", True)
    ENABLE_GPT = config.get("enable_gpt_integration", True)
    TRADED_PAIRS = config.get("traded_pairs", [])
    AGG_INTERVAL = config.get("trade_interval_seconds", 300)

    risk_controls = config.get("risk_controls", {
        "initial_spending_account": 50.0,
        "purchase_upper_limit_percent": 75.0,
        "minimum_buy_amount": 8.0,
        "max_position_value": 25.0
    })

    load_dotenv()
    KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "FAKE_KEY")
    KRAKEN_API_SECRET = os.getenv("KRAKEN_SECRET_API_KEY", "FAKE_SECRET")

    # 1) init DB
    init_db()

    # 2) Optional training
    if ENABLE_TRAINING:
        logger.info("[Main] Potential training step here. Omitted for brevity...")

    # 3) create AIStrategy => multi-coin GPT approach
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
        gpt_max_tokens=5000
    )
    logger.info(f"[Main] AIStrategy => multi-coin GPT => pairs={TRADED_PAIRS}, GPT={ENABLE_GPT}")

    # 4) get private token => build private feed if success
    token_json = get_ws_token(KRAKEN_API_KEY, KRAKEN_API_SECRET)
    token_str = None
    if token_json and "result" in token_json and "token" in token_json["result"]:
        token_str = token_json["result"]["token"]
        logger.info(f"[Main] Got private WS token => {token_str[:10]}...")

    # 5) aggregator with multi-coin approach + ability to fetch ledger
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
