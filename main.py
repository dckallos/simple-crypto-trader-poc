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

We keep single-coin logic in AIStrategy for optional usage later, but this aggregator
calls the multi-coin approach by default.

Environment:
    - KRAKEN_API_KEY, KRAKEN_SECRET_API_KEY for private feed usage (balance queries & real trades)
    - OPENAI_API_KEY if GPT is used in AIStrategy
    - config.yaml for aggregator settings (pairs, intervals, risk_controls, etc.)

ASCII Flow:

    +-------------------+
    |  Public WS Feed   | (KrakenPublicWSClient)
    |  (ticker events)  |
    +-------------------+
              |
              v
    aggregator_app.on_ticker() => store in DB => aggregator_app tries aggregator_cycle_all_coins()
                  (every aggregator_interval seconds)
              |
              v
    +---------------------------------+
    | aggregator_cycle_all_coins()   |
    | => _fetch_kraken_balance()     |
    | => build aggregator data       |
    | => AIStrategy => GPT (multi)   |
    +---------------------------------+
              |
       "decisions":[
         {"pair":"ETH/USD","action":"BUY","size":0.001},
         {"pair":"XBT/USD","action":"SELL","size":0.002}
       ]
              |
              v
 +----------------------------------+
 |  Private WS Feed (KrakenPrivate) |
 |  => placeOrder => real trades    |
 +----------------------------------+

Usage:
    python main.py
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


def get_ws_token(api_key: str, api_secret: str) -> Optional[dict]:
    """
    Retrieves a WebSockets authentication token from Kraken's REST API.
    That token is used for the private feed (wss://ws-auth.kraken.com).

    :param api_key: Your Kraken API Key
    :type api_key: str

    :param api_secret: Your Kraken Secret Key
    :type api_secret: str

    :return: JSON with e.g. {"error":[],"result":{"token":"..."}}
    :rtype: dict or None if error
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


class HybridApp:
    """
    A combined aggregator that:
      1) Receives live ticker updates from public WS => store in DB => track last known price
      2) At aggregator_interval => aggregator_cycle_all_coins => gather aggregator data for ALL pairs
         => AIStrategy => single GPT call => produce decisions for each coin in one shot
      3) If final_action=BUY/SELL => place real trades on private feed.
      4) Calls fetch_kraken_balance() before aggregator logic to ensure we have funds
         info that we can optionally pass into AIStrategy if needed.
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
        :type pairs: list of str

        :param strategy: AIStrategy instance
        :type strategy: AIStrategy

        :param aggregator_interval: time in seconds between aggregator cycles, default=300
        :type aggregator_interval: int

        :param private_ws_client: optional => private feed for real trades
        :type private_ws_client: KrakenPrivateWSClient or None

        :param kraken_api_key: for fetch_kraken_balance calls
        :type kraken_api_key: str

        :param kraken_api_secret: for fetch_kraken_balance calls
        :type kraken_api_secret: str
        """
        self.pairs = pairs
        self.strategy = strategy
        self.aggregator_interval = aggregator_interval
        self.ws_private = private_ws_client
        self.kraken_api_key = kraken_api_key
        self.kraken_api_secret = kraken_api_secret

        # Track last ticker price
        self.latest_prices: Dict[str, float] = {}
        for p in pairs:
            self.latest_prices[p] = 0.0

        self.last_agg_ts = 0

    def on_ticker(self, pair: str, last_price: float):
        """
        Called by public feed on each new ticker. We store in memory, then
        check if aggregator_interval is up => aggregator_cycle_all_coins().

        :param pair: e.g. "ETH/USD"
        :type pair: str

        :param last_price: The most recent last traded price
        :type last_price: float
        """
        self.latest_prices[pair] = last_price
        now = time.time()
        if (now - self.last_agg_ts) >= self.aggregator_interval:
            self.aggregator_cycle_all_coins()
            self.last_agg_ts = now

    def aggregator_cycle_all_coins(self):
        """
        Steps:
          1) fetch kraken account balance => e.g. {'ZUSD':..., 'XXBT':..., ...}
          2) gather aggregator data for all pairs => aggregator_list
          3) gather global trade_history => open_positions => pass with aggregator_list to AIStrategy => single GPT call
          4) place trades if final action=BUY/SELL
        """
        logger.info("[Aggregator] aggregator_cycle_all_coins => Checking Kraken balance first...")

        # 1) fetch balances
        balance_info = {}
        if self.kraken_api_key and self.kraken_api_secret:
            balance_info = fetch_kraken_balance(self.kraken_api_key, self.kraken_api_secret)
            logger.info(f"[Aggregator] Current Kraken Balances => {balance_info}")
        else:
            logger.warning("[Aggregator] No kraken_api_key/secret => skipping balance check.")

        # 2) aggregator_list
        aggregator_list: List[Dict[str, Any]] = []
        for pair in self.pairs:
            aggregator_list.append(self._build_aggregator_for_pair(pair))

        # 3) build trade history + open positions
        trade_history = self._build_global_trade_history(limit=10)
        open_positions_txt = self._build_open_positions_list()

        logger.info("[Aggregator] aggregator_list => %s", aggregator_list)

        # We store the balance info in the AIStrategy as an attribute if needed
        # (Simplistic approach; you could do more refined usage.)
        self.strategy.current_account_balance = balance_info

        # AIStrategy => multi coin approach => single GPT call
        decisions = self.strategy.predict_multi_coins(
            aggregator_list=aggregator_list,
            trade_history=trade_history,
            max_trades=2,
            open_positions=open_positions_txt
        )
        logger.info("[Aggregator] GPT decisions => %s", decisions)

        # 4) place trades if needed
        for p, (action, size) in decisions.items():
            if action in ("BUY", "SELL") and size > 0:
                side_str = "buy" if action == "BUY" else "sell"
                px = self._lookup_price_for_pair(p, aggregator_list)
                logger.info(f"[Aggregator] => place real order => {p}, side={side_str}, vol={size:.6f}, px={px}")
                if self.ws_private and px > 0:
                    self.ws_private.send_order(
                        pair=p,
                        side=side_str,
                        ordertype="market",
                        volume=size
                    )
                else:
                    logger.warning("[Aggregator] no private feed or no valid price => cannot place real order.")

    def _build_aggregator_for_pair(self, pair: str) -> Dict[str, Any]:
        """
        Return a single aggregator item for 'pair'.

        :param pair: e.g. "ETH/USD"
        :type pair: str

        :return: {"pair":"ETH/USD","price":1850.0,"aggregator_data":"..."}
        :rtype: dict
        """
        last_price = self.latest_prices.get(pair, 0.0)
        px_stats = self._fetch_recent_price_trends(pair)
        vol = px_stats.get("price_volatility", 0.0)
        ma10 = px_stats.get("price_ma_10", 0.0)

        base_symbol = pair.split("/")[0].upper()
        sentiment_text = self._load_sentiment_for_symbol(base_symbol)
        aggregator_str = f"price={last_price}, vol={vol}, ma10={ma10}, {sentiment_text}"

        return {
            "pair": pair,
            "price": last_price,
            "aggregator_data": aggregator_str
        }

    def _fetch_recent_price_trends(self, pair: str) -> Dict[str, Any]:
        """
        Query the local DB for recent price_history data for 'pair'.
        Build a small dictionary with volatility, moving average, etc.

        :param pair: e.g. "ETH/USD"
        :type pair: str

        :return: e.g. {"price_volatility":0.02,"price_ma_10":1805.6}
        :rtype: dict
        """
        try:
            conn = sqlite3.connect(DB_FILE)
            q = """
                SELECT last_price
                FROM price_history
                WHERE pair=?
                ORDER BY timestamp DESC
                LIMIT 100
            """
            df = pd.read_sql_query(q, conn, params=[pair])
        except Exception as e:
            logger.exception(f"[Aggregator] DB error => {e}")
            return {}
        finally:
            conn.close()

        if df.empty:
            return {}
        df = df.iloc[::-1].reset_index(drop=True)
        df["ma_10"] = df["last_price"].rolling(10).mean()
        df["volatility"] = df["last_price"].pct_change().std()

        last_row = df.iloc[-1]
        return {
            "price_ma_10": float(last_row["ma_10"]) if not pd.isnull(last_row["ma_10"]) else 0.0,
            "price_volatility": float(last_row["volatility"]) if not pd.isnull(last_row["volatility"]) else 0.0
        }

    def _load_sentiment_for_symbol(self, symbol: str) -> str:
        """
        Return a short string describing the sentiment/gscore/alt_rank for 'symbol'
        from the latest lunarcrush_timeseries row.

        :param symbol: e.g. "ETH"
        :type symbol: str

        :return: e.g. "sentiment=..., galaxy=..., alt_rank=..."
        :rtype: str
        """
        out = "sentiment=0.0, galaxy=0.0, alt_rank=999999"
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        try:
            c = conn.cursor()
            c.execute(
                "SELECT lunarcrush_id FROM lunarcrush_data WHERE UPPER(symbol)=? ORDER BY id DESC LIMIT 1",
                (symbol.upper(),)
            )
            row = c.fetchone()
            if row and row["lunarcrush_id"]:
                coin_id = str(row["lunarcrush_id"])
            else:
                return out

            c.execute("""
                SELECT sentiment, galaxy_score, alt_rank
                FROM lunarcrush_timeseries
                WHERE coin_id=?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (coin_id,))
            row2 = c.fetchone()
            if row2:
                s = row2["sentiment"] if row2["sentiment"] else 0.0
                g = row2["galaxy_score"] if row2["galaxy_score"] else 0.0
                a = row2["alt_rank"] if row2["alt_rank"] is not None else 999999
                out = f"sentiment={s}, galaxy={g}, alt_rank={a}"
        except Exception as e:
            logger.exception(f"[Aggregator] error loading sentiment => {e}")
        finally:
            conn.close()
        return out

    def _lookup_price_for_pair(self, pair: str, aggregator_list: List[Dict[str, Any]]) -> float:
        """
        Find 'price' in aggregator_list for 'pair', or 0.0 if missing.

        :param pair: e.g. "ETH/USD"
        :type pair: str

        :param aggregator_list: The list created in aggregator_cycle_all_coins
        :type aggregator_list: list of dict

        :return: e.g. 1850.0
        :rtype: float
        """
        for item in aggregator_list:
            if item["pair"] == pair:
                return item.get("price", 0.0)
        return 0.0

    def _build_global_trade_history(self, limit=10) -> List[str]:
        """
        Gather up to 'limit' rows from trades table => reversed => lines.

        :param limit: default=10
        :type limit: int

        :return: e.g. ["2025-01-01 10:00 BUY ETH/USD 0.001@25000", ...]
        :rtype: list of str
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
        Return textual list of open sub_positions for GPT usage.

        :return: e.g. ["ETH/USD LONG 0.002, entry=1860.0", "XBT/USD SHORT 0.001, entry=29500.0"]
        :rtype: list of str
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
    Main entry point for demonstrating the aggregator approach:
      1) Reads config.yaml for pairs, intervals, GPT usage, etc.
      2) Creates AIStrategy => multi-coin GPT approach
      3) Creates a HybridApp aggregator => calls aggregator_cycle_all_coins at intervals
      4) Subscribes to KrakenPublicWSClient => on_ticker => aggregator cycle
      5) If a valid WS token is found => also subscribe to private feed => place real trades
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
        "initial_spending_account": 30.0,
        "purchase_upper_limit_percent": 50.0,
        "minimum_buy_amount": 10.0,
        "max_position_value": 25.0
    })

    load_dotenv()
    KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "FAKE_KEY")
    KRAKEN_API_SECRET = os.getenv("KRAKEN_SECRET_API_KEY", "FAKE_SECRET")

    # 1) init DB
    init_db()

    # 2) Optional training
    if ENABLE_TRAINING:
        logger.info("[Main] Potential training step here. Omitted for brevity.")

    # 3) create AIStrategy => multi-coin GPT approach (which references our updated GPTManager)
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
        gpt_max_tokens=25000
    )
    logger.info(f"[Main] AIStrategy => multi-coin GPT => pairs={TRADED_PAIRS}, GPT={ENABLE_GPT}")

    # 4) get private token => we might do balance calls, etc. if we want
    token_json = get_ws_token(KRAKEN_API_KEY, KRAKEN_API_SECRET)
    token_str = None
    if token_json and "result" in token_json and "token" in token_json["result"]:
        token_str = token_json["result"]["token"]
        logger.info(f"[Main] Got private WS token => {token_str[:10]}... (truncated)")
    else:
        logger.warning("[Main] Could not retrieve token => private feed disabled")

    # 5) aggregator with multi-coin approach + ability to fetch balance
    aggregator_app = HybridApp(
        pairs=TRADED_PAIRS,
        strategy=ai_strategy,
        aggregator_interval=AGG_INTERVAL,
        private_ws_client=None,
        kraken_api_key=KRAKEN_API_KEY,
        kraken_api_secret=KRAKEN_API_SECRET
    )

    # Build public feed => pass aggregator_app.on_ticker => aggregator interval => aggregator_cycle_all_coins
    pub_client = KrakenPublicWSClient(
        pairs=TRADED_PAIRS,
        feed_type="ticker",
        on_ticker_callback=aggregator_app.on_ticker
    )

    # Build private feed if token
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

    logger.info("[Main] Running multi-coin aggregator approach with balance checks. Press Ctrl+C to exit.")
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
