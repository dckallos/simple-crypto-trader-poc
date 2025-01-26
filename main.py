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

We store each incoming ticker/trade in 'price_history', and call an AIStrategy
for each aggregator cycle to decide whether to place trades. Orders are placed
on the private feed if we have a valid token.

Enhanced Features:
    - A method `load_lunarcrush_sentiment_history(...)` to fetch multi-row
      sentiment data for a given coin from the local DB. We insert that into
      aggregator_data for GPT-based logic in AIStrategy.
    - AIStrategy can do advanced multi-coin decisions in a single GPT call
      (if you adapt aggregator_app to gather aggregator data for all coins
       and call `predict_multi_coins`). The aggregator below still calls
      single `predict` per pair, but you can update it for the multi approach
      if desired.

Usage:
    python main.py

Environment:
    - KRAKEN_API_KEY, KRAKEN_SECRET_API_KEY for private feed
    - OPENAI_API_KEY if GPT is used in AIStrategy
    - LUNARCRUSH_API_KEY (optional, if you want real-time sentiment from
      LunarCrush or local DB references)
    - config.yaml for aggregator settings (pairs, intervals, risk_controls, etc.)

ASCII Flow:

    +-------------------+
    |  Public WS Feed   | (KrakenPublicWSClient)
    |  (ticker events)  |
    +-------------------+
              |
              v
    aggregator_app.on_tick()  -- every N secs --> aggregator_app._aggregator_cycle()
              |
              v
    +------------------------+
    |   AIStrategy decides  |
    |   (BUY / SELL / HOLD) |
    +------------------------+
              |
       if "BUY"/"SELL" + size
              |
              v
 +----------------------------------+
 |  Private WS Feed (KrakenPrivate) |
 |  => placeOrder => real trades    |
 +----------------------------------+

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

# Optional: read from a LOG_CONFIG dict to set advanced logs
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
            'level': 'INFO',
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


def load_lunarcrush_sentiment_history(symbol: str, limit: int = 3) -> str:
    """
    Loads the last 'limit' rows of sentiment (and possibly galaxy_score, alt_rank)
    from 'lunarcrush_timeseries' for the given symbol, returning a short text
    to convey changes over time.

    Example usage:
        hist_str = load_lunarcrush_sentiment_history("ETH", limit=5)
        # -> "1) ts=1693400000, sentiment=0.2, galaxy_score=60, alt_rank=200; 2) ..."

    Adjust as needed for your schema or fields.
    """
    import sqlite3
    from db import DB_FILE
    logger = logging.getLogger(__name__)

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row

    # 1) find coin_id from symbol in 'lunarcrush_data'
    coin_id = None
    try:
        c = conn.cursor()
        c.execute("""
            SELECT lunarcrush_id
            FROM lunarcrush_data
            WHERE UPPER(symbol)=?
            ORDER BY id DESC
            LIMIT 1
        """, (symbol.upper(),))
        row = c.fetchone()
        if row and row["lunarcrush_id"]:
            coin_id = str(row["lunarcrush_id"])
    except Exception as e:
        logger.exception(f"[SentHistory] error => {e}")
    finally:
        conn.close()

    if not coin_id:
        return f"No coin_id for {symbol} in DB."

    # 2) get last 'limit' rows from 'lunarcrush_timeseries'
    text_list = []
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        c = conn.cursor()
        c.execute("""
            SELECT timestamp, sentiment, galaxy_score, alt_rank
            FROM lunarcrush_timeseries
            WHERE coin_id=?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (coin_id, limit))
        rows = c.fetchall()
        if not rows:
            return f"No timeseries for coin_id={coin_id}."

        rows = rows[::-1]  # chronological
        idx=1
        for r in rows:
            ts = r["timestamp"]
            sent = r["sentiment"] or 0.0
            gal = r["galaxy_score"] or 0.0
            alt = r["alt_rank"] if r["alt_rank"] is not None else 999999
            text_list.append(f"{idx}) ts={ts}, sentiment={sent}, galaxy={gal}, alt_rank={alt}")
            idx+=1
    except Exception as e:
        logger.exception(f"[SentHistory] error => {e}")
    finally:
        conn.close()

    return "; ".join(text_list)


def get_ws_token(api_key: str, api_secret: str) -> Optional[dict]:
    """
    Retrieves a WebSockets authentication token from Kraken's REST API.
    That token is used for the private feed (wss://ws-auth.kraken.com).
    """
    url = "https://api.kraken.com/0/private/GetWebSocketsToken"
    path = "/0/private/GetWebSocketsToken"
    nonce = str(int(time.time() * 1000))

    data = {"nonce": nonce}
    postdata = f"nonce={nonce}"

    # Build signature
    import hashlib, hmac, base64
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
      1) Receives live ticker from the public WebSocket feed => aggregator_interval => aggregator cycle
      2) aggregator cycle => read local DB data + sentiment => AIStrategy => trade decisions
      3) If final_action=BUY/SELL => we call private_ws_client.send_order => real trade
    """

    def __init__(
        self,
        pairs: List[str],
        strategy: AIStrategy,
        aggregator_interval: int = 60,
        private_ws_client: Optional[KrakenPrivateWSClient] = None
    ):
        """
        :param pairs: e.g. ["ETH/USD","XBT/USD"]
        :param strategy: AIStrategy instance
        :param aggregator_interval: aggregator cycle interval (sec)
        :param private_ws_client: optional private feed for real trade placement
        """
        self.pairs = pairs
        self.strategy = strategy
        self.aggregator_interval = aggregator_interval
        self.ws_private = private_ws_client
        self.last_call_ts = {p: 0 for p in pairs}

    def on_tick(self, pair: str, last_price: float):
        """
        Called by public feed on each new ticker event. We only run aggregator
        if aggregator_interval has passed.
        """
        now = time.time()
        if (now - self.last_call_ts[pair]) >= self.aggregator_interval:
            self._aggregator_cycle(pair, last_price)
            self.last_call_ts[pair] = int(now)

    def _aggregator_cycle(self, pair: str, last_price: float):
        """
        1) Query DB for short-term price stats => ma_10 + vol
        2) Load local LunarCrush sentiment => aggregator_data["lunarcrush_sentiment"]
        3) AIStrategy => final action/size
        4) If final_action=BUY/SELL => place real order => record DB
        """
        px_stats = self._fetch_recent_price_trends(pair)
        if not px_stats:
            logger.info(f"[Aggregator] skip {pair}, insufficient price_history data.")
            return

        base_symbol = pair.split("/")[0].upper()

        # We load local sentiment history so GPT can see changes over time
        sentiment_hist = load_lunarcrush_sentiment_history(base_symbol, limit=5)

        aggregator_data = {
            "pair": pair,
            "price": last_price,
            "sentiment_history": sentiment_hist,
            **px_stats
        }
        final_action, final_size = self.strategy.predict(aggregator_data)

        if final_action in ("BUY","SELL") and final_size>0:
            side_str = "buy" if final_action=="BUY" else "sell"
            logger.info(f"[Aggregator] => place real order => {pair}, side={side_str}, vol={final_size:.6f}, px={last_price}")

            if self.ws_private:
                self.ws_private.send_order(
                    pair=pair,
                    side=side_str,
                    ordertype="market",
                    volume=final_size
                )
                record_trade_in_db(
                    final_action,
                    final_size,
                    last_price,
                    "PENDING_KRAKEN_TXID",
                    pair
                )
            else:
                logger.warning("[Aggregator] no private feed => cannot place real order.")
        else:
            logger.debug(f"[Aggregator] => no trade => action={final_action}, size={final_size:.6f}")

    def _fetch_recent_price_trends(self, pair: str) -> Dict[str, Any]:
        """
        Read last ~100 price rows from 'price_history', compute ma_10 + volatility
        Return => dict or empty if insufficient data
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
            "price_ma_10": last_row["ma_10"],
            "price_volatility": last_row["volatility"]
        }


def main():
    """
    1) Load config
    2) init_db
    3) create AIStrategy
    4) get private token
    5) aggregator => pass aggregator_data => AIStrategy => real trades
    """
    logging.config.dictConfig(LOG_CONFIG)

    CONFIG_FILE = "config.yaml"
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

    ENABLE_TRAINING = config.get("enable_training", True)
    ENABLE_GPT = config.get("enable_gpt_integration", True)
    TRADED_PAIRS = config.get("traded_pairs", [])
    AGG_INTERVAL = config.get("trade_interval_seconds", 300)

    risk_controls = config.get("risk_controls", {
        "initial_spending_account": 55.0,
        "purchase_upper_limit_percent": 50.0,
        "minimum_buy_amount": 10.0,
        "max_position_value": 20.0
    })

    load_dotenv()
    KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY","FAKE_KEY")
    KRAKEN_API_SECRET = os.getenv("KRAKEN_SECRET_API_KEY","FAKE_SECRET")

    # 1) init DB
    init_db()

    # 2) (Optional) training
    if ENABLE_TRAINING:
        logger.info("[Main] Some training step might go here, omitted for brevity.")

    # 3) AIStrategy
    ai_strategy = AIStrategy(
        pairs=TRADED_PAIRS,
        use_openai=ENABLE_GPT,
        max_position_size=3,
        stop_loss_pct=0.05,
        take_profit_pct=0.01,
        max_daily_drawdown=-0.02,
        risk_controls=risk_controls,
        gpt_model="gpt-4o-mini",
        gpt_temperature=1.0,
        gpt_max_tokens=2000
    )
    logger.info(f"[Main] AIStrategy => pairs={TRADED_PAIRS}, GPT={ENABLE_GPT}")

    # 4) get private token => private feed
    token_json = get_ws_token(KRAKEN_API_KEY, KRAKEN_API_SECRET)
    token_str = None
    if token_json and "result" in token_json and "token" in token_json["result"]:
        token_str = token_json["result"]["token"]
        logger.info(f"[Main] Got private WS token => {token_str[:10]}... truncated")
    else:
        logger.warning("[Main] No private feed token => private feed disabled.")

    # 5) aggregator app
    aggregator_app = HybridApp(
        pairs=TRADED_PAIRS,
        strategy=ai_strategy,
        aggregator_interval=AGG_INTERVAL,
        private_ws_client=None  # set after creating priv_client if we want
    )

    # build public feed for ticker
    pub_client = KrakenPublicWSClient(
        pairs=TRADED_PAIRS,
        feed_type="ticker",
        on_ticker_callback=aggregator_app.on_tick
    )

    # optional private feed
    priv_client = None
    if token_str:
        def on_private_event(evt_dict):
            logger.debug(f"[PrivateWS] => {evt_dict}")

        priv_client = KrakenPrivateWSClient(
            token=token_str,
            on_private_event_callback=on_private_event
        )
        aggregator_app.ws_private = priv_client

    # start
    pub_client.start()
    if priv_client:
        priv_client.start()

    logger.info("[Main] Running. Press Ctrl+C to exit.")
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
