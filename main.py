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

Added Enhancement:
    - A method to fetch LunarCrush sentiment data for a given symbol,
      so we can pass that to AIStrategy (and ultimately to GPT).

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

Usage:
    python main.py

Environment:
    - KRAKEN_API_KEY
    - KRAKEN_SECRET_API_KEY
      (Used by get_ws_token to get a token for private WS feed)
    - OPENAI_API_KEY (if using GPT logic in AIStrategy)
    - LUNARCRUSH_API_KEY (optional, if you want real-time sentiment from LunarCrush)

See config.yaml for:
  - enable_training
  - traded_pairs
  - trade_interval_seconds
  - risk_controls
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
from typing import Optional, Dict, Any

from db import init_db, DB_FILE, record_trade_in_db
from ai_strategy import AIStrategy  # assume your AIStrategy is in ai_strategy.py
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
    Loads the last 'limit' rows of sentiment (or other metrics) from 'lunarcrush_timeseries'
    for the given symbol, returning a short text to convey changes over time.

    Example usage:
        hist_str = load_lunarcrush_sentiment_history("ETH", limit=5)
        # -> "1) ts=1693400000, sentiment=0.2; 2) ts=1693403600, sentiment=0.22; ..."

    Adjust as needed for your schema or fields (like galaxy_score, alt_rank, etc.).
    """
    import sqlite3
    from db import DB_FILE
    import logging
    logger = logging.getLogger(__name__)

    # Step 1) find coin_id from symbol in 'lunarcrush_data'
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
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
        logger.exception(f"[SentimentHistory] error finding coin_id => {e}")
    finally:
        conn.close()

    if not coin_id:
        return f"No coin_id for {symbol} in DB."

    # Step 2) retrieve last 'limit' rows from 'lunarcrush_timeseries' where coin_id=?
    # We want them in descending order so we can see the most recent first, or ascending for chronological changes
    history_text = []
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
            return f"No sentiment data found for coin_id={coin_id}."

        # Step 3) Build a short text summarizing these rows
        rows = rows[::-1]  # reverse to chronological if you want oldest->newest
        idx = 1
        for r in rows:
            ts = r["timestamp"]
            sent = r["sentiment"] if r["sentiment"] else 0.0
            gscore = r["galaxy_score"] if r["galaxy_score"] else 0.0
            arank = r["alt_rank"] if r["alt_rank"] is not None else 999999
            history_text.append(
                f"{idx}) ts={ts}, sentiment={sent}, galaxy_score={gscore}, alt_rank={arank}"
            )
            idx += 1
    except Exception as e:
        logger.exception(f"[SentimentHistory] error retrieving rows => {e}")
    finally:
        conn.close()

    return "; ".join(history_text)


def get_ws_token(api_key: str, api_secret: str) -> Optional[dict]:
    """
    Retrieves a WebSockets authentication token from Kraken's REST API.
    That token is used for the private feed (wss://ws-auth.kraken.com).

    :param api_key: Your Kraken API Key string
    :param api_secret: Your Kraken API Secret string
    :return: parsed JSON response if success, or None if error
    """
    url = "https://api.kraken.com/0/private/GetWebSocketsToken"
    path = "/0/private/GetWebSocketsToken"
    nonce = str(int(time.time() * 1000))

    data = {"nonce": nonce}
    postdata = f"nonce={nonce}"

    # Build signature
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


def fetch_lunarcrush_sentiment(symbol: str) -> float:
    """
    Fetches sentiment data from LunarCrush for the given symbol, or returns 0.0 if unavailable.

    This function is a simplified placeholder. In a real scenario:
      1) You would have a LUNARCRUSH_API_KEY and make a request to the LunarCrush API.
      2) Possibly parse 'galaxy_score' or 'alt_rank' or a custom sentiment metric.
      3) Or you may read from a local DB (lunarcrush_data) if you store snapshot data.

    Example usage in aggregator cycle => aggregator_data["lunarcrush_sentiment"] = fetch_lunarcrush_sentiment("ETH")

    :param symbol: e.g. "ETH" or "BTC" or similar
    :return: float representing sentiment or 0.0 if not found
    """
    # We'll do a quick demonstration call. Replace with your real endpoint/params:
    # NOTE: This is purely illustrative. The actual LunarCrush endpoint and response may differ.
    LUNARCRUSH_API_KEY = os.getenv("LUNARCRUSH_API_KEY", "")
    if not LUNARCRUSH_API_KEY:
        logger.warning("No LUNARCRUSH_API_KEY set => returning 0.0 for sentiment.")
        return 0.0

    base_url = "https://lunarcrush.com/api4/public/coins/list/v2"
    params = {
        "key": LUNARCRUSH_API_KEY,
        "symbol": symbol.upper()
    }
    try:
        r = requests.get(base_url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        # Suppose we find the coin in data["data"] with a "galaxy_score" or something
        # We'll do a naive approach => if "data" array not empty => parse "galaxy_score"
        coins = data.get("data", [])
        if not coins:
            return 0.0
        # The first coin might match our symbol
        coin = coins[0]
        galaxy_score = coin.get("galaxy_score", 0.0)
        sentiment = float(galaxy_score) / 100.0  # e.g. scale 0..1
        logger.info(f"LunarCrush => {symbol}, galaxy_score={galaxy_score}, => sentiment={sentiment}")
        return sentiment
    except Exception as e:
        logger.exception(f"Error fetching LunarCrush for {symbol} => {e}")
        return 0.0


class HybridApp:
    """
    A combined aggregator that:
      1) Receives live ticker from public WS => every aggregator_interval, do aggregator cycle
      2) aggregator cycle => fetch recent price data => AIStrategy => decide (BUY/SELL/HOLD)
      3) If final_action=BUY/SELL => we call private_ws_client.send_order() => real trade
    """

    def __init__(
        self,
        pairs,
        strategy: AIStrategy,
        aggregator_interval=60,
        private_ws_client: Optional[KrakenPrivateWSClient] = None
    ):
        """
        :param pairs: e.g. ["ETH/USD","XBT/USD"]
        :param strategy: AIStrategy instance
        :param aggregator_interval: run aggregator cycle every N seconds per pair
        :param private_ws_client: pass your KrakenPrivateWSClient if you want real orders
        """
        self.pairs = pairs
        self.strategy = strategy
        self.aggregator_interval = aggregator_interval
        self.ws_private = private_ws_client
        self.last_call_ts = {p: 0 for p in pairs}

    def on_tick(self, pair: str, last_price: float):
        """
        Called by the public feed on every new ticker event. We only run aggregator
        if aggregator_interval has passed.
        """
        now = time.time()
        if (now - self.last_call_ts[pair]) >= self.aggregator_interval:
            self._aggregator_cycle(pair, last_price)
            self.last_call_ts[pair] = int(now)

    def _aggregator_cycle(self, pair: str, last_price: float):
        """
        1) fetch recent price_history => e.g. 100 bars
        2) fetch LunarCrush sentiment => aggregator_data["lunarcrush_sentiment"]
        3) pass aggregator_data to AIStrategy => final_action, final_size
        4) If final_action=BUY/SELL => call send_order + record in DB
        """
        px_stats = self._fetch_recent_price_trends(pair)
        if not px_stats:
            logger.info(f"[Aggregator] skip {pair}, insufficient data in price_history.")
            return

        # Suppose pair is e.g. "ETH/USD", we want "ETH" for LunarCrush:
        base_symbol = pair.split("/")[0].upper()  # "ETH", "XBT"

        # 2) fetch sentiment from LunarCrush
        lunarcrush_value = load_lunarcrush_sentiment_history(base_symbol, limit=5)

        aggregator_data = {
            "pair": pair,
            "price": last_price,
            "lunarcrush_sentiment": lunarcrush_value,
            **px_stats
        }
        # AIStrategy => final action/size
        final_action, final_size = self.strategy.predict(aggregator_data)

        if final_action in ("BUY", "SELL") and final_size > 0:
            # We have a real trade to place
            side_str = "buy" if final_action == "BUY" else "sell"
            logger.info(f"[Aggregator] => place real order => {pair}, side={side_str}, volume={final_size:.6f}, px={last_price}")

            if self.ws_private:
                # Actually send to Kraken
                self.ws_private.send_order(
                    pair=pair,
                    side=side_str,
                    ordertype="market",
                    volume=final_size
                )

                # Also record in DB so we can see it in trades table
                # We don't have the real Kraken TXID yet; we'll store a placeholder.
                record_trade_in_db(
                    final_action,
                    final_size,
                    last_price,
                    "PENDING_KRAKEN_TXID",
                    pair
                )
            else:
                logger.warning("[Aggregator] no private WS => cannot place real order.")
        else:
            logger.debug(f"[Aggregator] => no trade => action={final_action}, size={final_size:.6f}")

    def _fetch_recent_price_trends(self, pair: str) -> Dict[str, Any]:
        """
        Read last ~100 last_price from 'price_history', compute ma_10 + volatility
        Return a dict or empty if insufficient data
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
    The main runner:
      1) load config
      2) init_db
      3) create AIStrategy
      4) get private token
      5) build aggregator => pass it to a public feed => aggregator calls AIStrategy
         => if buy/sell => calls private feed => real order
    """
    logging.config.dictConfig(LOG_CONFIG)

    CONFIG_FILE = "config.yaml"
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

    ENABLE_TRAINING = config.get("enable_training", True)
    ENABLE_GPT = config.get("enable_gpt_integration", True)
    TRADED_PAIRS = config.get("traded_pairs", [])
    AGG_INTERVAL = config.get("trade_interval_seconds", 300)

    # risk controls from config
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

    # 2) (Optional) training step
    if ENABLE_TRAINING:
        logger.info("[Main] Training step would go here (omitted for brevity).")

    # 3) create AIStrategy
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

    # 4) get private token
    token_json = get_ws_token(KRAKEN_API_KEY, KRAKEN_API_SECRET)
    token_str = None
    if token_json and "result" in token_json and "token" in token_json["result"]:
        token_str = token_json["result"]["token"]
        logger.info(f"[Main] Got private WS token => {token_str[:10]}... (truncated)")
    else:
        logger.warning("[Main] Could not retrieve token => private feed disabled")

    # 5) create aggregator app
    aggregator_app = HybridApp(
        pairs=TRADED_PAIRS,
        strategy=ai_strategy,
        aggregator_interval=AGG_INTERVAL,
        private_ws_client=None  # we will set this after we create priv_client
    )

    # 6) Build public feed
    pub_client = KrakenPublicWSClient(
        pairs=TRADED_PAIRS,
        feed_type="ticker",
        on_ticker_callback=aggregator_app.on_tick,
    )

    # 7) private feed
    priv_client = None
    if token_str:
        def on_private_event(evt_dict):
            # If you want to see addOrderStatus or openOrders
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
