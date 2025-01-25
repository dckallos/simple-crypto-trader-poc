# ==============================================================================
# FILE: main.py
# ==============================================================================
"""
main.py

A mature application that:
1) Initializes the DB (init_db from db.py).
2) Optionally runs training if config says so.
3) Retrieves a Kraken WebSockets token (if desired), enabling private feed usage.
4) Starts a Kraken WebSocket feed for real-time data, storing it in price_history.
5) Uses enriched data (sentiment and price trends) to pass to AIStrategy (with GPT or fallback).
6) Skips trading pairs without sufficient data.
7) Places real orders via Kraken WebSocket when AIStrategy indicates BUY/SELL.
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

from db import init_db, DB_FILE
from ai_strategy import AIStrategy
from ws_data_feed import KrakenWSClient

logging.basicConfig(level=logging.INFO)
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


def get_ws_token(api_key, api_secret):
    """
    Retrieves a WebSockets authentication token from Kraken's REST API.
    Example response:
        {
          "error": [],
          "result": {
            "expires": 900,
            "token": "..."
          }
        }
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
    except requests.exceptions.RequestException as e:
        logger.error(f"Error retrieving WebSockets token: {e}")
        return None

    return resp.json()


def main():
    logging.config.dictConfig(LOG_CONFIG)

    CONFIG_FILE = "config.yaml"
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

    ENABLE_TRAINING = config.get("enable_training", True)
    ENABLE_GPT_INTEGRATION = config.get("enable_gpt_integration", True)
    TRADED_PAIRS = config.get("traded_pairs", [])
    AGGREGATOR_INTERVAL_SECONDS = config.get("trade_interval_seconds", 120)

    # Load risk controls from config if present
    risk_controls = config.get("risk_controls", {
        "initial_spending_account": 40.0,
        "purchase_upper_limit_percent": 1.0,
        "minimum_buy_amount": 10.0,
        "max_position_value": 20.0
    })

    load_dotenv()
    KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "FAKE_KEY")
    KRAKEN_API_SECRET = os.getenv("KRAKEN_SECRET_API_KEY", "FAKE_SECRET")

    # 1) Initialize the DB
    init_db()

    # 2) Optionally perform training/aggregator setup (skipped in this demo)
    if ENABLE_TRAINING:
        logger.info("Training step would go here. Skipping for brevity.")
    else:
        logger.info("Skipping training step per config.")

    # 3) Create the AIStrategy instance
    model_path = "trained_model.pkl" if os.path.exists("trained_model.pkl") else None
    ai_model = AIStrategy(
        pairs=TRADED_PAIRS,
        model_path=model_path,
        use_openai=ENABLE_GPT_INTEGRATION,
        max_position_size=0.001,
        stop_loss_pct=0.05,
        take_profit_pct=0.01,
        max_daily_drawdown=-0.02,
        risk_controls=risk_controls
    )
    logger.info(f"AIStrategy loaded with pairs={TRADED_PAIRS}, GPT integration={ENABLE_GPT_INTEGRATION}")

    # 4) Retrieve the Kraken WebSockets token
    token_json = get_ws_token(KRAKEN_API_KEY, KRAKEN_API_SECRET)
    token_str = None
    if token_json and "result" in token_json and "token" in token_json["result"]:
        token_str = token_json["result"]["token"]
        logger.info(f"Retrieved WebSocket token: {token_str}")
    else:
        logger.warning("Failed to retrieve WebSocket token.")

    # 5) Define the hybrid aggregator
    class HybridApp:
        def __init__(self, pairs, strategy, aggregator_interval=60, ws_client=None):
            self.pairs = pairs
            self.strategy = strategy
            self.aggregator_interval = aggregator_interval
            self.last_call_ts = {p: 0 for p in pairs}
            self.ws_client = ws_client

        def on_tick(self, pair: str, last_price: float):
            now = time.time()
            if (now - self.last_call_ts[pair]) >= self.aggregator_interval:
                self._aggregator_cycle(pair, last_price)
                self.last_call_ts[pair] = int(now)

        def _aggregator_cycle(self, pair: str, last_price: float):
            # Fetch sentiment trends and price trends
            sentiment_stats = self._fetch_cryptopanic_sentiment_trends(pair)
            price_stats = self._fetch_recent_price_trends(pair)

            # Skip if we lack sufficient data
            if not sentiment_stats or not price_stats:
                logger.info(f"Skipping {pair}: Insufficient sentiment or trend data.")
                return

            # Prepare enriched aggregator data
            aggregator_data = {
                "pair": pair,
                "price": last_price,
                **sentiment_stats,  # Include sentiment trends
                **price_stats       # Include price trends/statistics
            }

            # Predict action/size using AIStrategy
            final_action, final_size = self.strategy.predict(aggregator_data)

            # Place orders if action is BUY or SELL
            if final_action in ("BUY", "SELL") and final_size > 0.0:
                side_str = "buy" if final_action == "BUY" else "sell"
                logger.info(f"Placing real order: {pair}, {side_str}, size={final_size}")
                self.ws_client.send_order(
                    pair=pair,
                    side=side_str,
                    ordertype="market",
                    volume=final_size
                )

        def _fetch_cryptopanic_sentiment_trends(self, pair: str) -> dict:
            conn = sqlite3.connect(DB_FILE)
            try:
                symbol = pair.split("/")[0]
                query = """
                    SELECT timestamp, sentiment_score
                    FROM cryptopanic_news
                    WHERE symbol = ?
                    AND timestamp >= strftime('%s', 'now', '-30 days')
                    ORDER BY timestamp ASC
                """
                df = pd.read_sql_query(query, conn, params=[symbol])
                if df.empty:
                    return {}

                # Calculate rolling averages and volatility
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
                df.set_index("datetime", inplace=True)
                df["sentiment_ma_7d"] = df["sentiment_score"].rolling("7D").mean()
                df["sentiment_ma_14d"] = df["sentiment_score"].rolling("14D").mean()
                df["sentiment_volatility"] = df["sentiment_score"].rolling("14D").std()

                return {
                    "sentiment_score": df["sentiment_score"].iloc[-1],          # Latest sentiment
                    "sentiment_ma_7d": df["sentiment_ma_7d"].iloc[-1],          # 7-day average
                    "sentiment_ma_14d": df["sentiment_ma_14d"].iloc[-1],        # 14-day average
                    "sentiment_volatility": df["sentiment_volatility"].iloc[-1] # Sentiment volatility
                }
            except Exception as e:
                logger.exception(f"Error fetching sentiment trends for {pair}: {e}")
                return {}
            finally:
                conn.close()

        def _fetch_recent_price_trends(self, pair: str) -> dict:
            conn = sqlite3.connect(DB_FILE)
            try:
                query = """
                    SELECT last_price
                    FROM price_history
                    WHERE pair = ?
                    ORDER BY timestamp DESC
                    LIMIT 100
                """
                df = pd.read_sql_query(query, conn, params=[pair])
                if df.empty:
                    return {}

                # Compute trends
                df["moving_avg_10"] = df["last_price"].rolling(10).mean().iloc[-1]
                df["moving_avg_30"] = df["last_price"].rolling(30).mean().iloc[-1]
                df["volatility"] = df["last_price"].pct_change().std()

                return {
                    "moving_avg_10": df["moving_avg_10"],
                    "moving_avg_30": df["moving_avg_30"],
                    "volatility": df["volatility"]
                }
            except Exception as e:
                logger.exception(f"Error fetching price trends for {pair}: {e}")
                return {}
            finally:
                conn.close()

    # 6) Create the WebSocket client with the aggregator
    app = HybridApp(
        pairs=TRADED_PAIRS,
        strategy=ai_model,
        aggregator_interval=AGGREGATOR_INTERVAL_SECONDS
    )
    ws_client = KrakenWSClient(
        pairs=TRADED_PAIRS,
        feed_type="ticker",
        api_key=KRAKEN_API_KEY,
        api_secret=KRAKEN_API_SECRET,
        on_ticker_callback=app.on_tick,
        start_in_private_feed=True,
        private_token=token_str
    )
    app.ws_client = ws_client

    # 7) Start the WebSocket feed
    ws_client.start()
    logger.info("Press Ctrl+C to exit the main loop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Exiting main loop due to KeyboardInterrupt.")
    finally:
        ws_client.stop()
        logger.info("Stopped WebSocket and main app.")

if __name__ == "__main__":
    main()