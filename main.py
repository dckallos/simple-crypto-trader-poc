# ==============================================================================
# FILE: main.py
# ==============================================================================
"""
main.py

A mature application that:
1) Initializes the DB (init_db from db.py).
2) Optionally runs training if config says so (fetch_cryptopanic, fetch_lunarcrush, then train_model).
3) Starts a Kraken WebSocket feed for real-time data, storing it in price_history.
4) Uses aggregator intervals to pass data to AIStrategy (with GPT or scikit).
5) Forcibly closes sub-positions if RiskManager triggers stop-loss or take-profit.
6) Demonstrates how to connect to the private feed for order management if desired.

Key Points:
- We rely on 'ai_strategy.py' to store decisions in 'ai_decisions' for each final action.
- We unify aggregator data so that 'market_data["price"]' is recognized by AIStrategy.
- We remove any leftover snippet that might spam direct record_trade_in_db("BUY", ...).

Everything else remains consistent with your multi-position logic,
private feed placeholders, etc.
"""

import time
import requests
import hashlib
import hmac
import base64
import math
import logging
import logging.config
import yaml
import os
import json
from dotenv import load_dotenv

from db import init_db
from ai_strategy import AIStrategy
from fetch_cryptopanic import fetch_cryptopanic_data
from fetch_lunarcrush import fetch_lunarcrush_data
from train_model import main as training_main
from ws_data_feed import KrakenWSClient

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
        '': {  # root logger
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False
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

def main():
    logging.config.dictConfig(LOG_CONFIG)
    # requests & urllib3 logs at DEBUG, root at INFO below.

    CONFIG_FILE = "config.yaml"
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

    ENABLE_TRAINING = config.get("enable_training", True)
    print(f"Loaded config: {config}")
    ENABLE_LIVE_AI_INFERENCE = config.get("enable_live_ai_inference", True)
    ENABLE_GPT_INTEGRATION = config.get("enable_gpt_integration", True)
    TRADED_PAIRS = config.get("traded_pairs", [])
    DB_FILE = "trades.db"
    AGGREGATOR_INTERVAL_SECONDS = config.get("trade_interval_seconds", 60)

    load_dotenv()
    KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "FAKE_KEY")
    KRAKEN_API_SECRET = os.getenv("KRAKEN_SECRET_API_KEY", "FAKE_SECRET")

    # 1) init DB
    init_db()

    # 2) Possibly run training
    if ENABLE_TRAINING:
        fetch_cryptopanic_data()
        fetch_lunarcrush_data()
        training_main()
        logger.info("Training done.")
    else:
        logger.info("Skipping training step per config.")

    # 3) AIStrategy creation
    model_path = "trained_model.pkl" if os.path.exists("trained_model.pkl") else None
    ai_model = AIStrategy(
        pairs=TRADED_PAIRS,
        model_path=model_path,
        use_openai=ENABLE_GPT_INTEGRATION,
        max_position_size=0.001,
        stop_loss_pct=0.05,
        take_profit_pct=0.01,
        max_daily_drawdown=-0.02
    )
    logger.info(f"AIStrategy loaded with pairs={TRADED_PAIRS}, GPT={ENABLE_GPT_INTEGRATION}")

    # A Hybrid aggregator approach:
    class HybridApp:
        """
        A minimal aggregator approach that calls AIStrategy after a certain time
        has elapsed. Each new ticker update calls on_tick(...).
        """
        def __init__(self, pairs, strategy, aggregator_interval=60):
            self.pairs = pairs
            self.strategy = strategy
            self.aggregator_interval = aggregator_interval
            self.last_call_ts = {p: 0 for p in pairs}

        def on_tick(self, pair: str, last_price: float):
            now = time.time()
            if (now - self.last_call_ts[pair]) >= self.aggregator_interval:
                self._aggregator_cycle(pair, last_price)
                self.last_call_ts[pair] = int(now)

        def _aggregator_cycle(self, pair: str, last_price: float):
            # If you want advanced aggregator merges from DB or cryptopanic,
            # do that here. For demonstration, we pass a minimal dictionary with "price".
            aggregator_data = {
                "pair": pair,
                "price": last_price
            }
            self.strategy.predict(aggregator_data)

    app = HybridApp(
        pairs=TRADED_PAIRS,
        strategy=ai_model,
        aggregator_interval=AGGREGATOR_INTERVAL_SECONDS
    )

    logger.debug("Starting Kraken WebSocket for public data with websockets approach.")
    ws_client = KrakenWSClient(
        pairs=TRADED_PAIRS,
        feed_type="ticker",
        api_key=KRAKEN_API_KEY,
        api_secret=KRAKEN_API_SECRET,
        on_ticker_callback=app.on_tick
    )

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

    # If you want private feed usage, you can do something like:
    # token_json = get_ws_token(KRAKEN_API_KEY, KRAKEN_API_SECRET)
    # if token_json and "result" in token_json and "token" in token_json["result"]:
    #     token_str = token_json["result"]["token"]
    #     logger.info(f"Retrieved token: {token_str}")
    #     ws_client.connect_private_feed(token_str)
    #     # subscribe private feed or send orders
    # else:
    #     logger.warning("Failed to retrieve token or parse it.")


if __name__ == "__main__":
    main()
