# ==============================================================================
# FILE: main.py
# ==============================================================================
"""
main.py

Entrypoint for the lightweight AI trading application. This file:
1. Reads toggles from config.yaml (enable_training, enable_live_ai_inference, enable_gpt_integration).
2. Optionally runs model training if 'enable_training' is True.
3. Starts a WebSocket feed to capture real-time price data from Kraken for multiple pairs.
4. If 'enable_live_ai_inference' is True, it loads the AIStrategy (with or without GPT).
   Otherwise, it uses a "dummy" strategy that never trades.
5. Places trades and records them in a simplified database if the strategy suggests it.

NOTE (Single-Threaded Trade Logic):
- The only concurrency here is the background thread for the WebSocket feed, which
  writes price data to the DB. All trade placement occurs in this main thread.
  This ensures no concurrency issues for placing orders.
"""

import time
import os
import logging
import sqlite3
from dotenv import load_dotenv
import yaml

# ------------------------------------------------------------------------------
# Import your AIStrategy class
# ------------------------------------------------------------------------------
from ai_strategy import AIStrategy
from db import init_db, record_trade_in_db
from ws_data_feed import KrakenWSClient

# (OPTIONAL) If you have a function to run training in train_model.py, you might import it:
from fetch_lunarcrush import fetch_lunarcrush_data
from fetch_cryptopanic import fetch_cryptopanic_data
from train_model import main as training_main
# or do some other approach like subprocess.

# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Configuration + Toggles
# ------------------------------------------------------------------------------
load_dotenv()
KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "FAKE_KEY")        # replace with real
KRAKEN_API_SECRET = os.getenv("KRAKEN_SECRET_API_KEY", "FAKE_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "FAKE_OPENAI_KEY")

# Load toggles (and other settings) from config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

TRADED_PAIRS = config.get("traded_pairs", [])

POLL_INTERVAL_SECONDS = 15  # how often we fetch data & run the AI
MAX_POSITION_SIZE = 0.001   # example max size for a single trade in BTC
DB_ENABLED = True           # toggle whether we record trades in sqlite

# New toggles from config.yaml
ENABLE_TRAINING = config.get("enable_training", False)
ENABLE_LIVE_AI_INFERENCE = config.get("enable_live_ai_inference", True)
ENABLE_GPT_INTEGRATION = config.get("enable_gpt_integration", False)

DB_FILE = "trades.db"       # name of the DB file

# ------------------------------------------------------------------------------
# Utility: Get the latest price from our DB
# ------------------------------------------------------------------------------
def get_latest_price_from_db(pair: str) -> dict:
    """
    Fetch the most recent price entry for a given pair from the 'price_history' table.

    :param pair: e.g. 'XBT/USD'.
    :return: dict with 'price' and 'timestamp' (and optionally other fields),
             or empty dict if none found.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            SELECT timestamp, bid_price, ask_price, last_price, volume
            FROM price_history
            WHERE pair=?
            ORDER BY id DESC
            LIMIT 1
        """, (pair,))
        row = c.fetchone()
        if row:
            # row: (timestamp, bid, ask, last, volume)
            return {
                "timestamp": row[0],
                "bid": row[1],
                "ask": row[2],
                "price": row[3],  # we'll use 'price' as a general field
                "volume": row[4]
            }
        else:
            return {}
    except Exception as e:
        logger.exception(f"Error fetching latest price from DB: {e}")
        return {}
    finally:
        conn.close()

# ------------------------------------------------------------------------------
# Order Placement (Mock)
# ------------------------------------------------------------------------------
def place_order(pair: str, side: str, volume: float) -> str:
    """
    Places an order on Kraken (mock implementation).

    :param pair: The trading pair, e.g. 'XBT/USD'.
    :param side: 'BUY' or 'SELL'.
    :param volume: Quantity to trade.
    :return: A mock order ID string. Replace with real Kraken REST calls in production.
    """
    logger.info(f"Placing {side} order for {volume} of {pair} (mock).")
    mock_order_id = f"MOCK-{side}-{int(time.time())}"
    return mock_order_id

# ------------------------------------------------------------------------------
# Optional: A DummyStrategy to skip AI logic
# ------------------------------------------------------------------------------
class DummyStrategy:
    """
    A fallback strategy that always returns HOLD, so no trades are placed.
    """
    def predict(self, market_data: dict):
        # always hold
        return ("HOLD", 0.0)

# ------------------------------------------------------------------------------
# Main Loop
# ------------------------------------------------------------------------------
def main():
    """
    Main loop for:
      1. Optionally run training if 'enable_training' is True.
      2. Initializing DB (if desired).
      3. Starting WebSocket for real-time data capture of all pairs (background thread).
      4. Either load AIStrategy (if 'enable_live_ai_inference' is True)
         or a DummyStrategy (if it's False).
      5. Periodically fetch DB data, run strategy predict, place trades, etc.
    """
    logger.info("Starting AI trading app...")

    # 1) Optional training
    if ENABLE_TRAINING:
        logger.info("ENABLE_TRAINING is True, running training routine now...")
        # If you have a direct function:
        fetch_cryptopanic_data()
        fetch_lunarcrush_data()
        training_main()
        # Or if you want to run a separate script:
        # import subprocess
        # subprocess.run(["python", "train_model.py"], check=True)
        logger.info("Training complete (placeholder).")
    else:
        logger.info("ENABLE_TRAINING is False. Skipping model training step.")

    # 2) Init the DB if desired
    if DB_ENABLED:
        init_db()
        logger.info("Database initialized.")

    # 3) Start the WebSocket feed in background
    ws_client = KrakenWSClient(TRADED_PAIRS, feed_type="ticker")
    ws_client.start()

    # 4) Decide if we do real AI or a dummy approach
    if ENABLE_LIVE_AI_INFERENCE:
        logger.info("ENABLE_LIVE_AI_INFERENCE is True => Loading AIStrategy.")
        ai_model = AIStrategy(
            pairs=TRADED_PAIRS,
            model_path="trained_model.pkl",
            use_openai=ENABLE_GPT_INTEGRATION
        )
        logger.info(f"AIStrategy loaded with pairs: {TRADED_PAIRS}")
    else:
        logger.info("ENABLE_LIVE_AI_INFERENCE is False => Using DummyStrategy. No trades will be placed.")
        ai_model = DummyStrategy()

    # 5) Enter main trading loop
    while True:
        try:
            # For each pair, we fetch the latest data from DB
            for pair in TRADED_PAIRS:
                market_data = get_latest_price_from_db(pair)

                # If there's no data in DB for this pair yet, skip it
                if not market_data or 'price' not in market_data:
                    logger.warning(f"No WS data yet for {pair}. Skipping this pair.")
                    continue

                logger.info(f"Latest price for {pair} = {market_data['price']}")

                # AI (or dummy) strategy decides what to do
                signal, suggested_size = ai_model.predict(market_data)
                logger.info(f"Strategy suggests: {signal} with size={suggested_size} for {pair}")

                # Minimal risk check: do not exceed MAX_POSITION_SIZE
                final_size = min(suggested_size, MAX_POSITION_SIZE)

                if signal in ("BUY", "SELL"):
                    order_id = place_order(pair, signal, final_size)

                    if DB_ENABLED:
                        record_trade_in_db(
                            side=signal,
                            quantity=final_size,
                            price=market_data["price"],
                            order_id=order_id,
                            pair=pair
                        )
                        logger.info(f"Trade recorded in DB for {pair}: order_id={order_id}")
                else:
                    logger.info(f"No action taken ({signal}) for {pair}.")

            logger.info(f"Sleeping {POLL_INTERVAL_SECONDS} seconds...\n")
            time.sleep(POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("Shutting down trading app...")
            break
        except Exception as e:
            logger.exception(f"Unexpected error in main loop: {e}")
            logger.info("Continuing after exception...")

    # 6) Cleanly stop the WebSocket
    ws_client.stop()


if __name__ == "__main__":
    main()
