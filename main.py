# ==============================================================================
# FILE: main.py
# ==============================================================================
"""
main.py

Entrypoint for the lightweight AI trading application. This file:
1. Initializes simple configuration (API keys, intervals).
2. Starts a WebSocket feed to capture real-time price data from Kraken for multiple pairs.
3. Fetches or reads market data from the DB (no HTTP fallback).
4. Feeds data to the AIStrategy.
5. Places trades and records them in a simplified database.

NOTE (Single-Threaded Trade Logic):
- The only concurrency here is the background thread for the WebSocket feed, which
  writes price data to the DB. All trade placement occurs in the main while-loop below.
  This ensures no concurrency issues for trade execution: only one loop can place orders.
"""

import time
import os
import logging
import sqlite3
from dotenv import load_dotenv
import yaml

from ai_strategy import AIStrategy
from db import init_db, record_trade_in_db
from ws_data_feed import KrakenWSClient  # reference the WebSocket module

# ------------------------------------------------------------------------------
# Logging Setup: simple console-based logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
load_dotenv()
KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "FAKE_KEY")     # replace with real
KRAKEN_API_SECRET = os.getenv("KRAKEN_SECRET_API_KEY", "FAKE_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "FAKE_OPENAI_KEY")

# Load pairs from your config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

TRADED_PAIRS = config.get("traded_pairs", [])

POLL_INTERVAL_SECONDS = 15  # how often we fetch data & run the AI
MAX_POSITION_SIZE = 0.001   # example max size for a single trade in BTC
DB_ENABLED = True           # toggle whether we record trades in sqlite

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
# Main Loop (Single-Threaded Trade Logic)
# ------------------------------------------------------------------------------
def main():
    """
    Main loop for:
      1. Initializing DB (if desired).
      2. Loading AI strategy (with multi-pair awareness).
      3. Starting WebSocket for real-time data capture of all pairs (background thread).
      4. Periodically fetching the latest data from DB.
      5. Getting AI signals and placing trades for each pair if data is available.

    Concurrency:
      - The WebSocket feed runs in a separate thread to store price data in DB.
      - Only this main thread places trades, ensuring no concurrency risk for orders.
    """
    logger.info("Starting AI trading app...")

    # Init the DB (if enabled)
    if DB_ENABLED:
        init_db()
        logger.info("Database initialized.")

    # Start the WebSocket feed (feed_type="trade" or "ticker") in background
    ws_client = KrakenWSClient(TRADED_PAIRS, feed_type="ticker")
    ws_client.start()

    # Load the AI strategy, passing the same pairs for multi-coin consideration
    ai_model = AIStrategy(pairs=TRADED_PAIRS, model_path="trained_model.pkl", use_openai=True)
    logger.info(f"AI strategy loaded with pairs: {TRADED_PAIRS}")

    while True:
        try:
            # For each pair, we fetch the latest data from DB
            for pair in TRADED_PAIRS:
                market_data = get_latest_price_from_db(pair)

                # If there's no data in DB for this pair yet, skip it
                if not market_data or 'price' not in market_data:
                    logger.warning(f"No WS data yet for {pair}. Skipping this pair.")
                    continue  # skip the rest of logic for this pair

                logger.info(f"Latest price for {pair} = {market_data['price']}")

                # AI strategy decides what to do
                signal, suggested_size = ai_model.predict(market_data)
                logger.info(f"AIStrategy suggests: {signal} with size={suggested_size} for {pair}")

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
                    logger.info(f"No action taken (HOLD) for {pair}.")

            logger.info(f"Sleeping {POLL_INTERVAL_SECONDS} seconds...\n")
            time.sleep(POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("Shutting down trading app...")
            break
        except Exception as e:
            logger.exception(f"Unexpected error in main loop: {e}")
            logger.info("Continuing after exception...")

    # Cleanly stop the WebSocket
    ws_client.stop()


if __name__ == "__main__":
    main()
