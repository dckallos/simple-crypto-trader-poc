# ==============================================================================
# FILE: main.py
# ==============================================================================
"""
main.py

Entrypoint for the lightweight AI trading application. This file:
1. Initializes simple configuration (API keys, intervals).
2. (New) Starts a WebSocket feed to capture real-time price data from Kraken.
3. Fetches or reads market data (from the DB or fallback HTTP request).
4. Feeds data to the AIStrategy.
5. Places trades and records them in a simplified database.
"""

import time
import os
import logging
import requests
import sqlite3
from dotenv import load_dotenv

from ai_strategy import AIStrategy
from db import init_db, record_trade_in_db
from ws_data_feed import KrakenWSClient  # <-- NEW: reference the WebSocket module

# ------------------------------------------------------------------------------
# Logging Setup: simple console-based logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
load_dotenv()
KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "FAKE_KEY")  # replace with real
KRAKEN_API_SECRET = os.getenv("KRAKEN_SECRET_API_KEY", "FAKE_SECRET")
TRADING_PAIR = "ETH/USD"  # symbol for demonstration
POLL_INTERVAL_SECONDS = 15  # how often we fetch data & run the AI
MAX_POSITION_SIZE = 0.001  # example max size for a single trade in BTC
DB_ENABLED = True  # toggle whether we record trades in sqlite

DB_FILE = "trades.db"  # name of the DB file

# ------------------------------------------------------------------------------
# Market Data Fetch (HTTP Fallback or for additional data)
# ------------------------------------------------------------------------------
def fetch_kraken_data(pair: str) -> dict:
    """
    Fetches basic market data for the given Kraken pair using Kraken's
    public Ticker endpoint. This function is synchronous for simplicity.

    :param pair: Friendly pair string, e.g. "XBT/USD".
    :return: A dict containing at least {"price": float, "timestamp": float}.
    """
    # Convert "XBT/USD" -> "XBTUSD" for Kraken's REST API call
    pair_for_url = pair.replace("/", "")
    url = f"https://api.kraken.com/0/public/Ticker?pair={pair_for_url}"

    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("error"):
            logger.warning(f"Kraken Ticker error: {data['error']}")
            return {"price": 0.0, "timestamp": time.time()}

        # Extract the first ticker entry
        result = data.get("result", {})
        if not result:
            logger.warning("No 'result' in Kraken response.")
            return {"price": 0.0, "timestamp": time.time()}

        first_key = list(result.keys())[0]
        ticker_info = result[first_key]
        last_trade_price_str = ticker_info["c"][0]  # "c" => last trade [price, lot volume]

        return {
            "price": float(last_trade_price_str),
            "timestamp": time.time()
        }

    except Exception as e:
        logger.exception(f"Error fetching data from Kraken: {e}")
        return {"price": 0.0, "timestamp": time.time()}


# ------------------------------------------------------------------------------
# Utility: get the latest price from our DB
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
# Main Loop
# ------------------------------------------------------------------------------
def main():
    """
    Main loop for:
      1. Initializing DB (if desired).
      2. Loading AI strategy.
      3. Starting WebSocket for real-time data capture.
      4. Periodically fetching the latest data from DB (or fallback to HTTP).
      5. Getting AI signals and placing trades.
    """
    logger.info("Starting AI trading app...")

    # Init the DB (if enabled)
    if DB_ENABLED:
        init_db()
        logger.info("Database initialized.")

    # Start the WebSocket feed in background
    ws_client = KrakenWSClient([TRADING_PAIR])
    ws_client.start()

    # Load the AI strategy
    ai_model = AIStrategy()
    logger.info("AI strategy loaded.")

    while True:
        try:
            # Attempt to get latest price from DB (populated by WebSocket)
            market_data = get_latest_price_from_db(TRADING_PAIR)

            # If DB doesn't have data yet (e.g., just started), fallback to HTTP
            if not market_data or 'price' not in market_data:
                logger.info("No WS data yet, fetching from HTTP as fallback...")
                fallback_data = fetch_kraken_data(TRADING_PAIR)
                # Format fallback to match the 'market_data' structure
                market_data = {
                    "timestamp": fallback_data["timestamp"],
                    "price": fallback_data["price"],
                    "bid": fallback_data["price"],
                    "ask": fallback_data["price"],
                    "volume": 0.0
                }

            logger.info(f"Latest price for {TRADING_PAIR} = {market_data['price']}")

            # AI strategy decides what to do
            signal, suggested_size = ai_model.predict(market_data)
            logger.info(f"AIStrategy suggests: {signal} with size={suggested_size}")

            # Minimal risk check: do not exceed MAX_POSITION_SIZE
            final_size = min(suggested_size, MAX_POSITION_SIZE)

            if signal in ("BUY", "SELL"):
                order_id = place_order(TRADING_PAIR, signal, final_size)

                if DB_ENABLED:
                    record_trade_in_db(
                        side=signal,
                        quantity=final_size,
                        price=market_data["price"],
                        order_id=order_id,
                        pair=TRADING_PAIR
                    )
                    logger.info(f"Trade recorded in DB: order_id={order_id}")
            else:
                logger.info("No action taken (HOLD).")

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
