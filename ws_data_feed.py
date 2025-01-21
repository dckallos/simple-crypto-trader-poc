# ==============================================================================
# FILE: ws_data_feed.py
# ==============================================================================
"""
ws_data_feed.py

Connects to Kraken's WebSocket API using the 'websockets' library with a secure
SSL context (via certifi). Subscribes to the "ticker" stream for specified pairs
and stores real-time data into 'price_history' using the store_price_history()
function from db.py.

Usage (from main.py or elsewhere):
    ws_client = KrakenWSClient(["XBT/USD", "ETH/USD"])
    ws_client.start()   # Runs an event loop in a background thread

    # Later, when stopping:
    ws_client.stop()
"""

import ssl
import certifi
import asyncio
import json
import logging
import threading

import websockets

from db import store_price_history

logger = logging.getLogger(__name__)


def create_secure_ssl_context() -> ssl.SSLContext:
    """
    Creates an SSL context using certifi's trusted CA certificates.
    Ensures SSLv2 and SSLv3 are disabled, requires certificate validation, and
    checks hostname.

    :return: An ssl.SSLContext configured for secure server auth.
    """
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    # Disable old/insecure protocols
    context.options |= ssl.OP_NO_SSLv2
    context.options |= ssl.OP_NO_SSLv3
    # Require certificate validation
    context.verify_mode = ssl.CERT_REQUIRED
    context.check_hostname = True
    # Load certifi certificate bundle
    context.load_verify_locations(cafile=certifi.where())
    return context


class KrakenWSClient:
    """
    KrakenWSClient manages a continuous WebSocket connection to Kraken's public
    ticker feed. It uses an async approach internally, but provides a simple
    start() method that runs in a background thread for convenience.
    """

    def __init__(self, pairs):
        """
        :param pairs: List of trading pairs, e.g. ["XBT/USD", "ETH/USD"].
        """
        self.url = "wss://ws.kraken.com"
        self.pairs = pairs
        self.ssl_context = create_secure_ssl_context()
        self.running = False
        self._thread = None

    def start(self):
        """
        Starts the WebSocket client in a background thread, so main.py can
        proceed without blocking. This method sets up an asyncio event loop
        and calls self.run().

        Call stop() to end gracefully.
        """
        if self.running:
            logger.warning("KrakenWSClient is already running.")
            return

        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Kraken WebSocket client started (async, in background thread).")

    def _run_loop(self):
        """
        Internal method that runs the asyncio event loop for self.run().
        """
        asyncio.run(self.run())

    async def run(self):
        """
        Main async routine that connects to the Kraken WebSocket,
        subscribes to ticker data, and processes incoming messages.
        Reconnects on errors if self.running is still True.
        """
        while self.running:
            try:
                async with websockets.connect(self.url, ssl=self.ssl_context) as ws:
                    # Subscribe to the ticker feed for the requested pairs
                    await self._subscribe(ws)

                    # Continuously receive messages
                    while self.running:
                        message = await ws.recv()
                        await self._handle_message(message)

            except Exception as e:
                # On any exception (network error, SSL error, etc.), we log it,
                # then wait briefly before reconnecting (if still running).
                logger.exception(f"Error in WS connection: {e}")
                if not self.running:
                    break
                logger.info("WebSocket disconnected. Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def _subscribe(self, ws):
        """
        Sends the subscribe message for the 'ticker' feed to Kraken.
        """
        subscribe_msg = {
            "event": "subscribe",
            "pair": self.pairs,
            "subscription": {"name": "ticker"}
        }
        await ws.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to ticker feed for pairs: {self.pairs}")

    async def _handle_message(self, message: str):
        """
        Parses incoming messages, which can be either subscription events
        or ticker data arrays. For ticker data, store to DB.
        """
        try:
            data = json.loads(message)

            if isinstance(data, dict):
                # Possibly a system status or subscription status message
                event_type = data.get("event")
                if event_type in ["subscribe", "subscriptionStatus", "heartbeat", "systemStatus"]:
                    logger.debug(f"WS event: {data}")
                return

            # Ticker data is usually a list with at least 4 elements
            if isinstance(data, list) and len(data) > 3:
                # data format example: [channelID, {tickerData}, "ticker", "XBT/USD"]
                pair = data[3]
                update_obj = data[1]  # dict with keys like 'a', 'b', 'c', 'v', etc.

                ask_info = update_obj.get("a", [])
                bid_info = update_obj.get("b", [])
                last_trade_info = update_obj.get("c", [])
                volume_info = update_obj.get("v", [])

                ask_price = float(ask_info[0]) if len(ask_info) > 0 else 0.0
                bid_price = float(bid_info[0]) if len(bid_info) > 0 else 0.0
                last_price = float(last_trade_info[0]) if len(last_trade_info) > 0 else 0.0
                volume = float(volume_info[0]) if len(volume_info) > 0 else 0.0

                # Store in the DB
                store_price_history(
                    pair=pair,
                    bid=bid_price,
                    ask=ask_price,
                    last=last_price,
                    volume=volume
                )
        except Exception as e:
            logger.exception(f"Error processing WS message: {e}")

    def stop(self):
        """
        Signals to stop the WS client. This will exit the run() loop after
        the next reconnect or next iteration.
        """
        self.running = False
        logger.info("KrakenWSClient stop requested.")
        # If there's an active websocket or loop, it will exit gracefully soon.
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            logger.info("KrakenWSClient background thread joined.")
