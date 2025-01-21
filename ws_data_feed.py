# ==============================================================================
# FILE: ws_data_feed.py
# ==============================================================================
"""
ws_data_feed.py

Connects to Kraken's WebSocket API using 'websockets' with a secure
SSL context (via certifi). Subscribes to multiple pairs for ticker updates.
Optionally, you can also subscribe to the 'trade' feed if you want more frequent
records.

Stores real-time data into 'price_history' via store_price_history(). If you
subscribe to the 'trade' feed, you can also create a separate table or store it
in price_history with appropriate fields.
"""

import ssl
import certifi
import asyncio
import json
import logging
import threading
import websockets

from db import store_price_history  # We assume price_history has (timestamp, pair, bid, ask, last, volume)

logger = logging.getLogger(__name__)


def create_secure_ssl_context() -> ssl.SSLContext:
    """
    Creates an SSL context using certifi's trusted CA certificates.
    Ensures SSLv2 and SSLv3 are disabled, requires certificate validation, and
    checks hostname.
    """
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    context.options |= ssl.OP_NO_SSLv2
    context.options |= ssl.OP_NO_SSLv3
    context.verify_mode = ssl.CERT_REQUIRED
    context.check_hostname = True
    context.load_verify_locations(cafile=certifi.where())
    return context


class KrakenWSClient:
    """
    KrakenWSClient manages a continuous WebSocket connection to Kraken's public
    feeds. It uses an async approach internally, but provides a simple
    start() method that runs in a background thread.
    """

    def __init__(self, pairs, feed_type="trade"):
        """
        :param pairs: List of trading pairs, e.g. ["XBT/USD", "ETH/USD", "SOL/USD"].
        :param feed_type: "ticker" for summary updates, "trade" for every trade, etc.
                          - 'ticker' feed typically updates on each new best bid/ask change.
                          - 'trade' feed sends each trade in real time.
        """
        self.url = "wss://ws.kraken.com"
        self.pairs = pairs
        self.feed_type = feed_type  # "ticker" or "trade" or other
        self.ssl_context = create_secure_ssl_context()
        self.running = False
        self._thread = None

    def start(self):
        """
        Starts the WebSocket client in a background thread, so main.py can
        proceed without blocking.
        """
        if self.running:
            logger.warning("KrakenWSClient is already running.")
            return

        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"Kraken WebSocket client started in background thread (feed_type={self.feed_type}).")

    def _run_loop(self):
        """
        Internal method that runs the asyncio event loop for self.run().
        """
        asyncio.run(self.run())

    async def run(self):
        """
        Main async routine that connects to the Kraken WebSocket,
        subscribes to data, and processes incoming messages.
        Reconnects on errors if self.running is still True.
        """
        while self.running:
            try:
                async with websockets.connect(self.url, ssl=self.ssl_context) as ws:
                    # Subscribe to the feed for the requested pairs
                    await self._subscribe(ws)

                    # Continuously receive messages
                    while self.running:
                        message = await ws.recv()
                        await self._handle_message(message)

            except Exception as e:
                logger.exception(f"Error in WS connection: {e}")
                if not self.running:
                    break
                logger.info("WebSocket disconnected. Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def _subscribe(self, ws):
        """
        Sends the subscription message for the chosen feed to Kraken.
        For the 'ticker' feed, we get best bid/ask and last trade info.
        For the 'trade' feed, we get each trade individually.
        """
        # Example: subscription object could be {"name": "ticker"} or {"name": "trade"}
        subscription_obj = {"name": self.feed_type}
        subscribe_msg = {
            "event": "subscribe",
            "pair": self.pairs,
            "subscription": subscription_obj
        }
        await ws.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to {self.feed_type} feed for pairs: {self.pairs}")

    async def _handle_message(self, message: str):
        """
        Parses incoming messages. For 'ticker' or 'trade' feed, store to DB.
        """
        try:
            data = json.loads(message)

            # Could be system events, heartbeats, etc.
            if isinstance(data, dict):
                event_type = data.get("event")
                if event_type in ["subscribe", "subscriptionStatus", "heartbeat", "systemStatus"]:
                    logger.debug(f"WS event: {data}")
                return

            # If it's a list, it typically holds feed data
            if isinstance(data, list):
                # Ticker feed format example: [channelID, {tickerData}, "ticker", "XBT/USD"]
                # Trade feed format example: [channelID, [["price","volume","time","side","type","misc"]], "trade", "XBT/USD"]
                if len(data) < 4:
                    return  # Not enough data

                feed_name = data[2]  # "ticker", "trade", etc.
                pair = data[3]

                # TICKER FEED
                if feed_name == "ticker":
                    update_obj = data[1]  # a dict with keys like "a", "b", "c", "v"
                    ask_info = update_obj.get("a", [])
                    bid_info = update_obj.get("b", [])
                    last_trade_info = update_obj.get("c", [])
                    volume_info = update_obj.get("v", [])

                    ask_price = float(ask_info[0]) if len(ask_info) > 0 else 0.0
                    bid_price = float(bid_info[0]) if len(bid_info) > 0 else 0.0
                    last_price = float(last_trade_info[0]) if len(last_trade_info) > 0 else 0.0
                    volume = float(volume_info[0]) if len(volume_info) > 0 else 0.0

                    store_price_history(
                        pair=pair,
                        bid=bid_price,
                        ask=ask_price,
                        last=last_price,
                        volume=volume
                    )

                # TRADE FEED
                elif feed_name == "trade":
                    # data[1] is a list of trades, each trade is [price, volume, time, side, order_type, misc]
                    trades_list = data[1]
                    for trade in trades_list:
                        # Parse the fields
                        price = float(trade[0])
                        vol = float(trade[1])
                        # time = float(trade[2])  # you could store this if you have a suitable column
                        # side = trade[3]         # "b" or "s"
                        # order_type = trade[4]   # "l" or "m"
                        # misc = trade[5]

                        # We'll just store into price_history for demonstration,
                        # but you might want a separate 'trades_feed' table.
                        # Because there's no concept of "bid" or "ask" here, we store them as 0.0
                        store_price_history(
                            pair=pair,
                            bid=0.0,
                            ask=0.0,
                            last=price,
                            volume=vol
                        )

        except Exception as e:
            logger.exception(f"Error processing WS message: {e}")

    def stop(self):
        """
        Signals to stop the WS client. The active connection will exit, and the
        background thread will join.
        """
        self.running = False
        logger.info("KrakenWSClient stop requested.")
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            logger.info("KrakenWSClient background thread joined.")
