# ==============================================================================
# FILE: ws_data_feed.py
# ==============================================================================
"""
ws_data_feed.py

Connects to Kraken's WebSocket API using the 'websockets' library (async).
- Subscribes to public feeds (ticker, trade) for real-time data,
  storing them in DB (via store_price_history).
- Also can connect to private feeds (wss://ws-auth.kraken.com) for
  'openOrders', 'ownTrades', etc., after retrieving a token from REST.
- Offers methods to add/cancel orders in private feed.

We run the async code in a background thread to avoid blocking
the main thread in your `main.py`.

Usage example:
    from ws_data_feed import KrakenWSClient

    client = KrakenWSClient(
        pairs=["XBT/USD", "ETH/USD"],
        feed_type="ticker",
        api_key="YOUR_KEY",
        api_secret="YOUR_SECRET"
    )
    # Start in background thread
    client.start()

    # Optionally connect private feed
    # client.connect_private_feed("YOUR_TOKEN")
    # client.subscribe_private_feed("openOrders")
    # client.send_order("XBT/USD", "buy", "limit", 0.01, 18000.0)

    # ...
    client.stop()
"""

import ssl
import certifi
import asyncio
import json
import logging
import threading
import websockets
import time

from db import store_price_history

logger = logging.getLogger(__name__)


def create_secure_ssl_context() -> ssl.SSLContext:
    """
    Creates an SSL context using certifi's trusted CA certificates.
    Ensures SSLv2 and SSLv3 are disabled, requires certificate validation,
    and checks hostname. This is used by 'websockets' to ensure secure TLS.
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
    KrakenWSClient manages a WebSocket connection to Kraken's public or private
    feeds using 'websockets' in an async event loop.

    PUBLIC FEEDS:
      - "ticker" or "trade" to store price data in 'price_history'.
    PRIVATE FEEDS:
      - Use connect_private_feed(...) with a valid token, then subscribe to
        "openOrders", "ownTrades", etc. Also methods for addOrder/cancelOrder.

    We run everything in a background thread so main code is not blocked.
    """

    def __init__(
        self,
        pairs,
        feed_type="trade",
        api_key=None,
        api_secret=None,
        on_ticker_callback=None
    ):
        """
        :param pairs: List of trading pairs, e.g. ["XBT/USD", "ETH/USD"].
        :param feed_type: "ticker" or "trade" or your custom (public) feed.
        :param api_key: For private feeds if needed.
        :param api_secret: For private feeds if needed.
        """
        self.url_public = "wss://ws.kraken.com"
        self.url_private = "wss://ws-auth.kraken.com"
        self.on_ticker_callback = on_ticker_callback

        self.pairs = pairs
        self.feed_type = feed_type
        self.api_key = api_key
        self.api_secret = api_secret

        # We'll store the selected URL. By default, public feed.
        self.url = self.url_public
        self.token = None

        # For concurrency
        self.running = False
        self._thread = None

        # We'll store the event loop & ssl context
        self.loop = None
        self.ssl_context = create_secure_ssl_context()

    def start(self):
        """
        Starts the WebSocket client in a background thread, so main code
        can continue. We create an asyncio loop and run it.
        """
        if self.running:
            logger.warning("KrakenWSClient is already running.")
            return

        self.running = True
        # Start background thread that runs our async loop
        self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()
        logger.info(f"KrakenWSClient started in background thread (feed_type={self.feed_type}).")

    def _run_async_loop(self):
        """
        The target for our background thread.
        Creates a new event loop, sets it, runs run() until stop is requested.
        """
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            self.loop.run_until_complete(self._main_loop())
        except Exception as e:
            logger.exception(f"Exception in run_async_loop => {e}")
        finally:
            logger.info("Exiting the event loop.")
            self.loop.close()

    async def _main_loop(self):
        """
        Reconnect logic: while self.running, connect => run => on disconnect => retry.
        If not self.running => exit.
        """
        while self.running:
            try:
                logger.info(f"Connecting to {self.url} ...")
                async with websockets.connect(self.url, ssl=self.ssl_context) as ws:
                    logger.info(f"WebSocket connected => feed_type={self.feed_type}")
                    # If it's the public feed => subscribe to feed_type
                    # If private => we might handle differently
                    await self._subscribe(ws)

                    # continuously receive messages
                    await self._consume_messages(ws)
            except Exception as e:
                logger.exception(f"WebSocket error => {e}")
                if not self.running:
                    break
                logger.info("WebSocket disconnected, reconnecting in 5s...")
                await asyncio.sleep(5)

        logger.info("main_loop => done because self.running=False")

    async def _subscribe(self, ws):
        """
        If we're using the public feed, we subscribe with pairs and feed_type.
        If private, we might do it differently, or wait for user code to call
        subscribe_private_feed with token, etc.
        """
        if self.url == self.url_public:
            # public subscription
            subscription_obj = {"name": self.feed_type}
            msg = {
                "event": "subscribe",
                "pair": self.pairs,
                "subscription": subscription_obj
            }
            await ws.send(json.dumps(msg))
            logger.info(f"Subscribed to public {self.feed_type} feed for pairs: {self.pairs}")
        else:
            # private feed => user code might call subscribe_private_feed
            # so do nothing here
            pass

    async def _consume_messages(self, ws):
        """
        Wait for messages and handle them. If self.running is set to False => we break.
        """
        while self.running:
            try:
                message = await ws.recv()
                await self._handle_message(message, ws)
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Connection closed => break consume_messages")
                break
            except Exception as e:
                logger.exception(f"Error receiving WS message => {e}")
                break

    async def _handle_message(self, message: str, ws):
        """
        Parse JSON, route to ticker/trade or system events.
        For private events => handle them too.
        """
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.error(f"JSON parse error => {message}")
            return

        # Could be dict => system events, or list => feed data
        if isinstance(data, dict):
            event_type = data.get("event")
            if event_type in ["subscribe", "subscriptionStatus", "heartbeat", "systemStatus"]:
                logger.debug(f"System/WS event => {data}")
            else:
                await self._handle_system_message(data, ws)
            return

        if isinstance(data, list):
            # e.g. [channelID, payload, "ticker", "XBT/USD"]
            if len(data) < 4:
                logger.debug(f"Short data list => {data}")
                return
            feed_name = data[2]
            pair = data[3]
            if feed_name == "ticker":
                await self._handle_ticker(data[1], pair)
            elif feed_name == "trade":
                await self._handle_trade(data[1], pair)
            else:
                logger.debug(f"Unrecognized feed_name={feed_name}, data={data}")
            return

    async def _handle_system_message(self, data: dict, ws):
        """
        For private or system messages (like addOrderStatus, subscription status, etc.)
        """
        event_type = data.get("event")
        if event_type == "subscriptionStatus":
            logger.info(f"Subscription status => {data}")
        elif event_type == "addOrderStatus":
            status = data.get("status")
            txid = data.get("txid", "")
            if status == "ok":
                logger.info(f"Order placed => txid={txid}, data={data}")
            else:
                logger.warning(f"Order add failed => {data}")
        elif event_type == "cancelOrderStatus":
            status = data.get("status")
            if status == "ok":
                logger.info("Order(s) cancelled successfully.")
            else:
                logger.warning(f"Order cancel failed => data={data}")
        elif event_type == "error":
            logger.error(f"WS error event => {data}")
        elif event_type in ["challenge", "heartbeat"]:
            logger.debug(f"System msg => {data}")
        else:
            logger.debug(f"Unrecognized system message => {data}")

    async def _handle_ticker(self, update_obj, pair):
        """
        For the 'ticker' feed => parse 'a','b','c','v', store to DB.
        """
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

    async def _handle_trade(self, trades_list, pair):
        """
        For the 'trade' feed => data is a list of trades.
        Each trade => [price, volume, time, side, order_type, misc]
        We'll store them in price_history with bid=ask=0.
        """
        for trade in trades_list:
            price = float(trade[0])
            vol = float(trade[1])
            store_price_history(
                pair=pair,
                bid=0.0,
                ask=0.0,
                last=price,
                volume=vol
            )

    def stop(self):
        """
        Stop the WS client => set running=False => event loop stops => join thread.
        """
        if not self.running:
            logger.info("KrakenWSClient is not running or already stopped.")
            return
        self.running = False
        logger.info("KrakenWSClient stop requested.")
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            logger.info("KrakenWSClient background thread joined.")

    # --------------------------------------------------------------------------
    # Private feed / Orders
    # --------------------------------------------------------------------------
    def connect_private_feed(self, token: str):
        """
        Switch to the private feed URL, store the token, and start the event loop if not running.
        This is a simplistic approach => same event loop for private feed or public feed.
        If you want separate, you'd do so differently.
        """
        self.token = token
        self.url = self.url_private
        # If we're already running => we can either stop & restart or do dynamic approach
        if self.running:
            logger.warning("Already running => we will reconnect to private feed on next cycle.")
        else:
            self.start()

    async def _handle_ticker(self, update_obj, pair):
        ask_info = update_obj.get("a", [])
        bid_info = update_obj.get("b", [])
        last_trade_info = update_obj.get("c", [])
        volume_info = update_obj.get("v", [])

        ask_price = float(ask_info[0]) if ask_info else 0.0
        bid_price = float(bid_info[0]) if bid_info else 0.0
        last_price = float(last_trade_info[0]) if last_trade_info else 0.0
        volume = float(volume_info[0]) if volume_info else 0.0

        store_price_history(
            pair=pair,
            bid=bid_price,
            ask=ask_price,
            last=last_price,
            volume=volume
        )

        # NEW: If we have a callback, call it
        if self.on_ticker_callback and last_price > 0:
            # pass pair & last_price
            self.on_ticker_callback(pair, last_price)

    async def _subscribe_private(self, ws, feed_name="openOrders"):
        """
        Called from subscribe_private_feed => or we can do it on connection
        """
        if not self.token:
            logger.warning("No token => cannot sub to private feed.")
            return
        msg = {
            "event": "subscribe",
            "subscription": {
                "name": feed_name,
                "token": self.token
            }
        }
        await ws.send(json.dumps(msg))
        logger.info(f"Attempted to subscribe to private feed => {feed_name}")

    def subscribe_private_feed(self, feed_name="openOrders"):
        """
        Because we are in an async approach, we do a short ephemeral task
        to send subscription. Or store the feed_name for next reconnection logic.
        """
        if not self.running:
            logger.warning("WS not running => call connect_private_feed(token) or start() first.")
            return
        # We can do a thread-safe approach => schedule a coroutine on our event loop:
        async def _sub():
            # We'll find an open websockets connection => hack:
            # Actually, it's simpler to store feed_name => do on next connect
            # or we keep track. We'll do a direct approach:
            logger.info(f"Subscribing to private feed => {feed_name}, token={self.token}")
            # We can't easily get a reference to the current ws => so we might need a new approach
            pass

        # For a robust approach, you'd implement a method to store the feed_name & token,
        # then on next connect we do _subscribe_private. Or manage a global "current_ws" reference.
        logger.warning("Method subscribe_private_feed is a stub in this example => call on connect.")
        # or we do a more advanced approach with a persistent ws reference.

    def send_order(self, pair: str, side: str, ordertype: str, volume: float, price: float = None):
        """
        For the private feed => 'addOrder' event. We'll do a stub approach as well,
        because we need the actual 'ws' reference in the running connection to send messages.
        """
        logger.warning("send_order is a stub => in websockets approach, we need an open reference to ws.")


    def cancel_order(self, txids: list):
        """
        same approach => stub. We don't have a direct reference to the current websockets connection
        in a simple design.
        In a robust design, you'd keep a 'current_ws' that is updated in _main_loop or _consume_messages.
        """
        logger.warning("cancel_order is a stub => see above. We need direct ws access.")
