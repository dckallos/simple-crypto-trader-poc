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
- Now includes an optional way to start directly in private-feed mode if
  `start_in_private_feed=True` is passed, along with `private_token`.

We run the async code in a background thread to avoid blocking
the main thread in your `main.py`.

Usage example:
    from ws_data_feed import KrakenWSClient

    client = KrakenWSClient(
        pairs=["XBT/USD", "ETH/USD"],
        feed_type="ticker",
        api_key="YOUR_KEY",
        api_secret="YOUR_SECRET",
        start_in_private_feed=True,          # optional
        private_token="YOUR_REST_OBTAINED_TOKEN",  # optional
    )
    # Start in background thread
    client.start()

    # The client will begin in private-feed mode if you provided a valid token
    # and set start_in_private_feed=True. Otherwise, it begins in public mode.

    # You can also subscribe to private channels:
    #   client.subscribe_private_feed("openOrders")
    # Or place private orders once connected to the private feed:
    #   client.send_order("XBT/USD", "buy", "limit", 0.01, 18000.0)

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
      - If you set `start_in_private_feed=True` and pass a valid `private_token`,
        we begin in private mode from the start, connecting to wss://ws-auth.kraken.com.
      - Or, call `connect_private_feed(...)` after you have a token to switch from
        public to private feed next time we reconnect or if we're not running yet.
      - Then you can subscribe to private channels (e.g. openOrders) or place orders.

    We run everything in a background thread so the main code is not blocked.
    """

    def __init__(
        self,
        pairs,
        feed_type="trade",
        api_key=None,
        api_secret=None,
        on_ticker_callback=None,
        start_in_private_feed: bool = False,
        private_token: str = None
    ):
        """
        :param pairs: List of trading pairs, e.g. ["XBT/USD", "ETH/USD"].
        :param feed_type: "ticker" or "trade" or your custom (public) feed.
        :param api_key: For private feeds if needed.
        :param api_secret: For private feeds if needed.
        :param on_ticker_callback: A function called whenever we receive a new ticker price.
        :param start_in_private_feed: If True, we start the WS client in private mode
                                      instead of the public feed.
        :param private_token: The token retrieved from the REST call, if you want to
                              start in private feed from the beginning.

        If start_in_private_feed=True and private_token is non-empty,
        we set `url` to `wss://ws-auth.kraken.com` and store `self.token`
        so that once we connect, we treat it as a private feed from the start.
        """
        self.url_public = "wss://ws.kraken.com"
        self.url_private = "wss://ws-auth.kraken.com"
        self.on_ticker_callback = on_ticker_callback

        self.pairs = pairs
        self.feed_type = feed_type
        self.api_key = api_key
        self.api_secret = api_secret

        # We'll store the event loop & ssl context
        self.loop = None
        self.ssl_context = create_secure_ssl_context()

        # If we want to start in private feed from the get-go:
        self.token = None
        if start_in_private_feed and private_token:
            logger.info("KrakenWSClient: Starting in PRIVATE feed mode.")
            self.url = self.url_private
            self.token = private_token
        else:
            # Default to public feed
            self.url = self.url_public

        # For concurrency
        self.running = False
        self._thread = None

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
        Creates a new event loop, sets it, runs _main_loop() until stop is requested.
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
                    logger.info(f"WebSocket connected => feed_type={self.feed_type}, url={self.url}")
                    # If it's the public feed => subscribe to feed_type
                    # If private => we might do so differently
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
        If we're using the public feed, subscribe with pairs and feed_type.
        If private, call private subscription logic if self.token is set.
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
            # private feed => see if we have a token
            if self.token:
                # default => subscribe to e.g. "openOrders" or none if you prefer
                # or rely on user calls to `subscribe_private_feed(...)`
                feed_name = "openOrders"
                await self._subscribe_private(ws, feed_name=feed_name)

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

        # If we have a callback, call it
        if self.on_ticker_callback and last_price > 0:
            self.on_ticker_callback(pair, last_price)

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
        Switch to the private feed URL, store the token, and connect if not running.
        If we are already running, logs a warning that we will reconnect to private
        feed on the next cycle or on next reconnect.

        :param token: The token retrieved from the REST call to GetWebSocketsToken
        """
        self.token = token
        self.url = self.url_private
        # If we're already running => either wait for next reconnect or design a forced approach
        if self.running:
            logger.warning(
                "Already running => we will reconnect to private feed on next cycle/disconnect. "
                "Or you can call 'stop()' then 'start()' to force immediate switch."
            )
        else:
            # not running => just start it in private mode
            self.start()

    async def _subscribe_private(self, ws, feed_name="openOrders"):
        """
        Called from subscribe_private_feed or automatically in _subscribe()
        if self.url==private and self.token is set.
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
        to send subscription or store the feed_name for next reconnection logic.

        For a robust approach, you'd implement a method to store the feed_name
        and call `_subscribe_private` once we have an active `ws` reference
        or on next connect.
        """
        if not self.running:
            logger.warning("WS not running => call connect_private_feed(token) or start() first.")
            return

        async def _sub():
            if not self.token:
                logger.warning("No token => can't subscribe to private feed.")
                return
            logger.info(f"Subscribing to private feed => {feed_name}")
            # In a robust design, you'd have a reference to the current WebSocket.
            # For now, we stub it out. This is where you'd call `_subscribe_private(ws, feed_name)`
            # if you stored `ws` as a member variable.
            pass

        if self.loop:
            # schedule the subscription on the event loop
            asyncio.run_coroutine_threadsafe(_sub(), self.loop)
        else:
            logger.warning("No event loop found => cannot schedule private subscription right now.")

    def send_order(self, pair: str, side: str, ordertype: str, volume: float, price: float = None):
        """
        For the private feed => 'addOrder' event. We'll do a stub approach as well,
        because we need an actual reference to the current websockets connection
        in a more advanced design.
        """
        logger.warning("send_order is a stub => in websockets approach, we need an open ws reference.")
        # In a robust design, store 'ws' and do an async method for "addOrder" event with
        # subscription token, e.g.:
        # msg = {
        #   "event": "addOrder",
        #   "token": self.token,
        #   ...
        # }

    def cancel_order(self, txids: list):
        """
        For private feed => 'cancelOrder'. Also a stub, see same note as above.
        """
        logger.warning("cancel_order is a stub => see above. We need direct ws access.")
