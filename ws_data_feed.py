#!/usr/bin/env python3
# =============================================================================
# FILE: ws_data_feed.py
# =============================================================================
"""
ws_data_feed.py

Implements two separate WebSocket client classes:

1) KrakenPublicWSClient:
   - Connects to wss://ws.kraken.com
   - Subscribes to public market data feeds like "ticker" or "trade"
   - Saves price data in 'price_history' (or calls a callback)

2) KrakenPrivateWSClient:
   - Connects to wss://ws-auth.kraken.com
   - Subscribes to private feeds (e.g. "openOrders", "ownTrades", addOrderStatus, etc.)
   - Can place/cancel orders
   - Requires an API token from Kraken's REST call to GetWebSocketsToken

**ENHANCEMENT**:
   - We now parse certain private feed messages to handle partial fills, final fills, or rejections
     in the 'trades' table. For instance, if an "addOrderStatus" message indicates "error",
     we call `set_trade_rejected(...)`. If it indicates a partial or final fill, we call
     `update_trade_fill(...)`. This ensures your 'trades' table is kept up-to-date with
     actual fill size, fees, and status.

Usage in your main code:
    # A) For public data
    pub_client = KrakenPublicWSClient(
        pairs=["XBT/USD","ETH/USD"],
        feed_type="ticker",             # or "trade"
        on_ticker_callback=some_function,
    )
    pub_client.start()

    # B) For private data
    priv_client = KrakenPrivateWSClient(
        token="YOUR_WEBSOCKETS_TOKEN",
        on_private_event_callback=some_function_for_open_orders
    )
    priv_client.start()

Important: Do **not** attempt to subscribe to public feeds using the private endpoint,
or vice versa. This is why we maintain two different classes.
"""

import ssl
import certifi
import asyncio
import json
import logging
import websockets
import time
import threading
import sqlite3

from db import (
    store_price_history,
    update_trade_fill,
    set_trade_rejected
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_secure_ssl_context() -> ssl.SSLContext:
    """
    Creates an SSL context using certifi's trusted CA certificates.
    Ensures SSLv2 and SSLv3 are disabled, requires certificate validation,
    and checks hostname.
    """
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    context.options |= ssl.OP_NO_SSLv2
    context.options |= ssl.OP_NO_SSLv3
    context.verify_mode = ssl.CERT_REQUIRED
    context.check_hostname = True
    context.load_verify_locations(cafile=certifi.where())
    return context


# ------------------------------------------------------------------------------
# PUBLIC FEED: KrakenPublicWSClient
# ------------------------------------------------------------------------------
class KrakenPublicWSClient:
    """
    A WebSocket client dedicated to PUBLIC feeds from Kraken, connecting
    to wss://ws.kraken.com. It can subscribe to "ticker", "trade", or any
    other public channel that Kraken offers. It does NOT handle private data.

    - Use 'feed_type' = "ticker" or "trade"
    - Provide an on_ticker_callback if you want to do something with the
      real-time ticker updates. (Similarly you can handle 'trade' updates.)
    """

    def __init__(
        self,
        pairs,
        feed_type="ticker",
        on_ticker_callback=None,
    ):
        """
        :param pairs: list of pairs, e.g. ["XBT/USD","ETH/USD"]
        :param feed_type: "ticker" or "trade" or other public feed
        :param on_ticker_callback: function called when a new ticker arrives
        """
        self.url_public = "wss://ws.kraken.com"
        self.pairs = pairs
        self.feed_type = feed_type
        self.on_ticker_callback = on_ticker_callback

        self.ssl_context = create_secure_ssl_context()
        self.loop = None
        self._ws = None
        self.running = False
        self._thread = None

    def start(self):
        """Start the WebSocket client in a background thread."""
        if self.running:
            logger.warning("KrakenPublicWSClient is already running.")
            return
        self.running = True

        self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"KrakenPublicWSClient started (feed_type={self.feed_type}, pairs={self.pairs})."
        )

    def stop(self):
        """Stop the WebSocket client => the event loop ends => the thread joins."""
        if not self.running:
            logger.info("KrakenPublicWSClient is not running or already stopped.")
            return
        self.running = False
        logger.info("KrakenPublicWSClient stop requested.")
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            logger.info("KrakenPublicWSClient background thread joined.")

    def _run_async_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._main_loop())
        except Exception as e:
            logger.exception(f"[KrakenPublicWSClient] Exception => {e}")
        finally:
            logger.info("[KrakenPublicWSClient] Exiting event loop.")
            self.loop.close()

    async def _main_loop(self):
        """Tries to connect => subscribe => consume messages => if disconnected, tries again."""
        while self.running:
            try:
                logger.info(f"[PublicWS] Connecting to {self.url_public} ...")
                async with websockets.connect(self.url_public, ssl=self.ssl_context) as ws:
                    self._ws = ws
                    logger.info(
                        f"[PublicWS] Connected => feed={self.feed_type}, pairs={self.pairs}"
                    )

                    await self._subscribe(ws)
                    await self._consume_messages(ws)

                    self._ws = None
            except Exception as e:
                logger.exception(f"[PublicWS] Error => {e}")
                self._ws = None
                if not self.running:
                    break
                logger.info("[PublicWS] Disconnected => retry in 5s...")
                await asyncio.sleep(5)

        logger.info("[PublicWS] main_loop => done because self.running=False")

    async def _subscribe(self, ws):
        """Subscribes to the public feed (e.g. ticker) for the specified pairs."""
        sub_obj = {"name": self.feed_type}
        msg = {
            "event": "subscribe",
            "pair": self.pairs,
            "subscription": sub_obj
        }
        await ws.send(json.dumps(msg))
        logger.info(f"[PublicWS] Subscribed => feed={self.feed_type}, pairs={self.pairs}")

    async def _consume_messages(self, ws):
        """Receive messages in a loop until we stop or the connection closes."""
        while self.running:
            try:
                message = await ws.recv()
                await self._handle_message(message)
            except websockets.exceptions.ConnectionClosed:
                logger.warning("[PublicWS] Connection closed => break")
                break
            except Exception as e:
                logger.exception(f"[PublicWS] Error receiving => {e}")
                break

    async def _handle_message(self, raw_msg: str):
        """Parses JSON => routes to ticker/trade logic or system events."""
        try:
            data = json.loads(raw_msg)
        except json.JSONDecodeError:
            logger.error(f"[PublicWS] JSON parse error => {raw_msg}")
            return

        # 2 main shapes: dict or list
        if isinstance(data, dict):
            ev = data.get("event")
            if ev in ["subscribe", "subscriptionStatus", "heartbeat", "systemStatus"]:
                logger.debug(f"[PublicWS] System event => {data}")
            else:
                logger.debug(f"[PublicWS] Unrecognized dict => {data}")
            return

        if isinstance(data, list):
            # e.g. [channelID, payload, "ticker", "XBT/USD"]
            if len(data) < 4:
                logger.debug(f"[PublicWS] Short data => {data}")
                return
            feed_name = data[2]
            pair = data[3]
            if feed_name == "ticker":
                await self._handle_ticker(data[1], pair)
            elif feed_name == "trade":
                await self._handle_trade(data[1], pair)
            else:
                logger.debug(f"[PublicWS] Unknown feed={feed_name}, data={data}")
            return

    async def _handle_ticker(self, update_obj, pair: str):
        """Handle a ticker update => store in 'price_history' or call on_ticker_callback."""
        ask_info = update_obj.get("a", [])
        bid_info = update_obj.get("b", [])
        last_info = update_obj.get("c", [])
        vol_info = update_obj.get("v", [])

        ask = float(ask_info[0]) if ask_info else 0.0
        bid = float(bid_info[0]) if bid_info else 0.0
        last_price = float(last_info[0]) if last_info else 0.0
        volume = float(vol_info[0]) if vol_info else 0.0

        store_price_history(
            pair=pair,
            bid=bid,
            ask=ask,
            last=last_price,
            volume=volume
        )

        # If there's a callback => pass the final last_price
        if self.on_ticker_callback and last_price > 0:
            self.on_ticker_callback(pair, last_price)

    async def _handle_trade(self, trades_list, pair: str):
        """
        For 'trade' feed => each item => [price, volume, time, side, orderType, misc].
        We'll store partial or call a callback. We'll do a simplistic approach => store as last price.
        """
        # We'll just store the last price from each trade in price_history. This is naive
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


# ------------------------------------------------------------------------------
# PRIVATE FEED: KrakenPrivateWSClient
# ------------------------------------------------------------------------------
class KrakenPrivateWSClient:
    """
    A WebSocket client dedicated to PRIVATE feeds from Kraken, connecting
    to wss://ws-auth.kraken.com. This allows for "openOrders", "ownTrades",
    "addOrderStatus", etc. We can also place/cancel orders.
    We do NOT handle public feed subscriptions here.

    - Provide 'token' from the REST call to GetWebSocketsToken.
    - Provide an on_private_event_callback if you want custom logic
      in addition to our partial fill / final fill updates.

    We'll parse certain events:
      * "addOrderStatus" => if "error" => set trade rejected; if "closed" => fully filled
                            (and we store final fill in 'trades'), if "open" => partial fill.
      * "openOrders" => can also contain partial fill info (vol_exec, etc.) if you subscribe to it.
    """

    def __init__(
        self,
        token: str,
        on_private_event_callback=None
    ):
        """
        :param token: The token from REST => GetWebSocketsToken
        :param on_private_event_callback: optional function that receives private events
        """
        self.url_private = "wss://ws-auth.kraken.com"
        self.token = token
        self.on_private_event_callback = on_private_event_callback

        self.ssl_context = create_secure_ssl_context()
        self.loop = None
        self._ws = None
        self.running = False
        self._thread = None

    def start(self):
        """Start the WebSocket client in background thread."""
        if self.running:
            logger.warning("KrakenPrivateWSClient is already running.")
            return
        self.running = True

        self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()
        logger.info("KrakenPrivateWSClient started (private feed).")

    def stop(self):
        """Stop the private feed => event loop ends => thread joined."""
        if not self.running:
            logger.info("KrakenPrivateWSClient is not running or already stopped.")
            return
        self.running = False
        logger.info("KrakenPrivateWSClient stop requested.")
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            logger.info("KrakenPrivateWSClient background thread joined.")

    def _run_async_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._main_loop())
        except Exception as e:
            logger.exception(f"[KrakenPrivateWSClient] => {e}")
        finally:
            logger.info("[KrakenPrivateWSClient] Exiting event loop.")
            self.loop.close()

    async def _main_loop(self):
        """Loop that connects to wss://ws-auth.kraken.com => subscribe => consume messages."""
        while self.running:
            try:
                logger.info("[PrivateWS] Connecting to wss://ws-auth.kraken.com ...")
                async with websockets.connect(self.url_private, ssl=self.ssl_context) as ws:
                    self._ws = ws
                    logger.info("[PrivateWS] Connected => Subscribing to openOrders, etc.")
                    await self._subscribe(ws)
                    await self._consume_messages(ws)
                    self._ws = None
            except Exception as e:
                logger.exception(f"[PrivateWS] Error => {e}")
                self._ws = None
                if not self.running:
                    break
                logger.info("[PrivateWS] Disconnected => retry in 5s...")
                await asyncio.sleep(5)

        logger.info("[PrivateWS] main_loop => done => self.running=False")

    async def _subscribe(self, ws):
        """
        Subscribe to private channels. By default => openOrders.
        If you want ownTrades, add it or call subscribe_private_feed with 'ownTrades'.
        """
        if not self.token:
            logger.warning("[PrivateWS] No token => can't subscribe to private feed.")
            return

        msg = {
            "event": "subscribe",
            "subscription": {
                "name": "openOrders",
                "token": self.token
            }
        }
        await ws.send(json.dumps(msg))
        logger.info("[PrivateWS] Subscribing => openOrders channel")

    async def _consume_messages(self, ws):
        while self.running:
            try:
                raw_msg = await ws.recv()
                await self._handle_message(raw_msg)
            except websockets.exceptions.ConnectionClosed:
                logger.warning("[PrivateWS] Connection closed => break")
                break
            except Exception as e:
                logger.exception(f"[PrivateWS] => {e}")
                break

    async def _handle_message(self, raw_msg: str):
        """
        Handle each message from the private feed. We call on_private_event_callback
        if provided. Additionally, we parse "addOrderStatus" or "openOrders" to
        handle partial fills or rejections in our 'trades' table.
        """
        try:
            data = json.loads(raw_msg)
        except json.JSONDecodeError:
            logger.error(f"[PrivateWS] JSON parse error => {raw_msg}")
            return

        # If the message is a dict => check if 'event' is e.g. "addOrderStatus", etc.
        if isinstance(data, dict):
            ev = data.get("event")
            if ev in ["subscriptionStatus", "heartbeat", "systemStatus"]:
                logger.debug(f"[PrivateWS] System event => {data}")
            else:
                # e.g. addOrderStatus, cancelOrderStatus, error, etc.
                logger.debug(f"[PrivateWS] Possibly private system => {data}")
                # attempt to process a recognized structure
                self._process_private_order_message(data)

            if self.on_private_event_callback:
                self.on_private_event_callback(data)
            return

        # If the message is a list => might be [ [some order object], "openOrders", {sequence:..} ]
        if isinstance(data, list):
            if len(data) < 2:
                logger.debug(f"[PrivateWS] short data => {data}")
                return

            feed_name = data[1]
            # If feed_name == "openOrders" => parse partial fill data from data[0]
            if feed_name == "openOrders":
                self._process_open_orders(data[0])

            # user callback
            if self.on_private_event_callback:
                self.on_private_event_callback({"feed_name": feed_name, "data": data})

    def _process_private_order_message(self, msg: dict):
        """
        Attempt to parse an 'addOrderStatus' or similar event to handle partial/final fill or rejections.
        For example:
          {
            "event": "addOrderStatus",
            "status": "error",
            "errorMessage": "EGeneral:Invalid arguments:volume minimum not met",
            "txid": "OOS2OH-BFAD3-LQ2LUD"
          }

          or

          {
            "event":"addOrderStatus",
            "status":"ok",
            "txid":"OOS2OH-BFAD3-LQ2LUD",
            "descr":"buy 0.04000000 AAVEUSD @ market"
          }

          or final data might also be included. If "status":"closed", you might see
          "vol_exec":"0.04000000","fee":"0.04","avg_price":"280.47"
        """
        if msg.get("event") != "addOrderStatus":
            return

        order_id = msg.get("txid", "")
        status_str = msg.get("status", "")

        # if "error" => the 'errorMessage' might have the reason
        if status_str == "error":
            # rejected
            logger.info(f"[PrivateWS] Order rejected => order_id={order_id}, reason={msg.get('errorMessage','')}")
            set_trade_rejected(order_id)
            return

        # if "ok" => means order is accepted or open; we might not do anything yet
        # We'll look for partial or final fills in the separate feed "openOrders" or
        # we might see them here if 'vol_exec' is included.
        logger.debug(f"[PrivateWS] addOrderStatus => order_id={order_id}, status={status_str}")

        # If the order is immediately closed or partially filled, we might get
        # 'vol_exec','avg_price','fee' here. Example:
        # { "status":"closed","vol_exec":"0.04000000","avg_price":"280.47000", "fee":"0.04488" }
        # But often that arrives in openOrders feed. We'll parse it here if present:
        vol_exec_str = msg.get("vol_exec", "0.0")
        avg_price_str = msg.get("avg_price", "0.0")
        fee_str = msg.get("fee", "0.0")

        try:
            vol_exec = float(vol_exec_str)
            avg_px = float(avg_price_str)
            fee_val = float(fee_str)
        except:
            vol_exec = 0.0
            avg_px = 0.0
            fee_val = 0.0

        # we can define an internal status for partial fill vs. closed
        # e.g. "closed" if vol_exec > 0 and status_str == "closed"
        # or "part_filled" if vol_exec > 0 but status='open'
        # For minimal approach: if status='closed', we do "closed",
        # otherwise if vol_exec>0 => "part_filled", else "open".
        if status_str == "closed":
            new_status = "closed"
        elif vol_exec > 0:
            new_status = "part_filled"
        else:
            new_status = "open"

        if order_id:
            update_trade_fill(
                order_id=order_id,
                filled_size=vol_exec,
                avg_fill_price=avg_px,
                fee=fee_val,
                status=new_status
            )

    def _process_open_orders(self, order_chunk):
        """
        If you subscribe to "openOrders", you'll get a list of orders with fields
        like 'vol_exec', 'avg_price', 'status', 'cost', 'fee', etc.

        Example data structure:
        [
          {
            "OOS2OH-BFAD3-LQ2LUD": {
              "vol_exec":"0.01000000",
              "cost":"12.45",
              "fee":"0.04980",
              "avg_price":"1245.0",
              "status":"open" or "closed" or "canceled",
              ...
            }
          }
        ]
        Then there's an array like ["openOrders", {sequence:...}] afterward.

        We'll parse each order in the chunk => update 'trades' table partial/final fill if present.
        """
        if not isinstance(order_chunk, list):
            return

        for item in order_chunk:
            # each item should be a dict with key=order_id
            if not isinstance(item, dict):
                continue
            for order_id, order_info in item.items():
                # parse fields
                status_str = order_info.get("status", "")
                vol_exec_str = order_info.get("vol_exec", "0.0")
                avg_price_str = order_info.get("avg_price", "0.0")
                fee_str = order_info.get("fee", "0.0")

                try:
                    vol_exec = float(vol_exec_str)
                    avg_px = float(avg_price_str)
                    fee_val = float(fee_str)
                except:
                    vol_exec = 0.0
                    avg_px = 0.0
                    fee_val = 0.0

                if status_str == "canceled":
                    # treat as rejected or closed with vol_exec partial?
                    # For simplicity, let's treat it as 'rejected' if vol_exec=0 else 'closed'
                    if vol_exec == 0:
                        logger.info(f"[PrivateWS] Order canceled => {order_id}")
                        set_trade_rejected(order_id)
                        continue
                    else:
                        # partial fill then canceled => set final fill
                        logger.info(f"[PrivateWS] Order canceled after partial fill => {order_id}")
                        update_trade_fill(
                            order_id=order_id,
                            filled_size=vol_exec,
                            avg_fill_price=avg_px,
                            fee=fee_val,
                            status="closed"
                        )
                        continue

                # possibly "open", "closed", or other
                if status_str == "open":
                    if vol_exec > 0:
                        new_status = "part_filled"
                    else:
                        new_status = "open"
                elif status_str == "closed":
                    new_status = "closed"
                else:
                    new_status = status_str  # e.g. "expired", "canceled"

                update_trade_fill(
                    order_id=order_id,
                    filled_size=vol_exec,
                    avg_fill_price=avg_px,
                    fee=fee_val,
                    status=new_status
                )

    # --------------------------------------------------------------------------
    # send_order / cancel_order for private feed
    # --------------------------------------------------------------------------
    def send_order(self, pair: str, side: str, ordertype: str, volume: float, price: float=None):
        """
        e.g. self.send_order("XBT/USD", "buy", "market", 0.01)
        This results in an 'addOrder' event => you get 'addOrderStatus' messages indicating success or error.
        """
        if not self.token:
            logger.warning("[PrivateWS] No token => cannot place order.")
            return
        if not self.running or not self._ws:
            logger.warning("[PrivateWS] Not connected => cannot place order.")
            return

        async def _send():
            msg = {
                "event": "addOrder",
                "token": self.token,
                "pair": pair,
                "ordertype": ordertype,
                "type": side,
                "volume": str(volume),
            }
            if price is not None:
                msg["price"] = str(price)
            logger.info(f"[PrivateWS] Sending addOrder => {msg}")
            await self._ws.send(json.dumps(msg))

        if self.loop:
            asyncio.run_coroutine_threadsafe(_send(), self.loop)

    def cancel_order(self, txids: list):
        """Cancel orders => e.g. self.cancel_order(["O6S6CF-ABC123-XYZ987"])"""
        if not self.token:
            logger.warning("[PrivateWS] No token => cannot cancel order.")
            return
        if not self.running or not self._ws:
            logger.warning("[PrivateWS] Not connected => cannot cancel order.")
            return

        async def _cancel():
            msg = {
                "event":"cancelOrder",
                "token": self.token,
                "txid": txids
            }
            logger.info(f"[PrivateWS] Sending cancelOrder => {msg}")
            await self._ws.send(json.dumps(msg))

        if self.loop:
            asyncio.run_coroutine_threadsafe(_cancel(), self.loop)
