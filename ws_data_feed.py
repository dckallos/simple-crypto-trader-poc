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

ENHANCEMENT (RISK MANAGER INTEGRATION):
   - If you pass a `risk_manager` instance to KrakenPrivateWSClient, we will call
     `risk_manager.create_sub_position(...)` once an order is fully filled.
   - This records an open sub-position in your 'sub_positions' table, in addition
     to recording a final row in 'trades' (for legacy usage).

**UPDATED**:
   - We remove partial fill references to the old 'trades' partial fill logic.
   - We store ephemeral states in 'pending_trades'.
   - Once an order is fully filled (status='closed'), we:
     1) Insert final trade record in 'trades'
     2) Mark the pending trade as closed in 'pending_trades'
     3) (NEW) Optionally create a new sub-position row in 'sub_positions' via risk_manager
   - If an order is rejected or canceled, we mark the pending trade as 'rejected' in DB.
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
    record_trade_in_db,             # for final trade insertion
    mark_pending_trade_open,
    mark_pending_trade_rejected,
    mark_pending_trade_closed,
    create_pending_trade
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
        We do a simplistic approach => store as last price in price_history table.
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
      in addition to ephemeral/final fill updates.
    - (NEW) Provide a 'risk_manager' if you want sub_positions updated automatically.

    Updated logic (no partial fill tracking in 'trades' table):
      * 'pending_trades' is used for ephemeral states
      * final trades are only inserted in 'trades' once we see a fully closed fill
      * if rejected, we mark the pending trade as 'rejected'
      * if closed with fill_size>0, we also call risk_manager.create_sub_position(...)
        to record an open sub-position in 'sub_positions' (if risk_manager is provided).
    """

    def __init__(
        self,
        token: str,
        on_private_event_callback=None,
        risk_manager=None
    ):
        """
        :param token: The token from REST => GetWebSocketsToken
        :param on_private_event_callback: optional function that receives private events
        :param risk_manager: optional instance of RiskManagerDB to create sub-positions
                             upon final fill (e.g. your updated risk_manager.py)
        """
        self.url_private = "wss://ws-auth.kraken.com"
        self.token = token
        self.on_private_event_callback = on_private_event_callback
        self.risk_manager = risk_manager  # <--- NEW

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
        update 'pending_trades' and possibly finalize trades in 'trades'.
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
            if feed_name == "openOrders":
                self._process_open_orders(data[0])

            if self.on_private_event_callback:
                self.on_private_event_callback({"feed_name": feed_name, "data": data})

    def _process_private_order_message(self, msg: dict):
        """
        Attempt to parse 'addOrderStatus' or similar event to handle
        final fill or rejections in 'pending_trades' table. Then if closed,
        create a final record in 'trades'.

        Example:
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
          or
          {
            "event":"addOrderStatus",
            "status":"closed",
            "txid":"OOS2OH-BFAD3-LQ2LUD",
            "vol_exec":"0.04000000","avg_price":"280.47","fee":"0.04488"
          }
        """
        if msg.get("event") != "addOrderStatus":
            return

        txid = msg.get("txid", "")
        status_str = msg.get("status", "")

        if status_str == "error":
            # Rejected => mark pending trade as 'rejected'
            reason = msg.get("errorMessage", "")
            logger.info(f"[PrivateWS] Order rejected => order_id={txid}, reason={reason}")
            mark_pending_trade_rejected_by_kraken_id(txid, reason)
            return

        logger.debug(f"[PrivateWS] addOrderStatus => order_id={txid}, status={status_str}")

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

        if status_str == "closed":
            # Final fill => record trade + mark pending closed
            self._finalize_trade_from_kraken(txid, vol_exec, avg_px, fee_val)
        elif status_str == "ok":
            # Possibly an 'open' order => mark pending as open
            mark_pending_trade_open_by_kraken_id(txid)
        else:
            # "open", "part_filled", or other states
            if status_str == "open":
                mark_pending_trade_open_by_kraken_id(txid)
            # We do not track partial fill. If needed, you can expand logic here.

    def _process_open_orders(self, order_chunk):
        """
        If you subscribe to "openOrders", you'll get a list of orders with fields
        like 'vol_exec','avg_price','fee','status' etc. We do final fill or rejection
        in 'pending_trades' as well. Then if final => record in 'trades' and optionally
        create a sub-position via risk_manager.

        Example data chunk:
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
        """
        if not isinstance(order_chunk, list):
            return

        for item in order_chunk:
            # each item should be a dict with key=order_id
            if not isinstance(item, dict):
                continue

            for txid, order_info in item.items():
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
                    # if vol_exec==0 => fully rejected. else => partial fill then canceled => treat as closed final fill
                    if vol_exec == 0:
                        logger.info(f"[PrivateWS] Order canceled => {txid}, no fill => reject")
                        mark_pending_trade_rejected_by_kraken_id(txid, "canceled")
                    else:
                        logger.info(f"[PrivateWS] Order canceled after partial fill => {txid}")
                        self._finalize_trade_from_kraken(txid, vol_exec, avg_px, fee_val)
                    continue

                if status_str == "closed":
                    logger.info(f"[PrivateWS] Order closed => {txid}, vol_exec={vol_exec}")
                    self._finalize_trade_from_kraken(txid, vol_exec, avg_px, fee_val)
                elif status_str == "open":
                    mark_pending_trade_open_by_kraken_id(txid)
                else:
                    logger.debug(f"[PrivateWS] order_id={txid}, status={status_str}, ignoring beyond logs.")

    def _finalize_trade_from_kraken(self, kraken_order_id: str, filled_size: float, avg_fill_price: float, fee: float):
        """
        Called when an order is definitely 'closed' or canceled-with-partial-fill from Kraken,
        to insert final row in 'trades', mark pending as closed, and (NEW) create a sub-position
        if filled_size>0 and a RiskManager is present.
        """
        # 1) We find the pending trade row by kraken_order_id => see below helper
        row = _fetch_pending_trade_by_kraken_id(kraken_order_id)
        if not row:
            logger.warning(f"[PrivateWS] No pending_trades row found for kraken_order_id={kraken_order_id}")
            return

        pending_id = row[0]
        pair = row[1]
        side = row[2]     # might be 'BUY' or 'SELL'
        requested_qty = row[3]

        # 2) If there's a final fill, record final trade in 'trades' table
        #    and optionally create a sub-position.
        if filled_size > 0:
            # Record final trade
            record_trade_in_db(
                side=side,
                quantity=filled_size,
                price=avg_fill_price,
                order_id=kraken_order_id,
                pair=pair
            )
            logger.info(f"[PrivateWS] Recorded final trade => side={side}, size={filled_size}, px={avg_fill_price}")

            # (NEW) If we have a RiskManager => create sub_position
            if self.risk_manager:
                sub_side = "long" if side.upper() == "BUY" else "short"
                sub_pos_id = self.risk_manager.create_sub_position(
                    pair=pair,
                    side=sub_side,
                    entry_price=avg_fill_price,
                    size=filled_size
                )
                logger.info(f"[PrivateWS] Created sub-position => id={sub_pos_id}, pair={pair}, side={sub_side}")

        else:
            logger.info(f"[PrivateWS] final fill_size=0 => no trade or sub-position creation, pair={pair}, side={side}")

        # 3) Mark the pending trade closed with optional reason
        reason_msg = f"final fill size={filled_size}, fee={fee}"
        mark_pending_trade_closed(pending_id, reason=reason_msg)

    # --------------------------------------------------------------------------
    # send_order / cancel_order for private feed
    # --------------------------------------------------------------------------
    def send_order(self, pair: str, side: str, ordertype: str, volume: float, price: float=None):
        """
        e.g. self.send_order("XBT/USD", "buy", "market", 0.01)
        This results in an 'addOrder' event => 'addOrderStatus' messages indicating success or error.
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


# ------------------------------------------------------------------------------
# HELPER: fetch pending_trades row by kraken_order_id
# ------------------------------------------------------------------------------
def _fetch_pending_trade_by_kraken_id(kraken_order_id: str):
    """
    Returns a tuple (id, pair, side, requested_qty) for the row in 'pending_trades'
    with kraken_order_id=?.
    """
    conn = sqlite3.connect("trades.db")
    try:
        c = conn.cursor()
        c.execute("""
            SELECT id, pair, side, requested_qty
            FROM pending_trades
            WHERE kraken_order_id=?
        """, (kraken_order_id,))
        row = c.fetchone()
        return row
    except Exception as e:
        logger.exception(f"Error fetching pending_trades row by kraken_order_id={kraken_order_id}: {e}")
        return None
    finally:
        conn.close()


def mark_pending_trade_open_by_kraken_id(kraken_order_id: str):
    """
    If an order is accepted or recognized as open, update pending_trades => status='open'.
    We don't do partial fill logic, just a quick status update if we have the row.
    """
    conn = sqlite3.connect("trades.db")
    try:
        c = conn.cursor()
        c.execute("""
            UPDATE pending_trades
            SET status='open'
            WHERE kraken_order_id=?
        """, (kraken_order_id,))
        conn.commit()
        if c.rowcount < 1:
            logger.debug(f"No pending_trades row found with kraken_order_id={kraken_order_id} to mark open.")
    except Exception as e:
        logger.exception(f"Error marking pending trade open by kraken_order_id={kraken_order_id}: {e}")
    finally:
        conn.close()


def mark_pending_trade_rejected_by_kraken_id(kraken_order_id: str, reason: str = None):
    """
    If order is rejected by Kraken, set status='rejected'.
    """
    conn = sqlite3.connect("trades.db")
    try:
        c = conn.cursor()
        c.execute("""
            UPDATE pending_trades
            SET status='rejected', reason=?
            WHERE kraken_order_id=?
        """, (reason, kraken_order_id))
        conn.commit()
        if c.rowcount < 1:
            logger.debug(f"No pending_trades row found with kraken_order_id={kraken_order_id} to reject.")
    except Exception as e:
        logger.exception(f"Error marking pending trade rejected by kraken_order_id={kraken_order_id}: {e}")
    finally:
        conn.close()
