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
   - Subscribes to private feeds ("openOrders", "ownTrades", etc.)
   - Can place/cancel orders
   - Requires an API token from Kraken’s REST call to GetWebSocketsToken
   - We store final fills in the 'trades' table. Sub-position logic has been
     removed/commented out; daily drawdown or realized PnL is now calculated
     purely from the trades table (or your own risk_manager logic).

Usage:
   - Create an instance of KrakenPrivateWSClient(token=..., risk_manager=...)
   - The client automatically subscribes to "ownTrades" and "openOrders"
     upon connection for real-time fill updates and open order status changes.
   - Completed fills are recorded into 'trades'. We no longer do local sub-positions.

Updates in This Version:
   - Removed calls to create_sub_position(...) and close_sub_position(...).
   - We still keep references to the local DB for 'trades' and 'pending_trades'.
   - The 'risk_manager' can remain if you use it for daily drawdown, but no
     sub-positions or partial fill logic is performed here.
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
    set_kraken_order_id_for_pending_trade,
    record_trade_in_db,
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
    to wss://ws.kraken.com. It can subscribe to "ticker", "trade" or
    any other public channel. It does NOT handle private data.

    If you supply:
      self.risk_manager = <RiskManagerDB instance>
      self.kraken_balances = <dict of real-time balances>

    then on each ticker update we call:
      self.risk_manager.on_price_update(
          pair=pair, current_price=last_price,
          kraken_balances=self.kraken_balances, ...
      )
    so that stop-loss / take-profit checks happen continuously.
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

        # Possibly set these externally in your main:
        #   self.risk_manager = <RiskManagerDB>
        #   self.kraken_balances = <dict>
        self.risk_manager = None
        self.kraken_balances = None

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

        if isinstance(data, dict):
            # subscriptionStatus, systemStatus, heartbeat, etc.
            ev = data.get("event")
            if ev in ["subscribe", "subscriptionStatus", "heartbeat", "systemStatus"]:
                logger.debug(f"[PublicWS] System event => {data}")
            else:
                logger.debug(f"[PublicWS] Unrecognized dict => {data}")
            return

        # Typically => [channelID, payload, "ticker"|"trade", "XBT/USD"]
        if isinstance(data, list):
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
        """
        Process each new ticker update => store in price_history, then
        (optionally) call risk_manager.on_price_update(...).
        """
        ask_info = update_obj.get("a", [])
        bid_info = update_obj.get("b", [])
        last_info = update_obj.get("c", [])
        vol_info = update_obj.get("v", [])

        ask = float(ask_info[0]) if ask_info else 0.0
        bid = float(bid_info[0]) if bid_info else 0.0
        last_price = float(last_info[0]) if last_info else 0.0
        volume = float(vol_info[0]) if vol_info else 0.0

        # Store in DB for reference
        store_price_history(pair=pair, bid=bid, ask=ask, last=last_price, volume=volume)

        # If there's a user callback => pass last_price
        if self.on_ticker_callback and last_price > 0:
            self.on_ticker_callback(pair, last_price)

        # Check stop-loss / take-profit on every ticker update if we have a risk_manager & balances
        from config_loader import ConfigLoader
        if self.risk_manager and self.kraken_balances and last_price > 0:
            self.risk_manager.on_price_update(
                pair=pair,
                current_price=last_price,
                kraken_balances=self.kraken_balances,
                stop_loss_pct=ConfigLoader.get_value("stop_loss_percent"),
                take_profit_pct=ConfigLoader.get_value("take_profit_percent")
            )

    async def _handle_trade(self, trades_list, pair: str):
        """
        For 'trade' feed => each item => [price, volume, time, side, orderType, misc].
        We'll store these in price_history as last_price for demonstration.
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
    "addOrderStatus", etc. We do NOT handle public feed subscriptions here.

    We subscribe to:
      - openOrders (optional, for reference)
      - ownTrades (for final fill confirmations, which we store in the 'trades' table).
    The local sub-position logic is removed. We rely on final fill records in 'trades'
    for PnL or daily drawdown checks (via your RiskManager).
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
        :param risk_manager: optional instance of RiskManagerDB for daily drawdown or
                             other checks (no sub-position logic here).
        """
        self.url_private = "wss://ws-auth.kraken.com"
        self.token = token
        self.on_private_event_callback = on_private_event_callback
        self.risk_manager = risk_manager

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
                    logger.info("[PrivateWS] Connected => Subscribing to private channels.")
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
        Subscribe to private channels: openOrders + ownTrades, both requiring the same token.
        """
        if not self.token:
            logger.warning("[PrivateWS] No token => can't subscribe to private feed.")
            return

        # 1) openOrders subscription
        msg_open_orders = {
            "event": "subscribe",
            "subscription": {
                "name": "openOrders",
                "token": self.token
            }
        }
        await ws.send(json.dumps(msg_open_orders))
        logger.info("[PrivateWS] Subscribing => openOrders channel")

        # 2) ownTrades subscription
        msg_own_trades = {
            "event": "subscribe",
            "subscription": {
                "name": "ownTrades",
                "token": self.token,
                "snapshot": True,           # get the last 50 trades
                "consolidate_taker": True   # merges taker fills
            }
        }
        await ws.send(json.dumps(msg_own_trades))
        logger.info("[PrivateWS] Subscribing => ownTrades channel")

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
        Handle each message from the private feed. This includes:
          - openOrders updates
          - ownTrades updates
          - addOrderStatus, cancelOrderStatus, errors, etc.
        """
        try:
            data = json.loads(raw_msg)
        except json.JSONDecodeError:
            logger.error(f"[PrivateWS] JSON parse error => {raw_msg}")
            return

        if isinstance(data, dict):
            # Typically 'event' => subscriptionStatus, heartbeat, systemStatus,
            # or addOrderStatus, cancelOrderStatus, error, etc.
            ev = data.get("event")
            if ev in ["subscriptionStatus", "heartbeat", "systemStatus"]:
                logger.debug(f"[PrivateWS] System event => {data}")
            else:
                # e.g. addOrderStatus => handle acceptance or error
                self._process_private_order_message(data)

            if self.on_private_event_callback:
                self.on_private_event_callback(data)
            return

        # If the message is a list => e.g. [ [some data], "ownTrades", {...} ]
        if isinstance(data, list):
            if len(data) < 2:
                logger.debug(f"[PrivateWS] short data => {data}")
                return

            feed_name = data[1]
            if feed_name == "openOrders":
                self._process_open_orders(data[0])
            elif feed_name == "ownTrades":
                self._process_own_trades(data[0])
            else:
                logger.debug(f"[PrivateWS] unknown feed={feed_name}, data={data}")

            if self.on_private_event_callback:
                self.on_private_event_callback({"feed_name": feed_name, "data": data})

    def _process_private_order_message(self, msg: dict):
        """
        Attempt to parse 'addOrderStatus' or other order messages to handle
        new or canceled orders in 'pending_trades'. We do NOT record final trades
        here (that's in ownTrades). We only finalize pending states.
        """
        ev = msg.get("event", "")
        if ev != "addOrderStatus":
            return

        txid = msg.get("txid", "")
        status_str = msg.get("status", "")
        userref = msg.get("userref")
        error_msg = msg.get("errorMessage", "")

        if status_str == "error":
            logger.info(f"[PrivateWS] Order rejected => {txid}, reason={error_msg}")
            mark_pending_trade_rejected_by_kraken_id(txid, error_msg)

        elif status_str in ["canceled", "expired"]:
            logger.info(f"[PrivateWS] Order canceled/expired => {txid}, reason={status_str}")
            vol_exec = float(msg.get("vol_exec", "0.0"))
            avg_price = float(msg.get("avg_price", "0.0"))
            fee_val = float(msg.get("fee", "0.0"))

            if vol_exec == 0:
                # Fully unfilled => treat as rejected
                mark_pending_trade_rejected_by_kraken_id(txid, status_str)
            else:
                # partial fill => ownTrades feed records fill => close pending
                self._finalize_trade_from_kraken(txid, vol_exec, avg_price, fee_val)

        elif status_str == "closed":
            logger.info(f"[PrivateWS] Order closed => {txid}")
            vol_exec = float(msg.get("vol_exec", "0.0"))
            avg_price = float(msg.get("avg_price", "0.0"))
            fee_val = float(msg.get("fee", "0.0"))
            # The real fill insertion is done by ownTrades => here we finalize pending
            self._finalize_trade_from_kraken(txid, vol_exec, avg_price, fee_val)

        elif status_str == "ok":
            # Means order is accepted. If userref => set kraken_order_id
            if userref:
                set_kraken_order_id_for_pending_trade(int(userref), txid)
            mark_pending_trade_open_by_kraken_id(txid)
        else:
            logger.debug(f"[PrivateWS] addOrderStatus => {txid}, status={status_str}")

    def _process_open_orders(self, order_chunk):
        """
        If you subscribe to "openOrders", you'll get an array: [ { orderID: {...}}, ... ]
        We only track 'pending_trades' status, not final trade logic.
        """
        if not isinstance(order_chunk, list):
            return

        for item in order_chunk:
            if not isinstance(item, dict):
                continue
            for txid, order_info in item.items():
                status_str = order_info.get("status", "")
                vol_exec_str = order_info.get("vol_exec", "0.0")
                vol_exec = float(vol_exec_str)

                if status_str in ("canceled", "expired"):
                    if vol_exec == 0:
                        mark_pending_trade_rejected_by_kraken_id(txid, status_str)
                    else:
                        # partial fill => ownTrades feed has it => close pending
                        self._finalize_trade_from_kraken(txid, vol_exec, 0.0, 0.0)
                elif status_str == "closed":
                    self._finalize_trade_from_kraken(txid, vol_exec, 0.0, 0.0)
                elif status_str == "open":
                    mark_pending_trade_open_by_kraken_id(txid)
                else:
                    logger.debug(f"[PrivateWS] order_id={txid}, status={status_str}, ignoring.")

    def _process_own_trades(self, trades_chunk):
        """
        This is the feed for final fills: each item in trades_chunk
        is a dict of {<trade_id>: {...fields...}}. We parse each trade, store
        it in 'trades' if it's new.
        """
        if not isinstance(trades_chunk, list):
            return

        for item in trades_chunk:
            if not isinstance(item, dict):
                continue

            for trade_id, trade_obj in item.items():
                self._store_own_trade_if_new(trade_id, trade_obj)

    def _store_own_trade_if_new(self, trade_id: str, trade_obj: dict):
        """
        Insert or skip if we already stored this trade_id in 'trades'.
        We do not call local sub-positions logic anymore; the risk manager
        can read these final fills from the trades table for daily PnL.
        """
        conn = sqlite3.connect("trades.db")
        try:
            c = conn.cursor()
            c.execute("""
                SELECT id FROM trades
                WHERE order_id=?
                LIMIT 1
            """, (trade_id,))
            row = c.fetchone()
            if row:
                # already recorded
                return

            side = trade_obj.get("type", "").upper()  # "BUY" or "SELL"
            price_str = trade_obj.get("price", "0.0")
            vol_str = trade_obj.get("vol", "0.0")
            fee_str = trade_obj.get("fee", "0.0")
            pair = trade_obj.get("pair", "UNKNOWN")

            time_f = float(trade_obj.get("time", "0"))
            ts = int(time_f)

            quantity = float(vol_str)
            fill_price = float(price_str)
            fill_fee = float(fee_str)

            # Insert new row into 'trades'
            c.execute("""
                INSERT INTO trades (
                    timestamp,
                    pair,
                    side,
                    quantity,
                    price,
                    order_id,
                    fee,
                    realized_pnl
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
            """, (
                ts, pair, side, quantity, fill_price, trade_id, fill_fee
            ))
            conn.commit()
            logger.info(
                f"[PrivateWS] Inserted new fill => trade_id={trade_id}, side={side}, "
                f"vol={quantity}, px={fill_price}"
            )

        except Exception as e:
            logger.exception(f"[PrivateWS] Error inserting ownTrades => {e}")
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # send_order / cancel_order for private feed
    # --------------------------------------------------------------------------
    def send_order(
            self,
            pair: str,
            side: str,
            ordertype: str,
            volume: float,
            price: float = None,
            userref: int = None
    ):
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
            if userref is not None:
                msg["userref"] = userref
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
                "event": "cancelOrder",
                "token": self.token,
                "txid": txids
            }
            logger.info(f"[PrivateWS] Sending cancelOrder => {msg}")
            await self._ws.send(json.dumps(msg))

        if self.loop:
            asyncio.run_coroutine_threadsafe(_cancel(), self.loop)

    def _finalize_trade_from_kraken(self, kraken_order_id: str, filled_size: float, avg_fill_price: float, fee: float):
        """
        Called from openOrders or addOrderStatus for partial/canceled orders,
        to finalize the 'pending_trades' row. The actual fill details come via
        ownTrades => we do not track local partial positions.
        """
        row = _fetch_pending_trade_by_kraken_id(kraken_order_id)
        if not row:
            logger.warning(f"[PrivateWS] No pending_trades row found for kraken_order_id={kraken_order_id}")
            return

        pending_id = row[0]
        pair = row[1]
        side = row[2]
        requested_qty = row[3]

        reason_msg = f"final fill size={filled_size}, fee={fee}"
        mark_pending_trade_closed(pending_id, reason=reason_msg)
        logger.info(
            f"[PrivateWS] Marked pending_id={pending_id} as closed => side={side}, partial_fill={filled_size}"
        )

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
