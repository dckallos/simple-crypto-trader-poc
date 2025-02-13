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
   - Requires an API token from Kraken’s REST call to GetWebSocketsToken
   - NEW: We also subscribe to the "ownTrades" channel to receive fill reports
     (past 50 trades + real-time).
   - On each fill, we insert a row in the 'trades' table if it's new, and optionally
     update sub_positions if a risk_manager is provided.

**Usage**:
   - Create an instance of KrakenPrivateWSClient(token=your_token, risk_manager=...)
   - The client automatically subscribes to "ownTrades" and "openOrders" upon connection.
   - If you want to see older trades in your DB, set snapshot=True so the last 50 trades
     come in as a snapshot on connect.
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

        if isinstance(data, dict):
            # subscriptionStatus, systemStatus, heartbeat, etc.
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
    "addOrderStatus", etc. We do NOT handle public feed subscriptions here.

    We subscribe to:
      - openOrders (optional, for pending/filled order states)
      - ownTrades for final fill confirmations (this is the
        new approach to record trades reliably in the DB).

    On each ownTrades update, we parse the trade(s) and store them in the
    'trades' table. If a risk_manager is provided, we also update sub_positions.
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
        :param risk_manager: optional instance of RiskManagerDB to create/update sub-positions
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

        # 1) openOrders subscription (optional, for reference)
        msg_open_orders = {
            "event": "subscribe",
            "subscription": {
                "name": "openOrders",
                "token": self.token
            }
        }
        await ws.send(json.dumps(msg_open_orders))
        logger.info("[PrivateWS] Subscribing => openOrders channel")

        # 2) ownTrades subscription (the recommended approach to see all fills)
        msg_own_trades = {
            "event": "subscribe",
            "subscription": {
                "name": "ownTrades",
                "token": self.token,
                "snapshot": True,              # get the last 50 trades
                "consolidate_taker": True      # merges taker fills
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

        # If the message is a list => could be [ [some trade data], "ownTrades", {...} ]
        if isinstance(data, list):
            if len(data) < 2:
                logger.debug(f"[PrivateWS] short data => {data}")
                return

            feed_name = data[1]
            if feed_name == "openOrders":
                # data[0] => list of order objects
                self._process_open_orders(data[0])
            elif feed_name == "ownTrades":
                # data[0] => list of trade dictionaries
                self._process_own_trades(data[0])
            else:
                logger.debug(f"[PrivateWS] unknown feed={feed_name}, data={data}")

            if self.on_private_event_callback:
                self.on_private_event_callback({"feed_name": feed_name, "data": data})

    def _process_private_order_message(self, msg: dict):
        """
        Attempt to parse 'addOrderStatus' or other order messages to handle
        new or canceled orders in 'pending_trades'.
        We do NOT record final trades here anymore, since ownTrades is the new source.
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
                # partial fill => the fill is handled by ownTrades, so we just finalize the pending trade
                self._finalize_trade_from_kraken(txid, vol_exec, avg_price, fee_val)
        elif status_str == "closed":
            logger.info(f"[PrivateWS] Order closed => {txid}")
            vol_exec = float(msg.get("vol_exec", "0.0"))
            avg_price = float(msg.get("avg_price", "0.0"))
            fee_val = float(msg.get("fee", "0.0"))
            # the real fill insertion is done by ownTrades => here we only finalize pending trade
            self._finalize_trade_from_kraken(txid, vol_exec, avg_price, fee_val)
        elif status_str == "ok":
            # Possibly means order is accepted. If userref => set kraken_order_id
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
                        # partial fill => ownTrades has it, we just close pending
                        self._finalize_trade_from_kraken(txid, vol_exec, 0.0, 0.0)
                elif status_str == "closed":
                    self._finalize_trade_from_kraken(txid, vol_exec, 0.0, 0.0)
                elif status_str == "open":
                    mark_pending_trade_open_by_kraken_id(txid)
                else:
                    logger.debug(f"[PrivateWS] order_id={txid}, status={status_str}, ignoring.")

    def _process_own_trades(self, trades_chunk):
        """
        This is the new approach for final fills: each item in trades_chunk
        is a dict of { <trade_id>: {...fields...} }. We parse each trade, store
        it in 'trades' if it's new, and optionally update sub_positions.
        """
        if not isinstance(trades_chunk, list):
            return

        for item in trades_chunk:
            # item => { "TDLH43-DVQXD-2KHVYY": {...}, "TDLH43-DVQXD-2KHVYY": {...} }
            if not isinstance(item, dict):
                continue

            for trade_id, trade_obj in item.items():
                self._store_own_trade_if_new(trade_id, trade_obj)

    def _store_own_trade_if_new(self, trade_id: str, trade_obj: dict):
        """
        Insert or skip if we already stored this trade_id. The 'trades' table
        stores final fills with columns:
          - timestamp
          - pair
          - side => 'BUY' or 'SELL'
          - quantity => from 'vol'
          - price => from 'price'
          - order_id => set to trade_id (unique fill ID)
          - fee => from 'fee'

        Then optionally call risk_manager to open or close sub-positions.
        """
        # 1) check if this trade is already in 'trades'
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
                # we already recorded this fill
                return

            # parse fields from trade_obj
            side = trade_obj.get("type", "").upper()  # "BUY" or "SELL"
            price_str = trade_obj.get("price", "0.0")
            vol_str = trade_obj.get("vol", "0.0")
            fee_str = trade_obj.get("fee", "0.0")
            pair = trade_obj.get("pair", "UNKNOWN")
            cost_str = trade_obj.get("cost", "0.0")

            # time => float string => convert to int
            time_f = float(trade_obj.get("time", "0"))
            ts = int(time_f)

            # convert numeric fields
            quantity = float(vol_str)
            fill_price = float(price_str)
            fill_fee = float(fee_str)

            # 2) insert a new row in trades
            c.execute("""
                INSERT INTO trades
                  (timestamp, pair, side, quantity, price, order_id, fee)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                ts, pair, side, quantity, fill_price, trade_id, fill_fee
            ))
            conn.commit()
            logger.info(f"[PrivateWS] Inserted new fill => trade_id={trade_id}, side={side}, vol={quantity}, px={fill_price}")
        except Exception as e:
            logger.exception(f"[PrivateWS] Error inserting ownTrades => {e}")
        finally:
            conn.close()

        # 3) If risk_manager => update sub_positions
        if self.risk_manager:
            self._update_sub_positions_with_trade(pair, side, fill_price, quantity)

    def _update_sub_positions_with_trade(self, pair: str, side: str, fill_price: float, size: float):
        """
        A naive example approach to sub-positions:
          - If side=BUY => create a new sub-position
          - If side=SELL => close the entire position if it exists
        Adjust for partial closes or short logic as needed.
        """
        if side == "BUY":
            # create a sub-position
            pos_id = self.risk_manager.create_sub_position(pair, "long", fill_price, size)
            logger.info(f"[PrivateWS] sub_positions => Created new position => id={pos_id}, pair={pair}, side=long")
        else:
            # SELL => find any open sub-position => close it
            # This is extremely naive (closing at the same size as the new trade).
            # In reality you might do partial close logic if `size` < open size.
            open_positions = self.risk_manager.get_open_positions_for_pair(pair)
            if not open_positions:
                logger.info(f"[PrivateWS] sub_positions => No open position to close for pair={pair}. (Skipping)")
                return

            # We'll just close the first open position, or close them all
            # for pos in open_positions: ...
            pos = open_positions[0]
            pos_id = pos["id"]
            entry_price = pos["entry_price"]
            entry_size = pos["size"]

            # realized PnL => (exit_price - entry_price)*size if side=long
            # but we pass it to risk_manager which might do the final calculation
            # We'll do a quick example:
            realized_pnl = (fill_price - entry_price) * entry_size

            self.risk_manager.close_sub_position(pos_id, fill_price, realized_pnl)
            logger.info(f"[PrivateWS] sub_positions => closed pos_id={pos_id}, realized_pnl={realized_pnl:.4f}")


    # --------------------------------------------------------------------------
    # Helper: finalize trade from the "addOrderStatus"/openOrders approach
    # (less relevant now that ownTrades is used for final fill.)
    # --------------------------------------------------------------------------
    def _finalize_trade_from_kraken(self, kraken_order_id: str, filled_size: float, avg_fill_price: float, fee: float):
        """
        Updated approach:
        - We DO NOT call record_trade_in_db here anymore, because ownTrades will handle
          final fill insertion. Instead, we only mark the pending_trades as 'closed'
          and optionally do sub-position logic for canceled partial fills if you prefer.
        """
        row = _fetch_pending_trade_by_kraken_id(kraken_order_id)
        if not row:
            logger.warning(f"[PrivateWS] No pending_trades row found for kraken_order_id={kraken_order_id}")
            return

        pending_id = row[0]
        pair = row[1]
        side = row[2]
        requested_qty = row[3]

        # If we got here because the order was canceled or expired with partial fill,
        # the actual fill is handled by the ownTrades feed. So do NOT record a trade row here.
        # We only finalize the 'pending_trades' row:
        reason_msg = f"final fill size={filled_size}, fee={fee}"
        mark_pending_trade_closed(pending_id, reason=reason_msg)
        logger.info(f"[PrivateWS] Marked pending_id={pending_id} as closed. side={side}, partial_fill={filled_size}")

        # If you want to do partial sub-position logic, do so here. Typically ownTrades does it.
        # For example:
        # if filled_size > 0 and self.risk_manager:
        #     # Maybe we do partial close logic or a fallback. Usually ownTrades is the single source now.
        pass


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
    We don't do partial fill logic here, just a quick status update.
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
