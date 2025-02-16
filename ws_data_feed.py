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
   - Saves price data in 'price_history' via store_price_history(...).
   - (OPTIONALLY) could call your risk_manager.on_price_update(...), but
     we're removing that direct call now so your RiskManager can derive
     prices from the DB on its own schedule.

2) KrakenPrivateWSClient:
   - Connects to wss://ws-auth.kraken.com
   - Subscribes to private feeds ("openOrders", "ownTrades", etc.)
   - Can place/cancel orders
   - Requires an API token from Kraken’s REST call to GetWebSocketsToken
   - We record final fills in the 'trades' table (via ownTrades). We do not
     maintain sub-positions here. Daily PnL or stop-loss is delegated to
     your RiskManager or aggregator logic.

Usage:
   - For public feed:
       public_ws = KrakenPublicWSClient(pairs=["ETH/USD","XBT/USD"], feed_type="ticker")
       public_ws.start()
   - For private feed:
       private_ws = KrakenPrivateWSClient(token=..., risk_manager=...)
       private_ws.start()

Db interactions:
   - price_history: store real-time ticker/trade updates
   - pending_trades/trades: final order fills and ephemeral states
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
    A WebSocket client dedicated to PUBLIC feeds from Kraken (wss://ws.kraken.com).
    Subscribes to "ticker" or "trade" channels and continuously receives live market data.

    Key responsibilities:
      - Maintaining a connection to Kraken's public feed(s).
      - Storing the incoming data to 'price_history' table (via store_price_history).
      - Optionally calling a callback (like on_ticker_callback) for user logic.

    In this version, we do *not* call risk_manager.on_price_update() from here,
    because we've moved that logic into the RiskManager's own background DB cycle.
    """

    def __init__(
        self,
        pairs,
        feed_type="ticker",
        on_ticker_callback=None
    ):
        """
        :param pairs: list of pairs, e.g. ["XBT/USD","ETH/USD"]
        :param feed_type: e.g. "ticker" or "trade"
        :param on_ticker_callback: optional function called when a new ticker arrives
        """
        self.url_public = "wss://ws.kraken.com"
        self.pairs = pairs
        self.feed_type = feed_type
        self.on_ticker_callback = on_ticker_callback

        # If you wanted risk_manager references, you can set them externally.
        self.risk_manager = None
        self.kraken_balances = None

        self.ssl_context = create_secure_ssl_context()
        self.loop = None
        self._ws = None
        self.running = False
        self._thread = None

    def start(self):
        """
        Start the WebSocket client in a background thread. Once started, it attempts to
        connect, subscribe, and process messages from Kraken until 'stop()' is called.
        """
        if self.running:
            logger.warning("KrakenPublicWSClient is already running => skipping start.")
            return
        self.running = True

        self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"[KrakenPublicWSClient] Started => feed_type={self.feed_type}, pairs={self.pairs}"
        )

    def stop(self):
        """
        Stop the WebSocket client => signals the main loop to exit,
        then joins the background thread.
        """
        if not self.running:
            logger.info("[KrakenPublicWSClient] Not running or already stopped.")
            return
        self.running = False
        logger.info("[KrakenPublicWSClient] stop requested. Cleaning up...")
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            logger.info("[KrakenPublicWSClient] background thread joined.")

    def _run_async_loop(self):
        """
        Private method that runs in the background thread. Creates an asyncio loop,
        attempts to connect, and consumes messages until 'stop()' is signaled.
        """
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._main_loop())
        except Exception as e:
            logger.exception(f"[KrakenPublicWSClient] run_async_loop => {e}")
        finally:
            logger.info("[KrakenPublicWSClient] Exiting event loop.")
            self.loop.close()

    async def _main_loop(self):
        """
        The main async routine:
          - connect to wss://ws.kraken.com
          - subscribe to your chosen channels
          - read messages
          - if disconnected, retry after 5s
        """
        while self.running:
            try:
                logger.info(f"[PublicWS] Connecting => {self.url_public}")
                async with websockets.connect(self.url_public, ssl=self.ssl_context) as ws:
                    self._ws = ws
                    logger.info(f"[PublicWS] Connected => feed={self.feed_type}, pairs={self.pairs}")

                    # subscribe to the channels
                    await self._subscribe(ws)
                    # read messages until a disconnection or stop
                    await self._consume_messages(ws)
                    # if we exit the consume loop, reset self._ws
                    self._ws = None

            except Exception as e:
                logger.exception(f"[PublicWS] Connection error => {e}")
                self._ws = None
                if not self.running:
                    break
                logger.info("[PublicWS] Will retry in 5 seconds...")
                await asyncio.sleep(5)

        logger.info("[PublicWS] main_loop => done => self.running=False => exit")

    async def _subscribe(self, ws):
        """
        Subscribe to a particular feed_type ("ticker" or "trade") for your pairs.
        """
        sub_data = {
            "event": "subscribe",
            "pair": self.pairs,
            "subscription": {"name": self.feed_type}
        }
        await ws.send(json.dumps(sub_data))
        logger.info(f"[PublicWS] Subscribed => feed={self.feed_type}, pairs={self.pairs}")

    async def _consume_messages(self, ws):
        """
        Continuously reads messages from the websocket until we stop or the connection closes.
        """
        while self.running:
            try:
                msg = await ws.recv()
                await self._handle_message(msg)
            except websockets.exceptions.ConnectionClosed:
                logger.warning("[PublicWS] Connection closed => break out of consume loop")
                break
            except Exception as ex:
                logger.exception(f"[PublicWS] Error in consume => {ex}")
                break

    async def _handle_message(self, raw_msg: str):
        """
        Parse the incoming JSON. If it's a system-level event, handle that.
        Otherwise, route 'ticker' or 'trade' data to the relevant method.
        """
        try:
            data = json.loads(raw_msg)
        except json.JSONDecodeError:
            logger.error(f"[PublicWS] JSON parse error => {raw_msg}")
            return

        if isinstance(data, dict):
            # Might be "subscriptionStatus", "systemStatus", "heartbeat", etc.
            event = data.get("event")
            if event in ["subscribe", "subscriptionStatus", "heartbeat", "systemStatus"]:
                logger.debug(f"[PublicWS] System event => {data}")
            else:
                logger.debug(f"[PublicWS] Unhandled dict => {data}")
            return

        # If it's a list => typically => [channelID, payload, feed_name, pair]
        if isinstance(data, list):
            if len(data) < 4:
                logger.debug(f"[PublicWS] short data => {data}")
                return
            feed_name = data[2]
            pair = data[3]
            if feed_name == "ticker":
                await self._handle_ticker(data[1], pair)
            elif feed_name == "trade":
                await self._handle_trade(data[1], pair)
            else:
                logger.debug(f"[PublicWS] unknown feed={feed_name}, data={data}")

    async def _handle_ticker(self, ticker_update: dict, pair: str):
        """
        For "ticker" updates => parse ask/bid/last/volume => store in price_history.
        Then call on_ticker_callback if any. We do *not* call risk_manager here,
        since your new approach is for the risk_manager to read from DB on its own schedule.
        """
        ask_info = ticker_update.get("a", [])
        bid_info = ticker_update.get("b", [])
        last_info = ticker_update.get("c", [])
        vol_info = ticker_update.get("v", [])

        ask_price = float(ask_info[0]) if ask_info else 0.0
        bid_price = float(bid_info[0]) if bid_info else 0.0
        last_price = float(last_info[0]) if last_info else 0.0
        volume_val = float(vol_info[0]) if vol_info else 0.0

        # store in DB => 'price_history' table
        store_price_history(
            pair=pair,
            bid=bid_price,
            ask=ask_price,
            last=last_price,
            volume=volume_val
        )

        # Optionally call user callback
        if self.on_ticker_callback and last_price > 0:
            self.on_ticker_callback(pair, last_price)

        # NOTE: We do *not* call risk_manager.on_price_update(...) here.
        # Instead, risk_manager handles it by reading from the DB.

    async def _handle_trade(self, trades_chunk, pair: str):
        """
        For 'trade' feed => each item => [price, volume, time, side, orderType, misc].
        We can store these trade prices in 'price_history' if you want a record
        (with last=trade_price, bid=0, ask=0). Typically you'd only store ticker data,
        but if you want trades as well, do it here.
        """
        for tr in trades_chunk:
            price_val = float(tr[0])
            vol_val = float(tr[1])
            # Possibly parse time, side, etc. if needed.

            store_price_history(
                pair=pair,
                bid=0.0,
                ask=0.0,
                last=price_val,
                volume=vol_val
            )


# ------------------------------------------------------------------------------
# PRIVATE FEED: KrakenPrivateWSClient
# ------------------------------------------------------------------------------
class KrakenPrivateWSClient:
    """
    A WebSocket client dedicated to PRIVATE feeds from Kraken (wss://ws-auth.kraken.com).
    Subscribes to openOrders, ownTrades, etc. We do not handle public feed logic here.
    Orders/fills are recorded in 'pending_trades' and 'trades' as they occur.
    """

    def __init__(
        self,
        token: str,
        on_private_event_callback=None,
        risk_manager=None
    ):
        """
        :param token: The token from REST => /0/private/GetWebSocketsToken
        :param on_private_event_callback: optional callback for private events
        :param risk_manager: optional RiskManager instance if you want to do daily drawdown checks, etc.
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
        """Spins up a thread that runs an asyncio loop => connect => subscribe => read messages."""
        if self.running:
            logger.warning("KrakenPrivateWSClient is already running => skip.")
            return
        self.running = True

        self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()
        logger.info("[KrakenPrivateWSClient] started => private feed.")

    def stop(self):
        """Signals the loop to exit, then joins the background thread."""
        if not self.running:
            logger.info("[KrakenPrivateWSClient] Not running or already stopped.")
            return
        self.running = False
        logger.info("[KrakenPrivateWSClient] stop requested.")
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            logger.info("[KrakenPrivateWSClient] background thread joined.")

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
        """Attempt to connect => subscribe => read messages => if disconnected, retry in 5s."""
        while self.running:
            try:
                logger.info("[PrivateWS] Connecting to wss://ws-auth.kraken.com ...")
                async with websockets.connect(self.url_private, ssl=self.ssl_context) as ws:
                    self._ws = ws
                    logger.info("[PrivateWS] Connected => subscribing to private channels.")
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
        Subscribes to 'openOrders' and 'ownTrades' using your token.
        """
        if not self.token:
            logger.warning("[PrivateWS] No token => cannot subscribe.")
            return

        # 1) openOrders
        msg_open = {
            "event": "subscribe",
            "subscription": {
                "name": "openOrders",
                "token": self.token
            }
        }
        await ws.send(json.dumps(msg_open))
        logger.info("[PrivateWS] subscribed => openOrders")

        # 2) ownTrades
        msg_trades = {
            "event": "subscribe",
            "subscription": {
                "name": "ownTrades",
                "token": self.token,
                "snapshot": True,
                "consolidate_taker": True
            }
        }
        await ws.send(json.dumps(msg_trades))
        logger.info("[PrivateWS] subscribed => ownTrades")

    async def _consume_messages(self, ws):
        while self.running:
            try:
                raw_msg = await ws.recv()
                await self._handle_message(raw_msg)
            except websockets.exceptions.ConnectionClosed:
                logger.warning("[PrivateWS] Connection closed => break loop")
                break
            except Exception as e:
                logger.exception(f"[PrivateWS] => {e}")
                break

    async def _handle_message(self, raw_msg: str):
        """
        Handles any private feed messages:
          - addOrderStatus, cancelOrderStatus, openOrders, ownTrades, etc.
        """
        try:
            data = json.loads(raw_msg)
        except json.JSONDecodeError:
            logger.error(f"[PrivateWS] JSON parse error => {raw_msg}")
            return

        if isinstance(data, dict):
            ev = data.get("event")
            if ev in ["subscriptionStatus", "heartbeat", "systemStatus"]:
                logger.debug(f"[PrivateWS] system event => {data}")
            else:
                self._process_private_order_message(data)
            if self.on_private_event_callback:
                self.on_private_event_callback(data)
            return

        # If data is a list => e.g. [ [some data], "ownTrades", {...} ]
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
        Called when we get addOrderStatus or cancelOrderStatus or error messages.
        We only update 'pending_trades', do not finalize actual trade fills here.
        """
        event_type = msg.get("event", "")
        if event_type != "addOrderStatus":
            return

        txid = msg.get("txid", "")
        status_str = msg.get("status", "")
        userref = msg.get("userref")
        error_msg = msg.get("errorMessage", "")

        if status_str == "error":
            logger.info(f"[PrivateWS] Order rejected => {txid}, reason={error_msg}")
            mark_pending_trade_rejected_by_kraken_id(txid, error_msg)
        elif status_str in ["canceled", "expired"]:
            logger.info(f"[PrivateWS] Order canceled => {txid}, reason={status_str}")
            vol_exec = float(msg.get("vol_exec", "0.0"))
            avg_price = float(msg.get("avg_price", "0.0"))
            fee_val = float(msg.get("fee", "0.0"))
            if vol_exec == 0:
                # fully unfilled => reject
                mark_pending_trade_rejected_by_kraken_id(txid, status_str)
            else:
                # partial => final close
                self._finalize_trade_from_kraken(txid, vol_exec, avg_price, fee_val)
        elif status_str == "closed":
            logger.info(f"[PrivateWS] Order closed => {txid}")
            vol_exec = float(msg.get("vol_exec", "0.0"))
            avg_price = float(msg.get("avg_price", "0.0"))
            fee_val = float(msg.get("fee", "0.0"))
            self._finalize_trade_from_kraken(txid, vol_exec, avg_price, fee_val)
        elif status_str == "ok":
            # order accepted
            if userref:
                set_kraken_order_id_for_pending_trade(int(userref), txid)
            mark_pending_trade_open_by_kraken_id(txid)
        else:
            logger.debug(f"[PrivateWS] addOrderStatus => {txid}, status={status_str}")

    def _process_open_orders(self, order_chunk):
        """
        openOrders feed => [ { txid: {...} }, ...]
        We only manage 'pending_trades'. Actual trade fills come from 'ownTrades'.
        """
        if not isinstance(order_chunk, list):
            return

        for item in order_chunk:
            if not isinstance(item, dict):
                continue
            for txid, info in item.items():
                status_str = info.get("status", "")
                vol_exec_str = info.get("vol_exec", "0.0")
                vol_exec = float(vol_exec_str)

                if status_str in ("canceled", "expired"):
                    if vol_exec == 0:
                        mark_pending_trade_rejected_by_kraken_id(txid, status_str)
                    else:
                        self._finalize_trade_from_kraken(txid, vol_exec, 0.0, 0.0)
                elif status_str == "closed":
                    self._finalize_trade_from_kraken(txid, vol_exec, 0.0, 0.0)
                elif status_str == "open":
                    mark_pending_trade_open_by_kraken_id(txid)
                else:
                    logger.debug(f"[PrivateWS] order_id={txid}, status={status_str}, ignoring.")

    def _process_own_trades(self, trades_chunk):
        """
        ownTrades => actual fills.
        Each item => { trade_id: {"type":"buy","vol":"0.01","pair":"ETH/USD","price":"1800.00", ...} }.
        We'll parse and call _store_own_trade_if_new(...) with all parameters.
        """
        if not isinstance(trades_chunk, list):
            return

        for item in trades_chunk:
            if not isinstance(item, dict):
                continue

            for trade_id, trade_obj in item.items():
                # Extract fields from trade_obj
                side_str = trade_obj.get("type", "").upper()  # "BUY" or "SELL"
                pair_str = trade_obj.get("pair", "UNKNOWN")
                vol_str = trade_obj.get("vol", "0.0")
                px_str = trade_obj.get("price", "0.0")
                fee_str = trade_obj.get("fee", "0.0")

                # Convert numeric strings
                quantity = float(vol_str)
                fill_price = float(px_str)
                fill_fee = float(fee_str)

                # Optionally parse 'time', but we can get a timestamp from time.time() if we want.
                # time_float = float(trade_obj.get("time", "0"))  # if needed

                self._store_own_trade_if_new(
                    trade_id=trade_id,
                    side=side_str,
                    pair=pair_str,
                    quantity=quantity,
                    fill_price=fill_price,
                    fee=fill_fee,
                    source=None,  # or fill from somewhere if available
                    rationale=None  # or fill from somewhere if available
                )

    def _store_own_trade_if_new(
            self,
            trade_id: str,
            side: str,
            pair: str,
            quantity: float,
            fill_price: float,
            fee: float,
            source: str = None,
            rationale: str = None
    ):
        """
        Insert a new row into 'trades' if we haven't inserted this trade_id yet.

        Additionally:
          1) We'll look up 'source' and 'rationale' in the 'pending_trades' table
             (matching kraken_order_id = trade_id).
          2) If found, we override the 'source' or 'rationale' parameters with those
             from the pending_trades table, unless they're None there as well.
          3) Then we insert into 'trades' with the final source/rationale.

        :param trade_id: The unique Kraken trade ID or order ID
        :param side: "BUY" or "SELL"
        :param pair: e.g. "ETH/USD"
        :param quantity: how many coins filled
        :param fill_price: the fill price
        :param fee: total fee
        :param source: e.g. "ai_strategy" or "risk_manager" — overridden if found in pending_trades
        :param rationale: e.g. GPT text or "stop-loss" / "take-profit" — overridden if found
        """
        conn = sqlite3.connect("trades.db")
        try:
            c = conn.cursor()

            # 1) Check if we've already inserted this trade_id => skip
            c.execute("""
                SELECT id
                FROM trades
                WHERE order_id=?
                LIMIT 1
            """, (trade_id,))
            row = c.fetchone()
            if row:
                return  # Already recorded => exit early

            # 2) Attempt to fetch source, rationale from pending_trades if they exist
            c.execute("""
                SELECT source, rationale
                FROM pending_trades
                WHERE kraken_order_id=?
                LIMIT 1
            """, (trade_id,))
            pend_row = c.fetchone()
            if pend_row:
                pending_source, pending_rationale = pend_row
                # Only override if they exist
                if pending_source is not None:
                    source = pending_source
                if pending_rationale is not None:
                    rationale = pending_rationale

            # 3) Insert new row into 'trades' with final source/rationale
            ts = int(time.time())  # or parse real fill-time if you like

            c.execute("""
                INSERT INTO trades (
                    timestamp,
                    pair,
                    side,
                    quantity,
                    price,
                    order_id,
                    fee,
                    realized_pnl,
                    source,
                    rationale
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
            """, (
                ts,
                pair,
                side,
                quantity,
                fill_price,
                trade_id,
                fee,
                source,
                rationale
            ))
            conn.commit()

            logger.info(
                f"[PrivateWS] Inserted new fill => trade_id={trade_id}, side={side}, "
                f"qty={quantity}, px={fill_price}, source={source}, rationale={rationale}"
            )

        except Exception as e:
            logger.exception(f"[PrivateWS] Error inserting ownTrades => {e}")
        finally:
            conn.close()

    def send_order(self, pair: str, side: str, ordertype: str, volume: float, price: float = None, userref: int = None):
        """
        e.g. self.send_order("XBT/USD", "buy", "market", 0.01)
        This triggers an 'addOrder' event => 'addOrderStatus' to confirm success or error.
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
                "volume": str(volume)
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
        """
        Cancel orders => e.g. self.cancel_order(["O6S6CF-ABC123-XYZ987"]).
        """
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
        Called whenever an order is canceled/expired/closed => finalize pending_trades row.
        Then we also let risk_manager know if there's a related lot.

        1) Look up the pending_trades row (which includes source, rationale, etc.).
        2) Mark it 'closed'.
        3) Insert a final record into 'trades' via _store_own_trade_if_new(...) if it isn't inserted yet.
           (Because ownTrades feed might also do it, but if it's partial fill we handle it here.)
        4) If there's a lot_id => call risk_manager.on_pending_trade_closed(...).
        """
        row = _fetch_pending_trade_by_kraken_id(kraken_order_id)
        if not row:
            logger.warning(f"[PrivateWS] No pending_trade row for kraken_order_id={kraken_order_id}")
            return

        # row: (id, pair, side, requested_qty, lot_id, source, rationale)
        pending_id = row[0]
        pair = row[1]
        side = row[2]
        requested_qty = row[3]
        lot_id = row[4]
        source = row[5]  # "risk_manager" or "ai_strategy", etc.
        rationale = row[6]  # "stop-loss", "take-profit", or GPT text, etc.

        reason = f"final fill size={filled_size}, fee={fee}"
        mark_pending_trade_closed(pending_id, reason=reason)
        logger.info(f"[PrivateWS] Marked pending_id={pending_id} closed => fill={filled_size}, fee={fee}")

        # Insert row into `trades` if needed (with the new source/rationale).
        # Usually, ownTrades feed also calls _store_own_trade_if_new. But if you want
        # to force an entry here, you can do so. The typical approach is to rely on
        # ownTrades to do the actual insertion. If you want them to definitely have source/rationale,
        # we can do:
        self._store_own_trade_if_new(
            trade_id=kraken_order_id,  # we treat the Kraken ID as 'order_id'
            side=side,
            pair=pair,
            quantity=filled_size,
            fill_price=avg_fill_price,
            fee=fee,
            source=source,
            rationale=rationale
        )

        # If there's a lot => inform risk_manager
        if lot_id is not None and self.risk_manager:
            self.risk_manager.on_pending_trade_closed(
                lot_id=lot_id,
                fill_size=filled_size,
                side=side,
                fill_price=avg_fill_price
            )


# ------------------------------------------------------------------------------
# HELPER: fetch pending_trades row by kraken_order_id
# ------------------------------------------------------------------------------
def _fetch_pending_trade_by_kraken_id(kraken_order_id: str):
    """
    Returns (id, pair, side, requested_qty, lot_id, source, rationale)
    from 'pending_trades' where kraken_order_id=?.

    We fetch 'source' and 'rationale' so that final trades can store them.
    """
    conn = sqlite3.connect("trades.db")
    try:
        c = conn.cursor()
        c.execute("""
            SELECT
                id,
                pair,
                side,
                requested_qty,
                lot_id,
                source,
                rationale
            FROM pending_trades
            WHERE kraken_order_id=?
        """, (kraken_order_id,))
        return c.fetchone()
    except Exception as e:
        logger.exception(f"Error fetching pending_trades by kraken_order_id={kraken_order_id}: {e}")
        return None
    finally:
        conn.close()

def mark_pending_trade_open_by_kraken_id(kraken_order_id: str):
    """
    If the order is accepted => pending_trades => status='open'.
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
            logger.debug(f"[PrivateWS] No pending_trades row found with kraken_order_id={kraken_order_id} to mark open.")
    except Exception as e:
        logger.exception(f"Error marking trade open => {e}")
    finally:
        conn.close()


def mark_pending_trade_rejected_by_kraken_id(kraken_order_id: str, reason: str = None):
    """
    If an order is rejected by Kraken, set status='rejected'.
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
            logger.debug(f"[PrivateWS] No pending row found with kraken_order_id={kraken_order_id} to reject.")
    except Exception as e:
        logger.exception(f"Error marking trade rejected => {e}")
    finally:
        conn.close()
