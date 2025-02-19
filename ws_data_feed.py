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
     we keep that in the RiskManager's own background DB cycle.
   - Does *not* place trades—only receives data.

2) KrakenPrivateWSClient:
   - Connects to wss://ws-auth.kraken.com
   - Subscribes to private feeds ("openOrders", "ownTrades", etc.)
   - Can place/cancel orders (via send_order, cancel_order)
   - Requires an API token from Kraken’s /0/private/GetWebSocketsToken
   - Records final fills in the 'trades' table (via ownTrades or order closures).
   - Optionally calls risk_manager if we want to confirm partial-lot changes, etc.

Db interactions:
 - price_history: store real-time ticker/trade updates
 - pending_trades/trades: track ephemeral states & final fills
 - Possibly updates lot status in risk_manager on SELL fill
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
      - Maintain a connection to Kraken's public feed(s).
      - Store the incoming data to 'price_history' (via store_price_history).
      - Optionally call a user-supplied callback (on_ticker_callback) for each new ticker.

    We do NOT call risk_manager.on_price_update() here. The RiskManager
    does that by polling the DB on its own schedule.
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

        # Optional references
        self.risk_manager = None
        self.kraken_balances = None

        self.ssl_context = create_secure_ssl_context()
        self.loop = None
        self._ws = None
        self.running = False
        self._thread = None

    def start(self):
        """
        Start the WebSocket client in a background thread.
        Once started, it attempts to connect, subscribe, and process messages
        from Kraken until 'stop()' is called.
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
        Private method that runs in the background thread.
        Creates an asyncio loop, attempts to connect, and consumes messages
        until 'stop()' is signaled.
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
          - subscribe to the chosen feed (ticker or trade)
          - read messages
          - on disconnection or error => retry after 5s
        """
        while self.running:
            try:
                logger.info(f"[PublicWS] Connecting => {self.url_public}")
                async with websockets.connect(self.url_public, ssl=self.ssl_context) as ws:
                    self._ws = ws
                    logger.info(f"[PublicWS] Connected => feed={self.feed_type}, pairs={self.pairs}")

                    # subscribe to the channels
                    await self._subscribe(ws)

                    # read messages until a disconnection or 'stop'
                    await self._consume_messages(ws)

                    # if we exit => reset
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
        Subscribe to "ticker" or "trade" feeds for the given pairs.
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
                logger.warning("[PublicWS] Connection closed => stopping read loop.")
                break
            except Exception as ex:
                logger.exception(f"[PublicWS] Error in consume => {ex}")
                break

    async def _handle_message(self, raw_msg: str):
        """
        Parse the incoming JSON. System-level messages (heartbeat, subscriptionStatus, etc.)
        get logged or ignored. Ticker or trade updates get passed along.
        """
        try:
            data = json.loads(raw_msg)
        except json.JSONDecodeError:
            logger.error(f"[PublicWS] JSON parse error => {raw_msg}")
            return

        # If data is a dict, it might be a system event
        if isinstance(data, dict):
            event = data.get("event")
            if event in ["subscribe", "subscriptionStatus", "heartbeat", "systemStatus"]:
                logger.debug(f"[PublicWS] System event => {data}")
            else:
                logger.debug(f"[PublicWS] Unhandled dict => {data}")
            return

        # If data is a list => typically => [channelID, payload, feed_name, pair]
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
        Then optionally call on_ticker_callback.
        """
        ask_info = ticker_update.get("a", [])
        bid_info = ticker_update.get("b", [])
        last_info = ticker_update.get("c", [])
        vol_info = ticker_update.get("v", [])

        ask_price = float(ask_info[0]) if ask_info else 0.0
        bid_price = float(bid_info[0]) if bid_info else 0.0
        last_price = float(last_info[0]) if last_info else 0.0
        volume_val = float(vol_info[0]) if vol_info else 0.0

        # Store in DB => 'price_history'
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

    async def _handle_trade(self, trades_chunk, pair: str):
        """
        For 'trade' feed => each item => [price, volume, time, side, orderType, misc].
        We'll store these as price_history rows with last=trade_price, volume=trade_volume.
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
    Subscribes to openOrders, ownTrades, etc.
    Orders/fills are tracked in 'pending_trades' and 'trades'.
    """

    def __init__(
        self,
        token: str,
        on_private_event_callback=None,
        risk_manager=None
    ):
        """
        :param token: The API token from REST => /0/private/GetWebSocketsToken
        :param on_private_event_callback: optional callback for private events
        :param risk_manager: optional RiskManager instance for e.g. partial-lot updates
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
        """Start the Private WS in a background thread => connect => subscribe => read messages."""
        if self.running:
            logger.warning("KrakenPrivateWSClient is already running => skip.")
            return
        self.running = True

        self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()
        logger.info("[KrakenPrivateWSClient] started => private feed.")

    def stop(self):
        """Stops the Private WS => signals exit => joins the thread."""
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
        """Connect => subscribe => read messages => if disconnected, retry in 5s until stopped."""
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

        # 1) openOrders subscription
        msg_open = {
            "event": "subscribe",
            "subscription": {
                "name": "openOrders",
                "token": self.token
            }
        }
        await ws.send(json.dumps(msg_open))
        logger.info("[PrivateWS] subscribed => openOrders")

        # 2) ownTrades subscription
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
                logger.warning("[PrivateWS] Connection closed => break loop.")
                break
            except Exception as e:
                logger.exception(f"[PrivateWS] => {e}")
                break

    async def _handle_message(self, raw_msg: str):
        """
        Handles private feed messages: openOrders, ownTrades, addOrderStatus, etc.
        """
        try:
            data = json.loads(raw_msg)
        except json.JSONDecodeError:
            logger.error(f"[PrivateWS] JSON parse error => {raw_msg}")
            return

        # Possibly system events
        if isinstance(data, dict):
            ev = data.get("event")
            if ev in ["subscriptionStatus", "heartbeat", "systemStatus"]:
                logger.debug(f"[PrivateWS] system event => {data}")
            else:
                self._process_private_order_message(data)
            if self.on_private_event_callback:
                self.on_private_event_callback(data)
            return

        # If data is a list => e.g. [someList, "ownTrades", {...}] or [someList, "openOrders", {...}]
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
        We'll update 'pending_trades' accordingly. Actual fills happen in ownTrades or by final closure.
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
                # fully unfilled => just mark rejected
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
                try:
                    set_kraken_order_id_for_pending_trade(int(userref), txid)
                except ValueError:
                    logger.warning(f"[PrivateWS] userref not an int => {userref}")
            mark_pending_trade_open_by_kraken_id(txid)
        else:
            logger.debug(f"[PrivateWS] addOrderStatus => {txid}, status={status_str} => ignoring.")

    def _process_open_orders(self, order_chunk):
        """
        openOrders feed => [ { txid: {...} }, ...]
        We'll keep 'pending_trades' in sync. Final fills come from ownTrades.
        """
        if not isinstance(order_chunk, list):
            return

        for item in order_chunk:
            if not isinstance(item, dict):
                continue
            for txid, info in item.items():
                userref = info.get("userref")
                status_str = info.get("status", "")
                vol_exec_str = info.get("vol_exec", "0.0")
                vol_exec = float(vol_exec_str)

                # If we can link userref => pending_id
                if userref:
                    try:
                        set_kraken_order_id_for_pending_trade(int(userref), txid)
                        logger.debug(f"[PrivateWS] Mapped userref={userref} -> kraken_order_id={txid}")
                    except ValueError:
                        logger.warning(f"[PrivateWS] userref is not int => {userref}")

                if status_str in ("canceled", "expired"):
                    if vol_exec == 0:
                        # no fills => treat as rejected
                        mark_pending_trade_rejected_by_kraken_id(txid, status_str)
                    else:
                        # partial fill => finalize
                        self._finalize_trade_from_kraken(txid, vol_exec, 0.0, 0.0)
                elif status_str == "closed":
                    self._finalize_trade_from_kraken(txid, vol_exec, 0.0, 0.0)
                elif status_str == "open":
                    mark_pending_trade_open_by_kraken_id(txid)
                else:
                    logger.debug(f"[PrivateWS] openOrders => order_id={txid}, status={status_str}, ignoring.")

    def _process_own_trades(self, trades_chunk):
        """
        ownTrades => actual fills.
        Each item => { 'TRADE_ID': {"type":"buy","vol":"0.01","pair":"ETH/USD","price":"1800.00", ...} }.
        We'll parse and call _store_own_trade_if_new(...) with all parameters.
        We may store exchange_fill_time if 'time' is available.
        """
        if not isinstance(trades_chunk, list):
            return

        for item in trades_chunk:
            if not isinstance(item, dict):
                continue

            for trade_id, trade_obj in item.items():
                # Example fields => "type", "vol", "pair", "price", "fee", "time"
                side_str = trade_obj.get("type", "").upper()  # "BUY" / "SELL"
                pair_str = trade_obj.get("pair", "UNKNOWN")
                vol_str = trade_obj.get("vol", "0.0")
                px_str = trade_obj.get("price", "0.0")
                fee_str = trade_obj.get("fee", "0.0")

                quantity = float(vol_str)
                fill_price = float(px_str)
                fill_fee = float(fee_str)

                # If we want to store the exchange-provided fill time:
                # It's often a float epoch, e.g. 1675797060.1234
                exchange_fill_time = None
                if "time" in trade_obj:
                    try:
                        exchange_fill_time = int(float(trade_obj["time"]))
                    except:
                        pass

                self._store_own_trade_if_new(
                    trade_id=trade_id,
                    side=side_str,
                    pair=pair_str,
                    quantity=quantity,
                    fill_price=fill_price,
                    fee=fill_fee,
                    source=None,   # We'll override from pending_trades if found
                    rationale=None, # We'll override from pending_trades if found
                    exchange_fill_time=exchange_fill_time
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
            rationale: str = None,
            exchange_fill_time: int = None
    ):
        """
        Insert a new row into 'trades' if we haven't inserted this trade_id yet.

        Steps:
          1) Check if 'trades' already has this order_id=trade_id => skip if so.
          2) If not, look up 'pending_trades' (kraken_order_id=trade_id) =>
             retrieve (lot_id, source, rationale).
          3) Insert final row in 'trades' with these extended fields:
               - lot_id
               - source
               - rationale (or any 'ai_rationale' if you want)
               - exchange_fill_time
        """
        conn = sqlite3.connect("trades.db")
        try:
            c = conn.cursor()

            # 1) Already exist?
            c.execute("""
                SELECT id FROM trades
                WHERE order_id=?
                LIMIT 1
            """, (trade_id,))
            row = c.fetchone()
            if row:
                return  # already recorded

            # 2) Attempt to fetch lot_id, source, rationale from pending_trades
            c.execute("""
                SELECT lot_id, source, rationale
                FROM pending_trades
                WHERE kraken_order_id=?
                LIMIT 1
            """, (trade_id,))
            pend_row = c.fetchone()
            lot_id = None
            if pend_row:
                lot_id, pending_source, pending_rationale = pend_row
                if pending_source is not None:
                    source = pending_source
                if pending_rationale is not None:
                    rationale = pending_rationale

            # 3) Insert into 'trades', using extended columns.
            ts_local = int(time.time())  # local insertion time
            # realized_pnl => set to NULL until realized is calculated
            # also store 'exchange_fill_time' if present

            c.execute("""
                INSERT INTO trades (
                    timestamp,        -- local insertion time
                    pair,
                    side,
                    quantity,
                    price,
                    order_id,
                    fee,
                    realized_pnl,
                    source,
                    rationale,
                    lot_id,
                    ai_rationale,
                    exchange_fill_time
                ) 
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?)
            """, (
                ts_local,
                pair,
                side,
                quantity,
                fill_price,
                trade_id,
                fee,
                source,
                rationale,
                lot_id,
                rationale,  # or store separately if you want a different ai_rationale
                exchange_fill_time
            ))
            conn.commit()

            logger.info(
                f"[PrivateWS] Inserted new fill => trade_id={trade_id}, side={side}, "
                f"qty={quantity}, px={fill_price}, fee={fee}, lot_id={lot_id}, source={source}"
            )

            # Optionally we can do risk_manager callback, but typically we rely on finalize or
            # on_pending_trade_closed for that. So we skip here.

        except Exception as e:
            logger.exception(f"[PrivateWS] Error inserting ownTrades => {e}")
        finally:
            conn.close()

    def send_order(self, pair: str, side: str, ordertype: str, volume: float,
                   price: float = None, userref: int = None):
        """
        e.g. send_order("XBT/USD", "buy", "market", 0.01)
        Triggers 'addOrder' => 'addOrderStatus' for confirmation.
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
                msg["userref"] = str(userref)
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

    def _finalize_trade_from_kraken(self, kraken_order_id: str, filled_size: float,
                                    avg_fill_price: float, fee: float):
        """
        Called if an order is canceled/expired/closed => finalize pending_trades
        row. Insert final trade in `trades` if not yet inserted. Then, if we have
        a lot_id => notify risk_manager.on_pending_trade_closed(...).
        """
        row = _fetch_pending_trade_by_kraken_id(kraken_order_id)
        if not row:
            logger.warning(f"[PrivateWS] No pending_trade row for kraken_order_id={kraken_order_id}")
            return

        # row => (id, pair, side, requested_qty, lot_id, source, rationale)
        pending_id = row[0]
        pair = row[1]
        side = row[2]
        requested_qty = row[3]
        lot_id = row[4]
        source = row[5]
        rationale = row[6]

        reason = f"final fill size={filled_size}, fee={fee}"
        mark_pending_trade_closed(pending_id, reason=reason)
        logger.info(f"[PrivateWS] Marked pending_id={pending_id} closed => fill={filled_size}, fee={fee}")

        # Insert row into trades if not exist
        # or call _store_own_trade_if_new so it also merges lot_id, source, rationale
        self._store_own_trade_if_new(
            trade_id=kraken_order_id,
            side=side,
            pair=pair,
            quantity=filled_size,
            fill_price=avg_fill_price,
            fee=fee,
            source=source,
            rationale=rationale,
            exchange_fill_time=None  # we do not have a direct time here in addOrderStatus
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
    If the order is accepted => set pending_trades.status='open'.
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
    If an order is rejected/canceled by Kraken, set status='rejected'.
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
            logger.debug(f"[PrivateWS] No pending_trades row found with kraken_order_id={kraken_order_id} => cannot reject.")
    except Exception as e:
        logger.exception(f"Error marking trade rejected => {e}")
    finally:
        conn.close()
