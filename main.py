# ==============================================================================
# FILE: main.py
# ==============================================================================
"""
main.py

A hybrid AI trading application where:
1) The AI is polled every 5 minutes for strategic decisions (BUY, SELL, HOLD)
   based on aggregated data from 'price_history'.
2) A separate RiskManager forcibly closes positions if a stop-loss or
   take-profit condition is met, checking real-time price ticks from
   the WebSocket feed (rather than waiting for the AI's next cycle).
3) We track a single position per pair in memory.
4) Model training and aggregator merges remain similar to previous examples.

Usage:
    python main.py
"""

import time
import os
import math
import logging
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
import yaml

# ------------------------------------------------------------------------------
# Import modules
# ------------------------------------------------------------------------------
from ai_strategy import AIStrategy
from risk_manager import RiskManager  # NEW: For real-time stop-loss / take-profit
from db import init_db, record_trade_in_db
from ws_data_feed import KrakenWSClient

# If you have a function to run training in train_model.py, you might import it:
from fetch_lunarcrush import fetch_lunarcrush_data
from fetch_cryptopanic import fetch_cryptopanic_data
from train_model import main as training_main

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Load environment for any keys
# ------------------------------------------------------------------------------
load_dotenv()
KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "FAKE_KEY")
KRAKEN_API_SECRET = os.getenv("KRAKEN_SECRET_API_KEY", "FAKE_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "FAKE_OPENAI_KEY")

# ------------------------------------------------------------------------------
# Load toggles (and other settings) from config.yaml
# ------------------------------------------------------------------------------
CONFIG_FILE = "config.yaml"
with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

TRADED_PAIRS = config.get("traded_pairs", [])
ENABLE_TRAINING = config.get("enable_training", False)
ENABLE_LIVE_AI_INFERENCE = config.get("enable_live_ai_inference", True)
ENABLE_GPT_INTEGRATION = config.get("enable_gpt_integration", False)
DB_FILE = "trades.db"
DB_ENABLED = True
MAX_POSITION_SIZE = 0.001

# We'll do 5-minute aggregator cycles => 300s
AGGREGATOR_INTERVAL_SECONDS = config.get("trade_interval_seconds", 300)

# ------------------------------------------------------------------------------
# (1) get_aggregated_data: aggregator for the last X seconds
# ------------------------------------------------------------------------------
def get_aggregated_data(pair: str, period_seconds: int) -> dict:
    """
    Aggregates data from the last 'period_seconds' for 'pair' in 'price_history'.
    Returns a dict with average, min, max price, total volume, etc.
    """
    cutoff_ts = math.floor(time.time()) - period_seconds
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute(f"""
            SELECT
                AVG(last_price) as avg_price,
                MIN(last_price) as min_price,
                MAX(last_price) as max_price,
                SUM(volume) as total_volume
            FROM price_history
            WHERE pair=? AND timestamp >= ?
        """, (pair, cutoff_ts))
        row = c.fetchone()
        if not row or row[0] is None:
            logger.warning(f"No aggregator data for {pair} in last {period_seconds} seconds.")
            return {}

        avg_p, min_p, max_p, vol_sum = row
        return {
            "avg_price": float(avg_p),
            "min_price": float(min_p),
            "max_price": float(max_p),
            "total_volume": float(vol_sum),
            "pair": pair,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.exception(f"Error aggregating data for {pair}: {e}")
        return {}
    finally:
        conn.close()

# ------------------------------------------------------------------------------
# (2) Single price fetch (optional leftover from older approach)
# ------------------------------------------------------------------------------
def get_latest_price_from_db(pair: str) -> dict:
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            SELECT timestamp, bid_price, ask_price, last_price, volume
            FROM price_history
            WHERE pair=?
            ORDER BY id DESC
            LIMIT 1
        """, (pair,))
        row = c.fetchone()
        if row:
            return {
                "timestamp": row[0],
                "bid": row[1],
                "ask": row[2],
                "price": row[3],
                "volume": row[4]
            }
        else:
            return {}
    except Exception as e:
        logger.exception(f"Error fetching latest price from DB: {e}")
        return {}
    finally:
        conn.close()

# ------------------------------------------------------------------------------
# (3) place_order mock
# ------------------------------------------------------------------------------
def place_order(pair: str, side: str, volume: float) -> str:
    logger.info(f"Placing {side} order for {volume} of {pair} (mock).")
    mock_order_id = f"MOCK-{side}-{int(time.time())}"
    return mock_order_id

# ------------------------------------------------------------------------------
# (4) DummyStrategy if not using AI
# ------------------------------------------------------------------------------
class DummyStrategy:
    def predict(self, market_data: dict):
        return ("HOLD", 0.0)

# ------------------------------------------------------------------------------
# (5) The main "Mature + Hybrid" application
# ------------------------------------------------------------------------------
class MatureHybridApp:
    """
    A hybrid approach:
    - We poll the AI every aggregator interval (5-10 min by default) to get a strategy decision.
    - Meanwhile, a separate RiskManager checks real-time ticks for forced exits (stop-loss, take-profit).
    - We store a single position per pair in memory and update it based on AI or forced exits.
    """

    def __init__(self, pairs, ai_strategy, risk_manager, aggregator_interval=300):
        self.pairs = pairs
        self.ai_strategy = ai_strategy
        self.risk_manager = risk_manager
        self.aggregator_interval = aggregator_interval

        # Track in-memory positions: pair => size
        self.current_positions = {p: 0.0 for p in pairs}

        # Timestamps to track last aggregator call for each pair
        self.last_agg_call = {p: 0 for p in pairs}

    def on_tick(self, pair: str, latest_price: float):
        """
        Called by the WebSocket feed whenever new tick data arrives for a pair.
        1) We let the RiskManager check if we must forcibly exit.
        2) If enough time has passed since the last aggregator cycle for that pair, we do an aggregator-based AI call.
        """
        # 1) Risk manager checks if we have an open position and must close
        pos_size = self.current_positions.get(pair, 0.0)
        if pos_size != 0.0:
            # If there's a forced exit, we do it
            should_exit, reason = self.risk_manager.check_exit(
                pair=pair,
                current_price=latest_price,
                position_size=pos_size
            )
            if should_exit:
                logger.info(f"RiskManager forced exit for {pair} => {reason}")
                self._close_position(pair, latest_price, "FORCED_EXIT")
                return

        # 2) Check if aggregator interval is up for this pair
        now = time.time()
        if now - self.last_agg_call[pair] >= self.aggregator_interval:
            # aggregator call => fetch aggregated data => AI => possibly open/close
            self._aggregator_cycle(pair)
            self.last_agg_call[pair] = now

    def _aggregator_cycle(self, pair: str):
        """
        Once aggregator interval hits for a pair, we gather aggregated data
        and pass it to the AI strategy.
        """
        market_data = get_aggregated_data(pair, period_seconds=self.aggregator_interval)
        if not market_data or "avg_price" not in market_data:
            logger.warning(f"No aggregator data for {pair}, skipping cycle.")
            return

        # Add current position info
        market_data["current_position"] = self.current_positions[pair]

        # Strategy call
        signal, suggested_size = self.ai_strategy.predict(market_data)
        logger.info(f"AI => {pair} => {signal}, size={suggested_size}")

        final_size = min(suggested_size, MAX_POSITION_SIZE)

        if signal == "BUY":
            if self.current_positions[pair] == 0.0:
                # open position
                fill_price = market_data["avg_price"]
                oid = place_order(pair, "BUY", final_size)
                if DB_ENABLED:
                    record_trade_in_db(
                        side="BUY",
                        quantity=final_size,
                        price=fill_price,
                        order_id=oid,
                        pair=pair
                    )
                self.current_positions[pair] = final_size
                logger.info(f"Opened position for {pair}, size={final_size}, price={fill_price}")
            else:
                logger.info(f"Already have a position on {pair}, ignoring BUY.")
        elif signal == "SELL":
            if self.current_positions[pair] > 0.0:
                # close existing long
                fill_price = market_data["avg_price"]
                oid = place_order(pair, "SELL", self.current_positions[pair])
                if DB_ENABLED:
                    record_trade_in_db(
                        side="SELL",
                        quantity=self.current_positions[pair],
                        price=fill_price,
                        order_id=oid,
                        pair=pair
                    )
                logger.info(f"Closed position on {pair} of size {self.current_positions[pair]} at ~{fill_price}")
                self.current_positions[pair] = 0.0
            else:
                logger.info(f"No long position to SELL for {pair}. Ignoring.")
        else:
            logger.info(f"HOLD => no aggregator action for {pair}.")

    def _close_position(self, pair: str, latest_price: float, exit_reason: str):
        """
        Force-close any open position for 'pair' at 'latest_price',
        e.g. for a risk manager forced exit.
        """
        if self.current_positions[pair] > 0:
            side = "SELL"
            size = self.current_positions[pair]
            oid = place_order(pair, side, size)
            if DB_ENABLED:
                record_trade_in_db(
                    side=side,
                    quantity=size,
                    price=latest_price,
                    order_id=oid,
                    pair=pair
                )
            logger.info(f"Force-closed {pair} position of size {size} at {latest_price}, reason={exit_reason}")
            self.current_positions[pair] = 0.0


# ------------------------------------------------------------------------------
# Mock function for the WebSocket feed => We'll integrate the "on_tick" logic
# ------------------------------------------------------------------------------
class CustomKrakenWSClient(KrakenWSClient):
    """
    Subclass of your existing KrakenWSClient that, upon receiving a new ticker
    message, calls 'app.on_tick(...)' for risk manager checks.
    """
    def __init__(self, pairs, feed_type, app_ref):
        super().__init__(pairs, feed_type=feed_type)
        self.app_ref = app_ref  # reference to MatureHybridApp

    async def _handle_message(self, message: str):
        """
        We'll do the normal parse, then call app_ref.on_tick(...) with the new last_price
        if it's a ticker feed.
        """
        import json
        try:
            data = json.loads(message)
            if isinstance(data, dict):
                event_type = data.get("event")
                if event_type in ["subscribe", "subscriptionStatus", "heartbeat", "systemStatus"]:
                    logger.debug(f"WS event: {data}")
                return

            if isinstance(data, list):
                if len(data) < 4:
                    return
                feed_name = data[2]
                pair = data[3]
                if feed_name == "ticker":
                    update_obj = data[1]
                    last_trade_info = update_obj.get("c", [])
                    if len(last_trade_info) > 0:
                        last_price = float(last_trade_info[0])
                        # call risk manager => on_tick
                        self.app_ref.on_tick(pair, last_price)

                    # store in DB
                    # you can do store_price_history(...) if you want
                    # We'll rely on the original logic from super:
                    await super()._handle_message(message)

        except Exception as e:
            logger.exception(f"Error in custom handle_message: {e}")


# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------
def main():
    logger.info("Starting hybrid AI app with real-time risk manager + 5-min aggregator AI")

    # 1) Optional training
    if ENABLE_TRAINING:
        fetch_cryptopanic_data()
        fetch_lunarcrush_data()
        training_main()
        logger.info("Training complete.")
    else:
        logger.info("Skipping training step.")

    # 2) DB init
    init_db()

    # 3) Decide AI or dummy
    if ENABLE_LIVE_AI_INFERENCE:
        ai_model = AIStrategy(
            pairs=TRADED_PAIRS,
            model_path="trained_model.pkl",
            use_openai=ENABLE_GPT_INTEGRATION
        )
    else:
        ai_model = DummyStrategy()

    # 4) Create a risk manager
    # For demonstration, let's say we forcibly exit if we lose 2% or gain 5%
    # in real-time. See risk_manager.py for logic.
    rm = RiskManager(stop_loss_pct=0.02, take_profit_pct=0.05)

    # 5) Create the main app instance
    app = MatureHybridApp(
        pairs=TRADED_PAIRS,
        ai_strategy=ai_model,
        risk_manager=rm,
        aggregator_interval=AGGREGATOR_INTERVAL_SECONDS
    )

    # 6) Start a custom WS client that calls app.on_tick
    ws_client = CustomKrakenWSClient(TRADED_PAIRS, feed_type="ticker", app_ref=app)
    ws_client.start()

    logger.info("Press Ctrl+C to exit.")
    try:
        # We'll just wait indefinitely, the on_tick logic handles the rest
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Exiting main loop.")
    finally:
        ws_client.stop()
        logger.info("Stopped.")


if __name__ == "__main__":
    main()
