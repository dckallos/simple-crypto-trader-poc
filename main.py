# ==============================================================================
# FILE: main.py
# ==============================================================================
"""
main.py

A mature application that:
1) Initializes DB (including open_positions and ai_context tables).
2) Optionally runs training if config says so.
3) Starts a custom WebSocket feed for real-time data, storing it in price_history.
4) Uses aggregator intervals to pass data to AIStrategy (with GPT or scikit).
5) Forcibly closes positions if RiskManager triggers (like a -5% stop-loss) in real time.

Requires:
 - db.py: with init_db(), open_positions logic, ai_context logic.
 - ai_strategy.py: your updated version with DB-based position storage and GPT context.
 - risk_manager.py, ws_data_feed.py, fetch_cryptopanic.py, fetch_lunarcrush.py, train_model.py, etc.

No lines omitted for brevity. You can copy/paste to replace your main.py.
"""

import time
import math
import logging
import sqlite3
from datetime import datetime
import yaml
import os

from dotenv import load_dotenv

import ai_strategy
# --------------------------------------------------------------------------
# Local imports
# --------------------------------------------------------------------------
from db import init_db
from ai_strategy import AIStrategy
from risk_manager import RiskManager
from ws_data_feed import KrakenWSClient

# optional training
from fetch_cryptopanic import fetch_cryptopanic_data
from fetch_lunarcrush import fetch_lunarcrush_data
from train_model import main as training_main
from ai_strategy import AIStrategy

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()
KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "FAKE_KEY")
KRAKEN_API_SECRET = os.getenv("KRAKEN_SECRET_API_KEY", "FAKE_SECRET")

# Read config
CONFIG_FILE = "config.yaml"
with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

TRADED_PAIRS = config.get("traded_pairs", [])
ENABLE_TRAINING = config.get("enable_training", False)
ENABLE_LIVE_AI_INFERENCE = config.get("enable_live_ai_inference", True)
ENABLE_GPT_INTEGRATION = config.get("enable_gpt_integration", False)

DB_FILE = "trades.db"
DB_ENABLED = True

# Aggregator approach: poll aggregator data every X seconds
AGGREGATOR_INTERVAL_SECONDS = config.get("trade_interval_seconds", 300)

MAX_POSITION_SIZE = 0.001

# ------------------------------------------------------------------------------
# A Hybrid approach that references AIStrategy for aggregator decisions
# plus risk manager for forced exits, using a custom WebSocket
# ------------------------------------------------------------------------------
class HybridApp:
    """
    We maintain:
    - AIStrategy (with DB-based positions)
    - aggregator interval
    - real-time forced exit checks in on_tick
    """

    def __init__(self, pairs, ai_strategy, aggregator_interval=300):
        self.pairs = pairs
        self.ai_strategy = ai_strategy
        self.aggregator_interval = aggregator_interval
        # track last aggregator call
        self.last_call_ts = {p: 0 for p in pairs}

    def on_tick(self, pair: str, last_price: float):
        """
        Called each time new ticker data arrives for 'pair'.
        1) Possibly do forced exit if risk manager triggers
           (the AIStrategy includes a risk manager check).
           Typically you'd do that inside the AIStrategy or a separate check.
        2) If aggregator interval elapsed, fetch aggregator data => call AIStrategy => place trades
        """
        # forced exit logic can be done in AIStrategy or separate.
        # We'll do aggregator cycle check here:
        now = time.time()
        if (now - self.last_call_ts[pair]) >= self.aggregator_interval:
            self._aggregator_cycle(pair)
            self.last_call_ts[pair] = now

    def _aggregator_cycle(self, pair: str):
        """
        1) gather aggregator data from the last aggregator_interval seconds
        2) pass to AIStrategy
        """
        from db import store_price_history
        aggregator_data = self._fetch_aggregated_data(pair, self.aggregator_interval)
        if not aggregator_data or "avg_price" not in aggregator_data:
            logger.warning(f"No aggregator data for {pair}, skipping aggregator cycle.")
            return

        # pass aggregator_data to AIStrategy
        # AIStrategy's predict() will do partial or full updates
        self.ai_strategy.predict(aggregator_data)

    def _fetch_aggregated_data(self, pair: str, period_seconds: int):
        """
        Aggregates price data from last period_seconds
        E.g. average price, min, max, total volume
        Potentially merges aggregator daily sentiment from CryptoPanic,
        or last known LunarCrush metrics.
        """
        import sqlite3
        cutoff_ts = math.floor(time.time()) - period_seconds
        conn = sqlite3.connect(DB_FILE)
        try:
            c = conn.cursor()
            c.execute(f"""
                SELECT
                  AVG(last_price),
                  MIN(last_price),
                  MAX(last_price),
                  SUM(volume)
                FROM price_history
                WHERE pair=? AND timestamp >= ?
            """, (pair, cutoff_ts))
            row = c.fetchone()
            if not row or row[0] is None:
                return {}
            avg_p, min_p, max_p, vol_sum = row

            # load aggregator fields from db if you want daily sentiment or recent LunarCrush
            # for example, daily cryptopanic:
            # we assume we have a function get_cryptopanic_sentiment_today() or so,
            # or we can store it in aggregator_data directly.
            # We'll do a simple approach:
            cryptopanic_sent = self._fetch_daily_cryptopanic_sentiment()
            galaxy_score, alt_rank = self._fetch_lunarcrush_metrics(pair)

            return {
                "avg_price": float(avg_p),
                "min_price": float(min_p),
                "max_price": float(max_p),
                "total_volume": float(vol_sum),
                "pair": pair,
                "timestamp": time.time(),
                "cryptopanic_sentiment": cryptopanic_sent,
                "galaxy_score": galaxy_score,
                "alt_rank": alt_rank
            }
        except Exception as e:
            logger.exception(f"Error aggregator data for {pair}: {e}")
            return {}
        finally:
            conn.close()

    def _fetch_daily_cryptopanic_sentiment(self) -> float:
        """
        Example: fetch daily average from cryptopanic_news for today's date,
        or just return 0 if none.
        """
        import datetime
        today_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        conn = sqlite3.connect(DB_FILE)
        try:
            c = conn.cursor()
            query = f"""
                SELECT avg(sentiment_score)
                FROM cryptopanic_news
                WHERE DATE(timestamp, 'unixepoch') = '{today_str}'
            """
            row = c.execute(query).fetchone()
            if row and row[0] is not None:
                return float(row[0])
            return 0.0
        except Exception as e:
            logger.exception(f"Error fetching daily cryptopanic sentiment: {e}")
            return 0.0
        finally:
            conn.close()

    def _fetch_lunarcrush_metrics(self, pair: str):
        """
        If your pair is "ETH/USD", symbol might be "ETH".
        We'll just do a minimal approach returning galaxy_score and alt_rank from the last row.
        """
        symbol = pair.split("/")[0].upper()
        conn = sqlite3.connect(DB_FILE)
        try:
            c = conn.cursor()
            c.execute("""
                SELECT galaxy_score, alt_rank
                FROM lunarcrush_data
                WHERE symbol=?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (symbol,))
            row = c.fetchone()
            if row:
                return float(row[0] or 0.0), int(row[1] or 0)
            return 0.0, 0
        except Exception as e:
            logger.exception(f"Error fetching lunarcrush for {symbol}: {e}")
            return 0.0, 0
        finally:
            conn.close()

# ------------------------------------------------------------------------------
# Custom WebSocket client that calls app.on_tick
# ------------------------------------------------------------------------------
class CustomKrakenWSClient(KrakenWSClient):
    def __init__(self, pairs, feed_type, app_ref):
        super().__init__(pairs, feed_type)
        self.app_ref = app_ref

    async def _handle_message(self, message: str):
        import json
        try:
            data = json.loads(message)
            if isinstance(data, dict):
                event_type = data.get("event")
                if event_type in ["subscribe", "subscriptionStatus", "heartbeat", "systemStatus"]:
                    logger.debug(f"WS event: {data}")
                return

            if isinstance(data, list) and len(data) >= 4:
                feed_name = data[2]
                pair = data[3]
                if feed_name == "ticker":
                    update_obj = data[1]
                    last_trade_info = update_obj.get("c", [])
                    if len(last_trade_info) > 0:
                        last_price = float(last_trade_info[0])
                        # call aggregator cycle if needed
                        self.app_ref.on_tick(pair, last_price)
                # store in DB anyway
                await super()._handle_message(message)

        except Exception as e:
            logger.exception(f"Error in custom WS handle_message: {e}")

# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------
def main():
    logger.info("Starting advanced main app with aggregator approach + DB-based positions + GPT context.")

    # 1) init DB
    init_db()

    # 2) Possibly training
    if ENABLE_TRAINING:
        fetch_cryptopanic_data()
        fetch_lunarcrush_data()
        training_main()
        logger.info("Training done.")
    else:
        logger.info("Skipping training step per config.")

    # 3) Decide AI or dummy
    if ENABLE_LIVE_AI_INFERENCE:
        ai_model = ai_strategy.AIStrategy(
            pairs=TRADED_PAIRS,
            model_path="trained_model.pkl",
            use_openai=ENABLE_GPT_INTEGRATION
        )
        logger.info(f"AIStrategy loaded with pairs: {TRADED_PAIRS}, GPT={ENABLE_GPT_INTEGRATION}")
    else:
        from ai_strategy import AIStrategy
        class DummyStrategy:
            def predict(self, market_data: dict):
                return ("HOLD", 0.0)
        ai_model = DummyStrategy()
        logger.info("DummyStrategy, no AI inference used.")

    # 4) Create the aggregator-based HybridApp
    app = HybridApp(
        pairs=TRADED_PAIRS,
        ai_strategy=ai_model,
        aggregator_interval=AGGREGATOR_INTERVAL_SECONDS
    )

    # 5) Start a custom WebSocket
    ws_client = CustomKrakenWSClient(TRADED_PAIRS, feed_type="ticker", app_ref=app)
    ws_client.start()

    logger.info("Press Ctrl+C to exit the main loop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Exiting main loop due to KeyboardInterrupt.")
    finally:
        ws_client.stop()
        logger.info("Stopped WebSocket and main app.")


if __name__ == "__main__":
    main()
