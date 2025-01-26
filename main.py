#!/usr/bin/env python3
# ==============================================================================
# FILE: main.py
# ==============================================================================
"""
main.py

Demonstration of consuming both the PUBLIC and PRIVATE Kraken websockets feeds,
using the classes defined in ws_data_feed.py:

- KrakenPublicWSClient: for public data like ticker/trade
- KrakenPrivateWSClient: for private data like openOrders, placing/canceling orders

We store each incoming ticker/trade in 'price_history'. Then at a fixed interval,
we gather aggregator data for ALL tracked coin pairs (e.g. sentiment, volatility,
moving averages, etc.) in a single pass. We pass it to AIStrategy => predict_multi_coins(...)
for a single GPT invocation that can decide collectively on each coin:
 - BUY / SELL / HOLD
 - Possibly close/swap an existing trade for a better opportunity.

We do keep single-coin logic in AIStrategy for optional usage in the future,
but this aggregator calls the multi-coin approach by default now.

Environment:
    - KRAKEN_API_KEY, KRAKEN_SECRET_API_KEY for private feed usage
    - OPENAI_API_KEY if GPT is used in AIStrategy
    - config.yaml for aggregator settings (pairs, intervals, risk_controls, etc.)

ASCII Flow:

    +-------------------+
    |  Public WS Feed   | (KrakenPublicWSClient)
    |  (ticker events)  |
    +-------------------+
              |
              v
    aggregator_app.on_ticker()  => store in DB => aggregator_app tries aggregator_cycle_all_coins()
                  (every aggregator_interval seconds)
              |
              v
    +------------------------+
    |   AIStrategy (multi)  |
    |   (GPT sees all coins)|
    +------------------------+
              |
       "decisions":[
         {"pair":"ETH/USD","action":"BUY","size":...},
         {"pair":"XBT/USD","action":"SELL","size":...}
       ]
              |
              v
 +----------------------------------+
 |  Private WS Feed (KrakenPrivate) |
 |  => placeOrder => real trades    |
 +----------------------------------+

Usage:
    python main.py
"""

import time
import requests
import hashlib
import hmac
import base64
import logging
import logging.config
import yaml
import os
import json
from dotenv import load_dotenv
import warnings
import sqlite3
import pandas as pd
from typing import Optional, Dict, Any, List

from db import init_db, DB_FILE, record_trade_in_db
from ai_strategy import AIStrategy
from ws_data_feed import KrakenPublicWSClient, KrakenPrivateWSClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s %(levelname)s [%(name)s] %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
    },
    'loggers': {
        '': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True
        },
        'requests': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True
        },
        'urllib3': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True
        },
    }
}


def get_ws_token(api_key: str, api_secret: str) -> Optional[dict]:
    """
    Retrieves a WebSockets authentication token from Kraken's REST API.
    That token is used for the private feed (wss://ws-auth.kraken.com).
    """
    url = "https://api.kraken.com/0/private/GetWebSocketsToken"
    path = "/0/private/GetWebSocketsToken"
    nonce = str(int(time.time() * 1000))

    data = {"nonce": nonce}
    postdata = f"nonce={nonce}"

    sha256 = hashlib.sha256((nonce + postdata).encode("utf-8")).digest()
    message = path.encode("utf-8") + sha256
    secret = base64.b64decode(api_secret)
    sig = hmac.new(secret, message, hashlib.sha512)
    signature = base64.b64encode(sig.digest())

    headers = {
        "API-Key": api_key,
        "API-Sign": signature.decode()
    }

    try:
        resp = requests.post(url, headers=headers, data=data, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error retrieving WS token from Kraken: {e}")
        return None


class HybridApp:
    """
    A combined aggregator that:
      1) Receives live ticker updates from public WS => store in DB => track last known price
      2) At aggregator_interval => aggregator_cycle_all_coins => gather aggregator data for ALL pairs
         => AIStrategy => single GPT call => produce decisions for each coin in one shot
      3) If final_action=BUY/SELL => place real trades on private feed.
    """

    def __init__(
        self,
        pairs: List[str],
        strategy: AIStrategy,
        aggregator_interval: int = 300,
        private_ws_client: Optional[KrakenPrivateWSClient] = None
    ):
        """
        :param pairs: e.g. ["ETH/USD","XBT/USD"] => tracked coins
        :param strategy: AIStrategy instance
        :param aggregator_interval: time in seconds between aggregator cycles
        :param private_ws_client: optional => private feed for real trades
        """
        self.pairs = pairs
        self.strategy = strategy
        self.aggregator_interval = aggregator_interval
        self.ws_private = private_ws_client

        # For storing last ticker price in memory
        self.latest_prices: Dict[str, float] = {}
        for p in pairs:
            self.latest_prices[p] = 0.0

        # For tracking last aggregator time
        self.last_agg_ts = 0

    def on_ticker(self, pair: str, last_price: float):
        """
        Called by public feed on each new ticker. We store in memory, then
        check if aggregator_interval is up => aggregator_cycle_all_coins.
        """
        self.latest_prices[pair] = last_price
        now = time.time()
        if (now - self.last_agg_ts) >= self.aggregator_interval:
            self.aggregator_cycle_all_coins()
            self.last_agg_ts = now

    def aggregator_cycle_all_coins(self):
        """
        Gather aggregator data for ALL pairs => single GPT call => multi-coin decisions.
        aggregator_list => e.g. [
           {
             "pair":"ETH/USD",
             "price":..., # from self.latest_prices or DB
             "aggregator_data":"rsi=...,price=...,vol=...,sentiment=..."
           },
           ...
        ]
        Then calls self.strategy.predict_multi_coins(...).
        Then places trades if final action is buy/sell.
        """
        logger.info("[Aggregator] aggregator_cycle_all_coins => building aggregator_list for all pairs")
        aggregator_list: List[Dict[str,Any]] = []

        # We'll also gather global trade history if we like
        trade_history = self._build_global_trade_history(limit=10)
        # If you have open_positions in sub_positions, we can build a textual representation:
        open_positions_txt = self._build_open_positions_list()

        for pair in self.pairs:
            pair_data = self._build_aggregator_for_pair(pair)
            aggregator_list.append(pair_data)

        logger.info("[Aggregator] aggregator_list => %s", aggregator_list)

        # Now call AIStrategy => multi approach
        decisions = self.strategy.predict_multi_coins(
            aggregator_list=aggregator_list,
            trade_history=trade_history,
            max_trades=10,
            open_positions=open_positions_txt
        )
        logger.info("[Aggregator] GPT decisions => %s", decisions)

        # For each coin => if action=buy/sell => place trades
        for p, (action, size) in decisions.items():
            if action in ("BUY","SELL") and size>0:
                side_str = "buy" if action=="BUY" else "sell"
                px = self._lookup_price_for_pair(p, aggregator_list)
                logger.info(f"[Aggregator] => place real order => {p}, side={side_str}, vol={size:.6f}, px={px}")
                if self.ws_private and px>0:
                    self.ws_private.send_order(
                        pair=p,
                        side=side_str,
                        ordertype="market",
                        volume=size
                    )
                    record_trade_in_db(
                        action,
                        size,
                        px,
                        "PENDING_KRAKEN_TXID",
                        p
                    )
                else:
                    logger.warning("[Aggregator] no private feed or no valid price => cannot place real order.")

    # --------------------------------------------------------------------------
    # BUILD aggregator data for each pair
    # --------------------------------------------------------------------------
    def _build_aggregator_for_pair(self, pair: str) -> Dict[str, Any]:
        """
        Return => {
          "pair":"ETH/USD",
          "price":1850.0,
          "aggregator_data":"rsi=44, price=1850, vol=0.02, sentiment=someVal, ma_10=..., etc."
        }
        We'll fetch from price_history, compute vol/ma, also load sentiment from DB, etc.
        """
        # current price from memory or DB
        last_price = self.latest_prices.get(pair, 0.0)

        # fetch short-term stats from DB
        px_stats = self._fetch_recent_price_trends(pair)
        vol = px_stats.get("price_volatility",0.0)
        ma10 = px_stats.get("price_ma_10",0.0)

        # load local sentiment => e.g. a short text or numeric
        base_symbol = pair.split("/")[0].upper()
        sentiment_text = self._load_sentiment_for_symbol(base_symbol)

        aggregator_str = f"price={last_price}, vol={vol}, ma10={ma10}, {sentiment_text}"

        return {
            "pair": pair,
            "price": last_price,
            "aggregator_data": aggregator_str
        }

    def _fetch_recent_price_trends(self, pair: str) -> Dict[str, Any]:
        """
        Query 'price_history' => last 100 => compute ma_10 + vol
        """
        try:
            conn = sqlite3.connect(DB_FILE)
            q = """
                SELECT last_price
                FROM price_history
                WHERE pair=?
                ORDER BY timestamp DESC
                LIMIT 100
            """
            df = pd.read_sql_query(q, conn, params=[pair])
        except Exception as e:
            logger.exception(f"[Aggregator] DB error => {e}")
            return {}
        finally:
            conn.close()

        if df.empty:
            return {}
        df = df.iloc[::-1].reset_index(drop=True)
        df["ma_10"] = df["last_price"].rolling(10).mean()
        df["volatility"] = df["last_price"].pct_change().std()
        last_row = df.iloc[-1]
        return {
            "price_ma_10": float(last_row["ma_10"]) if not pd.isnull(last_row["ma_10"]) else 0.0,
            "price_volatility": float(last_row["volatility"]) if not pd.isnull(last_row["volatility"]) else 0.0
        }

    def _load_sentiment_for_symbol(self, symbol: str) -> str:
        """
        Return e.g. "sentiment=..., galaxy=..., alt_rank=..."
        We'll do a short text from lunarcrush_timeseries or data.
        We'll just reference a function that returns a textual aggregator.
        """
        from db import DB_FILE
        # We'll do a short approach => pick last row from lunarcrush_timeseries
        # or we can do your 'load_lunarcrush_sentiment_history' if you like a multi-row summary:
        # We'll do a single-row aggregator approach for brevity:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        out="sentiment=0.0, galaxy=0.0, alt_rank=999999"
        try:
            c = conn.cursor()
            # find coin_id
            c.execute("SELECT lunarcrush_id FROM lunarcrush_data WHERE UPPER(symbol)=? ORDER BY id DESC LIMIT 1",(symbol.upper(),))
            row = c.fetchone()
            if row and row["lunarcrush_id"]:
                coin_id = str(row["lunarcrush_id"])
            else:
                return "sentiment=0.0, galaxy=0.0, alt_rank=999999"

            # find last row
            c.execute("""
                SELECT sentiment, galaxy_score, alt_rank
                FROM lunarcrush_timeseries
                WHERE coin_id=?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (coin_id,))
            row2 = c.fetchone()
            if row2:
                s = row2["sentiment"] if row2["sentiment"] else 0.0
                g = row2["galaxy_score"] if row2["galaxy_score"] else 0.0
                a = row2["alt_rank"] if row2["alt_rank"] is not None else 999999
                out = f"sentiment={s}, galaxy={g}, alt_rank={a}"
        except Exception as e:
            logger.exception(f"[Aggregator] error loading sentiment => {e}")
        finally:
            conn.close()

        return out

    def _lookup_price_for_pair(self, pair: str, aggregator_list: List[Dict[str,Any]]) -> float:
        """
        Helper to find the 'price' field from aggregator_list for the given pair,
        or fallback if missing.
        """
        for item in aggregator_list:
            if item["pair"]==pair:
                return item.get("price",0.0)
        return 0.0

    # --------------------------------------------------------------------------
    # Build a global trade history (the last N trades across all pairs), if desired
    # --------------------------------------------------------------------------
    def _build_global_trade_history(self, limit=10) -> List[str]:
        """
        Return a list of up to 'limit' lines describing the most recent trades
        across all pairs. e.g. "2025-01-21 10:15 BUY ETH/USD 0.001@2500"
        Sorted by ID desc for demonstration.
        """
        lines=[]
        conn = sqlite3.connect(DB_FILE)
        try:
            c = conn.cursor()
            c.execute("""
            SELECT timestamp, pair, side, quantity, price
            FROM trades
            ORDER BY id DESC
            LIMIT ?
            """,(limit,))
            rows = c.fetchall()
            if not rows:
                return []
            rows=rows[::-1]  # chronological
            for r in rows:
                t, p, side, qty, px = r
                import datetime
                dt_s = datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")
                line = f"{dt_s} {side} {p} {qty}@{px}"
                lines.append(line)
        except Exception as e:
            logger.exception(f"[Aggregator] error building global trade history => {e}")
        finally:
            conn.close()
        return lines

    # --------------------------------------------------------------------------
    # Build textual list of open positions from sub_positions
    # --------------------------------------------------------------------------
    def _build_open_positions_list(self) -> List[str]:
        """
        Query sub_positions for all open positions => return strings like
        "ETH/USD LONG 0.002, entry=1860"
        So GPT can see if we have trades it might want to close/swap.
        """
        lines=[]
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory=sqlite3.Row
        try:
            c=conn.cursor()
            c.execute("""
            SELECT pair, side, entry_price, size
            FROM sub_positions
            WHERE closed_at IS NULL
            """)
            rows = c.fetchall()
            for r in rows:
                p = r["pair"]
                s = r["side"]
                e = r["entry_price"]
                z = r["size"]
                lines.append(f"{p} {s.upper()} {z}, entry={e}")
        except Exception as e:
            logger.exception(f"[Aggregator] error building open_positions => {e}")
        finally:
            conn.close()
        return lines


def main():
    """
    The main runner:
      1) load config
      2) init_db
      3) create AIStrategy (use_openai => multi-coin approach)
      4) get private feed token
      5) aggregator => pass aggregator_data => AIStrategy => multi GPT decisions
    """
    logging.config.dictConfig(LOG_CONFIG)

    CONFIG_FILE = "config.yaml"
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

    ENABLE_TRAINING = config.get("enable_training", True)
    ENABLE_GPT = config.get("enable_gpt_integration", True)
    TRADED_PAIRS = config.get("traded_pairs", [])
    AGG_INTERVAL = config.get("trade_interval_seconds", 300)

    risk_controls = config.get("risk_controls", {
        "initial_spending_account": 50.0,
        "purchase_upper_limit_percent": 50.0,
        "minimum_buy_amount": 10.0,
        "max_position_value": 20.0
    })

    load_dotenv()
    KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY","FAKE_KEY")
    KRAKEN_API_SECRET = os.getenv("KRAKEN_SECRET_API_KEY","FAKE_SECRET")

    # 1) init DB
    init_db()

    # 2) (Optional) training step
    if ENABLE_TRAINING:
        logger.info("[Main] Potential training step here. Omitted for brevity.")

    # 3) create AIStrategy => includes single+multi coin logic
    ai_strategy = AIStrategy(
        pairs=TRADED_PAIRS,
        use_openai=ENABLE_GPT,
        max_position_size=3,
        stop_loss_pct=0.05,
        take_profit_pct=0.01,
        max_daily_drawdown=-0.02,
        risk_controls=risk_controls,
        gpt_model="o1-mini",          # Example model name
        gpt_temperature=1.0,
        gpt_max_tokens=2000
    )
    logger.info(f"[Main] AIStrategy => multi-coin GPT => pairs={TRADED_PAIRS}, GPT={ENABLE_GPT}")

    # 4) get private token
    token_json = get_ws_token(KRAKEN_API_KEY, KRAKEN_API_SECRET)
    token_str = None
    if token_json and "result" in token_json and "token" in token_json["result"]:
        token_str = token_json["result"]["token"]
        logger.info(f"[Main] Got private WS token => {token_str[:10]}... (truncated)")
    else:
        logger.warning("[Main] Could not retrieve token => private feed disabled")

    # 5) Build aggregator app => calls aggregator_cycle_all_coins
    aggregator_app = HybridApp(
        pairs=TRADED_PAIRS,
        strategy=ai_strategy,
        aggregator_interval=AGG_INTERVAL,
        private_ws_client=None  # attach after creation if we have token
    )

    # Build public feed
    pub_client = KrakenPublicWSClient(
        pairs=TRADED_PAIRS,
        feed_type="ticker",
        # We'll define on_ticker_callback => aggregator_app.on_ticker to store last price,
        # then aggregator_app tries aggregator_cycle_all_coins every aggregator_interval
        on_ticker_callback=aggregator_app.on_ticker
    )

    # Build private feed if token
    priv_client = None
    if token_str:
        def on_private_event(evt_dict):
            logger.debug(f"[PrivateWS] Event => {evt_dict}")

        priv_client = KrakenPrivateWSClient(
            token=token_str,
            on_private_event_callback=on_private_event
        )
        aggregator_app.ws_private = priv_client

    # start them
    pub_client.start()
    if priv_client:
        priv_client.start()

    logger.info("[Main] Running multi-coin aggregator approach. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("[Main] user exit => stopping.")
    finally:
        pub_client.stop()
        if priv_client:
            priv_client.stop()
        logger.info("[Main] aggregator halted.")


if __name__ == "__main__":
    main()
