# ==============================================================================
# FILE: ai_strategy.py
# ==============================================================================
"""
ai_strategy.py

Contains the AIStrategy class, which now optionally fetches and uses
multiple pairs' data for normalization or expanded feature sets.
"""

import logging
import sqlite3
import numpy as np

logger = logging.getLogger(__name__)

DB_FILE = "trades.db"

class AIStrategy:
    """
    AIStrategy class that decides whether to BUY, SELL, or HOLD based on
    AI-driven signals. This version can incorporate multi-coin data.
    """

    def __init__(self, pairs=None):
        """
        :param pairs: List of trading pairs to incorporate, e.g. ["XBT/USD", "ETH/USD"].
                      If None, defaults to single pair logic.
        """
        self.pairs = pairs if pairs else ["XBT/USD"]
        # Potentially load a trained model here if desired, e.g.:
        # self.model = joblib.load("trained_model.pkl")

    def predict(self, market_data: dict):
        """
        Given the immediate market data for a single pair, plus the possibility
        of referencing data for multiple coins from the DB, decide:
          - "BUY", "SELL", or "HOLD"
          - trade size
        :param market_data: A dict with {"price": float, "timestamp": float, ...} for one pair.
        :return: (signal, size)
        """
        # 1) Example: incorporate multi-coin context. We'll fetch the latest
        #    price from each pair in self.pairs, do a quick normalization.
        multi_prices = self._fetch_latest_prices_for_pairs()
        if len(multi_prices) < len(self.pairs):
            logger.warning("Not all pairs had data. Running fallback logic.")
            # fallback logic if incomplete data
            return self._dummy_logic(market_data)

        # 2) Normalization approach (dummy example: min-max across pairs)
        prices_array = np.array(list(multi_prices.values()), dtype=float)
        min_p = prices_array.min()
        max_p = prices_array.max()
        if max_p == min_p:  # avoid divide-by-zero
            norm_prices = np.ones_like(prices_array)
        else:
            norm_prices = (prices_array - min_p) / (max_p - min_p)

        # 3) Decide based on your single pair's normalized price
        #    Let's say the pair in market_data is the last in self.pairs
        #    or you identify it by key. (We'll do a simple approach.)
        my_pair = self.pairs[-1]  # assume market_data belongs to the last pair
        my_price_index = self.pairs.index(my_pair)
        my_norm = norm_prices[my_price_index]
        logger.info(f"Multi-pair normalization: {multi_prices}, normalized={norm_prices}, my_norm={my_norm:.4f}")

        # DUMMY logic: if normalized price < 0.5 => BUY, else HOLD
        if my_norm < 0.5:
            return ("BUY", 0.0005)
        else:
            return ("HOLD", 0.0)

    def _dummy_logic(self, market_data: dict):
        """
        Fallback logic if multi-coin data not available or incomplete.
        """
        price = market_data.get("price", 0.0)
        if price < 20000:
            return ("BUY", 0.0005)
        else:
            return ("HOLD", 0.0)

    def _fetch_latest_prices_for_pairs(self):
        """
        Retrieves the most recent 'last_price' for each pair in self.pairs
        from the 'price_history' table.
        Returns a dict: { "XBT/USD": <float>, "ETH/USD": <float>, ... }
        """
        multi_prices = {}
        conn = sqlite3.connect(DB_FILE)
        try:
            c = conn.cursor()
            for pair in self.pairs:
                c.execute("""
                    SELECT last_price 
                    FROM price_history
                    WHERE pair=?
                    ORDER BY id DESC
                    LIMIT 1
                """, (pair,))
                row = c.fetchone()
                if row:
                    multi_prices[pair] = row[0]
        except Exception as e:
            logger.exception(f"Error fetching multi-pair data: {e}")
        finally:
            conn.close()
        return multi_prices
