#!/usr/bin/env python3

import sqlite3
import logging

DB_FILE = "trades.db"  # or your actual path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_price_history_problems(db_path=DB_FILE):
    """
    Connects to the SQLite DB and aggregates 'price_history' rows to find
    zero or negative 'last_price', plus overall min/max. This helps identify
    if any pairs have unusual or invalid data leading to extreme backtest results.
    """
    conn = sqlite3.connect(db_path)
    try:
        query = """
        SELECT 
            pair,
            COUNT(*) AS total_rows,
            SUM(CASE WHEN last_price <= 0 THEN 1 ELSE 0 END) AS zero_or_negative_prices,
            MIN(last_price) AS min_price,
            MAX(last_price) AS max_price
        FROM price_history
        GROUP BY pair
        ORDER BY pair;
        """
        cursor = conn.execute(query)
        rows = cursor.fetchall()

        logger.info("Check of price_history for zero/negative or extreme prices:")
        logger.info("pair | total_rows | zero_or_negative_prices | min_price | max_price")
        for row in rows:
            pair, total_rows, zero_neg, minp, maxp = row
            logger.info(f"{pair} | {total_rows} | {zero_neg} | {minp} | {maxp}")

    except Exception as e:
        logger.exception(f"Error while checking price history: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_price_history_problems()
