#!/usr/bin/env python3
# =============================================================================
# FILE: db_lookup.py
# =============================================================================
"""
db_lookup.py

This module provides quick, helpful database lookups related to Kraken asset pairs and
LunarCrush price data. It is intended to help with:
1) Retrieving the base coin symbol from kraken_asset_pairs given a wsname (e.g. "ETH/USD").
2) Retrieving the minimum order size (`ordermin`) from kraken_asset_pairs.
3) Calculating the minimum cost in USD by multiplying `ordermin` with the latest coin price
   from lunarcrush_data.

Usage Example:
    from db_lookup import get_base_asset, get_ordermin, get_minimum_cost_in_usd

    # 1) Base asset for "ETH/USD"
    base = get_base_asset("ETH/USD")
    print(f"Base asset for ETH/USD: {base}")

    # 2) Minimum order size for "ETH/USD"
    minimum_order = get_ordermin("ETH/USD")
    print(f"Minimum order size in coin units: {minimum_order}")

    # 3) Minimum cost in USD for "ETH/USD"
    cost_in_usd = get_minimum_cost_in_usd("ETH/USD")
    print(f"Minimum cost in USD: {cost_in_usd:.2f}")

Implementation Details:
    - kraken_asset_pairs table must have columns: (wsname, base, ordermin, ...)
    - lunarcrush_data table must have at least (timestamp, symbol, price).
    - We'll grab the latest price from lunarcrush_data by ordering timestamp DESC.
    - If no matching row is found, or if conversions fail, the functions return fallback
      values (empty string or 0.0) rather than raise errors.
    - We do a small normalization on 'base' symbols (e.g. strip leading "X" or "Z") before
      matching them in lunarcrush_data.

Author: Your Team
"""

import sqlite3
import logging
import os

###############################################################################
# Adjust DB_FILE if your DB path is different
###############################################################################
DB_FILE = "trades.db"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def _normalize_symbol(symbol: str) -> str:
    """
    Internal helper function to strip leading 'X' or 'Z' from a Kraken base symbol if present.
    Ensures uppercase for lookup in lunarcrush_data.

    Examples:
        _normalize_symbol("XETH") => "ETH"
        _normalize_symbol("ZUSD") => "USD"
        _normalize_symbol("BTC")  => "BTC"
    """
    if not symbol:
        return ""
    # For safety, strip multiple leading X/Z characters, though typically there's only one.
    symbol = symbol.lstrip("XZ").upper()
    return symbol


def get_base_asset(wsname: str) -> str:
    """
    Returns the 'base' column from kraken_asset_pairs where wsname = ?.

    :param wsname: The WebSocket name of the pair, e.g. "ETH/USD" or "XBT/USD".
    :type wsname: str
    :return: The raw 'base' value from the table, e.g. "XETH", "XXBT", or "" if not found.
    :rtype: str
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            SELECT base
            FROM kraken_asset_pairs
            WHERE wsname = ?
            LIMIT 1
        """, (wsname,))
        row = c.fetchone()
        if row:
            return row[0]  # e.g. "XETH"
        logger.warning(f"[db_lookup] No matching base asset for wsname={wsname}")
        return ""
    except Exception as e:
        logger.exception(f"[db_lookup] Error in get_base_asset(wsname='{wsname}'): {e}")
        return ""
    finally:
        conn.close()

def get_asset_value_for_pair(wsname: str, value: str) -> str:
    """
    Returns the 'base' column from kraken_asset_pairs where wsname = ?.

    :param value:
    :param wsname: The WebSocket name of the pair, e.g. "ETH/USD" or "XBT/USD".
    :type wsname: str
    :return: The raw 'base' value from the table, e.g. "XETH", "XXBT", or "" if not found.
    :rtype: str
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute(f"""
            SELECT {value.upper()}
            FROM kraken_asset_pairs
            WHERE wsname = ?
            LIMIT 1
        """, (wsname,))
        row = c.fetchone()
        if row:
            return row[0]  # e.g. "XETH"
        logger.warning(f"[db_lookup] No matching base asset for wsname={wsname}")
        return ""
    except Exception as e:
        logger.exception(f"[db_lookup] Error in get_base_asset(wsname='{wsname}'): {e}")
        return ""
    finally:
        conn.close()


def get_ordermin(wsname: str) -> float:
    """
    Returns the 'ordermin' value (minimum order size in coin units) from kraken_asset_pairs
    for a given wsname. The 'ordermin' is stored as text in the DB, so we convert it to float.

    :param wsname: The WebSocket name of the pair, e.g. "ETH/USD".
    :type wsname: str
    :return: The minimum order size as a float. Returns 0.0 if not found or on error.
    :rtype: float
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            SELECT ordermin
            FROM kraken_asset_pairs
            WHERE wsname = ?
            LIMIT 1
        """, (wsname,))
        row = c.fetchone()
        if row and row[0] is not None:
            return float(row[0])
        logger.warning(f"[db_lookup] Could not find ordermin for wsname={wsname}")
        return 0.0
    except Exception as e:
        logger.exception(f"[db_lookup] Error in get_ordermin(wsname='{wsname}'): {e}")
        return 0.0
    finally:
        conn.close()


def get_formatted_name_from_pair_name(wsname: str) -> str | None:
    """
    Returns the 'ordermin' value (minimum order size in coin units) from kraken_asset_pairs
    for a given wsname. The 'ordermin' is stored as text in the DB, so we convert it to float.

    :param wsname: The WebSocket name of the pair, e.g. "ETH/USD".
    :type wsname: str
    :return: The minimum order size as a float. Returns 0.0 if not found or on error.
    :rtype: float
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            SELECT formatted_base_asset_name
            FROM kraken_asset_name_lookup
            WHERE wsname = ?
            LIMIT 1
        """, (wsname,))
        row = c.fetchone()
        if row and row[0] is not None:
            return str(row[0])
        logger.warning(f"[db_lookup] Could not find ordermin for wsname={wsname}")
        return None
    except Exception as e:
        logger.exception(f"[db_lookup] Error in get_ordermin(wsname='{wsname}'): {e}")
        return None
    finally:
        conn.close()


def get_minimum_cost_in_usd(wsname: str) -> float:
    """
    Calculate the minimum cost in USD by:
      1) Retrieving the coin's 'ordermin' from kraken_asset_pairs
      2) Finding the coin's latest price (in lunarcrush_data.price)
      3) Multiplying them => returns float cost in USD

    :param wsname: The WebSocket name of the pair, e.g. "ETH/USD".
    :type wsname: str
    :return: The minimum cost in USD. Returns 0.0 if not found or on error.
    :rtype: float
    """
    ordermin_val = get_ordermin(wsname)
    if ordermin_val <= 0:
        logger.warning(f"[db_lookup] ordermin is zero or negative for wsname={wsname}")
        return 0.0

    base = get_base_asset(wsname)
    if not base:
        logger.warning(f"[db_lookup] No base asset found for wsname={wsname}")
        return 0.0

    # Normalize base to match how it's stored in lunarcrush_data.symbol
    symbol = _normalize_symbol(base)

    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            SELECT price
            FROM lunarcrush_data
            WHERE UPPER(symbol) = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (symbol,))
        row = c.fetchone()
        if not row:
            logger.warning(f"[db_lookup] Could not find lunarcrush_data for symbol={symbol}")
            return 0.0

        latest_price = float(row[0])
        if latest_price <= 0:
            logger.warning(f"[db_lookup] Latest price is zero/negative for symbol={symbol}")
            return 0.0

        return ordermin_val * latest_price

    except Exception as e:
        logger.exception(f"[db_lookup] Error in get_minimum_cost_in_usd(wsname='{wsname}'): {e}")
        return 0.0
    finally:
        conn.close()

def get_recent_timeseries_for_coin(coin_id: str, limit: int = 5) -> dict:
    """
    Returns up to `limit` most recent records from lunarcrush_timeseries for the given `coin_id`.
    The returned dictionary has timestamps (DESC order) as keys, and each value is another dict
    with open_price, close_price, high_price, and low_price.

    Example return format:
    {
      1675854200: {
        "open_price": 123.45,
        "close_price": 130.12,
        "high_price": 135.50,
        "low_price": 122.75
      },
      1675850600: {
        "open_price": 120.00,
        "close_price": 123.45,
        ...
      },
      ...
    }
    """

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # So we can access row fields by name
    results_dict = {}

    try:
        c = conn.cursor()
        c.execute("""
            SELECT
                timestamp,
                open_price,
                close_price,
                high_price,
                low_price,
                volatility
            FROM lunarcrush_timeseries
            WHERE coin_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (coin_id.upper(), limit))
        rows = c.fetchall()

        # Populate the dictionary with timestamps as keys
        for row in rows:
            t = row["timestamp"]
            results_dict[t] = {
                "open_price":  round(row["open_price"], 2),
                "close_price": round(row["close_price"], 2),
                "high_price":  round(row["high_price"], 2),
                "low_price":   round(row["low_price"], 2),
                "volatility":  round(row["volatility"], 2)
            }

    except Exception as e:
        logger.exception(f"[db_lookup] Error in get_recent_timeseries_for_coin(coin_id='{coin_id}'): {e}")
    finally:
        conn.close()

    return results_dict



if __name__ == "__main__":
    # Example usage / self-test
    example_wsname = "DOGE/USD"

    print(f"Testing db_lookup.py for wsname={example_wsname} ...")

    base_asset = get_base_asset(example_wsname)
    print(f"Base asset => {base_asset}")

    order_min = get_ordermin(example_wsname)
    print(f"ordermin => {order_min}")

    min_cost_usd = get_minimum_cost_in_usd(example_wsname)
    print(f"Minimum cost in USD => {min_cost_usd}")

    tick_size = get_asset_value_for_pair(example_wsname, "tick_size")
    print(f"tick_size => {tick_size}")