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
from typing import List, Optional
import datetime

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


def get_symbols_for_time_series(
    traded_pairs: list[str],
    db_path: str = DB_FILE
) -> list[str]:
    """
    Given a list of Kraken wsname pairs (e.g. ["ETH/USD", "SOL/USD"]),
    fetches the 'formatted_base_asset_name' from 'kraken_asset_name_lookup'
    for each pair. Returns a list of symbols (e.g. ["ETH","SOL"]) in
    the same order, or None for any that are not found.

    :param traded_pairs: A list of wsname strings (e.g. ["ETH/USD","SOL/USD"]).
    :param db_path: Path to the SQLite database. Defaults to DB_FILE.
    :return: A list of symbols (e.g. ["ETH","SOL","ADA"]), or None entries if missing.
    """
    if not traded_pairs:
        return []

    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        # Build a parameterized query: SELECT wsname, formatted_base_asset_name ...
        placeholders = ",".join("?" for _ in traded_pairs)
        query = f"""
            SELECT wsname, formatted_base_asset_name
            FROM kraken_asset_name_lookup
            WHERE wsname IN ({placeholders})
        """
        c.execute(query, traded_pairs)
        rows = c.fetchall()

        # Put results into a dict => { "ETH/USD": "ETH", "SOL/USD": "SOL" }
        wsname_to_symbol = {}
        for (ws, formatted_name) in rows:
            wsname_to_symbol[ws] = formatted_name

    except Exception as e:
        logger.exception(f"[db_lookup] Error in get_symbols_for_time_series => {e}")
        return []
    finally:
        conn.close()

    # Build the output list in the same order as `traded_pairs`
    result = []
    for ws in traded_pairs:
        symbol = wsname_to_symbol.get(ws, None)
        result.append(symbol)

    return result



def is_table_underpopulated(
    table_name: str,
    db_file: str,
    grouping_vars: list[str],
    min_row_count: int
) -> bool:
    """
    Checks whether a table has enough rows overall (if no grouping vars)
    or enough rows on average per group (if grouping vars exist).

    :param table_name: The name of the SQLite table to query.
    :param db_file: Path to the SQLite database file.
    :param grouping_vars: A list of column names on which to GROUP BY.
                         E.g. ["symbol","side"] or [] if none.
    :param min_row_count: The threshold:
       - If grouping_vars is empty, the table's total row count must exceed this.
       - If grouping_vars is non-empty, the average (mean) row count across all groups must exceed this.
    :return: True if the condition is satisfied, False otherwise.
    """
    if not table_name:
        logger.error("No table_name provided.")
        return False

    conn = sqlite3.connect(db_file)
    try:
        c = conn.cursor()

        # CASE 1: No grouping variables => total row count
        if not grouping_vars:
            query = f"SELECT COUNT(*) FROM {table_name}"
            c.execute(query)
            row = c.fetchone()
            total_rows = row[0] if row else 0

            logger.debug(f"[check_table_minimum_rows] {table_name} => total_rows={total_rows}")
            return total_rows < min_row_count

        # CASE 2: We have grouping variables => average row count per group
        # e.g. SELECT col1, col2, COUNT(*) as cnt FROM table_name GROUP BY col1, col2
        col_list = ",".join(grouping_vars)
        query = (
            f"SELECT {col_list}, COUNT(*) as group_count "
            f"FROM {table_name} "
            f"GROUP BY {col_list}"
        )
        c.execute(query)
        rows = c.fetchall()
        if not rows:
            # If grouping yields no rows, average is effectively zero
            logger.debug(f"[check_table_minimum_rows] {table_name} => no groups found.")
            return False

        # compute average of group_count
        sum_counts = 0
        for r in rows:
            # last column is group_count (index = len(grouping_vars))
            group_count = r[len(grouping_vars)]
            sum_counts += group_count

        avg_count = float(sum_counts) / len(rows)
        logger.debug(
            f"[check_table_minimum_rows] {table_name} => found {len(rows)} groups, avg_count={avg_count:.2f}"
        )

        return avg_count < min_row_count

    except Exception as e:
        logger.exception(f"Error checking row counts for table={table_name}: {e}")
        return False
    finally:
        conn.close()


def map_kraken_asset_pairs(
        search_value: str,
        input_colname: str = "wsname",
        output_colname: str = "base",
        db_file: str = DB_FILE
) -> str:
    """
    Retrieves a specific column value (`output_colname`) from the `kraken_asset_pairs` table,
    by matching `input_colname` against a given `search_value`.

    Example: If you're storing pairs with "wsname"="ETH/USD", calling
    map_kraken_asset_pairs("ETH/USD", "wsname", "base") might return "XETH".

    :param search_value: The actual value to look up in `input_colname`.
                        e.g. "ETH/USD" if input_colname="wsname".
    :type search_value: str
    :param input_colname: The column name to match in the query. Defaults to "wsname".
                         Other valid columns might be "altname", etc.
    :type input_colname: str
    :param output_colname: The column name to retrieve if a match is found. Defaults to "base".
                           e.g. "base", "quote", "wsname", etc.
    :type output_colname: str
    :param db_file: Path to your SQLite database. Defaults to the module-level DB_FILE.
    :type db_file: str

    :return: The string value of `output_colname` if found, otherwise "".
    :rtype: str
    """
    conn = sqlite3.connect(db_file)
    try:
        c = conn.cursor()
        query = f"""
            SELECT {output_colname}
            FROM kraken_asset_pairs
            WHERE {input_colname} = ?
            LIMIT 1
        """
        c.execute(query, (search_value,))
        row = c.fetchone()
        if row:
            return row[0]  # e.g. "XETH" or "XXBT"
        logger.warning(
            f"[db_lookup] No matching row in 'kraken_asset_pairs' "
            f"for {input_colname}='{search_value}' => cannot retrieve '{output_colname}'."
        )
        return ""
    except Exception as e:
        logger.exception(
            f"[db_lookup] Error retrieving '{output_colname}' where {input_colname}='{search_value}': {e}"
        )
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


def get_websocket_name_from_base_asset(base_asset_name: str) -> str:
    """
    Returns the 'base' column from kraken_asset_pairs where wsname = ?.

    :param base_asset_name: The raw 'base' value from the table, e.g. "XETH", "XXBT", or "" if not found.
    :type base_asset_name: str
    :return: The WebSocket name of the pair, e.g. "ETH/USD" or "XBT/USD".
    :rtype: str
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            SELECT wsname
            FROM kraken_asset_name_lookup
            WHERE base_asset = ?
            LIMIT 1
        """, (base_asset_name,))
        row = c.fetchone()
        if row:
            return row[0]  # e.g. "ETH/USD"
        logger.warning(f"[db_lookup] No matching base asset for wsname={base_asset_name}")
        return ""
    except Exception as e:
        logger.exception(f"[db_lookup] Error in get_base_asset(wsname='{base_asset_name}'): {e}")
        return ""
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
    symbol = get_formatted_name_from_pair_name(wsname)

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

# --------------------------------------------------------------------------
# Retrieve recent ledger trades for a given pair (BUY/SELL), with USD cost
# --------------------------------------------------------------------------
def get_recent_ledger_entries_for_pair(
    pair: str,
    limit: int = 5,
    trade_type: Optional[str] = None
) -> List[str]:
    """
    Retrieves up to `limit` most recent trades (ledger entries) for the given pair,
    returning a list of strings formatted as:

        <epoch_time> <YYYY-MM-DD HH:MM> <BUY|SELL> <pair> <quantity>@<price_in_usd> balance=<balance>

    - We detect BUY vs SELL by sign of 'amount' in the base asset row:
        amount>0 => BUY
        amount<0 => SELL
    - We also fetch the matching ZUSD row (same 'refid') to see the cost or proceeds
      in USD (absolute value).  price = (abs(zusd_amount) / abs(base_amount)).
    - The ledger 'balance' is taken from the base-asset row's 'balance' column.
    - `trade_type` can be "BUY" or "SELL" if you only want that type. Otherwise None returns both.
    - We order by time DESC.

    Example output line:
       1739675593 2025-02-15 21:50 BUY ETH/USD 0.002@2692.23 balance=0.034
    """
    base_asset = get_base_asset(pair)
    if not base_asset:
        logger.warning(f"[db_lookup] Cannot retrieve recent ledger entries => no base_asset for {pair}")
        return []

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    results = []
    try:
        c = conn.cursor()
        # Grab the last `limit` ledger rows for base_asset, ignoring type != 'trade'
        q = """
        SELECT ledger_id, refid, time, amount, balance
        FROM ledger_entries
        WHERE asset=? 
          AND type='trade'
        ORDER BY time DESC
        LIMIT ?
        """
        c.execute(q, (base_asset, limit))
        rows = c.fetchall()

        for row in rows:
            ledger_id = row["ledger_id"]
            refid = row["refid"]
            t_epoch = float(row["time"])
            amt_base = float(row["amount"])
            bal_base = float(row["balance"] or 0.0)

            # Determine if it's buy or sell
            side = "BUY" if amt_base > 0 else "SELL"
            if amt_base == 0:
                # skip edge cases (0.0 amounts)
                continue
            # If trade_type is specified, skip mismatch
            if trade_type and trade_type.upper() != side:
                continue

            abs_qty = abs(amt_base)

            # For price in USD => find the matching ZUSD row for the same refid
            c2 = conn.cursor()
            c2.execute("""
            SELECT amount 
            FROM ledger_entries
            WHERE refid=? 
              AND asset='ZUSD'
              AND type='trade'
            LIMIT 1
            """, (refid,))
            row_zusd = c2.fetchone()
            cost_usd = 0.0
            if row_zusd:
                cost_usd = abs(float(row_zusd[0]))

            px = 0.0
            if abs_qty > 0:
                px = cost_usd / abs_qty

            # Convert epoch to readable date
            dt = datetime.datetime.utcfromtimestamp(t_epoch)
            dt_str = dt.strftime("%Y-%m-%d %H:%M")

            # Build final string
            line = (
                f"{int(t_epoch)} {dt_str} {side} {pair} "
                f"{abs_qty:.4g}@{px:.3f} balance={bal_base:.4g}"
            )
            results.append(line)

        return results

    except Exception as e:
        logger.exception(f"[db_lookup] Error in get_recent_ledger_entries_for_pair({pair}): {e}")
        return []
    finally:
        conn.close()


def get_recent_buys_for_pair(pair: str, limit: int = 5) -> List[str]:
    """
    Returns only the most recent BUY trades for a given pair.
    Internally calls get_recent_ledger_entries_for_pair(..., trade_type='BUY').
    """
    return get_recent_ledger_entries_for_pair(pair, limit=limit, trade_type='BUY')


def get_recent_sells_for_pair(pair: str, limit: int = 5) -> List[str]:
    """
    Returns only the most recent SELL trades for a given pair.
    Internally calls get_recent_ledger_entries_for_pair(..., trade_type='SELL').
    """
    return get_recent_ledger_entries_for_pair(pair, limit=limit, trade_type='SELL')



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