import sqlite3
import pandas as pd
from ta.volatility import AverageTrueRange  # Requires 'ta' library
import logging
from typing import Optional, Tuple, List
from config_loader import ConfigLoader

# Configuration
DB_FILE = "trades.db"  # Path to SQLite database
ATR_PERIOD = 14
STOP_LOSS_MULTIPLIER = ConfigLoader.get_value("stop_loss_percent")*100
TAKE_PROFIT_MULTIPLIER = ConfigLoader.get_value("take_profit_percent")*100

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def ensure_db_table():
    """Create the coin_thresholds table if it doesn’t exist."""
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS coin_thresholds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL UNIQUE,
                atr REAL,
                stop_loss REAL,
                take_profit REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        logger.info("[ThresholdManager] coin_thresholds table ensured.")
    except Exception as e:
        logger.exception(f"[ThresholdManager] Error creating coin_thresholds table: {e}")
    finally:
        conn.close()

def calculate_atr(pair: str, period: int = ATR_PERIOD) -> Optional[float]:
    """
    Calculate the ATR for a given pair using the last 'period' price points.
    Uses max(bid_price, ask_price) as high, min(bid_price, ask_price) as low, and last_price as close.

    Args:
        pair (str): The trading pair (e.g., 'ETH/USD').
        period (int): Number of periods for ATR calculation (default: 14).

    Returns:
        Optional[float]: The latest ATR value, or None if insufficient data.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        query = """
            SELECT timestamp, bid_price, ask_price, last_price
            FROM price_history
            WHERE pair = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(pair, period))
        if len(df) < period:
            logger.warning(f"[ThresholdManager] Insufficient data for ATR calculation: {pair}, only {len(df)} rows.")
            return None
        # Sort by timestamp ascending for correct chronological order
        df = df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
        # Compute high, low, close
        df['high'] = df[['bid_price', 'ask_price']].max(axis=1)
        df['low'] = df[['bid_price', 'ask_price']].min(axis=1)
        df['close'] = df['last_price']
        # Calculate ATR
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=period).average_true_range()
        return atr.iloc[-1]  # Latest ATR value
    except Exception as e:
        logger.exception(f"[ThresholdManager] Error calculating ATR for {pair}: {e}")
        return None
    finally:
        conn.close()

def calculate_thresholds(current_price: float, atr: float, stop_loss_multiplier: float = STOP_LOSS_MULTIPLIER, take_profit_multiplier: float = TAKE_PROFIT_MULTIPLIER) -> Tuple[float, float]:
    """
    Calculate stop-loss and take-profit thresholds based on current price and ATR.

    Args:
        current_price (float): The latest price of the pair.
        atr (float): The ATR value.
        stop_loss_multiplier (float): Multiplier for stop-loss (default: 1.5).
        take_profit_multiplier (float): Multiplier for take-profit (default: 2.0).

    Returns:
        Tuple[float, float]: (stop_loss, take_profit)
    """
    stop_loss = current_price - (atr * stop_loss_multiplier)
    take_profit = current_price + (atr * take_profit_multiplier)
    return stop_loss, take_profit

def store_thresholds(pair: str, atr: float, stop_loss: float, take_profit: float):
    """
    Store the calculated ATR, stop-loss, and take-profit in the database.

    Args:
        pair (str): The trading pair.
        atr (float): The ATR value.
        stop_loss (float): The stop-loss threshold.
        take_profit (float): The take-profit threshold.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO coin_thresholds (pair, atr, stop_loss, take_profit, last_updated)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (pair, atr, stop_loss, take_profit))
        conn.commit()
        logger.info(f"[ThresholdManager] Stored thresholds for {pair}: ATR={atr}, SL={stop_loss}, TP={take_profit}")
    except Exception as e:
        logger.exception(f"[ThresholdManager] Error storing thresholds for {pair}: {e}")
    finally:
        conn.close()

def fetch_latest_price(pair: str) -> Optional[float]:
    """
    Fetch the latest last_price for the given pair.

    Args:
        pair (str): The trading pair.

    Returns:
        Optional[float]: The latest last_price, or None if not available.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("SELECT last_price FROM price_history WHERE pair = ? ORDER BY timestamp DESC LIMIT 1", (pair,))
        row = c.fetchone()
        if row:
            return float(row[0])
        logger.warning(f"[ThresholdManager] No price data found for {pair}")
        return None
    except Exception as e:
        logger.exception(f"[ThresholdManager] Error fetching latest price for {pair}: {e}")
        return None
    finally:
        conn.close()

def update_pair_thresholds(pairs: List[str]):
    """
    Update thresholds for the given list of pairs.

    Args:
        pairs (List[str]): List of trading pairs to update (e.g., ["ETH/USD", "BTC/USD"]).
    """
    for pair in pairs:
        current_price = fetch_latest_price(pair)
        if current_price is None:
            continue  # Skip if no price data
        atr = calculate_atr(pair)
        if atr is None:
            continue  # Skip if ATR calculation fails
        stop_loss, take_profit = calculate_thresholds(current_price, atr)
        store_thresholds(pair, atr, stop_loss, take_profit)

# Ensure the database table exists on module load
ensure_db_table()